"""HLS 출력 (명세 §4.6): 장기 실행 ffmpeg 인코더 1개에 믹스된 PCM 을 흘려 넣고, 자막은 WebVTT 플레이리스트로 직접 쓴다.

디렉터리 `<hls_dir>/<session_id>/`:
- `master.m3u8` (우리가 씀): 오디오(+영상) 스트림 + 자막 렌디션 2개(번역 `sub_tgt`, 원문 `sub_src`)
- `media.m3u8`, `media_NNNNN.ts` (ffmpeg): 1 s 세그먼트, 슬라이딩 창, `program_date_time`
- `sub_tgt.m3u8`, `sub_tgt_NNNNN.vtt`, `sub_src.*` (우리가 씀): 미디어와 같은 번호·길이의 세그먼트. 큐 시각은 t_cap 기준이고
  `X-TIMESTAMP-MAP` 으로 미디어 PTS 에 맞춘다 (첫 세그먼트의 start_time 을 ffprobe 로 읽어 보정).
- 영상(`video_path` 가 있으면): ingest 가 기록 중인 raw.ts 를 `-re` 로 뒤따라 읽어(지연은 시작 시점으로 생김) 복사 먹싱.
- 첫 세그먼트는 시작 후 약 2 s 안에 생겨야 한다 (파이프 입력 분석 끔). 늦으면 `playing` 이 D 보다 늦게 나간다.

세션이 끝나면 ENDLIST 를 쓰고, 디렉터리는 세션 삭제 때 지운다 (방송 내용을 디스크에 남기지 않음, §13).
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
import threading
from collections import deque
from pathlib import Path

from livetrans.core.audio import OUT_RATE, float_to_pcm16_bytes
from livetrans.core.wincap import _NO_WINDOW
from livetrans.pipeline.timeline import Cue, vtt_segment

log = logging.getLogger("livetrans.hls")

CONTENT_TYPES = {
    ".m3u8": "application/vnd.apple.mpegurl",
    ".ts": "video/mp2t",
    ".vtt": "text/vtt; charset=utf-8",
    ".m4s": "video/iso.segment",
    ".mp4": "video/mp4",
}


def write_text_atomic(path: Path, text: str) -> None:
    """플레이리스트를 임시 파일에 쓴 뒤 바꿔 넣는다 — 읽는 쪽(플레이어 폴링)이 반쯤 쓰인 파일을 보지 않게.

    Windows 는 다른 핸들이 열어 둔 파일 위로 바꿔 넣지 못할 수 있어(PermissionError) 그때만 제자리에 쓴다.
    """
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    try:
        tmp.replace(path)
    except PermissionError:
        path.write_text(text, encoding="utf-8")
        tmp.unlink(missing_ok=True)


def find_ffmpeg(configured: str | None) -> str | None:
    """설정값 → PATH → 흔한 설치 위치 순으로 ffmpeg 실행 파일을 찾는다."""
    for cand in (configured, "ffmpeg"):
        if cand and shutil.which(cand):
            return shutil.which(cand)
    if configured and Path(configured).is_file():
        return configured
    home = Path.home()
    for p in sorted(home.glob("Downloads/ffmpeg*/bin/ffmpeg.exe"), reverse=True):
        return str(p)
    return None


class HlsWriter:
    def __init__(
        self,
        session_dir: Path,
        *,
        ffmpeg: str,
        segment_s: float = 1.0,
        list_size: int = 60,
        video_path: Path | None = None,
        video_codec: str = "copy",
        langs: tuple[str, str] = ("ko", "src"),
    ) -> None:
        self.dir = session_dir
        self.ffmpeg = ffmpeg
        self.segment_s = segment_s
        self.list_size = list_size
        self.video_path = video_path
        self.video_codec = video_codec
        self.tgt_lang, self.src_lang = langs
        self._proc: subprocess.Popen | None = None
        self._stderr: deque[str] = deque(maxlen=50)
        self._vtt_seq: dict[str, deque[tuple[int, Path]]] = {"tgt": deque(), "src": deque()}
        self._mpegts_offset: int | None = None
        self._closed = False

    # --- 수명 ---

    def start(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        args = [self.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
        maps = []
        if self.video_path is not None:
            args += ["-re", "-i", str(self.video_path)]
            maps += ["-map", "0:v:0"]
        # 파이프 PCM 은 포맷을 이미 아니 분석을 끄지 않으면 ffmpeg 가 기본 5 s 치 오디오가 들어올 때까지
        # 출력 초기화를 미뤄 첫 세그먼트가 그만큼 늦어진다 (2026-10-24 측정: 5.8 s → 1.8 s)
        args += ["-analyzeduration", "0", "-probesize", "32"]
        args += ["-f", "s16le", "-ar", str(OUT_RATE), "-ac", "2", "-i", "pipe:0"]
        maps += ["-map", f"{1 if self.video_path is not None else 0}:a:0"]
        args += maps
        if self.video_path is not None:
            args += ["-c:v", self.video_codec]
            if self.video_codec != "copy":
                args += ["-preset", "p4", "-b:v", "4M", "-g", str(int(self.segment_s * 60))]
        args += [
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-muxdelay",
            "0",
            "-muxpreload",
            "0",
            "-f",
            "hls",
            "-hls_time",
            str(self.segment_s),
            "-hls_list_size",
            str(self.list_size),
            "-hls_flags",
            "delete_segments+independent_segments+program_date_time+omit_endlist",
            "-hls_segment_type",
            "mpegts",
            "-hls_segment_filename",
            str(self.dir / "media_%05d.ts"),
            str(self.dir / "media.m3u8"),
        ]
        self._proc = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            creationflags=_NO_WINDOW,
        )
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        # 입력(raw.ts 등)을 못 열면 바로 죽는다 → 짧게 기다려 확인하고 호출자가 폴백하게 한다
        try:
            self._proc.wait(timeout=0.8)
        except subprocess.TimeoutExpired:
            pass
        if self._proc.poll() is not None:
            raise RuntimeError(
                f"HLS 인코더가 시작 직후 종료 (rc={self._proc.returncode}): {self.last_error}"
            )
        self._closed = False
        self._write_master()
        for kind in ("tgt", "src"):
            self._write_vtt_playlist(kind)

    def restart_without_video(self) -> None:
        """영상 입력을 열 수 없을 때 소리 전용으로 다시 시작한다 (자막·TTS 는 그대로)."""
        log.warning("HLS 영상 입력 실패 → 소리 전용으로 전환: %s", self.last_error)
        self.video_path = None
        self._proc = None
        self.start()

    def _drain_stderr(self) -> None:
        assert self._proc and self._proc.stderr
        for raw in self._proc.stderr:
            line = raw.decode("utf-8", "replace").strip()
            if line:
                self._stderr.append(line)
                log.warning("ffmpeg(hls): %s", line)

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def ready(self) -> bool:
        """플레이어가 열 수 있는 상태: 미디어 플레이리스트에 세그먼트가 하나 이상."""
        m = self.dir / "media.m3u8"
        try:
            return m.is_file() and ".ts" in m.read_text(encoding="utf-8")
        except OSError:
            return False

    @property
    def last_error(self) -> str:
        return " | ".join(list(self._stderr)[-3:])

    def write_segment(self, k: int, pcm_stereo, cues: dict[str, list[Cue]]) -> None:
        """세그먼트 k: PCM 을 인코더에, 자막을 VTT 세그먼트에."""
        if self._proc is None or self._proc.stdin is None or self._closed:
            return
        try:
            self._proc.stdin.write(float_to_pcm16_bytes(pcm_stereo))
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            log.error("HLS 인코더에 쓰기 실패: %s (%s)", e, self.last_error)
            self._closed = True
            return
        for kind in ("tgt", "src"):
            self._write_vtt(kind, k, cues.get(kind, []))

    def close(self) -> None:
        if self._closed and self._proc is None:
            return
        self._closed = True
        if self._proc is not None:
            try:
                if self._proc.stdin:
                    self._proc.stdin.close()
                self._proc.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                self._proc.kill()
        for kind in ("tgt", "src"):
            self._write_vtt_playlist(kind, end=True)

    def remove(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    # --- 플레이리스트 ---

    def _write_master(self) -> None:
        lines = [
            "#EXTM3U",
            "#EXT-X-VERSION:3",
            f'#EXT-X-MEDIA:TYPE=SUBTITLES,GROUP-ID="subs",NAME="translation",LANGUAGE="{self.tgt_lang}",DEFAULT=YES,AUTOSELECT=YES,FORCED=NO,URI="sub_tgt.m3u8"',
            f'#EXT-X-MEDIA:TYPE=SUBTITLES,GROUP-ID="subs",NAME="original",LANGUAGE="{self.src_lang}",DEFAULT=NO,AUTOSELECT=NO,FORCED=NO,URI="sub_src.m3u8"',
        ]
        if self.video_path is not None:
            lines.append(
                '#EXT-X-STREAM-INF:BANDWIDTH=4500000,CODECS="avc1.640028,mp4a.40.2",SUBTITLES="subs"'
            )
        else:
            lines.append('#EXT-X-STREAM-INF:BANDWIDTH=200000,CODECS="mp4a.40.2",SUBTITLES="subs"')
        lines.append("media.m3u8")
        write_text_atomic(self.dir / "master.m3u8", "\n".join(lines) + "\n")

    def _write_vtt(self, kind: str, k: int, cues: list[Cue]) -> None:
        if self._mpegts_offset is None:
            self._mpegts_offset = self._probe_offset()
        path = self.dir / f"sub_{kind}_{k:05d}.vtt"
        write_text_atomic(path, vtt_segment(cues, mpegts_offset=self._mpegts_offset or 0))
        seq = self._vtt_seq[kind]
        seq.append((k, path))
        while len(seq) > self.list_size:
            _, old = seq.popleft()
            old.unlink(missing_ok=True)
        self._write_vtt_playlist(kind)

    def _write_vtt_playlist(self, kind: str, *, end: bool = False) -> None:
        seq = self._vtt_seq[kind]
        first = seq[0][0] if seq else 0
        lines = [
            "#EXTM3U",
            "#EXT-X-VERSION:3",
            f"#EXT-X-TARGETDURATION:{max(1, round(self.segment_s))}",
            f"#EXT-X-MEDIA-SEQUENCE:{first}",
        ]
        for _, path in seq:
            lines += [f"#EXTINF:{self.segment_s:.3f},", path.name]
        if end:
            lines.append("#EXT-X-ENDLIST")
        write_text_atomic(self.dir / f"sub_{kind}.m3u8", "\n".join(lines) + "\n")

    def _probe_offset(self) -> int | None:
        """첫 미디어 세그먼트의 start_time(초) → MPEGTS 90 kHz 틱. 아직 없으면 None (다음 세그먼트에서 다시)."""
        first = self.dir / "media_00000.ts"
        if not first.is_file():
            return None
        ffprobe = str(Path(self.ffmpeg).with_name("ffprobe" + Path(self.ffmpeg).suffix))
        try:
            out = subprocess.run(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-show_entries",
                    "format=start_time",
                    "-of",
                    "json",
                    str(first),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
                creationflags=_NO_WINDOW,
            )
            start = float(json.loads(out.stdout)["format"]["start_time"])
            return round(start * 90000)
        except Exception as e:  # noqa: BLE001
            log.warning("ffprobe 로 세그먼트 시작 시각을 못 읽음 → 0 으로: %s", e)
            return 0


class AsyncHlsWriter:
    """HlsWriter 를 스레드로 감싼 비동기 래퍼 (ffmpeg 파이프 쓰기가 이벤트 루프를 막지 않게)."""

    def __init__(self, inner: HlsWriter) -> None:
        self.inner = inner

    async def start(self) -> None:
        await asyncio.to_thread(self.inner.start)

    async def write_segment(self, k: int, pcm, cues) -> None:
        await asyncio.to_thread(self.inner.write_segment, k, pcm, cues)

    async def close(self) -> None:
        await asyncio.to_thread(self.inner.close)

    @property
    def alive(self) -> bool:
        return self.inner.alive

    @property
    def ready(self) -> bool:
        return self.inner.ready
