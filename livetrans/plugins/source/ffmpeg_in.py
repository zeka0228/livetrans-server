"""모드 A 입력: ffmpeg 가 디먹싱·디코딩한 오디오를 받는다 (명세 §4.1 `obs_stream`, `file`).

- `url`: `srt://…`(OBS 송출, listener/caller 모두 ffmpeg 가 처리), `rtmp://`, `udp://`, 또는 파일 경로.
  파일은 `realtime=True` 면 `-re` 로 실시간 속도로 읽는다 (테스트·벤치용 = 명세의 `file` 소스).
- 출력 두 갈래: ① 48 kHz stereo(원본 출력용, `on_output` 콜백 → 컴포저 링) ② 16 kHz mono(분석용, `frames()`).
  t_cap 은 16 kHz 샘플 수로 센다 (I2). 48 kHz 쪽은 정확히 3배라 같은 축이다.
- `record_video=True` 면 두 번째 ffmpeg 가 입력을 `raw.ts` 로 복사 기록한다 (HLS 쪽이 지연을 두고 뒤따라 읽는다).
  SRT 를 두 프로세스가 동시에 받을 수 없으므로 영상이 있을 땐 **한 프로세스**가 `-map` 으로 오디오 PCM(stdout)과 raw.ts(파일)를 같이 낸다.
- 입력이 끝나거나 끊기면 `lost_reason` 을 두고 `frames()` 를 끝낸다 (세션은 E_CAPTURE_LOST → stopped).
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import threading
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import numpy as np

from livetrans.core.audio import OUT_RATE, decimate_48k_to_16k, downmix_mono, pcm16_bytes_to_float
from livetrans.core.clock import SampleClock
from livetrans.core.wincap import _NO_WINDOW
from livetrans.plugins.source.base import AudioFrame

log = logging.getLogger("livetrans.source.ffmpeg")

READ_BYTES = OUT_RATE * 2 * 2 // 10  # 100 ms of 48 kHz stereo PCM16
OutputCallback = Callable[[np.ndarray, int], None]  # (stereo float32 [n,2], t_start_ms)


class FfmpegSource:
    name = "obs_stream"

    def __init__(
        self,
        url: str,
        *,
        ffmpeg: str,
        realtime: bool = False,
        record_video: bool = False,
        record_path: Path | None = None,
        sample_rate: int = 16000,
        on_output: OutputCallback | None = None,
    ) -> None:
        self.url = url
        self.ffmpeg = ffmpeg
        self.realtime = realtime
        self.record_video = record_video
        self.record_path = record_path
        self.has_video = record_video
        self.clock = SampleClock(sample_rate)
        self.on_output = on_output
        self.lost_reason: str | None = None
        self._queue: asyncio.Queue[AudioFrame | None] = asyncio.Queue()
        self._proc: subprocess.Popen | None = None
        self._stopping = False
        self._stderr: list[str] = []
        self._carry = b""
        self._started = asyncio.Event()

    # --- 수명 ---

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        args = [self.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error"]
        if self.realtime:
            args += ["-re"]
        if self.url.startswith("srt://") or self.url.startswith("udp://"):
            args += ["-fflags", "nobuffer", "-flags", "low_delay"]
        args += ["-i", self.url]
        if self.record_video:
            assert self.record_path is not None
            self.record_path.parent.mkdir(parents=True, exist_ok=True)
            # flush_packets: raw.ts 를 패킷마다 디스크에 내려 HLS 쪽이 읽을 때 최신 상태가 되게 (기본은 256 KiB 단위라
            # 저비트레이트 방송에서 수 초 늦게 보임)
            args += ["-map", "0", "-c", "copy", "-flush_packets", "1"]
            args += ["-f", "mpegts", str(self.record_path)]
        args += ["-map", "0:a:0", "-vn", "-ac", "2", "-ar", str(OUT_RATE), "-f", "s16le", "pipe:1"]
        try:
            self._proc = subprocess.Popen(  # noqa: ASYNC220
                args,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=_NO_WINDOW,
            )
        except OSError as e:
            raise RuntimeError(f"ffmpeg 를 실행할 수 없습니다 ({self.ffmpeg}): {e}") from e
        threading.Thread(target=self._read_stderr, daemon=True).start()
        threading.Thread(target=self._read_stdout, args=(loop,), daemon=True).start()
        # 첫 오디오가 올 때까지 기다린다. 파일은 바로 온다. SRT/UDP listener 는 OBS 가 붙을 때까지 소리가 없으므로
        # 포트 바인드 실패 같은 즉시 오류만 잡고(3 s) 돌아간다 — 그 뒤의 대기는 세션이 `timeline.input_wait_s` 로 관리한다
        network = "://" in self.url
        try:
            await asyncio.wait_for(self._started.wait(), 3.0 if network else 15.0)
        except TimeoutError:
            if self._proc.poll() is not None:
                raise RuntimeError(f"ffmpeg 가 입력을 열지 못했습니다: {self._tail()}") from None
            log.info("ffmpeg 입력 대기 중 (%s)", self.url)

    def _read_stderr(self) -> None:
        assert self._proc and self._proc.stderr
        for raw in self._proc.stderr:
            line = raw.decode("utf-8", "replace").strip()
            if line:
                self._stderr.append(line)
                log.warning("ffmpeg(in): %s", line)

    def _tail(self) -> str:
        return " | ".join(self._stderr[-3:])

    def _read_stdout(self, loop: asyncio.AbstractEventLoop) -> None:
        assert self._proc and self._proc.stdout
        try:
            while True:
                data = self._proc.stdout.read(READ_BYTES)
                if not data:
                    break
                data = self._carry + data
                usable = len(data) // 4 * 4  # 2 ch × 2 bytes
                self._carry = data[usable:]
                if not usable:
                    continue
                stereo = pcm16_bytes_to_float(data[:usable], 2)
                mono16 = decimate_48k_to_16k(downmix_mono(stereo))
                t_start_ms = self.clock.advance(len(mono16))
                if self.on_output is not None:
                    self.on_output(stereo, t_start_ms)
                loop.call_soon_threadsafe(
                    self._deliver, AudioFrame(pcm=mono16, t_start_ms=t_start_ms)
                )
        finally:
            if not self._stopping:
                self.lost_reason = f"입력이 끝나거나 끊겼습니다 ({self.url}){': ' + self._tail() if self._stderr else ''}"
            loop.call_soon_threadsafe(self._deliver, None)

    def _deliver(self, frame: AudioFrame | None) -> None:
        if frame is not None and not self._started.is_set():
            self._started.set()
        self._queue.put_nowait(frame)

    async def frames(self) -> AsyncIterator[AudioFrame]:
        while True:
            frame = await self._queue.get()
            if frame is None:
                return
            yield frame

    def raw_media_dir(self) -> Path | None:
        return self.record_path.parent if self.record_path else None

    async def stop(self) -> None:
        self._stopping = True
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.terminate()
                await asyncio.to_thread(self._proc.wait, 5)
            except (OSError, subprocess.TimeoutExpired):
                self._proc.kill()
        if not self._started.is_set():
            self._queue.put_nowait(None)
