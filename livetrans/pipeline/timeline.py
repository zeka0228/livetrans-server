"""모드 A 타임라인·스케줄러·컴포저 (명세 §3, §4.5) — 순수 로직, 시계는 주입.

- 시간축: `t_cap`(ms, 입력 샘플 기준). 세그먼트 `k` = `[kS, (k+1)S)`.
- 벽시계 매핑: 캡처 시작 벽시계 `start_wall` 에 대해 `t_cap = t` 는 `start_wall + t + D` 에 재생되고 (`play_wall`),
  세그먼트 `k` 는 `start_wall + (k+1)S + D − H` 에 합성·게시한다 (`compose_wall`, §3.2).
- 자막 큐·TTS 클립은 결과가 도착하는 대로 등록되고, 컴포저가 세그먼트를 만들 때 포함한다.
  **데드라인 사다리 (§3.5)**: 원래 시각의 세그먼트가 이미 합성됐으면 → 다음 세그먼트부터 보여 준다(L2, `late_ms`).
  `max_lag_s` 보다 늦으면 버린다(L3). 제때 온 것은 L1 없음(= 정상). 사다리 단계는 `deadline.event` 로 보고한다.
- 믹스: 원본(48 kHz stereo) × 덕킹 엔벨로프(TTS 가 나는 동안 `duck_db`) + TTS(mono → 양쪽) × tts 게인.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from livetrans.core.audio import OUT_RATE, db_to_gain, duck_envelope


@dataclass
class Timeline:
    delay_s: float
    start_wall: float  # 캡처 시작(t_cap = 0) 벽시계, time.time()
    segment_s: float = 1.0
    holdback_s: float = 3.0
    guard_s: float = 1.0

    def compose_wall(self, k: int) -> float:
        return self.start_wall + (k + 1) * self.segment_s + self.delay_s - self.holdback_s

    def play_wall(self, t_ms: int) -> float:
        return self.start_wall + t_ms / 1000 + self.delay_s

    @property
    def play_start_wall_ms(self) -> int:
        return round((self.start_wall + self.delay_s) * 1000)

    def segment_of(self, t_ms: int) -> int:
        return int(t_ms / 1000 // self.segment_s)

    def segment_bounds_ms(self, k: int) -> tuple[int, int]:
        return round(k * self.segment_s * 1000), round((k + 1) * self.segment_s * 1000)


class OutputRing:
    """원본 출력 오디오(48 kHz stereo float32)를 t_cap 순서로 보관. 컴포저가 세그먼트 단위로 잘라 간다."""

    def __init__(self, rate: int = OUT_RATE, keep_s: float = 120.0) -> None:
        self.rate = rate
        self._keep = int(keep_s * rate)
        self._chunks: deque[tuple[int, np.ndarray]] = deque()  # (시작 샘플, [n, 2])
        self._total = 0  # 받은 샘플 수
        self._lock = threading.Lock()  # push 는 ffmpeg 리더 스레드, slice 는 이벤트 루프

    @property
    def total_ms(self) -> int:
        return self._total * 1000 // self.rate

    def push(self, stereo: np.ndarray, t_start_ms: int | None = None) -> None:
        if stereo.ndim == 1:
            stereo = np.stack([stereo, stereo], axis=1)
        with self._lock:
            self._chunks.append((self._total, stereo.astype(np.float32, copy=False)))
            self._total += len(stereo)
            cutoff = self._total - self._keep
            while self._chunks and self._chunks[0][0] + len(self._chunks[0][1]) <= cutoff:
                self._chunks.popleft()

    def slice(self, a_ms: int, b_ms: int) -> np.ndarray:
        """[a, b) ms 구간의 stereo. 아직 안 온 부분·버린 부분은 0."""
        a = a_ms * self.rate // 1000
        b = b_ms * self.rate // 1000
        out = np.zeros((max(0, b - a), 2), dtype=np.float32)
        with self._lock:
            chunks = list(self._chunks)
        for start, chunk in chunks:
            end = start + len(chunk)
            lo, hi = max(a, start), min(b, end)
            if hi > lo:
                out[lo - a : hi - a] = chunk[lo - start : hi - start]
        return out


@dataclass
class Cue:
    seg_id: int
    kind: str  # "src" | "tgt"
    t_start_ms: int
    t_end_ms: int
    text: str
    speaker: str | None = None
    shown_from_ms: int | None = None  # 실제로 보이기 시작하는 시각 (늦으면 뒤로 밀림)


@dataclass
class TtsClip:
    seg_id: int
    t_start_ms: int  # 원래 재생 시작 (발화 시작)
    pcm: np.ndarray  # mono float32, OUT_RATE
    speaker: str | None = None
    placed_at_ms: int | None = None

    @property
    def duration_ms(self) -> int:
        return len(self.pcm) * 1000 // OUT_RATE


@dataclass
class DeadlineEvent:
    seg_id: int
    kind: str  # "subtitle" | "tts"
    level: str  # "L2" | "L3"
    late_ms: int


@dataclass
class SegmentOut:
    k: int
    pcm: np.ndarray  # stereo float32 OUT_RATE, 길이 = segment
    cues: dict[str, list[Cue]]  # kind → 이 세그먼트에 걸친 큐
    events: list[DeadlineEvent] = field(default_factory=list)


class Composer:
    """세그먼트 합성기. `add_cue`/`add_tts` 로 결과를 받고 `compose(k)` 로 세그먼트를 낸다. 시계와 무관(호출 시점은 세션이 정함)."""

    MIN_CUE_MS = 1000  # 자막 최소 표시
    MAX_CUE_MS = 7000

    def __init__(
        self,
        timeline: Timeline,
        ring: OutputRing,
        *,
        duck_db: float = -12.0,
        max_lag_s: float = 4.0,
        ramp_ms: int = 50,
    ) -> None:
        self.tl = timeline
        self.ring = ring
        self.duck_gain = db_to_gain(duck_db)
        self.max_lag_ms = int(max_lag_s * 1000)
        self.ramp = int(ramp_ms * OUT_RATE / 1000)
        self.volume = {"original": 1.0, "tts": 1.0}
        self.next_k = 0  # 다음에 합성할 세그먼트
        self._cues: list[Cue] = []
        self._tts: list[TtsClip] = []
        self.stats = {"subtitle_total": 0, "subtitle_l3": 0, "tts_total": 0, "tts_l3": 0}

    # --- 등록 ---

    def _place(
        self, kind: str, seg_id: int, t_start_ms: int
    ) -> tuple[int | None, DeadlineEvent | None]:
        """원래 시각이 이미 합성된 세그먼트면 다음 세그먼트 시작으로 민다 (L2). 너무 늦으면 None (L3)."""
        composed_until_ms = self.tl.segment_bounds_ms(self.next_k)[0]  # 이 시각 전은 이미 나갔다
        if t_start_ms >= composed_until_ms:
            return t_start_ms, None
        late = composed_until_ms - t_start_ms
        if late > self.max_lag_ms:
            return None, DeadlineEvent(seg_id, kind, "L3", late)
        return composed_until_ms, DeadlineEvent(seg_id, kind, "L2", late)

    def add_cue(
        self,
        seg_id: int,
        kind: str,
        t_start_ms: int,
        t_end_ms: int,
        text: str,
        speaker: str | None = None,
    ) -> DeadlineEvent | None:
        if not text:
            return None
        self.stats["subtitle_total"] += 1
        shown, ev = self._place("subtitle", seg_id, t_start_ms)
        if shown is None:
            self.stats["subtitle_l3"] += 1
            return ev
        end = max(t_end_ms, shown + self.MIN_CUE_MS)
        end = min(end, shown + self.MAX_CUE_MS)
        self._cues.append(Cue(seg_id, kind, t_start_ms, end, text, speaker, shown_from_ms=shown))
        return ev

    def add_tts(
        self, seg_id: int, t_start_ms: int, pcm: np.ndarray, speaker: str | None = None
    ) -> DeadlineEvent | None:
        if len(pcm) == 0:
            return None
        self.stats["tts_total"] += 1
        placed, ev = self._place("tts", seg_id, t_start_ms)
        if placed is None:
            self.stats["tts_l3"] += 1
            return ev
        # 같은 화자의 직전 클립과 겹치지 않게 뒤로 민다 (문장이 섞여 들리지 않게)
        for c in self._tts:
            if c.speaker == speaker and c.placed_at_ms is not None:
                c_end = c.placed_at_ms + c.duration_ms
                if c.placed_at_ms <= placed < c_end:
                    placed = c_end
        self._tts.append(TtsClip(seg_id, t_start_ms, pcm, speaker, placed_at_ms=placed))
        return ev

    def replace_cue(self, seg_id: int, kind: str, speaker: str | None, text: str) -> bool:
        """아직 한 번도 합성에 실리지 않은 큐의 문구를 바꾼다 (2차 패스). 이미 나간 큐면 False."""
        composed_until_ms = self.tl.segment_bounds_ms(self.next_k)[0]
        done = False
        for c in self._cues:
            if c.seg_id == seg_id and c.kind == kind and c.speaker == speaker:
                if (c.shown_from_ms or c.t_start_ms) < composed_until_ms:
                    return False
                c.text = text
                done = True
        return done

    def replace_tts(self, seg_id: int, speaker: str | None, pcm: np.ndarray) -> bool:
        """아직 합성 전인 TTS 클립의 소리를 바꾼다 (2차 패스로 번역이 달라졌을 때). 이미 나갔으면 False."""
        composed_until_ms = self.tl.segment_bounds_ms(self.next_k)[0]
        for c in self._tts:
            if c.seg_id == seg_id and c.speaker == speaker:
                if c.placed_at_ms is None or c.placed_at_ms < composed_until_ms:
                    return False
                c.pcm = pcm
                return True
        return False

    # --- 합성 ---

    def compose(self, k: int | None = None) -> SegmentOut:
        k = self.next_k if k is None else k
        a_ms, b_ms = self.tl.segment_bounds_ms(k)
        n = (b_ms - a_ms) * OUT_RATE // 1000
        a_s = a_ms * OUT_RATE // 1000
        out = self.ring.slice(a_ms, b_ms)
        if len(out) < n:
            out = np.concatenate([out, np.zeros((n - len(out), 2), dtype=np.float32)])
        tts_mix = np.zeros(n, dtype=np.float32)
        active: list[tuple[int, int]] = []
        for clip in self._tts:
            c0 = clip.placed_at_ms * OUT_RATE // 1000
            c1 = c0 + len(clip.pcm)
            lo, hi = max(c0, a_s), min(c1, a_s + n)
            if hi > lo:
                tts_mix[lo - a_s : hi - a_s] += clip.pcm[lo - c0 : hi - c0]
                active.append((lo - a_s, hi - a_s))
        env = duck_envelope(n, active, self.duck_gain, self.ramp)
        mixed = out * (env * self.volume["original"])[:, None]
        if active:
            mixed += (tts_mix * self.volume["tts"])[:, None]
        np.clip(mixed, -1.0, 1.0, out=mixed)
        cues: dict[str, list[Cue]] = {"src": [], "tgt": []}
        for c in self._cues:
            if c.shown_from_ms < b_ms and c.t_end_ms > a_ms:
                cues.setdefault(c.kind, []).append(c)
        if k >= self.next_k:
            self.next_k = k + 1
        self._gc(a_ms)
        return SegmentOut(k, mixed, cues)

    def _gc(self, before_ms: int) -> None:
        self._cues = [c for c in self._cues if c.t_end_ms > before_ms - 10000]
        self._tts = [c for c in self._tts if c.placed_at_ms + c.duration_ms > before_ms - 10000]

    @property
    def miss_rate(self) -> float:
        total = self.stats["subtitle_total"] + self.stats["tts_total"]
        if not total:
            return 0.0
        return (self.stats["subtitle_l3"] + self.stats["tts_l3"]) / total


# --- WebVTT ---


def vtt_timestamp(ms: int) -> str:
    ms = max(0, int(ms))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def vtt_segment(cues: list[Cue], *, mpegts_offset: int = 0, show_speaker: bool = True) -> str:
    """세그먼트용 WebVTT 본문. 큐 시각은 t_cap(ms) → 00:00:00 기준, X-TIMESTAMP-MAP 으로 미디어 PTS 에 맞춘다."""
    lines = ["WEBVTT", f"X-TIMESTAMP-MAP=MPEGTS:{int(mpegts_offset)},LOCAL:00:00:00.000", ""]
    for c in sorted(cues, key=lambda c: (c.shown_from_ms, c.speaker or "")):
        start = c.shown_from_ms if c.shown_from_ms is not None else c.t_start_ms
        text = c.text.replace("\n", " ").strip()
        if show_speaker and c.speaker:
            text = f"<v {c.speaker}>{text}"
        lines += [f"{vtt_timestamp(start)} --> {vtt_timestamp(c.t_end_ms)}", text, ""]
    return "\n".join(lines) + "\n"
