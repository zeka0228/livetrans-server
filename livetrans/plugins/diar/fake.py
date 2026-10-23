"""테스트용 가짜 화자 분할기 — 발화마다 정해진 화자 배치를 돌려준다 (GPU 없음). 스트리밍용 가짜도 함께."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from livetrans.core.speaker import SpeakerTurn
from livetrans.pipeline.segmenter import SpeechSegment

TurnsFn = Callable[[SpeechSegment], list[SpeakerTurn]]


def one_speaker(seg: SpeechSegment) -> list[SpeakerTurn]:
    return [SpeakerTurn("S0", seg.t_start_ms, seg.t_end_ms)]


class FakeDiarizer:
    def __init__(self, turns_fn: TurnsFn) -> None:
        self._fn = turns_fn
        self.calls: list[int] = []
        self.resets = 0

    async def diarize(self, seg: SpeechSegment) -> list[SpeakerTurn]:
        self.calls.append(seg.seg_id)
        return self._fn(seg)

    def reset(self) -> None:
        self.resets += 1


ActivityFn = Callable[
    [int, int], list[tuple[str, int, int]]
]  # (from_ms, to_ms) → [(speaker, a, b)]


class FakeStreamDiarizer:
    """스트리밍 가짜: 정해진 활동 표에서 [이전 확정, 새 확정) 구간을 잘라 돌려준다. 오디오는 그대로 들고 있다."""

    def __init__(
        self,
        activity_fn: ActivityFn,
        *,
        hop_s: float = 1.0,
        right_context_s: float = 0.5,
        latency_ms: int = 7,
    ) -> None:
        self._fn = activity_fn
        self._hop = int(hop_s * 16000)
        self._right = int(right_context_s * 16000)
        self._chunks: list[np.ndarray] = []
        self._total = 0
        self._committed = 0
        self._last_step = 0
        self.steps = 0
        self.resets = 0
        self.last_latency_ms = latency_ms
        self.last_window_s = 0.0

    def push(self, pcm: np.ndarray) -> None:
        self._chunks.append(np.asarray(pcm, dtype=np.float32))
        self._total += len(pcm)

    @property
    def total_ms(self) -> int:
        return self._total * 1000 // 16000

    @property
    def committed_ms(self) -> int:
        return self._committed * 1000 // 16000

    @property
    def ready(self) -> bool:
        return self._total - self._last_step >= self._hop

    def audio(self, t_start_ms: int, t_end_ms: int) -> np.ndarray:
        buf = np.concatenate(self._chunks) if self._chunks else np.zeros(0, dtype=np.float32)
        a, b = t_start_ms * 16, t_end_ms * 16
        out = np.zeros(max(0, b - a), dtype=np.float32)
        lo, hi = max(a, 0), min(b, len(buf))
        if hi > lo:
            out[lo - a : hi - a] = buf[lo:hi]
        return out

    def committed_turns(self, t_start_ms: int, t_end_ms: int):
        from livetrans.core.speaker import SpeakerTurn

        turns = []
        for s, a, b in self._fn(0, self.committed_ms):
            lo, hi = max(a, t_start_ms), min(b, t_end_ms)
            if hi > lo:
                turns.append(SpeakerTurn(s, lo, hi))
        turns.sort(key=lambda t: (t.t_start_ms, t.speaker))
        return turns

    async def step(self, *, final: bool = False):
        from livetrans.core.speaker import SpeakerTurn

        self.steps += 1
        self._last_step = self._total
        commit_to = self._total if final else max(self._committed, self._total - self._right)
        prev_ms, commit_ms = self._committed * 1000 // 16000, commit_to * 1000 // 16000
        self._committed = commit_to
        turns = []
        for s, a, b in self._fn(prev_ms, commit_ms):
            lo, hi = max(a, prev_ms), min(b, commit_ms)
            if hi > lo:
                turns.append(SpeakerTurn(s, lo, hi))
        turns.sort(key=lambda t: (t.t_start_ms, t.speaker))
        return turns

    def reset(self) -> None:
        self.resets += 1


class FakeDiarizerEngine:
    name = "fake"
    model_id = "fake"

    def __init__(
        self,
        turns_fn: TurnsFn = one_speaker,
        activity_fn: ActivityFn | None = None,
        **stream_kw,
    ) -> None:
        self._fn = turns_fn
        self._activity_fn = activity_fn
        self._stream_kw = stream_kw
        self.created: list[FakeDiarizer] = []
        self.streams: list[FakeStreamDiarizer] = []

    def create_stream(self) -> FakeStreamDiarizer:
        if self._activity_fn is None:
            raise NotImplementedError(
                "activity_fn 이 없는 가짜 엔진은 스트림 모드를 지원하지 않는다"
            )
        s = FakeStreamDiarizer(self._activity_fn, **self._stream_kw)
        self.streams.append(s)
        return s

    async def load(self) -> None:
        return None

    def create(self) -> FakeDiarizer:
        d = FakeDiarizer(self._fn)
        self.created.append(d)
        return d

    async def close(self) -> None:
        return None
