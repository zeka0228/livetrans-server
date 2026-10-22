"""테스트용 가짜 화자 분할기 — 발화마다 정해진 화자 배치를 돌려준다 (GPU 없음)."""

from __future__ import annotations

from collections.abc import Callable

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


class FakeDiarizerEngine:
    name = "fake"
    model_id = "fake"

    def __init__(self, turns_fn: TurnsFn = one_speaker) -> None:
        self._fn = turns_fn
        self.created: list[FakeDiarizer] = []

    async def load(self) -> None:
        return None

    def create(self) -> FakeDiarizer:
        d = FakeDiarizer(self._fn)
        self.created.append(d)
        return d

    async def close(self) -> None:
        return None
