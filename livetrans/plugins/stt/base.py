"""STT 엔진 인터페이스 (명세 §4.3)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from livetrans.pipeline.segmenter import SpeechSegment


@dataclass
class Word:
    text: str
    t0_ms: int  # t_cap 기준 절대 시각
    t1_ms: int
    prob: float


@dataclass
class Transcript:
    seg_id: int
    text: str
    lang: str  # ISO 639-1
    lang_prob: float
    words: list[Word] = field(default_factory=list)
    avg_logprob: float = 0.0
    latency_ms: int = 0
    speaker: str | None = None  # v0.4: 화자 조건부 엔진이면 대상 화자, 아니면 None


class STTEngine(Protocol):
    name: str
    model_id: str
    languages: tuple[str, ...]
    supports_speaker_mask: bool  # v0.4: True 면 speaker·seg.speaker_turns 를 사용 (DiCoW)

    async def load(self) -> None: ...

    async def transcribe(
        self,
        seg: SpeechSegment,
        *,
        lang: str | None,
        prompt: str | None,
        final: bool,
        speaker: str | None = None,
    ) -> Transcript: ...

    async def close(self) -> None: ...
