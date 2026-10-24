"""번역(MT) 인터페이스 (명세 §4.4)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

ContextPair = tuple[str, str]  # (원문, 번역)


@dataclass
class Translation:
    seg_id: int
    text: str
    src_lang: str
    tgt_lang: str
    engine: str
    fallback: bool
    latency_ms: int
    speaker: str | None = None  # v0.4
    source_fixed: str | None = (
        None  # 모델이 고른/교정한 원문 (후보 재채점·2차 패스, S-30). 원문과 같으면 None
    )
    confidence: float | None = None  # 모델 확신도 0~1 (요청했을 때만)


class TranslationError(RuntimeError):
    """엔진이 번역을 내지 못함 (타임아웃·연결 실패·형식 오류). 폴백 체인이 다음 엔진으로 넘어간다."""


class Translator(Protocol):
    name: str
    model_id: str

    async def load(self) -> None: ...

    async def health(self) -> bool: ...

    async def translate(
        self,
        text: str,
        *,
        src: str,
        tgt: str,
        context: list[ContextPair],
        glossary: dict[str, str],
        length_budget_s: float | None = None,
    ) -> str: ...

    async def close(self) -> None: ...
