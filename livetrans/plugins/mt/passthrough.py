"""원문을 그대로 돌려주는 최종 폴백 (명세 §4.4)."""

from __future__ import annotations

from livetrans.plugins.mt.base import ContextPair


class PassthroughTranslator:
    name = "passthrough"
    model_id = "-"

    async def load(self) -> None:
        return None

    async def health(self) -> bool:
        return True

    async def translate(
        self,
        text: str,
        *,
        src: str,
        tgt: str,
        context: list[ContextPair],
        glossary: dict[str, str],
        length_budget_s: float | None = None,
    ) -> str:
        return text

    async def close(self) -> None:
        return None
