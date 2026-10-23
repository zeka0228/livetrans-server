"""실제 Ollama 번역 (로컬 서버 + 기본 모델이 있을 때만). 없으면 skip."""

from __future__ import annotations

import httpx
import pytest

from livetrans.core.config import MtConfig
from livetrans.plugins.mt.ollama_llm import OllamaTranslator

CFG = MtConfig(engine="ollama_llm")


def _available() -> bool:
    try:
        r = httpx.get(f"{CFG.url}/api/tags", timeout=2.0)
        names = {m.get("name", "") for m in r.json().get("models", [])}
        return CFG.model in names
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(not _available(), reason=f"Ollama 또는 모델 {CFG.model} 없음")


async def test_translates_japanese_to_korean_with_context():
    mt = OllamaTranslator(CFG)
    await mt.load()
    try:
        out = await mt.translate(
            "今日はいい天気ですね",
            src="ja",
            tgt="ko",
            context=[("こんにちは", "안녕하세요")],
            glossary={"配信": "방송"},
        )
    finally:
        await mt.close()
    assert out and out != "今日はいい天気ですね"
    assert any("가" <= ch <= "힣" for ch in out)  # 한글이 들어 있다
