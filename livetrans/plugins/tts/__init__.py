"""TTS 엔진 생성. 정식 플러그인 레지스트리(entry points)는 M2 에서 `core/registry.py` 로 옮긴다."""

from __future__ import annotations

from typing import Any

from livetrans.plugins.tts.base import TTSEngine


def create_tts(name: str, cfg: dict[str, Any]) -> TTSEngine:
    kind = cfg.get("type")
    if kind == "system":
        from livetrans.plugins.tts.system import SystemTTS

        return SystemTTS(name)
    if kind == "openjtalk":
        from livetrans.plugins.tts.openjtalk import OpenJTalkTTS

        return OpenJTalkTTS(name)
    if kind == "fake":
        from livetrans.plugins.tts.fake import FakeTTS

        return FakeTTS(name)
    raise ValueError(f"알 수 없는 TTS 엔진 type: {kind} (engine={name})")
