"""STT 엔진 생성. 정식 플러그인 레지스트리(entry points)는 M2 에서 `core/registry.py` 로 옮긴다."""

from __future__ import annotations

from livetrans.core.config import SttConfig
from livetrans.plugins.stt.base import STTEngine


def create_stt(cfg: SttConfig) -> STTEngine:
    if cfg.engine == "faster_whisper":
        from livetrans.plugins.stt.faster_whisper import FasterWhisperSTT

        return FasterWhisperSTT(cfg)
    if cfg.engine == "fake":
        from livetrans.plugins.stt.fake import FakeSTT

        return FakeSTT()
    raise ValueError(f"알 수 없는 STT 엔진: {cfg.engine}")
