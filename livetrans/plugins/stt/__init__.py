"""STT 엔진 생성. 정식 플러그인 레지스트리(entry points)는 M2 에서 `core/registry.py` 로 옮긴다."""

from __future__ import annotations

from livetrans.core.config import SttConfig
from livetrans.plugins.stt.base import STTEngine


def create_stt(cfg: SttConfig) -> STTEngine:
    """설정의 엔진을 만든다. `fallback` 이 있으면 로드 실패 시 그 엔진으로 넘어가는 래퍼로 감싼다."""
    engine = _create(cfg.engine, cfg.model, cfg)
    if cfg.fallback and cfg.fallback != cfg.engine:
        from livetrans.plugins.stt.fallback import FallbackSTT

        return FallbackSTT(engine, _create(cfg.fallback, cfg.fallback_model, cfg))
    return engine


def _create(engine: str, model: str, cfg: SttConfig) -> STTEngine:
    cfg = cfg.model_copy(update={"engine": engine, "model": model})
    if engine == "dicow":
        from livetrans.plugins.stt.dicow import DiCoWSTT

        return DiCoWSTT(cfg)
    if engine == "faster_whisper":
        from livetrans.plugins.stt.faster_whisper import FasterWhisperSTT

        return FasterWhisperSTT(cfg)
    if cfg.engine == "fake":
        from livetrans.plugins.stt.fake import FakeSTT

        return FakeSTT()
    raise ValueError(f"알 수 없는 STT 엔진: {engine}")
