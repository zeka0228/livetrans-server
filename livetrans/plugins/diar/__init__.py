"""화자 분할 엔진 생성. `engine: none` 이면 None (세션은 화자 없이 동작)."""

from __future__ import annotations

from livetrans.core.config import DiarConfig
from livetrans.plugins.diar.base import DiarizerEngine


def create_diarizer(cfg: DiarConfig) -> DiarizerEngine | None:
    if cfg.engine in ("none", None, ""):
        return None
    if cfg.engine == "pyannote":
        from livetrans.plugins.diar.pyannote import PyannoteDiarizerEngine

        return PyannoteDiarizerEngine(cfg)
    if cfg.engine == "fake":
        from livetrans.plugins.diar.fake import FakeDiarizerEngine

        return FakeDiarizerEngine()
    raise ValueError(f"알 수 없는 화자 분할 엔진: {cfg.engine}")
