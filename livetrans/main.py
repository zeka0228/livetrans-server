"""FastAPI 앱. 모델 로딩은 lifespan 에서 1회 (명세 §2.3)."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI

from livetrans.api.capture import router as capture_router
from livetrans.api.rest import router as rest_router
from livetrans.api.sessions import router as sessions_router
from livetrans.core.config import Settings, load_settings
from livetrans.core.gpu import prepare_cuda_dlls
from livetrans.plugins.source.loopback import CaptureBackend
from livetrans.plugins.stt import create_stt
from livetrans.plugins.stt.base import STTEngine

log = logging.getLogger("livetrans")

VadFactory = Callable[[], Callable[[np.ndarray], float]]


def _silero_factory() -> Callable[[np.ndarray], float]:
    from livetrans.pipeline.vad import StreamingSileroVAD

    return StreamingSileroVAD()


def create_app(
    settings: Settings | None = None,
    *,
    stt: STTEngine | None = None,
    vad_factory: VadFactory | None = None,
    capture: CaptureBackend | None = None,
) -> FastAPI:
    """stt·vad_factory·capture 는 테스트에서 가짜를 주입할 때만 넘긴다."""
    settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        added = prepare_cuda_dlls()
        if added:
            log.info("CUDA DLL 경로 등록: %s", ", ".join(map(str, added)))
        engine = stt or create_stt(settings.stt)
        log.info("STT 모델 로드: %s (%s)", engine.name, engine.model_id)
        await engine.load()
        app.state.settings = settings
        app.state.stt = engine
        app.state.vad_factory = vad_factory or _silero_factory
        app.state.capture = capture or CaptureBackend()
        app.state.models_loaded = [f"stt:{engine.name}:{engine.model_id}"]
        app.state.sessions = {}
        yield
        for session in list(app.state.sessions.values()):
            await session.abort()
        await engine.close()

    app = FastAPI(title="livetrans", version="0.1.0", lifespan=lifespan)
    app.include_router(rest_router)
    app.include_router(sessions_router)
    app.include_router(capture_router)
    return app


app = create_app()
