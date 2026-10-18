"""FastAPI 앱. 모델 로딩은 lifespan 에서 1회 (명세 §2.3)."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from livetrans.api.rest import router as rest_router
from livetrans.core.config import Settings, load_settings
from livetrans.core.gpu import prepare_cuda_dlls

log = logging.getLogger("livetrans")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        added = prepare_cuda_dlls()
        if added:
            log.info("CUDA DLL 경로 등록: %s", ", ".join(map(str, added)))
        app.state.settings = settings
        app.state.models_loaded = []
        yield

    app = FastAPI(title="livetrans", version="0.1.0", lifespan=lifespan)
    app.include_router(rest_router)
    return app


app = create_app()
