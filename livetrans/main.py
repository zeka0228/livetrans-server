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
from livetrans.api.tts import router as tts_router
from livetrans.core.config import Settings, load_settings
from livetrans.core.gpu import prepare_cuda_dlls
from livetrans.core.voices import VoiceRegistry
from livetrans.plugins.source.loopback import CaptureBackend
from livetrans.plugins.stt import create_stt
from livetrans.plugins.stt.base import STTEngine
from livetrans.plugins.tts import create_tts
from livetrans.plugins.tts.base import TTSEngine

log = logging.getLogger("livetrans")

VadFactory = Callable[[], Callable[[np.ndarray], float]]


def _silero_factory() -> Callable[[np.ndarray], float]:
    from livetrans.pipeline.vad import StreamingSileroVAD

    return StreamingSileroVAD()


async def _load_tts(settings: Settings) -> dict[str, TTSEngine]:
    """설정의 TTS 엔진을 만들고 로드한다. 실패한 엔진은 경고만 남기고 뺀다 (서버는 계속 뜬다)."""
    engines: dict[str, TTSEngine] = {}
    for name, cfg in settings.tts.engines.items():
        try:
            tts = create_tts(name, cfg)
            await tts.load()
        except Exception as e:  # noqa: BLE001 — 엔진 하나가 안 돼도 서버는 떠야 한다
            log.warning("TTS 엔진 %s 로드 실패: %s", name, e)
            continue
        log.info("TTS 엔진 로드: %s (%s)", name, cfg.get("type"))
        engines[name] = tts
    return engines


def create_app(
    settings: Settings | None = None,
    *,
    stt: STTEngine | None = None,
    vad_factory: VadFactory | None = None,
    capture: CaptureBackend | None = None,
    tts_engines: dict[str, TTSEngine] | None = None,
) -> FastAPI:
    """stt·vad_factory·capture·tts_engines 는 테스트에서 가짜를 주입할 때만 넘긴다."""
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
        voices = VoiceRegistry(
            tts_engines if tts_engines is not None else await _load_tts(settings),
            settings.tts.defaults,
            cache_items=settings.tts.cache_items,
        )
        await voices.refresh()
        app.state.voices = voices
        app.state.models_loaded = [f"stt:{engine.name}:{engine.model_id}"] + [
            f"tts:{name}" for name in voices.engines
        ]
        app.state.sessions = {}
        yield
        for session in list(app.state.sessions.values()):
            await session.abort()
        await engine.close()
        for tts in voices.engines.values():
            await tts.close()

    app = FastAPI(title="livetrans", version="0.1.0", lifespan=lifespan)
    app.include_router(rest_router)
    app.include_router(sessions_router)
    app.include_router(capture_router)
    app.include_router(tts_router)
    return app


app = create_app()
