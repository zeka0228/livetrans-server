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
from livetrans.core.config import Settings, SttConfig, load_settings
from livetrans.core.gpu import prepare_cuda_dlls
from livetrans.core.voices import VoiceRegistry
from livetrans.plugins.diar import create_diarizer
from livetrans.plugins.diar.base import DiarizerEngine
from livetrans.plugins.mt import create_translator
from livetrans.plugins.mt.base import Translator
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


async def _load_diarizer(
    settings: Settings, injected: DiarizerEngine | None
) -> DiarizerEngine | None:
    """설정의 화자 분할 엔진을 로드한다. 실패하면 경고만 남기고 화자 없이 뜬다 (자막은 계속 나와야 한다)."""
    engine = injected or create_diarizer(settings.diar)
    if engine is None:
        return None
    try:
        await engine.load()
    except Exception as e:  # noqa: BLE001 — 토큰·모델 문제로 화자 분할이 안 떠도 서버는 떠야 한다
        log.warning(
            "화자 분할 엔진 %s(%s) 로드 실패 → 화자 없이 진행: %s", engine.name, engine.model_id, e
        )
        return None
    log.info("화자 분할 로드: %s (%s)", engine.name, engine.model_id)
    return engine


async def _load_rescue(settings: Settings, injected: STTEngine | None) -> STTEngine | None:
    """환각 복구용 두 번째 STT. 실패해도 서버는 뜬다 (복구 없이 비운다)."""
    stt = settings.stt
    if injected is not None:
        engine = injected
    elif stt.rescue and stt.rescue not in (stt.engine, "none"):
        engine = create_stt(
            SttConfig(
                **(
                    stt.model_dump()
                    | {"engine": stt.rescue, "model": stt.rescue_model, "fallback": None}
                )
            )
        )
    else:
        return None
    try:
        await engine.load()
    except Exception as e:  # noqa: BLE001
        log.warning(
            "구조 STT %s(%s) 로드 실패 → 환각 복구 없이 진행: %s", engine.name, engine.model_id, e
        )
        return None
    log.info("구조 STT 로드: %s (%s)", engine.name, engine.model_id)
    return engine


async def _load_translator(settings: Settings, injected: Translator | None) -> Translator | None:
    """번역 체인 로드. Ollama 가 없으면 체인이 passthrough 만 남긴다. 전부 실패하면 번역 없이 (자막만)."""
    engine = injected or create_translator(settings.mt)
    try:
        await engine.load()
    except Exception as e:  # noqa: BLE001
        log.warning("번역 엔진 로드 실패 → 번역 없이 진행: %s", e)
        return None
    log.info("번역 엔진: %s (%s)", engine.name, engine.model_id)
    return engine


def create_app(
    settings: Settings | None = None,
    *,
    stt: STTEngine | None = None,
    vad_factory: VadFactory | None = None,
    capture: CaptureBackend | None = None,
    tts_engines: dict[str, TTSEngine] | None = None,
    diar: DiarizerEngine | None = None,
    stt_rescue: STTEngine | None = None,
    translator: Translator | None = None,
) -> FastAPI:
    """stt·vad_factory·capture·tts_engines·diar 는 테스트에서 가짜를 주입할 때만 넘긴다."""
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
        app.state.stt_rescue = await _load_rescue(settings, stt_rescue)
        app.state.diar = await _load_diarizer(settings, diar)
        app.state.mt = await _load_translator(settings, translator)
        app.state.vad_factory = vad_factory or _silero_factory
        app.state.capture = capture or CaptureBackend()
        voices = VoiceRegistry(
            tts_engines if tts_engines is not None else await _load_tts(settings),
            settings.tts.defaults,
            cache_items=settings.tts.cache_items,
        )
        await voices.refresh()
        app.state.voices = voices
        app.state.models_loaded = [f"stt:{engine.name}:{engine.model_id}"]
        if app.state.stt_rescue is not None:
            r = app.state.stt_rescue
            app.state.models_loaded.append(f"stt_rescue:{r.name}:{r.model_id}")
        if app.state.diar is not None:
            app.state.models_loaded.append(f"diar:{app.state.diar.name}:{app.state.diar.model_id}")
        if app.state.mt is not None and app.state.mt.name != "passthrough":
            app.state.models_loaded.append(f"mt:{app.state.mt.name}:{app.state.mt.model_id}")
        app.state.models_loaded += [f"tts:{name}" for name in voices.engines]
        app.state.sessions = {}
        yield
        for session in list(app.state.sessions.values()):
            await session.abort()
        await engine.close()
        if app.state.stt_rescue is not None:
            await app.state.stt_rescue.close()
        if app.state.diar is not None:
            await app.state.diar.close()
        if app.state.mt is not None:
            await app.state.mt.close()
        for tts in voices.engines.values():
            await tts.close()

    app = FastAPI(title="livetrans", version="0.1.0", lifespan=lifespan)
    app.include_router(rest_router)
    app.include_router(sessions_router)
    app.include_router(capture_router)
    app.include_router(tts_router)
    return app


app = create_app()
