"""TTS REST (명세 §6.1 `/api/voices`, D-018 `/api/tts`)."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

from livetrans.core.voices import NoVoiceError
from livetrans.plugins.tts.base import TTSError

log = logging.getLogger("livetrans.api.tts")

router = APIRouter(prefix="/api")

MIN_RATE, MAX_RATE = 0.5, 2.0


class TtsRequest(BaseModel):
    text: str
    lang: str | None = None
    voice_ref: str | None = None
    rate: float = 1.0


def _error(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message})


@router.get("/voices")
async def voices(request: Request, lang: str | None = None) -> list[dict]:
    return [v.to_dict() for v in request.app.state.voices.voices(lang)]


@router.get("/tts")
async def tts_get(
    request: Request,
    text: str,
    lang: str | None = None,
    voice_ref: str | None = None,
    rate: float = 1.0,
) -> Response:
    """앱이 오디오 플레이어에 URL 로 바로 넘길 수 있도록 GET 도 제공한다."""
    return await _synthesize(
        request, TtsRequest(text=text, lang=lang, voice_ref=voice_ref, rate=rate)
    )


@router.post("/tts")
async def tts_post(body: TtsRequest, request: Request) -> Response:
    return await _synthesize(request, body)


async def _synthesize(request: Request, req: TtsRequest) -> Response:
    settings = request.app.state.settings
    text = req.text.strip()
    if not text:
        raise _error(400, "E_TTS_TEXT", "읽을 텍스트가 비어 있습니다")
    if len(text) > settings.tts.max_chars:
        raise _error(400, "E_TTS_TEXT", f"텍스트가 너무 깁니다 (최대 {settings.tts.max_chars}자)")
    if not MIN_RATE <= req.rate <= MAX_RATE:
        raise _error(400, "E_TTS_PARAM", f"rate 는 {MIN_RATE}~{MAX_RATE} 사이여야 합니다")
    try:
        result = await request.app.state.voices.synthesize(
            text, lang=req.lang, voice_ref=req.voice_ref, rate=req.rate
        )
    except NoVoiceError as e:
        raise _error(404, "E_TTS_NO_VOICE", str(e)) from e
    except TTSError as e:
        log.exception("TTS 실패")
        raise _error(500, "E_TTS_FAILED", str(e)) from e
    return Response(
        content=result.wav,
        media_type="audio/wav",
        headers={
            "X-Voice-Ref": result.voice.voice_ref,
            "X-Duration-Ms": str(result.duration_ms),
            "X-Cache": "hit" if result.cached else "miss",
            "Cache-Control": "private, max-age=3600",
        },
    )
