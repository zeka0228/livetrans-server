"""세션 REST + WebSocket (명세 §6.1, §6.2).

지원 조합: 모드 C + `mic_ws`(앱 마이크), 모드 B + `loopback`(PC 소리 캡처, D-016).
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from livetrans.output.hls import AsyncHlsWriter, HlsWriter
from livetrans.pipeline.session import Session
from livetrans.plugins.source.ffmpeg_in import FfmpegSource
from livetrans.plugins.source.loopback import parse_target
from livetrans.plugins.source.mic_ws import MicWsSource

log = logging.getLogger("livetrans.api")

router = APIRouter()

WS_CLOSE_UNKNOWN_SESSION = 4404
WS_CLOSE_ALREADY_CONNECTED = 4409


class SourceSpec(BaseModel):
    type: str
    params: dict[str, Any] = Field(default_factory=dict)


class TtsSpec(BaseModel):
    enabled: bool = False
    voice_ref: str | None = None
    duck_db: float = -12


class DisplaySpec(BaseModel):
    show_original: bool = True
    show_speaker: bool = True


class SessionCreate(BaseModel):
    mode: str
    source: SourceSpec
    src_lang: str | None = "auto"  # "auto" 와 null 모두 자동 감지 (API 계약 Q-API-1)
    tgt_lang: str = "ko"
    delay_s: int | None = None
    tts: TtsSpec = TtsSpec()
    display: DisplaySpec = DisplaySpec()


def _error(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message})


def _sessions(app) -> dict[str, Session]:
    return app.state.sessions


def _diarizer_kwargs(app) -> dict:
    """화자 분할 설정에 따라 세션에 넘길 분할기: stream(D-028) 이면 스트리밍, 아니면 발화 단위 (D-027)."""
    engine = app.state.diar
    if engine is None:
        return {}
    mode = app.state.settings.diar.mode
    if mode == "stream" and hasattr(engine, "create_stream"):
        try:
            return {"stream_diarizer": engine.create_stream()}
        except NotImplementedError:
            pass  # 스트림을 지원하지 않는 엔진 → 발화 단위
    return {"diarizer": engine.create()}


@router.post("/api/sessions")
async def create_session(body: SessionCreate, request: Request) -> dict:
    app = request.app
    sample_rate = app.state.settings.audio.sample_rate
    combo = (body.mode, body.source.type)
    session_id = uuid.uuid4().hex[:12]
    extra: dict[str, Any] = {}
    if combo == ("C", "mic_ws"):
        source = MicWsSource(sample_rate)
    elif combo == ("B", "loopback"):
        source = await _loopback_source(app, body.source.params, sample_rate)
    elif body.mode == "A" and body.source.type in ("obs_stream", "file", "loopback"):
        source, extra = await _mode_a_setup(app, body, session_id, sample_rate)
    else:
        raise _error(
            501,
            "E_NOT_IMPLEMENTED",
            f"mode={body.mode}, source={body.source.type} 는 아직 지원하지 않습니다"
            " (현재: mode C + mic_ws, mode B + loopback, mode A + obs_stream|file|loopback)",
        )
    session = Session(
        app.state.settings,
        app.state.stt,
        app.state.vad_factory(),
        source,
        mode=body.mode,
        src_lang=None if body.src_lang in (None, "auto") else body.src_lang,
        tgt_lang=body.tgt_lang,
        session_id=session_id,
        rescue_stt=getattr(app.state, "stt_rescue", None),
        translator=getattr(app.state, "mt", None),
        **_diarizer_kwargs(app),
        **extra,
    )
    if isinstance(source, FfmpegSource):
        source.on_output = session.ring.push if session.ring is not None else None
    await session.start()
    _sessions(app)[session.session_id] = session
    return {
        "session_id": session.session_id,
        "delay_s": session.delay_s,
        "hls_url": session.hls_url,
        "ws_url": f"/ws/sessions/{session.session_id}",
    }


async def _mode_a_setup(app, body: SessionCreate, session_id: str, sample_rate: int):
    """모드 A: 입력(ffmpeg 또는 loopback) + HLS 출력 준비. 세션에 넘길 추가 인자를 돌려준다."""
    settings = app.state.settings
    ffmpeg = getattr(app.state, "ffmpeg", None)
    if not ffmpeg:
        raise _error(
            501,
            "E_NOT_IMPLEMENTED",
            "ffmpeg 가 없어 모드 A(HLS 출력)를 쓸 수 없습니다 (tools.ffmpeg)",
        )
    params = body.source.params
    hls_dir = Path(settings.output.hls_dir) / session_id
    if body.source.type == "loopback":
        source = await _loopback_source(app, params, sample_rate)
        video_path = None
    else:
        url = params.get("url") or params.get("path")
        if not url:
            raise _error(
                400, "E_SOURCE_PARAM", "source.params.url (srt://… 또는 파일 경로) 가 필요합니다"
            )
        if body.source.type == "file" and not Path(str(url)).is_file():
            raise _error(400, "E_SOURCE_PARAM", f"파일이 없습니다: {url}")
        video = bool(params.get("video", body.source.type == "obs_stream"))
        video_path = hls_dir / "raw.ts" if video else None
        source = FfmpegSource(
            str(url),
            ffmpeg=ffmpeg,
            realtime=body.source.type == "file" or bool(params.get("realtime")),
            record_video=video,
            record_path=video_path,
            sample_rate=sample_rate,
        )
    src = "und" if body.src_lang in (None, "auto") else str(body.src_lang)
    writer = AsyncHlsWriter(
        HlsWriter(
            hls_dir,
            ffmpeg=ffmpeg,
            segment_s=settings.timeline.segment_s,
            video_path=video_path,
            video_codec=settings.output.video_codec,
            langs=(body.tgt_lang, src),
        )
    )
    delay = float(body.delay_s) if body.delay_s else float(settings.timeline.default_delay_s)
    return source, {
        "delay_s": delay,
        "hls_writer": writer,
        "voices": getattr(app.state, "voices", None),
        "tts_enabled": body.tts.enabled,
        "tts_voice_ref": body.tts.voice_ref,
        "show_speaker": body.display.show_speaker,
    }


async def _loopback_source(app, params: dict[str, Any], sample_rate: int):
    backend = app.state.capture
    if not backend.available():
        raise _error(
            503, "E_CAPTURE_UNAVAILABLE", "캡처 헬퍼가 없습니다 (scripts\\build_wincap.bat)"
        )
    target = str(params.get("target", "system"))
    try:
        parse_target(target)
    except ValueError as e:
        raise _error(400, "E_CAPTURE_TARGET", str(e)) from e
    # 창 목록은 UI 용. 대상은 살아 있는 프로세스면 된다 (창 없는 프로세스의 소리도 캡처 가능)
    if target != "system" and (
        not backend.app_capture_supported() or not backend.process_alive(int(target[4:]))
    ):
        raise _error(
            400,
            "E_CAPTURE_TARGET",
            f"캡처할 수 없는 대상입니다 (창이 닫혔거나 미지원): {target}",
        )
    return backend.create_source(target, sample_rate)


@router.post("/api/sessions/{session_id}/stop")
async def stop_session(session_id: str, request: Request) -> dict:
    session = _sessions(request.app).get(session_id)
    if session is None:
        raise _error(404, "E_SESSION_NOT_FOUND", f"세션이 없습니다: {session_id}")
    await session.stop()
    _sessions(request.app).pop(session_id, None)
    return {"session_id": session_id, "stopped": True}


@router.websocket("/ws/sessions/{session_id}")
async def session_ws(ws: WebSocket, session_id: str) -> None:
    session = _sessions(ws.app).get(session_id)
    await ws.accept()
    if session is None:
        await ws.close(code=WS_CLOSE_UNKNOWN_SESSION, reason="unknown session")
        return
    if getattr(session, "ws_attached", False):
        await ws.close(code=WS_CLOSE_ALREADY_CONNECTED, reason="already connected")
        return
    session.ws_attached = True

    async def sender() -> None:
        while True:
            env = await session.outbox.get()
            await ws.send_text(env.to_json())
            if env.type == "session.state" and env.payload.get("state") == "stopped":
                return

    send_task = asyncio.create_task(sender())
    stop_task: asyncio.Task | None = None
    try:
        while not send_task.done():
            recv = asyncio.create_task(ws.receive())
            done, _ = await asyncio.wait({recv, send_task}, return_when=asyncio.FIRST_COMPLETED)
            if recv not in done:
                recv.cancel()
                break
            msg = recv.result()
            if msg["type"] == "websocket.disconnect":
                raise WebSocketDisconnect(msg.get("code", 1000))
            if msg.get("bytes") is not None:
                session.feed(msg["bytes"])
            elif msg.get("text"):
                kind, payload = _parse_message(msg["text"])
                action = payload.get("action") if kind == "control" else None
                if action == "start":
                    await session.begin()
                elif action == "stop" and stop_task is None:
                    stop_task = asyncio.create_task(session.stop())
                elif kind == "config.update":
                    session.update_config(payload)
        # stopped 를 보냈으면 서버가 닫는다
        await send_task
        await ws.close()
    except WebSocketDisconnect:
        send_task.cancel()
        if stop_task is None and not session.stopped:
            await session.abort()
    except Exception:
        log.exception("WS 처리 오류 session=%s", session_id)
        send_task.cancel()
        await session.abort()
    finally:
        _sessions(ws.app).pop(session_id, None)


def _parse_message(text: str) -> tuple[str | None, dict]:
    """앱 → 서버 텍스트 메시지 `{"type": ..., "payload": {...}}` → (type, payload). 형식이 틀리면 (None, {})."""
    try:
        msg = json.loads(text)
    except json.JSONDecodeError:
        return None, {}
    if not isinstance(msg, dict):
        return None, {}
    payload = msg.get("payload")
    return msg.get("type"), payload if isinstance(payload, dict) else {}


def _parse_control(text: str) -> str | None:
    kind, payload = _parse_message(text)
    return payload.get("action") if kind == "control" else None
