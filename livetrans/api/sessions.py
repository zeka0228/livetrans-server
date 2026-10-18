"""세션 REST + WebSocket (명세 §6.1, §6.2). M1 에서는 모드 C(`mic_ws`)만 지원한다."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from livetrans.pipeline.session import MicSession

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


def _sessions(app) -> dict[str, MicSession]:
    return app.state.sessions


@router.post("/api/sessions")
async def create_session(body: SessionCreate, request: Request) -> dict:
    if body.mode != "C" or body.source.type != "mic_ws":
        raise _error(
            501,
            "E_NOT_IMPLEMENTED",
            f"mode={body.mode}, source={body.source.type} 는 아직 지원하지 않습니다 (현재: mode C + mic_ws)",
        )
    app = request.app
    session = MicSession(
        app.state.settings,
        app.state.stt,
        app.state.vad_factory(),
        src_lang=None if body.src_lang in (None, "auto") else body.src_lang,
    )
    await session.start()
    _sessions(app)[session.session_id] = session
    return {
        "session_id": session.session_id,
        "delay_s": None,
        "hls_url": None,
        "ws_url": f"/ws/sessions/{session.session_id}",
    }


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
                action = _parse_control(msg["text"])
                if action == "start":
                    session.begin()
                elif action == "stop" and stop_task is None:
                    stop_task = asyncio.create_task(session.stop())
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


def _parse_control(text: str) -> str | None:
    """`{"type":"control","payload":{"action":...}}` 의 action. 그 외 메시지(config.update 등)는 아직 무시."""
    try:
        msg = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(msg, dict) or msg.get("type") != "control":
        return None
    payload = msg.get("payload")
    return payload.get("action") if isinstance(payload, dict) else None
