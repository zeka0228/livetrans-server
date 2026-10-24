"""HLS 정적 서빙 (명세 §6.1 `/hls/{id}/master.m3u8`). 세션 디렉터리의 플레이리스트·세그먼트·VTT 만 내준다."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from livetrans.output.hls import CONTENT_TYPES

router = APIRouter()


@router.get("/hls/{session_id}/{name}")
async def hls_file(session_id: str, name: str, request: Request) -> FileResponse:
    if Path(name).name != name or not session_id.isalnum():
        raise HTTPException(404)
    root = Path(request.app.state.settings.output.hls_dir) / session_id
    path = root / name
    if not path.is_file():
        raise HTTPException(404, detail={"code": "E_HLS_NOT_FOUND", "message": f"{name} 없음"})
    media_type = CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")
    return FileResponse(
        path,
        media_type=media_type,
        headers={"Cache-Control": "no-store"},  # CORS 헤더는 CORSMiddleware 가 붙인다 (server.cors_origins)
    )
