"""HLS 정적 서빙 (명세 §6.1 `/hls/{id}/master.m3u8`). 세션 디렉터리의 플레이리스트·세그먼트·VTT 만 내준다."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from livetrans.output.hls import CONTENT_TYPES

router = APIRouter()


@router.get("/hls/{session_id}/{name}")
async def hls_file(session_id: str, name: str, request: Request) -> Response:
    if Path(name).name != name or not session_id.isalnum():
        raise HTTPException(404)
    root = Path(request.app.state.settings.output.hls_dir) / session_id
    path = root / name
    try:
        # 파일을 통째로 읽어 보낸다 (1 s 세그먼트·플레이리스트라 작다). FileResponse 는 크기를 먼저 재고 뒤에 읽어,
        # 쓰는 중인 플레이리스트가 그 사이 자라면 "Response content longer than Content-Length" 로 응답이 깨졌다
        data = path.read_bytes()
    except (FileNotFoundError, IsADirectoryError, PermissionError):
        raise HTTPException(
            404, detail={"code": "E_HLS_NOT_FOUND", "message": f"{name} 없음"}
        ) from None
    media_type = CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")
    # CORS 헤더는 CORSMiddleware 가 붙인다 (server.cors_origins)
    return Response(content=data, media_type=media_type, headers={"Cache-Control": "no-store"})
