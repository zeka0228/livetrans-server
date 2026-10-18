"""REST API (명세 §6.1)."""

from __future__ import annotations

from fastapi import APIRouter, Request

from livetrans.core.gpu import gpu_info

router = APIRouter(prefix="/api")


@router.get("/health")
async def health(request: Request) -> dict:
    loaded = list(getattr(request.app.state, "models_loaded", []))
    return {"status": "ok", "gpu": gpu_info(), "models_loaded": loaded}
