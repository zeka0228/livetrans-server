"""REST API (명세 §6.1)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from livetrans.core.gpu import gpu_info, gpu_util

router = APIRouter(prefix="/api")


@router.post("/calibrate")
async def calibrate(request: Request) -> dict:
    """권장 D (명세 §3.4). 아직 측정 기반 보정은 없고 설정의 기본 D 를 돌려준다 (API 계약 Q-API-11)."""
    tl = request.app.state.settings.timeline
    return {
        "recommended_delay_s": tl.default_delay_s,
        "p99_end_to_end_ms": None,
        "gpu_util": gpu_util(),
        "note": "고정값 — 실측 보정(샘플 실시간 처리)은 미구현",
    }


@router.get("/audio/devices")
async def audio_devices() -> dict:
    """출력 장치 목록 (명세 §4.9, M5c). wincap 에 장치 열거가 아직 없다."""
    raise HTTPException(
        status_code=501,
        detail={
            "code": "E_NOT_IMPLEMENTED",
            "message": "출력 장치 목록은 아직 지원하지 않습니다 (M5c)",
        },
    )


@router.get("/health")
async def health(request: Request) -> dict:
    loaded = list(getattr(request.app.state, "models_loaded", []))
    return {"status": "ok", "gpu": gpu_info(), "models_loaded": loaded}
