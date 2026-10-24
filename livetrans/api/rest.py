"""REST API (명세 §6.1)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from livetrans.core.gpu import gpu_info, gpu_util, process_memory_mb

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


@router.get("/engines")
async def engines(request: Request) -> dict:
    """세션에서 고를 수 있는 엔진·모델 (API 계약 S-26): STT 는 떠 있는 엔진만, 번역 모델은 Ollama 에 설치된 것.

    앱은 세션 요청에 `stt: {engine}`(auto | 이름) 과 `mt: {model}` 로 선택을 넘긴다.
    """
    st = request.app.state
    stt_list: list[dict] = []
    for role, e in (("primary", st.stt), ("rescue", getattr(st, "stt_rescue", None))):
        if e is None or any(x["name"] == e.name for x in stt_list):
            continue
        stt_list.append(
            {
                "name": e.name,
                "model": e.model_id,
                "role": role,
                "languages": list(getattr(e, "languages", ())),
                "speaker_mask": bool(getattr(e, "supports_speaker_mask", False)),
            }
        )
    mt = getattr(st, "mt", None)
    models: list[dict] = []
    if mt is not None and hasattr(mt, "list_models"):
        try:
            models = await mt.list_models()
        except Exception:  # noqa: BLE001 — 목록 실패는 빈 목록 (선택 UI 만 비게)
            models = []
    return {
        "stt": {
            "default": "auto",
            "route_by_lang": dict(st.settings.stt.route_by_lang),
            "engines": stt_list,
        },
        "mt": {
            "engine": mt.name if mt is not None else "none",
            "default_model": mt.model_id if mt is not None and mt.name != "passthrough" else None,
            "models": models,
        },
    }


@router.get("/health")
async def health(request: Request) -> dict:
    loaded = list(getattr(request.app.state, "models_loaded", []))
    # process: 서버 프로세스 RAM (rss_mb 상주, private_mb 커밋). 메모리 추적용 (2026-10-25)
    return {
        "status": "ok",
        "gpu": gpu_info(),
        "process": process_memory_mb(),
        "models_loaded": loaded,
    }
