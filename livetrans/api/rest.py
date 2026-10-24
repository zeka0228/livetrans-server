"""REST API (명세 §6.1)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

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


class RouteCheckBody(BaseModel):
    router: str = "spare_device"
    isolation_device: str  # "device:{0.0.0.00000000}.{…}" (GET /api/audio/devices 의 id)


def _device_id(ref: str) -> str:
    return ref.removeprefix("device:")


async def _devices(request: Request) -> list[dict]:
    """wincap 의 출력 장치 목록. 헬퍼 없음 → 503, 옛 빌드 → 501."""
    from livetrans.core.wincap import WincapTooOld

    backend = request.app.state.capture
    if not backend.available():
        raise HTTPException(
            503,
            detail={
                "code": "E_CAPTURE_UNAVAILABLE",
                "message": "캡처 헬퍼가 없습니다 (scripts\\build_wincap.bat)",
            },
        )
    try:
        return await backend.list_devices()
    except WincapTooOld as e:
        raise HTTPException(
            501,
            detail={
                "code": "E_NOT_IMPLEMENTED",
                "message": "wincap 이 옛 빌드입니다 — scripts\\build_wincap.bat 로 다시 빌드하세요",
            },
        ) from e


@router.get("/audio/devices")
async def audio_devices(request: Request) -> dict:
    """출력 장치 목록 (명세 §4.9, API 계약 Q-API-11): `{devices:[{id, name, state, is_default, active}], router}`.
    `id` 는 세션 요청의 loopback `target` 과 `route.isolation_device` 에 그대로 쓴다."""
    devices = await _devices(request)
    return {
        "devices": [
            {
                "id": f"device:{d['id']}",
                "name": d["name"],
                "state": d["state"],
                "is_default": bool(d["is_default"]),
                "active": d["state"] == "active",
            }
            for d in devices
        ],
        "router": request.app.state.settings.route.router,
    }


@router.post("/audio/route/check")
async def audio_route_check(body: RouteCheckBody, request: Request) -> dict:
    """소리 격리 점검 (명세 §4.9 `check()`): 격리 장치가 살아 있고, 사용자 기본 출력 장치와 다르고, 원본이 실제로 격리 장치로
    나가며(피크 레벨), 사용자 장치로는 새지 않는지. 통과하면 모드 A 를 시작해도 사용자는 지연 출력만 듣는다."""
    cfg = request.app.state.settings.route
    devices = await _devices(request)
    by_id = {d["id"]: d for d in devices}
    default = next((d for d in devices if d["is_default"]), None)
    iso_id = _device_id(body.isolation_device)
    iso = by_id.get(iso_id)
    problems: list[str] = []
    router = body.router or cfg.router
    if router == "none":
        return {
            "isolated": True,
            "capture_target": f"device:{iso_id}" if iso else "system",
            "user_output_device": f"device:{default['id']}" if default else None,
            "problems": [],
            "setup_guide": None,
            "restore_guide": None,
        }
    if iso is None:
        problems.append("격리 장치를 찾을 수 없습니다 — 목록을 새로 고쳐 다시 고르세요")
    elif iso["state"] != "active":
        problems.append(
            f"격리 장치 '{iso['name']}' 가 꺼져 있거나 연결되지 않았습니다 ({iso['state']})"
        )
    if default is not None and iso is not None and iso_id == default["id"]:
        problems.append(
            f"격리 장치가 사용자 기본 출력 장치 '{default['name']}' 와 같습니다 — 우리 앱의 지연 소리가 다시 캡처됩니다(되먹임)"
        )
    if not problems and iso is not None:
        ids = [iso_id] + ([default["id"]] if default and default["id"] != iso_id else [])
        levels = await request.app.state.capture.meter(ids, cfg.check_ms)
        iso_peak = float(levels.get(iso_id, {}).get("peak", -1))
        if iso_peak < cfg.signal_threshold:
            problems.append(
                f"격리 장치 '{iso['name']}' 에서 소리가 감지되지 않습니다 (피크 {max(iso_peak, 0):.3f}) — 원본 방송이 재생 중인지, "
                "브라우저(또는 OBS 오디오 소스)의 출력 장치가 이 장치로 돼 있는지 확인하세요"
            )
        if default is not None and default["id"] != iso_id:
            def_peak = float(levels.get(default["id"], {}).get("peak", -1))
            if def_peak >= cfg.leak_threshold:
                problems.append(
                    f"사용자 출력 장치 '{default['name']}' 에서 소리가 납니다 (피크 {def_peak:.3f}) — 원본 방송이 아직 스피커로 "
                    "나오고 있거나 다른 앱이 소리를 내는 중입니다. 점검하는 동안은 원본 방송만 격리 장치로 재생하세요"
                )
    iso_name = iso["name"] if iso else "격리 장치"
    return {
        "isolated": not problems,
        "capture_target": f"device:{iso_id}" if iso else None,
        "user_output_device": f"device:{default['id']}" if default else None,
        "problems": problems,
        "setup_guide": (
            f"Windows 설정 → 시스템 → 소리 → 볼륨 믹서에서 원본 방송을 재생하는 앱(브라우저)의 출력 장치를 '{iso_name}' 으로 바꾸세요. "
            f"OBS 로 영상까지 보내면 OBS 의 '오디오 출력 캡처' 소스 장치도 '{iso_name}' 으로. 우리 앱(이 창)은 기본 장치 그대로 둡니다. "
            "격리 장치가 모니터(HDMI)면 모니터 스피커 볼륨을 0 으로."
        ),
        "restore_guide": "시청이 끝나면 같은 볼륨 믹서에서 그 앱의 출력 장치를 '기본값' 으로 되돌리세요.",
    }


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
