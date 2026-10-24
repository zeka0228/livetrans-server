"""출력 격리 (명세 §4.9, API 계약 Q-API-11·S-29): /api/audio/devices, /api/audio/route/check, 모드 A loopback 의 격리 장치 캡처.
wincap 없이 가짜 백엔드로."""

import pytest
from fastapi.testclient import TestClient

from livetrans.core.config import RouteConfig, Settings, SttConfig
from livetrans.main import create_app
from livetrans.plugins.source.loopback import parse_target
from livetrans.plugins.stt.fake import FakeSTT
from tests.integration.test_loopback_session import FakeBackend
from tests.integration.test_mic_session import energy_vad

SPK = "{0.0.0.00000000}.{AAAAAAAA-0000-0000-0000-000000000001}"  # 사용자 스피커 (기본)
HDMI = "{0.0.0.00000000}.{BBBBBBBB-0000-0000-0000-000000000002}"  # 격리 후보 (모니터)
OFF = "{0.0.0.00000000}.{CCCCCCCC-0000-0000-0000-000000000003}"  # 연결 안 됨


class RouteBackend(FakeBackend):
    def __init__(self, levels=None, **kw):
        super().__init__(**kw)
        self.levels = levels or {}
        self.meter_calls = []

    async def list_devices(self):
        return [
            {"id": SPK, "name": "스피커(USB)", "state": "active", "is_default": True},
            {"id": HDMI, "name": "LG HDR 4K(NVIDIA)", "state": "active", "is_default": False},
            {"id": OFF, "name": "BS NO.7(NVIDIA)", "state": "unplugged", "is_default": False},
        ]

    async def meter(self, ids, ms):
        self.meter_calls.append((list(ids), ms))
        return {
            i: {"peak": self.levels.get(i, 0.0), "mean": self.levels.get(i, 0.0) / 2} for i in ids
        }


def make_client(backend, router="spare_device"):
    settings = Settings(stt=SttConfig(engine="fake"), route=RouteConfig(router=router, check_ms=10))
    app = create_app(
        settings, stt=FakeSTT(), vad_factory=lambda: energy_vad, tts_engines={}, capture=backend
    )
    return TestClient(app)


def test_parse_target_accepts_device_ids():
    assert parse_target(f"device:{HDMI}") == ["device", HDMI]
    with pytest.raises(ValueError):
        parse_target("device:not-a-guid")


def test_devices_endpoint_shape_and_router():
    with make_client(RouteBackend()) as c:
        body = c.get("/api/audio/devices").json()
    assert body["router"] == "spare_device"
    assert [(d["id"], d["is_default"], d["active"], d["state"]) for d in body["devices"]] == [
        (f"device:{SPK}", True, True, "active"),
        (f"device:{HDMI}", False, True, "active"),
        (f"device:{OFF}", False, False, "unplugged"),
    ]
    assert body["devices"][1]["name"] == "LG HDR 4K(NVIDIA)"


def test_route_check_passes_when_isolation_has_signal_and_speaker_is_silent():
    backend = RouteBackend(levels={HDMI: 0.4, SPK: 0.0})
    with make_client(backend) as c:
        r = c.post(
            "/api/audio/route/check",
            json={"router": "spare_device", "isolation_device": f"device:{HDMI}"},
        ).json()
    assert r["isolated"] is True and r["problems"] == []
    assert r["capture_target"] == f"device:{HDMI}" and r["user_output_device"] == f"device:{SPK}"
    assert "LG HDR 4K(NVIDIA)" in r["setup_guide"] and r["restore_guide"]
    assert backend.meter_calls == [([HDMI, SPK], 10)]


def test_route_check_fails_on_no_signal_and_on_leak():
    with make_client(RouteBackend(levels={HDMI: 0.0, SPK: 0.0})) as c:
        r = c.post("/api/audio/route/check", json={"isolation_device": f"device:{HDMI}"}).json()
    assert r["isolated"] is False and any("소리가 감지되지 않습니다" in p for p in r["problems"])
    with make_client(RouteBackend(levels={HDMI: 0.3, SPK: 0.3})) as c:
        r = c.post("/api/audio/route/check", json={"isolation_device": f"device:{HDMI}"}).json()
    assert r["isolated"] is False and any("사용자 출력 장치" in p for p in r["problems"])


def test_route_check_rejects_default_device_and_inactive_device_without_metering():
    backend = RouteBackend(levels={SPK: 0.5})
    with make_client(backend) as c:
        same = c.post("/api/audio/route/check", json={"isolation_device": f"device:{SPK}"}).json()
        off = c.post("/api/audio/route/check", json={"isolation_device": f"device:{OFF}"}).json()
        missing = c.post(
            "/api/audio/route/check", json={"isolation_device": "device:{nope}"}
        ).json()
    assert not same["isolated"] and any("기본 출력 장치" in p for p in same["problems"])
    assert not off["isolated"] and any("꺼져 있거나" in p for p in off["problems"])
    assert not missing["isolated"] and any("찾을 수 없습니다" in p for p in missing["problems"])
    assert backend.meter_calls == []  # 장치 문제면 레벨을 재지 않는다


def test_route_check_with_router_none_skips_checks():
    with make_client(RouteBackend(), router="none") as c:
        r = c.post(
            "/api/audio/route/check", json={"router": "none", "isolation_device": f"device:{HDMI}"}
        ).json()
    assert r["isolated"] is True and r["problems"] == []


def _create_mode_a(c, params, route=None):
    body = {
        "mode": "A",
        "source": {"type": "loopback", "params": params},
        "src_lang": "ko",
        "tgt_lang": "ko",
        "delay_s": 5,
    }
    if route is not None:
        body["route"] = route
    return c.post("/api/sessions", json=body)


def test_mode_a_loopback_rejects_system_capture_when_isolating():
    backend = RouteBackend()
    with make_client(backend) as c:
        r = _create_mode_a(c, {})
        assert r.status_code == 400 and r.json()["detail"]["code"] == "E_ROUTE_NOT_ISOLATED"
        r = _create_mode_a(
            c,
            {"target": "system"},
            route={"router": "spare_device", "isolation_device": f"device:{HDMI}"},
        )
        assert r.status_code == 400  # 명시적 system 도 거부
    assert backend.sources == []


def test_mode_a_loopback_uses_isolation_device_from_route_or_target():
    backend = RouteBackend()
    with make_client(backend) as c:
        r = _create_mode_a(
            c, {}, route={"router": "spare_device", "isolation_device": f"device:{HDMI}"}
        )
        assert r.status_code == 200, r.text
        r2 = _create_mode_a(c, {"target": f"device:{HDMI}"})
        assert r2.status_code == 200, r2.text
    assert [s.target for s in backend.sources] == [f"device:{HDMI}", f"device:{HDMI}"]


def test_mode_a_loopback_system_allowed_when_router_none():
    backend = RouteBackend()
    with make_client(backend, router="none") as c:
        assert _create_mode_a(c, {}).status_code == 200
    assert [s.target for s in backend.sources] == ["system"]
