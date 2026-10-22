import asyncio
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from livetrans.core.clock import SampleClock
from livetrans.core.config import Settings, SttConfig
from livetrans.main import create_app
from livetrans.plugins.source.base import AudioFrame
from livetrans.plugins.source.loopback import CaptureError
from livetrans.plugins.stt.fake import FakeSTT

from .test_mic_session import collect_until_stopped, energy_vad, silence, tone


class FakeLoopbackSource:
    name = "loopback"
    has_video = False

    def __init__(self, audio: np.ndarray, *, lose: bool = False, fail_start: bool = False):
        self.audio = audio
        self.lose = lose
        self.fail_start = fail_start
        self.lost_reason = None
        self.started = False
        self._queue: asyncio.Queue = asyncio.Queue()

    async def start(self):
        if self.fail_start:
            raise CaptureError("ERROR test")
        self.started = True
        clock = SampleClock(16000)
        for i in range(0, len(self.audio), 1600):
            chunk = self.audio[i : i + 1600]
            self._queue.put_nowait(AudioFrame(chunk, clock.advance(len(chunk))))
        if self.lose:
            self.lost_reason = "ERROR target process exited"
            self._queue.put_nowait(None)

    async def frames(self):
        while True:
            f = await self._queue.get()
            if f is None:
                return
            yield f

    def raw_media_dir(self):
        return None

    async def stop(self):
        self._queue.put_nowait(None)


class FakeBackend:
    def __init__(self, **source_kw):
        self.source_kw = source_kw
        self.sources: list[FakeLoopbackSource] = []

    def available(self):
        return True

    def app_capture_supported(self):
        return True

    def process_alive(self, pid):
        return pid == 4242

    async def list_windows(self):
        return [{"pid": 4242, "process": "chrome.exe", "title": "配信 - YouTube - Chrome"}]

    def create_source(self, target, sample_rate):
        audio = np.concatenate([silence(0.5), tone(1.0), silence(1.0), tone(0.8), silence(1.0)])
        src = FakeLoopbackSource(audio, **self.source_kw)
        src.target = target
        self.sources.append(src)
        return src


def make_client(**source_kw):
    backend = FakeBackend(**source_kw)
    app = create_app(
        Settings(stt=SttConfig(engine="fake")),
        stt=FakeSTT(lang="ja"),
        vad_factory=lambda: energy_vad,
        capture=backend,
        tts_engines={},
    )
    return TestClient(app), backend


def create_b(client, target="pid:4242"):
    return client.post(
        "/api/sessions",
        json={
            "mode": "B",
            "source": {"type": "loopback", "params": {"target": target}},
            "src_lang": "ja",
        },
    )


def control(action):
    return json.dumps({"type": "control", "payload": {"action": action}})


def test_targets_endpoint():
    client, _ = make_client()
    with client:
        body = client.get("/api/capture/targets").json()
    assert body["capture_available"] is True and body["app_capture_supported"] is True
    assert [t["id"] for t in body["targets"]] == ["system", "pid:4242"]


def test_mode_b_flow_captures_after_start_and_stops():
    client, backend = make_client()
    with client:
        r = create_b(client)
        assert r.status_code == 200, r.text
        src = backend.sources[0]
        assert src.target == "pid:4242" and not src.started  # start 전에는 캡처하지 않음
        with client.websocket_connect(r.json()["ws_url"]) as ws:
            assert json.loads(ws.receive_text())["payload"]["state"] == "starting"
            ws.send_text(control("start"))
            first = json.loads(ws.receive_text())
            assert first["payload"]["state"] == "playing" and src.started
            ws.send_text(control("stop"))
            events = [first, *collect_until_stopped(ws)]
    finals = [e["payload"] for e in events if e["type"] == "segment.final"]
    assert [f["seg_id"] for f in finals] == [0, 1]
    assert all(f["lang"] == "ja" for f in finals)


def test_mode_b_ignores_binary_frames():
    client, _ = make_client()
    with client:
        r = create_b(client)
        with client.websocket_connect(r.json()["ws_url"]) as ws:
            ws.receive_text()
            ws.send_bytes(
                b"\x00" * 640
            )  # 모드 B 는 앱이 오디오를 보내지 않는다 — 와도 무시, 시작으로 보지 않음
            ws.send_text(control("start"))
            assert json.loads(ws.receive_text())["payload"]["state"] == "playing"
            ws.send_text(control("stop"))
            collect_until_stopped(ws)


def test_capture_lost_emits_error_and_stops_by_itself():
    client, _ = make_client(lose=True)
    with client:
        r = create_b(client)
        with client.websocket_connect(r.json()["ws_url"]) as ws:
            ws.receive_text()
            ws.send_text(control("start"))
            events = collect_until_stopped(ws)
    errors = [e["payload"] for e in events if e["type"] == "error"]
    assert errors == [
        {"code": "E_CAPTURE_LOST", "message": "ERROR target process exited", "recoverable": False}
    ]
    # 끊기기 전까지 받은 발화는 끝까지 인식하고, 오류는 모든 final 뒤 · stopped 바로 앞에 온다
    assert len([e for e in events if e["type"] == "segment.final"]) == 2
    assert [e["type"] for e in events[-2:]] == ["error", "session.state"]


def test_capture_start_failure():
    client, _ = make_client(fail_start=True)
    with client:
        r = create_b(client)
        with client.websocket_connect(r.json()["ws_url"]) as ws:
            ws.receive_text()
            ws.send_text(control("start"))
            events = collect_until_stopped(ws)
    assert events[0]["type"] == "error" and events[0]["payload"]["code"] == "E_CAPTURE_START"
    assert events[-1]["payload"]["state"] == "stopped"


@pytest.mark.parametrize("target", ["pid:999", "chrome", "pid:abc"])
def test_invalid_target_400(target):
    client, _ = make_client()
    with client:
        r = create_b(client, target)
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "E_CAPTURE_TARGET"


def test_system_target_allowed():
    client, backend = make_client()
    with client:
        r = create_b(client, "system")
    assert r.status_code == 200 and backend.sources[0].target == "system"
