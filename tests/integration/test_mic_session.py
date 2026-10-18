import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from livetrans.core.config import Settings, SttConfig
from livetrans.main import create_app
from livetrans.plugins.stt.fake import FakeSTT

SR = 16000


def energy_vad(window: np.ndarray) -> float:
    return 1.0 if float(np.sqrt(np.mean(window**2))) > 0.05 else 0.0


def pcm16(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()


def tone(seconds: float) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


@pytest.fixture
def client():
    stt = FakeSTT()
    app = create_app(
        Settings(stt=SttConfig(engine="fake")), stt=stt, vad_factory=lambda: energy_vad
    )
    with TestClient(app) as c:
        c.stt = stt
        yield c


def create(client, src_lang="auto"):
    r = client.post(
        "/api/sessions",
        json={
            "mode": "C",
            "source": {"type": "mic_ws", "params": {}},
            "src_lang": src_lang,
            "tgt_lang": "ko",
        },
    )
    assert r.status_code == 200, r.text
    return r.json()


def send_audio(ws, audio: np.ndarray, frame_bytes: int = 640):
    data = pcm16(audio)
    for i in range(0, len(data), frame_bytes):
        ws.send_bytes(data[i : i + frame_bytes])


def collect_until_stopped(ws) -> list[dict]:
    events = []
    while True:
        ev = json.loads(ws.receive_text())
        events.append(ev)
        if ev["type"] == "session.state" and ev["payload"]["state"] == "stopped":
            return events


def test_create_session_response_shape(client):
    body = create(client)
    assert set(body) == {"session_id", "delay_s", "hls_url", "ws_url"}
    assert body["ws_url"] == f"/ws/sessions/{body['session_id']}"
    assert body["delay_s"] is None and body["hls_url"] is None


def test_unsupported_mode_returns_501(client):
    r = client.post("/api/sessions", json={"mode": "A", "source": {"type": "obs_stream"}})
    assert r.status_code == 501
    assert r.json()["detail"]["code"] == "E_NOT_IMPLEMENTED"


def test_mic_flow_emits_finals_and_flushes_on_stop(client):
    body = create(client, src_lang="ja")
    with client.websocket_connect(body["ws_url"]) as ws:
        first = json.loads(ws.receive_text())
        assert first["type"] == "session.state" and first["payload"]["state"] == "starting"
        ws.send_text(json.dumps({"type": "control", "payload": {"action": "start"}}))
        # 발화 1개는 무음으로 닫히고, 마지막 발화는 stop 의 flush 로 닫힌다
        send_audio(ws, np.concatenate([silence(0.5), tone(1.5), silence(1.0), tone(0.8)]))
        ws.send_text(json.dumps({"type": "control", "payload": {"action": "stop"}}))
        events = collect_until_stopped(ws)

    states = [e["payload"]["state"] for e in events if e["type"] == "session.state"]
    assert states == ["playing", "stopped"]
    finals = [e["payload"] for e in events if e["type"] == "segment.final"]
    assert [f["seg_id"] for f in finals] == [0, 1]
    assert abs(finals[0]["t_start_ms"] - 300) <= 32
    assert finals[1]["t_end_ms"] <= 3800
    assert all(f["lang"] == "ja" for f in finals)
    assert {
        "seg_id",
        "t_start_ms",
        "t_end_ms",
        "text",
        "lang",
        "lang_prob",
        "words",
        "forced_cut",
        "stt_ms",
    } == set(finals[0])
    seqs = [e["seq"] for e in events]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
    # 확정 이후 같은 seg_id 의 중간 결과는 오지 않는다
    for f in finals:
        idx = next(
            i
            for i, e in enumerate(events)
            if e["type"] == "segment.final" and e["payload"]["seg_id"] == f["seg_id"]
        )
        assert not any(
            e["type"] == "segment.partial" and e["payload"]["seg_id"] == f["seg_id"]
            for e in events[idx:]
        )
    assert client.stt.calls[0]["lang"] == "ja"
    assert body["session_id"] not in client.app.state.sessions


def test_odd_byte_frames_keep_time_axis(client):
    body = create(client)
    with client.websocket_connect(body["ws_url"]) as ws:
        ws.receive_text()
        send_audio(ws, np.concatenate([silence(1.0), tone(1.0), silence(1.0)]), frame_bytes=333)
        ws.send_text(json.dumps({"type": "control", "payload": {"action": "stop"}}))
        events = collect_until_stopped(ws)
    finals = [e["payload"] for e in events if e["type"] == "segment.final"]
    assert len(finals) == 1
    assert abs(finals[0]["t_start_ms"] - 800) <= 32
    assert abs(finals[0]["t_end_ms"] - 2200) <= 32


def test_rest_stop_ends_session(client):
    body = create(client)
    with client.websocket_connect(body["ws_url"]) as ws:
        ws.receive_text()
        r = client.post(f"/api/sessions/{body['session_id']}/stop")
        assert r.json() == {"session_id": body["session_id"], "stopped": True}
        events = collect_until_stopped(ws)
    assert events[-1]["payload"]["state"] == "stopped"
    assert client.post(f"/api/sessions/{body['session_id']}/stop").status_code == 404


def test_unknown_session_ws_closes_4404(client):
    with (
        client.websocket_connect("/ws/sessions/nope") as ws,
        pytest.raises(WebSocketDisconnect) as exc,
    ):
        ws.receive_text()
    assert exc.value.code == 4404


def test_null_src_lang_means_auto_and_audio_without_start_begins(client):
    r = client.post(
        "/api/sessions",
        json={"mode": "C", "source": {"type": "mic_ws"}, "src_lang": None},
    )
    assert r.status_code == 200
    with client.websocket_connect(r.json()["ws_url"]) as ws:
        ws.receive_text()
        send_audio(ws, np.concatenate([tone(1.0), silence(1.0)]))
        ws.send_text(json.dumps({"type": "control", "payload": {"action": "stop"}}))
        events = collect_until_stopped(ws)
    states = [e["payload"]["state"] for e in events if e["type"] == "session.state"]
    assert states == ["playing", "stopped"]
    assert client.stt.calls[-1]["lang"] is None
