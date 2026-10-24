"""STT 엔진 선택 (API 계약 S-26): 세션 요청 `stt.engine`, 언어별 라우팅 `stt.route_by_lang`, `GET /api/engines`."""

import json

import numpy as np
from fastapi.testclient import TestClient

from livetrans.core.config import Settings, SttConfig
from livetrans.main import create_app
from livetrans.plugins.stt.fake import FakeSTT
from tests.integration.test_mic_session import (
    collect_until_stopped,
    energy_vad,
    send_audio,
    silence,
    tone,
)

TWO = np.concatenate([silence(0.5), tone(1.2), silence(1.0), tone(1.2), silence(1.0)])


class Primary(FakeSTT):
    name = "dicow"
    model_id = "fake-dicow"
    supports_speaker_mask = True


class Alt(FakeSTT):
    name = "faster_whisper"
    model_id = "fake-fw"
    supports_speaker_mask = False


def make_app(route=None, primary_lang="ko"):
    settings = Settings(
        stt=SttConfig(engine="fake", route_by_lang=route if route is not None else {})
    )
    primary, alt = Primary(lang=primary_lang), Alt(lang=primary_lang)
    app = create_app(
        settings, stt=primary, vad_factory=lambda: energy_vad, tts_engines={}, stt_rescue=alt
    )
    return app, primary, alt


def run(client, body_extra: dict, src_lang="auto"):
    r = client.post(
        "/api/sessions",
        json={
            "mode": "C",
            "source": {"type": "mic_ws", "params": {}},
            "src_lang": src_lang,
            "tgt_lang": "ko",
            **body_extra,
        },
    )
    if r.status_code != 200:
        return r, []
    with client.websocket_connect(r.json()["ws_url"]) as ws:
        ws.receive_text()
        ws.send_text(json.dumps({"type": "control", "payload": {"action": "start"}}))
        send_audio(ws, TWO)
        ws.send_text(json.dumps({"type": "control", "payload": {"action": "stop"}}))
        events = collect_until_stopped(ws)
    return r, [e["payload"] for e in events if e["type"] == "segment.final"]


def finals_of(engine):
    return [c for c in engine.calls if c["final"]]


def test_engines_endpoint_lists_loaded_stt_and_mt():
    app, _, _ = make_app(route={"ko": "faster_whisper"})
    with TestClient(app) as client:
        body = client.get("/api/engines").json()
    assert body["stt"]["default"] == "auto"
    assert body["stt"]["route_by_lang"] == {"ko": "faster_whisper"}
    assert [(e["name"], e["role"], e["speaker_mask"]) for e in body["stt"]["engines"]] == [
        ("dicow", "primary", True),
        ("faster_whisper", "rescue", False),
    ]
    assert body["mt"]["engine"] == "passthrough" and body["mt"]["default_model"] is None
    assert body["mt"]["models"] == []


def test_auto_routes_detected_korean_to_alt_engine_from_first_utterance():
    app, primary, alt = make_app(route={"ko": "faster_whisper"}, primary_lang="ko")
    with TestClient(app) as client:
        _, finals = run(client, {})
    assert len(finals) >= 2
    # 언어 투표가 모이기 전: 기본 엔진이 감지(ko, 확신) → 라우팅된 엔진으로 같은 발화를 다시 인식.
    # 투표가 모인 뒤: 바로 라우팅된 엔진으로. 어느 쪽이든 확정 결과는 전부 라우팅된 엔진이 낸다
    assert len(finals_of(alt)) == len(finals), "ko 발화가 faster_whisper 로 가지 않았다"
    assert len(finals_of(primary)) <= len(finals)  # 감지용 호출만
    # 마스크 미지원 엔진에는 speaker 를 넘기지 않는다
    assert all(c["speaker"] is None for c in finals_of(alt))


def test_explicit_src_lang_routes_without_detection():
    app, primary, alt = make_app(route={"ja": "faster_whisper"})
    with TestClient(app) as client:
        _, finals = run(client, {}, src_lang="ja")
    assert finals and finals_of(alt) and not finals_of(primary)


def test_english_stays_on_primary_when_not_routed():
    app, primary, alt = make_app(route={"ko": "faster_whisper"}, primary_lang="en")
    with TestClient(app) as client:
        _, finals = run(client, {})
    assert finals and finals_of(primary) and not finals_of(alt)


def test_session_stt_engine_overrides_routing():
    app, primary, alt = make_app(route={"ko": "faster_whisper"}, primary_lang="ko")
    with TestClient(app) as client:
        _, finals = run(client, {"stt": {"engine": "dicow"}})
    assert finals and finals_of(primary) and not finals_of(alt)


def test_unknown_stt_engine_is_rejected():
    app, _, _ = make_app()
    with TestClient(app) as client:
        r, _ = run(client, {"stt": {"engine": "whisperx"}})
    assert r.status_code == 400 and r.json()["detail"]["code"] == "E_STT_ENGINE"
