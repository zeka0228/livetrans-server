"""세션 stream 모드 (D-028): 가짜 스트리밍 Diarizer + 가짜 STT — 화자 레인이 발화를 자르고 speaker 가 붙은 partial/final 을 낸다."""

import json

from fastapi.testclient import TestClient

from livetrans.core.config import DiarConfig, Settings, SttConfig
from livetrans.main import create_app
from livetrans.plugins.diar.fake import FakeDiarizerEngine, one_speaker
from livetrans.plugins.stt.fake import FakeSTT
from tests.integration.test_mic_session import (
    collect_until_stopped,
    create,
    energy_vad,
    send_audio,
    tone,
)

# 활동 표 (ms): S0 500~2500, S1 2000~4000 (500 ms 겹침), S0 4800~5500
ACTIVITY = [("S0", 500, 2500), ("S1", 2000, 4000), ("S0", 4800, 5500)]


def activity(from_ms: int, to_ms: int):
    return [(s, a, b) for s, a, b in ACTIVITY if b > from_ms and a < to_ms]


def run(mode="stream", engine=None, seconds=6.0):
    stt = FakeSTT()
    engine = engine or FakeDiarizerEngine(activity_fn=activity, hop_s=1.0, right_context_s=0.5)
    app = create_app(
        Settings(stt=SttConfig(engine="fake"), diar=DiarConfig(engine="fake", mode=mode)),
        stt=stt,
        vad_factory=lambda: energy_vad,
        tts_engines={},
        diar=engine,
    )
    with TestClient(app) as client:
        body = create(client, src_lang="ko")
        with client.websocket_connect(body["ws_url"]) as ws:
            ws.receive_text()
            ws.send_text(json.dumps({"type": "control", "payload": {"action": "start"}}))
            send_audio(ws, tone(seconds))
            ws.send_text(json.dumps({"type": "control", "payload": {"action": "stop"}}))
            events = collect_until_stopped(ws)
    return stt, engine, events


def test_lanes_cut_utterances_per_speaker_with_speaker_field():
    stt, engine, events = run()
    finals = [e["payload"] for e in events if e["type"] == "segment.final"]
    # 레인별 발화: S0 (500~2500), S1 (2000~4000), S0 (4800~5500) — 시간 범위는 그 화자가 말한 범위
    assert [(f["seg_id"], f["speaker"], f["t_start_ms"], f["t_end_ms"]) for f in finals] == [
        (0, "S0", 500, 2500),
        (1, "S1", 2000, 4000),
        (2, "S0", 4800, 5500),
    ]
    assert all(f["text"].startswith(f"발화 {f['seg_id']} 확정") for f in finals)
    assert all(f["diar_ms"] == 7 for f in finals)
    # 확정 STT 는 레인 화자 1명만 대상으로 (마스크), 발화 오디오는 패딩 포함 구간
    final_calls = [c for c in stt.calls if c["final"]]
    assert [c["speaker"] for c in final_calls] == ["S0", "S1", "S0"]
    assert (final_calls[0]["t_start_ms"], final_calls[0]["t_end_ms"]) == (300, 2700)
    assert final_calls[0]["n_samples"] == (2700 - 300) * 16
    # 겹친 S1 구간은 마스크용으로 함께 넘어간다 → speakers 에 S1 포함 (FakeSTT 는 기록만)
    stream = engine.streams[0]
    assert stream.steps >= 6 and stream.resets == 1
    assert events[-1]["payload"]["state"] == "stopped"


def test_partials_carry_speaker_and_share_seg_id_with_final():
    _, _, events = run()
    partials = [e["payload"] for e in events if e["type"] == "segment.partial"]
    finals = [e["payload"] for e in events if e["type"] == "segment.final"]
    assert partials, "레인이 열려 있는 동안 중간 결과가 나와야 한다"
    assert all("speaker" in p for p in partials)
    final_ids = {f["seg_id"]: f["speaker"] for f in finals}
    for p in partials:
        assert final_ids.get(p["seg_id"]) == p["speaker"]
    # 같은 seg_id 의 final 뒤에는 partial 이 없다
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


def test_utterance_mode_is_used_when_configured_or_unsupported():
    engine = FakeDiarizerEngine(turns_fn=one_speaker, activity_fn=activity)
    _, engine, events = run(mode="utterance", engine=engine)
    assert engine.created and not engine.streams
    finals = [e["payload"] for e in events if e["type"] == "segment.final"]
    assert finals and all(f["speaker"] == "S0" for f in finals)
    # 스트림을 지원하지 않는 엔진은 stream 설정이어도 발화 단위로
    plain = FakeDiarizerEngine(turns_fn=one_speaker)
    _, plain, events = run(mode="stream", engine=plain)
    assert plain.created and not plain.streams


def test_stream_step_failure_does_not_kill_session():
    class Flaky(FakeDiarizerEngine):
        def create_stream(self):
            s = super().create_stream()
            orig = s.step

            async def step(*, final=False):
                if s.steps == 1:
                    s.steps += 1
                    raise RuntimeError("GPU 잠깐 죽음")
                return await orig(final=final)

            s.step = step
            return s

    _, _, events = run(engine=Flaky(activity_fn=activity, hop_s=1.0, right_context_s=0.5))
    assert events[-1]["payload"]["state"] == "stopped"
    assert [e["payload"]["speaker"] for e in events if e["type"] == "segment.final"] == [
        "S0",
        "S1",
        "S0",
    ]
