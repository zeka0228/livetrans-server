"""세션 + 화자 분할 (가짜 Diarizer·가짜 STT, GPU 없음).

검증하는 계약 (명세 §6.2 v0.4, API 계약 S-16): 한 구간에 화자가 2명이면 **같은 `seg_id` 로 `segment.final` 이 화자마다 1개씩**,
`speaker` 는 있을 때만 포함, 화자 분할이 실패해도 자막은 계속 나온다.
"""

import json

import numpy as np
from fastapi.testclient import TestClient

from livetrans.core.config import DiarConfig, Settings, SttConfig
from livetrans.core.speaker import SpeakerTurn
from livetrans.main import create_app
from livetrans.plugins.diar.fake import FakeDiarizerEngine, one_speaker
from livetrans.plugins.stt.base import Transcript
from livetrans.plugins.stt.fake import FakeSTT
from tests.integration.test_mic_session import (
    collect_until_stopped,
    create,
    energy_vad,
    send_audio,
    silence,
    tone,
)

ONE_UTTERANCE = np.concatenate([silence(0.5), tone(1.5), silence(1.0)])
TWO_UTTERANCES = np.concatenate([silence(0.5), tone(1.5), silence(1.0), tone(0.8)])


def two_speakers(seg) -> list[SpeakerTurn]:
    """S0 가 앞 2/3, S1 이 뒤 1/3 — 200 ms 겹침."""
    mid = seg.t_start_ms + (seg.t_end_ms - seg.t_start_ms) * 2 // 3
    return [
        SpeakerTurn("S0", seg.t_start_ms, mid),
        SpeakerTurn("S1", mid - 200, seg.t_end_ms, overlap=True),
    ]


class NoMaskSTT(FakeSTT):
    supports_speaker_mask = False


class ConstantSTT(FakeSTT):
    """확정 결과가 항상 같은 문장 — 화자별 반복 제거 검증용."""

    async def transcribe(self, seg, *, lang, prompt, final, speaker=None):
        tr = await super().transcribe(seg, lang=lang, prompt=prompt, final=final, speaker=speaker)
        if final:
            tr.text = "같은 문장"
        return tr


class EmptySTT(FakeSTT):
    async def transcribe(self, seg, *, lang, prompt, final, speaker=None) -> Transcript:
        tr = await super().transcribe(seg, lang=lang, prompt=prompt, final=final, speaker=speaker)
        tr.text, tr.words = "", []
        return tr


class RaisingDiarizer:
    def __init__(self) -> None:
        self.resets = 0

    async def diarize(self, seg):
        raise RuntimeError("모델 죽음")

    def reset(self) -> None:
        self.resets += 1


class RaisingDiarizerEngine(FakeDiarizerEngine):
    def create(self):
        d = RaisingDiarizer()
        self.created.append(d)
        return d


class FailingLoadEngine(FakeDiarizerEngine):
    async def load(self) -> None:
        raise RuntimeError("토큰 없음")


def make_client(diar, stt=None, settings=None):
    stt = stt or FakeSTT()
    app = create_app(
        settings or Settings(stt=SttConfig(engine="fake")),
        stt=stt,
        vad_factory=lambda: energy_vad,
        tts_engines={},
        diar=diar,
    )
    client = TestClient(app)
    client.stt = stt
    return client


def run_mic(client, audio, src_lang="ko") -> list[dict]:
    body = create(client, src_lang=src_lang)
    with client.websocket_connect(body["ws_url"]) as ws:
        ws.receive_text()  # starting
        ws.send_text(json.dumps({"type": "control", "payload": {"action": "start"}}))
        send_audio(ws, audio)
        ws.send_text(json.dumps({"type": "control", "payload": {"action": "stop"}}))
        return collect_until_stopped(ws)


def finals(events) -> list[dict]:
    return [e["payload"] for e in events if e["type"] == "segment.final"]


def test_health_lists_diarizer_between_stt_and_tts():
    with make_client(FakeDiarizerEngine()) as client:
        models = client.get("/api/health").json()["models_loaded"]
    assert models == ["stt:fake:fake", "diar:fake:fake"]


def test_settings_engine_fake_is_created_by_factory():
    settings = Settings(stt=SttConfig(engine="fake"), diar=DiarConfig(engine="fake"))
    with make_client(None, settings=settings) as client:
        assert "diar:fake:fake" in client.get("/api/health").json()["models_loaded"]


def test_diarizer_load_failure_keeps_server_up_without_speakers():
    with make_client(FailingLoadEngine()) as client:
        assert client.get("/api/health").json()["models_loaded"] == ["stt:fake:fake"]
        fs = finals(run_mic(client, ONE_UTTERANCE))
    assert len(fs) == 1 and "speaker" not in fs[0] and "diar_ms" not in fs[0]


def test_two_speakers_emit_one_final_per_speaker_with_same_seg_id():
    engine = FakeDiarizerEngine(two_speakers)
    with make_client(engine) as client:
        events = run_mic(client, ONE_UTTERANCE)
    fs = finals(events)
    assert [f["seg_id"] for f in fs] == [0, 0]
    assert [f["speaker"] for f in fs] == ["S0", "S1"]
    # 화자별 시간 범위: S0 는 구간 앞쪽, S1 은 뒤쪽 (200 ms 겹침)
    assert fs[0]["t_start_ms"] < fs[1]["t_start_ms"] < fs[0]["t_end_ms"] < fs[1]["t_end_ms"]
    assert fs[1]["t_start_ms"] == fs[0]["t_end_ms"] - 200
    assert all(isinstance(f["diar_ms"], int) and f["diar_ms"] >= 0 for f in fs)
    assert all(f["text"] for f in fs)
    # STT 는 화자마다 1회(final), 중간 결과는 화자 없이
    final_calls = [c for c in client.stt.calls if c["final"]]
    assert [c["speaker"] for c in final_calls] == ["S0", "S1"]
    assert all(c["speaker"] is None for c in client.stt.calls if not c["final"])
    # 세션 종료 시 화자 임베딩 폐기 (§13)
    diarizer = engine.created[0]
    assert diarizer.calls == [0] and diarizer.resets == 1
    states = [e["payload"]["state"] for e in events if e["type"] == "session.state"]
    assert states == ["playing", "stopped"]


def test_single_speaker_final_carries_speaker_and_full_range():
    with make_client(FakeDiarizerEngine(one_speaker)) as client:
        fs = finals(run_mic(client, ONE_UTTERANCE))
    assert len(fs) == 1 and fs[0]["speaker"] == "S0"
    assert abs(fs[0]["t_start_ms"] - 300) <= 32
    assert client.stt.calls[-1]["speaker"] == "S0"


def test_no_turns_means_no_speaker_field():
    with make_client(FakeDiarizerEngine(lambda seg: [])) as client:
        fs = finals(run_mic(client, ONE_UTTERANCE))
    assert len(fs) == 1 and "speaker" not in fs[0] and fs[0]["diar_ms"] >= 0
    assert client.stt.calls[-1]["speaker"] is None


def test_diarizer_exception_does_not_block_subtitles():
    with make_client(RaisingDiarizerEngine()) as client:
        events = run_mic(client, ONE_UTTERANCE)
    fs = finals(events)
    assert len(fs) == 1 and "speaker" not in fs[0] and "diar_ms" not in fs[0]
    assert not [e for e in events if e["type"] == "error"]


def test_engine_without_speaker_mask_runs_once_and_shows_main_speaker():
    with make_client(FakeDiarizerEngine(two_speakers), stt=NoMaskSTT()) as client:
        fs = finals(run_mic(client, ONE_UTTERANCE))
    assert len(fs) == 1 and fs[0]["speaker"] == "S0"
    assert client.stt.calls[-1]["speaker"] is None  # 마스크 미지원 엔진에는 화자를 넘기지 않는다
    assert fs[0]["t_end_ms"] < 1800  # 시간 범위는 주 화자(S0) 범위로 좁힌다


def test_all_empty_texts_still_emit_one_final_so_app_can_clear_partial():
    with make_client(FakeDiarizerEngine(two_speakers), stt=EmptySTT()) as client:
        fs = finals(run_mic(client, ONE_UTTERANCE))
    assert len(fs) == 1 and fs[0]["text"] == "" and fs[0]["seg_id"] == 0


def test_repeat_suppression_is_per_speaker():
    def turns(seg):
        return two_speakers(seg) if seg.seg_id == 0 else one_speaker(seg)

    with make_client(FakeDiarizerEngine(turns), stt=ConstantSTT()) as client:
        fs = finals(run_mic(client, TWO_UTTERANCES))
    # 발화 0: S0·S1 모두 처음이라 둘 다 문장 유지. 발화 1: S0 가 같은 문장 반복 → 비움
    assert [(f["seg_id"], f["speaker"], f["text"]) for f in fs] == [
        (0, "S0", "같은 문장"),
        (0, "S1", "같은 문장"),
        (1, "S0", ""),
    ]


def test_max_speakers_per_seg_limits_stt_runs():
    def three(seg):
        third = (seg.t_end_ms - seg.t_start_ms) // 3
        a = seg.t_start_ms
        return [
            SpeakerTurn("S0", a, a + third),
            SpeakerTurn("S1", a + third, a + 2 * third + 100),  # 가장 길게 말함
            SpeakerTurn("S2", a + 2 * third + 100, seg.t_end_ms),
        ]

    settings = Settings(stt=SttConfig(engine="fake"), diar=DiarConfig(max_speakers_per_seg=2))
    with make_client(FakeDiarizerEngine(three), settings=settings) as client:
        fs = finals(run_mic(client, ONE_UTTERANCE))
    assert sorted(f["speaker"] for f in fs) == ["S0", "S1"]  # 가장 짧게 말한 S2 는 제외
    assert fs[0]["speaker"] == "S1"  # 말한 시간 순
