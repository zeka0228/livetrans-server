"""환각 복구 체인 (재시도 → 구조 엔진 → 살리기) — 가짜 STT·가짜 Diarizer, GPU 없음."""

import json

import numpy as np
from fastapi.testclient import TestClient

from livetrans.core.config import Settings, SttConfig
from livetrans.core.speaker import SpeakerTurn
from livetrans.main import create_app
from livetrans.plugins.diar.fake import FakeDiarizerEngine
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
LOOP = "oh yeah i am i am i am i am i am i am i am i am i am i am"


class LoopingSTT(FakeSTT):
    """확정은 항상 반복 환각. anti_repeat 재시도 결과는 생성자로 정한다."""

    def __init__(self, retry_text: str | None = None) -> None:
        super().__init__()
        self.retry_text = retry_text

    async def transcribe(self, seg, *, lang, prompt, final, speaker=None, anti_repeat=False):
        tr = await super().transcribe(
            seg, lang=lang, prompt=prompt, final=final, speaker=speaker, anti_repeat=anti_repeat
        )
        if final:
            tr.text = self.retry_text if (anti_repeat and self.retry_text) else LOOP
        return tr


def two_speakers(seg) -> list[SpeakerTurn]:
    mid = seg.t_start_ms + (seg.t_end_ms - seg.t_start_ms) // 2
    return [SpeakerTurn("S0", seg.t_start_ms, mid), SpeakerTurn("S1", mid, seg.t_end_ms)]


def run(stt, rescue=None, diar=None):
    app = create_app(
        Settings(stt=SttConfig(engine="fake")),
        stt=stt,
        vad_factory=lambda: energy_vad,
        tts_engines={},
        diar=diar,
        stt_rescue=rescue,
    )
    with TestClient(app) as client:
        health = client.get("/api/health").json()["models_loaded"]
        body = create(client, src_lang="ko")
        with client.websocket_connect(body["ws_url"]) as ws:
            ws.receive_text()
            ws.send_text(json.dumps({"type": "control", "payload": {"action": "start"}}))
            send_audio(ws, ONE_UTTERANCE)
            ws.send_text(json.dumps({"type": "control", "payload": {"action": "stop"}}))
            events = collect_until_stopped(ws)
    return health, [e["payload"] for e in events if e["type"] == "segment.final"]


def test_retry_with_anti_repeat_recovers_text():
    stt = LoopingSTT(retry_text="oh yeah i am fine")
    _, finals = run(stt)
    assert [f["text"] for f in finals] == ["oh yeah i am fine"]
    final_calls = [c for c in stt.calls if c["final"]]
    assert [c["anti_repeat"] for c in final_calls] == [False, True]
    assert final_calls[1]["prompt"] is None  # 재시도는 프롬프트 없이


def test_rescue_engine_transcribes_only_the_speaker_span():
    stt = LoopingSTT()  # 재시도도 환각
    rescue = FakeSTT(lang="ko")
    health, finals = run(stt, rescue=rescue, diar=FakeDiarizerEngine(two_speakers))
    assert "stt_rescue:fake:fake" in health
    assert [f["speaker"] for f in finals] == ["S0", "S1"]
    assert all(f["text"].startswith("발화 0 확정") for f in finals)  # 구조 엔진 결과
    # 구조 엔진에는 화자 범위만 잘라서 넘긴다 (S0 = 앞 절반, S1 = 뒤 절반)
    assert [c["speaker"] for c in rescue.calls] == [None, None]
    c0, c1 = rescue.calls
    assert c0["t_end_ms"] == c1["t_start_ms"]
    assert abs(c0["n_samples"] - c1["n_samples"]) <= 16 * 2
    assert c0["n_samples"] == (c0["t_end_ms"] - c0["t_start_ms"]) * 16
    assert finals[0]["t_start_ms"] == c0["t_start_ms"] and finals[1]["t_end_ms"] == c1["t_end_ms"]
    # stt_ms 에 재시도·구조 엔진 시간이 더해진다 (가짜는 0 이라 값만 존재)
    assert all(isinstance(f["stt_ms"], int) for f in finals)


def test_rescue_failure_falls_back_to_salvaged_prefix():
    class BrokenRescue(FakeSTT):
        async def transcribe(self, seg, *, lang, prompt, final, speaker=None, anti_repeat=False):
            raise RuntimeError("구조 엔진 죽음")

    _, finals = run(LoopingSTT(), rescue=BrokenRescue())
    assert [f["text"] for f in finals] == ["oh yeah i am"]  # 반복 직전까지 + 단위 1개


def test_without_rescue_salvages_or_blanks():
    _, finals = run(LoopingSTT())
    assert [f["text"] for f in finals] == ["oh yeah i am"]

    class PureLoop(LoopingSTT):
        async def transcribe(self, seg, *, lang, prompt, final, speaker=None, anti_repeat=False):
            tr = await FakeSTT.transcribe(
                self, seg, lang=lang, prompt=prompt, final=final, speaker=speaker
            )
            if final:
                tr.text = "ok" * 100
            return tr

    _, finals = run(PureLoop())
    assert [f["text"] for f in finals] == [""]  # 살릴 것이 없으면 빈 final (앱이 partial 을 지우게)
