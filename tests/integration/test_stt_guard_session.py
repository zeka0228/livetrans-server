"""세션 단계의 환각 방지와 세션 기록(JSONL·오디오 덤프) — 가짜 STT, GPU 없음."""

import json
import wave
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from livetrans.core.config import LoggingConfig, Settings, SttConfig
from livetrans.main import create_app
from livetrans.plugins.stt.fake import FakeSTT
from tests.integration.test_mic_session import (
    collect_until_stopped,
    create,
    energy_vad,
    send_audio,
    silence,
    tone,
)

TWO_UTTERANCES = np.concatenate([silence(0.5), tone(1.5), silence(1.0), tone(0.8)])


class HallucinatingSTT(FakeSTT):
    """발화 0 은 중간·확정 모두 'hohoho…' 반복, 발화 1 은 정상."""

    async def transcribe(self, seg, *, lang, prompt, final, speaker=None):
        tr = await super().transcribe(seg, lang=lang, prompt=prompt, final=final, speaker=speaker)
        if seg.seg_id == 0:
            tr.text = "ho" * 200
        return tr


def run(stt, settings) -> list[dict]:
    app = create_app(settings, stt=stt, vad_factory=lambda: energy_vad, tts_engines={})
    with TestClient(app) as client:
        body = create(client, src_lang="ko")
        with client.websocket_connect(body["ws_url"]) as ws:
            first = json.loads(ws.receive_text())  # starting
            ws.send_text(json.dumps({"type": "control", "payload": {"action": "start"}}))
            send_audio(ws, TWO_UTTERANCES)
            ws.send_text(json.dumps({"type": "control", "payload": {"action": "stop"}}))
            events = [first] + collect_until_stopped(ws)
    return events


def test_hallucinated_text_is_blanked_and_not_used_as_prompt():
    stt = HallucinatingSTT()
    events = run(stt, Settings(stt=SttConfig(engine="fake")))
    finals = [e["payload"] for e in events if e["type"] == "segment.final"]
    partials = [e["payload"] for e in events if e["type"] == "segment.partial"]
    assert [f["seg_id"] for f in finals] == [0, 1]
    assert finals[0]["text"] == "" and finals[1]["text"].startswith("발화 1 확정")
    assert finals[0]["words"] == []
    assert not any(p["seg_id"] == 0 for p in partials)  # 환각 중간 결과는 보내지 않는다
    seg1_final = next(c for c in stt.calls if c["final"] and c["seg_id"] == 1)
    assert seg1_final["prompt"] is None  # 환각 문장이 다음 발화의 프롬프트로 새지 않는다


def test_session_jsonl_and_audio_dump(tmp_path: Path):
    settings = Settings(
        stt=SttConfig(engine="fake"),
        logging=LoggingConfig(dir=str(tmp_path), per_session_jsonl=True, dump_audio=True),
    )
    events = run(HallucinatingSTT(), settings)
    session_id = events[0]["session_id"]
    logs = list(tmp_path.glob(f"*_{session_id}.jsonl"))
    assert len(logs) == 1
    records = [json.loads(line) for line in logs[0].read_text(encoding="utf-8").splitlines()]
    types = [r["type"] for r in records]
    # WS 로 보낸 이벤트가 전부 들어 있고(내용 동일), 디버그 기록이 덧붙는다
    sent = [r for r in records if not r["type"].startswith("debug.")]
    assert [(r["type"], r["seq"]) for r in sent] == [(e["type"], e["seq"]) for e in events]
    assert types.count("debug.segment") == 2
    assert any(
        r["type"] == "debug.hallucination" and r["reason"].startswith("반복 단위") for r in records
    )
    dumps = sorted((tmp_path / "audio").glob(f"*_{session_id}/seg*.wav"))
    assert [p.name for p in dumps] == ["seg0000.wav", "seg0001.wav"]
    with wave.open(str(dumps[0])) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2
        # 발화 0 = 1.5 s 톤 + 앞뒤 패딩 (segmenter speech_pad 200 ms) 근처
        assert 1.4 <= w.getnframes() / 16000 <= 2.2
    seg_rec = next(r for r in records if r["type"] == "debug.segment" and r["seg_id"] == 0)
    assert Path(seg_rec["file"]) == dumps[0] and seg_rec["speakers"] == []


def test_logging_off_writes_nothing(tmp_path: Path):
    settings = Settings(stt=SttConfig(engine="fake"), logging=LoggingConfig(dir=str(tmp_path)))
    run(FakeSTT(), settings)
    assert list(tmp_path.iterdir()) == []
