"""세션 번역 (명세 §4.4·§6.2 `translation.final`) — 가짜 번역기, 가짜 STT, 가짜 스트림 Diarizer."""

import json

from fastapi.testclient import TestClient

from livetrans.core.config import DiarConfig, MtConfig, Settings, SttConfig
from livetrans.main import create_app
from livetrans.plugins.diar.fake import FakeDiarizerEngine
from livetrans.plugins.mt import FallbackTranslator
from livetrans.plugins.mt.base import TranslationError
from livetrans.plugins.mt.passthrough import PassthroughTranslator
from livetrans.plugins.stt.fake import FakeSTT
from tests.integration.test_mic_session import (
    collect_until_stopped,
    create,
    energy_vad,
    send_audio,
    tone,
)
from tests.integration.test_stream_session import activity


class FakeTranslator:
    name = "fake_mt"
    model_id = "fake"

    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    async def load(self):
        return None

    async def health(self):
        return True

    async def close(self):
        return None

    async def translate(self, text, *, src, tgt, context, glossary, length_budget_s=None):
        self.calls.append(
            {
                "text": text,
                "src": src,
                "tgt": tgt,
                "context": list(context),
                "glossary": dict(glossary),
            }
        )
        if self.fail:
            raise TranslationError("엔진 죽음")
        return f"[{src}->{tgt}] {text}"


def run(translator, *, src_lang="ja", stream=True, glossary=None):
    stt = FakeSTT(lang="ja")
    settings = Settings(
        stt=SttConfig(engine="fake"),
        diar=DiarConfig(engine="fake" if stream else "none"),
        mt=MtConfig(engine="passthrough", glossary=glossary),
    )
    diar = (
        FakeDiarizerEngine(activity_fn=activity, hop_s=1.0, right_context_s=0.5) if stream else None
    )
    app = create_app(
        settings,
        stt=stt,
        vad_factory=lambda: energy_vad,
        tts_engines={},
        diar=diar,
        translator=translator,
    )
    with TestClient(app) as client:
        health = client.get("/api/health").json()["models_loaded"]
        body = create(client, src_lang=src_lang)
        with client.websocket_connect(body["ws_url"]) as ws:
            ws.receive_text()
            ws.send_text(json.dumps({"type": "control", "payload": {"action": "start"}}))
            send_audio(ws, tone(6.0))
            ws.send_text(json.dumps({"type": "control", "payload": {"action": "stop"}}))
            events = collect_until_stopped(ws)
    return health, events


def test_each_final_gets_a_translation_with_speaker_and_context():
    mt = FakeTranslator()
    health, events = run(mt)
    assert "mt:fake_mt:fake" in health
    finals = [e["payload"] for e in events if e["type"] == "segment.final"]
    trs = [e["payload"] for e in events if e["type"] == "translation.final"]
    assert [(t["seg_id"], t["speaker"]) for t in trs] == [
        (f["seg_id"], f["speaker"]) for f in finals
    ]
    for f, t in zip(finals, trs, strict=True):
        assert t["text"] == f"[ja->ko] {f['text']}"
        assert (t["src_lang"], t["tgt_lang"], t["engine"], t["fallback"]) == (
            "ja",
            "ko",
            "fake_mt",
            False,
        )
        assert isinstance(t["mt_ms"], int)
    # 번역은 해당 final 뒤에 오고, stopped 전에 전부 도착한다
    for t in trs:
        i_final = next(
            i
            for i, e in enumerate(events)
            if e["type"] == "segment.final" and e["payload"]["seg_id"] == t["seg_id"]
        )
        i_tr = next(
            i
            for i, e in enumerate(events)
            if e["type"] == "translation.final" and e["payload"]["seg_id"] == t["seg_id"]
        )
        assert i_final < i_tr
    assert events[-1]["type"] == "session.state"
    # 화자별 문맥: S0 의 3번째 발화(seg 2) 번역 때 문맥은 S0 의 앞 발화(seg 0)만 (S1 의 seg 1 은 제외)
    third = mt.calls[2]
    assert [c for c, _ in third["context"]] == [finals[0]["text"]]
    assert mt.calls[1]["context"] == []  # S1 첫 발화


def test_same_language_passes_through_without_calling_engine():
    mt = FakeTranslator()
    _, events = run(mt, src_lang="ko")  # FakeSTT 가 ko 로 인식, tgt ko
    trs = [e["payload"] for e in events if e["type"] == "translation.final"]
    assert trs and all(t["engine"] == "passthrough" and not t["fallback"] for t in trs)
    assert mt.calls == []


def test_engine_failure_falls_back_to_original_text():
    _, events = run(FakeTranslator(fail=True))
    trs = [e["payload"] for e in events if e["type"] == "translation.final"]
    finals = {
        e["payload"]["seg_id"]: e["payload"]["text"] for e in events if e["type"] == "segment.final"
    }
    assert trs and all(t["engine"] == "passthrough" and t["fallback"] is True for t in trs)
    assert all(t["text"] == finals[t["seg_id"]] for t in trs)


def test_fallback_chain_in_session_reports_fallback_engine():
    chain = FallbackTranslator([FakeTranslator(fail=True), PassthroughTranslator()])
    _, events = run(chain)
    trs = [e["payload"] for e in events if e["type"] == "translation.final"]
    assert trs and all(t["engine"] == "passthrough" and t["fallback"] is True for t in trs)


def test_no_translator_still_emits_translation_final_as_passthrough():
    _, events = run(None)
    trs = [e["payload"] for e in events if e["type"] == "translation.final"]
    finals = [e["payload"] for e in events if e["type"] == "segment.final"]
    assert len(trs) == len(finals) and all(t["engine"] == "passthrough" for t in trs)


def test_glossary_is_loaded_per_language_pair(tmp_path):
    g = tmp_path / "glossary.yaml"
    g.write_text('"ja->ko": {配信: 방송}\n"en->ko": {chat: 채팅}\n', encoding="utf-8")
    mt = FakeTranslator()
    run(mt, glossary=str(g))
    assert mt.calls and all(c["glossary"] == {"配信": "방송"} for c in mt.calls)


def test_utterance_mode_translation_without_speaker():
    mt = FakeTranslator()
    _, events = run(mt, stream=False)
    trs = [e["payload"] for e in events if e["type"] == "translation.final"]
    assert trs and all("speaker" not in t for t in trs)


class DialogueFakeTranslator(FakeTranslator):
    """`dialogue`·`speaker` 를 받는 엔진 (Ollama 처럼). 세션이 최근 대화를 넘기는지 본다."""

    async def translate(
        self,
        text,
        *,
        src,
        tgt,
        context,
        glossary,
        length_budget_s=None,
        dialogue=None,
        speaker=None,
    ):
        out = await super().translate(
            text,
            src=src,
            tgt=tgt,
            context=context,
            glossary=glossary,
            length_budget_s=length_budget_s,
        )
        self.calls[-1]["dialogue"] = list(dialogue or [])
        self.calls[-1]["speaker"] = speaker
        return out


def test_session_passes_recent_dialogue_of_all_speakers_to_engine():
    mt = DialogueFakeTranslator()
    _, events = run(mt)
    finals = [e["payload"] for e in events if e["type"] == "segment.final"]
    assert len(mt.calls) >= 3
    # S1 의 첫 발화(seg 1) 번역 때: 같은 화자 문맥은 비어 있지만 대화 문맥에는 S0 의 seg 0 이 있다
    second = mt.calls[1]
    assert second["context"] == [] and second["speaker"] == finals[1]["speaker"]
    assert second["dialogue"] == [
        (finals[0]["speaker"], finals[0]["text"], f"[ja->ko] {finals[0]['text']}")
    ]
    # 세 번째 호출은 앞의 두 줄을 순서대로 본다
    assert [d[1] for d in mt.calls[2]["dialogue"]] == [finals[0]["text"], finals[1]["text"]]


def test_engine_without_dialogue_parameter_still_works():
    mt = FakeTranslator()  # translate 에 dialogue 인자가 없다 → 세션이 넘기지 않는다
    _, events = run(mt)
    assert [e for e in events if e["type"] == "translation.final"]
    assert all("dialogue" not in c for c in mt.calls)


class RevisingFakeTranslator(DialogueFakeTranslator):
    """2차 패스(`revise`)를 지원하는 엔진: 번역을 '[rev]' 로, 원문의 '확정' 을 '교정' 으로 바꿔 돌려준다."""

    def __init__(self, *, change=True):
        super().__init__()
        self.revise_calls = []
        self.change = change

    async def revise(
        self, text, *, src, tgt, context, glossary, dialogue=None, speaker=None, upcoming=None
    ):
        self.revise_calls.append(
            {"text": text, "speaker": speaker, "upcoming": list(upcoming or [])}
        )
        if not self.change:
            return f"[{src}->{tgt}] {text}", text
        return f"[rev] {text}", text.replace("확정", "교정")


def _updates(events, kind):
    return [e["payload"] for e in events if e["type"] == kind]


def test_second_pass_revises_earlier_utterance_after_next_final():
    mt = RevisingFakeTranslator()
    _, events = run(mt)
    finals = [e["payload"] for e in events if e["type"] == "segment.final"]
    tr_updates = _updates(events, "translation.update")
    src_updates = _updates(events, "segment.update")
    assert len(finals) >= 2
    first = finals[0]
    # 첫 발화는 다음 발화 확정 뒤 2차 패스: 뒤 발화가 upcoming 으로 들어갔고, 번역·원문 갱신이 왔다
    assert mt.revise_calls and mt.revise_calls[0]["upcoming"][0][1] == finals[1]["text"]
    up = next(u for u in tr_updates if u["seg_id"] == first["seg_id"])
    assert up["text"] == f"[rev] {first['text']}" and up["prev_text"] == f"[ja->ko] {first['text']}"
    assert up["speaker"] == first["speaker"] and up["reason"] == "context"
    su = next(u for u in src_updates if u["seg_id"] == first["seg_id"])
    assert su["text"] == first["text"].replace("확정", "교정") and su["prev_text"] == first["text"]
    # 갱신은 그 발화의 translation.final 뒤에 온다
    i_final = next(
        i
        for i, e in enumerate(events)
        if e["type"] == "translation.final" and e["payload"]["seg_id"] == first["seg_id"]
    )
    i_up = next(
        i
        for i, e in enumerate(events)
        if e["type"] == "translation.update" and e["payload"]["seg_id"] == first["seg_id"]
    )
    assert i_final < i_up
    # 마지막 발화는 뒤 문맥이 없어 2차 패스가 없다
    last = finals[-1]
    assert all(u["seg_id"] != last["seg_id"] for u in tr_updates + src_updates)
    # 발화당 한 번만
    assert len({c["text"] for c in mt.revise_calls}) == len(mt.revise_calls)


def test_second_pass_emits_nothing_when_unchanged():
    mt = RevisingFakeTranslator(change=False)
    _, events = run(mt)
    assert mt.revise_calls
    assert not _updates(events, "translation.update") and not _updates(events, "segment.update")


def test_engine_without_revise_skips_second_pass():
    mt = DialogueFakeTranslator()
    _, events = run(mt)
    assert not _updates(events, "translation.update") and not _updates(events, "segment.update")
