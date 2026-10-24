"""번역 플러그인 (명세 §4.4) — Ollama 어댑터는 httpx MockTransport 로, 네트워크 없이."""

import json
from pathlib import Path

import httpx
import pytest

from livetrans.core.config import MtConfig
from livetrans.plugins.mt import FallbackTranslator, create_translator, load_glossary
from livetrans.plugins.mt.base import TranslationError
from livetrans.plugins.mt.ollama_llm import (
    OllamaTranslator,
    build_messages,
    parse_result,
    parse_translation,
)
from livetrans.plugins.mt.passthrough import PassthroughTranslator


def test_build_messages_has_system_context_and_glossary():
    msgs = build_messages(
        "今日はいい天気ですね",
        src="ja",
        tgt="ko",
        context=[("こんにちは", "안녕하세요")],
        glossary={"配信": "방송"},
        length_budget_s=3.0,
    )
    assert msgs[0]["role"] == "system"
    assert "Japanese" in msgs[0]["content"] and "Korean" in msgs[0]["content"]
    assert "配信 -> 방송" in msgs[0]["content"] and "3 seconds" in msgs[0]["content"]
    assert [m["role"] for m in msgs[1:]] == ["user", "assistant", "user"]
    assert json.loads(msgs[2]["content"]) == {"translation": "안녕하세요"}
    assert msgs[-1]["content"] == "今日はいい天気ですね"


def test_build_messages_puts_dialogue_and_correction_instruction_in_system():
    msgs = build_messages(
        "불레",
        src="ko",
        tgt="ja",
        context=[],
        glossary={},
        dialogue=[("S0", "같이 영화 볼래?", "一緒に映画見る？"), ("S1", "응 좋아", "うん、いいよ")],
        speaker="S1",
    )
    system = msgs[0]["content"]
    assert "Recent dialogue" in system and "S0: 같이 영화 볼래? -> 一緒に映画見る？" in system
    assert "> S1: 응 좋아 -> うん、いいよ" in system  # 현재 화자 표시
    assert "The current speaker is S1" in system
    assert "correct such errors" in system and "omitted subjects" in system
    assert [m["role"] for m in msgs[1:]] == ["user"] and msgs[-1]["content"] == "불레"
    # dialogue 가 없으면 대화 블록도 없다
    plain = build_messages("hi", src="en", tgt="ko", context=[], glossary={})
    assert "Recent dialogue" not in plain[0]["content"]


def test_build_messages_adds_upcoming_lines_for_disambiguation_only():
    msgs = build_messages(
        "그 키 어디 있어",
        src="ko",
        tgt="en",
        context=[],
        glossary={},
        upcoming=[("S1", "파란 열쇠 말이야?"), ("S0", "응 상자 열 때 쓰는 거")],
    )
    system = msgs[0]["content"]
    assert "spoken right after the current utterance" in system
    assert "S1: 파란 열쇠 말이야?" in system and "S0: 응 상자 열 때 쓰는 거" in system
    assert "never translate or merge" in system
    assert msgs[-1]["content"] == "그 키 어디 있어"
    assert (
        "spoken right after"
        not in build_messages("hi", src="en", tgt="ko", context=[], glossary={})[0]["content"]
    )


def test_build_messages_second_pass_asks_for_corrected_source():
    msgs = build_messages("불레", src="ko", tgt="ja", context=[], glossary={}, want_source=True)
    system = msgs[0]["content"]
    assert "second pass" in system and '"source": "<corrected original>"' in system
    assert (
        '"source"'
        not in build_messages("불레", src="ko", tgt="ja", context=[], glossary={})[0]["content"]
    )


def test_parse_result_returns_translation_and_optional_source():
    assert parse_result('{"translation": "見る？", "source": "볼래?"}') == ("見る？", "볼래?")
    assert parse_result('{"translation": "見る？"}') == ("見る？", None)
    assert parse_result('{"translation": "見る？", "source": "  "}') == ("見る？", None)
    assert parse_result('{"source": "볼래?"}') == (None, None)
    assert parse_result("garbage") == (None, None)


async def test_ollama_revise_returns_translation_and_source():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen2.5:7b"}]})
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "message": {"content": '{"translation": "一緒に見る？", "source": "같이 볼래?"}'}
            },
        )

    mt = _ollama(handler)
    out = await mt.revise(
        "같이 불레",
        src="ko",
        tgt="ja",
        context=[],
        glossary={},
        dialogue=[("S0", "영화 어때", "映画どう")],
        speaker="S1",
        upcoming=[("S0", "액션으로")],
    )
    assert out == ("一緒に見る？", "같이 볼래?")
    system = seen["body"]["messages"][0]["content"]
    assert (
        "second pass" in system and "Recent dialogue" in system and "spoken right after" in system
    )


def test_parse_translation_accepts_json_and_rejects_garbage():
    assert parse_translation('{"translation": "안녕"}') == "안녕"
    assert parse_translation('Sure! {"translation": "안녕"} ok') == "안녕"
    assert parse_translation("") is None
    assert parse_translation("just text") is None
    assert parse_translation('{"other": 1}') is None
    assert parse_translation('{"translation": 3}') is None


def _ollama(handler, **cfg) -> OllamaTranslator:
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(base_url="http://ollama.test", transport=transport)
    return OllamaTranslator(MtConfig(engine="ollama_llm", model="qwen2.5:7b", **cfg), client=client)


async def test_ollama_translate_posts_chat_with_json_format():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen2.5:7b"}]})
        seen["body"] = json.loads(req.content)
        return httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": '{"translation": "오늘 날씨 좋네요"}'}
            },
        )

    mt = _ollama(handler)
    await mt.load()
    out = await mt.translate("今日はいい天気", src="ja", tgt="ko", context=[], glossary={})
    assert out == "오늘 날씨 좋네요"
    body = seen["body"]
    assert body["model"] == "qwen2.5:7b" and body["format"] == "json" and body["stream"] is False
    assert body["messages"][-1] == {"role": "user", "content": "今日はいい天気"}
    assert body["options"]["temperature"] == 0.1


async def test_ollama_retries_once_on_bad_json_then_raises():
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(json.loads(req.content))
        return httpx.Response(200, json={"message": {"content": "not json at all"}})

    mt = _ollama(handler)
    with pytest.raises(TranslationError):
        await mt.translate("x", src="en", tgt="ko", context=[], glossary={})
    assert len(calls) == 2
    assert "Invalid format" in calls[1]["messages"][-1]["content"]


async def test_ollama_timeout_and_http_error_become_translation_error():
    def timeout(req: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=req)

    with pytest.raises(TranslationError):
        await _ollama(timeout).translate("x", src="en", tgt="ko", context=[], glossary={})

    def err(req: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    with pytest.raises(TranslationError):
        await _ollama(err).translate("x", src="en", tgt="ko", context=[], glossary={})


async def test_ollama_load_fails_when_model_missing():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"name": "other:1b"}]})

    with pytest.raises(TranslationError, match="qwen2.5:7b"):
        await _ollama(handler).load()


class Boom:
    name = "boom"
    model_id = "x"

    def __init__(self, fail_load=False):
        self.fail_load = fail_load

    async def load(self):
        if self.fail_load:
            raise TranslationError("없음")

    async def health(self):
        return True

    async def translate(self, text, *, src, tgt, context, glossary, length_budget_s=None):
        raise TranslationError("실패")

    async def close(self):
        return None


async def test_fallback_chain_marks_fallback_and_drops_dead_engines():
    chain = FallbackTranslator([Boom(), PassthroughTranslator()])
    await chain.load()
    tr = await chain.translate_full(
        7, "hello", src="en", tgt="ko", context=[], glossary={}, speaker="S1"
    )
    assert (tr.text, tr.engine, tr.fallback, tr.seg_id, tr.speaker) == (
        "hello",
        "passthrough",
        True,
        7,
        "S1",
    )

    dead = FallbackTranslator([Boom(fail_load=True), PassthroughTranslator()])
    await dead.load()
    assert [e.name for e in dead.active] == ["passthrough"] and dead.name == "passthrough"
    tr = await dead.translate_full(1, "hi", src="en", tgt="ko", context=[], glossary={})
    assert tr.fallback is False  # 살아 있는 첫 엔진이 곧 1순위

    with pytest.raises(TranslationError):
        await FallbackTranslator([Boom(fail_load=True)]).load()


def test_create_translator_appends_passthrough():
    chain = create_translator(MtConfig(engine="ollama_llm", fallback=[]))
    assert isinstance(chain, FallbackTranslator)
    assert [e.name for e in chain.active] == ["ollama_llm", "passthrough"]
    assert isinstance(
        create_translator(MtConfig(engine="passthrough", fallback=[])), PassthroughTranslator
    )
    with pytest.raises(ValueError):
        create_translator(MtConfig(engine="nope"))


def test_load_glossary_merges_pair_and_wildcards(tmp_path: Path):
    base = tmp_path / "glossary.yaml"
    base.write_text(
        '"*->ko": {Twitch: 트위치}\n"ja->ko": {配信: 방송}\n"en->ko": {chat: 채팅}\n',
        encoding="utf-8",
    )
    (tmp_path / "glossary.local.yaml").write_text(
        '"ja->ko": {リスナー: 시청자, 配信: 스트리밍}\n', encoding="utf-8"
    )
    g = load_glossary(base, "ja", "ko")
    assert g == {"Twitch": "트위치", "配信": "스트리밍", "リスナー": "시청자"}  # local 이 덮어쓴다
    assert load_glossary(base, "en", "ja") == {}
    assert (
        load_glossary(None, "ja", "ko") == {}
        and load_glossary(tmp_path / "none.yaml", "ja", "ko") == {}
    )


def test_script_problem_detects_wrong_writing_system():
    from livetrans.plugins.mt.ollama_llm import script_problem

    assert script_problem("차를 열为啥要开这个车呢", "ko") is not None
    assert script_problem("안녕하세요 여러분 자막 테스트입니다", "ko") is None
    assert script_problem("Today's weather is nice.", "en") is None
    assert script_problem("Today 날씨 is nice", "en") is not None
    assert script_problem("今日はいい天気ですね", "ja") is None
    assert script_problem("今日は 날씨", "ja") is not None
    assert script_problem("1234 !!", "ko") is None


async def test_ollama_retries_when_output_script_is_wrong_then_raises():
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(json.loads(req.content))
        return httpx.Response(
            200, json={"message": {"content": '{"translation": "차를 열为啥要开这个车呢"}'}}
        )

    with pytest.raises(TranslationError, match="한자"):
        await _ollama(handler).translate(
            "can i ask why", src="en", tgt="ko", context=[], glossary={}
        )
    assert len(calls) == 2 and "Korean script" in calls[1]["messages"][-1]["content"]
