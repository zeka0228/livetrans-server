"""Ollama 로컬 LLM 번역 어댑터 (명세 §4.4 `ollama_llm`).

- `POST {url}/api/chat`, `format: "json"` 으로 **JSON 한 줄** `{"translation": "..."}` 만 받는다. 파싱 실패 시 1회 재시도 후 `TranslationError`.
- 프롬프트: 역할(생방송 자막 번역가) + 원어·대상어 + 용어집 + 직전 문맥 쌍(화자별, 세션이 넘김) + 원문.
- `keep_alive` 로 모델을 상주시킨다. 첫 호출(모델 로드)만 수 초.
- 2026-10-25 벤치 (RTX 4070 Ti SUPER, 서버와 동시 상주, 40문장 chrF ja/en): **gemma4:e4b 56/43, 0.2 s, 3.3 GB (기본, think=false)**,
  gemma3:12b-it-qat 41/34 1.0 s 8.8 GB, gemma2:9b 39/24 0.3 s 6.3 GB (2026-10-23 까지 기본). qwen2.5:7b 는 더 작지만
  한국어 출력 중 중국어로 새는 일이 잦고(`script_problem` 으로 걸러 폴백), qwen2.5:3b 는 오역, gemma4:e4b 는 JSON 강제가 안 됨. 모델은 `mt.model`.
  VRAM 이 모자라면(다른 Ollama 모델이 함께 올라와 있으면) CPU 로 밀려 수 초가 걸린다 — `ollama ps` 로 100 % GPU 인지 확인.
- Ollama 는 별도 프로세스가 GPU 를 쓴다 → `GPU_LOCK` 밖. VRAM 예산(§10)에 모델 크기만큼 더해 잡아야 한다 (7b Q4 ≈ 5 GB).
"""

from __future__ import annotations

import json
import logging
import re
import time

import httpx

from livetrans.core.config import MtConfig
from livetrans.plugins.mt.base import ContextPair, TranslationError

log = logging.getLogger("livetrans.mt.ollama")

LANG_NAMES = {
    "ko": "Korean",
    "en": "English",
    "ja": "Japanese",
    "zh": "Chinese",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ru": "Russian",
    "pt": "Portuguese",
    "id": "Indonesian",
    "th": "Thai",
    "vi": "Vietnamese",
}


DialogueLine = tuple[str | None, str, str]  # (화자, 원문, 번역)
UpcomingLine = tuple[str | None, str]  # (화자, 원문) — 아직 번역 안 된 뒤 발화
MAX_DIALOGUE_LINES = 12
MAX_UPCOMING_LINES = 3
MAX_HYPOTHESES = 4
MAX_UNCERTAIN_WORDS = 8


def _clip(s: str, n: int = 80) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def lang_name(code: str) -> str:
    return LANG_NAMES.get(code.split("-")[0].lower(), code)


def build_messages(
    text: str,
    *,
    src: str,
    tgt: str,
    context: list[ContextPair],
    glossary: dict[str, str],
    length_budget_s: float | None = None,
    dialogue: list[DialogueLine] | None = None,
    speaker: str | None = None,
    upcoming: list[UpcomingLine] | None = None,
    want_source: bool = False,
    hypotheses: list[str] | None = None,
    uncertain_words: list[str] | None = None,
) -> list[dict[str, str]]:
    """Ollama chat 메시지. 같은 화자의 문맥 쌍은 (user 원문 → assistant JSON) 대화로 넣어 모델이 형식을 따라가게 하고,
    최근 대화(모든 화자)는 시스템 메시지에 "화자: 원문 -> 번역" 줄로 넣는다 (지시어·생략 주어·STT 오류 교정 재료)."""
    system = (
        f"You are a live-broadcast subtitle translator. Translate the user's {lang_name(src)} utterance "
        f"into natural, concise {lang_name(tgt)} as spoken subtitles for viewers. Keep the speaker's tone. "
        "Keep proper names, game terms and numbers; do not add explanations or omit content. "
        "The input is automatic speech recognition output without punctuation and may contain recognition errors "
        "(homophones, dropped particles, split or merged words). First silently correct such errors using the "
        "dialogue context, then translate the intended meaning. Resolve pronouns, omitted subjects and the level of "
        "politeness from the dialogue, and keep names and terms consistent with earlier lines."
    )
    if dialogue:
        tag = f" The current speaker is {speaker}." if speaker else ""
        shown = "\n".join(
            f"{'> ' if speaker and who == speaker else ''}{who or 'speaker'}: {_clip(s)} -> {_clip(t)}"
            for who, s, t in dialogue[-MAX_DIALOGUE_LINES:]
        )
        system += " Recent dialogue, oldest first (speaker: original -> translation):\n"
        system += f"{shown}\n{tag}"
    if upcoming:
        nxt = "\n".join(
            f"{who or 'speaker'}: {_clip(s)}" for who, s in upcoming[:MAX_UPCOMING_LINES]
        )
        system += (
            " Lines spoken right after the current utterance (not yet translated). Use them only to "
            "disambiguate words and fix recognition errors in the current line; never translate or merge "
            f"them into the output:\n{nxt}\n"
        )
    if glossary:
        pairs = "; ".join(f"{k} -> {v}" for k, v in list(glossary.items())[:60])
        system += f" Use these fixed translations when the term appears: {pairs}."
    if length_budget_s:
        system += f" The translation will be spoken aloud in about {length_budget_s:.0f} seconds, so keep it short."
    rescoring = bool(hypotheses and len(hypotheses) > 1) or bool(uncertain_words)
    candidates = hypotheses[:MAX_HYPOTHESES] if hypotheses and len(hypotheses) > 1 else None
    if candidates:
        # 후보는 사용자 턴에 목록으로 준다 (아래). 시스템 메시지에 두고 1번을 사용자 턴으로 주면 모델이 사용자 턴을
        # '정답'으로 고착해 문맥상 틀린 후보를 그대로 둔다 (2026-10-25 gemma4 실험: 5문장 중 2 → 6문장 중 5 정답)
        system += (
            " The recognizer was unsure about this line and produced several candidate transcriptions; they "
            "differ exactly where it was unsure. Decide which candidate is what the speaker most likely said, "
            "judging by meaning, common collocations and the conversation (a candidate that makes sense in "
            "context beats one that does not, regardless of order). Then translate that candidate and return it "
            'verbatim as "source".'
        )
    if uncertain_words:
        system += (
            " Low-confidence words in the current line: "
            + ", ".join(f'"{w}"' for w in uncertain_words[:MAX_UNCERTAIN_WORDS])
            + ". Consider likely misrecognitions of these words given the context."
        )
    if want_source or rescoring:
        want_source = True
    if want_source:
        system += (
            ' This is a second pass with more context. Also return the corrected original line as "source": '
            "the same language and the same content with recognition errors fixed; if nothing needs fixing, "
            'copy the original unchanged. Also return "confidence": a number from 0 to 1 for how sure you are '
            "that the final source is what was actually said (1 = certain). Respond with JSON only: "
            '{"translation": "<translated text>", "source": "<corrected original>", "confidence": <0-1>}'
        )
    else:
        system += ' Respond with JSON only: {"translation": "<translated text>"}'
    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    for s, t in context:
        messages.append({"role": "user", "content": s})
        messages.append(
            {"role": "assistant", "content": json.dumps({"translation": t}, ensure_ascii=False)}
        )
    if candidates:
        listed = "\n".join(f"{i}) {_clip(h)}" for i, h in enumerate(candidates, 1))
        messages.append(
            {"role": "user", "content": f"Candidate transcriptions of the current line:\n{listed}"}
        )
    else:
        messages.append({"role": "user", "content": text})
    return messages


_HAN = re.compile(r"[一-鿿㐀-䶿]")
_HANGUL = re.compile(r"[가-힣]")
_KANA = re.compile(r"[぀-ヿ]")


def script_problem(text: str, tgt: str) -> str | None:
    """대상 언어와 맞지 않는 문자 체계가 섞였으면 이유 (qwen 계열이 한국어 출력 중 중국어로 새는 사례 — 2026-10-23 측정)."""
    code = tgt.split("-")[0].lower()
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return None
    if code == "ko":
        han = len(_HAN.findall(text))
        if han and han / len(letters) > 0.05:
            return f"한국어 출력에 한자 {han}자"
        if _KANA.search(text):
            return "한국어 출력에 가나 문자"
    elif code == "en":
        if _HANGUL.search(text) or _KANA.search(text) or _HAN.search(text):
            return "영어 출력에 CJK 문자"
    elif code == "ja":
        if _HANGUL.search(text):
            return "일본어 출력에 한글"
    return None


def parse_translation(content: str) -> str | None:
    """모델 출력(JSON 한 줄)에서 번역문. 형식이 어긋나면 None."""
    return parse_result(content)[0]


def parse_result(content: str) -> tuple[str | None, str | None, float | None]:
    """모델 출력에서 (번역문, 교정 원문, 확신도). 번역문이 없으면 (None, None, None); 뒤 둘은 요청했을 때만 온다."""
    content = content.strip()
    if not content:
        return None, None, None
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # 앞뒤에 군말이 붙은 경우 첫 { … } 만 시도
        a, b = content.find("{"), content.rfind("}")
        if a < 0 or b <= a:
            return None, None, None
        try:
            data = json.loads(content[a : b + 1])
        except json.JSONDecodeError:
            return None, None, None
    if isinstance(data, dict):
        tr = data.get("translation")
        if not isinstance(tr, str) or not tr.strip():
            return None, None, None
        src = data.get("source")
        src = src.strip() if isinstance(src, str) and src.strip() else None
        conf = data.get("confidence")
        try:
            conf = min(1.0, max(0.0, float(conf))) if conf is not None else None
        except (TypeError, ValueError):
            conf = None
        return tr.strip(), src, conf
    return None, None, None


class OllamaTranslator:
    name = "ollama_llm"

    def __init__(self, cfg: MtConfig, client: httpx.AsyncClient | None = None) -> None:
        self.cfg = cfg
        self.model_id = cfg.model
        self._client = client or httpx.AsyncClient(base_url=cfg.url, timeout=cfg.timeout_s)
        self._owns_client = client is None

    async def load(self) -> None:
        """서버 연결·모델 존재 확인 후 짧은 호출로 모델을 올려 둔다 (keep_alive)."""
        try:
            r = await self._client.get("/api/tags", timeout=5.0)
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise TranslationError(f"Ollama 에 연결할 수 없습니다 ({self.cfg.url}): {e}") from e
        names = {m.get("name", "") for m in r.json().get("models", [])}
        if self.model_id not in names and f"{self.model_id}:latest" not in names:
            raise TranslationError(
                f"Ollama 에 모델 {self.model_id} 이 없습니다 (ollama pull {self.model_id}). 있는 모델: {sorted(names)}"
            )
        t0 = time.perf_counter()
        await self.translate("hello", src="en", tgt="ko", context=[], glossary={})
        log.info("Ollama 번역 모델 준비: %s (%.1f s)", self.model_id, time.perf_counter() - t0)

    async def health(self) -> bool:
        try:
            r = await self._client.get("/api/tags", timeout=3.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def list_models(self) -> list[dict]:
        """Ollama 에 설치된 모델 (세션 요청 `mt.model` 후보, API 계약 S-26). 못 받으면 빈 목록."""
        try:
            r = await self._client.get("/api/tags", timeout=3.0)
            r.raise_for_status()
        except httpx.HTTPError:
            return []
        out = []
        for m in r.json().get("models", []):
            name = m.get("name") or ""
            if name and "embed" not in name:  # 임베딩 모델은 번역에 못 쓴다
                out.append({"name": name, "size_mb": int(int(m.get("size", 0)) // (1024 * 1024))})
        return out

    async def translate(
        self,
        text: str,
        *,
        src: str,
        tgt: str,
        context: list[ContextPair],
        glossary: dict[str, str],
        length_budget_s: float | None = None,
        dialogue: list[DialogueLine] | None = None,
        speaker: str | None = None,
        upcoming: list[UpcomingLine] | None = None,
        hypotheses: list[str] | None = None,
        uncertain_words: list[str] | None = None,
    ) -> str:
        return (
            await self.translate_ex(
                text,
                src=src,
                tgt=tgt,
                context=context,
                glossary=glossary,
                length_budget_s=length_budget_s,
                dialogue=dialogue,
                speaker=speaker,
                upcoming=upcoming,
                hypotheses=hypotheses,
                uncertain_words=uncertain_words,
            )
        )[0]

    async def translate_ex(
        self,
        text: str,
        *,
        src: str,
        tgt: str,
        context: list[ContextPair],
        glossary: dict[str, str],
        length_budget_s: float | None = None,
        dialogue: list[DialogueLine] | None = None,
        speaker: str | None = None,
        upcoming: list[UpcomingLine] | None = None,
        hypotheses: list[str] | None = None,
        uncertain_words: list[str] | None = None,
    ) -> tuple[str, str | None, float | None]:
        """1차 번역 (번역, 고른/교정한 원문 또는 None, 확신도 또는 None). 후보·불확실 단어가 있으면 모델이 읽기를 고르고
        확신도를 돌려준다 (문맥 재채점, S-30); 없으면 평소처럼 번역만."""
        messages = build_messages(
            text,
            src=src,
            tgt=tgt,
            context=context,
            glossary=glossary,
            length_budget_s=length_budget_s,
            dialogue=dialogue,
            speaker=speaker,
            upcoming=upcoming,
            hypotheses=hypotheses,
            uncertain_words=uncertain_words,
        )
        return await self._ask(messages, tgt)

    async def revise(
        self,
        text: str,
        *,
        src: str,
        tgt: str,
        context: list[ContextPair],
        glossary: dict[str, str],
        dialogue: list[DialogueLine] | None = None,
        speaker: str | None = None,
        upcoming: list[UpcomingLine] | None = None,
        hypotheses: list[str] | None = None,
        uncertain_words: list[str] | None = None,
    ) -> tuple[str, str | None, float | None]:
        """2차 패스: 앞뒤 문맥으로 다시 번역하고 교정 원문·확신도도 받는다 → (번역, 교정 원문 또는 None, 확신도)."""
        messages = build_messages(
            text,
            src=src,
            tgt=tgt,
            context=context,
            glossary=glossary,
            dialogue=dialogue,
            speaker=speaker,
            upcoming=upcoming,
            want_source=True,
            hypotheses=hypotheses,
            uncertain_words=uncertain_words,
        )
        return await self._ask(messages, tgt)

    async def _ask(
        self, messages: list[dict[str, str]], tgt: str
    ) -> tuple[str, str | None, float | None]:
        """모델에 묻고 (번역, 교정 원문, 확신도) 를 돌려준다. 형식·문자 체계가 어긋나면 한 번 고쳐 달라고 한다."""
        why = "형식 오류"
        for attempt in range(2):
            content = await self._chat(messages)
            out, fixed, conf = parse_result(content)
            problem = None if out is None else script_problem(out, tgt)
            if out is not None and problem is None:
                return out, fixed, conf
            why = "형식 오류" if out is None else problem
            log.warning("Ollama 번역 %s (시도 %d): %r", why, attempt + 1, content[:120])
            fix = (
                'Invalid format. Respond with JSON only: {"translation": "..."}'
                if out is None
                else f"The translation must be written entirely in {lang_name(tgt)} script "
                f"(no other writing systems except proper names). Translate again. JSON only."
            )
            messages = messages + [
                {"role": "assistant", "content": content},
                {"role": "user", "content": fix},
            ]
        raise TranslationError(f"Ollama 번역 결과를 쓸 수 없습니다 ({why})")

    async def _chat(self, messages: list[dict[str, str]]) -> str:
        body = {
            "model": self.model_id,
            "messages": messages,
            "stream": False,
            "format": "json",
            "keep_alive": self.cfg.keep_alive,
            "think": self.cfg.think,  # 추론 모델의 사고 출력 (MtConfig.think). 비추론 모델은 무시한다
            "options": {
                "temperature": self.cfg.temperature,
                "num_predict": self.cfg.max_tokens,
                # 러너 로딩 방식 (MtConfig.use_mmap). 모델을 처음 올리는 요청에만 영향, 이후 요청은 무시된다
                "use_mmap": self.cfg.use_mmap,
            },
        }
        try:
            r = await self._client.post("/api/chat", json=body, timeout=self.cfg.timeout_s)
            r.raise_for_status()
        except httpx.TimeoutException as e:
            raise TranslationError(f"Ollama 번역 시간 초과 ({self.cfg.timeout_s}s)") from e
        except httpx.HTTPError as e:
            raise TranslationError(f"Ollama 요청 실패: {e}") from e
        try:
            return r.json()["message"]["content"]
        except (KeyError, ValueError, TypeError) as e:
            raise TranslationError(f"Ollama 응답 형식 오류: {e}") from e

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()
