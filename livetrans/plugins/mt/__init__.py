"""번역 엔진 생성 + 폴백 체인 (명세 §4.4: `primary → secondary → passthrough`)."""

from __future__ import annotations

import inspect
import logging
import time
from pathlib import Path

import yaml

from livetrans.core.config import MtConfig
from livetrans.plugins.mt.base import ContextPair, Translation, TranslationError, Translator

log = logging.getLogger("livetrans.mt")


def _create(engine: str, cfg: MtConfig) -> Translator:
    if engine == "ollama_llm":
        from livetrans.plugins.mt.ollama_llm import OllamaTranslator

        return OllamaTranslator(cfg)
    if engine == "passthrough":
        from livetrans.plugins.mt.passthrough import PassthroughTranslator

        return PassthroughTranslator()
    raise ValueError(f"알 수 없는 번역 엔진: {engine}")


def create_translator(cfg: MtConfig) -> Translator:
    """설정의 엔진 체인. 마지막에 `passthrough` 가 없으면 붙인다 (자막은 번역이 실패해도 원문으로 나와야 한다)."""
    names = [cfg.engine, *cfg.fallback]
    if "passthrough" not in names:
        names.append("passthrough")
    engines = [_create(n, cfg) for n in names]
    return engines[0] if len(engines) == 1 else FallbackTranslator(engines)


def load_glossary(path: str | Path | None, src: str, tgt: str) -> dict[str, str]:
    """`config/glossary.yaml` (+ `glossary.local.yaml`) 에서 언어쌍 용어집. 형식: `{"ko->en": {용어: 번역}, "*->ko": {...}}`."""
    if not path:
        return {}
    merged: dict[str, str] = {}
    base = Path(path)
    for p in (base, base.with_name(base.stem + ".local" + base.suffix)):
        if not p.is_file():
            continue
        try:
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as e:
            log.warning("용어집을 읽을 수 없음 (%s): %s", p, e)
            continue
        if not isinstance(data, dict):
            continue
        for key in (f"*->{tgt}", f"{src}->*", f"{src}->{tgt}"):
            table = data.get(key)
            if isinstance(table, dict):
                merged.update({str(k): str(v) for k, v in table.items()})
    return merged


def _translate_params_of(fn) -> frozenset[str]:
    try:
        return frozenset(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return frozenset()


def _translate_params(engine) -> frozenset[str]:
    """엔진 translate 의 키워드 인자 이름 (1회 검사·캐시). passthrough·테스트 가짜는 문맥 인자를 안 받을 수 있다."""
    cached = getattr(engine, "_translate_params", None)
    if cached is None:
        try:
            cached = frozenset(inspect.signature(engine.translate).parameters)
        except (TypeError, ValueError):
            cached = frozenset()
        try:
            engine._translate_params = cached
        except AttributeError:
            pass
    return cached


def accepts_dialogue(engine) -> bool:
    return "dialogue" in _translate_params(engine)


def context_kwargs(engine, *, dialogue, speaker, upcoming) -> dict:
    """엔진이 받는 문맥 인자만 골라 넘긴다 (dialogue·speaker·upcoming)."""
    params = _translate_params(engine)
    out: dict = {}
    if "dialogue" in params:
        out["dialogue"] = dialogue
        if "speaker" in params:
            out["speaker"] = speaker
    if "upcoming" in params:
        out["upcoming"] = upcoming
    return out


class FallbackTranslator:
    """엔진 체인: 앞 엔진이 로드에 실패하면 빼고, 번역에 실패하면 다음 엔진으로. `fallback` 플래그를 Translation 에 남긴다."""

    def __init__(self, engines: list[Translator]) -> None:
        self._engines = engines
        self.active: list[Translator] = list(engines)

    @property
    def name(self) -> str:
        return self.active[0].name if self.active else "none"

    @property
    def model_id(self) -> str:
        return self.active[0].model_id if self.active else "-"

    async def load(self) -> None:
        active: list[Translator] = []
        for e in self._engines:
            try:
                await e.load()
                active.append(e)
            except Exception as err:  # noqa: BLE001 — 한 엔진이 안 떠도 체인은 살아야 한다
                log.warning(
                    "번역 엔진 %s(%s) 로드 실패 → 체인에서 제외: %s", e.name, e.model_id, err
                )
        if not active:
            raise TranslationError("쓸 수 있는 번역 엔진이 없습니다")
        self.active = active

    async def health(self) -> bool:
        return bool(self.active) and await self.active[0].health()

    async def translate(
        self,
        text: str,
        *,
        src: str,
        tgt: str,
        context: list[ContextPair],
        glossary: dict[str, str],
        length_budget_s: float | None = None,
        dialogue: list | None = None,
        speaker: str | None = None,
        upcoming: list | None = None,
    ) -> str:
        return (
            await self.translate_full(
                0,
                text,
                src=src,
                tgt=tgt,
                context=context,
                glossary=glossary,
                length_budget_s=length_budget_s,
                speaker=speaker,
                dialogue=dialogue,
                upcoming=upcoming,
            )
        ).text

    async def translate_full(
        self,
        seg_id: int,
        text: str,
        *,
        src: str,
        tgt: str,
        context: list[ContextPair],
        glossary: dict[str, str],
        length_budget_s: float | None = None,
        speaker: str | None = None,
        dialogue: list | None = None,
        upcoming: list | None = None,
    ) -> Translation:
        t0 = time.perf_counter()
        last: Exception | None = None
        for i, e in enumerate(self.active):
            try:
                extra = context_kwargs(e, dialogue=dialogue, speaker=speaker, upcoming=upcoming)
                out = await e.translate(
                    text,
                    src=src,
                    tgt=tgt,
                    context=context,
                    glossary=glossary,
                    length_budget_s=length_budget_s,
                    **extra,
                )
                return Translation(
                    seg_id=seg_id,
                    text=out,
                    src_lang=src,
                    tgt_lang=tgt,
                    engine=e.name,
                    fallback=i > 0,
                    latency_ms=int((time.perf_counter() - t0) * 1000),
                    speaker=speaker,
                )
            except Exception as err:  # noqa: BLE001
                last = err
                log.warning("번역 실패 %s seg=%s → 다음 엔진: %s", e.name, seg_id, err)
        raise TranslationError(f"모든 번역 엔진 실패: {last}")

    async def revise(self, text: str, **kw) -> tuple[str, str | None] | None:
        """2차 패스를 지원하는 첫 활성 엔진(Ollama)으로. 지원 엔진이 없으면 None (2차 패스 생략)."""
        for e in self.active:
            fn = getattr(e, "revise", None)
            if fn is None:
                continue
            params = _translate_params_of(fn)
            return await fn(text, **{k: v for k, v in kw.items() if k in params})
        return None

    async def close(self) -> None:
        for e in self._engines:
            await e.close()
