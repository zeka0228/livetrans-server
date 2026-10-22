"""Voice Registry (명세 §5.1): 언어 → 음성 결정, 합성 결과 캐시.

언어는 STT 가 주는 ISO 639-1("ja") 또는 BCP-47("ja-JP") 둘 다 받는다.
기본 음성은 설정 `tts.defaults` 의 `"<엔진>:<voice_id>"`, `voice_id` 가 `auto` 면 그 엔진에서 언어가 맞는 첫 음성.
기본값으로 못 찾으면 언어가 맞는 아무 음성으로 폴백한다.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass

from livetrans.plugins.tts.base import TTSEngine, VoiceInfo

log = logging.getLogger("livetrans.voices")

DEFAULT_LOCALE = {"ko": "ko-KR", "en": "en-US", "ja": "ja-JP"}


class NoVoiceError(LookupError):
    pass


def normalize_lang(lang: str) -> str:
    lang = lang.strip()
    return lang if "-" in lang else DEFAULT_LOCALE.get(lang.lower(), lang.lower())


def _primary(lang: str) -> str:
    return lang.split("-")[0].lower()


def matches(voice: VoiceInfo, lang: str) -> bool:
    """정확한 로캘이 같거나, 주 언어(ja-JP 의 ja)가 같으면 맞는 음성."""
    want = normalize_lang(lang)
    return any(v.lower() == want.lower() or _primary(v) == _primary(want) for v in voice.languages)


@dataclass
class SynthResult:
    wav: bytes
    voice: VoiceInfo
    duration_ms: int
    cached: bool = False


class VoiceRegistry:
    def __init__(
        self, engines: dict[str, TTSEngine], defaults: dict[str, str], *, cache_items: int = 128
    ) -> None:
        self.engines = engines
        self.defaults = {normalize_lang(k): v for k, v in defaults.items()}
        self._voices: dict[str, list[VoiceInfo]] = {}
        self._cache: OrderedDict[tuple, SynthResult] = OrderedDict()
        self._cache_items = cache_items

    async def refresh(self) -> None:
        self._voices = {name: await e.list_voices() for name, e in self.engines.items()}

    def all_voices(self) -> list[VoiceInfo]:
        return [v for name in self.engines for v in self._voices.get(name, [])]

    def voices(self, lang: str | None = None) -> list[VoiceInfo]:
        return [v for v in self.all_voices() if lang is None or matches(v, lang)]

    def resolve(self, lang: str | None, voice_ref: str | None = None) -> VoiceInfo:
        if voice_ref:
            voice = next((v for v in self.all_voices() if v.voice_ref == voice_ref), None)
            if voice is None:
                raise NoVoiceError(f"음성이 없습니다: {voice_ref}")
            return voice
        if not lang:
            raise NoVoiceError("lang 또는 voice_ref 가 필요합니다")
        locale = normalize_lang(lang)
        ref = self.defaults.get(locale) or next(
            (r for loc, r in self.defaults.items() if _primary(loc) == _primary(locale)), None
        )
        if ref:
            engine, _, voice_id = ref.partition(":")
            for v in self._voices.get(engine, []):
                if (voice_id == "auto" and matches(v, locale)) or v.voice_ref == ref:
                    return v
            log.info("기본 음성 %s 를 못 찾아 폴백 (lang=%s)", ref, locale)
        candidates = self.voices(locale)
        if not candidates:
            raise NoVoiceError(f"이 언어의 음성이 없습니다: {locale}")
        return candidates[0]

    async def synthesize(
        self, text: str, *, lang: str | None, voice_ref: str | None = None, rate: float = 1.0
    ) -> SynthResult:
        voice = self.resolve(lang, voice_ref)
        key = (voice.voice_ref, round(rate, 3), text)
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            return SynthResult(hit.wav, hit.voice, hit.duration_ms, cached=True)
        engine_name, _, voice_id = voice.voice_ref.partition(":")
        clip = await self.engines[engine_name].synthesize(text, voice_id, rate=rate)
        result = SynthResult(clip.to_wav(), voice, int(clip.duration_s * 1000))
        self._cache[key] = result
        if len(self._cache) > self._cache_items:
            self._cache.popitem(last=False)
        return result
