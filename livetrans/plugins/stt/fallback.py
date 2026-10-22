"""기본 엔진 로드 실패 시 폴백 엔진으로 넘어가는 래퍼 (명세 §4.3 "DiCoW 실패 시 faster-whisper 폴백", R-15).

로드 단계에서만 폴백한다. 인식 중 오류는 세션이 `E_STT_FAILED` 로 처리하므로 여기서 다시 시도하지 않는다
(폴백 모델까지 미리 올리면 VRAM 을 두 배로 쓰기 때문).
"""

from __future__ import annotations

import logging

from livetrans.pipeline.segmenter import SpeechSegment
from livetrans.plugins.stt.base import STTEngine, Transcript

log = logging.getLogger("livetrans.stt")


class FallbackSTT:
    def __init__(self, primary: STTEngine, fallback: STTEngine) -> None:
        self.primary = primary
        self.fallback = fallback
        self.active: STTEngine = primary
        self.fell_back = False

    @property
    def name(self) -> str:
        return self.active.name

    @property
    def model_id(self) -> str:
        return self.active.model_id

    @property
    def languages(self) -> tuple[str, ...]:
        return self.active.languages

    @property
    def supports_speaker_mask(self) -> bool:
        return getattr(self.active, "supports_speaker_mask", False)

    async def load(self) -> None:
        try:
            await self.primary.load()
            self.active = self.primary
        except Exception as e:  # noqa: BLE001 — 어떤 이유로든 기본 엔진이 안 뜨면 폴백으로 서버는 떠야 한다
            log.warning(
                "STT 기본 엔진 %s(%s) 로드 실패 → 폴백 %s(%s): %s",
                self.primary.name,
                self.primary.model_id,
                self.fallback.name,
                self.fallback.model_id,
                e,
            )
            await self.fallback.load()
            self.active = self.fallback
            self.fell_back = True

    async def transcribe(
        self,
        seg: SpeechSegment,
        *,
        lang: str | None,
        prompt: str | None,
        final: bool,
        speaker: str | None = None,
    ) -> Transcript:
        return await self.active.transcribe(
            seg, lang=lang, prompt=prompt, final=final, speaker=speaker
        )

    async def close(self) -> None:
        await self.primary.close()
        await self.fallback.close()
