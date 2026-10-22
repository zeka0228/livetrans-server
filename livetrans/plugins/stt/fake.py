"""테스트·앱 개발용 가짜 STT. GPU 없이 고정 지연으로 정해진 텍스트를 돌려준다."""

from __future__ import annotations

import asyncio

from livetrans.pipeline.segmenter import SpeechSegment
from livetrans.plugins.stt.base import Transcript, Word


class FakeSTT:
    name = "fake"
    model_id = "fake"
    languages = ("ko", "en", "ja")
    supports_speaker_mask = True

    def __init__(
        self, *, latency_s: float = 0.0, lang: str = "ko", lang_prob: float = 0.99
    ) -> None:
        self.latency_s = latency_s
        self.lang = lang
        self.lang_prob = lang_prob
        self.calls: list[dict] = []

    async def load(self) -> None:
        return None

    async def transcribe(
        self,
        seg: SpeechSegment,
        *,
        lang: str | None,
        prompt: str | None,
        final: bool,
        speaker: str | None = None,
    ) -> Transcript:
        self.calls.append(
            {
                "seg_id": seg.seg_id,
                "lang": lang,
                "prompt": prompt,
                "final": final,
                "speaker": speaker,
            }
        )
        if self.latency_s:
            await asyncio.sleep(self.latency_s)
        dur = seg.t_end_ms - seg.t_start_ms
        kind = "확정" if final else "중간"
        text = f"발화 {seg.seg_id} {kind} ({dur} ms)"
        words = [Word(text, seg.t_start_ms, seg.t_end_ms, 1.0)] if final else []
        return Transcript(
            seg_id=seg.seg_id,
            text=text,
            lang=lang or self.lang,
            lang_prob=1.0 if lang else self.lang_prob,
            words=words,
            latency_ms=int(self.latency_s * 1000),
            speaker=speaker,
        )

    async def close(self) -> None:
        return None
