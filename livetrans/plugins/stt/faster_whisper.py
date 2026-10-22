"""faster-whisper STT 어댑터 (명세 §4.3).

모델 호출은 블로킹이라 스레드에서 돌리고, GPU 작업은 프로세스 전체에서 락 하나로 직렬화한다 (§2.3).
세그먼터가 이미 발화를 잘랐으므로 faster-whisper 내장 VAD 필터는 끈다.
"""

from __future__ import annotations

import asyncio
import time

from livetrans.core.config import SttConfig
from livetrans.core.gpu import GPU_LOCK
from livetrans.pipeline.segmenter import SpeechSegment
from livetrans.plugins.stt.base import Transcript, Word

PROMPT_MAX_CHARS = 200


class FasterWhisperSTT:
    name = "faster_whisper"
    languages = ("ko", "en", "ja")
    supports_speaker_mask = False  # speaker 는 무시하고 구간 전체를 1회 인식한다 (§4.3)

    def __init__(self, cfg: SttConfig) -> None:
        self.cfg = cfg
        self.model_id = cfg.model
        self._model = None

    async def load(self) -> None:
        if self._model is None:
            self._model = await asyncio.to_thread(self._load_blocking)

    def _load_blocking(self):
        from faster_whisper import WhisperModel

        return WhisperModel(
            self.cfg.model, device=self.cfg.device, compute_type=self.cfg.compute_type
        )

    async def transcribe(
        self,
        seg: SpeechSegment,
        *,
        lang: str | None,
        prompt: str | None,
        final: bool,
        speaker: str | None = None,
    ) -> Transcript:
        if self._model is None:
            raise RuntimeError("STT 모델이 로드되지 않았습니다")
        tr = await asyncio.to_thread(self._transcribe_blocking, seg, lang, prompt, final)
        tr.speaker = speaker
        return tr

    def _transcribe_blocking(
        self, seg: SpeechSegment, lang: str | None, prompt: str | None, final: bool
    ) -> Transcript:
        t0 = time.perf_counter()
        with GPU_LOCK:
            segments, info = self._model.transcribe(
                seg.pcm,
                language=lang,
                beam_size=self.cfg.beam_final if final else self.cfg.beam_partial,
                word_timestamps=final,
                initial_prompt=prompt[-PROMPT_MAX_CHARS:] if prompt else None,
                condition_on_previous_text=False,
                vad_filter=False,
            )
            parts = list(segments)  # 제너레이터라 여기서 실제 디코딩이 일어난다
        words: list[Word] = []
        for p in parts:
            for w in p.words or ():
                words.append(
                    Word(
                        text=w.word,
                        t0_ms=seg.t_start_ms + int(w.start * 1000),
                        t1_ms=seg.t_start_ms + int(w.end * 1000),
                        prob=float(w.probability),
                    )
                )
        logprobs = [p.avg_logprob for p in parts]
        return Transcript(
            seg_id=seg.seg_id,
            text="".join(p.text for p in parts).strip(),
            lang=info.language,
            lang_prob=float(info.language_probability),
            words=words,
            avg_logprob=float(sum(logprobs) / len(logprobs)) if logprobs else 0.0,
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )

    async def close(self) -> None:
        self._model = None
