"""DiCoW STT 어댑터 — Diarization-Conditioned Whisper (명세 §4.3, D-019).

가중치 `BUT-FIT/DiCoW_v3_3` (CC BY 4.0, 출처 표기 — THIRD_PARTY_NOTICES.md) 를 Hugging Face transformers 의
`trust_remote_code` 로 로드한다. Whisper large-v3-turbo 기반이라 한·영·일을 포함한 다국어를 인식하고,
발화 구간에 화자 마스크(STNO, `stno.py`)를 함께 넣으면 **대상 화자의 말만** 받아 적는다.

- 화자 구간이 없는 세그먼트(화자 분할 미사용)는 전체를 target 으로 → 일반 Whisper 와 같은 조건.
- 모델은 블로킹이라 스레드에서 돌리고, GPU 작업은 프로세스 전역 `GPU_LOCK` 아래에서만 실행한다 (§2.3).
- 출력은 DiCoW 학습 정규화 때문에 **소문자·구두점 없음** (2026-10-23 실측). 단어 타임스탬프는 아직 없다 (`words=[]`).
- 언어 자동 감지는 Whisper 방식(첫 디코더 스텝의 언어 토큰 확률)으로 직접 구한다.
"""

from __future__ import annotations

import asyncio
import logging
import time

import numpy as np

from livetrans.core.config import SttConfig
from livetrans.core.gpu import GPU_LOCK
from livetrans.pipeline.segmenter import SpeechSegment
from livetrans.plugins.stt.base import Transcript
from livetrans.plugins.stt.guard import max_new_tokens_for
from livetrans.plugins.stt.stno import N_FRAMES_30S, build_stno_mask

log = logging.getLogger("livetrans.stt.dicow")

SAMPLE_RATE = 16000
MAX_INPUT_S = 30.0
PROMPT_MAX_CHARS = 200


def _lower_uppercase_mapping(tokenizer) -> None:
    """DiCoW 가 대문자 토큰을 소문자 토큰으로 묶어 디코딩할 때 쓰는 표 (BUTSpeechFIT/DiCoW inference.py 와 동일)."""
    tokenizer.upper_cased_tokens = {}
    vocab = tokenizer.get_vocab()
    for token, index in vocab.items():
        if not token:
            continue
        if token[0] == "Ġ" and len(token) > 1:
            lower = token[0] + token[1].lower() + (token[2:] if len(token) > 2 else "")
        else:
            lower = token[0].lower() + token[1:]
        if lower != token:
            lower_index = vocab.get(lower)
            if lower_index is not None:
                tokenizer.upper_cased_tokens[lower_index] = index


class DiCoWSTT:
    name = "dicow"
    languages = ("ko", "en", "ja")
    supports_speaker_mask = True

    def __init__(self, cfg: SttConfig) -> None:
        self.cfg = cfg
        self.model_id = cfg.model
        self._model = None
        self._fe = None
        self._tok = None
        self._lang_ids: dict[str, int] = {}  # "ko" → 토큰 id
        self._id_lang: dict[int, str] = {}
        self._device = None
        self._dtype = None

    # --- 수명 주기 ---

    async def load(self) -> None:
        if self._model is None:
            await asyncio.to_thread(self._load_blocking)

    def _load_blocking(self) -> None:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoTokenizer, WhisperFeatureExtractor

        t0 = time.perf_counter()
        self._device = torch.device(self.cfg.device)
        self._dtype = torch.float16 if "float16" in self.cfg.compute_type else torch.float32
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id, trust_remote_code=True, torch_dtype=self._dtype
        )
        model = model.to(self._device).eval()
        tok = AutoTokenizer.from_pretrained(self.model_id)
        _lower_uppercase_mapping(tok)
        model.set_tokenizer(tok)
        self._fe = WhisperFeatureExtractor.from_pretrained(self.model_id)
        self._tok = tok
        self._model = model
        lang_to_id = model.generation_config.lang_to_id or {}
        self._lang_ids = {k.strip("<|>"): v for k, v in lang_to_id.items()}
        self._id_lang = {v: k for k, v in self._lang_ids.items()}
        log.info(
            "DiCoW 로드 %s (%s, %s) %.1f s",
            self.model_id,
            self._device,
            self._dtype,
            time.perf_counter() - t0,
        )

    async def close(self) -> None:
        self._model = None
        self._fe = None
        self._tok = None

    # --- 인식 ---

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
        return await asyncio.to_thread(self._transcribe_blocking, seg, lang, prompt, final, speaker)

    def _transcribe_blocking(
        self,
        seg: SpeechSegment,
        lang: str | None,
        prompt: str | None,
        final: bool,
        speaker: str | None,
    ) -> Transcript:
        import torch

        t0 = time.perf_counter()
        pcm = np.asarray(seg.pcm, dtype=np.float32)
        max_samples = int(MAX_INPUT_S * SAMPLE_RATE)
        if len(pcm) > max_samples:  # 세그먼터가 8 s 로 자르므로 보통 안 온다
            pcm = pcm[:max_samples]
        duration_ms = len(pcm) * 1000 // SAMPLE_RATE
        stno_np = build_stno_mask(
            duration_ms,
            turns=seg.speaker_turns,
            speakers=seg.speakers,
            target=speaker,
            seg_t_start_ms=seg.t_start_ms,
            n_frames=N_FRAMES_30S,
        )
        feats = self._fe(pcm, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        x = feats.input_features.to(self._device, self._dtype)  # [1, 128, 3000]
        stno = torch.from_numpy(stno_np)[None].to(self._device, self._dtype)  # [1, 4, 1500]
        attn = torch.ones(1, x.shape[-1], dtype=torch.bool, device=self._device)
        prompt_ids = None
        if prompt:
            prompt_ids = self._tok.get_prompt_ids(
                prompt[-PROMPT_MAX_CHARS:], return_tensors="pt"
            ).to(self._device)

        with GPU_LOCK, torch.inference_mode():
            lang_prob = 1.0
            if lang is None:
                lang, lang_prob = self._detect_language(x, stno)
            gen_kwargs = {
                "input_features": x,
                "attention_mask": attn,
                "stno_mask": stno,
                "language": lang,
                "task": "transcribe",
                "num_beams": self.cfg.beam_final if final else self.cfg.beam_partial,
                "return_timestamps": False,
                # 반복 환각이 토큰 상한(448)까지 달리지 않게 오디오 길이에 비례해 자른다 (guard.py)
                "max_new_tokens": max_new_tokens_for(duration_ms),
            }
            if prompt_ids is not None:
                gen_kwargs["prompt_ids"] = prompt_ids
            out = self._model.generate(**gen_kwargs)
        seqs = out["sequences"] if isinstance(out, dict) else out
        text = self._tok.batch_decode(seqs, skip_special_tokens=True)[0].strip()
        if prompt and text.startswith(prompt[-PROMPT_MAX_CHARS:].strip()):
            # 일부 경로에서 프롬프트가 출력 앞에 그대로 붙어 나오면 떼어낸다
            text = text[len(prompt[-PROMPT_MAX_CHARS:].strip()) :].strip()
        return Transcript(
            seg_id=seg.seg_id,
            text=text,
            lang=lang,
            lang_prob=float(lang_prob),
            words=[],
            avg_logprob=0.0,
            latency_ms=int((time.perf_counter() - t0) * 1000),
            speaker=speaker,
        )

    def _detect_language(self, x, stno) -> tuple[str, float]:
        """첫 디코더 스텝에서 언어 토큰들만 남겨 softmax → (코드, 확률). 호출자가 GPU_LOCK 을 잡고 있어야 한다."""
        import torch

        model = self._model
        gen_cfg = model.generation_config
        dec = torch.full(
            (x.shape[0], 1), gen_cfg.decoder_start_token_id, dtype=torch.long, device=self._device
        )
        logits = model(
            input_features=x, decoder_input_ids=dec, use_cache=False, stno_mask=stno
        ).logits[:, -1]
        ids = list(self._id_lang.keys())
        probs = logits[0, ids].float().softmax(-1)
        best = int(probs.argmax())
        return self._id_lang[ids[best]], float(probs[best])
