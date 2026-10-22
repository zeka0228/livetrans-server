"""pyannote 화자 분할 어댑터 (명세 §4.8, D-021).

파이프라인 `pyannote/speaker-diarization-community-1` (CC BY 4.0 — THIRD_PARTY_NOTICES.md) 을 발화(SpeechSegment) 단위로
돌린다: 분할 모델 → 화자 임베딩 → VBx 클러스터링. 발화 안 라벨(SPEAKER_00 …)은 발화마다 새로 매겨지므로
`SpeakerRegistry` 가 세션 화자 ID("S0" …)로 바꾼다 (증분 클러스터링).

- 모델은 Hugging Face 이용 조건 동의 + 토큰이 있어야 받는다 (`scripts/download_models.py --diar`). 서버 실행은 캐시만 쓴다.
- GPU 작업은 `GPU_LOCK` 아래 스레드에서 (§2.3). 2026-10-23 실측: 워밍업 후 발화당 25~105 ms, VRAM 약 0.6 GB.
- 화자 임베딩은 세션 객체(`PyannoteDiarizer`) 메모리에만 있고 디스크에 쓰지 않는다 (§13).
- `torchcodec` 이 없어도 된다 — 오디오를 메모리 파형으로 넘긴다 (pyannote 가 import 시 경고만 낸다).
"""

from __future__ import annotations

import asyncio
import logging
import time
import warnings
from collections import defaultdict

import numpy as np

from livetrans.core.config import DiarConfig
from livetrans.core.gpu import GPU_LOCK
from livetrans.core.speaker import SpeakerTurn
from livetrans.pipeline.segmenter import SpeechSegment
from livetrans.plugins.diar.registry import SpeakerRegistry

log = logging.getLogger("livetrans.diar.pyannote")

SAMPLE_RATE = 16000


class PyannoteDiarizerEngine:
    name = "pyannote"

    def __init__(self, cfg: DiarConfig) -> None:
        self.cfg = cfg
        self.model_id = cfg.model
        self._pipe = None
        self._transform = None  # 파이프라인의 x-vector 변환 (센터링·LDA·길이 정규화)

    async def load(self) -> None:
        if self._pipe is None:
            await asyncio.to_thread(self._load_blocking)

    def _load_blocking(self) -> None:
        import torch

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # torchcodec 미설치 경고 등
            from pyannote.audio import Pipeline

            t0 = time.perf_counter()
            pipe = Pipeline.from_pretrained(self.model_id)
            if pipe is None:
                raise RuntimeError(
                    f"화자 분할 모델을 열 수 없습니다: {self.model_id} "
                    "(HF 이용 조건 동의 + 토큰 후 scripts/download_models.py --diar)"
                )
            pipe.to(torch.device(self.cfg.device))
            plda = getattr(pipe, "_plda", None)
            self._transform = getattr(plda, "_xvec_tf", None)
            # 첫 호출이 느리므로(약 1 s) 잡음 1 s 로 워밍업
            with GPU_LOCK, torch.inference_mode():
                noise = torch.randn(1, SAMPLE_RATE) * 0.01
                pipe({"waveform": noise, "sample_rate": SAMPLE_RATE})
        self._pipe = pipe
        log.info(
            "pyannote 로드 %s (%s) %.1f s, 임베딩 변환 %s",
            self.model_id,
            self.cfg.device,
            time.perf_counter() - t0,
            "LDA" if self._transform else "없음",
        )

    def create(self) -> PyannoteDiarizer:
        if self._pipe is None:
            raise RuntimeError("화자 분할 모델이 로드되지 않았습니다")
        registry = SpeakerRegistry(
            threshold=self.cfg.match_threshold,
            max_speakers=self.cfg.max_speakers,
            transform=self._transform,
        )
        return PyannoteDiarizer(self._pipe, self.cfg, registry)

    async def close(self) -> None:
        self._pipe = None
        self._transform = None


class PyannoteDiarizer:
    """세션 하나의 화자 분할기. 세션이 끝나면 `reset()` 으로 임베딩을 버린다."""

    def __init__(self, pipe, cfg: DiarConfig, registry: SpeakerRegistry) -> None:
        self._pipe = pipe
        self._cfg = cfg
        self.registry = registry
        self.last_latency_ms = 0

    async def diarize(self, seg: SpeechSegment) -> list[SpeakerTurn]:
        return await asyncio.to_thread(self._diarize_blocking, seg)

    def reset(self) -> None:
        self.registry.reset()

    def _diarize_blocking(self, seg: SpeechSegment) -> list[SpeakerTurn]:
        import torch

        t0 = time.perf_counter()
        wav = torch.from_numpy(np.ascontiguousarray(seg.pcm, dtype=np.float32))[None]
        with GPU_LOCK, torch.inference_mode(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = self._pipe(
                {"waveform": wav, "sample_rate": SAMPLE_RATE},
                max_speakers=self._cfg.max_speakers,
            )
        ann = out.speaker_diarization
        labels = list(ann.labels())
        self.last_latency_ms = int((time.perf_counter() - t0) * 1000)
        if not labels:
            return []

        # 발화 안 라벨별 구간(ms, t_cap 기준)과 말한 시간
        tracks: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for segment, _, label in ann.itertracks(yield_label=True):
            a = seg.t_start_ms + round(segment.start * 1000)
            b = seg.t_start_ms + round(segment.end * 1000)
            a, b = max(seg.t_start_ms, a), min(seg.t_end_ms, b)
            if b > a:
                tracks[label].append((a, b))
        spoken = {lb: sum(b - a for a, b in tracks[lb]) for lb in labels}

        # 너무 짧게 말한 라벨은 버린다 (단, 전부 짧으면 가장 긴 하나는 남긴다)
        kept = [lb for lb in labels if spoken[lb] >= self._cfg.min_speaker_ms and tracks[lb]]
        if not kept:
            best = max(labels, key=lambda lb: spoken[lb])
            kept = [best] if tracks[best] else []
        if not kept:
            return []

        emb = out.speaker_embeddings
        if emb is None or emb.shape[0] < len(labels):
            emb = np.full((len(labels), 1), np.nan, dtype=np.float32)
        idx = [labels.index(lb) for lb in kept]
        ids = self.registry.assign(emb[idx], [spoken[lb] / 1000.0 for lb in kept])
        label_to_id = dict(zip(kept, ids, strict=True))

        turns: list[SpeakerTurn] = []
        for lb in kept:
            for a, b in tracks[lb]:
                overlap = any(
                    a < b2 and a2 < b for other in kept if other != lb for a2, b2 in tracks[other]
                )
                turns.append(SpeakerTurn(label_to_id[lb], a, b, overlap=overlap))
        turns.sort(key=lambda t: (t.t_start_ms, t.speaker))
        return turns
