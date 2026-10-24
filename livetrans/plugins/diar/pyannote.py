"""pyannote 화자 분할 어댑터 (명세 §4.8, D-021).

파이프라인 `pyannote/speaker-diarization-community-1` (CC BY 4.0 — THIRD_PARTY_NOTICES.md) 을 발화(SpeechSegment) 단위로
돌린다: 분할 모델 → 화자 임베딩 → VBx 클러스터링. 호출마다 라벨(SPEAKER_00 …)이 새로 매겨지므로 세션 화자 ID("S0" …)로
바꿔야 한다. 두 단계로 한다 (2026-10-23 합성음 측정 근거):

1. **문맥 창 + 연속성**: 발화 하나(≤ 8 s)만 넣으면 오프라인 클러스터링이 순차 2화자를 한 화자로 합친다 (ko+en 8 s 클립은
   `num_speakers=2` 를 줘도 1화자, 같은 음성을 이어 붙인 20~28 s 창은 2화자). 그래서 **직전 발화들을 `context_s` 만큼 앞에 이어
   붙여 함께 분할**한다. 창 안에서 라벨은 일관되므로, 이전 발화 구간(세션 ID 를 이미 아는 구간)과 **겹친 시간이 가장 긴 라벨 ↔ 세션 화자**
   로 1:1 매칭하면 ID 가 이어진다. 임베딩에 의존하지 않아 같은 화자의 임베딩이 창 구성에 따라 흔들려도(단독 ↔ 창: 0.56) 끊기지 않는다.
2. **임베딩 등록부** (`SpeakerRegistry`): 문맥에 없던 라벨(새로 등장한 화자, 또는 `context_s` 보다 오래 침묵했던 화자)만 임베딩으로
   기존 화자와 비교해 붙이거나 새 ID 를 준다.

- 모델은 Hugging Face 이용 조건 동의 + 토큰이 있어야 받는다 (`scripts/download_models.py --diar`). 서버 실행은 캐시만 쓴다.
- GPU 작업은 `GPU_LOCK` 아래 스레드에서 (§2.3).
- 화자 임베딩·문맥 파형은 세션 객체(`PyannoteDiarizer`) 메모리에만 있고 디스크에 쓰지 않는다 (§13).
- `torchcodec` 이 없어도 된다 — 오디오를 메모리 파형으로 넘긴다 (pyannote 가 import 시 경고만 낸다).
"""

from __future__ import annotations

import asyncio
import logging
import time
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

from livetrans.core.config import DiarConfig
from livetrans.core.gpu import GPU_LOCK
from livetrans.core.speaker import SpeakerTurn
from livetrans.pipeline.segmenter import SpeechSegment
from livetrans.plugins.diar.registry import SpeakerRegistry

log = logging.getLogger("livetrans.diar.pyannote")

SAMPLE_RATE = 16000
CONTEXT_GAP_S = 0.3  # 문맥 창에서 발화 사이에 끼우는 무음
MIN_CONTINUITY_S = 0.3  # 라벨 ↔ 세션 화자를 잇는 데 필요한 최소 겹침


@dataclass
class _Past:
    """문맥 창에 넣는 지난 발화: 파형과, 그 안에서 이미 배정한 (시작 샘플, 끝 샘플, 세션 화자)."""

    pcm: np.ndarray
    turns: list[tuple[int, int, str]]


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
            # 배치를 줄여 창당 피크 VRAM 과 캐시 예약을 낮춘다 (DiarConfig.batch_size 측정 주석 참고)
            for attr in ("segmentation_batch_size", "embedding_batch_size"):
                if hasattr(pipe, attr):
                    setattr(pipe, attr, self.cfg.batch_size)
            plda = getattr(pipe, "_plda", None)
            self._transform = getattr(plda, "_xvec_tf", None)
            # 첫 호출이 느리므로(약 1 s) 잡음 1 s 로 워밍업
            with GPU_LOCK, torch.inference_mode():
                noise = torch.randn(1, SAMPLE_RATE) * 0.01
                pipe({"waveform": noise, "sample_rate": SAMPLE_RATE})
        self._pipe = pipe
        log.info(
            "pyannote 로드 %s (%s) %.1f s, 임베딩 변환 %s, 문맥 창 %.0f s",
            self.model_id,
            self.cfg.device,
            time.perf_counter() - t0,
            "LDA" if self._transform else "없음",
            self.cfg.context_s,
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

    def create_stream(self):
        """스트리밍 화자 분할기 (D-028). 같은 파이프라인을 연속 오디오 창에 쓴다."""
        from livetrans.plugins.diar.stream import StreamDiarizer

        if self._pipe is None:
            raise RuntimeError("화자 분할 모델이 로드되지 않았습니다")
        registry = SpeakerRegistry(
            threshold=self.cfg.match_threshold,
            max_speakers=self.cfg.max_speakers,
            transform=self._transform,
        )
        return StreamDiarizer(self._pipe, self.cfg, registry)

    async def close(self) -> None:
        self._pipe = None
        self._transform = None


class PyannoteDiarizer:
    """세션 하나의 화자 분할기. 세션이 끝나면 `reset()` 으로 임베딩·문맥을 버린다."""

    def __init__(self, pipe, cfg: DiarConfig, registry: SpeakerRegistry) -> None:
        self._pipe = pipe
        self._cfg = cfg
        self.registry = registry
        self.last_latency_ms = 0
        self.last_window_s = 0.0  # 마지막 호출에 넣은 파형 길이 (문맥 포함)
        self._history: deque[_Past] = deque()
        self._history_samples = 0
        self._gap = np.zeros(int(CONTEXT_GAP_S * SAMPLE_RATE), dtype=np.float32)

    async def diarize(self, seg: SpeechSegment) -> list[SpeakerTurn]:
        return await asyncio.to_thread(self._diarize_blocking, seg)

    def reset(self) -> None:
        self.registry.reset()
        self._history.clear()
        self._history_samples = 0

    # --- 문맥 창 ---

    def _window(self, cur: np.ndarray) -> tuple[np.ndarray, int, list[tuple[float, float, str]]]:
        """(문맥 발화들 + 무음 + 현재 발화, 현재 발화의 샘플 오프셋, 문맥 안의 (시작 s, 끝 s, 세션 화자))."""
        if not self._history:
            return cur, 0, []
        parts: list[np.ndarray] = []
        known: list[tuple[float, float, str]] = []
        pos = 0
        for past in self._history:
            for a, b, sid in past.turns:
                known.append(((pos + a) / SAMPLE_RATE, (pos + b) / SAMPLE_RATE, sid))
            parts += [past.pcm, self._gap]
            pos += len(past.pcm) + len(self._gap)
        parts.append(cur)
        return np.concatenate(parts), pos, known

    def _remember(self, cur: np.ndarray, turns: list[tuple[int, int, str]]) -> None:
        budget = int(self._cfg.context_s * SAMPLE_RATE)
        if budget <= 0:
            return
        self._history.append(_Past(cur, turns))
        self._history_samples += len(cur)
        while self._history_samples > budget and len(self._history) > 1:
            self._history_samples -= len(self._history.popleft().pcm)

    def _relabel_history(self, ann, label_to_id: dict[str, str]) -> None:
        """이번 창의 결과로 문맥 발화들의 화자 구간을 다시 쓴다.

        창이 길수록 클러스터링이 정확하므로 최신 결과를 믿는다 — 문맥이 짧았을 때(세션 초반) 합쳐졌던 화자가 나중에 갈리면
        지난 구간의 ID 도 따라 고쳐져 다음 연속성 매칭이 어긋나지 않는다. 이미 보낸 `segment.final` 은 바꾸지 않는다.
        """
        pos = 0
        for past in self._history:
            a0, b0 = pos / SAMPLE_RATE, (pos + len(past.pcm)) / SAMPLE_RATE
            new: list[tuple[int, int, str]] = []
            for segment, _, label in ann.itertracks(yield_label=True):
                sid = label_to_id.get(label)
                if sid is None:
                    continue
                a, b = max(segment.start, a0), min(segment.end, b0)
                if b > a:
                    new.append((round((a - a0) * SAMPLE_RATE), round((b - a0) * SAMPLE_RATE), sid))
            if new:
                past.turns = new
            pos += len(past.pcm) + len(self._gap)

    # --- 분할 ---

    def _diarize_blocking(self, seg: SpeechSegment) -> list[SpeakerTurn]:
        import torch

        t0 = time.perf_counter()
        cur = np.ascontiguousarray(seg.pcm, dtype=np.float32)
        window, offset, known = self._window(cur)
        self.last_window_s = len(window) / SAMPLE_RATE
        wav = torch.from_numpy(window)[None]
        with GPU_LOCK, torch.inference_mode(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = self._pipe(
                {"waveform": wav, "sample_rate": SAMPLE_RATE},
                max_speakers=self._cfg.max_speakers,
            )
        ann = out.speaker_diarization
        labels = list(ann.labels())
        self.last_latency_ms = int((time.perf_counter() - t0) * 1000)

        # 라벨별 트랙: 현재 발화 구간은 t_cap(ms) 로, 문맥 구간은 창 초 단위로 (연속성 매칭용)
        cur_start_s = offset / SAMPLE_RATE
        cur_end_s = (offset + len(cur)) / SAMPLE_RATE
        tracks: dict[str, list[tuple[int, int]]] = defaultdict(list)
        past_tracks: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for segment, _, label in ann.itertracks(yield_label=True):
            if segment.start < cur_start_s:
                past_tracks[label].append((segment.start, min(segment.end, cur_start_s)))
            a_s, b_s = max(segment.start, cur_start_s), min(segment.end, cur_end_s)
            if b_s <= a_s:
                continue
            a = seg.t_start_ms + round((a_s - cur_start_s) * 1000)
            b = seg.t_start_ms + round((b_s - cur_start_s) * 1000)
            a, b = max(seg.t_start_ms, a), min(seg.t_end_ms, b)
            if b > a:
                tracks[label].append((a, b))
        present = [lb for lb in labels if tracks[lb]]  # 문맥에만 있던 화자는 제외
        if not present:
            self._remember(cur, [])
            return []
        spoken = {lb: sum(b - a for a, b in tracks[lb]) for lb in present}

        # 너무 짧게 말한 라벨은 버린다 (단, 전부 짧으면 가장 긴 하나는 남긴다)
        kept = [lb for lb in present if spoken[lb] >= self._cfg.min_speaker_ms]
        if not kept:
            kept = [max(present, key=lambda lb: spoken[lb])]

        # 1) 문맥 연속성: 이전 발화에서 이미 배정한 구간과 겹친 시간이 긴 (라벨, 세션 화자) 쌍부터 1:1.
        #    문맥에만 있는 라벨도 매칭해 둔다 — 그 세션 화자는 "이번 창에서 다른 사람" 이므로 새 라벨이 임베딩으로 가져가면 안 된다
        label_to_id: dict[str, str] = {}
        if known:
            scores: list[tuple[float, str, str]] = []
            for lb in labels:
                by_sid: dict[str, float] = defaultdict(float)
                for a, b in past_tracks.get(lb, ()):
                    for a2, b2, sid in known:
                        by_sid[sid] += max(0.0, min(b, b2) - max(a, a2))
                scores += [(ov, lb, sid) for sid, ov in by_sid.items() if ov >= MIN_CONTINUITY_S]
            scores.sort(reverse=True)
            for _, lb, sid in scores:
                if lb in label_to_id or sid in label_to_id.values():
                    continue
                label_to_id[lb] = sid

        # 2) 나머지 라벨은 임베딩으로 (등록부가 1:1 + 새 화자 처리). 임베딩은 창 전체 기준, labels 순서
        emb = out.speaker_embeddings
        if emb is None or emb.shape[0] < len(labels):
            emb = np.full((len(labels), 1), np.nan, dtype=np.float32)
        rest = [lb for lb in kept if lb not in label_to_id]
        if rest:
            idx = [labels.index(lb) for lb in rest]
            ids = self.registry.assign(
                emb[idx], [spoken[lb] / 1000.0 for lb in rest], exclude=label_to_id.values()
            )
            label_to_id.update(zip(rest, ids, strict=True))
        for lb in kept:
            if lb not in rest:
                self.registry.reinforce(label_to_id[lb], emb[labels.index(lb)], spoken[lb] / 1000.0)

        turns: list[SpeakerTurn] = []
        for lb in kept:
            for a, b in tracks[lb]:
                overlap = any(
                    a < b2 and a2 < b for other in kept if other != lb for a2, b2 in tracks[other]
                )
                turns.append(SpeakerTurn(label_to_id[lb], a, b, overlap=overlap))
        turns.sort(key=lambda t: (t.t_start_ms, t.speaker))
        self._relabel_history(ann, label_to_id)
        self._remember(
            cur,
            [
                (
                    round((a - seg.t_start_ms) * SAMPLE_RATE / 1000),
                    round((b - seg.t_start_ms) * SAMPLE_RATE / 1000),
                    label_to_id[lb],
                )
                for lb in kept
                for a, b in tracks[lb]
            ],
        )
        return turns
