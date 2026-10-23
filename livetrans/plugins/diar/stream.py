"""스트리밍 화자 분할 — 연속 오디오에 긴 창을 주기적으로 돌려 화자 활동 구간을 확정한다 (D-028, 명세 §4.8).

발화 단위(`pyannote.py`)는 실방송에서 화자를 뭉쳤다 (덤프 25개 중 18개가 한 화자). 오프라인으로 전체를 돌리면 가르므로,
**연속 오디오 링버퍼에 `window_s`(120 s) 창을 `hop_s`(3 s) 마다** 돌리고 **`right_context_s`(2 s) 앞까지만 확정**한다.
시뮬레이션(실방송 덤프): 120 s/3 s/2 s 가 오프라인 전체 분할 대비 98.2 % 일치, 창당 ≈ 0.6 s, GPU ≈ 20 %. 20~60 s 창은 69~71 %.

세션 화자 ID 잇기:
1. 창 라벨 ↔ 세션 화자: 이미 확정한 구간과 겹친 시간이 긴 쌍부터 1:1 (연속성). 창이 직전 창과 거의 전부 겹치므로 안정적이다.
2. 겹치는 구간이 없는 라벨(새 화자, 또는 창보다 오래 침묵했던 화자)만 임베딩 등록부(`SpeakerRegistry`)로.
3. 창 결과로 창 범위 안의 과거 확정 구간을 다시 쓴다 (창이 길수록 정확). 이미 내보낸 자막은 바꾸지 않는다.

이 객체는 세션마다 하나이며 오디오·임베딩은 메모리에만 있다 (§13). GPU 작업은 `GPU_LOCK` 아래 스레드에서.
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
from livetrans.plugins.diar.registry import SpeakerRegistry

log = logging.getLogger("livetrans.diar.stream")

SAMPLE_RATE = 16000
MIN_CONTINUITY_S = 0.3  # 라벨 ↔ 세션 화자를 잇는 최소 겹침
MIN_TURN_MS = 40  # 이보다 짧은 조각은 버림 (창 경계 잘림 등)

Interval = tuple[int, int]  # (시작 ms, 끝 ms), t_cap 기준


def _merge(ivs: list[Interval]) -> list[Interval]:
    out: list[Interval] = []
    for a, b in sorted(ivs):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _overlap_ms(x: list[Interval], y: list[Interval]) -> int:
    return sum(max(0, min(b, d) - max(a, c)) for a, b in x for c, d in y)


def _with_overlap(raw: list[tuple[str, int, int]]) -> list[SpeakerTurn]:
    """(화자, 시작, 끝) → 다른 화자와 겹치면 overlap=True 인 SpeakerTurn, 시간순."""
    turns = [
        SpeakerTurn(
            sid,
            a,
            b,
            overlap=any(o_sid != sid and oa < b and a < ob for o_sid, oa, ob in raw),
        )
        for sid, a, b in raw
    ]
    turns.sort(key=lambda t: (t.t_start_ms, t.speaker))
    return turns


class StreamDiarizer:
    """세션 하나의 스트리밍 화자 분할기. `push` 로 오디오를 넣고, `ready` 가 되면 `step` 으로 새로 확정된 구간을 받는다."""

    def __init__(self, pipe, cfg: DiarConfig, registry: SpeakerRegistry) -> None:
        self._pipe = pipe
        self._cfg = cfg
        self.registry = registry
        self._window = int(cfg.window_s * SAMPLE_RATE)
        self._hop = int(cfg.hop_s * SAMPLE_RATE)
        self._right = int(cfg.right_context_s * SAMPLE_RATE)
        cap = self._window + self._hop + self._right + SAMPLE_RATE
        self._buf = np.zeros(cap, dtype=np.float32)
        self._buf_len = 0  # 버퍼에 들어 있는 유효 샘플 수 (끝 정렬)
        self._total = 0  # 지금까지 받은 샘플 수 (= t_cap)
        self._committed = 0  # 활동을 확정한 샘플 경계
        self._last_step_total = 0
        self._activity: dict[str, list[Interval]] = defaultdict(list)  # 세션 화자 → 확정 구간
        self.last_latency_ms = 0
        self.last_window_s = 0.0

    # --- 입력 ---

    def push(self, pcm: np.ndarray) -> None:
        pcm = np.asarray(pcm, dtype=np.float32)
        n = len(pcm)
        if n == 0:
            return
        cap = len(self._buf)
        if n >= cap:
            self._buf[:] = pcm[-cap:]
            self._buf_len = cap
        else:
            keep = min(self._buf_len, cap - n)
            if keep:
                self._buf[cap - n - keep : cap - n] = self._buf[cap - keep :]
            self._buf[cap - n :] = pcm
            self._buf_len = keep + n
        self._total += n

    @property
    def total_ms(self) -> int:
        return self._total * 1000 // SAMPLE_RATE

    @property
    def committed_ms(self) -> int:
        return self._committed * 1000 // SAMPLE_RATE

    @property
    def ready(self) -> bool:
        return self._total - self._last_step_total >= self._hop

    def audio(self, t_start_ms: int, t_end_ms: int) -> np.ndarray:
        """[t_start, t_end) 구간의 파형. 버퍼 밖(너무 오래된) 부분은 0 으로 채운다."""
        a = t_start_ms * SAMPLE_RATE // 1000
        b = t_end_ms * SAMPLE_RATE // 1000
        out = np.zeros(max(0, b - a), dtype=np.float32)
        buf_start = self._total - self._buf_len
        lo, hi = max(a, buf_start), min(b, self._total)
        if hi > lo:
            off = len(self._buf) - self._buf_len
            out[lo - a : hi - a] = self._buf[off + (lo - buf_start) : off + (hi - buf_start)]
        return out

    def committed_turns(self, t_start_ms: int, t_end_ms: int) -> list[SpeakerTurn]:
        """[t_start, t_end] 에 걸친 모든 화자의 확정 구간 (STT 화자 마스크용)."""
        raw: list[tuple[str, int, int]] = []
        for sid, ivs in self._activity.items():
            for a, b in ivs:
                lo, hi = max(a, t_start_ms), min(b, t_end_ms)
                if hi > lo:
                    raw.append((sid, lo, hi))
        return _with_overlap(raw)

    def reset(self) -> None:
        self.registry.reset()
        self._activity.clear()
        self._buf_len = 0

    # --- 분할 ---

    async def step(self, *, final: bool = False) -> list[SpeakerTurn]:
        """새로 확정된 화자 구간(이전 확정 경계 ~ 새 경계)을 돌려준다. `final` 이면 오른쪽 문맥 없이 끝까지 확정."""
        return await asyncio.to_thread(self._step_blocking, final)

    def _step_blocking(self, final: bool) -> list[SpeakerTurn]:
        import torch

        t0 = time.perf_counter()
        self._last_step_total = self._total
        commit_to = self._total if final else max(self._committed, self._total - self._right)
        prev_commit = self._committed
        if commit_to <= prev_commit and not final:
            return []
        n = min(self._buf_len, self._window)
        if n < SAMPLE_RATE // 2:  # 0.5 s 미만이면 돌리지 않는다
            self._committed = commit_to
            return []
        w0 = self._total - n  # 창 시작 샘플
        off = len(self._buf) - n
        window = np.ascontiguousarray(self._buf[off:])
        self.last_window_s = n / SAMPLE_RATE
        with GPU_LOCK, torch.inference_mode(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = self._pipe(
                {"waveform": torch.from_numpy(window)[None], "sample_rate": SAMPLE_RATE},
                max_speakers=self._cfg.max_speakers,
            )
        ann = out.speaker_diarization
        labels = list(ann.labels())
        w0_ms = w0 * 1000 // SAMPLE_RATE
        prev_ms = prev_commit * 1000 // SAMPLE_RATE
        commit_ms = commit_to * 1000 // SAMPLE_RATE

        # 창 라벨별 구간 (절대 ms), 확정 경계까지만
        tracks: dict[str, list[Interval]] = defaultdict(list)
        for segment, _, label in ann.itertracks(yield_label=True):
            a = w0_ms + round(segment.start * 1000)
            b = min(commit_ms, w0_ms + round(segment.end * 1000))
            if b - a >= MIN_TURN_MS:
                tracks[label].append((a, b))
        tracks = {lb: _merge(ivs) for lb, ivs in tracks.items() if ivs}

        # 1) 연속성: 이미 확정한 구간 [w0, prev_commit) 과 겹친 시간 (ms)
        past = {
            sid: [
                (max(a, w0_ms), min(b, prev_ms)) for a, b in ivs if min(b, prev_ms) > max(a, w0_ms)
            ]
            for sid, ivs in self._activity.items()
        }
        scores = [
            (_overlap_ms(ivs, past[sid]), lb, sid)
            for lb, ivs in tracks.items()
            for sid in past
            if past[sid]
        ]
        scores = [s for s in scores if s[0] >= MIN_CONTINUITY_S * 1000]
        scores.sort(reverse=True)
        label_to_id: dict[str, str] = {}
        for _, lb, sid in scores:
            if lb in label_to_id or sid in label_to_id.values():
                continue
            label_to_id[lb] = sid

        # 2) 새 구간에서 말하는데 못 이은 라벨 → 임베딩 등록부 (새 화자 또는 오래전 화자)
        emb = out.speaker_embeddings
        if emb is None or emb.shape[0] < len(labels):
            emb = np.full((len(labels), 1), np.nan, dtype=np.float32)
        rest = [
            lb for lb in tracks if lb not in label_to_id and any(b > prev_ms for _, b in tracks[lb])
        ]
        if rest:
            idx = [labels.index(lb) for lb in rest]
            weights = [sum(b - a for a, b in tracks[lb]) / 1000.0 for lb in rest]
            ids = self.registry.assign(emb[idx], weights, exclude=label_to_id.values())
            label_to_id.update(zip(rest, ids, strict=True))
        for lb, sid in label_to_id.items():
            if lb not in rest:
                self.registry.reinforce(
                    sid, emb[labels.index(lb)], sum(b - a for a, b in tracks[lb]) / 1000.0
                )

        # 3) 창 범위의 확정 구간을 창 결과로 다시 쓴다 (이번 창에 있는 화자만)
        for sid in set(label_to_id.values()):
            self._activity[sid] = [
                iv for iv in self._activity[sid] if iv[1] <= w0_ms or iv[0] >= commit_ms
            ]
        for lb, sid in label_to_id.items():
            self._activity[sid] = _merge(self._activity[sid] + tracks[lb])
        # 오래된 구간은 버린다 (창 밖)
        for sid in list(self._activity):
            self._activity[sid] = [iv for iv in self._activity[sid] if iv[1] > w0_ms]
            if not self._activity[sid]:
                del self._activity[sid]

        # 새로 확정된 구간 [prev_commit, commit_to)
        raw: list[tuple[str, int, int]] = []
        for lb, sid in label_to_id.items():
            for a, b in tracks[lb]:
                lo, hi = max(a, prev_ms), min(b, commit_ms)
                if hi - lo >= MIN_TURN_MS:
                    raw.append((sid, lo, hi))
        self._committed = commit_to
        self.last_latency_ms = int((time.perf_counter() - t0) * 1000)
        return _with_overlap(raw)
