"""화자 레인 세그먼터 (D-028): 화자별로 "말 시작 ~ 말 끝" 이 곧 발화 경계다.

스트리밍 Diarizer 가 확정한 화자 활동 구간을 화자마다 따로 받아, 전역 VAD 없이 발화를 자른다.
- 같은 화자의 구간 사이 무음이 `min_silence_ms` 이하면 같은 발화로 잇는다.
- 발화가 `max_utterance_s` 에 닿으면 강제 분할 (`forced_cut`, 이어지는 조각은 `prev_seg_id`).
- 확정 경계(`committed_ms`)에서 마지막 구간 뒤로 `min_silence_ms` 이상 지났으면 발화를 닫는다 — 즉 자막은
  화자가 말을 멈춘 뒤 `min_silence + right_context + hop(최대)` 뒤에 확정된다 (지연은 D 로 흡수, 사용자 결정).
- 말한 시간이 `min_speech_ms` 미만이면 버린다 (강제 분할 뒤 조각은 예외).

시각은 전부 t_cap(ms). 오디오는 들고 있지 않다 — 세션이 Diarizer 링버퍼에서 잘라 온다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from livetrans.core.config import VadConfig
from livetrans.core.speaker import SpeakerTurn

Interval = tuple[int, int]


@dataclass
class LaneUtterance:
    seg_id: int
    speaker: str
    t_start_ms: int
    t_end_ms: int
    turns: list[SpeakerTurn]  # 이 화자의 구간들
    forced_cut: bool = False
    prev_seg_id: int | None = None


@dataclass
class LanePartial:
    """진행 중 레인 발화의 중간 결과 요청."""

    seg_id: int
    speaker: str
    t_start_ms: int
    t_end_ms: int
    pcm: np.ndarray
    turns: list[SpeakerTurn] = field(default_factory=list)


class LaneSegmenter:
    def __init__(self, speaker: str, cfg: VadConfig, alloc_id) -> None:
        self.speaker = speaker
        self._min_silence = cfg.min_silence_ms
        self._min_speech = cfg.min_speech_ms
        self._pad = cfg.speech_pad_ms
        self._max_len = int(cfg.max_utterance_s * 1000)
        self._alloc_id = alloc_id  # 세션 공용 seg_id 발급기 (레인끼리 겹치지 않게)
        self._open: list[Interval] = []
        self._seg_id: int | None = None
        self._prev_id: int | None = None
        self._forced_head = False

    # --- 조회 ---

    @property
    def open_span(self) -> tuple[int, int, int] | None:
        """진행 중 발화 (seg_id, 시작 ms, 지금까지 끝 ms). 없으면 None."""
        if not self._open or self._seg_id is None:
            return None
        return self._seg_id, max(0, self._open[0][0] - self._pad), self._open[-1][1]

    @property
    def open_turns(self) -> list[SpeakerTurn]:
        return [SpeakerTurn(self.speaker, a, b) for a, b in self._open]

    # --- 입력 ---

    def push(self, intervals: list[Interval], committed_ms: int) -> list[LaneUtterance]:
        """이 화자의 새 확정 구간들과 확정 경계를 넣고, 닫힌 발화를 돌려준다."""
        out: list[LaneUtterance] = []
        for a, b in sorted(intervals):
            if b <= a:
                continue
            if self._open and a - self._open[-1][1] <= self._min_silence:
                if a <= self._open[-1][1]:
                    self._open[-1] = (self._open[-1][0], max(self._open[-1][1], b))
                else:
                    self._open.append((a, b))
            else:
                if self._open:
                    out.extend(self._close(forced=False))
                self._open = [(a, b)]
                self._seg_id = self._alloc_id()
                self._prev_id = None
                self._forced_head = False
            out.extend(self._maybe_force())
        if self._open and committed_ms - self._open[-1][1] >= self._min_silence:
            out.extend(self._close(forced=False))
        return out

    def flush(self) -> list[LaneUtterance]:
        return self._close(forced=False) if self._open else []

    # --- 내부 ---

    def _maybe_force(self) -> list[LaneUtterance]:
        out: list[LaneUtterance] = []
        while self._open and self._open[-1][1] - self._open[0][0] >= self._max_len:
            cut = self._open[0][0] + self._max_len
            head: list[Interval] = []
            tail: list[Interval] = []
            for a, b in self._open:
                if b <= cut:
                    head.append((a, b))
                elif a >= cut:
                    tail.append((a, b))
                else:
                    head.append((a, cut))
                    tail.append((cut, b))
            self._open = head
            out.extend(self._close(forced=True, next_open=tail))
        return out

    def _close(
        self, *, forced: bool, next_open: list[Interval] | None = None
    ) -> list[LaneUtterance]:
        ivs, seg_id = self._open, self._seg_id
        spoken = sum(b - a for a, b in ivs)
        keep = spoken >= self._min_speech or self._forced_head
        utts: list[LaneUtterance] = []
        if keep and ivs and seg_id is not None:
            utts.append(
                LaneUtterance(
                    seg_id=seg_id,
                    speaker=self.speaker,
                    t_start_ms=max(0, ivs[0][0] - self._pad),
                    t_end_ms=ivs[-1][1] + (0 if forced else self._pad),
                    turns=[SpeakerTurn(self.speaker, a, b) for a, b in ivs],
                    forced_cut=forced,
                    prev_seg_id=self._prev_id,
                )
            )
        if forced and next_open:
            self._open = next_open
            self._prev_id = seg_id if keep else self._prev_id
            self._seg_id = self._alloc_id()
            self._forced_head = True
        else:
            self._open, self._seg_id, self._prev_id, self._forced_head = [], None, None, False
        return utts
