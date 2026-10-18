"""VAD 기반 발화 분할기 (명세 §4.2, 강제 분할 §3.6).

오디오를 512 샘플(32 ms) 창으로 나눠 음성 확률을 받고, 상태 기계로 발화를 자른다.
모든 시각은 세션 시작부터 받은 샘플 수로 계산한다 (I2, `t_cap`).

- 시작: 확률 ≥ threshold 인 창이 나오면, 직전 `speech_pad_ms` 만큼을 앞에 붙여 발화를 연다.
- 끝: 마지막 음성 창 뒤로 무음이 `min_silence_ms` 이어지면, 마지막 음성 + `speech_pad_ms` 에서 자른다.
  음성 길이가 `min_speech_ms` 미만이면 버린다 (잡음·짧은 클릭).
- 강제 분할: 발화가 `max_utterance_s` 에 닿으면 최근 1.5 s 안에서 에너지가 가장 낮은 창의 가운데에서 자르고,
  나머지는 `prev_seg_id` 로 이어지는 새 발화로 계속한다.
- 중간 결과: `partial_interval_s` 가 주어지면 진행 중 발화를 그 간격마다 `PartialRequest` 로 내보낸다 (모드 B·C).

VAD 는 `Callable[[np.ndarray], float]` 로 주입한다. 실제로는 `StreamingSileroVAD`, 테스트에서는 에너지 기반 가짜.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from livetrans.core.config import VadConfig

WINDOW_SAMPLES = 512
FORCED_CUT_SEARCH_S = 1.5


@dataclass
class SpeechSegment:
    seg_id: int
    t_start_ms: int
    t_end_ms: int
    pcm: np.ndarray  # float32 mono
    forced_cut: bool = False
    prev_seg_id: int | None = None


@dataclass
class PartialRequest:
    seg_id: int
    t_start_ms: int
    pcm: np.ndarray


SegmenterEvent = SpeechSegment | PartialRequest


class Segmenter:
    def __init__(
        self,
        cfg: VadConfig,
        vad: Callable[[np.ndarray], float],
        *,
        sample_rate: int = 16000,
        partial_interval_s: float | None = None,
    ) -> None:
        self._vad = vad
        self._sr = sample_rate
        self._threshold = cfg.threshold
        self._pad = self._ms(cfg.speech_pad_ms)
        self._min_silence = self._ms(cfg.min_silence_ms)
        self._min_speech = self._ms(cfg.min_speech_ms)
        self._max_len = int(cfg.max_utterance_s * sample_rate)
        self._search = int(FORCED_CUT_SEARCH_S * sample_rate)
        self._partial = int(partial_interval_s * sample_rate) if partial_interval_s else None

        self._pending = np.zeros(0, dtype=np.float32)
        self._pos = 0  # 창으로 처리한 누적 샘플 수
        self._next_id = 0

        # 발화 밖: 앞 패딩용으로 최근 오디오를 보관
        self._history = np.zeros(0, dtype=np.float32)

        # 발화 안
        self._active = False
        self._buf: list[np.ndarray] = []
        self._buf_len = 0
        self._seg_start = 0
        self._last_speech_end = 0
        self._speech = 0
        self._prev_id: int | None = None
        self._forced_head = False
        self._last_partial = 0

    def _ms(self, ms: int) -> int:
        return ms * self._sr // 1000

    def _to_ms(self, samples: int) -> int:
        return samples * 1000 // self._sr

    @property
    def active(self) -> bool:
        return self._active

    def push(self, pcm: np.ndarray) -> list[SegmenterEvent]:
        data = np.concatenate([self._pending, pcm.astype(np.float32, copy=False)])
        n = len(data) // WINDOW_SAMPLES * WINDOW_SAMPLES
        self._pending = data[n:].copy()
        events: list[SegmenterEvent] = []
        for i in range(0, n, WINDOW_SAMPLES):
            events.extend(self._step(data[i : i + WINDOW_SAMPLES]))
        return events

    def flush(self) -> list[SpeechSegment]:
        """입력 끝: 진행 중 발화를 닫는다. 32 ms 미만 잔여 샘플은 버린다."""
        self._pending = np.zeros(0, dtype=np.float32)
        if not self._active:
            return []
        seg = self._close(min(self._last_speech_end + self._pad, self._pos))
        return [seg] if seg else []

    def _step(self, window: np.ndarray) -> list[SegmenterEvent]:
        start = self._pos
        end = start + WINDOW_SAMPLES
        self._pos = end
        is_speech = self._vad(window) >= self._threshold

        if not self._active:
            if is_speech:
                self._open(start, window)
            else:
                self._history = (
                    np.concatenate([self._history, window])[-self._pad :]
                    if self._pad
                    else self._history
                )
            return []

        self._append(window)
        if is_speech:
            self._last_speech_end = end
            self._speech += WINDOW_SAMPLES

        events: list[SegmenterEvent] = []
        if end - self._last_speech_end >= self._min_silence:
            seg = self._close(min(self._last_speech_end + self._pad, end))
            if seg:
                events.append(seg)
            return events

        if end - self._seg_start >= self._max_len:
            events.append(self._forced_cut())

        if self._partial and self._active and end - self._last_partial >= self._partial:
            self._last_partial = end
            events.append(
                PartialRequest(self._next_id, self._to_ms(self._seg_start), self._concat())
            )
        return events

    def _open(self, start: int, window: np.ndarray) -> None:
        pre = self._history
        self._history = np.zeros(0, dtype=np.float32)
        self._active = True
        self._seg_start = start - len(pre)
        self._buf = [pre, window] if len(pre) else [window]
        self._buf_len = len(pre) + len(window)
        self._last_speech_end = start + len(window)
        self._speech = len(window)
        self._last_partial = self._seg_start
        self._forced_head = False
        self._prev_id = None

    def _append(self, window: np.ndarray) -> None:
        self._buf.append(window)
        self._buf_len += len(window)

    def _concat(self) -> np.ndarray:
        return np.concatenate(self._buf) if self._buf else np.zeros(0, dtype=np.float32)

    def _close(self, cut: int) -> SpeechSegment | None:
        """[seg_start, cut) 를 발화로 내보내고, cut 이후 오디오는 앞 패딩 후보로 돌린다."""
        audio = self._concat()
        rel = cut - self._seg_start
        head, tail = audio[:rel], audio[rel:]
        keep = self._speech >= self._min_speech or self._forced_head
        seg = None
        if keep:
            seg = SpeechSegment(
                seg_id=self._next_id,
                t_start_ms=self._to_ms(self._seg_start),
                t_end_ms=self._to_ms(cut),
                pcm=head,
                forced_cut=False,
                prev_seg_id=self._prev_id,
            )
            self._next_id += 1
        # 버린 발화의 id 는 다음 발화가 재사용한다 (중간 결과가 같은 id 로 덮어써지도록)
        self._active = False
        self._buf, self._buf_len = [], 0
        self._history = tail[-self._pad :] if self._pad else np.zeros(0, dtype=np.float32)
        return seg

    def _forced_cut(self) -> SpeechSegment:
        audio = self._concat()
        end = self._seg_start + len(audio)
        lo = max(self._seg_start + WINDOW_SAMPLES, end - self._search)
        best, best_energy = end, float("inf")
        for w_start in range(end - WINDOW_SAMPLES, lo - 1, -WINDOW_SAMPLES):
            rel = w_start - self._seg_start
            energy = float(np.mean(audio[rel : rel + WINDOW_SAMPLES] ** 2))
            if energy < best_energy:
                best_energy, best = energy, w_start + WINDOW_SAMPLES // 2
        rel = best - self._seg_start
        seg = SpeechSegment(
            seg_id=self._next_id,
            t_start_ms=self._to_ms(self._seg_start),
            t_end_ms=self._to_ms(best),
            pcm=audio[:rel],
            forced_cut=True,
            prev_seg_id=self._prev_id,
        )
        self._next_id += 1
        rest = audio[rel:]
        self._prev_id = seg.seg_id
        self._seg_start = best
        self._buf, self._buf_len = [rest], len(rest)
        self._last_speech_end = max(self._last_speech_end, best)
        self._speech = len(rest)
        self._last_partial = best
        self._forced_head = True  # 이어지는 조각은 짧아도 버리지 않는다 (문장 끝이 잘려나가지 않게)
        return seg
