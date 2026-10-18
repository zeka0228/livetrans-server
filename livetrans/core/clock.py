"""캡처 타임라인 t_cap: 수신 오디오 샘플 수가 마스터 클럭이다 (명세 §3.1, 설계 D-010)."""

from __future__ import annotations


class SampleClock:
    """샘플 카운터 → t_cap(ms). 벽시계를 쓰지 않는다.

    `advance(n)` 은 n 샘플을 받았음을 기록하고, 그 샘플 묶음이 시작된 t_cap(ms) 를 돌려준다.
    """

    def __init__(self, sample_rate: int) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate 는 양수여야 합니다")
        self.sample_rate = sample_rate
        self._samples = 0

    @property
    def samples(self) -> int:
        return self._samples

    @property
    def now_ms(self) -> int:
        """지금까지 받은 오디오의 끝 시각 (ms, 내림)."""
        return self.samples_to_ms(self._samples)

    def advance(self, num_samples: int) -> int:
        if num_samples < 0:
            raise ValueError("num_samples 는 음수일 수 없습니다")
        start_ms = self.now_ms
        self._samples += num_samples
        return start_ms

    def samples_to_ms(self, num_samples: int) -> int:
        return num_samples * 1000 // self.sample_rate

    def ms_to_samples(self, ms: int) -> int:
        return ms * self.sample_rate // 1000
