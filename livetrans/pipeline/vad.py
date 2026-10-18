"""Silero VAD 스트리밍 래퍼.

faster-whisper 가 번들한 Silero VAD v6 ONNX 를 쓴다 (별도 torch 의존성 없음). faster-whisper 의
`SileroVADModel` 은 파일 전체를 배치로 처리하므로, 같은 세션을 512 샘플(32 ms) 창마다 LSTM 상태(h, c)와
직전 64 샘플 문맥을 이어서 호출하는 방식으로 스트리밍화했다. 배치 결과와 동일함은 단위 테스트로 확인한다.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000
WINDOW_SAMPLES = 512  # 32 ms @ 16 kHz
CONTEXT_SAMPLES = 64


class StreamingSileroVAD:
    """한 번에 512 샘플 창을 받아 음성 확률(0~1)을 돌려준다."""

    def __init__(self) -> None:
        from faster_whisper.vad import get_vad_model

        self._session = get_vad_model().session
        self.reset()

    def reset(self) -> None:
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._context = np.zeros(CONTEXT_SAMPLES, dtype=np.float32)

    def __call__(self, window: np.ndarray) -> float:
        if window.shape != (WINDOW_SAMPLES,):
            raise ValueError(f"창 크기는 {WINDOW_SAMPLES} 샘플이어야 합니다: {window.shape}")
        x = np.concatenate([self._context, window.astype(np.float32, copy=False)])[None, :]
        probs, self._h, self._c = self._session.run(None, {"input": x, "h": self._h, "c": self._c})
        self._context = window[-CONTEXT_SAMPLES:].astype(np.float32, copy=True)
        return float(np.asarray(probs).reshape(-1)[0])
