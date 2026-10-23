"""오디오 보조: 리샘플(선형), 다운믹스, dB↔게인, 덕킹 엔벨로프, PCM16 변환. numpy 만 쓴다."""

from __future__ import annotations

import numpy as np

OUT_RATE = 48000  # 모드 A 출력(원본·TTS 믹스) 샘플레이트
ANALYSIS_RATE = 16000  # STT·화자 분할 입력


def db_to_gain(db: float) -> float:
    return float(10 ** (db / 20))


def pcm16_bytes_to_float(data: bytes, channels: int) -> np.ndarray:
    """little-endian PCM16 → float32 [n, channels]."""
    arr = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        n = len(arr) // channels * channels
        arr = arr[:n].reshape(-1, channels)
    return arr


def float_to_pcm16_bytes(x: np.ndarray) -> bytes:
    return (np.clip(x, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def downmix_mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1).astype(np.float32) if x.ndim == 2 else x.astype(np.float32, copy=False)


def resample_linear(x: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """선형 보간 리샘플 (mono). TTS 클립(22.05/48 kHz)과 원본을 48 kHz 로 맞추는 용도 — 음질보다 단순함 우선."""
    if src_rate == dst_rate or len(x) == 0:
        return x.astype(np.float32, copy=False)
    n_dst = round(len(x) * dst_rate / src_rate)
    t_src = np.arange(len(x)) / src_rate
    t_dst = np.arange(n_dst) / dst_rate
    return np.interp(t_dst, t_src, x).astype(np.float32)


def decimate_48k_to_16k(mono48: np.ndarray) -> np.ndarray:
    """48 kHz → 16 kHz: 3샘플 평균(박스 필터). STT 입력용으로 충분하고 지연이 없다."""
    n = len(mono48) // 3 * 3
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    return mono48[:n].reshape(-1, 3).mean(axis=1).astype(np.float32)


def duck_envelope(n: int, active: list[tuple[int, int]], duck_gain: float, ramp: int) -> np.ndarray:
    """길이 n 의 게인 엔벨로프. `active` 구간(샘플 인덱스)에서는 duck_gain, 경계는 `ramp` 샘플 동안 선형으로."""
    env = np.ones(n, dtype=np.float32)
    if duck_gain >= 1.0 or not active:
        return env
    for a, b in active:
        a, b = max(0, a), min(n, b)
        if b <= a:
            continue
        env[a:b] = np.minimum(env[a:b], duck_gain)
        if ramp > 0:
            ra = max(0, a - ramp)
            if ra < a:
                env[ra:a] = np.minimum(
                    env[ra:a], np.linspace(1.0, duck_gain, a - ra, dtype=np.float32)
                )
            rb = min(n, b + ramp)
            if b < rb:
                env[b:rb] = np.minimum(
                    env[b:rb], np.linspace(duck_gain, 1.0, rb - b, dtype=np.float32)
                )
    return env
