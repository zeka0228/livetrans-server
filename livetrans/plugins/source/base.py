"""입력 소스 인터페이스 (명세 §4.1)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass
class AudioFrame:
    pcm: np.ndarray  # float32 mono 16 kHz, [-1, 1]
    t_start_ms: int  # t_cap 기준


class Source(Protocol):
    name: str
    has_video: bool

    async def start(self) -> None: ...

    def frames(self) -> AsyncIterator[AudioFrame]: ...

    def raw_media_dir(self) -> Path | None: ...

    async def stop(self) -> None: ...


def pcm16_to_float32(data: bytes) -> np.ndarray:
    """little-endian PCM16 바이트 → float32 [-1, 1]."""
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
