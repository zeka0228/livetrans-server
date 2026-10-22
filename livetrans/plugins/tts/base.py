"""TTS 엔진 인터페이스 (명세 §5.2)."""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class VoiceInfo:
    voice_ref: str  # "<engine_instance>:<voice_id>" 전역 식별자
    display_name: str
    languages: tuple[str, ...]  # BCP-47 ("ja-JP", "ko-KR", "en-US")
    styles: tuple[str, ...] = ()
    supports_rate: bool = False
    license_note: str = ""

    def to_dict(self) -> dict:
        return {
            "voice_ref": self.voice_ref,
            "display_name": self.display_name,
            "languages": list(self.languages),
            "styles": list(self.styles),
            "supports_rate": self.supports_rate,
            "license_note": self.license_note,
        }


@dataclass
class AudioClip:
    pcm: np.ndarray  # float32 mono [-1, 1]
    sample_rate: int
    meta: dict = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return len(self.pcm) / self.sample_rate if self.sample_rate else 0.0

    def to_wav(self) -> bytes:
        """16-bit PCM mono WAV."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sample_rate)
            w.writeframes((np.clip(self.pcm, -1.0, 1.0) * 32767).astype("<i2").tobytes())
        return buf.getvalue()


class TTSError(RuntimeError):
    pass


class TTSEngine(Protocol):
    name: str  # 엔진 인스턴스 이름 (설정의 키)

    async def load(self) -> None: ...

    async def health(self) -> bool: ...

    async def list_voices(self) -> list[VoiceInfo]: ...

    async def synthesize(
        self, text: str, voice_id: str, *, rate: float = 1.0, style: str | None = None
    ) -> AudioClip: ...

    async def close(self) -> None: ...
