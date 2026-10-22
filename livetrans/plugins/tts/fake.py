"""테스트용 가짜 TTS. 글자 수에 비례한 길이의 사인파를 돌려준다."""

from __future__ import annotations

import numpy as np

from livetrans.plugins.tts.base import AudioClip, TTSError, VoiceInfo


class FakeTTS:
    def __init__(
        self, name: str = "fake", languages: tuple[str, ...] = ("ko-KR", "en-US", "ja-JP")
    ):
        self.name = name
        self.languages = languages
        self.calls: list[tuple[str, str, float]] = []

    async def load(self) -> None:
        return None

    async def health(self) -> bool:
        return True

    async def list_voices(self) -> list[VoiceInfo]:
        return [
            VoiceInfo(f"{self.name}:{lang}", f"Fake {lang}", (lang,), supports_rate=True)
            for lang in self.languages
        ]

    async def synthesize(
        self, text: str, voice_id: str, *, rate: float = 1.0, style: str | None = None
    ) -> AudioClip:
        if voice_id not in self.languages:
            raise TTSError(f"음성이 없습니다: {voice_id}")
        self.calls.append((text, voice_id, rate))
        sr = 16000
        n = int(sr * 0.1 * len(text) / rate)
        t = np.arange(n) / sr
        return AudioClip((0.2 * np.sin(2 * np.pi * 330 * t)).astype(np.float32), sr)

    async def close(self) -> None:
        return None
