"""일본어 TTS 어댑터 `openjtalk` (D-018). Windows 에 일본어 음성이 없어서 넣은 기본 일본어 음성.

pyopenjtalk-plus(MIT, Open JTalk + 사전 + HTS 기본 음성 번들)를 쓴다. 공식 pyopenjtalk 는 Windows 에서 C++ 빌드가
필요해 미리 빌드된 휠이 있는 포크를 택했다. 엔진 기본 음성(nitech 남성 화자)만 제공한다 ([[권리 원칙]] 2).
"""

from __future__ import annotations

import asyncio
import threading

import numpy as np

from livetrans.plugins.tts.base import AudioClip, TTSError, VoiceInfo

VOICE_ID = "default"


class OpenJTalkTTS:
    def __init__(self, name: str = "openjtalk") -> None:
        self.name = name
        self._lock = threading.Lock()  # pyopenjtalk 는 전역 엔진을 쓰므로 직렬화
        self._loaded = False

    async def load(self) -> None:
        # 사전 로딩을 겸해 짧은 문장을 한 번 합성한다 (첫 요청 지연 제거)
        await asyncio.to_thread(self._tts_blocking, "テスト", 1.0)
        self._loaded = True

    async def health(self) -> bool:
        return self._loaded

    async def list_voices(self) -> list[VoiceInfo]:
        return [
            VoiceInfo(
                voice_ref=f"{self.name}:{VOICE_ID}",
                display_name="Open JTalk (기본 음성)",
                languages=("ja-JP",),
                supports_rate=True,
                license_note="Open JTalk 엔진 기본 음성 (pyopenjtalk-plus 번들)",
            )
        ]

    async def synthesize(
        self, text: str, voice_id: str, *, rate: float = 1.0, style: str | None = None
    ) -> AudioClip:
        if voice_id != VOICE_ID:
            raise TTSError(f"음성이 없습니다: {voice_id}")
        return await asyncio.to_thread(self._tts_blocking, text, rate)

    def _tts_blocking(self, text: str, rate: float) -> AudioClip:
        import pyopenjtalk

        with self._lock:
            x, sr = pyopenjtalk.tts(text, speed=rate)
        # pyopenjtalk 는 int16 범위의 float64 를 돌려준다
        return AudioClip(pcm=(np.asarray(x) / 32768.0).astype(np.float32), sample_rate=int(sr))

    async def close(self) -> None:
        return None
