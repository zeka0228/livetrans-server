"""Windows 기본 음성 TTS 어댑터 `system` (명세 §5.3). SAPI 5 를 comtypes 로 호출한다.

SAPI COM 객체는 만든 스레드에서만 쓰므로, 전용 스레드 1개(executor)에서 COM 초기화·합성을 모두 한다.
설치된 음성은 Windows 언어 팩에 따라 다르다 (이 PC: Heami ko-KR, Zira en-US — 일본어 없음).
"""

from __future__ import annotations

import asyncio
import ctypes
import math
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from livetrans.plugins.tts.base import AudioClip, TTSError, VoiceInfo

SAFT_22KHZ_16BIT_MONO = 22
SAMPLE_RATE = 22050
SVSF_DEFAULT = 0


def lcid_to_bcp47(lcid_hex: str) -> str | None:
    """SAPI 음성 토큰의 Language 속성("412" 또는 "412;409") → "ko-KR"."""
    first = lcid_hex.split(";")[0].strip()
    try:
        lcid = int(first, 16)
    except ValueError:
        return None
    buf = ctypes.create_unicode_buffer(85)
    if ctypes.windll.kernel32.LCIDToLocaleName(lcid, buf, len(buf), 0) == 0:
        return None
    return buf.value


def rate_to_sapi(rate: float) -> int:
    """배속(1.0 = 기본) → SAPI Rate(-10~10). SAPI 는 10 단계에 약 3배로 본다."""
    if rate <= 0:
        return 0
    return max(-10, min(10, round(10 * math.log(rate, 3))))


class SystemTTS:
    def __init__(self, name: str = "system") -> None:
        self.name = name
        self._executor = ThreadPoolExecutor(max_workers=1, initializer=self._init_com)
        self._voices: list[tuple[VoiceInfo, object]] = []
        self._voice = None

    @staticmethod
    def _init_com() -> None:
        import comtypes

        comtypes.CoInitialize()

    async def _run(self, fn, *args):
        return await asyncio.get_running_loop().run_in_executor(self._executor, fn, *args)

    async def load(self) -> None:
        if sys.platform != "win32":
            raise TTSError("system TTS 는 Windows 전용입니다")
        await self._run(self._load_blocking)

    def _load_blocking(self) -> None:
        import comtypes.client as cc

        self._voice = cc.CreateObject("SAPI.SpVoice")
        tokens = self._voice.GetVoices()
        voices = []
        for i in range(tokens.Count):
            token = tokens.Item(i)
            name = token.GetDescription()
            lang = lcid_to_bcp47(token.GetAttribute("Language") or "")
            info = VoiceInfo(
                voice_ref=f"{self.name}:{name}",
                display_name=name,
                languages=(lang,) if lang else (),
                supports_rate=True,
                license_note="Windows 기본 제공 음성",
            )
            voices.append((info, token))
        self._voices = voices

    async def health(self) -> bool:
        return bool(self._voices)

    async def list_voices(self) -> list[VoiceInfo]:
        return [info for info, _ in self._voices]

    async def synthesize(
        self, text: str, voice_id: str, *, rate: float = 1.0, style: str | None = None
    ) -> AudioClip:
        token = next((t for info, t in self._voices if info.display_name == voice_id), None)
        if token is None:
            raise TTSError(f"음성이 없습니다: {voice_id}")
        return await self._run(self._synthesize_blocking, text, token, rate)

    def _synthesize_blocking(self, text: str, token, rate: float) -> AudioClip:
        import comtypes.client as cc

        fmt = cc.CreateObject("SAPI.SpAudioFormat")
        fmt.Type = SAFT_22KHZ_16BIT_MONO
        stream = cc.CreateObject("SAPI.SpMemoryStream")
        stream.Format = fmt
        self._voice.AudioOutputStream = stream
        self._voice.Voice = token
        self._voice.Rate = rate_to_sapi(rate)
        self._voice.Speak(text, SVSF_DEFAULT)
        data = bytes(stream.GetData())
        pcm = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
        return AudioClip(pcm=pcm, sample_rate=SAMPLE_RATE)

    async def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
