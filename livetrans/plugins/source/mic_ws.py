"""모드 C 입력: 앱이 WebSocket 바이너리로 보내는 PCM16 LE 16 kHz mono (명세 §4.1, §6.2).

t_cap 은 수신 바이트 수로 계산한다 (I2). 프레임 경계가 홀수 바이트로 끊겨도 다음 프레임과 이어 붙인다.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

from livetrans.core.clock import SampleClock
from livetrans.plugins.source.base import AudioFrame, pcm16_to_float32


class MicWsSource:
    name = "mic_ws"
    has_video = False

    def __init__(self, sample_rate: int = 16000) -> None:
        self.clock = SampleClock(sample_rate)
        self._queue: asyncio.Queue[AudioFrame | None] = asyncio.Queue()
        self._carry = b""
        self._ended = False

    async def start(self) -> None:
        return None

    def feed(self, data: bytes) -> None:
        if self._ended or not data:
            return
        data = self._carry + data
        usable = len(data) // 2 * 2
        self._carry = data[usable:]
        if not usable:
            return
        pcm = pcm16_to_float32(data[:usable])
        self._queue.put_nowait(AudioFrame(pcm=pcm, t_start_ms=self.clock.advance(len(pcm))))

    async def frames(self) -> AsyncIterator[AudioFrame]:
        while True:
            frame = await self._queue.get()
            if frame is None:
                return
            yield frame

    def raw_media_dir(self) -> Path | None:
        return None

    async def stop(self) -> None:
        if not self._ended:
            self._ended = True
            self._queue.put_nowait(None)
