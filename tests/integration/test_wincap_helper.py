"""실제 wincap 헬퍼로 PC 전체 소리 캡처: 무음이어도 시간축이 벽시계와 맞는지 (헬퍼가 없으면 skip)."""

import asyncio
import os
import time

import pytest

from livetrans.core.wincap import list_windows, process_alive, wincap_path
from livetrans.plugins.source.loopback import LoopbackSource

pytestmark = pytest.mark.skipif(wincap_path() is None, reason="wincap.exe 미빌드")


async def test_system_capture_keeps_wall_clock():
    src = LoopbackSource("system")
    await src.start()
    t0 = time.monotonic()
    samples = 0

    async def consume():
        nonlocal samples
        async for f in src.frames():
            samples += len(f.pcm)

    task = asyncio.create_task(consume())
    await asyncio.sleep(2.0)
    elapsed = time.monotonic() - t0
    await src.stop()
    await task
    assert src.lost_reason is None
    assert abs(samples / 16000 - elapsed) < 0.4


def test_process_alive():
    assert process_alive(os.getpid())
    assert not process_alive(99_999_999)


async def test_list_windows_shape():
    windows = await list_windows()
    assert all({"pid", "process", "title"} <= set(w) for w in windows)
