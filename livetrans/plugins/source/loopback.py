"""모드 B 입력: PC 에서 재생 중인 소리를 캡처 (명세 §4.1 `loopback`, D-016).

`target` = `"system"`(기본 출력 장치 전체), `"pid:<N>"`(그 프로세스 트리의 소리만), `"device:<ID>"`(특정 출력 장치 — 모드 A
격리 장치, 명세 §4.9).
캡처는 `wincap` 헬퍼가 하고, 16 kHz mono PCM16 을 stdout 으로 받는다. 헬퍼가 무음 구간을 0 으로 채우므로
수신 샘플 수가 곧 벽시계와 맞는 t_cap 이다 (I2).

uvicorn 이 어떤 이벤트 루프를 쓰든 동작하도록 asyncio subprocess 대신 Popen + 리더 스레드를 쓴다.
"""

from __future__ import annotations

import asyncio
import subprocess
import threading
from collections.abc import AsyncIterator
from pathlib import Path

from livetrans.core.clock import SampleClock
from livetrans.core.wincap import _NO_WINDOW, wincap_path
from livetrans.plugins.source.base import AudioFrame, pcm16_to_float32

READ_BYTES = 3200  # 100 ms
READY_TIMEOUT_S = 8.0


class CaptureError(RuntimeError):
    pass


def parse_target(target: str) -> list[str]:
    """`system` / `pid:<N>` / `device:<ID>` → wincap 인자. 형식이 틀리면 ValueError."""
    if target == "system":
        return ["system"]
    if target.startswith("pid:") and target[4:].isdigit():
        return ["pid", target[4:]]
    if target.startswith("device:") and target[7:].startswith("{"):
        return ["device", target[7:]]
    raise ValueError(f"알 수 없는 캡처 대상: {target!r}")


class LoopbackSource:
    name = "loopback"
    has_video = False

    def __init__(self, target: str, sample_rate: int = 16000) -> None:
        self.target = target
        self._args = parse_target(target)
        self.clock = SampleClock(sample_rate)
        self.lost_reason: str | None = None
        self._queue: asyncio.Queue[AudioFrame | None] = asyncio.Queue()
        self._proc: subprocess.Popen | None = None
        self._stopping = False
        self._stderr: list[str] = []

    async def start(self) -> None:
        exe = wincap_path()
        if exe is None:
            raise CaptureError(
                "캡처 헬퍼(wincap.exe)가 없습니다. scripts\\build_wincap.bat 으로 빌드하세요"
            )
        loop = asyncio.get_running_loop()
        ready: asyncio.Future[str] = loop.create_future()
        self._proc = subprocess.Popen(  # noqa: ASYNC220 — 모듈 docstring 참고
            [str(exe), "capture", *self._args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=_NO_WINDOW,
        )
        threading.Thread(target=self._read_stderr, args=(loop, ready), daemon=True).start()
        try:
            await asyncio.wait_for(ready, READY_TIMEOUT_S)
        except (TimeoutError, CaptureError):
            self._kill()
            raise
        threading.Thread(target=self._read_stdout, args=(loop,), daemon=True).start()

    def _read_stderr(self, loop: asyncio.AbstractEventLoop, ready: asyncio.Future) -> None:
        assert self._proc and self._proc.stderr
        for raw in self._proc.stderr:
            line = raw.decode("utf-8", "replace").strip()
            self._stderr.append(line)
            if line.startswith("READY"):
                loop.call_soon_threadsafe(_set_result, ready, line)
            elif line.startswith("ERROR") and not ready.done():
                loop.call_soon_threadsafe(_set_exception, ready, CaptureError(line))
        # 헬퍼가 READY 전에 죽은 경우
        loop.call_soon_threadsafe(
            _set_exception, ready, CaptureError("; ".join(self._stderr) or "캡처 헬퍼 종료")
        )

    def _read_stdout(self, loop: asyncio.AbstractEventLoop) -> None:
        assert self._proc and self._proc.stdout
        carry = b""
        while True:
            data = self._proc.stdout.read(READ_BYTES)
            if not data:
                break
            data = carry + data
            usable = len(data) // 2 * 2
            carry = data[usable:]
            pcm = pcm16_to_float32(data[:usable])
            loop.call_soon_threadsafe(self._push, pcm)
        loop.call_soon_threadsafe(self._ended)

    def _push(self, pcm) -> None:
        if not self._stopping:
            self._queue.put_nowait(AudioFrame(pcm=pcm, t_start_ms=self.clock.advance(len(pcm))))

    def _ended(self) -> None:
        if not self._stopping:
            errors = [line for line in self._stderr if line.startswith("ERROR")]
            self.lost_reason = errors[-1] if errors else "캡처가 예기치 않게 끝났습니다"
        self._queue.put_nowait(None)

    async def frames(self) -> AsyncIterator[AudioFrame]:
        while True:
            frame = await self._queue.get()
            if frame is None:
                return
            yield frame

    def raw_media_dir(self) -> Path | None:
        return None

    async def stop(self) -> None:
        if self._stopping:
            return
        self._stopping = True
        if self._proc is None:
            self._queue.put_nowait(None)
            return
        await asyncio.to_thread(self._close)
        self._queue.put_nowait(None)

    def _close(self) -> None:
        assert self._proc
        try:
            if self._proc.stdin:
                self._proc.stdin.close()  # 헬퍼는 stdin 이 닫히면 정상 종료
            self._proc.wait(timeout=3)
        except (OSError, subprocess.TimeoutExpired):
            self._kill()

    def _kill(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.kill()


def _set_result(fut: asyncio.Future, value) -> None:
    if not fut.done():
        fut.set_result(value)


def _set_exception(fut: asyncio.Future, exc: BaseException) -> None:
    if not fut.done():
        fut.set_exception(exc)


class CaptureBackend:
    """캡처 가능 여부·창 목록·소스 생성을 한곳에 모은다. 테스트에서는 가짜로 바꿔 끼운다."""

    def available(self) -> bool:
        return wincap_path() is not None

    def app_capture_supported(self) -> bool:
        from livetrans.core.wincap import process_loopback_supported

        return self.available() and process_loopback_supported()

    def process_alive(self, pid: int) -> bool:
        from livetrans.core.wincap import process_alive

        return process_alive(pid)

    async def list_devices(self) -> list[dict]:
        from livetrans.core.wincap import list_devices

        return await list_devices()

    async def meter(self, device_ids: list[str], ms: int) -> dict[str, dict]:
        from livetrans.core.wincap import meter

        return await meter(device_ids, ms)

    async def list_windows(self) -> list[dict]:
        from livetrans.core.wincap import list_windows

        return await list_windows()

    def create_source(self, target: str, sample_rate: int) -> LoopbackSource:
        return LoopbackSource(target, sample_rate)
