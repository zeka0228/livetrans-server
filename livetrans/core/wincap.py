"""Windows 오디오 캡처 헬퍼(`tools/wincap`) 위치 확인과 창 목록 조회 (D-016).

헬퍼는 C#(NAudio) 콘솔 앱이다. `scripts/build_wincap.bat` 으로 `tools/wincap/out/wincap.exe` 에 빌드한다.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

ENV_WINCAP = "LIVETRANS_WINCAP"
DEFAULT_WINCAP = Path(__file__).resolve().parents[2] / "tools" / "wincap" / "out" / "wincap.exe"
# process loopback(AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK) 최소 빌드
PROCESS_LOOPBACK_MIN_BUILD = 20348


def wincap_path() -> Path | None:
    path = Path(os.environ.get(ENV_WINCAP, DEFAULT_WINCAP))
    return path if sys.platform == "win32" and path.is_file() else None


def process_loopback_supported() -> bool:
    return sys.platform == "win32" and sys.getwindowsversion().build >= PROCESS_LOOPBACK_MIN_BUILD


def _list_windows_blocking(exe: Path) -> list[dict]:
    out = subprocess.run(
        [str(exe), "list"], capture_output=True, timeout=10, check=True, creationflags=_NO_WINDOW
    ).stdout
    return json.loads(out.decode("utf-8"))


async def list_windows() -> list[dict]:
    """보이는 최상위 창 `[{pid, process, title}]`. 헬퍼가 없으면 빈 목록."""
    exe = wincap_path()
    if exe is None:
        return []
    return await asyncio.to_thread(_list_windows_blocking, exe)


_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def process_alive(pid: int) -> bool:
    """프로세스가 살아 있는지. (Windows 의 os.kill(pid, 0) 은 프로세스를 종료시키므로 쓰지 않는다)"""
    if sys.platform != "win32" or pid <= 0:
        return False
    import ctypes

    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)
