"""GPU 보조: pip 로 설치된 CUDA DLL 경로 등록, NVML 로 메모리 조회."""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Any

# STT·화자 분할·번역이 같은 GPU 를 쓰므로 GPU 작업은 이 락 아래에서만 실행한다 (명세 §2.3)
GPU_LOCK = threading.Lock()


def prepare_cuda_dlls() -> list[Path]:
    """Windows 에서 nvidia-*-cu12 pip 패키지의 bin 폴더를 DLL 검색 경로에 추가한다.

    CTranslate2 가 cublas64_12 / cudnn64_9 를 찾지 못하는 문제의 예방 조치 ([[Windows 환경 함정]]).
    추가한 경로 목록을 돌려준다 (없으면 빈 목록).
    """
    if sys.platform != "win32":
        return []
    try:
        import nvidia  # type: ignore[import-not-found]
    except ImportError:
        return []
    added: list[Path] = []
    for base in map(Path, nvidia.__path__):
        for sub in ("cublas", "cudnn", "cuda_nvrtc"):
            bin_dir = base / sub / "bin"
            if bin_dir.is_dir():
                os.add_dll_directory(str(bin_dir))
                added.append(bin_dir)
    return added


def gpu_info() -> dict[str, Any] | None:
    """첫 번째 GPU 의 이름·메모리(MB). NVML 을 못 쓰면 None."""
    try:
        import pynvml
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", "replace")
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return {
                "name": name,
                "mem_total_mb": int(mem.total // (1024 * 1024)),
                "mem_used_mb": int(mem.used // (1024 * 1024)),
            }
        finally:
            pynvml.nvmlShutdown()
    except pynvml.NVMLError:  # NVML 오류는 health 를 실패시키지 않는다
        return None
