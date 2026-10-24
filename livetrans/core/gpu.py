"""GPU 보조: pip 로 설치된 CUDA DLL 경로 등록, NVML 로 메모리 조회."""

from __future__ import annotations

import logging
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


def release_cached_gpu_memory() -> None:
    """torch 캐싱 할당자가 쥔 VRAM 을 드라이버에 돌려준다 (세션 끝). 세션 중 쌓인 다양한 크기의 블록이 남아
    유휴 때도 수 GB 가 예약된 채였고, Windows 는 이것을 프로세스 커밋(Private)에도 잡는다 (2026-10-25 측정)."""
    torch = sys.modules.get("torch")
    if torch is None:
        return
    try:
        if torch.cuda.is_available():
            with GPU_LOCK:
                torch.cuda.empty_cache()
    except Exception as e:  # noqa: BLE001 — 정리 실패는 무시 (다음 세션에 영향 없음)
        logging.getLogger("livetrans.gpu").debug("VRAM 캐시 반환 실패: %s", e)


def process_memory_mb() -> dict[str, int] | None:
    """이 프로세스의 RAM: rss(상주)·private(커밋). /api/health 에 실어 메모리 추적용. 못 재면 None."""
    if sys.platform == "win32":
        import ctypes
        import ctypes.wintypes as wt

        class _PMC(ctypes.Structure):
            _fields_ = [
                ("cb", wt.DWORD),
                ("PageFaultCount", wt.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        try:
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            fn = k32.K32GetProcessMemoryInfo
            fn.argtypes = [wt.HANDLE, ctypes.POINTER(_PMC), wt.DWORD]
            fn.restype = wt.BOOL
            k32.GetCurrentProcess.restype = wt.HANDLE
            pmc = _PMC()
            pmc.cb = ctypes.sizeof(_PMC)
            if not fn(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
                return None
            mb = 1024 * 1024
            return {
                "rss_mb": int(pmc.WorkingSetSize // mb),
                "private_mb": int(pmc.PrivateUsage // mb),
                "peak_rss_mb": int(pmc.PeakWorkingSetSize // mb),
            }
        except Exception:  # noqa: BLE001
            return None
    try:
        import resource

        peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return {"peak_rss_mb": int(peak_kb // 1024)}
    except Exception:  # noqa: BLE001
        return None


def gpu_util() -> float | None:
    """첫 GPU 의 사용률(0~1). NVML 을 못 쓰면 None."""
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            return pynvml.nvmlDeviceGetUtilizationRates(h).gpu / 100.0
        finally:
            pynvml.nvmlShutdown()
    except Exception:  # noqa: BLE001
        return None


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
