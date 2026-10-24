"""core.gpu 의 메모리 보조: 프로세스 RAM 조회, VRAM 캐시 반환 (torch 없이도 안전)."""

import sys

from livetrans.core.gpu import process_memory_mb, release_cached_gpu_memory


def test_process_memory_mb_reports_resident_memory():
    info = process_memory_mb()
    assert info is not None
    if sys.platform == "win32":
        assert info["rss_mb"] > 0 and info["private_mb"] >= info["rss_mb"] // 4
        assert info["peak_rss_mb"] >= info["rss_mb"]
    else:
        assert info["peak_rss_mb"] > 0


def test_release_cached_gpu_memory_is_safe_without_cuda():
    release_cached_gpu_memory()  # torch 미로드/비CUDA 환경에서도 예외 없이 지나간다
