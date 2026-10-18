"""캡처 대상 목록 (API 계약 S-9, D-016)."""

from __future__ import annotations

import os

from fastapi import APIRouter, Request

router = APIRouter(prefix="/api")

SYSTEM_TARGET = {
    "id": "system",
    "kind": "system",
    "title": "PC 전체 소리 (기본 출력 장치)",
    "process": None,
    "pid": None,
    "windows": [],
}


def build_targets(windows: list[dict], app_capture: bool) -> list[dict]:
    """맨 앞은 항상 system. 창은 프로세스 단위로 묶는다 (캡처가 프로세스 트리 단위라서)."""
    targets = [dict(SYSTEM_TARGET)]
    if not app_capture:
        return targets
    by_pid: dict[int, dict] = {}
    for w in windows:
        pid = int(w["pid"])
        if pid == os.getpid():
            continue
        entry = by_pid.get(pid)
        if entry is None:
            entry = by_pid[pid] = {
                "id": f"pid:{pid}",
                "kind": "app",
                "title": w["title"],
                "process": w["process"],
                "pid": pid,
                "windows": [],
            }
            targets.append(entry)
        entry["windows"].append(w["title"])
    return targets


@router.get("/capture/targets")
async def capture_targets(request: Request) -> dict:
    backend = request.app.state.capture
    available = backend.available()
    app_capture = available and backend.app_capture_supported()
    windows = await backend.list_windows() if app_capture else []
    return {
        "targets": build_targets(windows, app_capture) if available else [],
        "capture_available": available,
        "app_capture_supported": app_capture,
    }
