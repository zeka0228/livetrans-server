"""WebSocket 이벤트 봉투 (명세 §6.2). 모든 t_*_ms 는 t_cap 기준."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

PROTOCOL_VERSION = 1


@dataclass(frozen=True)
class Envelope:
    v: int
    type: str
    seq: int
    session_id: str
    ts_wall_ms: int
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))


class EventFactory:
    """세션 하나의 seq 를 단조 증가시키며 봉투를 만든다."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._seq = 0

    @property
    def last_seq(self) -> int:
        return self._seq

    def make(self, type_: str, payload: dict[str, Any]) -> Envelope:
        self._seq += 1
        return Envelope(
            v=PROTOCOL_VERSION,
            type=type_,
            seq=self._seq,
            session_id=self.session_id,
            ts_wall_ms=int(time.time() * 1000),
            payload=payload,
        )


# --- payload 빌더: 명세 §6.2 표와 1:1 ---

SESSION_STATES = ("starting", "calibrating", "buffering", "playing", "stopped", "error")


def session_state_payload(
    state: str, delay_s: float | None = None, play_start_wall_ms: int | None = None
) -> dict[str, Any]:
    if state not in SESSION_STATES:
        raise ValueError(f"알 수 없는 세션 상태: {state}")
    return {"state": state, "delay_s": delay_s, "play_start_wall_ms": play_start_wall_ms}


def error_payload(code: str, message: str, recoverable: bool) -> dict[str, Any]:
    return {"code": code, "message": message, "recoverable": recoverable}
