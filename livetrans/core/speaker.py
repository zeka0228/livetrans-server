"""화자 구간 (명세 §4.8 `SpeakerTurn`).

Diarizer 가 확정한 "누가 언제 말했나". 세션 안에서만 쓰는 익명 ID("S0", "S1", …)이며 디스크에 남기지 않는다 (§13).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpeakerTurn:
    speaker: str  # 세션 내 익명 ID
    t_start_ms: int  # t_cap 기준
    t_end_ms: int
    overlap: bool = False  # 다른 화자와 겹친 구간이면 True
