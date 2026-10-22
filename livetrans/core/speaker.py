"""화자 구간 (명세 §4.8 `SpeakerTurn`) 과 구간 → 발화 배치 보조 함수.

Diarizer 가 확정한 "누가 언제 말했나". 세션 안에서만 쓰는 익명 ID("S0", "S1", …)이며 디스크에 남기지 않는다 (§13).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class SpeakerTurn:
    speaker: str  # 세션 내 익명 ID
    t_start_ms: int  # t_cap 기준
    t_end_ms: int
    overlap: bool = False  # 다른 화자와 겹친 구간이면 True


def speaking_ms(turns: Iterable[SpeakerTurn]) -> dict[str, int]:
    """화자별 말한 시간(ms)."""
    total: dict[str, int] = defaultdict(int)
    for t in turns:
        total[t.speaker] += max(0, t.t_end_ms - t.t_start_ms)
    return dict(total)


def select_speakers(turns: Sequence[SpeakerTurn], max_per_seg: int) -> tuple[str, ...]:
    """발화에서 STT 를 돌릴 화자 — 말한 시간이 긴 순으로 최대 `max_per_seg` 명 (명세 §4.3 "화자별 실행 k 회")."""
    total = speaking_ms(turns)
    first_seen = {}
    for t in turns:
        first_seen.setdefault(t.speaker, t.t_start_ms)
    ranked = sorted(total, key=lambda s: (-total[s], first_seen[s]))
    return tuple(ranked[: max(1, max_per_seg)])


def speaker_extent(turns: Iterable[SpeakerTurn], speaker: str) -> tuple[int, int] | None:
    """해당 화자가 말한 전체 범위 (첫 시작, 마지막 끝). 구간이 없으면 None."""
    starts = [t.t_start_ms for t in turns if t.speaker == speaker]
    ends = [t.t_end_ms for t in turns if t.speaker == speaker]
    if not starts:
        return None
    return min(starts), max(ends)
