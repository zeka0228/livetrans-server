"""STT 환각 방지 (명세 §4.3 "환각 방지", 엔진 공용).

Whisper 계열은 말소리가 아닌 구간(웃음·BGM·효과음)이나 프롬프트가 오염됐을 때 같은 토큰을 토큰 상한(448)까지
반복한다 ("hohoho…", 2026-10-23 실방송 모드 B 에서 발생). 엔진이 무엇이든 세션이 결과 텍스트를 여기서 거른다.

판정 (하나라도 걸리면 환각):
- 압축비: `len(utf-8) / len(zlib)` 가 `COMPRESSION_RATIO_MAX`(2.4, Whisper 원 논문의 기본값) 초과. 짧은 문장은 zlib 헤더 때문에
  비율이 낮아 걸리지 않는다 (20 바이트 미만은 검사 안 함).
- 짧은 단위 반복: 1~8자 단위가 연속 `MIN_REPEATS`(6) 회 이상 (`하하하하하하하하` 포함 — 실제 웃음도 자막 가치가 없다).
- 단어 반복: 같은 단어(공백 구분)가 연속 `MIN_WORD_REPEATS`(5) 회 이상.

걸린 텍스트는 빈 문장으로 바꾸고 **프롬프트(직전 문장)로 쓰지 않는다** — 안 그러면 다음 발화까지 오염된다.
"""

from __future__ import annotations

import re
import zlib

COMPRESSION_RATIO_MAX = 2.4
MIN_REPEATS = 6
MIN_WORD_REPEATS = 5
_MIN_BYTES_FOR_RATIO = 20

_UNIT_REPEAT = re.compile(rf"(\S{{1,8}}?)\1{{{MIN_REPEATS - 1},}}")
_WORD_REPEAT = re.compile(rf"(?:^|\s)(\S+)(?:\s+\1){{{MIN_WORD_REPEATS - 1},}}(?=\s|$)")


def compression_ratio(text: str) -> float:
    data = text.encode("utf-8")
    if not data:
        return 0.0
    return len(data) / len(zlib.compress(data))


def hallucination_reason(text: str) -> str | None:
    """환각으로 보이면 이유 문자열, 아니면 None."""
    t = text.strip()
    if not t:
        return None
    m = _UNIT_REPEAT.search(t)
    if m:
        return f"반복 단위 {m.group(1)!r}"
    m = _WORD_REPEAT.search(t)
    if m:
        return f"반복 단어 {m.group(1)!r}"
    if len(t.encode("utf-8")) >= _MIN_BYTES_FOR_RATIO:
        ratio = compression_ratio(t)
        if ratio > COMPRESSION_RATIO_MAX:
            return f"압축비 {ratio:.1f}"
    return None


def max_new_tokens_for(
    duration_ms: int, *, tokens_per_s: float = 15.0, floor: int = 16, cap: int = 440
) -> int:
    """오디오 길이에 비례한 디코딩 토큰 상한. 보통 말은 초당 3~6 토큰이라 15 는 넉넉하고, 반복 환각은 여기서 끊긴다."""
    return int(min(cap, max(floor, duration_ms / 1000 * tokens_per_s + floor)))
