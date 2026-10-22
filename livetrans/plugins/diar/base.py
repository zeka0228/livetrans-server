"""화자 분할 인터페이스 (명세 §4.8).

명세의 `Diarizer.update(chunk)` 는 분석 트랙(Enhancer) 위의 스트리밍 API 인데, 분석 트랙이 아직 없으므로
지금은 **세그먼터가 닫은 발화 단위**로 돌린다 (`diarize(seg)`). 발화가 이미 닫혀 있어 오른쪽 문맥 `C_diar` 는 필요 없다.
스트리밍으로 바꿀 때 이 인터페이스를 명세대로 넓힌다 (명세서 대비 변경 이력 참고).

- `DiarizerEngine` 은 모델을 한 번 올리는 프로세스 전역 객체.
- `Diarizer` 는 **세션마다 하나** — 화자 ID("S0", "S1", …) 와 화자 임베딩을 세션 메모리에만 두고 디스크에 남기지 않는다 (§13).
"""

from __future__ import annotations

from typing import Protocol

from livetrans.core.speaker import SpeakerTurn
from livetrans.pipeline.segmenter import SpeechSegment


class Diarizer(Protocol):
    """세션 단위 화자 분할기. `diarize` 가 돌려주는 구간의 시각은 `t_cap` 기준(ms)이다."""

    async def diarize(self, seg: SpeechSegment) -> list[SpeakerTurn]: ...

    def reset(self) -> None: ...


class DiarizerEngine(Protocol):
    name: str
    model_id: str

    async def load(self) -> None: ...

    def create(self) -> Diarizer: ...

    async def close(self) -> None: ...
