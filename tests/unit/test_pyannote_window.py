"""pyannote 어댑터의 문맥 창·구간 매핑 — 파이프라인을 가짜로 바꿔 GPU 없이 검사한다."""

import numpy as np
import pytest
from pyannote.core import Annotation, Segment

from livetrans.core.config import DiarConfig
from livetrans.pipeline.segmenter import SpeechSegment
from livetrans.plugins.diar.pyannote import CONTEXT_GAP_S, SAMPLE_RATE, PyannoteDiarizer
from livetrans.plugins.diar.registry import SpeakerRegistry


class Output:
    def __init__(self, ann: Annotation, emb: np.ndarray | None) -> None:
        self.speaker_diarization = ann
        self.speaker_embeddings = emb


class FakePipe:
    """호출마다 받은 파형 길이를 기록하고, 미리 정한 (시작 s, 끝 s, 라벨) 트랙을 돌려준다."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.tracks: list[tuple[float, float, str]] = []
        self.embeddings: dict[str, np.ndarray] = {}

    def __call__(self, file, **kw):
        wav = file["waveform"]
        self.calls.append({"samples": wav.shape[-1], **kw})
        ann = Annotation()
        for i, (a, b, lb) in enumerate(self.tracks):
            ann[Segment(a, b), i] = lb
        labels = list(ann.labels())
        emb = np.stack([self.embeddings[lb] for lb in labels]) if labels else None
        return Output(ann, emb)


def seg(seg_id: int, t_start_ms: int, seconds: float) -> SpeechSegment:
    n = int(seconds * SAMPLE_RATE)
    return SpeechSegment(
        seg_id=seg_id,
        t_start_ms=t_start_ms,
        t_end_ms=t_start_ms + n * 1000 // SAMPLE_RATE,
        pcm=np.full(n, 0.1, dtype=np.float32),
    )


@pytest.fixture
def pipe():
    return FakePipe()


def make(pipe, **cfg) -> PyannoteDiarizer:
    c = DiarConfig(engine="pyannote", **cfg)
    return PyannoteDiarizer(pipe, c, SpeakerRegistry(threshold=c.match_threshold, max_speakers=4))


async def test_window_grows_with_history_and_is_trimmed_to_budget(pipe):
    d = make(pipe, context_s=5.0)
    pipe.tracks, pipe.embeddings = [(0.0, 2.0, "A")], {"A": np.array([1.0, 0.0])}
    await d.diarize(seg(0, 0, 2.0))
    assert pipe.calls[-1]["samples"] == 2 * SAMPLE_RATE and pipe.calls[-1]["max_speakers"] == 4
    await d.diarize(seg(1, 3000, 2.0))
    gap = int(CONTEXT_GAP_S * SAMPLE_RATE)
    assert pipe.calls[-1]["samples"] == 2 * SAMPLE_RATE + gap + 2 * SAMPLE_RATE
    assert d.last_window_s == pytest.approx(4.3)
    await d.diarize(seg(2, 6000, 2.0))  # 예산 5 s: 2+2+2 > 5 → 가장 오래된 것 제거 → 2+2
    assert pipe.calls[-1]["samples"] == 3 * (2 * SAMPLE_RATE) + 2 * gap  # 현재 호출은 제거 전 창
    await d.diarize(seg(3, 9000, 2.0))
    assert pipe.calls[-1]["samples"] == 2 * (2 * SAMPLE_RATE) + 2 * gap + 2 * SAMPLE_RATE
    d.reset()
    await d.diarize(seg(4, 12000, 2.0))
    assert pipe.calls[-1]["samples"] == 2 * SAMPLE_RATE  # reset 이 문맥도 비운다


async def test_context_zero_disables_history(pipe):
    d = make(pipe, context_s=0)
    pipe.tracks, pipe.embeddings = [(0.0, 1.0, "A")], {"A": np.array([1.0, 0.0])}
    await d.diarize(seg(0, 0, 1.0))
    await d.diarize(seg(1, 2000, 1.0))
    assert [c["samples"] for c in pipe.calls] == [SAMPLE_RATE, SAMPLE_RATE]


async def test_only_current_utterance_region_is_returned_in_t_cap(pipe):
    d = make(pipe, context_s=20.0)
    pipe.tracks, pipe.embeddings = [(0.0, 3.0, "A")], {"A": np.array([1.0, 0.0])}
    await d.diarize(seg(0, 1000, 3.0))  # 문맥: 3 s
    # 창 = [A 3.0 s][gap 0.3 s][현재 4.0 s]. 현재 구간은 창의 3.3~7.3 s
    pipe.tracks = [
        (0.0, 3.0, "A"),  # 문맥에만 있는 라벨 → 제외
        (3.3, 5.3, "A"),  # 현재 0~2 s
        (5.0, 7.3, "B"),  # 현재 1.7~4.0 s (A 와 0.3 s 겹침)
        (2.0, 3.4, "C"),  # 현재 구간에 0.1 s 만 걸침 → min_speaker_ms 미만이라 버림
    ]
    pipe.embeddings = {
        "A": np.array([1.0, 0.0]),
        "B": np.array([0.0, 1.0]),
        "C": np.array([0.7, 0.7]),
    }
    turns = await d.diarize(seg(1, 10000, 4.0))
    assert [(t.speaker, t.t_start_ms, t.t_end_ms, t.overlap) for t in turns] == [
        ("S0", 10000, 12000, True),
        ("S1", 11700, 14000, True),
    ]
    assert d.registry.speakers == ["S0", "S1"]


async def test_labels_outside_current_region_mean_no_turns(pipe):
    d = make(pipe, context_s=20.0)
    pipe.tracks, pipe.embeddings = [(0.0, 1.0, "A")], {"A": np.array([1.0, 0.0])}
    await d.diarize(seg(0, 0, 1.0))
    pipe.tracks = [(0.0, 1.0, "A")]  # 문맥 구간에만 트랙, 현재 발화(1.3~2.3 s)는 비어 있음
    assert await d.diarize(seg(1, 5000, 1.0)) == []
    assert d.registry.speakers == ["S0"]  # 등록부는 건드리지 않는다


async def test_missing_embeddings_fall_back_to_nan_rule(pipe):
    d = make(pipe, context_s=0)
    pipe.tracks = [(0.0, 1.0, "A")]
    pipe.embeddings = {"A": np.array([1.0, 0.0])}
    await d.diarize(seg(0, 0, 1.0))
    pipe.embeddings = {}

    class NoEmb(FakePipe):
        def __call__(self, file, **kw):
            out = super().__call__(file, **kw)
            out.speaker_embeddings = None
            return out

    d._pipe = NoEmb()
    d._pipe.tracks = [(0.0, 1.0, "X")]
    d._pipe.embeddings = {"X": np.array([0.0, 1.0])}  # 파이프라인이 None 으로 바꿔 돌려준다
    turns = await d.diarize(seg(1, 2000, 1.0))
    assert [t.speaker for t in turns] == ["S0"]  # 임베딩이 없으면 직전 화자


async def test_context_continuity_carries_id_even_if_embedding_differs(pipe):
    d = make(pipe, context_s=20.0)
    pipe.tracks, pipe.embeddings = [(0.0, 2.0, "A")], {"A": np.array([1.0, 0.0])}
    assert [t.speaker for t in await d.diarize(seg(0, 0, 2.0))] == ["S0"]
    # 창 = [발화0 2 s][0.3][발화1 2 s]. 라벨 이름이 바뀌고 임베딩도 S0 와 전혀 다르지만, 문맥 구간이 S0 자리와 겹치므로 S0
    pipe.tracks = [(0.0, 2.0, "Q"), (2.3, 4.3, "Q")]
    pipe.embeddings = {"Q": np.array([0.0, 1.0])}
    assert [t.speaker for t in await d.diarize(seg(1, 3000, 2.0))] == ["S0"]
    # 창 = [발화0][0.3][발화1][0.3][발화2]. Z 는 문맥(S0) 전체를 덮고, Y 는 현재 발화에만 → Y 는 임베딩이 S0 중심점과 비슷해도
    # (S0 는 이번 호출에서 Z 가 가져감) 새 화자 S1
    pipe.tracks = [(0.0, 2.0, "Z"), (2.3, 4.3, "Z"), (4.6, 6.6, "Y")]
    pipe.embeddings = {"Z": np.array([0.0, 1.0]), "Y": np.array([0.3, 1.0])}
    assert [t.speaker for t in await d.diarize(seg(2, 6000, 2.0))] == ["S1"]
    assert d.registry.speakers == ["S0", "S1"]
    # 다음 발화: 문맥에서 S0(Z 자리)·S1(Y 자리) 둘 다 보이고 현재 구간에 둘 다 말함 → 겹친 시간으로 각각 이어진다
    pipe.tracks = [
        (0.0, 2.0, "M"),
        (2.3, 4.3, "M"),
        (4.6, 6.6, "N"),
        (6.9, 7.9, "N"),
        (7.9, 8.9, "M"),
    ]
    pipe.embeddings = {"M": np.array([0.0, 1.0]), "N": np.array([0.3, 1.0])}
    turns = await d.diarize(seg(3, 9000, 2.0))
    assert [(t.speaker, t.t_start_ms, t.t_end_ms) for t in turns] == [
        ("S1", 9000, 10000),
        ("S0", 10000, 11000),
    ]


async def test_continuity_needs_minimum_overlap(pipe):
    d = make(pipe, context_s=20.0)
    pipe.tracks, pipe.embeddings = [(0.0, 2.0, "A")], {"A": np.array([1.0, 0.0])}
    await d.diarize(seg(0, 0, 2.0))
    # 문맥과 0.1 s 만 겹치는 라벨은 연속성으로 잇지 않고 임베딩으로 판단 → (0,1) 은 S0 와 다르므로 새 화자
    pipe.tracks = [(1.9, 2.0, "B"), (2.3, 4.3, "B")]
    pipe.embeddings = {"B": np.array([0.0, 1.0])}
    assert [t.speaker for t in await d.diarize(seg(1, 3000, 2.0))] == ["S1"]


async def test_history_is_relabeled_by_the_longer_window(pipe):
    d = make(pipe, context_s=20.0)
    pipe.tracks, pipe.embeddings = [(0.0, 2.0, "A")], {"A": np.array([1.0, 0.0])}
    await d.diarize(seg(0, 0, 2.0))  # S0
    pipe.tracks = [(0.0, 2.0, "A"), (2.3, 4.3, "B")]
    pipe.embeddings = {"A": np.array([1.0, 0.0]), "B": np.array([0.0, 1.0])}
    assert [t.speaker for t in await d.diarize(seg(1, 3000, 2.0))] == ["S1"]
    # 더 긴 창: 발화1 의 앞 절반은 사실 S0 였다고 판단 (P = 발화0 + 발화1 앞 절반, R = 발화1 뒤 절반 + 현재)
    pipe.tracks = [(0.0, 2.0, "P"), (2.3, 3.3, "P"), (3.3, 4.3, "R"), (4.6, 6.6, "R")]
    pipe.embeddings = {"P": np.array([1.0, 0.0]), "R": np.array([0.0, 1.0])}
    assert [t.speaker for t in await d.diarize(seg(2, 6000, 2.0))] == ["S1"]
    # 발화1 의 문맥 기록이 [S0 앞 절반, S1 뒤 절반] 으로 고쳐졌는지: 발화1 앞 절반과만 겹치는 라벨은 이제 S0 로 이어진다
    pipe.tracks = [(2.3, 3.3, "U"), (6.9, 8.9, "U")]
    pipe.embeddings = {"U": np.array([0.0, 1.0])}  # 임베딩은 S1 쪽이지만 연속성이 우선
    assert [t.speaker for t in await d.diarize(seg(3, 9000, 2.0))] == ["S0"]
