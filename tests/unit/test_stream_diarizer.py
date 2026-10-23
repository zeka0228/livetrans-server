"""스트리밍 화자 분할기 (D-028) — 파이프라인을 가짜로 바꿔 GPU 없이: 링버퍼, 확정 경계, 연속성, 되돌림."""

import numpy as np
import pytest
from pyannote.core import Annotation, Segment

from livetrans.core.config import DiarConfig
from livetrans.plugins.diar.registry import SpeakerRegistry
from livetrans.plugins.diar.stream import SAMPLE_RATE, StreamDiarizer


class Output:
    def __init__(self, ann, emb):
        self.speaker_diarization = ann
        self.speaker_embeddings = emb


class FakePipe:
    """절대 시각(ms)으로 적은 활동 표를, 받은 창 기준 상대 초로 바꿔 돌려준다."""

    def __init__(self) -> None:
        self.sd: StreamDiarizer | None = None
        self.tracks: dict[str, list[tuple[int, int]]] = {}
        self.embeddings: dict[str, np.ndarray] = {}
        self.calls: list[tuple[int, int]] = []  # (창 시작 ms, 창 길이 ms)

    def __call__(self, file, **kw):
        n = file["waveform"].shape[-1]
        w0_ms = (self.sd._total - n) * 1000 // SAMPLE_RATE
        w_len_ms = n * 1000 // SAMPLE_RATE
        self.calls.append((w0_ms, w_len_ms))
        ann = Annotation()
        for lb, ivs in self.tracks.items():
            for a, b in ivs:
                a2, b2 = max(a, w0_ms), min(b, w0_ms + w_len_ms)
                if b2 > a2:
                    ann[Segment((a2 - w0_ms) / 1000, (b2 - w0_ms) / 1000), f"{lb}@{a}"] = lb
        labels = list(ann.labels())
        emb = np.stack([self.embeddings[lb] for lb in labels]) if labels else None
        return Output(ann, emb)


def make(window_s=10.0, hop_s=2.0, right_s=1.0):
    pipe = FakePipe()
    cfg = DiarConfig(engine="pyannote", window_s=window_s, hop_s=hop_s, right_context_s=right_s)
    sd = StreamDiarizer(pipe, cfg, SpeakerRegistry(threshold=0.6, max_speakers=4))
    pipe.sd = sd
    return sd, pipe


def secs(s: float) -> np.ndarray:
    return np.full(int(s * SAMPLE_RATE), 0.1, dtype=np.float32)


def turns(ts):
    return [(t.speaker, t.t_start_ms, t.t_end_ms, t.overlap) for t in ts]


async def test_ready_after_hop_and_commit_leaves_right_context():
    sd, pipe = make()
    pipe.tracks, pipe.embeddings = {"A": [(0, 3000)]}, {"A": np.array([1.0, 0.0])}
    sd.push(secs(1.0))
    assert not sd.ready
    sd.push(secs(1.0))
    assert sd.ready and sd.total_ms == 2000
    got = await sd.step()
    assert turns(got) == [("S0", 0, 1000, False)]  # 2 s 중 오른쪽 1 s 는 아직 미확정
    assert sd.committed_ms == 1000 and not sd.ready
    assert pipe.calls == [(0, 2000)]


async def test_labels_are_linked_by_overlap_not_by_name_and_new_speaker_gets_new_id():
    sd, pipe = make()
    pipe.tracks, pipe.embeddings = {"A": [(0, 3000)]}, {"A": np.array([1.0, 0.0])}
    sd.push(secs(2.0))
    await sd.step()  # S0: [0, 1000)
    # 다음 창: 같은 사람이 라벨 B 로, 임베딩도 전혀 다르게 나와도 겹침으로 S0. C 는 새로 등장 → S1
    pipe.tracks = {"B": [(0, 3000)], "C": [(2500, 4000)]}
    pipe.embeddings = {"B": np.array([0.0, 1.0]), "C": np.array([0.0, 1.0])}
    sd.push(secs(2.0))
    got = await sd.step()  # 확정 [1000, 3000)
    assert turns(got) == [("S0", 1000, 3000, True), ("S1", 2500, 3000, True)]
    assert sd.registry.speakers == ["S0", "S1"]
    assert turns(sd.committed_turns(0, 3000)) == [("S0", 0, 3000, True), ("S1", 2500, 3000, True)]


async def test_history_is_rewritten_by_later_window():
    sd, pipe = make()
    pipe.tracks, pipe.embeddings = {"A": [(0, 3000)]}, {"A": np.array([1.0, 0.0])}
    sd.push(secs(4.0))
    await sd.step()  # S0: [0, 3000)
    # 더 긴 창에서 보니 2 s 이후는 다른 사람(Y) — X 가 S0 를 잇고(겹침 2 s), Y 는 새 화자
    pipe.tracks = {"X": [(0, 2000)], "Y": [(2000, 5000)]}
    pipe.embeddings = {"X": np.array([1.0, 0.0]), "Y": np.array([0.0, 1.0])}
    sd.push(secs(2.0))
    got = await sd.step()  # 확정 [3000, 5000)
    assert turns(got) == [("S1", 3000, 5000, False)]
    # 과거 [2000, 3000) 도 S1 로 고쳐졌다 (이미 보낸 자막은 그대로지만 다음 연속성 매칭 기준이 된다)
    assert turns(sd.committed_turns(0, 5000)) == [("S0", 0, 2000, False), ("S1", 2000, 5000, False)]


async def test_final_step_commits_everything_and_audio_slices_match():
    sd, pipe = make()
    pipe.tracks, pipe.embeddings = {"A": [(500, 1500)]}, {"A": np.array([1.0, 0.0])}
    ramp = np.arange(2 * SAMPLE_RATE, dtype=np.float32) / SAMPLE_RATE  # 값 = 초
    sd.push(ramp[: SAMPLE_RATE // 2])
    sd.push(ramp[SAMPLE_RATE // 2 :])
    got = await sd.step(final=True)
    assert turns(got) == [("S0", 500, 1500, False)] and sd.committed_ms == 2000
    a = sd.audio(500, 1500)
    assert (
        len(a) == SAMPLE_RATE
        and a[0] == pytest.approx(0.5)
        and a[-1] == pytest.approx(1.5 - 1 / SAMPLE_RATE)
    )


async def test_ring_buffer_keeps_only_recent_audio():
    sd, pipe = make(window_s=2.0, hop_s=1.0, right_s=0.5)  # 버퍼 용량 = 2 + 1 + 0.5 + 1 = 4.5 s
    pipe.tracks, pipe.embeddings = {"A": [(0, 100000)]}, {"A": np.array([1.0, 0.0])}
    for i in range(10):
        sd.push(np.full(SAMPLE_RATE, float(i), dtype=np.float32))
    assert sd.total_ms == 10000
    assert sd.audio(9000, 10000)[0] == 9.0 and sd.audio(6000, 7000)[0] == 6.0
    assert np.all(sd.audio(1000, 2000) == 0.0)  # 버퍼 밖은 0
    await sd.step()
    assert pipe.calls[-1][1] == 2000  # 창은 최근 2 s 만


async def test_short_turn_fragments_are_dropped_and_empty_window_is_fine():
    sd, pipe = make()
    pipe.tracks, pipe.embeddings = {"A": [(990, 1010)]}, {"A": np.array([1.0, 0.0])}
    sd.push(secs(2.0))
    assert await sd.step() == []  # 20 ms 조각은 버림
    pipe.tracks = {}
    sd.push(secs(2.0))
    assert await sd.step() == []
    sd.reset()
    assert sd.registry.speakers == [] and sd.committed_turns(0, 10000) == []
