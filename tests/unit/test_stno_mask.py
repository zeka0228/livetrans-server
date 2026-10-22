import numpy as np

from livetrans.core.speaker import SpeakerTurn
from livetrans.plugins.stt.stno import (
    NON_TARGET,
    OVERLAP,
    SILENCE,
    TARGET,
    build_stno_mask,
    speaker_activity,
    stno_from_activity,
)


def test_single_speaker_without_turns_is_all_target_then_silence():
    m = build_stno_mask(2000, n_frames=200)  # 2 s = 100 프레임
    assert m.shape == (4, 200)
    assert m[TARGET, :100].all() and not m[TARGET, 100:].any()
    assert m[SILENCE, 100:].all() and not m[SILENCE, :100].any()
    assert not m[NON_TARGET].any() and not m[OVERLAP].any()
    # 프레임마다 4채널 합은 1
    assert np.allclose(m.sum(axis=0), 1.0)


def test_two_speakers_overlap_categories():
    # S0: 0~1.0 s, S1: 0.6~2.0 s  (0.6~1.0 겹침). 발화 길이 2.0 s, 창 150 프레임(3 s)
    turns = [SpeakerTurn("S0", 0, 1000), SpeakerTurn("S1", 600, 2000)]
    act = speaker_activity(turns, ("S0", "S1"), seg_t_start_ms=0, n_frames=150)
    assert act[0, :50].all() and not act[0, 50:].any()
    assert act[1, 30:100].all() and not act[1, :30].any()

    m0 = stno_from_activity(act, 0)
    assert m0[TARGET, :30].all()  # S0 혼자
    assert m0[OVERLAP, 30:50].all()  # 둘 다
    assert m0[NON_TARGET, 50:100].all()  # S1 혼자
    assert m0[SILENCE, 100:].all()
    assert np.allclose(m0.sum(axis=0), 1.0)

    m1 = stno_from_activity(act, 1)
    assert m1[NON_TARGET, :30].all() and m1[OVERLAP, 30:50].all() and m1[TARGET, 50:100].all()


def test_build_with_turns_uses_segment_offset_and_longest_speaker_by_default():
    # 발화는 t_cap 10.0 s 에서 시작, S1 이 더 길게 말함 → target 미지정이면 S1
    turns = [SpeakerTurn("S0", 10000, 10400), SpeakerTurn("S1", 10400, 11800)]
    m = build_stno_mask(
        1800, turns=turns, speakers=("S0", "S1"), seg_t_start_ms=10000, n_frames=100
    )
    assert m[NON_TARGET, :20].all()  # S0 구간은 non-target
    assert m[TARGET, 20:90].all()
    assert m[SILENCE, 90:].all()  # 발화 끝 뒤는 silence
    m_s0 = build_stno_mask(
        1800, turns=turns, speakers=("S0", "S1"), target="S0", seg_t_start_ms=10000, n_frames=100
    )
    assert m_s0[TARGET, :20].all() and m_s0[NON_TARGET, 20:90].all()
