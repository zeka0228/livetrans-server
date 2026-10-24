"""화자 레인 세그먼터 (D-028) — 화자 활동 구간 → 발화 경계."""

from livetrans.core.config import VadConfig
from livetrans.pipeline.lanes import LaneSegmenter


def make(**kw) -> LaneSegmenter:
    ids = iter(range(100))
    return LaneSegmenter("S0", VadConfig(**kw), lambda: next(ids))


def test_intervals_within_min_silence_join_into_one_utterance():
    lane = make()  # min_silence 600, pad 200, min_speech 250
    assert lane.push([(1000, 2000)], committed_ms=2100) == []  # 아직 무음 판정 전
    assert lane.open_span == (0, 800, 2000)
    assert lane.push([(2400, 3000)], committed_ms=3100) == []  # 400 ms 틈 → 같은 발화
    utts = lane.push([], committed_ms=3700)  # 700 ms 무음 확정 → 닫힘
    assert len(utts) == 1
    u = utts[0]
    assert (u.seg_id, u.speaker, u.t_start_ms, u.t_end_ms, u.forced_cut) == (
        0,
        "S0",
        800,
        3200,
        False,
    )
    assert [(t.t_start_ms, t.t_end_ms) for t in u.turns] == [(1000, 2000), (2400, 3000)]
    assert u.prev_seg_id is None and lane.open_span is None


def test_gap_longer_than_min_silence_closes_previous_and_opens_new():
    lane = make()
    lane.push([(1000, 2000)], 2000)
    utts = lane.push([(3000, 3500)], 3500)  # 1000 ms 틈
    assert [(u.seg_id, u.t_start_ms, u.t_end_ms) for u in utts] == [(0, 800, 2200)]
    assert lane.open_span == (1, 2800, 3500)


def test_overlapping_or_touching_intervals_merge():
    lane = make()
    lane.push([(1000, 2000), (1500, 2500), (2500, 2600)], 2600)
    assert lane.open_span == (0, 800, 2600)
    assert [(t.t_start_ms, t.t_end_ms) for t in lane.open_turns] == [(1000, 2600)]


def test_too_short_speech_is_dropped():
    lane = make()
    assert lane.push([(1000, 1100)], 1100) == []
    assert lane.push([], 2000) == []  # 100 ms < min_speech 250 → 버림
    assert lane.open_span is None


def test_forced_cut_at_max_utterance_and_tail_keeps_chain():
    lane = make(max_utterance_s=2.0)
    utts = lane.push([(0, 2500)], 2500)
    assert [(u.seg_id, u.t_start_ms, u.t_end_ms, u.forced_cut) for u in utts] == [
        (0, 0, 2000, True)
    ]
    assert lane.open_span == (1, 1800, 2500)  # 이어지는 조각, 새 id
    tail = lane.push([], 3200)
    assert [(u.seg_id, u.prev_seg_id, u.t_start_ms, u.t_end_ms, u.forced_cut) for u in tail] == [
        (1, 0, 1800, 2700, False)
    ]


def test_long_run_is_cut_repeatedly():
    lane = make(max_utterance_s=1.0)
    utts = lane.push([(0, 3300)], 3300)
    assert [(u.seg_id, u.t_start_ms, u.t_end_ms, u.forced_cut) for u in utts] == [
        (0, 0, 1000, True),
        (1, 800, 2000, True),
        (2, 1800, 3000, True),
    ]
    assert utts[1].prev_seg_id == 0 and utts[2].prev_seg_id == 1
    assert lane.open_span == (3, 2800, 3300)


def test_flush_closes_open_utterance():
    lane = make()
    lane.push([(100, 900)], 900)
    utts = lane.flush()
    assert [(u.seg_id, u.t_start_ms, u.t_end_ms) for u in utts] == [(0, 0, 1100)]
    assert lane.flush() == []


def test_short_fragment_merges_into_next_utterance():
    lane = make()  # lane_min_speech 400, lane_merge_gap 1500
    assert lane.push([(1000, 1300)], 1300) == []  # 300 ms 조각
    assert lane.push([], 2000) == []  # 무음 700 ms → 닫히지만 내보내지 않고 보류
    assert lane.open_span is None
    assert lane.push([(2200, 3200)], 3200) == []  # 틈 900 ms ≤ 1500 → 조각을 앞에 붙여 새 발화
    assert lane.open_span[1:] == (800, 3200)
    utts = lane.push([], 4000)
    assert [(u.t_start_ms, u.t_end_ms) for u in utts] == [(800, 3400)]
    assert [(t.t_start_ms, t.t_end_ms) for t in utts[0].turns] == [(1000, 1300), (2200, 3200)]


def test_short_fragment_without_following_speech_is_dropped():
    lane = make()
    lane.push([(1000, 1300)], 1300)
    assert lane.push([], 2000) == []
    assert lane.push([(4000, 5000)], 5000) == []  # 틈 2700 ms > 1500 → 조각 버림
    assert lane.open_span[1:] == (3800, 5000)
    assert lane.flush()[0].turns[0].t_start_ms == 4000


def test_short_fragment_at_end_is_dropped_on_flush():
    lane = make()
    lane.push([(1000, 1300)], 1300)
    assert lane.flush() == []
    assert lane.flush() == []


def test_fragment_threshold_is_configurable():
    lane = make(lane_min_speech_ms=250)  # 예전 동작: 300 ms 도 바로 내보낸다
    lane.push([(1000, 1300)], 1300)
    assert [(u.t_start_ms, u.t_end_ms) for u in lane.push([], 2000)] == [(800, 1500)]
