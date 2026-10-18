import pytest

from livetrans.core.clock import SampleClock


def test_advance_returns_start_and_tracks_total():
    c = SampleClock(16000)
    assert c.advance(320) == 0
    assert c.advance(320) == 20
    assert c.samples == 640
    assert c.now_ms == 40


def test_one_second_of_samples_is_1000ms():
    c = SampleClock(16000)
    c.advance(16000)
    assert c.now_ms == 1000


def test_ms_floor_not_round():
    c = SampleClock(16000)
    c.advance(31)  # 1.9375 ms
    assert c.now_ms == 1


def test_conversions_roundtrip_on_frame_boundaries():
    c = SampleClock(16000)
    assert c.ms_to_samples(20) == 320
    assert c.samples_to_ms(320) == 20


def test_invalid_args():
    with pytest.raises(ValueError):
        SampleClock(0)
    with pytest.raises(ValueError):
        SampleClock(16000).advance(-1)
