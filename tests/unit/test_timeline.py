"""모드 A 타임라인·컴포저·WebVTT·오디오 보조 (명세 §3, §4.5) — 순수 로직."""

import numpy as np
import pytest

from livetrans.core.audio import (
    OUT_RATE,
    db_to_gain,
    decimate_48k_to_16k,
    duck_envelope,
    float_to_pcm16_bytes,
    pcm16_bytes_to_float,
    resample_linear,
)
from livetrans.pipeline.timeline import Composer, OutputRing, Timeline, vtt_segment, vtt_timestamp


def test_audio_helpers_roundtrip_and_resample():
    x = np.linspace(-1, 1, 480, dtype=np.float32)
    stereo = np.stack([x, -x], axis=1)
    back = pcm16_bytes_to_float(float_to_pcm16_bytes(stereo), 2)
    assert back.shape == (480, 2) and np.allclose(back, stereo, atol=1e-4)
    assert len(resample_linear(np.ones(22050, dtype=np.float32), 22050, 48000)) == 48000
    assert len(decimate_48k_to_16k(np.ones(4800, dtype=np.float32))) == 1600
    assert db_to_gain(-12) == pytest.approx(0.251, abs=0.001)
    env = duck_envelope(1000, [(300, 600)], 0.25, ramp=100)
    assert env[0] == 1.0 and env[450] == pytest.approx(0.25) and env[999] == 1.0
    assert 0.25 < env[250] < 1.0 and 0.25 < env[650] < 1.0  # 램프


def test_timeline_wall_mapping():
    tl = Timeline(delay_s=10, start_wall=1000.0, segment_s=1.0, holdback_s=3.0)
    assert tl.compose_wall(0) == pytest.approx(1000 + 1 + 10 - 3)
    assert tl.play_wall(2500) == pytest.approx(1012.5)
    assert tl.play_start_wall_ms == 1_010_000
    assert tl.segment_of(2999) == 2 and tl.segment_bounds_ms(3) == (3000, 4000)


def test_output_ring_slices_and_keeps_recent():
    ring = OutputRing(keep_s=1.0)
    ring.push(np.full((OUT_RATE, 2), 0.5, dtype=np.float32))
    ring.push(np.full((OUT_RATE, 2), -0.5, dtype=np.float32))
    assert ring.total_ms == 2000
    seg = ring.slice(1500, 2500)  # 뒤 500 ms 는 아직 없음 → 0
    assert seg.shape == (OUT_RATE, 2) and seg[0, 0] == -0.5 and seg[-1, 0] == 0.0
    assert np.all(ring.slice(0, 500) == 0.0)  # 1 s 보존 → 첫 1 s 는 버려짐


def make_composer(delay=10.0, holdback=3.0, max_lag=4.0):
    tl = Timeline(delay_s=delay, start_wall=0.0, holdback_s=holdback)
    ring = OutputRing(keep_s=60)
    ring.push(np.full((OUT_RATE * 10, 2), 0.4, dtype=np.float32))  # 10 s 원본
    return Composer(tl, ring, duck_db=-12, max_lag_s=max_lag), ring


def test_cue_on_time_is_shown_at_its_own_time():
    comp, _ = make_composer()
    assert comp.add_cue(1, "src", 2300, 3800, "hello", "S0") is None
    seg2 = comp.compose(0)
    assert seg2.cues["src"] == []
    seg = comp.compose(1)  # 1000~2000: 아직
    assert seg.cues["src"] == []
    seg = comp.compose(2)  # 2000~3000: 걸침
    assert [c.text for c in seg.cues["src"]] == ["hello"] and seg.cues["src"][
        0
    ].shown_from_ms == 2300


def test_late_cue_is_shifted_l2_and_very_late_dropped_l3():
    comp, _ = make_composer(max_lag=4.0)
    for _k in range(3):
        comp.compose()  # 0~3 s 는 이미 나감
    ev = comp.add_cue(1, "tgt", 1200, 2000, "late", "S0")
    assert ev is not None and (ev.kind, ev.level, ev.late_ms) == ("subtitle", "L2", 1800)
    seg = comp.compose()  # k=3
    assert [c.text for c in seg.cues["tgt"]] == ["late"]
    assert (
        seg.cues["tgt"][0].shown_from_ms == 3000 and seg.cues["tgt"][0].t_end_ms == 4000
    )  # 최소 1 s
    for _k in range(5):
        comp.compose()  # ~9 s
    ev = comp.add_cue(2, "tgt", 1000, 2000, "too late", "S0")
    assert ev is not None and ev.level == "L3"
    assert comp.stats["subtitle_l3"] == 1 and comp.miss_rate == pytest.approx(1 / 2)


def test_tts_is_mixed_with_ducking_and_same_speaker_clips_do_not_overlap():
    comp, _ = make_composer()
    tone = np.full(OUT_RATE // 2, 0.3, dtype=np.float32)  # 0.5 s
    assert comp.add_tts(1, 2000, tone, "S0") is None
    assert comp.add_tts(2, 2200, tone, "S0") is None  # 같은 화자: 2500 으로 밀림
    assert comp.add_tts(3, 2200, tone, "S1") is None  # 다른 화자: 겹쳐도 됨
    seg = comp.compose(2)  # 2000~3000
    pcm = seg.pcm
    # 0~0.5 s: 원본 0.4 × 덕킹 0.25 + S0 0.3 + S1 0.3(2200 부터) → 0.1 + 0.3 (+0.3)
    assert pcm[1000, 0] == pytest.approx(0.1 + 0.3, abs=0.02)
    assert pcm[OUT_RATE // 4, 0] == pytest.approx(0.1 + 0.3 + 0.3, abs=0.02)
    # 0.5~1.0 s: S0 두 번째 클립(2500~3000) + S1(2200~2700) → 2600 에서 0.1+0.3+0.3, 2800 에서 0.1+0.3
    assert pcm[int(OUT_RATE * 0.6), 0] == pytest.approx(0.7, abs=0.02)
    assert pcm[int(OUT_RATE * 0.8), 0] == pytest.approx(0.4, abs=0.02)
    seg1 = comp.compose(1)  # 지나간 세그먼트를 다시 합성해도 TTS 없음, 원본만
    assert seg1.pcm[100, 0] == pytest.approx(0.4, abs=0.01)


def test_volume_controls_apply():
    comp, _ = make_composer()
    comp.volume["original"] = 0.5
    seg = comp.compose(0)
    assert seg.pcm[100, 0] == pytest.approx(0.2, abs=0.01)
    comp.volume["tts"] = 0.0
    comp.add_tts(1, 1000, np.full(OUT_RATE // 2, 0.3, dtype=np.float32), "S0")
    seg = comp.compose(1)
    assert seg.pcm[100, 0] == pytest.approx(0.2 * 0.251, abs=0.01)  # 덕킹은 되지만 TTS 는 0


def test_vtt_segment_format():
    comp, _ = make_composer()
    comp.add_cue(1, "tgt", 1500, 3200, "안녕하세요\n여러분", "S0")
    comp.add_cue(2, "tgt", 1800, 2600, "hi", None)
    seg = comp.compose(1)
    text = vtt_segment(seg.cues["tgt"], mpegts_offset=126000)
    assert text.startswith("WEBVTT\nX-TIMESTAMP-MAP=MPEGTS:126000,LOCAL:00:00:00.000\n\n")
    assert "00:00:01.500 --> 00:00:03.200\n<v S0>안녕하세요 여러분\n" in text
    assert "00:00:01.800 --> 00:00:02.800\nhi\n" in text
    assert vtt_timestamp(3_723_004) == "01:02:03.004"
    assert vtt_segment([], mpegts_offset=0).count("-->") == 0
