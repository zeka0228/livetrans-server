import numpy as np

from livetrans.core.config import VadConfig
from livetrans.pipeline.segmenter import PartialRequest, Segmenter, SpeechSegment

SR = 16000


def energy_vad(window: np.ndarray) -> float:
    return 1.0 if float(np.sqrt(np.mean(window**2))) > 0.05 else 0.0


def tone(seconds: float, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


def run(seg: Segmenter, audio: np.ndarray, chunk: int = 320) -> list:
    events = []
    for i in range(0, len(audio), chunk):
        events.extend(seg.push(audio[i : i + chunk]))
    return events


def finals(events: list) -> list[SpeechSegment]:
    return [e for e in events if isinstance(e, SpeechSegment)]


def test_two_utterances_with_padding():
    seg = Segmenter(VadConfig(), energy_vad)
    audio = np.concatenate([silence(1.0), tone(1.0), silence(1.0), tone(0.5), silence(1.0)])
    out = finals(run(seg, audio))
    assert [s.seg_id for s in out] == [0, 1]
    # 시작은 음성 시작보다 speech_pad(200 ms) 앞, 창 단위(32 ms) 오차 허용
    assert abs(out[0].t_start_ms - 800) <= 32
    assert abs(out[0].t_end_ms - 2200) <= 32
    assert abs(out[1].t_start_ms - 2800) <= 32
    assert abs(out[1].t_end_ms - 3700) <= 32
    for s in out:
        assert abs(len(s.pcm) - (s.t_end_ms - s.t_start_ms) * SR // 1000) <= 16
        assert not s.forced_cut and s.prev_seg_id is None


def test_short_blip_is_dropped_and_id_reused():
    seg = Segmenter(VadConfig(), energy_vad)
    audio = np.concatenate([tone(0.1), silence(1.0), tone(1.0), silence(1.0)])
    out = finals(run(seg, audio))
    assert len(out) == 1
    assert out[0].seg_id == 0
    assert out[0].t_start_ms >= 900 - 32


def test_forced_cut_at_max_length_picks_low_energy_point():
    cfg = VadConfig(max_utterance_s=3.0)
    seg = Segmenter(cfg, energy_vad)
    # 2.3~2.4 s 구간에 작은 진폭(VAD 는 음성으로 보지만 에너지는 낮음)
    speech = np.concatenate([tone(2.3), tone(0.1, amp=0.06), tone(2.0)])
    out = finals(run(seg, np.concatenate([speech, silence(1.0)])))
    assert out[0].forced_cut is True
    assert 2300 <= out[0].t_end_ms <= 2400
    assert out[1].prev_seg_id == out[0].seg_id
    assert out[1].t_start_ms == out[0].t_end_ms
    assert out[1].forced_cut is False
    assert all(s.t_end_ms - s.t_start_ms <= 3000 for s in out)


def test_flush_closes_open_utterance():
    seg = Segmenter(VadConfig(), energy_vad)
    assert finals(run(seg, tone(1.0))) == []
    out = seg.flush()
    assert len(out) == 1 and out[0].t_start_ms == 0
    assert seg.flush() == []


def test_partial_requests_follow_interval_and_share_id():
    seg = Segmenter(VadConfig(), energy_vad, partial_interval_s=1.0)
    events = run(seg, np.concatenate([tone(2.5), silence(1.0)]))
    partials = [e for e in events if isinstance(e, PartialRequest)]
    final = finals(events)
    # 발화가 열려 있는 동안(꼬리 무음 0.6 s 포함) 1 s 마다: 1, 2, 3 s
    assert len(partials) == 3
    assert {p.seg_id for p in partials} == {final[0].seg_id}
    assert len(partials[1].pcm) > len(partials[0].pcm)


def test_chunking_does_not_change_result():
    audio = np.concatenate([silence(0.5), tone(1.2), silence(0.8), tone(0.7), silence(1.0)])
    a = finals(run(Segmenter(VadConfig(), energy_vad), audio, chunk=320))
    b = finals(run(Segmenter(VadConfig(), energy_vad), audio, chunk=4999))
    assert [(s.t_start_ms, s.t_end_ms) for s in a] == [(s.t_start_ms, s.t_end_ms) for s in b]
