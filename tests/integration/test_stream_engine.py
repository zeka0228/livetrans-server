"""실제 pyannote 스트리밍 화자 분할 (GPU). 모델·CUDA·SAPI 없으면 skip.

연속 음성: ko(Heami) → 무음 → en(Zira 피치 0.8 배) → 무음 → ko+en 겹침 → 무음. 20 ms 프레임으로 밀어 넣고
확정 구간의 화자가 영역별로 일관되는지 본다 (경계는 ±0.5 s 허용).
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from livetrans.core.config import DiarConfig
from livetrans.plugins.diar.pyannote import PyannoteDiarizerEngine

MODEL = "pyannote/speaker-diarization-community-1"
SR = 16000


def _model_cached() -> bool:
    from huggingface_hub import try_to_load_from_cache

    return isinstance(try_to_load_from_cache(MODEL, "config.yaml"), str)


def _cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


pytestmark = [
    pytest.mark.skipif(sys.platform != "win32", reason="시험 음성을 SAPI 로 만든다"),
    pytest.mark.skipif(not _cuda(), reason="CUDA 없음"),
    pytest.mark.skipif(not _model_cached(), reason=f"{MODEL} 미다운로드"),
]


def _resample(pcm, sr, target=SR):
    t_src = np.arange(len(pcm)) / sr
    t_dst = np.arange(int(len(pcm) * target / sr)) / target
    return np.interp(t_dst, t_src, pcm).astype(np.float32)


@pytest.fixture(scope="module")
async def stream_audio():
    from livetrans.plugins.tts.system import SystemTTS

    tts = SystemTTS()
    await tts.load()
    voices = {v.languages[0][:2]: v.display_name for v in await tts.list_voices() if v.languages}
    if "ko" not in voices or "en" not in voices:
        pytest.skip("SAPI 한국어·영어 음성 없음")
    ko1 = await tts.synthesize("안녕하세요. 실시간 방송 번역기 자막 테스트입니다.", voices["ko"])
    ko2 = await tts.synthesize("오늘 날씨가 정말 좋네요. 산책하기 좋은 날입니다.", voices["ko"])
    en = await tts.synthesize("Hello everyone, this is a subtitle test.", voices["en"])
    await tts.close()
    ko1, ko2 = _resample(ko1.pcm, ko1.sample_rate), _resample(ko2.pcm, ko2.sample_rate)
    en = _resample(_resample(en.pcm, en.sample_rate), int(SR * 0.8))
    gap = np.zeros(SR, dtype=np.float32)
    overlap = int(0.8 * SR)
    mix = np.zeros(len(ko2) + len(en) - overlap, dtype=np.float32)
    mix[: len(ko2)] += ko2
    mix[len(ko2) - overlap :] += en
    audio = np.concatenate([gap, ko1, gap, en, gap, mix, gap, gap])
    t = len(gap)
    regions = {"ko1": (t, t + len(ko1))}
    t += len(ko1) + len(gap)
    regions["en"] = (t, t + len(en))
    t += len(en) + len(gap)
    regions["mix_ko"] = (t, t + len(ko2) - overlap)
    regions["mix_en"] = (t + len(ko2), t + len(mix))
    return audio, {k: (a * 1000 // SR, b * 1000 // SR) for k, (a, b) in regions.items()}


def _dominant(turns, a_ms, b_ms):
    spoken = {}
    for t in turns:
        ov = max(0, min(t.t_end_ms, b_ms) - max(t.t_start_ms, a_ms))
        if ov:
            spoken[t.speaker] = spoken.get(t.speaker, 0) + ov
    return max(spoken, key=spoken.get) if spoken else None


async def test_stream_keeps_speaker_ids_consistent_across_regions(stream_audio):
    audio, regions = stream_audio
    eng = PyannoteDiarizerEngine(
        DiarConfig(engine="pyannote", model=MODEL, window_s=120, hop_s=3, right_context_s=2)
    )
    await eng.load()
    sd = eng.create_stream()
    turns = []
    frame = SR // 50  # 20 ms
    for i in range(0, len(audio), frame):
        sd.push(audio[i : i + frame])
        if sd.ready:
            turns += await sd.step()
    turns += await sd.step(final=True)
    await eng.close()
    assert turns and sd.last_latency_ms > 0
    ko1 = _dominant(turns, *regions["ko1"])
    en = _dominant(turns, *regions["en"])
    mix_ko = _dominant(turns, *regions["mix_ko"])
    mix_en = _dominant(turns, *regions["mix_en"])
    assert ko1 is not None and en is not None and ko1 != en
    assert mix_ko == ko1  # 같은 한국어 목소리 → 같은 세션 ID
    assert mix_en == en  # 겹침 뒤 영어 목소리 → 영어 화자 ID
    # 확정 구간은 단조 증가 경계로 이어져 있고 창 밖 시각이 없다
    assert all(0 <= t.t_start_ms < t.t_end_ms <= len(audio) * 1000 // SR for t in turns)
