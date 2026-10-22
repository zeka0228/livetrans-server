"""실제 DiCoW 엔진 (GPU). 모델이 HF 캐시에 없거나 CUDA 가 없으면 skip.

시험 음성은 Windows SAPI(Heami ko / Zira en) 로 만든다 — 다른 OS 면 skip.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from livetrans.core.config import SttConfig
from livetrans.core.speaker import SpeakerTurn
from livetrans.pipeline.segmenter import SpeechSegment

MODEL = "BUT-FIT/DiCoW_v3_3"


def _model_cached() -> bool:
    from huggingface_hub import try_to_load_from_cache

    return isinstance(try_to_load_from_cache(MODEL, "config.json"), str)


def _cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


pytestmark = [
    pytest.mark.skipif(sys.platform != "win32", reason="시험 음성을 SAPI 로 만든다"),
    pytest.mark.skipif(not _cuda(), reason="CUDA 없음"),
    pytest.mark.skipif(
        not _model_cached(), reason=f"{MODEL} 미다운로드 (scripts/download_models.py)"
    ),
]


def _resample(pcm: np.ndarray, sr: int, target: int = 16000) -> np.ndarray:
    t_src = np.arange(len(pcm)) / sr
    t_dst = np.arange(int(len(pcm) * target / sr)) / target
    return np.interp(t_dst, t_src, pcm).astype(np.float32)


@pytest.fixture(scope="module")
async def speech():
    from livetrans.plugins.tts.system import SystemTTS

    tts = SystemTTS()
    await tts.load()
    voices = {v.languages[0][:2]: v.display_name for v in await tts.list_voices() if v.languages}
    if "ko" not in voices or "en" not in voices:
        pytest.skip("SAPI 한국어·영어 음성 없음")
    ko = await tts.synthesize("안녕하세요. 실시간 방송 번역기 자막 테스트입니다.", voices["ko"])
    en = await tts.synthesize("Hello everyone, this is a subtitle test.", voices["en"])
    await tts.close()
    return _resample(ko.pcm, ko.sample_rate), _resample(en.pcm, en.sample_rate)


@pytest.fixture(scope="module")
async def engine():
    from livetrans.plugins.stt.dicow import DiCoWSTT

    stt = DiCoWSTT(SttConfig(engine="dicow", model=MODEL))
    await stt.load()
    yield stt
    await stt.close()


async def test_single_speaker_fixed_and_auto_language(engine, speech):
    ko, _ = speech
    seg = SpeechSegment(seg_id=1, t_start_ms=0, t_end_ms=len(ko) * 1000 // 16000, pcm=ko)
    tr = await engine.transcribe(seg, lang="ko", prompt=None, final=True)
    assert "번역기" in tr.text and tr.lang == "ko" and tr.latency_ms > 0 and tr.speaker is None
    auto = await engine.transcribe(seg, lang=None, prompt=None, final=True)
    assert auto.lang == "ko" and auto.lang_prob > 0.5 and "자막" in auto.text
    partial = await engine.transcribe(seg, lang="ko", prompt=None, final=False)
    assert "번역기" in partial.text


async def test_prompt_is_accepted_and_not_echoed(engine, speech):
    ko, _ = speech
    seg = SpeechSegment(seg_id=2, t_start_ms=0, t_end_ms=len(ko) * 1000 // 16000, pcm=ko)
    tr = await engine.transcribe(seg, lang="ko", prompt="직전 문장입니다", final=True)
    assert "번역기" in tr.text and not tr.text.startswith("직전 문장")


async def test_two_speakers_are_separated_by_mask(engine, speech):
    ko, en = speech
    overlap = int(0.8 * 16000)
    mix = np.zeros(len(ko) + len(en) - overlap, dtype=np.float32)
    mix[: len(ko)] += ko
    mix[len(ko) - overlap :] += en
    ko_end = len(ko) * 1000 // 16000
    total = len(mix) * 1000 // 16000
    seg = SpeechSegment(
        seg_id=3,
        t_start_ms=0,
        t_end_ms=total,
        pcm=mix,
        speakers=("S0", "S1"),
        speaker_turns=[SpeakerTurn("S0", 0, ko_end), SpeakerTurn("S1", ko_end - 800, total)],
    )
    s0 = await engine.transcribe(seg, lang=None, prompt=None, final=True, speaker="S0")
    s1 = await engine.transcribe(seg, lang=None, prompt=None, final=True, speaker="S1")
    assert s0.lang == "ko" and "번역기" in s0.text and s0.speaker == "S0"
    assert s1.lang == "en" and "subtitle" in s1.text.lower() and s1.speaker == "S1"
