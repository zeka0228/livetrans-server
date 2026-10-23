"""실제 pyannote 화자 분할 (GPU). 모델이 HF 캐시에 없거나 CUDA 가 없으면 skip.

시험 음성은 Windows SAPI(Heami ko / Zira en) — 다른 OS 면 skip. 모델은 HF 이용 조건 동의 + 토큰이 필요하므로
(`scripts/download_models.py --diar`) 서버 실행과 마찬가지로 **캐시만** 쓴다.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from livetrans.core.config import DiarConfig
from livetrans.pipeline.segmenter import SpeechSegment

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
    pytest.mark.skipif(
        not _model_cached(), reason=f"{MODEL} 미다운로드 (scripts/download_models.py --diar)"
    ),
]


def _resample(pcm: np.ndarray, sr: int, target: int = SR) -> np.ndarray:
    t_src = np.arange(len(pcm)) / sr
    t_dst = np.arange(int(len(pcm) * target / sr)) / target
    return np.interp(t_dst, t_src, pcm).astype(np.float32)


def _seg(seg_id: int, pcm: np.ndarray, t_start_ms: int = 0, **kw) -> SpeechSegment:
    return SpeechSegment(
        seg_id=seg_id,
        t_start_ms=t_start_ms,
        t_end_ms=t_start_ms + len(pcm) * 1000 // SR,
        pcm=pcm,
        **kw,
    )


@pytest.fixture(scope="module")
async def speech():
    """ko = Heami, en = Zira 를 **0.8 배 속도로 재생해 낮춘 목소리**.

    Heami·Zira 원음은 둘 다 여성 합성음이라 pyannote segmentation 이 한 화자로 본다 (2026-10-23 측정: LDA 코사인 0.56,
    `num_speakers=2` 를 줘도 1화자). 피치를 낮추면 LDA 코사인 0.14 로 분명히 다른 화자가 된다.
    """
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
    en_low = _resample(_resample(en.pcm, en.sample_rate), int(SR * 0.8))
    return _resample(ko1.pcm, ko1.sample_rate), _resample(ko2.pcm, ko2.sample_rate), en_low


@pytest.fixture(scope="module")
async def engine():
    from livetrans.plugins.diar.pyannote import PyannoteDiarizerEngine

    eng = PyannoteDiarizerEngine(DiarConfig(engine="pyannote", model=MODEL))
    await eng.load()
    yield eng
    await eng.close()


def _mix(a: np.ndarray, b: np.ndarray, overlap_s: float) -> np.ndarray:
    overlap = int(overlap_s * SR)
    mix = np.zeros(len(a) + len(b) - overlap, dtype=np.float32)
    mix[: len(a)] += a
    mix[len(a) - overlap :] += b
    return mix


async def test_same_voice_keeps_its_id_across_utterances(engine, speech):
    ko1, ko2, en = speech
    diar = engine.create()
    t1 = await diar.diarize(_seg(0, ko1))
    assert t1 and {t.speaker for t in t1} == {"S0"}
    assert diar.last_latency_ms > 0
    t2 = await diar.diarize(_seg(1, en, t_start_ms=5000))
    assert {t.speaker for t in t2} == {"S1"}  # 다른 목소리 → 새 화자
    t3 = await diar.diarize(_seg(2, ko2, t_start_ms=10000))
    assert {t.speaker for t in t3} == {"S0"}  # 같은 목소리(다른 문장) → 기존 화자
    assert all(10000 <= t.t_start_ms < t.t_end_ms <= 10000 + len(ko2) * 1000 // SR for t in t3)
    diar.reset()
    assert diar.registry.speakers == []


async def test_sequential_two_voices_split_with_context(engine, speech):
    """문맥 창 덕분에 순차 2화자(0.8 s 겹침) 발화가 갈린다. 첫 발화로 단독 실행하면 한 화자로 합쳐진다 (측정)."""
    ko1, ko2, en = speech
    diar = engine.create()
    await diar.diarize(_seg(0, ko1))  # S0
    await diar.diarize(_seg(1, en, t_start_ms=6000))  # S1
    ko_end = len(ko2) * 1000 // SR
    turns = await diar.diarize(_seg(2, _mix(ko2, en, 0.8), t_start_ms=12000))
    assert diar.last_window_s > 12
    by: dict[str, list] = {}
    for t in turns:
        by.setdefault(t.speaker, []).append(t)
    assert set(by) == {"S0", "S1"}
    # S0(한국어) 는 한국어 구간 안에서 끝나고, S1(영어) 은 영어 구간 근처에서 시작한다 (±500 ms)
    assert max(t.t_end_ms for t in by["S0"]) <= 12000 + ko_end + 500
    assert min(t.t_start_ms for t in by["S1"]) >= 12000 + ko_end - 800 - 500


async def test_simultaneous_two_voices_split_without_context(engine, speech):
    ko1, _, en = speech
    both = np.zeros(max(len(ko1), len(en)), dtype=np.float32)
    both[: len(ko1)] += ko1
    both[: len(en)] += en
    turns = await engine.create().diarize(_seg(0, both, t_start_ms=1000))
    assert {t.speaker for t in turns} == {"S0", "S1"}
    assert any(t.overlap for t in turns)
