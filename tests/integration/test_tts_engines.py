"""실제 TTS 엔진 (Windows SAPI, Open JTalk). 쓸 수 없는 환경이면 skip."""

import sys

import numpy as np
import pytest

from livetrans.plugins.tts.system import lcid_to_bcp47, rate_to_sapi


def test_rate_mapping():
    assert rate_to_sapi(1.0) == 0
    assert rate_to_sapi(3.0) == 10
    assert rate_to_sapi(1 / 3) == -10
    assert rate_to_sapi(100) == 10


@pytest.mark.skipif(sys.platform != "win32", reason="Windows 전용")
def test_lcid_to_bcp47():
    assert lcid_to_bcp47("412") == "ko-KR"
    assert lcid_to_bcp47("409;9") == "en-US"
    assert lcid_to_bcp47("zz") is None


@pytest.mark.skipif(sys.platform != "win32", reason="Windows 전용")
async def test_system_tts_synthesizes():
    from livetrans.plugins.tts.system import SystemTTS

    tts = SystemTTS()
    await tts.load()
    voices = await tts.list_voices()
    if not voices:
        pytest.skip("설치된 Windows 음성 없음")
    clip = await tts.synthesize("Hello world. 안녕하세요.", voices[0].display_name)
    await tts.close()
    assert clip.sample_rate == 22050 and clip.pcm.dtype == np.float32
    assert clip.duration_s > 0.5 and np.abs(clip.pcm).max() > 0.05


async def test_openjtalk_synthesizes_and_rate_changes_length():
    pytest.importorskip("pyopenjtalk")
    from livetrans.plugins.tts.openjtalk import OpenJTalkTTS

    tts = OpenJTalkTTS()
    await tts.load()
    assert [v.languages for v in await tts.list_voices()] == [("ja-JP",)]
    normal = await tts.synthesize("今日はいい天気ですね。", "default")
    fast = await tts.synthesize("今日はいい天気ですね。", "default", rate=1.5)
    assert normal.duration_s > 0.8 and np.abs(normal.pcm).max() > 0.05
    assert fast.duration_s < normal.duration_s * 0.85
