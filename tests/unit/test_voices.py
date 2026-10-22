import io
import wave

import numpy as np
import pytest

from livetrans.core.voices import NoVoiceError, VoiceRegistry, normalize_lang
from livetrans.plugins.tts.base import AudioClip
from livetrans.plugins.tts.fake import FakeTTS


@pytest.fixture
async def registry():
    reg = VoiceRegistry(
        {"win": FakeTTS("win", ("ko-KR", "en-US")), "jp": FakeTTS("jp", ("ja-JP",))},
        {"ko-KR": "win:auto", "en": "win:auto", "ja-JP": "jp:ja-JP"},
        cache_items=2,
    )
    await reg.refresh()
    return reg


@pytest.mark.parametrize(
    ("raw", "norm"), [("ja", "ja-JP"), ("KO", "ko-KR"), ("en-GB", "en-GB"), ("zh", "zh")]
)
def test_normalize_lang(raw, norm):
    assert normalize_lang(raw) == norm


async def test_resolve_by_iso_and_bcp47(registry):
    assert registry.resolve("ja").voice_ref == "jp:ja-JP"
    assert registry.resolve("ko-KR").voice_ref == "win:ko-KR"
    # 기본값 키가 "en" 이어도 en-US 로 정규화돼 맞는다
    assert registry.resolve("en").voice_ref == "win:en-US"
    # 정확한 로캘 기본값이 없으면 주 언어가 같은 음성
    assert registry.resolve("en-GB").voice_ref == "win:en-US"


async def test_resolve_explicit_voice_ref_and_errors(registry):
    assert registry.resolve(None, "win:ko-KR").display_name == "Fake ko-KR"
    with pytest.raises(NoVoiceError):
        registry.resolve("zh")
    with pytest.raises(NoVoiceError):
        registry.resolve(None, "nope:x")
    with pytest.raises(NoVoiceError):
        registry.resolve(None)


async def test_falls_back_when_default_engine_missing():
    reg = VoiceRegistry({"jp": FakeTTS("jp", ("ja-JP",))}, {"ja-JP": "gone:auto"})
    await reg.refresh()
    assert reg.resolve("ja").voice_ref == "jp:ja-JP"


async def test_synthesize_caches_lru(registry):
    a = await registry.synthesize("안녕", lang="ko")
    b = await registry.synthesize("안녕", lang="ko")
    assert not a.cached and b.cached and a.wav == b.wav
    await registry.synthesize("하나", lang="ko")
    await registry.synthesize("둘", lang="ko")  # cache_items=2 → "안녕" 밀려남
    c = await registry.synthesize("안녕", lang="ko")
    assert not c.cached
    assert len(registry.engines["win"].calls) == 4


def test_wav_encoding():
    clip = AudioClip(np.array([0.0, 0.5, -1.0, 2.0], dtype=np.float32), 22050)
    with wave.open(io.BytesIO(clip.to_wav())) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()) == (
            22050,
            1,
            2,
            4,
        )
        samples = np.frombuffer(w.readframes(4), dtype="<i2")
    assert samples.tolist() == [0, 16383, -32767, 32767]
