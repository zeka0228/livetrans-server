import numpy as np
import pytest

from livetrans.core.config import SttConfig
from livetrans.pipeline.segmenter import SpeechSegment
from livetrans.plugins.stt import create_stt
from livetrans.plugins.stt.dicow import DiCoWSTT
from livetrans.plugins.stt.fake import FakeSTT
from livetrans.plugins.stt.fallback import FallbackSTT
from livetrans.plugins.stt.faster_whisper import FasterWhisperSTT


class FailingSTT:
    name = "failing"
    model_id = "x"
    languages = ("ko",)
    supports_speaker_mask = False

    async def load(self) -> None:
        raise RuntimeError("모델 없음")

    async def transcribe(self, seg, *, lang, prompt, final, speaker=None):
        raise AssertionError("호출되면 안 됨")

    async def close(self) -> None:
        return None


def test_factory_builds_dicow_with_faster_whisper_fallback():
    cfg = SttConfig(
        engine="dicow",
        model="BUT-FIT/DiCoW_v3_3",
        fallback="faster_whisper",
        fallback_model="large-v3",
    )
    stt = create_stt(cfg)
    assert isinstance(stt, FallbackSTT)
    assert isinstance(stt.primary, DiCoWSTT) and stt.primary.model_id == "BUT-FIT/DiCoW_v3_3"
    assert isinstance(stt.fallback, FasterWhisperSTT) and stt.fallback.model_id == "large-v3"
    # 로드 전에는 기본 엔진 정보를 보여준다
    assert stt.name == "dicow" and stt.supports_speaker_mask is True


def test_factory_without_fallback_returns_engine_directly():
    assert isinstance(create_stt(SttConfig(engine="faster_whisper")), FasterWhisperSTT)
    assert isinstance(create_stt(SttConfig(engine="fake", fallback="fake")), FakeSTT)
    with pytest.raises(ValueError):
        create_stt(SttConfig(engine="nope"))


async def test_fallback_switches_when_primary_load_fails():
    fake = FakeSTT()
    stt = FallbackSTT(FailingSTT(), fake)
    await stt.load()
    assert stt.fell_back and stt.active is fake and stt.name == "fake"
    seg = SpeechSegment(seg_id=1, t_start_ms=0, t_end_ms=500, pcm=np.zeros(8000, dtype=np.float32))
    tr = await stt.transcribe(seg, lang="ko", prompt=None, final=True, speaker="S0")
    assert tr.speaker == "S0" and fake.calls[0]["speaker"] == "S0"


async def test_fallback_keeps_primary_when_it_loads():
    primary, fallback = FakeSTT(lang="ja"), FakeSTT()
    stt = FallbackSTT(primary, fallback)
    await stt.load()
    assert not stt.fell_back and stt.active is primary
