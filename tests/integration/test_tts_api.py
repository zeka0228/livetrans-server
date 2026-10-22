import io
import wave

import pytest
from fastapi.testclient import TestClient

from livetrans.core.config import Settings, SttConfig
from livetrans.main import create_app
from livetrans.plugins.tts.fake import FakeTTS


@pytest.fixture
def client():
    settings = Settings(stt=SttConfig(engine="fake"))
    settings.tts.defaults = {"ko-KR": "fake:auto", "ja-JP": "fake:auto"}
    app = create_app(settings, tts_engines={"fake": FakeTTS("fake", ("ko-KR", "ja-JP"))})
    with TestClient(app) as c:
        yield c


def test_voices_list_and_filter(client):
    all_voices = client.get("/api/voices").json()
    assert {v["voice_ref"] for v in all_voices} == {"fake:ko-KR", "fake:ja-JP"}
    ja = client.get("/api/voices", params={"lang": "ja"}).json()
    assert [v["voice_ref"] for v in ja] == ["fake:ja-JP"]
    assert set(ja[0]) == {
        "voice_ref",
        "display_name",
        "languages",
        "styles",
        "supports_rate",
        "license_note",
    }


def test_get_tts_returns_wav_with_headers(client):
    r = client.get("/api/tts", params={"text": "こんにちは", "lang": "ja"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    assert r.headers["x-voice-ref"] == "fake:ja-JP"
    assert r.headers["x-cache"] == "miss"
    with wave.open(io.BytesIO(r.content)) as w:
        assert w.getnframes() / w.getframerate() * 1000 == pytest.approx(
            int(r.headers["x-duration-ms"]), abs=1
        )
    again = client.get("/api/tts", params={"text": "こんにちは", "lang": "ja"})
    assert again.headers["x-cache"] == "hit" and again.content == r.content


def test_post_tts(client):
    r = client.post("/api/tts", json={"text": "안녕하세요", "lang": "ko", "rate": 1.25})
    assert r.status_code == 200 and r.headers["x-voice-ref"] == "fake:ko-KR"


@pytest.mark.parametrize(
    ("params", "status", "code"),
    [
        ({"text": "   ", "lang": "ko"}, 400, "E_TTS_TEXT"),
        ({"text": "가" * 501, "lang": "ko"}, 400, "E_TTS_TEXT"),
        ({"text": "hi", "lang": "ko", "rate": 3}, 400, "E_TTS_PARAM"),
        ({"text": "hello", "lang": "en"}, 404, "E_TTS_NO_VOICE"),
        ({"text": "hello"}, 404, "E_TTS_NO_VOICE"),
        ({"text": "hello", "voice_ref": "nope:x"}, 404, "E_TTS_NO_VOICE"),
    ],
)
def test_tts_errors(client, params, status, code):
    r = client.get("/api/tts", params=params)
    assert r.status_code == status
    assert r.json()["detail"]["code"] == code


def test_health_lists_tts_engines(client):
    assert "tts:fake" in client.get("/api/health").json()["models_loaded"]
