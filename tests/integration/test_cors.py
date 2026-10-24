"""CORS (API 계약 Q-API-13): 브라우저 앱 출처에서 REST·HLS 를 부를 수 있어야 한다."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from livetrans.core.config import Settings, SttConfig
from livetrans.main import create_app
from livetrans.plugins.tts.fake import FakeTTS

APP_ORIGIN = "http://127.0.0.1:8780"


def _client(cors_origins: list[str], tmp_path: Path) -> TestClient:
    settings = Settings(stt=SttConfig(engine="fake"))
    settings.server.cors_origins = cors_origins
    settings.tts.defaults = {"ko-KR": "fake:auto"}
    settings.output.hls_dir = str(tmp_path)
    app = create_app(settings, tts_engines={"fake": FakeTTS("fake", ("ko-KR",))})
    return TestClient(app)


def test_default_origins_are_flutter_web():
    assert Settings().server.cors_origins == ["http://127.0.0.1:8780", "http://localhost:8780"]


def test_preflight_and_simple_request_from_app_origin(tmp_path):
    with _client([APP_ORIGIN], tmp_path) as c:
        pre = c.options(
            "/api/sessions",
            headers={
                "Origin": APP_ORIGIN,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        assert pre.status_code == 200
        assert pre.headers["access-control-allow-origin"] == APP_ORIGIN
        assert "POST" in pre.headers["access-control-allow-methods"]
        assert "content-type" in pre.headers["access-control-allow-headers"].lower()

        r = c.get("/api/health", headers={"Origin": APP_ORIGIN})
        assert r.status_code == 200
        assert r.headers["access-control-allow-origin"] == APP_ORIGIN


def test_tts_headers_are_exposed(tmp_path):
    with _client([APP_ORIGIN], tmp_path) as c:
        r = c.get("/api/tts", params={"text": "안녕", "lang": "ko"}, headers={"Origin": APP_ORIGIN})
        assert r.status_code == 200
        exposed = {h.strip().lower() for h in r.headers["access-control-expose-headers"].split(",")}
        assert {"x-voice-ref", "x-duration-ms", "x-cache"} <= exposed


def test_hls_uses_same_cors_policy(tmp_path):
    (tmp_path / "abc123").mkdir()
    (tmp_path / "abc123" / "master.m3u8").write_text("#EXTM3U\n", encoding="utf-8")
    with _client([APP_ORIGIN], tmp_path) as c:
        r = c.get("/hls/abc123/master.m3u8", headers={"Origin": APP_ORIGIN})
        assert r.status_code == 200
        assert r.headers["access-control-allow-origin"] == APP_ORIGIN
        assert r.headers["cache-control"] == "no-store"
        other = c.get("/hls/abc123/master.m3u8", headers={"Origin": "http://evil.example"})
        assert "access-control-allow-origin" not in other.headers


def test_unknown_origin_gets_no_cors_headers(tmp_path):
    with _client([APP_ORIGIN], tmp_path) as c:
        r = c.get("/api/health", headers={"Origin": "http://evil.example"})
        assert r.status_code == 200
        assert "access-control-allow-origin" not in r.headers


def test_empty_list_disables_cors(tmp_path):
    with _client([], tmp_path) as c:
        r = c.get("/api/health", headers={"Origin": APP_ORIGIN})
        assert r.status_code == 200
        assert "access-control-allow-origin" not in r.headers
