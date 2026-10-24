from fastapi.testclient import TestClient

from livetrans.core.config import Settings, SttConfig
from livetrans.main import create_app


def test_health_shape():
    app = create_app(Settings(stt=SttConfig(engine="fake")), tts_engines={})
    with TestClient(app) as client:
        r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert set(body) == {"status", "gpu", "process", "models_loaded"}
    assert body["process"] is None or body["process"]["peak_rss_mb"] > 0  # 프로세스 RAM
    assert body["models_loaded"] == ["stt:fake:fake"]
    if body["gpu"] is not None:
        assert {"name", "mem_total_mb", "mem_used_mb"} <= set(body["gpu"])
