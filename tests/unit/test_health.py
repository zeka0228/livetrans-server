from fastapi.testclient import TestClient

from livetrans.core.config import Settings, SttConfig
from livetrans.main import create_app


def test_health_shape():
    app = create_app(Settings(stt=SttConfig(engine="fake")))
    with TestClient(app) as client:
        r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert set(body) == {"status", "gpu", "models_loaded"}
    assert body["models_loaded"] == ["stt:fake:fake"]
    if body["gpu"] is not None:
        assert {"name", "mem_total_mb", "mem_used_mb"} <= set(body["gpu"])
