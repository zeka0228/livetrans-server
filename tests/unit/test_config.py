from pathlib import Path

from livetrans.core.config import Settings, deep_merge, load_settings


def test_deep_merge_recurses_and_does_not_mutate():
    base = {"a": {"x": 1, "y": 2}, "b": 1}
    override = {"a": {"y": 20, "z": 30}, "c": 3}
    merged = deep_merge(base, override)
    assert merged == {"a": {"x": 1, "y": 20, "z": 30}, "b": 1, "c": 3}
    assert base == {"a": {"x": 1, "y": 2}, "b": 1}


def test_load_settings_applies_local_override(tmp_path: Path):
    (tmp_path / "default.yaml").write_text(
        "server:\n  host: '127.0.0.1'\n  port: 8765\nstt:\n  model: 'large-v3'\n", encoding="utf-8"
    )
    (tmp_path / "default.local.yaml").write_text("server:\n  port: 9000\n", encoding="utf-8")
    s = load_settings(tmp_path)
    assert s.server.port == 9000
    assert s.server.host == "127.0.0.1"
    assert s.stt.model == "large-v3"


def test_load_settings_without_files_uses_defaults(tmp_path: Path):
    s = load_settings(tmp_path)
    assert s == Settings()


def test_repo_default_yaml_is_valid():
    s = load_settings()
    assert s.audio.sample_rate == 16000
    assert s.vad.max_utterance_s == 8.0
