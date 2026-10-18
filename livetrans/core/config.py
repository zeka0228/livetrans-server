"""설정 로딩: config/default.yaml 위에 config/default.local.yaml 을 깊은 병합으로 덮어쓴다 (명세 §7)."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

# 저장소 루트/config
DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
ENV_CONFIG_DIR = "LIVETRANS_CONFIG_DIR"


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8765


class AudioConfig(BaseModel):
    sample_rate: int = 16000


class VadConfig(BaseModel):
    threshold: float = 0.5
    min_speech_ms: int = 250
    min_silence_ms: int = 600
    speech_pad_ms: int = 200
    max_utterance_s: float = 8.0


class SttConfig(BaseModel):
    engine: str = "faster_whisper"
    model: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    beam_final: int = 5
    beam_partial: int = 1
    language: str | None = None


class LoggingConfig(BaseModel):
    dir: str = "logs"
    per_session_jsonl: bool = True


class Settings(BaseModel):
    server: ServerConfig = ServerConfig()
    audio: AudioConfig = AudioConfig()
    vad: VadConfig = VadConfig()
    stt: SttConfig = SttConfig()
    logging: LoggingConfig = LoggingConfig()


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """dict 는 재귀 병합, 그 외 값은 override 가 이긴다. 입력은 변경하지 않는다."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"설정 파일의 최상위는 매핑이어야 합니다: {path}")
    return data


def load_settings(config_dir: Path | None = None) -> Settings:
    """default.yaml → default.local.yaml 순서로 읽어 병합한다.

    config_dir 가 None 이면 환경 변수 LIVETRANS_CONFIG_DIR, 없으면 저장소의 config/ 를 쓴다.
    """
    if config_dir is None:
        config_dir = Path(os.environ.get(ENV_CONFIG_DIR, DEFAULT_CONFIG_DIR))
    merged = deep_merge(
        _read_yaml(config_dir / "default.yaml"),
        _read_yaml(config_dir / "default.local.yaml"),
    )
    return Settings.model_validate(merged)
