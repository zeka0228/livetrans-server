"""설정 로딩: config/default.yaml 위에 config/default.local.yaml 을 깊은 병합으로 덮어쓴다 (명세 §7)."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

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
    engine: str = "faster_whisper"  # dicow | faster_whisper | fake
    model: str = (
        "large-v3"  # dicow: HF 모델 id (BUT-FIT/DiCoW_v3_3), faster_whisper: 모델명 또는 HF id
    )
    fallback: str | None = None  # 기본 엔진 로드 실패 시 쓸 엔진 (R-15)
    fallback_model: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    beam_final: int = 5
    beam_partial: int = 1
    language: str | None = None


class DiarConfig(BaseModel):
    """화자 분할 (명세 §4.8, D-021). `engine: none` 이면 화자 없이 동작한다."""

    engine: str = "none"  # none | pyannote
    model: str = "pyannote/speaker-diarization-community-1"
    device: str = "cuda"
    max_speakers: int = 4  # 세션당 화자 상한
    max_speakers_per_seg: int = 2  # 구간당 동시 화자 상한 k (명세 Q11)
    match_threshold: float = 0.6  # 기존 화자로 볼 코사인 하한 (pyannote LDA 공간). 합성음 측정 2026-10-23: 같은 화자 0.79, 다른 화자 0.14~0.56
    min_speaker_ms: int = 300  # 구간 안에서 이보다 짧게 말한 화자는 잡음으로 보고 버림
    context_s: float = (
        20.0  # 직전 발화를 이만큼 이어 붙여 함께 분할 (발화 하나만 넣으면 순차 2화자를 합침)
    )


def _default_tts_engines() -> dict[str, dict[str, Any]]:
    return {"system": {"type": "system"}, "openjtalk": {"type": "openjtalk"}}


def _default_tts_voices() -> dict[str, str]:
    return {"ko-KR": "system:auto", "en-US": "system:auto", "ja-JP": "openjtalk:default"}


class TtsConfig(BaseModel):
    """엔진 인스턴스와 언어별 기본 음성 (명세 §5.4 를 default.yaml 의 tts 섹션으로)."""

    engines: dict[str, dict[str, Any]] = Field(default_factory=_default_tts_engines)
    defaults: dict[str, str] = Field(default_factory=_default_tts_voices)
    max_chars: int = 500
    cache_items: int = 128


class LoggingConfig(BaseModel):
    """세션 기록 (명세 §11). 기본값은 꺼짐 — 저장소 `default.yaml` 이 켠다 (테스트가 파일을 남기지 않게)."""

    dir: str = "logs"
    per_session_jsonl: bool = False  # 세션마다 보낸 이벤트 전부를 logs/<날짜>_<세션>.jsonl 에
    dump_audio: bool = (
        False  # 확정 발화 PCM 을 logs/audio/<날짜>_<세션>/segNNNN.wav 로 (실패 사례 분석용)
    )


class Settings(BaseModel):
    server: ServerConfig = ServerConfig()
    audio: AudioConfig = AudioConfig()
    vad: VadConfig = VadConfig()
    stt: SttConfig = SttConfig()
    diar: DiarConfig = DiarConfig()
    tts: TtsConfig = TtsConfig()
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
