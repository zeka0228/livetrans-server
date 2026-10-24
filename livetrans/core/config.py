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
    # 브라우저 앱(Flutter 웹)이 REST·HLS 를 부를 수 있게 허용할 출처. 빈 목록이면 CORS 미들웨어를 켜지 않는다.
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://127.0.0.1:8780", "http://localhost:8780"]
    )


class AudioConfig(BaseModel):
    sample_rate: int = 16000


class VadConfig(BaseModel):
    threshold: float = 0.5
    min_speech_ms: int = 250
    min_silence_ms: int = 600
    speech_pad_ms: int = 200
    max_utterance_s: float = 8.0
    # 화자 레인(D-028) 경계 조각: 이보다 짧게 말한 발화는 바로 내보내지 않고 같은 화자의 다음 발화(틈 ≤ lane_merge_gap_ms)에
    # 붙인다. 다음 발화가 없으면 버린다. 2026-10-25: 0.4 s 조각("sigum")이 그대로 번역·TTS 까지 가던 문제
    lane_min_speech_ms: int = 400
    lane_merge_gap_ms: int = 1500


class SttConfig(BaseModel):
    engine: str = "faster_whisper"  # dicow | faster_whisper | fake
    model: str = (
        "large-v3"  # dicow: HF 모델 id (BUT-FIT/DiCoW_v3_3), faster_whisper: 모델명 또는 HF id
    )
    fallback: str | None = None  # 기본 엔진 로드 실패 시 쓸 엔진 (R-15)
    fallback_model: str = "large-v3"
    rescue: str | None = (
        None  # 환각 판정 시 그 발화만 다시 인식할 구조 엔진 (기본 엔진과 함께 로드, VRAM 추가)
    )
    rescue_model: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    beam_final: int = 5
    beam_partial: int = 1
    language: str | None = None


class DiarConfig(BaseModel):
    """화자 분할 (명세 §4.8, D-021). `engine: none` 이면 화자 없이 동작한다."""

    engine: str = "none"  # none | pyannote
    mode: str = "stream"  # stream: 연속 오디오 + 긴 창 + 화자 레인 (D-028) | utterance: 발화 단위 + 문맥 창 (D-027)
    window_s: float = 120.0  # stream: 창 길이 (실방송 시뮬레이션 120 s = 오프라인 대비 98.2 % 일치)
    hop_s: float = 3.0  # stream: 창 실행 간격
    right_context_s: float = 2.0  # stream: 확정을 미루는 오른쪽 문맥 (C_diar)
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


class TimelineConfig(BaseModel):
    """모드 A 시간 모델 (명세 §3). D 는 세션 요청의 delay_s 가 없을 때 기본값."""

    segment_s: float = 1.0
    holdback_s: float = 3.0  # H: 플레이어가 라이브 끝에서 떨어져 재생하는 버퍼
    guard_s: float = 1.0  # G
    default_delay_s: int = 36  # D 기본 (D-022 최악 가정). 테스트는 요청에서 작게
    ring_extra_s: float = 30.0  # 원본 출력 링버퍼 여유
    # control start 뒤 첫 입력 소리가 이 시간 안에 안 오면 E_CAPTURE_START 로 끝낸다. SRT listener 는 OBS 가 붙을 때까지
    # 소리가 없으므로, 타임라인(D)은 첫 소리가 도착한 순간부터 센다 (ERR-011)
    input_wait_s: float = 90.0
    max_lag_s: float = (
        8.0  # 결과가 이보다 늦으면 버림 (L3). L_max(8 s) 발화가 D 부족 시에도 버려지지 않게
    )


class ToolsConfig(BaseModel):
    ffmpeg: str | None = None  # 없으면 PATH → ~/Downloads/ffmpeg*/bin 순으로 찾는다


class OutputConfig(BaseModel):
    hls_dir: str = "runtime/hls"  # gitignore. 세션마다 하위 디렉터리, 세션 종료 후 삭제
    video_codec: str = (
        "copy"  # copy | h264_nvenc | libx264 (copy 는 입력 키프레임 간격으로 세그먼트가 잘린다)
    )


class TtsConfig(BaseModel):
    """엔진 인스턴스와 언어별 기본 음성 (명세 §5.4 를 default.yaml 의 tts 섹션으로)."""

    engines: dict[str, dict[str, Any]] = Field(default_factory=_default_tts_engines)
    defaults: dict[str, str] = Field(default_factory=_default_tts_voices)
    max_chars: int = 500
    cache_items: int = 128
    max_rate: float = 1.25  # 모드 A: TTS 가 슬롯보다 길면 이 배속까지 빠르게 (§3.6)
    max_overflow_s: float = 1.0  # 발화 길이보다 이만큼은 넘쳐도 그대로
    duck_db: float = -12.0  # TTS 가 나는 동안 원본 음량


class MtConfig(BaseModel):
    """번역 (명세 §4.4). 기본 Ollama 로컬 LLM, 실패하면 체인 → 마지막은 항상 원문(passthrough)."""

    engine: str = "passthrough"  # ollama_llm | passthrough. 모델 기본값은 passthrough(테스트가 네트워크를 안 타게), 저장소 default.yaml 이 ollama_llm 으로 켠다
    model: str = "gemma2:9b"  # 2026-10-23 측정: 190~440 ms/문장(서버와 함께 100 % GPU, 6.3 GB), 한국어 자연스러움. qwen2.5:7b 는 한국어 출력 중 중국어로 샘
    url: str = "http://127.0.0.1:11434"
    fallback: list[str] = Field(default_factory=lambda: ["passthrough"])
    context_pairs: int = 3  # 화자별 직전 (원문, 번역) 쌍 수
    timeout_s: float = 6.0  # 이 시간 안에 못 오면 다음 엔진
    keep_alive: str = "30m"
    temperature: float = 0.1
    max_tokens: int = 200
    glossary: str | None = "config/glossary.yaml"  # + glossary.local.yaml (gitignore)


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
    mt: MtConfig = MtConfig()
    timeline: TimelineConfig = TimelineConfig()
    tools: ToolsConfig = ToolsConfig()
    output: OutputConfig = OutputConfig()
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
