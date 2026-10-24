# livetrans-server

실시간 방송 번역기의 백엔드 서버입니다. 외국어 라이브 방송을 최악 처리 시간 이상으로 고정 지연(D초)해 재생하고, 그 사이에 STT → 번역 → TTS를 끝내 자막·번역 음성을 원래 발화 타이밍에 맞춰 내보내는 것이 목표입니다. 재생은 어떤 경우에도 멈추지 않습니다.

- 스택: Python 3.11 · FastAPI · DiCoW (PyTorch · transformers) / faster-whisper (CTranslate2) · Silero VAD
- 프론트엔드(Flutter)는 별도 저장소입니다.

## 설계 문서

설계·결정·이력 문서는 저장소가 아니라 메인테이너의 Obsidian 볼트(`실시간 방송 번역기/`)에서 관리합니다. 이 README는 실행 방법만 다룹니다.

## 실행

요구 사항: Windows, NVIDIA GPU(CUDA 12 드라이버), [uv](https://docs.astral.sh/uv/). PC 소리 캡처(모드 B)를 쓰려면 [.NET SDK 8+](https://dotnet.microsoft.com/download) 도 필요합니다 (`setup.bat` 이 `tools/wincap` 캡처 헬퍼를 빌드).

```bat
scripts\setup.bat
scripts\run_server.bat
```

기본 주소는 `http://127.0.0.1:8765` 이며 `GET /api/health` 로 상태를 확인합니다. 개인 설정은 `config/default.local.yaml` 에 두면 `config/default.yaml` 위에 덮어써집니다 (gitignore).

브라우저 앱(Flutter 웹)이 REST(`/api/*`)·HLS(`/hls/*`) 를 부를 수 있도록 `server.cors_origins`(기본 `http://127.0.0.1:8780`, `http://localhost:8780`) 출처에 CORS 를 허용합니다. 앱을 다른 포트에서 띄우면 `config/default.local.yaml` 의 `server.cors_origins` 에 추가하고, 빈 목록이면 CORS 를 끕니다. WebSocket 은 CORS 대상이 아닙니다.

### 모델 내려받기

STT 모델 가중치는 저장소에 없고 첫 실행 전에 받아야 합니다 (받지 않으면 서버 기동 시 Hugging Face 에서 자동으로 받느라 오래 걸립니다).

```bat
uv run python scripts\download_models.py            # DiCoW v3.3 + faster-whisper large-v3 (기본 STT 세트)
uv run python scripts\download_models.py --all      # + kotoba-whisper(일본어 대안), pyannote 화자 분할(토큰 필요), Bandit v2 분리 모델
```

- 기본 STT 는 **DiCoW** (화자 조건부 Whisper, `stt.engine: dicow`) 이고 로드에 실패하면 `stt.fallback` 의 faster-whisper large-v3 로 자동 전환됩니다. 엔진·모델은 `config/default.local.yaml` 의 `stt` 섹션으로 바꿉니다.
- **화자 분할**(pyannote `speaker-diarization-community-1`, CC BY 4.0)은 기본 **꺼져** 있습니다 (`diar.engine: none`). 켜려면 Hugging Face 에서 모델 이용 조건에 동의하고 토큰을 설정한 뒤(`hf auth login` 또는 `HF_TOKEN`) `uv run python scripts\download_models.py --diar` 로 받고, `config/default.local.yaml` 에 `diar: {engine: pyannote}` 를 둡니다. 켜지면 `segment.final` 에 `speaker`(`"S0"`, `"S1"`…) 가 붙고, 한 발화에 화자가 둘이면 같은 `seg_id` 로 화자별 final 이 하나씩 옵니다 (DiCoW 가 화자마다 따로 인식). 화자 임베딩은 세션 메모리에만 두고 저장하지 않습니다.
- **번역**은 로컬 [Ollama](https://ollama.com) 의 LLM(`mt.engine: ollama_llm`, 기본 `gemma2:9b`)으로 합니다. Ollama 가 떠 있고 모델이 받아져 있어야 하며(`ollama pull gemma2:9b`, VRAM 약 6 GB), 없으면 자동으로 원문 그대로(`passthrough`) 내보냅니다. 고유명사는 `config/glossary.yaml`(개인은 `glossary.local.yaml`)에 고정 번역으로 적습니다.
- **모드 A (동기 시청)** 는 ffmpeg 가 필요합니다 (PATH, `tools.ffmpeg`, 또는 `~/Downloads/ffmpeg*/bin` 자동 탐색). 세션 요청 `mode: "A"` + `source.type: "obs_stream"`(OBS → SRT/RTMP) 또는 `"file"`(테스트) 로 만들면 `hls_url` 이 돌아오고, 서버가 D 초 지연된 영상·원본 소리 + 번역 음성(덕킹) + 자막(WebVTT 번역/원문 트랙)을 HLS 로 내보냅니다. 테스트 영상은 `uv run python -X utf8 scripts/make_test_media.py` 로 만들고 `scripts/mode_a_e2e.py` 로 실서버를 점검합니다. 앱 없이 브라우저에서 바로 보려면 `uv run python -X utf8 scripts\mode_a_browser_demo.py runtime	est_media	est_broadcast.mp4 20` (세션을 만들고 `playing` 이 되면 hls.js 페이지 `scripts/hls_view.html` 을 기본 브라우저로 엽니다. `srt://…` 를 주면 OBS 입력도 됩니다). OBS 입력(SRT listener)은 `control start` 뒤 OBS 가 붙어 첫 소리가 올 때까지 `starting` 에 머물고 지연 D 는 그 순간부터 셉니다 (`timeline.input_wait_s`, 기본 90 s 안에 소리가 없으면 `E_CAPTURE_START`). HLS 산출물은 `runtime/hls/<세션>` 에 잠깐 있다가 세션 종료 2분 뒤 지워집니다.
- 사용하는 모델·소프트웨어의 출처와 라이선스(출처 표기 의무 포함)는 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 에 있습니다.

테스트:

```bat
uv run pytest
```

## Rights & Responsible Use

- 이 프로젝트는 개인 시청 보조 도구입니다. 번역 결과물을 업로드·재배포하는 기능은 제공하지 않습니다.
- 타인(스트리머·성우·캐릭터)의 목소리를 복제하는 기능이나 모델을 포함하지 않습니다. 기본 음성은 OS·엔진이 기본 제공하는 음성입니다.
- 사용자가 음성 모델을 연결할 때는 본인이 사용 권리를 가진 것만 사용해야 합니다. 음성 모델·엔진은 저장소에 번들하지 않습니다.
- 플랫폼 스트림을 직접 내려받는 입력 방식은 기본값이 아닙니다. 기본 입력은 화면·시스템 소리 캡처입니다.

## 라이선스

Apache License 2.0
