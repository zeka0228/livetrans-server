# livetrans-server

실시간 방송 번역기의 백엔드 서버입니다. 외국어 라이브 방송을 최악 처리 시간 이상으로 고정 지연(D초)해 재생하고, 그 사이에 STT → 번역 → TTS를 끝내 자막·번역 음성을 원래 발화 타이밍에 맞춰 내보내는 것이 목표입니다. 재생은 어떤 경우에도 멈추지 않습니다.

- 스택: Python 3.11 · FastAPI · faster-whisper (CTranslate2, CUDA) · Silero VAD
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
