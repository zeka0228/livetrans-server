# 서드파티 모델·소프트웨어 출처 및 라이선스 (Third-Party Notices)

이 프로젝트는 아래 공개 모델과 소프트웨어를 사용한다. **모델 가중치는 저장소에 포함하지 않으며** `scripts/download_models.py` 로 각 배포처에서 받는다 (Obsidian `권리 원칙` 3). 각 항목의 라이선스 조건, 특히 **출처 표기(attribution)** 를 아래와 같이 따른다.

> 라이선스 요약은 프로젝트 운영 참고용이며 법률 자문이 아니다. 원문 라이선스가 우선한다.

## 1. STT

### DiCoW — Diarization-Conditioned Whisper (BUT Speech@FIT)
- 가중치: [`BUT-FIT/DiCoW_v3_MLC`](https://huggingface.co/BUT-FIT/DiCoW_v3_MLC) (기본), [`BUT-FIT/DiCoW_v3_3`](https://huggingface.co/BUT-FIT/DiCoW_v3_3) — **CC BY 4.0** (출처 표기 필수). 기반 모델 OpenAI Whisper large-v3-turbo (MIT).
- 코드: [BUTSpeechFIT/DiCoW](https://github.com/BUTSpeechFIT/DiCoW) — Apache License 2.0. 이 저장소의 `livetrans/plugins/stt/dicow.py` 는 그 저장소의 `pipeline.py` 에 있는 STNO(silence/target/non-target/overlap) 마스크 구성 방식을 참고해 다시 구현했다.
- 저작자: Alexander Polok, Dominik Klement 외 (Brno University of Technology, Speech@FIT).
- 변경 사항: 가중치는 수정 없이 사용. 추론 코드는 이 프로젝트의 스트리밍 파이프라인에 맞게 재작성.
- 저작자가 요청한 인용:

```bibtex
@INPROCEEDINGS{polok2026sedicow,
  author={Polok, Alexander and Klement, Dominik and Cornell, Samuele and Wiesner, Matthew and Černocký, Jan and Khudanpur, Sanjeev and Burget, Lukáš},
  booktitle={ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  title={SE-DiCoW: Self-Enrolled Diarization-Conditioned Whisper},
  year={2026},
}

@article{POLOK2026101841,
    title = {DiCoW: Diarization-conditioned Whisper for target speaker automatic speech recognition},
    journal = {Computer Speech & Language},
    volume = {95},
    pages = {101841},
    year = {2026},
    issn = {0885-2308},
    doi = {https://doi.org/10.1016/j.csl.2025.101841},
    url = {https://www.sciencedirect.com/science/article/pii/S088523082500066X},
    author = {Alexander Polok and Dominik Klement and Martin Kocour and Jiangyu Han and Federico Landini and Bolaji Yusuf and Matthew Wiesner and Sanjeev Khudanpur and Jan Černocký and Lukáš Burget},
}

@INPROCEEDINGS{10887683,
  author={Polok, Alexander and Klement, Dominik and Wiesner, Matthew and Khudanpur, Sanjeev and Černocký, Jan and Burget, Lukáš},
  booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  title={Target Speaker ASR with Whisper},
  year={2025},
  pages={1-5},
  doi={10.1109/ICASSP49660.2025.10887683}
}
```

- `DiCoW_v3_MLC` 는 MLC-SLM 챌린지용 미세조정 모델이다 (Polok et al., "BUT System for the MLC-SLM Challenge", arXiv:2506.13414). 일본어·한국어·영어를 포함한 다국어 대화 데이터로 학습.
- 참고: DiCoW 공식 데모가 쓰는 화자 분할 모델 [`BUT-FIT/diarizen-wavlm-large-s80-mlc`](https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-mlc) 는 **CC BY-NC 4.0 (비상업 전용)** 이다. 이 프로젝트는 기본 화자 분할로 pyannote(아래)를 쓰고, Diarizen 을 쓰는 경우 비상업 용도에 한한다.

### faster-whisper / CTranslate2 / OpenAI Whisper
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (MIT), [CTranslate2](https://github.com/OpenNMT/CTranslate2) (MIT), 가중치 [`Systran/faster-whisper-large-v3`](https://huggingface.co/Systran/faster-whisper-large-v3) (MIT, OpenAI Whisper large-v3 변환본).
- OpenAI Whisper: Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision" (arXiv:2212.04356), MIT.

### kotoba-whisper v2.0 (Kotoba Technologies) — 일본어 대안
- 가중치: [`kotoba-tech/kotoba-whisper-v2.0`](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0) — Apache License 2.0. faster-whisper 변환본 [`kotoba-tech/kotoba-whisper-v2.0-faster`](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0-faster) — MIT.
- Whisper large-v3 를 교사로 ReazonSpeech 로 증류한 일본어 전용 모델. Distil-Whisper(Gandhi et al., arXiv:2311.00430) 방식.

## 2. 화자 분할 (Diarization)

### pyannote speaker-diarization-community-1
- 파이프라인: [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1) — **CC BY 4.0** (출처 표기 필수). 라이브러리 [pyannote.audio](https://github.com/pyannote/pyannote-audio) — MIT.
- Hugging Face 에서 이용 조건 동의 후 토큰으로 받는다 (서버 실행 자체는 오프라인).
- 저작자가 요청한 인용:
  - 화자 분할: A. Plaquet, H. Bredin, "Powerset multi-class cross entropy loss for neural speaker diarization", INTERSPEECH 2023.
  - 화자 임베딩: H. Wang et al., "Wespeaker: A research and production oriented speaker embedding learning toolkit", ICASSP 2023.
  - 클러스터링: F. Landini et al., "Bayesian HMM clustering of x-vector sequences (VBx) in speaker diarization", Computer Speech & Language, 2022.

## 3. 오디오 전단 (분석 트랙)

### Bandit v2 — Cinematic Audio Source Separation (대사/음악/효과음 분리)
- 가중치: Zenodo [10.5281/zenodo.12701995](https://zenodo.org/records/12701995) `checkpoint-multi.ckpt` (다국어, DnR v3 학습) — **CC BY-SA 4.0** (출처 표기 필수, 가중치를 개작해 배포하면 동일 조건). 이 프로젝트는 가중치를 수정·재배포하지 않는다.
- 코드: [kwatcharasupat/bandit-v2](https://github.com/kwatcharasupat/bandit-v2) — Apache License 2.0.
- 인용: K. N. Watcharasupat, C.-W. Wu, I. Orife, "Remastering Divide and Remaster: A Cinematic Audio Source Separation Dataset with Multilingual Support", arXiv:2407.07275 (2024).
- 저작자는 상업적 이익을 얻은 경우 음악 관련 비영리 단체 기부를 권고한다 (라이선스 조건은 아님).

### DeepFilterNet
- [Rikorose/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) — MIT / Apache License 2.0 (이중 라이선스). Hendrik Schröter 외.
- 인용: H. Schröter et al., "DeepFilterNet2: Towards Real-Time Speech Enhancement on Embedded Devices for Full-Band Audio", arXiv:2205.05474 (2022).

### Silero VAD
- [snakers4/silero-vad](https://github.com/snakers4/silero-vad) — MIT. faster-whisper 에 번들된 ONNX 모델을 스트리밍으로 래핑해 사용.

## 4. TTS
- Windows SAPI 기본 음성 — Windows 에 포함된 음성, 별도 배포 없음.
- [pyopenjtalk-plus](https://github.com/tsukumijima/pyopenjtalk-plus) (MIT) — [Open JTalk](https://open-jtalk.sourceforge.net/) (Modified BSD), HTS 기본 음성(nitech, Creative Commons / BSD 계열) 번들. 엔진 기본 음성만 사용.

## 5. 캡처
- [NAudio](https://github.com/naudio/NAudio) — MIT. `tools/wincap` 캡처 헬퍼가 사용.

---

항목을 추가·교체할 때 이 파일과 Obsidian `기술 명세서` §10 을 같은 PR 에서 갱신한다.
