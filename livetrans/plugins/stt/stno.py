"""DiCoW 용 STNO 마스크 (Silence / Target / Non-target / Overlap) 생성.

DiCoW 는 30 s 로그멜 입력과 함께 50 fps 프레임마다 "지금 누가 말하는가"를 4채널 확률로 받는다 (D-019).
구성 방식은 BUTSpeechFIT/DiCoW 의 `pipeline.py` (Apache-2.0) 를 따랐다 — THIRD_PARTY_NOTICES.md.

- silence    : 아무도 말하지 않음
- target     : 대상 화자만 말함
- non_target : 대상 화자는 침묵, 다른 화자가 말함
- overlap    : 대상 화자와 다른 화자가 함께 말함

화자 구간이 없으면(화자 분할 미사용) 발화 전체를 target 으로 둔다 → 일반 Whisper 와 같은 조건.
numpy 만 쓰므로 GPU 없이 단위 테스트할 수 있다.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from livetrans.core.speaker import SpeakerTurn

FPS = 50  # Whisper 인코더 출력 프레임률 (20 ms)
N_FRAMES_30S = 1500
SILENCE, TARGET, NON_TARGET, OVERLAP = 0, 1, 2, 3


def speaker_activity(
    turns: Sequence[SpeakerTurn], speakers: Sequence[str], seg_t_start_ms: int, n_frames: int
) -> np.ndarray:
    """화자별 활동 마스크 [n_spk, n_frames] (0/1). 구간 시각은 t_cap 기준이라 발화 시작으로 옮긴다."""
    act = np.zeros((len(speakers), n_frames), dtype=np.float32)
    index = {s: i for i, s in enumerate(speakers)}
    for t in turns:
        i = index.get(t.speaker)
        if i is None:
            continue
        a = max(0, round((t.t_start_ms - seg_t_start_ms) * FPS / 1000))
        b = min(n_frames, round((t.t_end_ms - seg_t_start_ms) * FPS / 1000))
        if b > a:
            act[i, a:b] = 1.0
    return act


def stno_from_activity(act: np.ndarray, target_index: int) -> np.ndarray:
    """활동 마스크 [n_spk, n_frames] → 대상 화자의 STNO [4, n_frames]."""
    others = np.ones(act.shape[0], dtype=bool)
    others[target_index] = False
    silence = np.prod(1.0 - act, axis=0)
    anyone_else = np.prod(1.0 - act[others], axis=0) if others.any() else np.ones(act.shape[1])
    target = act[target_index] * anyone_else
    non_target = (1.0 - act[target_index]) * (1.0 - anyone_else)
    overlap = act[target_index] - target
    return np.stack([silence, target, non_target, overlap]).astype(np.float32)


def build_stno_mask(
    duration_ms: int,
    *,
    turns: Sequence[SpeakerTurn] = (),
    speakers: Sequence[str] = (),
    target: str | None = None,
    seg_t_start_ms: int = 0,
    n_frames: int = N_FRAMES_30S,
) -> np.ndarray:
    """발화 하나에 대한 STNO 마스크 [4, n_frames].

    - `turns`/`speakers` 가 없으면: 발화 길이만큼 target=1, 나머지(30 s 패딩)는 silence=1.
    - 있으면: `target` 화자 기준 STNO. `target` 이 None 이면 가장 오래 말한 화자를 대상으로 한다.
    """
    n_speech = min(n_frames, max(0, round(duration_ms * FPS / 1000)))
    if not turns or not speakers:
        m = np.zeros((4, n_frames), dtype=np.float32)
        m[TARGET, :n_speech] = 1.0
        m[SILENCE, n_speech:] = 1.0
        return m
    act = speaker_activity(turns, speakers, seg_t_start_ms, n_frames)
    if target is None:
        target_index = int(act.sum(axis=1).argmax())
    else:
        target_index = list(speakers).index(target)
    m = stno_from_activity(act, target_index)
    # 발화 끝 이후(패딩)는 무조건 silence
    m[:, n_speech:] = 0.0
    m[SILENCE, n_speech:] = 1.0
    return m
