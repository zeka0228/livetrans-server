"""화자 배치 보조 함수(core/speaker), 세션 화자 등록부(diar/registry), 화자 분할 팩토리 — GPU·모델 없이."""

import numpy as np
import pytest

from livetrans.core.config import DiarConfig, load_settings
from livetrans.core.speaker import SpeakerTurn, select_speakers, speaker_extent, speaking_ms
from livetrans.plugins.diar import create_diarizer
from livetrans.plugins.diar.fake import FakeDiarizerEngine
from livetrans.plugins.diar.pyannote import PyannoteDiarizerEngine
from livetrans.plugins.diar.registry import SpeakerRegistry


def _vec(*xs: float) -> np.ndarray:
    return np.asarray(xs, dtype=np.float32)


# --- core/speaker 보조 함수 ---


def test_speaking_ms_sums_per_speaker():
    turns = [SpeakerTurn("S0", 0, 1000), SpeakerTurn("S1", 500, 900), SpeakerTurn("S0", 1200, 1500)]
    assert speaking_ms(turns) == {"S0": 1300, "S1": 400}


def test_select_speakers_ranks_by_time_then_first_seen():
    turns = [SpeakerTurn("S2", 0, 300), SpeakerTurn("S0", 300, 1300), SpeakerTurn("S1", 1300, 1600)]
    # S0 1000 ms > S2 300 ms(먼저 등장) = S1 300 ms
    assert select_speakers(turns, 2) == ("S0", "S2")
    assert select_speakers(turns, 1) == ("S0",)
    assert select_speakers(turns, 0) == ("S0",)  # 상한이 0 이어도 최소 1명
    assert select_speakers([], 2) == ()


def test_speaker_extent_spans_first_start_to_last_end():
    turns = [
        SpeakerTurn("S0", 100, 400),
        SpeakerTurn("S1", 300, 900),
        SpeakerTurn("S0", 1000, 1200),
    ]
    assert speaker_extent(turns, "S0") == (100, 1200)
    assert speaker_extent(turns, "S1") == (300, 900)
    assert speaker_extent(turns, "S9") is None


# --- SpeakerRegistry (증분 클러스터링) ---


def test_new_speakers_get_sequential_ids():
    reg = SpeakerRegistry(threshold=0.5, max_speakers=4)
    assert reg.assign(_vec(1, 0, 0), [1.0]) == ["S0"]
    assert reg.assign(_vec(0, 1, 0), [1.0]) == ["S1"]
    assert reg.speakers == ["S0", "S1"] and reg.last_speaker == "S1"


def test_similar_embedding_reuses_existing_speaker():
    reg = SpeakerRegistry(threshold=0.5, max_speakers=4)
    reg.assign(_vec(1, 0, 0), [1.0])
    assert reg.assign(_vec(0.9, 0.1, 0), [1.0]) == ["S0"]
    assert reg.speakers == ["S0"]


def test_two_speakers_in_one_utterance_never_share_an_id():
    reg = SpeakerRegistry(threshold=0.5, max_speakers=4)
    reg.assign(_vec(1, 0, 0), [1.0])
    # 둘 다 S0 와 임계값 이상으로 비슷하지만 1:1 매칭이라 더 비슷한 쪽만 S0 를 받는다
    ids = reg.assign(np.stack([_vec(0.8, 0.6, 0), _vec(1, 0.05, 0)]), [1.0, 1.0])
    assert ids == ["S1", "S0"]


def test_centroid_is_weighted_by_speaking_time():
    reg = SpeakerRegistry(threshold=0.5, max_speakers=4)
    reg.assign(_vec(1, 0, 0), [0.1])  # 아주 짧게 말함
    assert reg.assign(_vec(0.6, 0.8, 0), [10.0]) == ["S0"]  # 길게 말함 → 중심점을 지배
    # 가중 평균이면 중심점 ≈ (0.6, 0.8) 이라 (0, 1) 도 S0. 단순 평균이면 (0.8, 0.4) 라 새 화자가 됐을 것
    assert reg.assign(_vec(0, 1, 0), [1.0]) == ["S0"]


def test_max_speakers_caps_and_attaches_to_nearest():
    reg = SpeakerRegistry(threshold=0.9, max_speakers=2)
    reg.assign(_vec(1, 0, 0), [1.0])
    reg.assign(_vec(0, 1, 0), [1.0])
    # 임계값 미만이지만 자리가 없으므로 가장 가까운 S1 에 붙는다
    assert reg.assign(_vec(0.2, 0.7, 0.7), [1.0]) == ["S1"]
    assert reg.speakers == ["S0", "S1"]


def test_nan_embedding_falls_back_to_last_speaker():
    nan = np.full(3, np.nan, dtype=np.float32)
    reg = SpeakerRegistry(threshold=0.5, max_speakers=4)
    assert reg.assign(nan, [0.1]) == ["S0"]  # 아무도 없으면 S0 자리를 만든다
    assert reg.assign(_vec(1, 0, 0), [1.0]) == ["S0"]  # 비어 있던 S0 자리를 채운다
    assert reg.assign(_vec(0, 1, 0), [1.0]) == ["S1"]
    assert reg.assign(nan, [0.1]) == ["S1"]  # 직전 화자
    # 한 발화 안의 NaN 은 **직전 호출**의 마지막 화자(S1)를 쓴다 — 같은 발화의 다른 라벨(S0)과 합쳐지지 않게
    assert reg.assign(np.stack([_vec(1, 0, 0), nan]), [1.0, 0.1]) == ["S0", "S1"]


def test_reset_forgets_everyone():
    reg = SpeakerRegistry(threshold=0.5, max_speakers=4)
    reg.assign(_vec(1, 0, 0), [1.0])
    reg.reset()
    assert reg.speakers == [] and reg.last_speaker is None
    assert reg.assign(_vec(0, 1, 0), [1.0]) == ["S0"]


def test_transform_is_applied_before_similarity():
    # 변환이 첫 차원만 남기면 (1,0,0) 과 (1,0,5) 는 같은 화자. 원시 공간에서는 다른 화자
    reg = SpeakerRegistry(threshold=0.9, max_speakers=4, transform=lambda x: x[..., :1])
    reg.assign(_vec(1, 0, 0), [1.0])
    assert reg.assign(_vec(1, 0, 5), [1.0]) == ["S0"]
    raw = SpeakerRegistry(threshold=0.9, max_speakers=4)
    raw.assign(_vec(1, 0, 0), [1.0])
    assert raw.assign(_vec(1, 0, 5), [1.0]) == ["S1"]


def test_empty_input_returns_empty():
    assert SpeakerRegistry(threshold=0.5, max_speakers=4).assign(np.zeros((0, 3)), []) == []


# --- 팩토리·설정 ---


def test_create_diarizer_by_engine_name():
    assert create_diarizer(DiarConfig(engine="none")) is None
    assert create_diarizer(DiarConfig(engine="")) is None
    assert isinstance(create_diarizer(DiarConfig(engine="fake")), FakeDiarizerEngine)
    eng = create_diarizer(DiarConfig(engine="pyannote", model="org/model"))
    assert isinstance(eng, PyannoteDiarizerEngine)
    assert eng.name == "pyannote" and eng.model_id == "org/model"
    with pytest.raises(ValueError):
        create_diarizer(DiarConfig(engine="nope"))


def test_pyannote_create_before_load_raises():
    with pytest.raises(RuntimeError):
        PyannoteDiarizerEngine(DiarConfig(engine="pyannote")).create()


def test_repo_default_diar_is_off_until_app_supports_speaker():
    s = load_settings()
    assert s.diar.engine == "none"  # API 계약 S-16: 앱 반영 전까지 speaker 미송신
    assert s.diar.max_speakers_per_seg == 2 and s.diar.max_speakers == 4


def test_two_nan_embeddings_in_one_utterance_get_distinct_ids():
    nan = np.full(3, np.nan, dtype=np.float32)
    reg = SpeakerRegistry(threshold=0.5, max_speakers=4)
    assert reg.assign(np.stack([nan, nan]), [1.0, 1.0]) == [
        "S0",
        "S1",
    ]  # 겹친 2화자가 둘 다 NaN 이어도 합치지 않는다


def test_exclude_skips_speakers_assigned_elsewhere():
    reg = SpeakerRegistry(threshold=0.5, max_speakers=4)
    reg.assign(_vec(1, 0, 0), [1.0])
    # S0 와 거의 같지만 이번 발화에서 S0 는 이미 다른 라벨(문맥 연속성)이 가져갔다 → 새 화자
    assert reg.assign(_vec(0.99, 0.1, 0), [1.0], exclude=["S0"]) == ["S1"]
    # 자리가 꽉 찼을 때도 제외한 화자에는 붙이지 않는다
    full = SpeakerRegistry(threshold=0.9, max_speakers=2)
    full.assign(_vec(1, 0, 0), [1.0])
    full.assign(_vec(0, 1, 0), [1.0])
    assert full.assign(_vec(0.9, 0.3, 0), [1.0], exclude=["S0"]) == ["S1"]


def test_reinforce_updates_centroid_and_ignores_nan():
    reg = SpeakerRegistry(threshold=0.6, max_speakers=4)
    reg.assign(_vec(1, 0, 0), [1.0])
    reg.reinforce("S0", _vec(0, 1, 0), 10.0)  # 문맥 연속성으로 S0 로 정해진 임베딩
    assert reg.last_speaker == "S0"
    assert reg.assign(_vec(0, 1, 0), [1.0]) == ["S0"]  # 중심점이 옮겨가 (0,1,0) 도 S0
    reg.reinforce("S2", np.full(3, np.nan, dtype=np.float32), 1.0)  # 없는 자리도 만들되 NaN 은 무시
    assert reg.speakers == ["S0", "S1", "S2"] and reg.last_speaker == "S2"
    assert reg.assign(_vec(1, 0, 0), [1.0]) == ["S1"]  # 빈 자리 S1 부터 채운다
