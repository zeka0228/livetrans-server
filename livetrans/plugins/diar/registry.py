"""세션 화자 등록부 — 증분 클러스터링 (명세 §4.8 "핵심 난점").

발화마다 따로 돌린 화자 분할은 발화마다 라벨이 새로 매겨진다(SPEAKER_00 …). 세션 동안 화자별 임베딩 중심점을 들고
있다가 새 발화의 화자 임베딩을 **가장 비슷한 기존 화자**에 붙이고, 충분히 비슷한 화자가 없으면 새 ID 를 준다.

- 비교는 `transform` 으로 옮긴 공간의 코사인 유사도. pyannote 어댑터는 파이프라인의 x-vector 변환(센터링·LDA·길이 정규화)을
  넘긴다 — 원시 임베딩 코사인보다 같은/다른 화자 간격이 넓다 (2026-10-23 합성음 측정: 평균 0.65 vs 0.30).
- 중심점은 말한 시간으로 가중한 원시 임베딩 평균. 비교할 때마다 변환한다.
- 한 발화의 화자 둘이 같은 세션 화자에 붙지 않도록 (발화 화자, 세션 화자) 쌍을 유사도 높은 순으로 1:1 매칭한다.
- 화자 수가 `max_speakers` 에 닿으면 임계값 미만이라도 가장 가까운 기존 화자에 붙인다.
- 임베딩이 NaN(너무 짧은 발화)이면 직전에 배정한 화자를 다시 쓴다 (이어 말하는 경우가 흔하다는 가정, 미검증).

numpy 만 쓰고 모델에 의존하지 않아 GPU 없이 단위 테스트한다. 디스크에 아무것도 쓰지 않는다 (§13).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

Transform = Callable[[np.ndarray], np.ndarray]


def _unit(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, 1e-9)


class SpeakerRegistry:
    def __init__(
        self,
        *,
        threshold: float,
        max_speakers: int,
        transform: Transform | None = None,
    ) -> None:
        self._threshold = threshold
        self._max = max(1, max_speakers)
        self._transform = transform
        self._sums: list[np.ndarray | None] = []  # 화자별 가중 합 (원시 공간)
        self._weights: list[float] = []
        self._last: str | None = None

    # --- 조회 ---

    @property
    def speakers(self) -> list[str]:
        return [f"S{i}" for i in range(len(self._sums))]

    @property
    def last_speaker(self) -> str | None:
        return self._last

    def reset(self) -> None:
        self._sums.clear()
        self._weights.clear()
        self._last = None

    # --- 배정 ---

    def assign(self, embeddings: np.ndarray, weights: Sequence[float]) -> list[str]:
        """발화 안 화자 임베딩 [n, d] → 세션 화자 ID n개. `weights` 는 각 화자가 말한 시간(초)."""
        emb = np.asarray(embeddings, dtype=np.float32)
        if emb.ndim == 1:
            emb = emb[None]
        n = emb.shape[0]
        if n == 0:
            return []
        valid = ~np.isnan(emb).any(axis=1)
        result: list[str | None] = [None] * n

        # 1) 유효한 임베딩을 기존 화자와 1:1 매칭 (유사도 높은 쌍부터)
        if any(s is not None for s in self._sums) and valid.any():
            rows = np.flatnonzero(valid)
            sims = self._similarity(emb[rows])  # [n_valid, n_session]
            pairs = sorted(
                (
                    (float(sims[i, j]), i, j)
                    for i in range(sims.shape[0])
                    for j in range(sims.shape[1])
                ),
                reverse=True,
            )
            used: set[int] = set()
            for sim, i, j in pairs:
                if sim < self._threshold:
                    break
                if result[rows[i]] is not None or j in used:
                    continue
                result[rows[i]] = f"S{j}"
                used.add(j)
                self._update(j, emb[rows[i]], weights[rows[i]])

        # 2) 못 붙은 유효 임베딩: 자리가 있으면 새 화자, 없으면 가장 가까운 화자
        for i in range(n):
            if result[i] is not None or not valid[i]:
                continue
            empty = next((j for j, s in enumerate(self._sums) if s is None), None)
            if empty is not None:
                j = empty
            elif len(self._sums) < self._max:
                self._sums.append(None)
                self._weights.append(0.0)
                j = len(self._sums) - 1
            else:
                j = int(self._similarity(emb[i : i + 1])[0].argmax())
            result[i] = f"S{j}"
            self._update(j, emb[i], weights[i])

        # 3) NaN 임베딩: 직전 화자, 그것도 없으면 S0 (임베딩 없는 자리)
        for i in range(n):
            if result[i] is None:
                if self._last is None and not self._sums:
                    self._sums.append(None)
                    self._weights.append(0.0)
                result[i] = self._last or "S0"
        self._last = result[-1]
        return [r for r in result if r is not None]

    # --- 내부 ---

    def _update(self, j: int, emb: np.ndarray, weight: float) -> None:
        w = max(float(weight), 1e-3)
        cur = self._sums[j]
        add = emb.astype(np.float64) * w
        self._sums[j] = add if cur is None else cur + add
        self._weights[j] += w

    def _centroids(self) -> np.ndarray:
        dim = next(s.shape[0] for s in self._sums if s is not None)
        rows = []
        for s, w in zip(self._sums, self._weights, strict=True):
            if s is None or w <= 0:
                rows.append(np.full(dim, np.nan, dtype=np.float32))
            else:
                rows.append((s / w).astype(np.float32))
        return np.stack(rows)

    def _similarity(self, emb: np.ndarray) -> np.ndarray:
        cents = self._centroids()
        if self._transform is not None:
            emb, cents = self._transform(emb), self._transform(cents)
        sims = _unit(emb) @ _unit(cents).T
        return np.nan_to_num(sims, nan=-1.0)
