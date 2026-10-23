"""세션 화자 등록부 — 증분 클러스터링 (명세 §4.8 "핵심 난점").

발화마다 따로 돌린 화자 분할은 발화마다 라벨이 새로 매겨진다(SPEAKER_00 …). 세션 동안 화자별 임베딩 중심점을 들고
있다가 새 발화의 화자 임베딩을 **가장 비슷한 기존 화자**에 붙이고, 충분히 비슷한 화자가 없으면 새 ID 를 준다.

- 비교는 `transform` 으로 옮긴 공간의 코사인 유사도. pyannote 어댑터는 파이프라인의 x-vector 변환(센터링·LDA·길이 정규화)을
  넘긴다 — 원시 임베딩 코사인보다 같은/다른 화자 간격이 넓다 (2026-10-23 합성음 측정, 문맥 창 임베딩끼리: 같은 화자 0.92~1.00,
  다른 화자 0.04~0.19. 단독 발화 임베딩 ↔ 문맥 창 임베딩은 같은 화자라도 0.56 까지 떨어진다 → 어댑터가 문맥 연속성을 먼저 쓴다).
- 중심점은 말한 시간으로 가중한 원시 임베딩 평균. 비교할 때마다 변환한다.
- 한 발화의 화자 둘이 같은 세션 화자에 붙지 않도록 (발화 화자, 세션 화자) 쌍을 유사도 높은 순으로 1:1 매칭한다.
  `exclude` 로 "이번 발화에서 이미 다른 방법으로 배정된 화자" 를 빼고 매칭할 수 있다.
- 화자 수가 `max_speakers` 에 닿으면 임계값 미만이라도 가장 가까운 기존 화자에 붙인다.
- 임베딩이 NaN(겹침만 있거나 너무 짧은 발화)이면 직전에 배정한 화자를 다시 쓰되, 같은 발화의 다른 화자와는 겹치지 않게 한다.

numpy 만 쓰고 모델에 의존하지 않아 GPU 없이 단위 테스트한다. 디스크에 아무것도 쓰지 않는다 (§13).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

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
        self._sums: list[
            np.ndarray | None
        ] = []  # 화자별 가중 합 (원시 공간), None = 임베딩 없는 자리
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

    def assign(
        self,
        embeddings: np.ndarray,
        weights: Sequence[float],
        exclude: Iterable[str] = (),
    ) -> list[str]:
        """발화 안 화자 임베딩 [n, d] → 세션 화자 ID n개. `weights` 는 각 화자가 말한 시간(초).

        `exclude` 의 화자는 이번 호출에서 배정하지 않는다 (어댑터가 문맥 연속성으로 이미 배정한 화자).
        """
        emb = np.asarray(embeddings, dtype=np.float32)
        if emb.ndim == 1:
            emb = emb[None]
        n = emb.shape[0]
        if n == 0:
            return []
        valid = ~np.isnan(emb).any(axis=1)
        result: list[str | None] = [None] * n
        used: set[int] = {self._index(s) for s in exclude}  # 이번 호출에서 더 못 쓰는 세션 화자

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
            for sim, i, j in pairs:
                if sim < self._threshold:
                    break
                if result[rows[i]] is not None or j in used:
                    continue
                result[rows[i]] = f"S{j}"
                used.add(j)
                self._update(j, emb[rows[i]], weights[rows[i]])

        # 2) 못 붙은 유효 임베딩: 빈 자리/새 화자, 자리가 없으면 이번 호출에서 안 쓴 화자 중 가장 가까운 것
        for i in range(n):
            if result[i] is not None or not valid[i]:
                continue
            j = self._free_slot(used)
            if j is None:
                cands = [k for k, s in enumerate(self._sums) if k not in used and s is not None]
                if cands:
                    sims = self._similarity(emb[i : i + 1])[0]
                    j = max(cands, key=lambda k: sims[k])
                else:
                    j = self._new_slot()
            result[i] = f"S{j}"
            used.add(j)
            self._update(j, emb[i], weights[i])

        # 3) NaN 임베딩: 직전 화자(이번 호출에서 안 썼으면), 아니면 빈/새 자리, 그것도 없으면 직전 화자
        for i in range(n):
            if result[i] is not None:
                continue
            last = self._index(self._last) if self._last else None
            if last is not None and last not in used:
                j = last
            else:
                j = self._free_slot(used)
                if j is None:
                    j = last if last is not None else self._new_slot()
            result[i] = f"S{j}"
            used.add(j)
        self._last = result[-1]
        return [r for r in result if r is not None]

    def reinforce(self, speaker: str, embedding: np.ndarray, weight: float) -> None:
        """다른 방법(문맥 연속성)으로 이미 정해진 화자의 중심점을 이번 임베딩으로 갱신한다. NaN 은 무시."""
        j = self._index(speaker)
        while len(self._sums) <= j:
            self._new_slot()
        e = np.asarray(embedding, dtype=np.float32)
        if not np.isnan(e).any():
            self._update(j, e, weight)
        self._last = speaker

    # --- 내부 ---

    @staticmethod
    def _index(speaker: str) -> int:
        return int(speaker[1:])

    def _new_slot(self) -> int:
        self._sums.append(None)
        self._weights.append(0.0)
        return len(self._sums) - 1

    def _free_slot(self, used: set[int]) -> int | None:
        empty = next((j for j, s in enumerate(self._sums) if s is None and j not in used), None)
        if empty is not None:
            return empty
        if len(self._sums) < self._max:
            return self._new_slot()
        return None

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
