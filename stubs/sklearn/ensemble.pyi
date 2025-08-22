from __future__ import annotations

from typing import Sequence
import numpy as np
from numpy.typing import NDArray


class IsolationForest:
    def __init__(
        self,
        n_estimators: int = ...,
        contamination: float | str = ...,
        random_state: int | None = ...,
        **kwargs: object,
    ) -> None: ...
    def fit(
        self,
        X: NDArray[np.float64] | Sequence[Sequence[float]],
        y: Sequence[int] | NDArray[np.int64] | None = ...,
    ) -> IsolationForest: ...
    def decision_function(
        self,
        X: NDArray[np.float64] | Sequence[Sequence[float]],
    ) -> NDArray[np.float64]: ...
    def predict(
        self,
        X: NDArray[np.float64] | Sequence[Sequence[float]],
    ) -> NDArray[np.int64]: ...