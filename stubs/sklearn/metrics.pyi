from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

def f1_score(
    y_true: Sequence[int] | NDArray[np.int64],
    y_pred: Sequence[int] | NDArray[np.int64],
    average: str = ...,
    labels: Sequence[int] | None = ...,
    sample_weight: Sequence[float] | NDArray[np.float64] | None = ...,
) -> float: ...