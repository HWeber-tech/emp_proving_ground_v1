"""Legacy shim for RealTimeLearningEngine removed.

Real-time learning now resolves exclusively through
``src.sentient.learning.real_time_learning_engine``.
"""

from __future__ import annotations

raise ModuleNotFoundError(
    "src.thinking.learning.real_time_learner was removed. Import "
    "RealTimeLearningEngine from src.sentient.learning.real_time_learning_engine "
    "instead."
)

