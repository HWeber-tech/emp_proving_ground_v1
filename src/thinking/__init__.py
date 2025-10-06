"""
EMP Thinking Layer v1.1

The cognitive layer responsible for pattern recognition, analysis,
inference, and memory in the EMP Ultimate Architecture v1.1.

This layer owns all cognitive functions including:
- Pattern detection (trends, regimes, anomalies)
- Performance analysis and scoring
- Risk analysis and assessment
- Market inference and prediction
- Memory and learning systems
"""

from __future__ import annotations

from contextlib import suppress

with suppress(ModuleNotFoundError):
    from .analysis import *
with suppress(ModuleNotFoundError):
    from .inference import *
with suppress(ModuleNotFoundError):
    from .memory import *
with suppress(ModuleNotFoundError):
    from .models import *
with suppress(ModuleNotFoundError):
    from .patterns import *

__version__ = "1.1.0"
__author__ = "EMP System"
__description__ = "Cognitive layer for pattern recognition and analysis"
