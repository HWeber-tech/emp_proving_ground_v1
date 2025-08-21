from __future__ import annotations

# Local stubs for scikit-learn (minimal surface used by the project).
# Expose commonly used submodules as attributes of the package.
from . import cluster as cluster
from . import preprocessing as preprocessing
from . import covariance as covariance
from . import metrics as metrics
from . import ensemble as ensemble

__all__ = ["cluster", "preprocessing", "covariance", "metrics", "ensemble"]