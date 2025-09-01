from __future__ import annotations

# Local stubs for scikit-learn (minimal surface used by the project).
# Expose commonly used submodules as attributes of the package.
from . import cluster as cluster
from . import covariance as covariance
from . import ensemble as ensemble
from . import metrics as metrics
from . import preprocessing as preprocessing

__all__ = ["cluster", "preprocessing", "covariance", "metrics", "ensemble"]