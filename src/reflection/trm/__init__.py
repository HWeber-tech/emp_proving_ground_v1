"""Tiny Recursive Model (TRM) production runner utilities."""

from .ab_validator import SurrogateABValidationResult, validate_surrogate_alignment
from .config import AutoApplySettings, ModelConfig, RIMRuntimeConfig, TRMParams, load_runtime_config
from .runner import TRMRunResult, TRMRunner

__all__ = [
    "ModelConfig",
    "RIMRuntimeConfig",
    "TRMParams",
    "TRMRunResult",
    "TRMRunner",
    "load_runtime_config",
    "AutoApplySettings",
    "SurrogateABValidationResult",
    "validate_surrogate_alignment",
]
