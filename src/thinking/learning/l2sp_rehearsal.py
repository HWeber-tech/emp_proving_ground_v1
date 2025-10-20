"""L2-SP regularisation and equity rehearsal planner for FX adaptation (E.2.2).

The roadmap calls for anchoring the FX finetune against the equity starting
point via either Elastic Weight Consolidation or an L2 penalty to the starting
weights (L2-SP).  This module implements the L2-SP path together with a simple
rehearsal scheduler that keeps 20–30% of each minibatch sourced from the
equity market replay.  The helpers intentionally keep framework dependencies
optional so PyTorch, NumPy, or plain Python values can all interoperate.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

try:  # torch is optional at import time
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional dependency guard
    torch = None  # type: ignore[assignment]

try:  # NumPy is also optional in some environments
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - optional dependency guard
    np = None  # type: ignore[assignment]


class _UnsupportedParameterError(TypeError):
    """Raised when parameters contain unsupported value types."""


def _clone_anchor(value: Any) -> Any:
    """Create a detached copy of an anchor parameter."""

    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().clone()
    if np is not None and isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    if isinstance(value, (int, float)):
        return float(value)
    raise _UnsupportedParameterError(
        "Anchor parameters must be tensors, numpy arrays, or numeric scalars",
    )


def _coerce_anchor_for_current(current: Any, anchor: Any) -> Any:
    """Match anchor type/device/dtype to ``current`` for penalty calculation."""

    if torch is not None and isinstance(current, torch.Tensor):
        if not isinstance(anchor, torch.Tensor):
            anchor_tensor = torch.as_tensor(anchor, dtype=current.dtype, device=current.device)
        else:
            anchor_tensor = anchor.to(device=current.device, dtype=current.dtype)
        if current.shape != anchor_tensor.shape:
            raise ValueError("Anchor tensor shape mismatch for parameter")
        return anchor_tensor

    if np is not None and isinstance(current, np.ndarray):
        anchor_array = np.asarray(anchor, dtype=current.dtype)
        if current.shape != anchor_array.shape:
            raise ValueError("Anchor array shape mismatch for parameter")
        return anchor_array

    if isinstance(current, (int, float)):
        return float(anchor)

    raise _UnsupportedParameterError(
        "Current parameters must be tensors, numpy arrays, or numeric scalars",
    )


def _squared_error(current: Any, anchor: Any, *, reduction: str) -> tuple[Any, float]:
    """Return per-parameter penalty and its floating-point magnitude."""

    if reduction not in {"mean", "sum"}:
        raise ValueError("reduction must be 'mean' or 'sum'")

    if torch is not None and isinstance(current, torch.Tensor):
        anchor_tensor = _coerce_anchor_for_current(current, anchor)
        diff = current - anchor_tensor
        penalty_tensor = diff.pow(2).mean() if reduction == "mean" else diff.pow(2).sum()
        scalar = float(penalty_tensor.detach().cpu())
        return penalty_tensor, scalar

    if np is not None and isinstance(current, np.ndarray):
        anchor_array = _coerce_anchor_for_current(current, anchor)
        diff = current - anchor_array
        squared = diff * diff
        penalty_value = float(squared.mean() if reduction == "mean" else squared.sum())
        return penalty_value, penalty_value

    if isinstance(current, (int, float)):
        anchor_value = _coerce_anchor_for_current(current, anchor)
        diff = float(current) - float(anchor_value)
        penalty_value = diff * diff
        return penalty_value, penalty_value

    raise _UnsupportedParameterError(
        "Unsupported parameter type for L2-SP penalty calculation",
    )


@dataclass
class L2SPRegularizer:
    """L2-SP penalty anchored to pre-adaptation weights."""

    anchors: Mapping[str, Any]
    strength: float = 1.0
    reduction: str = "mean"

    def __post_init__(self) -> None:
        if self.strength < 0.0:
            raise ValueError("strength must be non-negative")
        if self.reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'")
        if not isinstance(self.anchors, Mapping):
            raise TypeError("anchors must be a mapping of parameter names to values")

        cloned: dict[str, Any] = {}
        for name, value in self.anchors.items():
            cloned[name] = _clone_anchor(value)
        self.anchors = cloned

    def penalty(
        self,
        current_parameters: Mapping[str, Any],
        *,
        return_breakdown: bool = False,
    ) -> Any | tuple[Any, Mapping[str, float]]:
        """Compute the L2-SP penalty against ``current_parameters``.

        Parameters
        ----------
        current_parameters:
            Mapping of parameter names to current values (tensors, arrays, or
            numeric scalars).  All keys present in the anchors must be provided.
        return_breakdown:
            When ``True`` a tuple ``(penalty, breakdown)`` is returned where
            ``breakdown`` maps parameter names to their contribution after the
            strength multiplier is applied.
        """

        if not isinstance(current_parameters, Mapping):
            raise TypeError("current_parameters must be a mapping")

        missing = set(self.anchors) - set(current_parameters)
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise KeyError(f"Missing parameters for L2-SP penalty: {missing_list}")

        penalty_value: Any | None = None
        breakdown: dict[str, float] | None = {} if return_breakdown else None

        for name, anchor_value in self.anchors.items():
            current_value = current_parameters[name]
            term, scalar = _squared_error(current_value, anchor_value, reduction=self.reduction)
            penalty_value = term if penalty_value is None else penalty_value + term
            if breakdown is not None:
                breakdown[name] = scalar * self.strength

        if penalty_value is None:
            raise ValueError("No parameters were processed for L2-SP penalty")

        penalty_value = penalty_value * self.strength

        if return_breakdown:
            assert breakdown is not None  # narrow type for mypy
            return penalty_value, breakdown
        return penalty_value


@dataclass(frozen=True)
class EquityRehearsalPlan:
    """Batch-level rehearsal allocation keeping equity fraction in range."""

    total_batch_size: int
    equity_batch_size: int
    fx_batch_size: int
    equity_fraction: float
    fx_fraction: float
    target_fraction: float
    fraction_range: tuple[float, float]
    within_range: bool

    def as_dict(self) -> Mapping[str, float]:  # pragma: no cover - trivial getter
        return {
            "total_batch_size": float(self.total_batch_size),
            "equity_batch_size": float(self.equity_batch_size),
            "fx_batch_size": float(self.fx_batch_size),
            "equity_fraction": self.equity_fraction,
            "fx_fraction": self.fx_fraction,
            "target_fraction": self.target_fraction,
            "fraction_lower": self.fraction_range[0],
            "fraction_upper": self.fraction_range[1],
            "within_range": float(self.within_range),
        }


def _validate_fraction_range(value: tuple[float, float]) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError("fraction_range must contain exactly two values")
    lower, upper = float(value[0]), float(value[1])
    if not (0.0 <= lower <= 1.0 and 0.0 <= upper <= 1.0):
        raise ValueError("fraction_range bounds must lie in [0, 1]")
    if lower > upper:
        raise ValueError("fraction_range lower bound cannot exceed upper bound")
    return lower, upper


def plan_equity_rehearsal(
    total_batch_size: int,
    *,
    target_fraction: float = 0.25,
    fraction_range: tuple[float, float] = (0.2, 0.3),
) -> EquityRehearsalPlan:
    """Allocate equity rehearsal samples for an FX adaptation minibatch.

    The equity fraction is clamped into ``fraction_range`` and rounded to the
    nearest feasible integer count so that, whenever possible, the realised
    fraction stays within 20–30% of the batch size.
    """

    if total_batch_size <= 0:
        raise ValueError("total_batch_size must be positive")

    lower, upper = _validate_fraction_range(fraction_range)
    clamped_target = min(max(float(target_fraction), lower), upper)

    proposed = int(round(total_batch_size * clamped_target))
    equity = max(0, min(total_batch_size, proposed))

    if equity == 0 and lower > 0.0:
        equity = max(1, int(math.ceil(total_batch_size * lower)))
    fx = total_batch_size - equity

    actual_fraction = equity / total_batch_size
    within_range = lower - 1e-9 <= actual_fraction <= upper + 1e-9

    if not within_range:
        min_equity = int(math.floor(upper * total_batch_size))
        max_equity = int(math.ceil(lower * total_batch_size))

        if actual_fraction < lower and max_equity <= total_batch_size:
            equity = max_equity
        elif actual_fraction > upper and min_equity >= 0:
            equity = min_equity

        equity = max(0, min(total_batch_size, equity))
        fx = total_batch_size - equity
        actual_fraction = equity / total_batch_size if total_batch_size else 0.0
        within_range = lower - 1e-9 <= actual_fraction <= upper + 1e-9

    fx_fraction = 1.0 - actual_fraction

    return EquityRehearsalPlan(
        total_batch_size=total_batch_size,
        equity_batch_size=equity,
        fx_batch_size=fx,
        equity_fraction=actual_fraction,
        fx_fraction=fx_fraction,
        target_fraction=clamped_target,
        fraction_range=(lower, upper),
        within_range=within_range,
    )


__all__ = [
    "L2SPRegularizer",
    "EquityRehearsalPlan",
    "plan_equity_rehearsal",
]

