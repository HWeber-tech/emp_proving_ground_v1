"""Retention gate evaluation for Phase E horizon requirements.

This module codifies the roadmap gate **E.2.3**: equity retention must stay
within the cap for every evaluation horizon (ev1, ev5, ev20, …).  The helper
exposes a small data model so CI or orchestration code can make a binary
promotion decision while still surfacing horizon-level details for reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import math

__all__ = [
    "HorizonRetentionGate",
    "RetentionGateDecision",
    "evaluate_retention_gates",
]


def _canonicalise_event_horizon(value: object) -> str:
    """Normalise roadmap event horizon labels (``ev1``/``1``/``EV05``).

    The retention gates operate over the event horizons called out in the
    roadmap (ev1, ev5, ev20).  The configuration frequently appears in YAML
    where values may be emitted as integers or upper-case strings, so we share a
    single coercion helper that keeps the public API forgiving while ensuring
    downstream consumers only see canonical labels.
    """

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(value):  # defensive guardrail
            raise ValueError("horizon label must be finite")
        integer = int(value)
        if not math.isclose(value, integer, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"event horizon must be an integer: {value!r}")
        if integer <= 0:
            raise ValueError("event horizon index must be positive")
        return f"ev{integer}"

    text = str(value).strip()
    if not text:
        raise ValueError("event horizon label cannot be blank")

    lower = text.lower()
    if lower.startswith("ev"):
        suffix = lower[2:]
        if suffix.isdigit() and int(suffix) > 0:
            return f"ev{int(suffix)}"
        raise ValueError(f"invalid event horizon suffix: {value!r}")

    if lower.isdigit() and int(lower) > 0:
        return f"ev{int(lower)}"

    raise ValueError(f"unsupported event horizon label: {value!r}")


def _coerce_percentage(value: object, *, field: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{field} must be finite")
    return numeric


@dataclass(frozen=True)
class HorizonRetentionGate:
    """Retention observation for a single horizon cap."""

    horizon: str
    cap_pct: float
    observed_pct: float | None

    @property
    def status(self) -> str:
        if self.observed_pct is None:
            return "missing"
        if self.observed_pct <= self.cap_pct + 1e-9:
            return "pass"
        return "fail"

    @property
    def exceeded_pct(self) -> float | None:
        if self.observed_pct is None:
            return None
        if self.observed_pct <= self.cap_pct:
            return 0.0
        return self.observed_pct - self.cap_pct

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "horizon": self.horizon,
            "cap_pct": self.cap_pct,
            "status": self.status,
        }
        if self.observed_pct is not None:
            payload["observed_pct"] = self.observed_pct
            payload["exceeded_pct"] = max(self.observed_pct - self.cap_pct, 0.0)
        return payload


@dataclass(frozen=True)
class RetentionGateDecision:
    """Aggregated retention gate outcome across all configured horizons."""

    passed: bool
    gates: tuple[HorizonRetentionGate, ...]
    missing_horizons: tuple[str, ...]
    extra_horizons: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "gates": [gate.as_dict() for gate in self.gates],
            "missing_horizons": list(self.missing_horizons),
            "extra_horizons": list(self.extra_horizons),
        }


def _normalise_caps(caps: Mapping[object, object]) -> tuple[list[str], dict[str, float]]:
    order: list[str] = []
    normalised: dict[str, float] = {}
    for horizon, cap in caps.items():
        canonical = _canonicalise_event_horizon(horizon)
        cap_value = _coerce_percentage(cap, field=f"cap for {canonical}")
        if canonical in normalised:
            if abs(normalised[canonical] - cap_value) > 1e-9:
                raise ValueError(
                    f"duplicate cap for horizon {canonical}: values {normalised[canonical]!r} and {cap!r}"
                )
            continue
        if cap_value < 0.0:
            raise ValueError("retention cap percentages must be non-negative")
        order.append(canonical)
        normalised[canonical] = cap_value
    if not order:
        raise ValueError("at least one retention cap must be supplied")
    return order, normalised


def _normalise_observations(observed: Mapping[object, object | None]) -> dict[str, float | None]:
    normalised: dict[str, float | None] = {}
    for horizon, drop in observed.items():
        canonical = _canonicalise_event_horizon(horizon)
        if canonical in normalised:
            raise ValueError(f"duplicate retention observation for horizon {canonical}")
        if drop is None:
            normalised[canonical] = None
            continue
        drop_value = _coerce_percentage(drop, field=f"drop for {canonical}")
        normalised[canonical] = drop_value
    return normalised


def evaluate_retention_gates(
    observed_drops_pct: Mapping[object, object | None],
    cap_pct: Mapping[object, object],
) -> RetentionGateDecision:
    """Evaluate equity retention drops against per-horizon caps.

    Parameters
    ----------
    observed_drops_pct:
        Mapping of horizon label to observed retention drop percentage.  The
        helper accepts canonical strings (``"ev1"``) as well as integers (``1``)
        for convenience.  ``None`` marks a missing measurement and fails the cap.
    cap_pct:
        Mapping of horizon label to allowed drop percentage.

    Returns
    -------
    RetentionGateDecision
        Rich outcome describing whether all horizons passed and, if not, which
        horizons violated their cap.
    """

    order, normalised_caps = _normalise_caps(cap_pct)
    normalised_observed = _normalise_observations(observed_drops_pct)

    gates: list[HorizonRetentionGate] = []
    failing = False

    for horizon in order:
        cap_value = normalised_caps[horizon]
        observed = normalised_observed.get(horizon)
        gate = HorizonRetentionGate(horizon=horizon, cap_pct=cap_value, observed_pct=observed)
        gates.append(gate)
        if gate.status != "pass":
            failing = True

    missing = tuple(sorted(h for h in normalised_caps if h not in normalised_observed))
    extra = tuple(sorted(h for h in normalised_observed if h not in normalised_caps))

    passed = not failing and not missing

    return RetentionGateDecision(
        passed=passed,
        gates=tuple(gates),
        missing_horizons=missing,
        extra_horizons=extra,
    )
