"""Decision latency baseline evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, MutableMapping

__all__ = ["DecisionLatencyBaseline", "evaluate_decision_latency"]


@dataclass(frozen=True)
class DecisionLatencyBaseline:
    """Baseline decision latency percentiles captured during M1."""

    p50_s: float
    p99_s: float
    samples: int | None = None

    def with_tolerance(self, *, tolerance_pct: float) -> "DecisionLatencyBaseline":
        """Return a baseline inflated by ``tolerance_pct`` (e.g. 0.05 for 5%)."""

        multiplier = 1.0 + max(tolerance_pct, 0.0)
        return DecisionLatencyBaseline(
            p50_s=self.p50_s * multiplier,
            p99_s=self.p99_s * multiplier,
            samples=self.samples,
        )


# Baseline captured from the bootstrap replay harness immediately after the M1 workstream.
_DEFAULT_BASELINE = DecisionLatencyBaseline(p50_s=0.0859, p99_s=0.1532, samples=21)


def evaluate_decision_latency(
    metrics: Mapping[str, object] | None,
    *,
    baseline: DecisionLatencyBaseline | None = None,
    tolerance_pct: float = 0.05,
    min_samples: int = 10,
) -> Mapping[str, object]:
    """Assess decision latency percentiles against the M1 baseline.

    Parameters
    ----------
    metrics:
        Mapping containing ``p50``, ``p99``, and ``samples`` entries.
    baseline:
        Optional override for the reference baseline (defaults to the M1 capture).
    tolerance_pct:
        Percentage slack applied to the baseline when computing thresholds.
    min_samples:
        Minimum number of samples required before enforcing the thresholds.
    """

    if metrics is None:
        return {
            "status": "no_data",
            "reason": "missing_metrics",
        }

    baseline = baseline or _DEFAULT_BASELINE
    thresholds = baseline.with_tolerance(tolerance_pct=tolerance_pct)

    samples = _coerce_int(metrics.get("samples"))
    p50 = _coerce_float(metrics.get("p50"))
    p99 = _coerce_float(metrics.get("p99"))

    result: MutableMapping[str, object] = {
        "status": "pass",
        "samples": samples,
        "current": {
            "p50_s": p50,
            "p99_s": p99,
        },
        "baseline": {
            "p50_s": baseline.p50_s,
            "p99_s": baseline.p99_s,
            "samples": baseline.samples,
        },
        "thresholds": {
            "p50_s": thresholds.p50_s,
            "p99_s": thresholds.p99_s,
        },
    }

    if samples is None or samples < min_samples:
        result["status"] = "warn"
        result["reason"] = "insufficient_samples"
        return result

    if p50 is None or p99 is None:
        result["status"] = "warn"
        result["reason"] = "incomplete_metrics"
        return result

    failures: list[str] = []
    if p50 > thresholds.p50_s:
        failures.append("p50")
    if p99 > thresholds.p99_s:
        failures.append("p99")

    if failures:
        result["status"] = "fail"
        result["reason"] = "threshold_exceeded"
        result["failures"] = tuple(failures)

    return result


def _coerce_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: object) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None
