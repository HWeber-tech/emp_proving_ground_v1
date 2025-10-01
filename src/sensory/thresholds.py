"""Threshold evaluation utilities for sensory organs."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ThresholdAssessment", "evaluate_thresholds"]


@dataclass(frozen=True, slots=True)
class ThresholdAssessment:
    """Describe the relationship between a signal value and configured limits."""

    state: str
    magnitude: float
    thresholds: dict[str, float]
    breached_level: str | None
    breach_ratio: float
    distance_to_warn: float
    distance_to_alert: float

    def as_dict(self) -> dict[str, float | str | None]:
        return {
            "state": self.state,
            "magnitude": self.magnitude,
            "breached_level": self.breached_level,
            "breach_ratio": self.breach_ratio,
            "thresholds": dict(self.thresholds),
            "distance_to_warn": self.distance_to_warn,
            "distance_to_alert": self.distance_to_alert,
        }


def evaluate_thresholds(
    value: float,
    warn_threshold: float,
    alert_threshold: float,
    *,
    mode: str = "absolute",
) -> ThresholdAssessment:
    """Return the threshold posture for ``value``.

    ``mode`` controls how ``value`` is converted into a magnitude:

    ``"absolute"``
        Use ``abs(value)`` so deviations in either direction contribute to the
        threshold assessment.

    ``"positive"``
        Clamp to the positive domain so only values greater than zero
        contribute to the assessment.
    """

    if mode not in {"absolute", "positive"}:
        raise ValueError(f"Unsupported evaluation mode: {mode}")

    if mode == "absolute":
        magnitude = abs(float(value))
    else:
        magnitude = max(0.0, float(value))

    warn = max(0.0, float(warn_threshold))
    alert = max(warn, float(alert_threshold))

    if magnitude >= alert:
        state = "alert"
        breached_level: str | None = "alert"
    elif magnitude >= warn:
        state = "warning"
        breached_level = "warn"
    else:
        state = "nominal"
        breached_level = None

    distance_to_warn = max(0.0, warn - magnitude)
    distance_to_alert = max(0.0, alert - magnitude)
    breach_ratio = magnitude / alert if alert > 0 else 0.0

    thresholds = {"warn": warn, "alert": alert}
    return ThresholdAssessment(
        state=state,
        magnitude=magnitude,
        thresholds=thresholds,
        breached_level=breached_level,
        breach_ratio=breach_ratio,
        distance_to_warn=distance_to_warn,
        distance_to_alert=distance_to_alert,
    )
