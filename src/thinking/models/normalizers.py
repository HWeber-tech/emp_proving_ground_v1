"""
Normalization helpers for boundary objects (structural typing)
=============================================================

These helpers convert loosely-typed or third-party objects into small,
well-defined dicts that our code consumes. Use these at integration
boundaries to avoid scattered casts and getattr chains.

Rules:
- Never raise: always return a safe default shape.
- Only include fields we actually consume in the codebase.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Optional, SupportsFloat, SupportsIndex, cast

from .types import AttackReportTD


logger = logging.getLogger(__name__)


def _to_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    try:
        Floatable = str | SupportsFloat | SupportsIndex
        return float(cast(Floatable, value))
    except (TypeError, ValueError):
        return default


def _call_dict_method(obj: object) -> Mapping[str, object] | None:
    """Best-effort ``dict()`` invocation that never raises."""

    dict_method = getattr(obj, "dict", None)
    if not callable(dict_method):
        return None
    try:
        candidate = dict_method()
    except Exception as exc:  # pragma: no cover - defensive logging only
        logger.debug("Object %s.dict() failed during normalization: %s", type(obj).__name__, exc, exc_info=exc)
        return None
    if isinstance(candidate, Mapping):
        return candidate
    logger.debug(
        "Object %s.dict() returned non-mapping payload of type %s", type(obj).__name__, type(candidate).__name__
    )
    return None


def _safe_getattr(obj: object, attr: str) -> object | None:
    """Fetch attribute while swallowing unexpected getter errors."""

    try:
        return getattr(obj, attr)
    except AttributeError:
        return None
    except Exception as exc:  # pragma: no cover - getters may raise
        logger.debug(
            "Attribute %s access failed on %s: %s", attr, type(obj).__name__, exc, exc_info=exc
        )
        return None


def normalize_prediction(p: object) -> dict[str, float]:
    """
    Normalize a prediction-like object into a minimal dict.

    Output keys:
      - confidence: float
      - probability: float
    """
    # If it looks like a pydantic/dataclass-like object with .dict()
    payload = _call_dict_method(p)
    if isinstance(payload, Mapping):
        return {
            "confidence": _to_float(payload.get("confidence", 0.0), 0.0),
            "probability": _to_float(payload.get("probability", 0.0), 0.0),
        }

    # Mapping path
    if isinstance(p, Mapping):
        return {
            "confidence": _to_float(p.get("confidence", 0.0), 0.0),
            "probability": _to_float(p.get("probability", 0.0), 0.0),
        }

    # Attribute path (structural access)
    conf = _to_float(_safe_getattr(p, "confidence"), 0.0)
    prob = _to_float(_safe_getattr(p, "probability"), 0.0)

    return {"confidence": conf, "probability": prob}


def normalize_survival_result(r: object) -> dict[str, float]:
    """
    Normalize a survival-like result into a minimal dict.

    Output keys:
      - survival_rate: float
    Notes:
    - Accepts either .survival_rate or .survival_probability (fallback).
    - Accepts mapping or attribute-based access.
    """
    # Mapping path
    if isinstance(r, Mapping):
        sr = r.get("survival_rate", r.get("survival_probability", 0.0))
        return {"survival_rate": _to_float(sr, 0.0)}

    # Attribute path (GAN discriminator results often expose .survival_rate)
    attr_value = _safe_getattr(r, "survival_rate")
    if attr_value is not None:
        return {"survival_rate": _to_float(attr_value, 0.0)}

    probability = _safe_getattr(r, "survival_probability")
    if probability is not None:
        return {"survival_rate": _to_float(probability, 0.0)}

    # Default
    return {"survival_rate": 0.0}


def normalize_attack_report(a: Mapping[str, object] | object) -> AttackReportTD:
    """
    Normalize a red-team attack report-like payload to an AttackReportTD.

    Output keys (total=False TypedDict):
      - attack_id: str
      - strategy_id: str
      - success: bool
      - impact: float
      - timestamp: str
      - error: str
      - anticipation_guard: dict[str, object]
      - observer_focus: str
      - camouflage_seed: str
      - observation_signature: dict[str, object]
    """

    def _as_mapping(obj: object) -> Optional[Mapping[str, object]]:
        if isinstance(obj, Mapping):
            return obj
        payload = _call_dict_method(obj)
        if isinstance(payload, Mapping):
            return payload
        return None

    m = _as_mapping(a)
    if m is None:
        # Fallback to attribute access
        out = AttackReportTD(
            attack_id=str(_safe_getattr(a, "attack_id") or ""),
            strategy_id=str(_safe_getattr(a, "strategy_id") or ""),
            success=bool(_safe_getattr(a, "success") or False),
            impact=_to_float(_safe_getattr(a, "impact"), 0.0),
            timestamp=str(_safe_getattr(a, "timestamp") or ""),
            error=str(_safe_getattr(a, "error") or ""),
        )
        guard_attr = _safe_getattr(a, "anticipation_guard")
        if isinstance(guard_attr, Mapping):
            out["anticipation_guard"] = dict(guard_attr)
        focus_attr = _safe_getattr(a, "observer_focus")
        if focus_attr is not None:
            out["observer_focus"] = str(focus_attr)
        seed_attr = _safe_getattr(a, "camouflage_seed")
        if seed_attr is not None:
            out["camouflage_seed"] = str(seed_attr)
        signature_attr = _safe_getattr(a, "observation_signature")
        if isinstance(signature_attr, Mapping):
            out["observation_signature"] = dict(signature_attr)
        return out

    # Mapping normalization
    out: AttackReportTD = AttackReportTD()
    if "attack_id" in m:
        out["attack_id"] = str(m.get("attack_id", ""))
    if "strategy_id" in m:
        out["strategy_id"] = str(m.get("strategy_id", ""))
    if "success" in m:
        out["success"] = bool(m.get("success", False))
    if "impact" in m:
        out["impact"] = _to_float(m.get("impact", 0.0), 0.0)
    if "timestamp" in m:
        out["timestamp"] = str(m.get("timestamp", ""))
    if "error" in m:
        out["error"] = str(m.get("error", ""))
    guard_payload = m.get("anticipation_guard")
    if isinstance(guard_payload, Mapping):
        out["anticipation_guard"] = dict(guard_payload)
    focus_value = m.get("observer_focus")
    if focus_value is not None:
        out["observer_focus"] = str(focus_value)
    seed_value = m.get("camouflage_seed")
    if seed_value is not None:
        out["camouflage_seed"] = str(seed_value)
    signature_payload = m.get("observation_signature")
    if isinstance(signature_payload, Mapping):
        out["observation_signature"] = dict(signature_payload)

    return out
