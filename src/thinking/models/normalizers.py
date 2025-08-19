from __future__ import annotations

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

from typing import Any, Dict, Mapping, Optional

from .types import AttackReportTD


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return default


def normalize_prediction(p: object) -> Dict[str, float]:
    """
    Normalize a prediction-like object into a minimal dict.

    Output keys:
      - confidence: float
      - probability: float
    """
    # If it looks like a pydantic/dataclass-like object with .dict()
    try:
        d = p.dict()  # type: ignore[attr-defined]
        if isinstance(d, Mapping):
            return {
                "confidence": _to_float(d.get("confidence", 0.0), 0.0),
                "probability": _to_float(d.get("probability", 0.0), 0.0),
            }
    except Exception:
        pass

    # Mapping path
    if isinstance(p, Mapping):
        return {
            "confidence": _to_float(p.get("confidence", 0.0), 0.0),
            "probability": _to_float(p.get("probability", 0.0), 0.0),
        }

    # Attribute path (structural access)
    try:
        conf = _to_float(getattr(p, "confidence", 0.0), 0.0)
    except Exception:
        conf = 0.0
    try:
        prob = _to_float(getattr(p, "probability", 0.0), 0.0)
    except Exception:
        prob = 0.0

    return {"confidence": conf, "probability": prob}


def normalize_survival_result(r: object) -> Dict[str, float]:
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
    try:
        if hasattr(r, "survival_rate"):
            return {"survival_rate": _to_float(getattr(r, "survival_rate", 0.0), 0.0)}
        if hasattr(r, "survival_probability"):
            return {"survival_rate": _to_float(getattr(r, "survival_probability", 0.0), 0.0)}
    except Exception:
        pass

    # Default
    return {"survival_rate": 0.0}


def normalize_attack_report(a: Mapping[str, Any] | object) -> AttackReportTD:
    """
    Normalize a red-team attack report-like payload to an AttackReportTD.

    Output keys (total=False TypedDict):
      - attack_id: str
      - strategy_id: str
      - success: bool
      - impact: float
      - timestamp: str
      - error: str
    """
    def _as_mapping(obj: object) -> Optional[Mapping[str, Any]]:
        try:
            if isinstance(obj, Mapping):
                return obj
            # Pydantic/dataclass-like
            if hasattr(obj, "dict"):
                d = obj.dict()  # type: ignore[attr-defined]
                if isinstance(d, Mapping):
                    return d
        except Exception:
            pass
        return None

    m = _as_mapping(a)
    if m is None:
        # Fallback to attribute access
        try:
            return AttackReportTD(
                attack_id=str(getattr(a, "attack_id", "")),
                strategy_id=str(getattr(a, "strategy_id", "")),
                success=bool(getattr(a, "success", False)),
                impact=_to_float(getattr(a, "impact", 0.0), 0.0),
                timestamp=str(getattr(a, "timestamp", "")),
                error=str(getattr(a, "error", "")) if hasattr(a, "error") else "",
            )
        except Exception:
            return AttackReportTD()

    # Mapping normalization
    out: AttackReportTD = AttackReportTD()
    try:
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
    except Exception:
        # Always return something shape-compatible
        return AttackReportTD()

    return out