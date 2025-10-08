"""Shared helpers for recording deterministic risk API metadata in execution modules."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from src.trading.risk.risk_api import (
    RISK_API_RUNBOOK,
    RiskApiError,
    build_runtime_risk_metadata,
)

RiskContextProvider = Callable[[], Any]


def capture_risk_context(
    provider: RiskContextProvider | None,
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    """Resolve the latest deterministic risk metadata and error payloads.

    The helper shields execution modules from provider failures, returning the
    canonical runbook tagged metadata on success or an error payload compatible
    with runtime telemetry surfaces.
    """

    if provider is None:
        return None, None

    try:
        candidate = provider()
    except Exception as exc:  # pragma: no cover - defensive metadata guard
        return None, {
            "message": "Risk context provider failed",
            "error": str(exc),
            "runbook": RISK_API_RUNBOOK,
        }

    if candidate is None:
        return None, {
            "message": "Risk context provider returned no trading manager",
            "runbook": RISK_API_RUNBOOK,
        }

    try:
        metadata = dict(build_runtime_risk_metadata(candidate))
    except RiskApiError as exc:
        return None, exc.to_metadata()
    except Exception as exc:  # pragma: no cover - unexpected metadata issue
        return None, {
            "message": "Unexpected risk metadata failure",
            "error": str(exc),
            "runbook": RISK_API_RUNBOOK,
        }

    return metadata, None


def describe_risk_context(
    metadata: Mapping[str, object] | None,
    error: Mapping[str, object] | None,
) -> dict[str, object]:
    """Render a serialisable risk context payload for telemetry surfaces."""

    payload: dict[str, object] = {
        "runbook": RISK_API_RUNBOOK,
        "risk_api_runbook": RISK_API_RUNBOOK,
    }

    if isinstance(metadata, Mapping) and metadata:
        payload["metadata"] = dict(metadata)

    if isinstance(error, Mapping) and error:
        payload["error"] = dict(error)

    return payload


__all__ = ["RiskContextProvider", "capture_risk_context", "describe_risk_context"]
