"""Utilities for deterministic signing of governance artefacts."""

from __future__ import annotations

import hmac
import json
import os
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Mapping, MutableMapping

try:  # pragma: no cover - optional dependency for normalisation
    import numpy as _np  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - tolerate missing numpy
    _np = None

__all__ = ["compute_audit_signature", "normalise_payload"]


def normalise_payload(value: Any) -> Any:
    """Return a JSON-serialisable representation with deterministic ordering."""

    if isinstance(value, Mapping):
        normalised: MutableMapping[str, Any] = {}
        for key, payload in value.items():
            normalised[str(key)] = normalise_payload(payload)
        return dict(sorted(normalised.items()))

    if isinstance(value, (list, tuple)):
        return [normalise_payload(item) for item in value]

    if isinstance(value, set):
        # Sort on the normalised representation to ensure deterministic ordering.
        serialisable = [normalise_payload(item) for item in value]
        return sorted(serialisable, key=lambda item: json.dumps(item, sort_keys=True, default=str))

    if isinstance(value, datetime):
        timestamp = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return timestamp.astimezone(UTC).isoformat()

    if isinstance(value, date):
        return value.isoformat()

    if isinstance(value, Path):
        return value.as_posix()

    if hasattr(value, "as_dict") and callable(value.as_dict):  # type: ignore[attr-defined]
        try:
            return normalise_payload(value.as_dict())
        except Exception:  # pragma: no cover - defensive fallback
            return repr(value)

    if hasattr(value, "dict") and callable(value.dict):  # type: ignore[attr-defined]
        try:
            return normalise_payload(value.dict())
        except Exception:  # pragma: no cover - defensive fallback
            return repr(value)

    if _np is not None:
        try:
            if isinstance(value, _np.generic):
                return normalise_payload(value.item())
            if isinstance(value, _np.ndarray):
                return normalise_payload(value.tolist())
        except Exception:  # pragma: no cover - defensive fallback
            return repr(value)

    if hasattr(value, "tolist") and callable(value.tolist):  # type: ignore[attr-defined]
        try:
            return normalise_payload(value.tolist())
        except Exception:  # pragma: no cover - defensive fallback
            return repr(value)

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return repr(value)


def compute_audit_signature(
    *,
    kind: str,
    payload: Mapping[str, Any],
    previous_signature: str | None = None,
) -> str:
    """Return a SHA-256 signature (optionally HMAC) for ``payload``."""

    normalised = {
        "kind": kind,
        "payload": normalise_payload(payload),
    }
    if previous_signature:
        normalised["previous"] = previous_signature

    message = json.dumps(normalised, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    secret = os.getenv("EMP_AUDIT_SIGNING_KEY")
    if secret:
        digest = hmac.new(secret.encode("utf-8"), message.encode("utf-8"), digestmod="sha256")
        return digest.hexdigest()
    return __sha256(message)


def __sha256(message: str) -> str:
    import hashlib

    return hashlib.sha256(message.encode("utf-8")).hexdigest()
