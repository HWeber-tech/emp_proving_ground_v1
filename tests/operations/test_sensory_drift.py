from datetime import datetime

import pytest

from src.operations.sensory_drift import DriftSeverity, evaluate_sensory_drift


def test_evaluate_sensory_drift_flags_alert_and_warn() -> None:
    audit_entries = [
        {
            "symbol": "EURUSD",
            "generated_at": datetime.utcnow().isoformat(),
            "unified_score": 0.62,
            "confidence": 0.78,
            "dimensions": {
                "why": {"signal": 0.55, "confidence": 0.74},
                "how": {"signal": -0.15, "confidence": 0.68},
            },
        },
        {
            "symbol": "EURUSD",
            "generated_at": datetime.utcnow().isoformat(),
            "unified_score": 0.10,
            "confidence": 0.45,
            "dimensions": {
                "why": {"signal": 0.05, "confidence": 0.40},
                "how": {"signal": 0.12, "confidence": 0.52},
            },
        },
        {
            "symbol": "EURUSD",
            "generated_at": datetime.utcnow().isoformat(),
            "unified_score": -0.05,
            "confidence": 0.48,
            "dimensions": {
                "why": {"signal": -0.02, "confidence": 0.42},
                "how": {"signal": 0.08, "confidence": 0.46},
            },
        },
    ]

    snapshot = evaluate_sensory_drift(audit_entries, metadata={"ingest_success": True})

    assert snapshot.status is DriftSeverity.alert
    assert snapshot.metadata["ingest_success"] is True
    assert snapshot.metadata["entries"] == len(audit_entries)

    why = snapshot.dimensions["why"]
    assert why.severity is DriftSeverity.alert
    assert why.baseline_signal is not None
    assert pytest.approx(why.baseline_signal, rel=1e-6) == 0.015
    assert why.delta is not None and why.delta > 0.5

    how = snapshot.dimensions["how"]
    assert how.severity is DriftSeverity.warn
    assert how.delta is not None and pytest.approx(how.delta, rel=1e-6) == -0.25
    markdown = snapshot.to_markdown()
    assert "why" in markdown and "how" in markdown


def test_evaluate_sensory_drift_handles_single_entry() -> None:
    audit_entries = [
        {
            "symbol": "GBPUSD",
            "generated_at": datetime.utcnow().isoformat(),
            "dimensions": {
                "why": {"signal": 0.25, "confidence": 0.6},
            },
        }
    ]

    snapshot = evaluate_sensory_drift(audit_entries)

    assert snapshot.status is DriftSeverity.normal
    assert snapshot.sample_window == 1
    why = snapshot.dimensions["why"]
    assert why.baseline_signal is None
    assert why.delta is None
    assert why.severity is DriftSeverity.normal
