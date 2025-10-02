"""Guardrail manifest checks for critical regression suites.

These tests ensure the ingest orchestration, risk policy, and observability
regressions remain part of the guardrail test matrix. The CI pipeline executes
``pytest -m guardrail`` on every change; if the manifest is broken or the guard
marker is removed, these checks fail and block the pipeline.
"""

from __future__ import annotations

from pathlib import Path

import pytest


pytestmark = pytest.mark.guardrail

_GUARDRAIL_TARGETS: dict[str, Path] = {
    "ingest_orchestration": Path("tests/data_foundation/test_timescale_backbone_orchestrator.py"),
    "ingest_scheduler": Path("tests/data_foundation/test_ingest_scheduler.py"),
    "risk_policy": Path("tests/trading/test_risk_policy.py"),
    "observability_event_bus": Path("tests/operations/test_event_bus_health.py"),
}


def test_guardrail_manifest_targets_exist() -> None:
    """Each guardrail target file must exist so CI exercises it."""

    missing = [label for label, path in _GUARDRAIL_TARGETS.items() if not path.exists()]
    assert not missing, f"Missing guardrail tests: {', '.join(missing)}"


def test_guardrail_manifest_targets_marked() -> None:
    """Ensure every guardrail module declares the guardrail pytest marker."""

    missing_marker = []
    for label, path in _GUARDRAIL_TARGETS.items():
        contents = path.read_text(encoding="utf-8")
        if "pytest.mark.guardrail" not in contents:
            missing_marker.append(label)
    assert not missing_marker, f"Guardrail marker missing in: {', '.join(missing_marker)}"
