"""Guardrail manifest checks for critical regression suites.

These tests ensure the ingest orchestration, risk policy, and observability
regressions remain part of the guardrail test matrix. The CI pipeline executes
``pytest -m guardrail`` on every change; if the manifest is broken or the guard
marker is removed, these checks fail and block the pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml


pytestmark = pytest.mark.guardrail

_GUARDRAIL_TARGETS: dict[str, Path] = {
    "ingest_orchestration": Path("tests/data_foundation/test_timescale_backbone_orchestrator.py"),
    "ingest_scheduler": Path("tests/data_foundation/test_ingest_scheduler.py"),
    "risk_policy": Path("tests/trading/test_risk_policy.py"),
    "observability_event_bus": Path("tests/operations/test_event_bus_health.py"),
}


def _load_ci_workflow() -> dict[str, Any]:
    """Load the CI workflow so tests can assert the guardrail coverage contract."""

    workflow_path = Path(".github/workflows/ci.yml")
    assert workflow_path.exists(), "CI workflow is missing"
    return yaml.safe_load(workflow_path.read_text(encoding="utf-8"))


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


def test_ci_guardrail_job_executes_guardrail_marker() -> None:
    """The guardrail job must run the guardrail marker to block regressions early."""

    workflow = _load_ci_workflow()
    guardrail_step = next(
        (
            step
            for step in workflow["jobs"]["tests"]["steps"]
            if step.get("name") == "Pytest (guardrail suite)"
        ),
        None,
    )
    assert guardrail_step is not None, "Guardrail pytest step missing from CI workflow"
    run_script = guardrail_step.get("run", "")
    assert "pytest -m guardrail" in run_script, "Guardrail pytest step does not target the guardrail marker"


def test_ci_guardrail_job_targets_ingest_risk_observability_domains() -> None:
    """Coverage job should enumerate guardrail-critical domains explicitly."""

    workflow = _load_ci_workflow()
    coverage_step = next(
        (
            step
            for step in workflow["jobs"]["tests"]["steps"]
            if step.get("name") == "Pytest (coverage)"
        ),
        None,
    )
    assert coverage_step is not None, "Coverage pytest step missing from CI workflow"
    run_script = coverage_step.get("run", "")
    for domain in (
        "tests/data_foundation",
        "tests/trading",
        "tests/operations",
        "tests/observability",
    ):
        assert domain in run_script, f"Coverage pytest step does not target {domain}"
