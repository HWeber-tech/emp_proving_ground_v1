from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from src.deployment import (
    SmokeTest,
    SmokeTestPlan,
    SmokeTestResult,
    execute_smoke_plan,
    load_smoke_plan,
    summarize_results,
)


@pytest.fixture()
def sample_plan_file(tmp_path: Path) -> Path:
    payload = {
        "metadata": {
            "name": "oracle-ci",
            "environment": "oracle-testing",
            "owner": "ops@emp",
            "rollback_command": ["./rollback.sh"],
        },
        "tests": [
            {"name": "ok", "command": ["echo", "ok"], "critical": True},
            {"name": "non-critical", "command": ["echo", "ok"], "critical": False},
        ],
    }
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(yaml.safe_dump(payload))
    return plan_path


@pytest.fixture()
def parsed_plan(sample_plan_file: Path) -> SmokeTestPlan:
    return load_smoke_plan(sample_plan_file)


def test_load_smoke_plan_parses_metadata(parsed_plan: SmokeTestPlan) -> None:
    assert parsed_plan.name == "oracle-ci"
    assert parsed_plan.environment == "oracle-testing"
    assert parsed_plan.owner == "ops@emp"
    assert parsed_plan.rollback_command == ("./rollback.sh",)
    assert len(parsed_plan.tests) == 2


def test_execute_smoke_plan_stops_on_critical_failure(parsed_plan: SmokeTestPlan) -> None:
    def runner(test: SmokeTest) -> SmokeTestResult:
        succeeded = test.name != "ok"
        return SmokeTestResult(
            test=test,
            succeeded=succeeded,
            returncode=0 if succeeded else 1,
            stdout="",
            stderr="",
            duration_seconds=0.01,
            started_at=datetime.now(timezone.utc),
        )

    results = execute_smoke_plan(parsed_plan, runner=runner)
    assert len(results) == 1
    assert results[0].test.name == "ok"
    assert not results[0].succeeded


def test_summarize_results_reports_failures(parsed_plan: SmokeTestPlan) -> None:
    result = SmokeTestResult(
        test=parsed_plan.tests[0],
        succeeded=False,
        returncode=1,
        stdout="",
        stderr="error",
        duration_seconds=0.5,
        started_at=datetime.now(timezone.utc),
    )
    summary = summarize_results([result])
    assert summary["total"] == 1
    assert summary["succeeded"] == 0
    assert summary["critical_failure"] == parsed_plan.tests[0].name
