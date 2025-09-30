"""Oracle Cloud smoke-test execution helpers.

The high-impact roadmap calls for automated smoke tests and a rollback plan
prior to promoting new builds into Oracle Cloud (or equivalent).  This module
provides a small, dependency-light harness that can be invoked from CI or the
local CLI.  It loads declarative plans, executes shell commands, and reports
structured results so deployment pipelines can block on failures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, MutableMapping, Sequence
import os
import shlex
import subprocess
import time

try:
    import yaml  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - handled in tests
    raise RuntimeError(
        "PyYAML is required to load smoke test plans. Install via `pip install pyyaml`."
    ) from exc

__all__ = [
    "SmokeTest",
    "SmokeTestPlan",
    "SmokeTestResult",
    "load_smoke_plan",
    "execute_smoke_plan",
    "summarize_results",
]


@dataclass(slots=True)
class SmokeTest:
    """Single smoke-test command executed against the deployment."""

    name: str
    command: Sequence[str]
    critical: bool = True
    timeout: float | None = None
    env: Mapping[str, str] | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "SmokeTest":
        name = str(payload.get("name") or "unnamed")
        raw_command = payload.get("command")
        if isinstance(raw_command, str):
            command: Sequence[str] = tuple(shlex.split(raw_command))
        elif isinstance(raw_command, Sequence):
            command = tuple(str(part) for part in raw_command)
        else:
            raise ValueError(f"Smoke test '{name}' missing command definition")

        critical = bool(payload.get("critical", True))
        timeout_raw = payload.get("timeout")
        timeout = float(timeout_raw) if timeout_raw is not None else None

        env_payload = payload.get("env")
        env: Mapping[str, str] | None
        if isinstance(env_payload, Mapping):
            env = {str(k): str(v) for k, v in env_payload.items()}
        else:
            env = None

        return cls(name=name, command=command, critical=critical, timeout=timeout, env=env)


@dataclass(slots=True)
class SmokeTestPlan:
    """Collection of smoke tests with metadata for orchestration."""

    name: str
    environment: str
    owner: str | None
    rollback_command: Sequence[str] | None
    tests: Sequence[SmokeTest]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def critical_tests(self) -> list[SmokeTest]:
        return [test for test in self.tests if test.critical]


@dataclass(slots=True)
class SmokeTestResult:
    """Outcome of a smoke test run."""

    test: SmokeTest
    succeeded: bool
    returncode: int
    stdout: str
    stderr: str
    duration_seconds: float
    started_at: datetime

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.test.name,
            "succeeded": self.succeeded,
            "returncode": self.returncode,
            "duration_seconds": self.duration_seconds,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "critical": self.test.critical,
            "timestamp": self.started_at.isoformat(),
        }


def load_smoke_plan(path: str | Path) -> SmokeTestPlan:
    """Load a smoke-test plan from a YAML file."""

    plan_path = Path(path)
    if not plan_path.exists():
        raise FileNotFoundError(f"Smoke plan not found: {plan_path}")

    with plan_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    metadata = payload.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata section must be a mapping")

    plan_name = str(metadata.get("name") or plan_path.stem)
    environment = str(metadata.get("environment") or "unknown")
    owner = metadata.get("owner")
    owner_str = str(owner) if owner is not None else None

    rollback_raw = metadata.get("rollback_command")
    rollback_command: Sequence[str] | None
    if isinstance(rollback_raw, str):
        rollback_command = tuple(shlex.split(rollback_raw))
    elif isinstance(rollback_raw, Sequence):
        rollback_command = tuple(str(part) for part in rollback_raw)
    elif rollback_raw is None:
        rollback_command = None
    else:
        raise ValueError("rollback_command must be a string or list of strings")

    tests_raw = payload.get("tests")
    if not isinstance(tests_raw, Iterable):
        raise ValueError("tests section must be an iterable of mappings")

    tests: list[SmokeTest] = []
    for item in tests_raw:
        if not isinstance(item, Mapping):
            raise ValueError("each smoke test entry must be a mapping")
        tests.append(SmokeTest.from_mapping(item))

    if not tests:
        raise ValueError("smoke plan must contain at least one test")

    return SmokeTestPlan(
        name=plan_name,
        environment=environment,
        owner=owner_str,
        rollback_command=rollback_command,
        tests=tuple(tests),
    )


Runner = Callable[[SmokeTest], SmokeTestResult]


def execute_smoke_plan(
    plan: SmokeTestPlan,
    *,
    runner: Runner | None = None,
) -> list[SmokeTestResult]:
    """Execute the smoke tests sequentially and return their results."""

    if runner is None:
        runner = _default_runner

    results: list[SmokeTestResult] = []
    for test in plan.tests:
        result = runner(test)
        results.append(result)
        if not result.succeeded and test.critical:
            break
    return results


def _default_runner(test: SmokeTest) -> SmokeTestResult:
    started = datetime.now(timezone.utc)
    start_time = time.perf_counter()
    try:
        completed = subprocess.run(
            list(test.command),
            capture_output=True,
            text=True,
            timeout=test.timeout,
            check=False,
            env=_merge_env(test.env),
        )
    except subprocess.TimeoutExpired as exc:
        duration = time.perf_counter() - start_time
        return SmokeTestResult(
            test=test,
            succeeded=False,
            returncode=-1,
            stdout=exc.stdout or "",
            stderr=(exc.stderr or "") + "\ncommand timed out",
            duration_seconds=duration,
            started_at=started,
        )

    duration = time.perf_counter() - start_time
    succeeded = completed.returncode == 0
    return SmokeTestResult(
        test=test,
        succeeded=succeeded,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        duration_seconds=duration,
        started_at=started,
    )


def _merge_env(extra: Mapping[str, str] | None) -> MutableMapping[str, str] | None:
    if not extra:
        return None
    merged: MutableMapping[str, str] = dict(os.environ)
    merged.update(extra)
    return merged


def summarize_results(results: Sequence[SmokeTestResult]) -> dict[str, object]:
    """Produce a structured summary payload for downstream reporting."""

    summary: MutableMapping[str, object] = {
        "total": len(results),
        "succeeded": sum(1 for result in results if result.succeeded),
        "failed": [result.test.name for result in results if not result.succeeded],
        "critical_failure": next(
            (result.test.name for result in results if result.test.critical and not result.succeeded),
            None,
        ),
        "results": [result.as_dict() for result in results],
    }
    return dict(summary)
