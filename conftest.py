from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Ensure `src` is on sys.path for test imports.
REPO_ROOT = Path(__file__).resolve().parent
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


_FAILURE_METADATA: list["FailureRecord"] = []
_RECORDED_NODES: set[str] = set()
_COLLECTED_TESTS = 0
_RUN_STARTED_AT: datetime | None = None
_FAILURE_METADATA_PATH: Path | None = None

_DEFAULT_FAILURE_METADATA = "build/pytest_failures.json"
_FAILURE_METADATA_ENV = "PYTEST_FAILURE_METADATA"


@dataclass
class FailureRecord:
    nodeid: str
    phase: str
    duration: float
    message: str
    module: str
    test: str
    run_at: str


def pytest_addoption(parser: Any) -> None:  # pragma: no cover - pytest hook
    group = parser.getgroup("failure-metadata")
    group.addoption(
        "--failure-metadata",
        action="store",
        default=None,
        help=(
            "Write JSON metadata about failing tests to the given path. "
            "Use 'none' to disable even if the PYTEST_FAILURE_METADATA environment "
            "variable is set."
        ),
    )
    group.addoption(
        "--no-failure-metadata",
        action="store_true",
        default=False,
        help="Disable writing JSON failure metadata for this pytest invocation.",
    )


def pytest_configure(config: Any) -> None:  # pragma: no cover - pytest hook
    global _RUN_STARTED_AT, _FAILURE_METADATA_PATH

    _RUN_STARTED_AT = datetime.now(timezone.utc)

    if config.getoption("--no-failure-metadata"):
        _FAILURE_METADATA_PATH = None
        return

    configured_path = config.getoption("--failure-metadata")
    env_path = os.environ.get(_FAILURE_METADATA_ENV)

    raw_path = configured_path or env_path or _DEFAULT_FAILURE_METADATA
    if raw_path and raw_path.lower() not in {"none", "off"}:
        _FAILURE_METADATA_PATH = Path(raw_path)
        _FAILURE_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    else:
        _FAILURE_METADATA_PATH = None


def pytest_collection_finish(session: Any) -> None:  # pragma: no cover - pytest hook
    global _COLLECTED_TESTS
    _COLLECTED_TESTS = len(getattr(session, "items", []))


def pytest_runtest_logreport(report: Any) -> None:  # pragma: no cover - pytest hook
    if _FAILURE_METADATA_PATH is None:
        return

    if not getattr(report, "failed", False):
        return

    nodeid = getattr(report, "nodeid", "")
    if not nodeid or nodeid in _RECORDED_NODES:
        return

    _RECORDED_NODES.add(nodeid)

    module, _, test = nodeid.partition("::")
    message = _extract_failure_message(report)

    record = FailureRecord(
        nodeid=nodeid,
        phase=getattr(report, "when", "call"),
        duration=float(getattr(report, "duration", 0.0) or 0.0),
        message=message,
        module=module,
        test=test or module,
        run_at=_RUN_STARTED_AT.isoformat() if _RUN_STARTED_AT else "",
    )
    _FAILURE_METADATA.append(record)


def pytest_sessionfinish(session: Any, exitstatus: int) -> None:  # pragma: no cover - pytest hook
    if _FAILURE_METADATA_PATH is None:
        return

    payload = {
        "run_at": (_RUN_STARTED_AT or datetime.now(timezone.utc)).isoformat(),
        "exit_status": exitstatus,
        "collected": _COLLECTED_TESTS,
        "failures": [record.__dict__ for record in _FAILURE_METADATA],
        "python": sys.version,
        "pytest_args": session.config.invocation_params.args,  # type: ignore[attr-defined]
    }

    _FAILURE_METADATA_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _extract_failure_message(report: Any) -> str:
    longrepr = getattr(report, "longrepr", None)
    if isinstance(longrepr, str):
        text = longrepr.strip()
    else:
        text = ""
        for attr in ("reprcrash", "longreprtext"):
            candidate = getattr(longrepr, attr, None)
            if isinstance(candidate, str):
                text = candidate.strip()
                break
        if not text and longrepr is not None:
            text = str(longrepr).strip()

    if not text:
        return getattr(report, "outcome", "failed")

    first_line = text.splitlines()[0]
    return first_line[:500]
