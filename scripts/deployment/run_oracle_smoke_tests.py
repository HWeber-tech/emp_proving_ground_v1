#!/usr/bin/env python3
"""Execute the Oracle Cloud smoke-test plan.

The script loads ``config/deployment/oracle_smoke_plan.yaml`` by default,
executes each test sequentially, prints a summary table, and exits with a
non-zero status when a critical test fails.  It is designed to run in CI right
after applying Kubernetes manifests so we can roll back before exposing the
release to live traffic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.deployment import execute_smoke_plan, load_smoke_plan, summarize_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "plan",
        type=Path,
        nargs="?",
        default=Path("config/deployment/oracle_smoke_plan.yaml"),
        help="Path to the smoke-test plan YAML file",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Emit machine-readable JSON summary",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = load_smoke_plan(args.plan)
    results = execute_smoke_plan(plan)
    summary = summarize_results(results)

    if args.json_output:
        print(json.dumps(summary, indent=2))
    else:
        _print_summary(summary)

    if summary["critical_failure"]:
        rollback_cmd = " ".join(plan.rollback_command or [])
        if rollback_cmd:
            print(f"Critical smoke test failed. Suggested rollback command: {rollback_cmd}", file=sys.stderr)
        else:
            print("Critical smoke test failed. No rollback command configured.", file=sys.stderr)
        return 1
    return 0


def _print_summary(summary: dict[str, object]) -> None:
    failed = summary.get("failed", [])
    print("Oracle Cloud smoke-test summary")
    print("===============================")
    print(f"Total tests: {summary.get('total')}")
    print(f"Succeeded: {summary.get('succeeded')}")
    if failed:
        print("Failed tests:")
        for name in failed:
            print(f"  - {name}")
    else:
        print("All tests passed")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
