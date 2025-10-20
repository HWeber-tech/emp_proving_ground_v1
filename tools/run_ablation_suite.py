#!/usr/bin/env python3
"""CI entrypoint to execute the deterministic ablation sweep and gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from src.thinking.evaluation import (
    build_ablation_payload,
    evaluate_ablation_gates,
    render_ablation_markdown,
    run_ablation_suite,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the roadmap ablation sweep and gates")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/ablations/results.json"),
        help="Path to write the JSON payload (default: artifacts/ablations/results.json)",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("artifacts/ablations/results.md"),
        help="Path to write the Markdown summary (default: artifacts/ablations/results.md)",
    )
    parser.set_defaults(enforce_gates=True)
    parser.add_argument(
        "--enforce-gates",
        dest="enforce_gates",
        action="store_true",
        help="Fail when any ablation gate does not pass (default)",
    )
    parser.add_argument(
        "--skip-gates",
        dest="enforce_gates",
        action="store_false",
        help="Do not fail the process when a gate fails",
    )
    return parser.parse_args(argv)


def _prepare_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    resolved = path.expanduser()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    results = run_ablation_suite()
    gates = evaluate_ablation_gates(results)
    payload = build_ablation_payload(results, gates)

    output_path = _prepare_path(args.output)
    if output_path is not None:
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summary_text = render_ablation_markdown(results, gates)
    summary_path = _prepare_path(args.summary)
    if summary_path is not None:
        summary_path.write_text(summary_text, encoding="utf-8")

    print(json.dumps(payload, indent=2))

    if args.enforce_gates and not all(gate.passed for gate in gates):
        for gate in gates:
            if not gate.passed:
                details = ", ".join(f"{k}={v}" for k, v in sorted(gate.details.items()))
                message = f"gate_failed: {gate.gate_id} ({gate.description})"
                if details:
                    message = f"{message} [{details}]"
                print(message, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
