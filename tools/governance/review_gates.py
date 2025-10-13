"""CLI helpers for governance review gates.

This utility loads the governance review gate definitions, merges them with the
recorded verdict state, and exposes subcommands to inspect gate posture or
record new decisions.  It complements the promotion/policy ledger CLI by
handling the roadmap requirement for sign-off gates with explicit verdict
tracking and Markdown/JSON evidence outputs.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Sequence

from src.compliance.workflow import ComplianceWorkflowSnapshot
from src.governance.review_gates import (
    ReviewCriterionStatus,
    ReviewGateDecision,
    ReviewGateRegistry,
    ReviewVerdict,
)

_DEFAULT_DEFINITION_PATH = Path("config/governance/review_gates.yaml")
_DEFAULT_STATE_PATH = Path("artifacts/governance/review_gates.json")


def _default_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Governance review gate workflows")
    parser.add_argument(
        "--definitions",
        type=Path,
        default=_DEFAULT_DEFINITION_PATH,
        help="Path to the review gate definitions YAML file (default: config/governance/review_gates.yaml).",
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=_DEFAULT_STATE_PATH,
        help="Path to the persisted review gate state JSON file (default: artifacts/governance/review_gates.json).",
    )
    subparsers = parser.add_subparsers(dest="command")

    status_parser = subparsers.add_parser("status", help="Render review gate status")
    status_parser.add_argument(
        "--format",
        choices=("json", "markdown"),
        default="json",
        help="Output format (default: json).",
    )
    status_parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the rendered summary (stdout always receives the content).",
    )
    status_parser.add_argument(
        "--workflow-output",
        type=Path,
        help="Optional path to write the compliance workflow snapshot as JSON.",
    )
    status_parser.add_argument(
        "--regulation",
        default="AlphaTrade Governance",
        help="Regulation label used for the compliance workflow snapshot (default: AlphaTrade Governance).",
    )

    decide_parser = subparsers.add_parser("decide", help="Record a governance gate verdict")
    decide_parser.add_argument(
        "--gate",
        required=True,
        help="Review gate identifier to update (matches gate_id in the definitions).",
    )
    decide_parser.add_argument(
        "--verdict",
        required=True,
        choices=[member.value for member in ReviewVerdict],
        help="Verdict to record (pass, warn, fail, waived).",
    )
    decide_parser.add_argument(
        "--decided-by",
        action="append",
        default=[],
        help="Reviewer responsible for the verdict (can be repeated).",
    )
    decide_parser.add_argument(
        "--note",
        action="append",
        default=[],
        help="Optional note attached to the verdict (can be repeated).",
    )
    decide_parser.add_argument(
        "--criterion",
        action="append",
        default=[],
        help="Criterion status in the form <id>=<status>; status is met, not_met, or waived.",
    )
    decide_parser.add_argument(
        "--decided-at",
        help="ISO8601 timestamp for the decision (default: now in UTC).",
    )
    decide_parser.add_argument(
        "--persist",
        type=Path,
        help="Override the state file path (defaults to --state).",
    )

    return parser


def _render_status(
    registry: ReviewGateRegistry,
    *,
    output_format: str,
    output_path: Path | None,
    workflow_path: Path | None,
    regulation: str,
) -> int:
    if output_format == "markdown":
        content = registry.to_markdown()
    else:
        content = json.dumps(registry.to_summary(), indent=2)

    print(content)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")

    if workflow_path is not None:
        snapshot: ComplianceWorkflowSnapshot = registry.to_workflow_snapshot(regulation=regulation)
        workflow_path.parent.mkdir(parents=True, exist_ok=True)
        workflow_path.write_text(
            json.dumps(snapshot.as_dict(), indent=2),
            encoding="utf-8",
        )
    return 0


def _parse_criteria(arguments: Sequence[str]) -> dict[str, ReviewCriterionStatus]:
    criteria: dict[str, ReviewCriterionStatus] = {}
    for item in arguments:
        if "=" not in item:
            raise ValueError(f"criterion must be in <id>=<status> form: {item}")
        key, value = item.split("=", 1)
        criterion_id = key.strip().lower()
        if not criterion_id:
            raise ValueError("criterion identifier cannot be empty")
        criteria[criterion_id] = ReviewCriterionStatus.from_value(value)
    return criteria


def _parse_decided_at(value: str | None) -> datetime:
    if not value:
        return datetime.now(tz=UTC)
    decided_at = datetime.fromisoformat(value)
    if decided_at.tzinfo is None:
        decided_at = decided_at.replace(tzinfo=UTC)
    else:
        decided_at = decided_at.astimezone(UTC)
    return decided_at


def main(argv: Sequence[str] | None = None) -> int:
    parser = _default_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    # Default command is status when none provided for backwards compatibility with simple invocations.
    command = args.command or "status"

    state_path: Path = args.state
    if command == "decide" and args.persist is not None:
        state_path = args.persist

    registry = ReviewGateRegistry.load(args.definitions, state_path=state_path)

    if command == "status":
        return _render_status(
            registry,
            output_format=args.format,
            output_path=args.output,
            workflow_path=args.workflow_output,
            regulation=args.regulation,
        )

    if command == "decide":
        try:
            criteria_status = _parse_criteria(args.criterion)
        except ValueError as exc:
            parser.error(str(exc))
        decided_at = _parse_decided_at(args.decided_at)
        verdict = ReviewVerdict.from_value(args.verdict)
        decision = ReviewGateDecision(
            gate_id=str(args.gate).strip().lower(),
            verdict=verdict,
            decided_at=decided_at,
            decided_by=tuple(item.strip() for item in args.decided_by if item.strip()),
            notes=tuple(item.strip() for item in args.note if item.strip()),
            criteria_status=criteria_status,
        )
        try:
            entry = registry.record_decision(decision)
        except KeyError as exc:
            parser.error(str(exc))
        state_path = args.persist or args.state
        registry.save(state_path)
        payload = {
            "gate": entry.definition.gate_id,
            "title": entry.definition.title,
            "verdict": decision.verdict.value,
            "status": entry.status().value,
            "decided_at": decision.decided_at.isoformat(),
            "decided_by": list(decision.decided_by),
            "notes": list(decision.notes),
            "criteria": {
                criterion_id: status.value for criterion_id, status in decision.criteria_status.items()
            },
            "state_path": str(state_path),
        }
        print(json.dumps(payload, indent=2))
        return 0

    parser.error(f"unknown command: {command}")
    return 1


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
try:  # Python < 3.11 fallback
    from datetime import UTC
except ImportError:  # pragma: no cover - compatibility branch
    from datetime import timezone

    UTC = timezone.utc  # type: ignore[assignment]
