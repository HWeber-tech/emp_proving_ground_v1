from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

try:  # pragma: no cover - Python < 3.11 fallback
    from datetime import UTC  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - fallback for Python < 3.11
    UTC = timezone.utc


def _now() -> datetime:
    return datetime.now(tz=UTC)


def _to_utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise ValueError(str(exc))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a CI alert timeline JSON payload for forced-failure drills."
        )
    )
    parser.add_argument(
        "--incident-id",
        required=True,
        help="Identifier for the drill or alert (e.g. ci-alert-2025-10-07)",
    )
    parser.add_argument(
        "--opened-at",
        required=True,
        help="Timestamp when the alert opened (ISO 8601)",
    )
    parser.add_argument(
        "--opened-channel",
        default="github",
        help="Channel that opened the alert (defaults to github)",
    )
    parser.add_argument(
        "--opened-actor",
        help="Actor responsible for opening the alert (optional)",
    )
    parser.add_argument(
        "--ack-at",
        help="Timestamp when the alert was acknowledged (ISO 8601)",
    )
    parser.add_argument(
        "--ack-channel",
        default="slack",
        help="Channel used to acknowledge the alert (defaults to slack)",
    )
    parser.add_argument(
        "--ack-actor",
        help="Actor who acknowledged the alert (optional)",
    )
    parser.add_argument(
        "--resolve-at",
        help="Timestamp when the alert was resolved (ISO 8601)",
    )
    parser.add_argument(
        "--resolve-channel",
        default="github",
        help="Channel used to resolve the alert (defaults to github)",
    )
    parser.add_argument(
        "--resolve-actor",
        help="Actor who resolved the alert (optional)",
    )
    parser.add_argument(
        "--note",
        help="Optional free-form note to include in the timeline",
    )
    parser.add_argument(
        "--drill",
        action="store_true",
        help="Flag the timeline as a forced-failure drill",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the drill timeline JSON",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        opened_at = _to_utc(args.opened_at)
    except ValueError as exc:
        parser.error(f"Invalid --opened-at timestamp: {exc}")

    ack_at = None
    if args.ack_at is not None:
        try:
            ack_at = _to_utc(args.ack_at)
        except ValueError as exc:
            parser.error(f"Invalid --ack-at timestamp: {exc}")

    resolve_at = None
    if args.resolve_at is not None:
        try:
            resolve_at = _to_utc(args.resolve_at)
        except ValueError as exc:
            parser.error(f"Invalid --resolve-at timestamp: {exc}")

    if ack_at is not None and ack_at < opened_at:
        parser.error("--ack-at must be greater than or equal to --opened-at")
    if resolve_at is not None and resolve_at < opened_at:
        parser.error("--resolve-at must be greater than or equal to --opened-at")
    if ack_at is not None and resolve_at is not None and resolve_at < ack_at:
        parser.error("--resolve-at must be greater than or equal to --ack-at")

    events = [
        {
            "type": "alert_opened",
            "timestamp": opened_at.isoformat(timespec="seconds"),
            "channel": args.opened_channel,
        }
    ]
    if args.opened_actor:
        events[0]["actor"] = args.opened_actor

    if ack_at is not None:
        ack_event = {
            "type": "alert_acknowledged",
            "timestamp": ack_at.isoformat(timespec="seconds"),
            "channel": args.ack_channel,
        }
        if args.ack_actor:
            ack_event["actor"] = args.ack_actor
        events.append(ack_event)

    if resolve_at is not None:
        resolve_event = {
            "type": "alert_resolved",
            "timestamp": resolve_at.isoformat(timespec="seconds"),
            "channel": args.resolve_channel,
        }
        if args.resolve_actor:
            resolve_event["actor"] = args.resolve_actor
        events.append(resolve_event)

    timeline: dict[str, object] = {
        "incident_id": args.incident_id,
        "label": args.incident_id,
        "drill": bool(args.drill),
        "generated_at": _now().isoformat(timespec="seconds"),
        "events": events,
    }

    if args.note:
        timeline["note"] = args.note

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(timeline, indent=2) + "\n")

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
