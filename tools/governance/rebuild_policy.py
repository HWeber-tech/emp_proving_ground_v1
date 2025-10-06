"""CLI to rebuild enforceable policy artefacts from the policy ledger."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from src.config.risk.risk_config import RiskConfig
from src.governance.policy_ledger import PolicyLedgerStore, build_policy_governance_workflow
from src.governance.policy_rebuilder import rebuild_policy_artifacts

try:  # Python < 3.11 compatibility
    from datetime import UTC  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - fallback for 3.10 runtimes
    UTC = timezone.utc  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay policy ledger entries to regenerate enforceable risk policies "
            "and router guardrail payloads for AlphaTrade tiers."
        )
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=Path("artifacts/governance/policy_ledger.json"),
        help="Path to the policy ledger JSON artifact. Defaults to artifacts/governance/policy_ledger.json.",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        help="Optional JSON file providing the baseline RiskConfig overrides.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional destination for the rebuilt policy payload (prints to stdout when omitted).",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Pretty-print indentation for JSON output (default: 2).",
    )
    return parser


def _load_base_config(path: Path | None) -> RiskConfig | None:
    if path is None:
        return None
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Base config not found at {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Base config at {path} is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Base config payload must be a mapping")
    return RiskConfig(**payload)


def _emit(payload: Mapping[str, object], output: Path | None, *, indent: int) -> None:
    text = json.dumps(payload, indent=indent, sort_keys=True)
    if output is None:
        print(text)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f"{text}\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    base_config = None
    try:
        base_config = _load_base_config(args.base_config)
    except Exception as exc:
        logger.error("Failed to load base config: %s", exc)
        return 1

    try:
        store = PolicyLedgerStore(args.ledger)
    except Exception as exc:
        logger.error("Failed to load policy ledger: %s", exc)
        return 1

    artifacts = rebuild_policy_artifacts(store, base_config=base_config)
    governance_snapshot = build_policy_governance_workflow(store)

    payload = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "ledger_path": str(args.ledger),
        "policy_count": len(artifacts),
        "policies": {artifact.policy_id: artifact.as_dict() for artifact in artifacts},
        "governance_workflow": governance_snapshot.as_dict(),
    }

    try:
        _emit(payload, args.output, indent=args.indent)
    except Exception as exc:
        logger.error("Failed to emit policy rebuild payload: %s", exc)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
