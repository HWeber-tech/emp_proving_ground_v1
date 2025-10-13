"""CLI to materialise a policy phenotype snapshot from the governance ledger."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from src.config.risk.risk_config import RiskConfig
from src.governance.policy_ledger import PolicyLedgerStore
from src.governance.policy_phenotype import (
    build_policy_phenotypes,
    select_policy_phenotype,
)

try:  # Python < 3.11 compatibility
    from datetime import UTC  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - fallback for 3.10 runtimes
    UTC = timezone.utc  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild a single policy phenotype from the governance ledger using "
            "the deterministic hash produced by the ledger replay pipeline."
        )
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=Path("artifacts/governance/policy_ledger.json"),
        help="Path to the policy ledger JSON artifact (default: artifacts/governance/policy_ledger.json).",
    )
    parser.add_argument(
        "--policy-hash",
        type=str,
        help="Phenotype hash emitted by the rebuild pipeline. Takes precedence over --policy-id when provided.",
    )
    parser.add_argument(
        "--policy-id",
        type=str,
        help="Fallback selector when the phenotype hash is unavailable.",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        help="Optional JSON file providing baseline RiskConfig overrides prior to applying ledger deltas.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional destination path for the phenotype payload (prints to stdout when omitted).",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Pretty-print indentation for JSON output (default: 2).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available phenotypes with their hashes and exit.",
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


def _emit_listing(phenotypes, *, indent: int) -> None:
    rows = [
        {
            "policy_id": phenotype.policy_id,
            "policy_hash": phenotype.policy_hash,
            "stage": phenotype.stage.value,
            "tactic_id": phenotype.tactic_id,
            "updated_at": phenotype.updated_at.isoformat(),
        }
        for phenotype in phenotypes
    ]
    print(json.dumps(rows, indent=indent, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

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

    phenotypes = build_policy_phenotypes(store, base_config=base_config)

    if args.list:
        _emit_listing(phenotypes, indent=args.indent)
        return 0

    try:
        phenotype = select_policy_phenotype(
            phenotypes,
            policy_hash=args.policy_hash,
            policy_id=args.policy_id,
        )
    except Exception as exc:
        logger.error("Failed to select policy phenotype: %s", exc)
        return 1

    payload = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "ledger_path": str(args.ledger),
        "phenotype": phenotype.as_dict(),
    }

    try:
        _emit(payload, args.output, indent=args.indent)
    except Exception as exc:
        logger.error("Failed to emit phenotype payload: %s", exc)
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

