"""CLI to regenerate canonical policy runtime payloads from the governance ledger."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Mapping, Sequence

from src.config.risk.risk_config import RiskConfig
from src.governance.policy_ledger import PolicyLedgerStore
from src.governance.policy_phenotype import (
    build_policy_phenotypes,
    select_policy_phenotype,
)
from src.governance.strategy_rebuilder import StrategyRuntimeConfig, rebuild_strategy

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
        help=(
            "Optional path for the canonical runtime payload. When provided the "
            "command writes byte-identical JSON generated from the ledger replay."
        ),
    )
    parser.add_argument(
        "--runtime-output",
        type=Path,
        help=(
            "Optional destination path for the canonical runtime configuration JSON. "
            "Defaults to runtime_config.json next to --output when omitted."
        ),
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=None,
        help=(
            "Pretty-print indentation for stdout output. When omitted the runtime "
            "payload is emitted using canonical JSON bytes."
        ),
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


def _emit_runtime_config(
    config: StrategyRuntimeConfig,
    output: Path | None,
    *,
    indent: int | None,
) -> None:
    if output is None:
        text = config.json(indent=indent)
        print(text)
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(config.json_bytes)


def _emit_listing(phenotypes, *, indent: int | None) -> None:
    listing_indent = 2 if indent is None else indent
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
    print(json.dumps(rows, indent=listing_indent, sort_keys=True))


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

    target_hash: str | None = None
    if args.policy_hash:
        candidate = args.policy_hash.strip()
        if not candidate:
            logger.error("policy hash cannot be blank when provided")
            return 1
        target_hash = candidate
    elif args.policy_id:
        try:
            phenotype = select_policy_phenotype(
                phenotypes,
                policy_id=args.policy_id,
            )
        except Exception as exc:
            logger.error("Failed to select policy phenotype: %s", exc)
            return 1
        target_hash = phenotype.policy_hash
    else:
        logger.error("Either --policy-hash or --policy-id must be provided")
        return 1

    try:
        runtime_config = rebuild_strategy(
            target_hash,
            store=store,
            base_config=base_config,
        )
    except Exception as exc:
        logger.error("Failed to rebuild runtime config: %s", exc)
        return 1

    try:
        _emit_runtime_config(runtime_config, args.output, indent=args.indent)
    except Exception as exc:
        logger.error("Failed to emit runtime config payload: %s", exc)
        return 1

    if args.output is None:
        return 0

    logger.info(
        "Runtime payload digest %s for policy %s",
        runtime_config.digest,
        runtime_config.policy_hash,
    )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
