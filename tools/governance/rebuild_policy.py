"""CLI to rebuild enforceable policy artefacts from the policy ledger."""

from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from src.config.risk.risk_config import RiskConfig
from src.governance.policy_changelog import (
    DEFAULT_POLICY_PROMOTION_RUNBOOK,
    render_policy_ledger_changelog,
)
from src.governance.policy_ledger import PolicyLedgerStore, build_policy_governance_workflow
from src.governance.policy_rebuilder import rebuild_policy_artifacts

try:  # Python < 3.11 compatibility
    from datetime import UTC  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - fallback for 3.10 runtimes
    UTC = timezone.utc  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_SAFE_FILENAME_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


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
        "--policy",
        help=(
            "Optional policy selector. Matches policy_id, metadata.policy_hash, "
            "or metadata.manifest_hash entries."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional destination for the rebuilt policy payload (prints to stdout when omitted).",
    )
    parser.add_argument(
        "--changelog",
        type=Path,
        help="Optional destination for a Markdown governance changelog derived from the ledger.",
    )
    parser.add_argument(
        "--runbook-url",
        type=str,
        default=DEFAULT_POLICY_PROMOTION_RUNBOOK,
        help=(
            "Runbook URL to embed in the governance changelog output. "
            f"Defaults to {DEFAULT_POLICY_PROMOTION_RUNBOOK}."
        ),
    )
    parser.add_argument(
        "--phenotype-dir",
        type=Path,
        help=(
            "Directory to write per-policy phenotype bundles (one JSON file per policy)."
        ),
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


def _emit_markdown(content: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    text = content if content.endswith("\n") else f"{content}\n"
    output.write_text(text, encoding="utf-8")


def _matches_selector(artifact: "PolicyRebuildArtifact", selector: str) -> bool:
    if artifact.policy_id == selector:
        return True
    metadata = artifact.metadata or {}
    for key in ("policy_hash", "manifest_hash", "phenotype_hash"):
        value = metadata.get(key)
        if isinstance(value, str) and value == selector:
            return True
    if len(selector) >= 6 and artifact.policy_id.endswith(selector):
        return True
    return False


def _filter_artifacts(
    artifacts: Sequence["PolicyRebuildArtifact"],
    selector: str | None,
) -> tuple["PolicyRebuildArtifact", ...]:
    if selector is None:
        return tuple(artifacts)
    trimmed = selector.strip()
    if not trimmed:
        return tuple(artifacts)
    matches = tuple(artifact for artifact in artifacts if _matches_selector(artifact, trimmed))
    if not matches:
        raise LookupError(f"No policy rebuild artifacts matched selector: {trimmed}")
    return matches


def _sanitise_filename(value: str) -> str:
    cleaned = _SAFE_FILENAME_PATTERN.sub("_", value.strip())
    cleaned = cleaned.strip("._") or "policy"
    return cleaned


def _write_phenotypes(
    artifacts: Iterable["PolicyRebuildArtifact"],
    directory: Path,
    *,
    indent: int,
) -> tuple[Path, ...]:
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for artifact in artifacts:
        filename = f"{_sanitise_filename(artifact.policy_id)}.json"
        destination = directory / filename
        payload = artifact.as_dict()
        text = json.dumps(payload, indent=indent, sort_keys=True) + "\n"
        destination.write_text(text, encoding="utf-8")
        paths.append(destination)
    return tuple(paths)


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
    try:
        selected_artifacts = _filter_artifacts(artifacts, args.policy)
    except LookupError as exc:
        logger.error("%s", exc)
        return 1
    governance_snapshot = build_policy_governance_workflow(store)

    payload = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "ledger_path": str(args.ledger),
        "policy_count": len(selected_artifacts),
        "policies": {
            artifact.policy_id: artifact.as_dict() for artifact in selected_artifacts
        },
        "governance_workflow": governance_snapshot.as_dict(),
    }

    phenotype_paths: tuple[Path, ...] = ()
    if args.phenotype_dir is not None:
        try:
            phenotype_paths = _write_phenotypes(
                selected_artifacts,
                args.phenotype_dir,
                indent=args.indent,
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.error("Failed to write phenotype bundles: %s", exc)
            return 1
    if phenotype_paths:
        payload["phenotype_paths"] = [str(path) for path in phenotype_paths]

    try:
        _emit(payload, args.output, indent=args.indent)
    except Exception as exc:
        logger.error("Failed to emit policy rebuild payload: %s", exc)
        return 1

    if args.changelog is not None:
        try:
            markdown = render_policy_ledger_changelog(
                store,
                runbook_url=args.runbook_url,
            )
            _emit_markdown(markdown, args.changelog)
        except Exception as exc:
            logger.error("Failed to emit policy governance changelog: %s", exc)
            return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
