"""Rebuild Terraform-managed compute resources flagged as degraded."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Sequence

_DEFAULT_TERRAFORM_DIR = Path("infra/hetzner")
_DEFAULT_DEGRADED_FILE = Path("artifacts/infra/degraded_nodes.json")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild Terraform resources that have been marked as degraded.",
    )
    parser.add_argument(
        "--terraform-dir",
        type=Path,
        default=_DEFAULT_TERRAFORM_DIR,
        help="Path to the Terraform working directory (defaults to infra/hetzner).",
    )
    parser.add_argument(
        "--degraded-file",
        type=Path,
        default=_DEFAULT_DEGRADED_FILE,
        help="JSON file enumerating degraded Terraform resource addresses.",
    )
    parser.add_argument(
        "--resource",
        action="append",
        dest="resources",
        default=[],
        help="Additional Terraform resource address to rebuild (repeatable).",
    )
    return parser.parse_args(argv)


def _normalise_resources(resources: Iterable[str]) -> list[str]:
    normalised: list[str] = []
    for resource in resources:
        trimmed = resource.strip()
        if not trimmed:
            raise ValueError("Encountered empty Terraform resource address.")
        normalised.append(trimmed)
    # Preserve order while removing duplicates.
    return list(dict.fromkeys(normalised))


def _load_resources_from_file(path: Path) -> list[str]:
    if not path.exists():
        return []

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - exercised via ValueError path
        raise ValueError(f"Failed to parse degraded resource file: {exc.msg}") from exc

    if isinstance(payload, dict):
        candidates = payload.get("resources", [])
    else:
        candidates = payload

    if isinstance(candidates, str):
        candidates = [candidates]

    if not isinstance(candidates, Iterable) or isinstance(candidates, (bytes, bytearray)):
        raise ValueError("Degraded resource file must provide a list of Terraform addresses.")

    return _normalise_resources(str(item) for item in candidates)


def _collect_resources(args: argparse.Namespace) -> list[str]:
    collected: list[str] = []
    if args.resources:
        collected.extend(_normalise_resources(args.resources))

    try:
        file_resources = _load_resources_from_file(args.degraded_file)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc

    collected.extend(file_resources)
    return list(dict.fromkeys(collected))


def _run_terraform(cmd: list[str], workdir: Path) -> None:
    subprocess.run(cmd, check=True, cwd=str(workdir))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    if not args.terraform_dir.exists():
        print(f"error: Terraform directory not found: {args.terraform_dir}", file=sys.stderr)
        return 1

    try:
        resources = _collect_resources(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not resources:
        return 0

    init_cmd = ["terraform", "init", "-input=false"]
    apply_cmd = ["terraform", "apply"]
    apply_cmd.extend(f"-replace={address}" for address in resources)
    apply_cmd.extend(["-input=false", "-auto-approve"])

    try:
        _run_terraform(init_cmd, args.terraform_dir)
        _run_terraform(apply_cmd, args.terraform_dir)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        print(f"error: command failed with exit code {exc.returncode}: {exc.cmd}", file=sys.stderr)
        return exc.returncode or 1

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
