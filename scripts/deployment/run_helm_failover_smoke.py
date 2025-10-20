#!/usr/bin/env python3
"""CLI for validating Helm failover readiness using a historical replay smoke test."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from collections.abc import Mapping
from typing import Sequence

from src.deployment import (
    HelmFailoverReplayConfig,
    HelmFailoverReplayError,
    run_helm_failover_replay_smoke,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a Helm deployment after failover by waiting for pods to be ready "
            "and executing a historical replay smoke command."
        )
    )
    parser.add_argument("--release", required=True, help="Helm release name")
    parser.add_argument(
        "--namespace",
        default="emp-system",
        help="Kubernetes namespace containing the release (default: emp-system)",
    )
    parser.add_argument(
        "--selector",
        help="Optional label selector override for pods (defaults to release selector)",
    )
    parser.add_argument(
        "--kubectl",
        default="kubectl",
        help="Path to the kubectl executable (default: kubectl)",
    )
    parser.add_argument(
        "--max-wait-seconds",
        type=float,
        default=120.0,
        help="Maximum time to wait for pods to become ready (default: 120)",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=5.0,
        help="Polling interval for pod readiness checks (default: 5)",
    )
    parser.add_argument(
        "--replay-command",
        help=(
            "Optional custom replay command. Supports {release} and {namespace} "
            "format tokens. Defaults to executing tools.runtime.run_simulation via kubectl exec."
        ),
    )
    parser.add_argument(
        "--replay-timeout",
        type=float,
        default=180.0,
        help="Timeout applied to the replay command (default: 180)",
    )
    parser.add_argument(
        "--replay-env",
        action="append",
        metavar="KEY=VALUE",
        help="Environment variable override passed to the replay command (may be repeated)",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Emit a JSON summary instead of human-readable output",
    )
    return parser


def _parse_env(entries: Sequence[str] | None) -> dict[str, str]:
    overrides: dict[str, str] = {}
    if not entries:
        return overrides
    for raw in entries:
        if "=" not in raw:
            raise ValueError(f"Environment override must be KEY=VALUE: {raw}")
        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Environment override has empty key: {raw}")
        overrides[key] = value
    return overrides


def _format_pod_line(result: Mapping[str, object]) -> str:
    name = result.get("name", "<pod>")
    ready = result.get("ready")
    reason = result.get("reason")
    status = "ready" if ready else "not ready"
    if reason:
        return f"  - {name}: {status} ({reason})"
    return f"  - {name}: {status}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    env_overrides = _parse_env(args.replay_env)
    selector = args.selector
    replay_command = None
    if args.replay_command:
        formatted = args.replay_command.format(
            release=args.release,
            namespace=args.namespace,
        )
        replay_command = tuple(shlex.split(formatted))

    config = HelmFailoverReplayConfig(
        release=args.release,
        namespace=args.namespace,
        selector=selector,
        kubectl=args.kubectl,
        max_wait_seconds=args.max_wait_seconds,
        poll_interval_seconds=args.poll_interval_seconds,
        replay_command=replay_command,
        replay_timeout=args.replay_timeout,
        replay_environment=env_overrides or None,
    )

    try:
        outcome = run_helm_failover_replay_smoke(config)
    except HelmFailoverReplayError as exc:
        if args.json_output:
            payload = {"status": "failed"}
            payload.update(exc.as_dict())
            print(json.dumps(payload, indent=2))
        else:
            print(
                f"Helm failover replay smoke failed: {exc}",
                file=sys.stderr,
            )
            if exc.pods:
                print("Pod readiness:", file=sys.stderr)
                for pod in exc.pods:
                    print(_format_pod_line(pod.as_dict()), file=sys.stderr)
            if exc.returncode is not None:
                print(
                    f"Historical replay returned exit code {exc.returncode}",
                    file=sys.stderr,
                )
            if exc.stdout:
                print("-- replay stdout --", file=sys.stderr)
                print(exc.stdout, file=sys.stderr)
            if exc.stderr:
                print("-- replay stderr --", file=sys.stderr)
                print(exc.stderr, file=sys.stderr)
        return 1

    if args.json_output:
        payload = {"status": "succeeded"}
        payload.update(outcome.as_dict())
        print(json.dumps(payload, indent=2))
    else:
        print(
            f"Helm failover replay smoke succeeded for release {outcome.release} in "
            f"{outcome.duration_seconds:.1f}s",
        )
        print("Pod readiness:")
        for pod in outcome.pods:
            print(_format_pod_line(pod.as_dict()))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
