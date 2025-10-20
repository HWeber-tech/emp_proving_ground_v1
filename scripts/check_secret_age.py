#!/usr/bin/env python3
"""Check the rotation age of API keys and secrets."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping

try:  # Optional dependency used across other scripts
    from dotenv import dotenv_values
except Exception:  # pragma: no cover - dotenv is optional at runtime
    dotenv_values = None


DEFAULT_API_KEY_THRESHOLD_DAYS = 90
DEFAULT_SECRET_THRESHOLD_DAYS = 30
DEFAULT_WARN_RATIO = 0.8

API_KEY_TOKENS: tuple[str, ...] = ("API_KEY", "API-KEY", "APIKEY", "ACCESS_KEY")
SECRET_TOKENS: tuple[str, ...] = (
    "SECRET",
    "PASSWORD",
    "TOKEN",
    "PRIVATE_KEY",
    "CLIENT_SECRET",
)

ROTATION_FIELDS: tuple[tuple[str, str], ...] = (
    ("_ROTATED_AT", "timestamp"),
    ("_ROTATED_ON", "timestamp"),
    ("_ROTATED", "timestamp"),
    ("_UPDATED_AT", "timestamp"),
    ("_ISSUED_AT", "timestamp"),
    ("_CREATED_AT", "timestamp"),
    ("_AGE_DAYS", "age"),
    ("_ROTATION_AGE_DAYS", "age"),
)

IGNORED_SUFFIXES: tuple[str, ...] = tuple(field for field, _ in ROTATION_FIELDS) + (
    "_EXPIRES_AT",
    "_EXPIRES_ON",
    "_EXPIRY",
    "_SECRET_REF",
    "_SECRET_REFERENCE",
    "_SECRET_NAME",
    "_SECRET_PATH",
)


class SecretKind(StrEnum):
    """Categories recognised by the rotation checker."""

    api_key = "api_key"
    secret = "secret"


class SecretStatus(StrEnum):
    """Evaluation status for a secret or API key."""

    ok = "ok"
    warn = "warn"
    stale = "stale"
    unknown = "unknown"
    missing = "missing"


@dataclass(frozen=True)
class RotationInfo:
    """Metadata describing where rotation information was sourced."""

    base_name: str
    variable: str
    raw_value: str
    mode: str
    priority: int


@dataclass(frozen=True)
class SecretRecord:
    """Evaluated rotation state for a secret or API key."""

    name: str
    kind: SecretKind
    variables: tuple[str, ...]
    rotation_source: str | None
    age_days: float | None
    threshold_days: float
    warn_threshold_days: float
    status: SecretStatus
    has_value: bool
    rotation_error: str | None = None


def resolve_default_env_file() -> Path:
    """Return the default env file path, honouring overrides."""

    override = os.environ.get("EMP_SECRETS_ENV_FILE")
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parents[1] / ".env"


def _load_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    if dotenv_values is not None:
        data = dotenv_values(path)
        return {str(key): str(value) for key, value in data.items() if value is not None}
    content = path.read_text(encoding="utf-8")
    mapping: dict[str, str] = {}
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        mapping[key.strip()] = value.strip()
    return mapping


def load_environment(paths: Iterable[Path]) -> dict[str, str]:
    """Load environment mappings from the supplied dotenv files and process env."""

    combined: dict[str, str] = {}
    for path in paths:
        combined.update(_load_env_file(path))
    for key, value in os.environ.items():
        combined[str(key)] = str(value)
    return combined


def parse_timestamp(raw: str | None) -> datetime | None:
    """Parse ISO-like timestamps or UNIX epochs into UTC datetimes."""

    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    if text.isdigit():
        try:
            return datetime.fromtimestamp(int(text), tz=timezone.utc)
        except (OSError, ValueError):
            return None
    candidate = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y"):
            try:
                parsed = datetime.strptime(candidate, fmt)
                break
            except ValueError:
                continue
        else:
            return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _parse_age_days(raw: str | None) -> float | None:
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    try:
        return max(float(text), 0.0)
    except ValueError:
        return None


def _normalise_mapping(mapping: Mapping[str, str]) -> dict[str, str]:
    normalised: dict[str, str] = {}
    for key, value in mapping.items():
        if key is None:
            continue
        key_text = str(key).strip()
        if not key_text:
            continue
        normalised[key_text] = str(value)
    return normalised


def _is_metadata_key(name_upper: str) -> bool:
    return any(name_upper.endswith(suffix) for suffix in IGNORED_SUFFIXES)


def _classify_from_tokens(name_upper: str) -> SecretKind | None:
    for token in API_KEY_TOKENS:
        if token in name_upper:
            return SecretKind.api_key
    for token in SECRET_TOKENS:
        if token in name_upper:
            return SecretKind.secret
    return None


def evaluate_secret_age(
    environment: Mapping[str, str],
    *,
    now: datetime,
    api_key_threshold_days: float = DEFAULT_API_KEY_THRESHOLD_DAYS,
    secret_threshold_days: float = DEFAULT_SECRET_THRESHOLD_DAYS,
    warn_ratio: float = DEFAULT_WARN_RATIO,
) -> list[SecretRecord]:
    """Evaluate rotation age across secrets/API keys present in the environment."""

    env = _normalise_mapping(environment)
    upper_to_key: dict[str, str] = {key.upper(): key for key in env.keys()}

    rotation_data: dict[str, RotationInfo] = {}
    for key, value in env.items():
        key_upper = key.upper()
        for priority, (suffix, mode) in enumerate(ROTATION_FIELDS):
            if not key_upper.endswith(suffix):
                continue
            base_upper = key_upper[: -len(suffix)]
            if not base_upper:
                continue
            base_name = key[: -len(suffix)]
            if base_upper not in rotation_data or priority < rotation_data[base_upper].priority:
                rotation_data[base_upper] = RotationInfo(
                    base_name=base_name,
                    variable=key,
                    raw_value=value,
                    mode=mode,
                    priority=priority,
                )
            break

    associated: dict[str, list[str]] = {base: [] for base in rotation_data}
    for key in env:
        key_upper = key.upper()
        if _is_metadata_key(key_upper):
            continue
        for base_upper in rotation_data:
            prefix = f"{base_upper}_"
            if key_upper.startswith(prefix):
                associated[base_upper].append(key)
                break

    candidates: dict[str, str] = {}
    for base_upper, info in rotation_data.items():
        display = info.base_name or upper_to_key.get(base_upper, base_upper)
        candidates[base_upper] = display

    for key in env:
        key_upper = key.upper()
        if _is_metadata_key(key_upper):
            continue
        if any(key_upper.startswith(f"{base_upper}_") for base_upper in rotation_data):
            continue
        kind = _classify_from_tokens(key_upper)
        if kind is None:
            continue
        candidates.setdefault(key_upper, key)

    records: list[SecretRecord] = []

    severity_order: dict[SecretStatus, int] = {
        SecretStatus.stale: 0,
        SecretStatus.missing: 1,
        SecretStatus.warn: 2,
        SecretStatus.unknown: 3,
        SecretStatus.ok: 4,
    }

    for base_upper, display_name in sorted(candidates.items(), key=lambda item: item[0]):
        rotation_info = rotation_data.get(base_upper)
        variables: list[str] = []

        if rotation_info is not None:
            variables.extend(sorted(associated.get(base_upper, ())))
            base_key = upper_to_key.get(base_upper)
            if base_key is not None and base_key not in variables:
                variables.insert(0, base_key)
        else:
            variables.append(display_name)

        kind = _classify_from_tokens(base_upper)
        if kind is None and rotation_info is not None:
            # Derive kind from associated variables
            derived_kind = None
            for variable in variables:
                derived_kind = _classify_from_tokens(variable.upper())
                if derived_kind is not None:
                    break
            kind = derived_kind or SecretKind.secret
        elif kind is None:
            kind = SecretKind.secret

        has_value = any(env.get(variable, "").strip() for variable in variables)

        age_days: float | None = None
        rotation_error: str | None = None
        rotation_source: str | None = None

        if rotation_info is not None:
            rotation_source = rotation_info.variable
            if rotation_info.mode == "timestamp":
                timestamp = parse_timestamp(rotation_info.raw_value)
                if timestamp is None:
                    rotation_error = "unable to parse rotation timestamp"
                else:
                    delta = now - timestamp
                    age_days = max(delta.total_seconds() / 86400.0, 0.0)
            else:
                age = _parse_age_days(rotation_info.raw_value)
                if age is None:
                    rotation_error = "unable to parse rotation age"
                else:
                    age_days = age

        threshold = api_key_threshold_days if kind is SecretKind.api_key else secret_threshold_days
        warn_threshold = max(threshold * warn_ratio, 0.0)

        if not has_value:
            status = SecretStatus.missing
        elif age_days is None:
            status = SecretStatus.unknown
        elif threshold <= 0:
            status = SecretStatus.ok
        elif age_days <= warn_threshold:
            status = SecretStatus.ok
        elif age_days <= threshold:
            status = SecretStatus.warn
        else:
            status = SecretStatus.stale

        records.append(
            SecretRecord(
                name=display_name,
                kind=kind,
                variables=tuple(variables),
                rotation_source=rotation_source,
                age_days=age_days,
                threshold_days=threshold,
                warn_threshold_days=warn_threshold,
                status=status,
                has_value=has_value,
                rotation_error=rotation_error,
            )
        )

    records.sort(key=lambda record: (severity_order[record.status], record.name))
    return records


def _format_age(age_days: float | None) -> str:
    if age_days is None:
        return "--"
    return f"{age_days:.1f}"


def _format_threshold(threshold_days: float) -> str:
    if threshold_days <= 0:
        return "--"
    return f"{threshold_days:.0f}"


def _render_text_report(
    records: Iterable[SecretRecord],
    *,
    now: datetime,
    env_files: Iterable[Path],
) -> None:
    entries = list(records)
    print(f"Secrets age report @ {now.isoformat()}")
    sources = [str(path) if path.exists() else f"{path} (missing)" for path in env_files]
    if sources:
        print(f"Sources: {', '.join(sources)}")
    if not entries:
        print("No secrets detected.")
        return
    print("Status  Name                           Kind     Age(d)  Limit  Rotation Source                 Notes")
    print("------  ------------------------------ -------- ------- ------ ------------------------------- --------------------------------")
    for record in entries:
        notes: str
        if record.status is SecretStatus.missing:
            notes = "value not set"
        elif record.status is SecretStatus.unknown:
            if record.rotation_source:
                notes = record.rotation_error or "rotation metadata invalid"
            else:
                notes = "rotation metadata missing"
        elif record.status is SecretStatus.warn:
            notes = "approaching rotation limit"
        elif record.status is SecretStatus.stale:
            notes = "exceeds rotation limit"
        else:
            notes = "within rotation target"
        rotation_display = record.rotation_source or "--"
        print(
            f"{record.status.value:<6}  {record.name:<30} {record.kind.value:<8} {_format_age(record.age_days):>7} "
            f"{_format_threshold(record.threshold_days):>6} {rotation_display:<31} {notes}"
        )


def _render_json_report(
    records: Iterable[SecretRecord],
    *,
    now: datetime,
    env_files: Iterable[Path],
    warn_ratio: float,
    api_key_threshold_days: float,
    secret_threshold_days: float,
) -> None:
    payload: MutableMapping[str, object] = {
        "generated_at": now.isoformat(),
        "sources": [str(path) for path in env_files],
        "warn_ratio": warn_ratio,
        "thresholds": {
            "api_key_days": api_key_threshold_days,
            "secret_days": secret_threshold_days,
        },
        "secrets": [
            {
                "name": record.name,
                "kind": record.kind.value,
                "status": record.status.value,
                "age_days": record.age_days,
                "threshold_days": record.threshold_days,
                "warn_threshold_days": record.warn_threshold_days,
                "rotation_source": record.rotation_source,
                "variables": list(record.variables),
                "has_value": record.has_value,
                "rotation_error": record.rotation_error,
            }
            for record in records
        ],
    }
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-file",
        action="append",
        type=Path,
        dest="env_files",
        help="Path to a dotenv file containing secrets (defaults to EMP_SECRETS_ENV_FILE or .env)",
    )
    parser.add_argument(
        "--api-key-threshold",
        type=float,
        default=DEFAULT_API_KEY_THRESHOLD_DAYS,
        help="Rotation limit in days for API keys (default: %(default)s)",
    )
    parser.add_argument(
        "--secret-threshold",
        type=float,
        default=DEFAULT_SECRET_THRESHOLD_DAYS,
        help="Rotation limit in days for secrets (default: %(default)s)",
    )
    parser.add_argument(
        "--warn-ratio",
        type=float,
        default=DEFAULT_WARN_RATIO,
        help="Warning threshold as a fraction of the rotation limit (default: %(default)s)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Emit JSON instead of a text table",
    )
    parser.add_argument(
        "--now",
        type=str,
        help="Override the current timestamp (ISO-8601)",
    )
    parser.add_argument(
        "--fail-on",
        action="append",
        choices=[status.value for status in SecretStatus] + ["none"],
        help=(
            "Statuses that should cause a non-zero exit code. "
            "May be specified multiple times. Defaults to 'stale' and 'missing'. Use 'none' to disable."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    env_files = args.env_files or []
    if not env_files:
        env_files = [resolve_default_env_file()]

    reference_time: datetime
    if args.now:
        parsed = parse_timestamp(args.now)
        if parsed is None:
            raise SystemExit(f"Invalid --now timestamp: {args.now}")
        reference_time = parsed.astimezone(UTC)
    else:
        reference_time = datetime.now(tz=UTC)

    environment = load_environment(env_files)

    records = evaluate_secret_age(
        environment,
        now=reference_time,
        api_key_threshold_days=args.api_key_threshold,
        secret_threshold_days=args.secret_threshold,
        warn_ratio=args.warn_ratio,
    )

    if args.json_output:
        _render_json_report(
            records,
            now=reference_time,
            env_files=env_files,
            warn_ratio=args.warn_ratio,
            api_key_threshold_days=args.api_key_threshold,
            secret_threshold_days=args.secret_threshold,
        )
    else:
        _render_text_report(records, now=reference_time, env_files=env_files)

    if args.fail_on:
        if "none" in args.fail_on:
            failure_statuses: set[SecretStatus] = set()
        else:
            failure_statuses = {SecretStatus(status) for status in args.fail_on}
    else:
        failure_statuses = {SecretStatus.stale, SecretStatus.missing}

    if any(record.status in failure_statuses for record in records):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SecretKind",
    "SecretStatus",
    "SecretRecord",
    "evaluate_secret_age",
    "load_environment",
    "parse_timestamp",
    "resolve_default_env_file",
]
