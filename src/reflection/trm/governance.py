"""Governance publishing helpers for TRM suggestions."""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Mapping, Sequence

from .types import RIMWindow

_QUEUE_NAME = "reflection.trm"
_TOP_SUGGESTION_LIMIT = 5


@dataclass(slots=True, frozen=True)
class GovernancePublication:
    """Container describing the governance artifacts emitted for a TRM run."""

    queue_path: Path | None
    digest_path: Path
    markdown_path: Path


def publish_governance_artifacts(
    suggestions: Sequence[Mapping[str, object]],
    *,
    run_id: str,
    run_timestamp: dt.datetime,
    window: RIMWindow,
    input_hash: str,
    model_hash: str,
    config_hash: str,
    queue_path: Path,
    digest_path: Path,
    markdown_path: Path,
    artifact_path: Path | None = None,
) -> GovernancePublication:
    """Persist governance-friendly artifacts summarising the TRM outputs."""

    timestamp = run_timestamp.astimezone(dt.timezone.utc)
    timestamp_iso = _isoformat(timestamp)

    queue_entries_written: Path | None = None
    if suggestions:
        queue_path.parent.mkdir(parents=True, exist_ok=True)
        with queue_path.open("a", encoding="utf-8") as handle:
            for index, suggestion in enumerate(suggestions, 1):
                entry = dict(suggestion)
                governance_payload = {
                    "queue": _QUEUE_NAME,
                    "run_id": run_id,
                    "position": index,
                    "enqueued_at": timestamp_iso,
                    "status": "pending",
                }
                if artifact_path is not None:
                    governance_payload["artifact_path"] = artifact_path.as_posix()
                governance_payload["window"] = {
                    "start": _isoformat(window.start),
                    "end": _isoformat(window.end),
                    "minutes": window.minutes,
                }
                entry["governance"] = governance_payload
                handle.write(json.dumps(entry, sort_keys=True) + "\n")
        queue_entries_written = queue_path

    digest = _build_digest(
        suggestions,
        run_id=run_id,
        timestamp_iso=timestamp_iso,
        window=window,
        input_hash=input_hash,
        model_hash=model_hash,
        config_hash=config_hash,
        queue_path=queue_entries_written,
        artifact_path=artifact_path,
    )
    digest_path.parent.mkdir(parents=True, exist_ok=True)
    digest_path.write_text(json.dumps(digest, indent=2, sort_keys=True), encoding="utf-8")

    markdown = _render_markdown(digest, suggestions)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown, encoding="utf-8")

    return GovernancePublication(
        queue_path=queue_entries_written,
        digest_path=digest_path,
        markdown_path=markdown_path,
    )


def _build_digest(
    suggestions: Sequence[Mapping[str, object]],
    *,
    run_id: str,
    timestamp_iso: str,
    window: RIMWindow,
    input_hash: str,
    model_hash: str,
    config_hash: str,
    queue_path: Path | None,
    artifact_path: Path | None,
) -> Mapping[str, object]:
    suggestion_count = len(suggestions)
    type_counts: dict[str, int] = {}
    confidence_values: list[float] = []
    target_stats: dict[str, dict[str, object]] = {}

    for suggestion in suggestions:
        suggestion_type = str(suggestion.get("type", "unknown"))
        type_counts[suggestion_type] = type_counts.get(suggestion_type, 0) + 1
        confidence = _safe_float(suggestion.get("confidence"))
        if confidence is not None:
            confidence_values.append(confidence)

        target = _derive_target(suggestion)
        stats = target_stats.setdefault(
            target,
            {
                "count": 0,
                "types": {},
                "max_confidence": 0.0,
            },
        )
        stats["count"] = int(stats["count"]) + 1
        typed = stats["types"]
        typed[suggestion_type] = typed.get(suggestion_type, 0) + 1
        if confidence is not None:
            stats["max_confidence"] = max(float(stats["max_confidence"]), confidence)

    if confidence_values:
        confidence_payload: Mapping[str, object] = {
            "min": min(confidence_values),
            "avg": fmean(confidence_values),
            "max": max(confidence_values),
        }
    else:
        confidence_payload = {
            "min": None,
            "avg": None,
            "max": None,
        }

    targets_payload = [
        {
            "target": target,
            "count": stats["count"],
            "types": {
                suggestion_type: count
                for suggestion_type, count in sorted(stats["types"].items())
            },
            "max_confidence": stats["max_confidence"],
        }
        for target, stats in sorted(
            target_stats.items(),
            key=lambda item: (-int(item[1]["count"]), item[0]),
        )
    ]

    top_suggestions = [
        _summarise_suggestion(suggestion)
        for suggestion in sorted(
            suggestions,
            key=lambda payload: _safe_float(payload.get("confidence")) or -1.0,
            reverse=True,
        )[:_TOP_SUGGESTION_LIMIT]
    ]

    digest: dict[str, object] = {
        "run_id": run_id,
        "generated_at": timestamp_iso,
        "suggestion_count": suggestion_count,
        "input_hash": input_hash,
        "model_hash": model_hash,
        "config_hash": config_hash,
        "window": {
            "start": _isoformat(window.start),
            "end": _isoformat(window.end),
            "minutes": window.minutes,
        },
        "queue_path": queue_path.as_posix() if queue_path else None,
        "artifact_path": artifact_path.as_posix() if artifact_path else None,
        "by_type": dict(sorted(type_counts.items())),
        "confidence": confidence_payload,
        "targets": targets_payload,
        "top_suggestions": top_suggestions,
    }
    return digest


def _render_markdown(
    digest: Mapping[str, object],
    suggestions: Sequence[Mapping[str, object]],
) -> str:
    lines = [
        "# TRM Governance Reflection Summary",
        "",
        f"- Run ID: {digest.get('run_id', 'unknown')}",
        f"- Generated: {digest.get('generated_at', 'unknown')}",
        f"- Suggestions: {digest.get('suggestion_count', 0)}",
    ]

    queue_path = digest.get("queue_path")
    artifact_path = digest.get("artifact_path")
    lines.append(f"- Queue: {queue_path or 'n/a'}")
    lines.append(f"- Artifact: {artifact_path or 'n/a'}")

    window = digest.get("window")
    if isinstance(window, Mapping):
        lines.append(
            "- Window: {start} → {end} ({minutes} minutes)".format(
                start=window.get("start", "n/a"),
                end=window.get("end", "n/a"),
                minutes=window.get("minutes", "n/a"),
            )
        )

    confidence = digest.get("confidence")
    if isinstance(confidence, Mapping):
        lines.append(
            "- Confidence (min/avg/max): {min}/{avg}/{max}".format(
                min=_format_confidence(confidence.get("min")),
                avg=_format_confidence(confidence.get("avg")),
                max=_format_confidence(confidence.get("max")),
            )
        )

    lines.append("")
    lines.append("## Suggestion breakdown")
    lines.append("| Target | Total | Types | Max confidence |")
    lines.append("| --- | --- | --- | --- |")
    targets = digest.get("targets")
    if isinstance(targets, Sequence) and targets:
        for entry in targets:
            if not isinstance(entry, Mapping):
                continue
            types = entry.get("types")
            if isinstance(types, Mapping):
                type_summary = ", ".join(
                    f"{key}:{value}" for key, value in sorted(types.items())
                )
            else:
                type_summary = "-"
            lines.append(
                "| {target} | {count} | {types} | {confidence} |".format(
                    target=entry.get("target", "n/a"),
                    count=entry.get("count", 0),
                    types=type_summary or "-",
                    confidence=_format_confidence(entry.get("max_confidence")),
                )
            )
    else:
        lines.append("| n/a | 0 | - | 0.00 |")

    if suggestions:
        lines.append("")
        lines.append("## Top suggestions")
        lines.append("| ID | Type | Target | Confidence | Rationale |")
        lines.append("| --- | --- | --- | --- | --- |")
        for entry in digest.get("top_suggestions", ()):  # type: ignore[assignment]
            if not isinstance(entry, Mapping):
                continue
            lines.append(
                "| {sid} | {stype} | {target} | {confidence} | {rationale} |".format(
                    sid=entry.get("suggestion_id", "n/a"),
                    stype=entry.get("type", "n/a"),
                    target=entry.get("target", "n/a"),
                    confidence=_format_confidence(entry.get("confidence")),
                    rationale=_escape_pipes(str(entry.get("rationale", "")).strip()),
                )
            )

    return "\n".join(lines) + "\n"


def _summarise_suggestion(suggestion: Mapping[str, object]) -> Mapping[str, object]:
    return {
        "suggestion_id": str(suggestion.get("suggestion_id", "")),
        "type": str(suggestion.get("type", "")),
        "target": _derive_target(suggestion),
        "confidence": _safe_float(suggestion.get("confidence")),
        "rationale": str(suggestion.get("rationale", "")).strip(),
    }


def _derive_target(suggestion: Mapping[str, object]) -> str:
    payload = suggestion.get("payload")
    if isinstance(payload, Mapping):
        if "strategy_id" in payload:
            return str(payload.get("strategy_id"))
        candidates = payload.get("strategy_candidates")
        if isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes)):
            for candidate in candidates:
                value = str(candidate).strip()
                if value:
                    return value
        experiment = payload.get("experiment_id")
        if isinstance(experiment, str) and experiment.strip():
            return experiment.strip()
    return "n/a"


def _safe_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_confidence(value: object) -> str:
    numeric = _safe_float(value)
    if numeric is None:
        return "0.00"
    return f"{numeric:.2f}"


def _isoformat(value: dt.datetime) -> str:
    ts = value.astimezone(dt.timezone.utc)
    return ts.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _escape_pipes(text: str) -> str:
    return text.replace("|", "\\|")


__all__ = [
    "GovernancePublication",
    "publish_governance_artifacts",
]
