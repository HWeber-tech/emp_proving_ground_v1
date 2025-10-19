"""Governance publishing helpers for TRM suggestions."""

from __future__ import annotations

import datetime as dt
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Mapping, MutableMapping, Sequence

from .types import RIMWindow

_QUEUE_NAME = "reflection.trm"
_TOP_SUGGESTION_LIMIT = 5


@dataclass(slots=True, frozen=True)
class GovernancePublication:
    """Container describing the governance artifacts emitted for a TRM run."""

    queue_path: Path | None
    digest_path: Path
    markdown_path: Path


@dataclass(slots=True, frozen=True)
class ProposalEvaluation:
    """Replay-derived metrics describing whether a proposal cleared safeguards."""

    suggestion_id: str
    oos_uplift: float | None
    risk_hits: int | None
    invariant_breaches: int | None = None
    budget_remaining: float | None = None
    budget_utilisation: float | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> Mapping[str, object]:
        payload: MutableMapping[str, object] = {
            "suggestion_id": self.suggestion_id,
            "oos_uplift": self.oos_uplift,
            "risk_hits": self.risk_hits,
        }
        if self.invariant_breaches is not None:
            payload["invariant_breaches"] = self.invariant_breaches
        if self.budget_remaining is not None:
            payload["budget_remaining"] = self.budget_remaining
        if self.budget_utilisation is not None:
            payload["budget_utilisation"] = self.budget_utilisation
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(slots=True, frozen=True)
class AutoApplyDecision:
    """Outcome of evaluating a proposal against the auto-apply rule."""

    suggestion_id: str
    auto_applied: bool
    reasons: tuple[str, ...] = ()
    evaluation: ProposalEvaluation | None = None

    def as_dict(self) -> Mapping[str, object]:
        payload: MutableMapping[str, object] = {
            "applied": self.auto_applied,
            "reasons": list(self.reasons),
        }
        if self.evaluation is not None:
            payload["evaluation"] = dict(self.evaluation.as_dict())
        return payload


@dataclass(slots=True, frozen=True)
class AutoApplyRuleConfig:
    """Configuration for the governance auto-apply rule."""

    uplift_threshold: float
    max_risk_hits: int = 0
    min_budget_remaining: float = 0.0
    max_budget_utilisation: float | None = None
    require_budget_metrics: bool = True

    def as_dict(self) -> Mapping[str, object]:
        payload: MutableMapping[str, object] = {
            "uplift_threshold": self.uplift_threshold,
            "max_risk_hits": self.max_risk_hits,
            "min_budget_remaining": self.min_budget_remaining,
            "require_budget_metrics": self.require_budget_metrics,
        }
        if self.max_budget_utilisation is not None:
            payload["max_budget_utilisation"] = self.max_budget_utilisation
        return payload

    def evaluate(self, evaluation: ProposalEvaluation) -> AutoApplyDecision:
        """Return the auto-apply decision for the supplied evaluation."""

        reasons: list[str] = []

        uplift = evaluation.oos_uplift
        if uplift is None:
            reasons.append("uplift_missing")
        elif uplift < self.uplift_threshold:
            reasons.append(
                f"uplift_below_threshold:{uplift:.6f}<{self.uplift_threshold:.6f}"
            )

        risk_hits = evaluation.risk_hits
        if risk_hits is None:
            reasons.append("risk_hits_unknown")
        elif risk_hits > self.max_risk_hits:
            reasons.append(f"risk_hits_exceeded:{risk_hits}>{self.max_risk_hits}")

        invariant_breaches = evaluation.invariant_breaches
        if invariant_breaches is None:
            reasons.append("invariants_unknown")
        elif invariant_breaches > 0:
            reasons.append(f"invariants_breached:{invariant_breaches}")

        budget_remaining = evaluation.budget_remaining
        budget_utilisation = evaluation.budget_utilisation

        if budget_remaining is not None and budget_remaining <= self.min_budget_remaining:
            reasons.append("budget_exhausted")

        if (
            self.max_budget_utilisation is not None
            and budget_utilisation is not None
            and budget_utilisation > self.max_budget_utilisation
        ):
            reasons.append(
                f"budget_over_utilised:{budget_utilisation:.6f}>{self.max_budget_utilisation:.6f}"
            )

        if self.require_budget_metrics and budget_remaining is None and budget_utilisation is None:
            reasons.append("budget_unknown")

        decision = AutoApplyDecision(
            suggestion_id=evaluation.suggestion_id,
            auto_applied=not reasons,
            reasons=tuple(reasons),
            evaluation=evaluation,
        )
        return decision


def evaluate_auto_apply(
    evaluations: Sequence[ProposalEvaluation],
    config: AutoApplyRuleConfig,
) -> Mapping[str, AutoApplyDecision]:
    """Evaluate a batch of proposals and return their auto-apply decisions."""

    decisions: dict[str, AutoApplyDecision] = {}
    for evaluation in evaluations:
        decision = config.evaluate(evaluation)
        decisions[evaluation.suggestion_id] = decision
    return decisions


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
    proposal_evaluations: Sequence[ProposalEvaluation] | None = None,
    auto_apply_config: AutoApplyRuleConfig | None = None,
) -> GovernancePublication:
    """Persist governance-friendly artifacts summarising the TRM outputs."""

    timestamp = run_timestamp.astimezone(dt.timezone.utc)
    timestamp_iso = _isoformat(timestamp)

    evaluation_lookup: dict[str, ProposalEvaluation] = {}
    if proposal_evaluations:
        evaluation_lookup = {
            evaluation.suggestion_id: evaluation
            for evaluation in proposal_evaluations
        }

    auto_apply_decisions: dict[str, AutoApplyDecision] = {}
    if auto_apply_config is not None and evaluation_lookup:
        for suggestion in suggestions:
            suggestion_id = str(suggestion.get("suggestion_id", "")).strip()
            if not suggestion_id:
                continue
            evaluation = evaluation_lookup.get(suggestion_id)
            if evaluation is None:
                continue
            auto_apply_decisions[suggestion_id] = auto_apply_config.evaluate(evaluation)

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
                suggestion_id = str(entry.get("suggestion_id", "")).strip()
                decision = auto_apply_decisions.get(suggestion_id)
                if decision is not None:
                    governance_payload["auto_apply"] = decision.as_dict()
                    if decision.auto_applied:
                        governance_payload["status"] = "auto_applied"
                        governance_payload["applied_at"] = timestamp_iso
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
        auto_apply_decisions=auto_apply_decisions,
        auto_apply_config=auto_apply_config,
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
    auto_apply_decisions: Mapping[str, AutoApplyDecision] | None = None,
    auto_apply_config: AutoApplyRuleConfig | None = None,
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

    auto_apply_payload: MutableMapping[str, object] | None = None
    if auto_apply_decisions:
        applied_ids = sorted(
            suggestion_id
            for suggestion_id, decision in auto_apply_decisions.items()
            if decision.auto_applied
        )
        pending_ids = sorted(
            suggestion_id
            for suggestion_id, decision in auto_apply_decisions.items()
            if not decision.auto_applied
        )
        failure_reasons = Counter(
            reason
            for decision in auto_apply_decisions.values()
            if not decision.auto_applied
            for reason in decision.reasons
        )
        auto_apply_payload = {
            "evaluated": len(auto_apply_decisions),
            "auto_applied": len(applied_ids),
            "pending": len(pending_ids),
        }
        if auto_apply_config is not None:
            auto_apply_payload["config"] = dict(auto_apply_config.as_dict())
        if applied_ids:
            auto_apply_payload["applied_suggestions"] = applied_ids
        if pending_ids:
            auto_apply_payload["pending_suggestions"] = pending_ids
        if failure_reasons:
            auto_apply_payload["failure_reasons"] = dict(sorted(failure_reasons.items()))

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
    if auto_apply_payload is not None:
        digest["auto_apply"] = auto_apply_payload
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

    auto_apply = digest.get("auto_apply")
    if isinstance(auto_apply, Mapping) and auto_apply.get("evaluated"):
        evaluated = int(auto_apply.get("evaluated", 0) or 0)
        applied = int(auto_apply.get("auto_applied", 0) or 0)
        pending = int(auto_apply.get("pending", 0) or 0)

        lines.append("")
        lines.append("## Auto-apply summary")
        lines.append(f"- Evaluated: {evaluated}")
        lines.append(f"- Auto-applied: {applied}")
        if pending:
            lines.append(f"- Pending review: {pending}")

        config_payload = auto_apply.get("config")
        if isinstance(config_payload, Mapping) and config_payload:
            config_pairs = ", ".join(
                f"{key}={config_payload[key]}"
                for key in sorted(config_payload)
            )
            lines.append(f"- Rule config: {config_pairs}")

        applied_ids = auto_apply.get("applied_suggestions")
        if isinstance(applied_ids, Sequence) and applied_ids:
            lines.append(
                "- Auto-applied suggestions: "
                + ", ".join(str(item) for item in applied_ids)
            )

        pending_ids = auto_apply.get("pending_suggestions")
        if isinstance(pending_ids, Sequence) and pending_ids:
            lines.append(
                "- Pending suggestions: "
                + ", ".join(str(item) for item in pending_ids)
            )

        failure_reasons = auto_apply.get("failure_reasons")
        if isinstance(failure_reasons, Mapping) and failure_reasons:
            lines.append("- Failure reasons:")
            for reason, count in sorted(failure_reasons.items()):
                lines.append(f"  - {reason}: {count}")

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
    "ProposalEvaluation",
    "AutoApplyDecision",
    "AutoApplyRuleConfig",
    "evaluate_auto_apply",
    "publish_governance_artifacts",
]
