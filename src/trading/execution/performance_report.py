"""Utilities for rendering execution throttle and throughput diagnostics."""

from __future__ import annotations

from typing import Any, Mapping, Iterable


def _format_float(value: object, *, precision: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number:.{precision}f}"


def _format_percentage(value: object, *, precision: int = 1) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number * 100:.{precision}f}%"


def _format_optional(value: object) -> str:
    if value in (None, "", "-"):
        return "—"
    return str(value)


def _format_bool(value: object) -> str:
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return "—"


def _format_seconds(value: object) -> str:
    """Render a human-friendly representation of seconds."""

    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return "—"

    if seconds < 0:
        seconds = 0.0

    if seconds >= 3600:
        hours = seconds / 3600.0
        if hours.is_integer():
            return f"{int(hours)}h"
        return f"{hours:.2f}h"
    if seconds >= 60:
        minutes = seconds / 60.0
        if minutes.is_integer():
            return f"{int(minutes)}m"
        return f"{minutes:.2f}m"
    if seconds.is_integer():
        return f"{int(seconds)}s"
    return f"{seconds:.2f}s"


def _render_mapping_lines(
    payload: Mapping[str, Any] | None, *, prefix: str = "-"
) -> list[str]:
    if not isinstance(payload, Mapping):
        return [f"{prefix} None"]
    items = []
    for key, value in sorted(payload.items()):
        if isinstance(value, (dict, Mapping)):
            value = dict(value)
        items.append(f"{prefix} {key}: {value}")
    return items or [f"{prefix} None"]


def build_execution_performance_report(stats: Mapping[str, object]) -> str:
    """Render a Markdown summary of execution throughput and throttle state."""

    submitted = stats.get("orders_submitted")
    executed = stats.get("orders_executed")
    failed = stats.get("orders_failed")

    lines = ["# Execution performance summary", ""]
    lines.append("## Order flow")
    lines.append(f"- Submitted: {int(submitted) if isinstance(submitted, int) else _format_optional(submitted)}")
    lines.append(f"- Executed: {int(executed) if isinstance(executed, int) else _format_optional(executed)}")
    lines.append(f"- Failed: {int(failed) if isinstance(failed, int) else _format_optional(failed)}")

    throttle_snapshot = stats.get("trade_throttle")
    if isinstance(throttle_snapshot, Mapping):
        lines.append("")
        lines.append("## Trade throttle")
        name = throttle_snapshot.get("name", "trade_throttle")
        state = throttle_snapshot.get("state", "unknown")
        lines.append(f"- Control: `{name}`")
        lines.append(f"- State: {state}")
        message = throttle_snapshot.get("message")
        if message:
            lines.append(f"- Message: {message}")
        reason = throttle_snapshot.get("reason")
        if reason:
            lines.append(f"- Reason: `{reason}`")
        metadata = throttle_snapshot.get("metadata")
        if isinstance(metadata, Mapping):
            retry = metadata.get("retry_at")
            if retry:
                lines.append(f"- Retry at: {retry}")
            context = metadata.get("context")
            if isinstance(context, Mapping) and context:
                context_pairs = ", ".join(f"{key}={value}" for key, value in sorted(context.items()))
                lines.append(f"- Context: {context_pairs}")
            max_trades = metadata.get("max_trades")
            remaining_trades = metadata.get("remaining_trades")
            if max_trades is not None or remaining_trades is not None:
                remaining_display = _format_optional(remaining_trades)
                max_display = _format_optional(max_trades)
                utilisation = metadata.get("window_utilisation")
                utilisation_display = (
                    f" ({_format_percentage(utilisation, precision=1)} utilised)"
                    if utilisation is not None
                    else ""
                )
                lines.append(
                    f"- Capacity: {remaining_display} / {max_display} trades remaining"
                    f"{utilisation_display}"
                )
            retry_in = metadata.get("retry_in_seconds")
            if retry_in is not None:
                lines.append(f"- Retry in: {_format_seconds(retry_in)}")
            window_reset_in = metadata.get("window_reset_in_seconds")
            window_reset_at = metadata.get("window_reset_at")
            if window_reset_in is not None or window_reset_at:
                label = _format_seconds(window_reset_in)
                if window_reset_at:
                    lines.append(
                        f"- Window resets in: {label} (at {window_reset_at})"
                    )
                else:
                    lines.append(f"- Window resets in: {label}")
            scope = metadata.get("scope")
            if isinstance(scope, Mapping) and scope:
                lines.append("- Scope:")
                lines.extend(_render_mapping_lines(scope, prefix="  -"))
            scope_key = throttle_snapshot.get("scope_key")
            if isinstance(scope_key, Iterable) and not isinstance(scope_key, (str, bytes)):
                scope_key_display = list(scope_key)
                lines.append(f"- Scope key: {scope_key_display}")

    throughput = stats.get("throughput")
    if isinstance(throughput, Mapping):
        lines.append("")
        lines.append("## Throughput window")
        lines.append("| Metric | Value |")
        lines.append("| --- | --- |")
        lines.append(f"| Samples | {_format_optional(throughput.get('samples'))} |")
        lines.append(
            f"| Avg processing (ms) | {_format_float(throughput.get('avg_processing_ms'))} |")
        lines.append(
            f"| P95 processing (ms) | {_format_float(throughput.get('p95_processing_ms'))} |")
        lines.append(
            f"| Max processing (ms) | {_format_float(throughput.get('max_processing_ms'))} |")
        lines.append(
            f"| Avg order latency (ms) | {_format_float(stats.get('avg_latency_ms'))} |")
        lines.append(
            f"| P99 order latency (ms) | {_format_float(stats.get('latency_p99_ms'))} |")
        lines.append(
            f"| Max order latency (ms) | {_format_float(stats.get('max_latency_ms'))} |")
        lines.append(f"| Avg lag (ms) | {_format_float(throughput.get('avg_lag_ms'))} |")
        lines.append(f"| Max lag (ms) | {_format_float(throughput.get('max_lag_ms'))} |")
        lines.append(
            f"| Throughput (per min) | {_format_float(throughput.get('throughput_per_min'))} |")

    backlog = stats.get("backlog")
    if isinstance(backlog, Mapping):
        lines.append("")
        lines.append("## Event backlog")
        lines.append(f"- Samples: {_format_optional(backlog.get('samples'))}")
        lines.append(
            f"- Threshold (ms): {_format_float(backlog.get('threshold_ms'))}"
        )
        lines.append(f"- Max lag (ms): {_format_float(backlog.get('max_lag_ms'))}")
        lines.append(f"- Avg lag (ms): {_format_float(backlog.get('avg_lag_ms'))}")
        lines.append(f"- Latest lag (ms): {_format_float(backlog.get('latest_lag_ms'))}")
        lines.append(f"- P95 lag (ms): {_format_float(backlog.get('p95_lag_ms'))}")
        breaches = backlog.get("breaches")
        lines.append(f"- Breaches: {_format_optional(breaches)}")
        lines.append(
            f"- Breach rate: {_format_percentage(backlog.get('breach_rate'))}"
        )
        lines.append(
            f"- Max breach streak: {_format_optional(backlog.get('max_breach_streak'))}"
        )
        worst_breach = backlog.get("worst_breach_ms")
        if worst_breach not in (None, ""):
            lines.append(f"- Worst breach (ms): {_format_float(worst_breach)}")
        last_breach = backlog.get("last_breach_at")
        if last_breach:
            lines.append(f"- Last breach: {last_breach}")
        healthy = backlog.get("healthy")
        if isinstance(healthy, bool):
            lines.append(f"- Healthy: {'Yes' if healthy else 'No'}")

    resource = stats.get("resource_usage")
    if isinstance(resource, Mapping):
        lines.append("")
        lines.append("## Resource usage snapshot")
        lines.append(f"- Sampled at: {_format_optional(resource.get('timestamp'))}")
        lines.append(f"- CPU (%): {_format_float(resource.get('cpu_percent'))}")
        lines.append(f"- Memory (MB): {_format_float(resource.get('memory_mb'))}")
        lines.append(
            f"- Memory (%): {_format_float(resource.get('memory_percent'))}"
        )
        lines.append(f"- IO read (MB): {_format_float(resource.get('io_read_mb'))}")
        lines.append(f"- IO write (MB): {_format_float(resource.get('io_write_mb'))}")
        lines.append(
            f"- IO read ops: {_format_optional(resource.get('io_read_count'))}"
        )
        lines.append(
            f"- IO write ops: {_format_optional(resource.get('io_write_count'))}"
        )

    return "\n".join(lines) + "\n"


def build_performance_health_report(assessment: Mapping[str, object]) -> str:
    """Render a Markdown summary of performance health checks."""

    lines = ["# Performance health assessment", ""]

    healthy = assessment.get("healthy")
    status = "Healthy" if healthy else "Unhealthy"
    lines.append(f"- Overall status: {status}")

    throughput = assessment.get("throughput")
    if isinstance(throughput, Mapping):
        lines.append("")
        lines.append("## Throughput")
        lines.append(
            f"- Healthy: {_format_bool(throughput.get('healthy'))}"
        )
        lines.append(
            f"- Max processing (ms): {_format_float(throughput.get('max_processing_ms'))}"
        )
        lines.append(
            f"- Max lag (ms): {_format_float(throughput.get('max_lag_ms'))}"
        )
        lines.append(f"- Samples: {_format_optional(throughput.get('samples'))}")
    else:
        lines.append("")
        lines.append("## Throughput")
        lines.append("- No throughput data available")

    backlog = assessment.get("backlog")
    if isinstance(backlog, Mapping):
        lines.append("")
        lines.append("## Backlog")
        lines.append(f"- Evaluated: {_format_bool(backlog.get('evaluated'))}")
        lines.append(f"- Healthy: {_format_bool(backlog.get('healthy'))}")
        lines.append(
            f"- Threshold (ms): {_format_float(backlog.get('threshold_ms'))}"
        )
        lines.append(f"- Max lag (ms): {_format_float(backlog.get('max_lag_ms'))}")
        lines.append(f"- P95 lag (ms): {_format_float(backlog.get('p95_lag_ms'))}")
        lines.append(f"- Samples: {_format_optional(backlog.get('samples'))}")
        lines.append(
            f"- Breach rate: {_format_percentage(backlog.get('breach_rate'))}"
        )
        lines.append(
            f"- Max breach streak: {_format_optional(backlog.get('max_breach_streak'))}"
        )
    else:
        lines.append("")
        lines.append("## Backlog")
        lines.append("- No backlog data available")

    resource = assessment.get("resource")
    if isinstance(resource, Mapping):
        lines.append("")
        lines.append("## Resource utilisation")
        lines.append(f"- Healthy: {_format_bool(resource.get('healthy'))}")
        lines.append(f"- Status: {resource.get('status', '—')}")
        limits = resource.get("limits")
        if isinstance(limits, Mapping):
            lines.append("- Limits:")
            lines.extend(_render_mapping_lines(limits, prefix="  -"))
        sample = resource.get("sample")
        if isinstance(sample, Mapping):
            lines.append("- Sample:")
            lines.extend(_render_mapping_lines(sample, prefix="  -"))
        violations = resource.get("violations")
        if violations:
            lines.append("- Violations:")
            violation_lines = _render_mapping_lines(violations, prefix="  -")
            lines.extend(violation_lines)
        else:
            lines.append("- Violations: None")
    else:
        lines.append("")
        lines.append("## Resource utilisation")
        lines.append("- No resource data available")

    throttle = assessment.get("throttle")
    if isinstance(throttle, Mapping):
        lines.append("")
        lines.append("## Trade throttle")
        lines.append(f"- Active: {_format_bool(throttle.get('active'))}")
        reason = throttle.get("reason")
        if reason:
            lines.append(f"- Reason: `{reason}`")
        message = throttle.get("message")
        if message:
            lines.append(f"- Message: {message}")
        retry_at = throttle.get("retry_at")
        if retry_at:
            lines.append(f"- Retry at: {retry_at}")
        remaining = throttle.get("remaining_trades")
        max_trades = throttle.get("max_trades")
        if remaining is not None or max_trades is not None:
            lines.append(
                f"- Remaining trades: {_format_optional(remaining)} / {_format_optional(max_trades)}"
            )
        utilisation = throttle.get("window_utilisation")
        if utilisation is not None:
            lines.append(
                f"- Window utilisation: {_format_percentage(utilisation, precision=1)}"
            )
        retry_in = throttle.get("retry_in_seconds")
        if retry_in is not None:
            lines.append(f"- Retry in: {_format_seconds(retry_in)}")
        window_reset_in = throttle.get("window_reset_in_seconds")
        if window_reset_in is not None:
            window_reset_at = throttle.get("window_reset_at")
            if window_reset_at:
                lines.append(
                    f"- Window resets in: {_format_seconds(window_reset_in)} (at {window_reset_at})"
                )
            else:
                lines.append(
                    f"- Window resets in: {_format_seconds(window_reset_in)}"
                )
        context = throttle.get("context")
        if isinstance(context, Mapping):
            lines.append("- Context:")
            lines.extend(_render_mapping_lines(context, prefix="  -"))
        scope = throttle.get("scope")
        if isinstance(scope, Mapping) and scope:
            lines.append("- Scope:")
            lines.extend(_render_mapping_lines(scope, prefix="  -"))
        scope_key = throttle.get("scope_key")
        if isinstance(scope_key, Iterable) and not isinstance(scope_key, (str, bytes)):
            lines.append(f"- Scope key: {list(scope_key)}")
    return "\n".join(lines) + "\n"


__all__ = [
    "build_execution_performance_report",
    "build_performance_health_report",
]
