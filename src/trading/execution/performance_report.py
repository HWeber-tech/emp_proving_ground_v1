"""Utilities for rendering execution throttle and throughput diagnostics."""

from __future__ import annotations

from typing import Mapping


def _format_float(value: object, *, precision: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number:.{precision}f}"


def _format_optional(value: object) -> str:
    if value in (None, "", "-", 0):
        return "—"
    return str(value)


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
        breaches = backlog.get("breaches")
        lines.append(f"- Breaches: {_format_optional(breaches)}")
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

    return "\n".join(lines) + "\n"


__all__ = ["build_execution_performance_report"]
