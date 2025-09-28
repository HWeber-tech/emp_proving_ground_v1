"""Reporting helpers for Evolution Lab experiments."""

from __future__ import annotations

from typing import Mapping, Sequence

__all__ = ["render_leaderboard_markdown"]


def _format_float(value: object, precision: int = 3) -> str:
    try:
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return "N/A"


def _format_bool(value: object) -> str:
    if value is None:
        return "N/A"
    return "✅" if bool(value) else "❌"


def render_leaderboard_markdown(
    manifests: Sequence[Mapping[str, object]],
    *,
    caption: str | None = None,
) -> str:
    """Render a Markdown leaderboard table from experiment manifests."""

    if not manifests:
        return "No experiments recorded yet.\n"

    sorted_manifests = sorted(
        manifests,
        key=lambda manifest: float(
            (manifest.get("best_metrics") or {}).get("fitness", float("-inf"))
        ),
        reverse=True,
    )

    header = (
        "| Experiment | Seed | Fitness | Sharpe | Sortino | Max Drawdown | Total Return | Short | Long | Risk | VaR Guard | Drawdown Guard |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
    )

    rows: list[str] = []
    for manifest in sorted_manifests:
        metrics = manifest.get("best_metrics") or {}
        genome = manifest.get("best_genome") or {}
        seed = manifest.get("seed")
        rows.append(
            "| {experiment} | {seed} | {fitness} | {sharpe} | {sortino} | {max_dd} | {total_return} | {short} | {long} | {risk} | {var_guard} | {dd_guard} |".format(
                experiment=manifest.get("experiment", "-"),
                seed="N/A" if seed is None else seed,
                fitness=_format_float(metrics.get("fitness")),
                sharpe=_format_float(metrics.get("sharpe")),
                sortino=_format_float(metrics.get("sortino")),
                max_dd=_format_float(metrics.get("max_drawdown")),
                total_return=_format_float(metrics.get("total_return")),
                short=_format_float(genome.get("short_window"), precision=0),
                long=_format_float(genome.get("long_window"), precision=0),
                risk=_format_float(genome.get("risk_fraction"), precision=2),
                var_guard=_format_bool(genome.get("use_var_guard")),
                dd_guard=_format_bool(genome.get("use_drawdown_guard")),
            )
        )

    body = header + "\n".join(rows) + "\n"

    if caption:
        return f"{caption}\n\n{body}"
    return body
