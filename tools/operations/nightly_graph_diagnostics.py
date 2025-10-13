"""Nightly job that captures graph diagnostics and evaluates health thresholds."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.artifacts import archive_artifact
from src.operations.graph_diagnostics import (
    GraphEvaluation,
    GraphHealthStatus,
    GraphMetrics,
    GraphThresholds,
    compute_graph_metrics,
    evaluate_graph_metrics,
)
from src.understanding import UnderstandingDiagnosticsBuilder

logger = logging.getLogger("tools.operations.nightly_graph_diagnostics")

DEFAULT_RUN_ROOT = Path("artifacts/graph_diagnostics")


@dataclass(frozen=True)
class GraphDiagnosticsContext:
    """Resolved file paths for a graph diagnostics invocation."""

    run_dir: Path
    metrics_path: Path
    snapshot_path: Path
    dot_path: Path
    markdown_path: Path
    timestamp: datetime
    run_id: str


@dataclass(frozen=True)
class GraphDiagnosticsJobResult:
    """Outcome of a graph diagnostics run."""

    context: GraphDiagnosticsContext
    metrics: GraphMetrics
    evaluation: GraphEvaluation


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=DEFAULT_RUN_ROOT,
        help="Directory that will contain timestamped graph runs (default: artifacts/graph_diagnostics).",
    )
    parser.add_argument(
        "--timestamp",
        help="Optional UTC timestamp (YYYYmmddTHHMMSSZ) used for deterministic run directories.",
    )
    parser.add_argument(
        "--min-average-degree",
        type=float,
        default=GraphThresholds.min_average_degree,
        help="Minimum acceptable average degree before failing the job (default: 1.0).",
    )
    parser.add_argument(
        "--min-modularity",
        type=float,
        default=GraphThresholds.min_modularity,
        help="Minimum acceptable modularity before warning (default: 0.0).",
    )
    parser.add_argument(
        "--min-core-ratio",
        type=float,
        default=GraphThresholds.min_core_ratio,
        help="Lower bound for the fraction of core nodes (default: 0.2).",
    )
    parser.add_argument(
        "--max-core-ratio",
        type=float,
        default=GraphThresholds.max_core_ratio,
        help="Upper bound for the fraction of core nodes (default: 0.75).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Logging level for the orchestrator (default: INFO).",
    )
    return parser.parse_args(argv)


def _resolve_timestamp(value: str | None) -> datetime:
    if not value:
        return datetime.now(tz=timezone.utc)
    cleaned = value.strip()
    if not cleaned:
        return datetime.now(tz=timezone.utc)
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    candidate = datetime.fromisoformat(cleaned)
    if candidate.tzinfo is None:
        candidate = candidate.replace(tzinfo=timezone.utc)
    return candidate.astimezone(timezone.utc)


def _format_run_id(ts: datetime) -> str:
    return f"graph-{ts.strftime('%Y%m%dT%H%M%SZ')}"


def _prepare_run_context(run_root: Path, timestamp: datetime) -> GraphDiagnosticsContext:
    run_root = run_root.expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    run_id = _format_run_id(timestamp)
    run_dir = run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = run_dir / "graph_metrics.json"
    snapshot_path = run_dir / "graph_snapshot.json"
    dot_path = run_dir / "graph.dot"
    markdown_path = run_dir / "graph_summary.md"

    return GraphDiagnosticsContext(
        run_dir=run_dir,
        metrics_path=metrics_path,
        snapshot_path=snapshot_path,
        dot_path=dot_path,
        markdown_path=markdown_path,
        timestamp=timestamp,
        run_id=run_id,
    )


def _build_markdown(
    context: GraphDiagnosticsContext,
    metrics: GraphMetrics,
    evaluation: GraphEvaluation,
) -> str:
    lines = [
        f"# Graph Diagnostics Summary (run_id={context.run_id})",
        "",
        f"- Status: {evaluation.status.value.upper()}",
        f"- Timestamp: {context.timestamp.isoformat()}",
        f"- Average degree: {metrics.average_degree:.3f}",
        f"- Modularity: {metrics.modularity:.3f}",
        f"- Core ratio: {metrics.core_ratio:.3f}",
        "",
        "## Degree Histogram",
    ]
    if metrics.degree_histogram:
        for degree, count in metrics.degree_histogram.items():
            lines.append(f"- Degree {degree}: {count}")
    else:
        lines.append("- (no edges recorded)")

    lines.append("")
    lines.append("## Core Nodes")
    lines.append(", ".join(metrics.core_nodes) or "(none)")
    lines.append("")
    lines.append("## Periphery Nodes")
    lines.append(", ".join(metrics.periphery_nodes) or "(none)")

    if evaluation.messages:
        lines.append("")
        lines.append("## Alerts")
        for message in evaluation.messages:
            lines.append(f"- {message}")

    return "\n".join(lines)


def run_graph_diagnostics_job(
    *,
    run_root: Path,
    thresholds: GraphThresholds,
    timestamp: datetime | None = None,
) -> GraphDiagnosticsJobResult:
    ts = timestamp or datetime.now(tz=timezone.utc)
    context = _prepare_run_context(run_root, ts)

    builder = UnderstandingDiagnosticsBuilder()
    artifacts = builder.build()
    metrics = compute_graph_metrics(artifacts.graph)
    evaluation = evaluate_graph_metrics(metrics, thresholds)

    snapshot_payload = {
        "graph": artifacts.graph.as_dict(),
        "snapshot": artifacts.to_snapshot().as_dict(),
    }
    context.snapshot_path.write_text(json.dumps(snapshot_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    metrics_payload = {
        "run_id": context.run_id,
        "timestamp": context.timestamp.isoformat(),
        "metrics": metrics.as_dict(),
        "evaluation": evaluation.as_dict(),
    }
    context.metrics_path.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    context.dot_path.write_text(artifacts.graph.to_dot() + "\n", encoding="utf-8")
    context.markdown_path.write_text(_build_markdown(context, metrics, evaluation) + "\n", encoding="utf-8")

    archive_artifact(
        "graph_diagnostics",
        context.metrics_path,
        timestamp=context.timestamp,
        run_id=context.run_id,
        target_name=context.metrics_path.name,
    )
    archive_artifact(
        "graph_diagnostics",
        context.snapshot_path,
        timestamp=context.timestamp,
        run_id=context.run_id,
        target_name=context.snapshot_path.name,
    )
    archive_artifact(
        "graph_diagnostics",
        context.dot_path,
        timestamp=context.timestamp,
        run_id=context.run_id,
        target_name=context.dot_path.name,
    )
    archive_artifact(
        "graph_diagnostics",
        context.markdown_path,
        timestamp=context.timestamp,
        run_id=context.run_id,
        target_name=context.markdown_path.name,
    )

    if evaluation.status is GraphHealthStatus.fail:
        logger.error(
            "Graph diagnostics failed thresholds", extra={"messages": list(evaluation.messages)}
        )
    elif evaluation.status is GraphHealthStatus.warn:
        logger.warning(
            "Graph diagnostics triggered warnings", extra={"messages": list(evaluation.messages)}
        )
    else:
        logger.info("Graph diagnostics healthy", extra={"run_id": context.run_id})

    return GraphDiagnosticsJobResult(context=context, metrics=metrics, evaluation=evaluation)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level))

    thresholds = GraphThresholds(
        min_average_degree=args.min_average_degree,
        min_modularity=args.min_modularity,
        min_core_ratio=args.min_core_ratio,
        max_core_ratio=args.max_core_ratio,
    )

    try:
        timestamp = _resolve_timestamp(args.timestamp)
    except ValueError as exc:  # pragma: no cover - user input surface
        logger.error("Invalid timestamp", exc_info=exc)
        return 2

    result = run_graph_diagnostics_job(run_root=args.run_root, thresholds=thresholds, timestamp=timestamp)

    if result.evaluation.status is GraphHealthStatus.fail:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
