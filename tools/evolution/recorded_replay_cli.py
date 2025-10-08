"""CLI to evaluate genomes against recorded sensory replay datasets."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

from src.core.genome import NoOpGenomeProvider
from src.evolution.evaluation.datasets import load_recorded_snapshots
from src.evolution.evaluation.recorded_replay import RecordedSensoryEvaluator
from src.evolution.evaluation.telemetry import RecordedReplayTelemetrySnapshot, summarise_recorded_replay

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate adaptive genomes against recorded sensory datasets to provide "
            "deterministic evidence aligned with the sensory + evolution roadmap goals."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Path to the JSONL file containing recorded sensory snapshots.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional maximum number of snapshots to load from the dataset.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Abort on malformed JSON records instead of skipping them.",
    )
    parser.add_argument(
        "--genome",
        action="append",
        help=(
            "Genome definition encoded as JSON. May be passed multiple times. "
            "Each definition should include an 'id' and optional 'parameters'."
        ),
    )
    parser.add_argument(
        "--genome-file",
        action="append",
        type=Path,
        help=(
            "Path to a JSON file containing genome definitions. Accepts either "
            "a list of objects or an object mapping identifiers to parameter maps."
        ),
    )
    parser.add_argument(
        "--dataset-id",
        help="Override dataset identifier embedded in telemetry outputs.",
    )
    parser.add_argument(
        "--evaluation-id",
        help="Identifier for this evaluation run recorded in telemetry outputs.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        help="Minimum sensory confidence enforced during replay evaluation.",
    )
    parser.add_argument(
        "--warn-drawdown",
        type=float,
        default=0.15,
        help="Drawdown threshold (%) that escalates telemetry status to WARN (default: 0.15).",
    )
    parser.add_argument(
        "--alert-drawdown",
        type=float,
        default=0.25,
        help="Drawdown threshold (%) that escalates telemetry status to ALERT (default: 0.25).",
    )
    parser.add_argument(
        "--format",
        choices=("json", "markdown"),
        default="json",
        help="Output format for the evaluation summary (default: json).",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indentation level for JSON output (ignored for markdown).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the rendered evaluation summary.",
    )
    return parser


def _emit(text: str, output: Path | None) -> None:
    if output is None:
        print(text)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f"{text}\n", encoding="utf-8")


def _load_snapshots(path: Path, *, limit: int | None, strict: bool) -> list:
    if not path.exists():
        raise FileNotFoundError(f"recorded dataset not found at {path}")
    snapshots = load_recorded_snapshots(path, limit=limit, strict=strict)
    if not snapshots:
        raise ValueError("recorded dataset did not yield any snapshots")
    return snapshots


def _parse_json_payload(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON payload: {exc}") from exc


def _normalise_genome_payload(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        if "parameters" in payload or any(k in payload for k in ("id", "genome_id", "name")):
            return [payload]
        genomes: list[Mapping[str, Any]] = []
        for identifier, params in payload.items():
            if isinstance(params, Mapping):
                genomes.append({"id": identifier, "parameters": params})
        if genomes:
            return genomes
        raise ValueError("unable to interpret genome mapping payload")
    if isinstance(payload, Iterable) and not isinstance(payload, (str, bytes)):
        genomes: list[Mapping[str, Any]] = []
        for item in payload:
            if not isinstance(item, Mapping):
                raise ValueError("genome list entries must be objects")
            genomes.append(item)
        if genomes:
            return genomes
    raise ValueError("unsupported genome payload structure")


def _load_genome_file(path: Path) -> list[Mapping[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"genome definition file not found at {path}")
    text = path.read_text(encoding="utf-8")
    try:
        payload = _parse_json_payload(text)
    except ValueError as exc:
        genomes: list[Mapping[str, Any]] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                item = _parse_json_payload(stripped)
            except ValueError as line_error:
                raise ValueError(f"invalid JSON on line {line_number} in {path}: {line_error}") from line_error
            if not isinstance(item, Mapping):
                raise ValueError(f"line {line_number} in {path} does not define an object")
            genomes.append(item)
        if not genomes:
            raise ValueError(f"no genome definitions found in {path}")
        return genomes
    return _normalise_genome_payload(payload)


def _collect_genomes(
    inline_genomes: Sequence[str] | None,
    genome_files: Sequence[Path] | None,
) -> list[Mapping[str, Any]]:
    definitions: list[Mapping[str, Any]] = []
    if genome_files:
        for file_path in genome_files:
            definitions.extend(_load_genome_file(file_path))
    if inline_genomes:
        for raw in inline_genomes:
            payload = _parse_json_payload(raw)
            definitions.extend(_normalise_genome_payload(payload))
    if not definitions:
        raise ValueError("at least one genome definition must be provided via --genome or --genome-file")
    return definitions


def _coerce_mapping(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    result: MutableMapping[str, Any] = {}
    for key, value in payload.items():
        try:
            result[str(key)] = value
        except Exception:
            continue
    return dict(result)


def _extract_genome_definition(raw: Mapping[str, Any], index: int) -> dict[str, Any]:
    genome_id: str | None = None
    for key in ("id", "genome_id", "identifier", "name"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            genome_id = value.strip()
            break
    if genome_id is None:
        genome_id = f"genome-{index + 1}"

    parameters_payload = raw.get("parameters") if isinstance(raw.get("parameters"), Mapping) else None
    if parameters_payload is None:
        # Fallback: treat numeric fields (excluding known metadata keys) as parameters
        parameters_payload = {}
        for key, value in raw.items():
            if key in {"id", "genome_id", "identifier", "name", "metadata", "dataset_id", "evaluation_id", "generation", "species_type"}:
                continue
            try:
                parameters_payload[key] = float(value)  # type: ignore[arg-type]
            except Exception:
                continue
    parameters = {}
    if isinstance(parameters_payload, Mapping):
        for key, value in parameters_payload.items():
            try:
                parameters[str(key)] = float(value)  # type: ignore[arg-type]
            except Exception:
                continue
    generation = raw.get("generation")
    species_type = raw.get("species_type")
    metadata = _coerce_mapping(raw.get("metadata"))
    dataset_override = raw.get("dataset_id")
    evaluation_override = raw.get("evaluation_id")
    return {
        "id": genome_id,
        "parameters": parameters,
        "generation": generation,
        "species_type": species_type,
        "metadata": metadata,
        "dataset_id": dataset_override if isinstance(dataset_override, str) else None,
        "evaluation_id": evaluation_override if isinstance(evaluation_override, str) else None,
    }


def _render_json(
    dataset: Path,
    snapshots_loaded: int,
    evaluations: Sequence[RecordedReplayTelemetrySnapshot],
    *,
    indent: int,
    warn_drawdown: float,
    alert_drawdown: float,
    min_confidence: float | None,
) -> str:
    payload: dict[str, Any] = {
        "dataset": str(dataset),
        "snapshots_loaded": snapshots_loaded,
        "min_confidence": min_confidence,
        "warn_drawdown": warn_drawdown,
        "alert_drawdown": alert_drawdown,
        "evaluations": [snapshot.as_dict() for snapshot in evaluations],
    }
    return json.dumps(payload, indent=indent, sort_keys=True)


def _render_markdown(
    dataset: Path,
    snapshots_loaded: int,
    evaluations: Sequence[RecordedReplayTelemetrySnapshot],
    *,
    warn_drawdown: float,
    alert_drawdown: float,
    min_confidence: float | None,
) -> str:
    lines = [
        "# Recorded sensory replay evaluation",
        "",
        f"- Dataset: {dataset}",
        f"- Snapshots analysed: {snapshots_loaded}",
        f"- Warn drawdown threshold: {warn_drawdown:.2%}",
        f"- Alert drawdown threshold: {alert_drawdown:.2%}",
    ]
    if min_confidence is not None:
        lines.append(f"- Minimum confidence: {min_confidence:.2f}")
    lines.append("")
    for snapshot in evaluations:
        header = f"## Genome `{snapshot.genome_id}` — status: {snapshot.status.upper()}"
        lines.extend((header, "", snapshot.to_markdown(), ""))
    return "\n".join(lines).strip() + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        snapshots = _load_snapshots(args.dataset, limit=args.limit, strict=args.strict)
    except (FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        return 1

    try:
        genome_definitions = _collect_genomes(args.genome, args.genome_file)
    except (ValueError, FileNotFoundError) as exc:
        logger.error(str(exc))
        return 1

    provider = NoOpGenomeProvider()
    evaluator = RecordedSensoryEvaluator(snapshots)
    telemetry_snapshots: list[RecordedReplayTelemetrySnapshot] = []

    for index, raw_definition in enumerate(genome_definitions):
        genome_definition = _extract_genome_definition(raw_definition, index)
        genome_id = genome_definition["id"]
        genome = provider.new_genome(
            genome_id,
            genome_definition["parameters"],
            generation=genome_definition.get("generation", 0) or 0,
            species_type=genome_definition.get("species_type"),
        )
        try:
            result = evaluator.evaluate(genome, min_confidence=args.min_confidence)
        except Exception:  # pragma: no cover - defensive guard around evaluation
            logger.exception("Replay evaluation failed for genome %s", genome_id)
            return 1

        metadata = genome_definition.get("metadata") or {}
        dataset_id = genome_definition.get("dataset_id") or args.dataset_id
        evaluation_id = genome_definition.get("evaluation_id") or args.evaluation_id
        telemetry = summarise_recorded_replay(
            result,
            genome_id=genome_id,
            dataset_id=dataset_id,
            evaluation_id=evaluation_id,
            parameters=getattr(genome, "parameters", genome_definition["parameters"]),
            metadata=metadata,
            warn_drawdown=args.warn_drawdown,
            alert_drawdown=args.alert_drawdown,
        )
        telemetry_snapshots.append(telemetry)

    if args.format == "json":
        rendered = _render_json(
            args.dataset,
            len(snapshots),
            telemetry_snapshots,
            indent=args.indent,
            warn_drawdown=args.warn_drawdown,
            alert_drawdown=args.alert_drawdown,
            min_confidence=args.min_confidence,
        )
    else:
        rendered = _render_markdown(
            args.dataset,
            len(snapshots),
            telemetry_snapshots,
            warn_drawdown=args.warn_drawdown,
            alert_drawdown=args.alert_drawdown,
            min_confidence=args.min_confidence,
        )

    _emit(rendered, args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
