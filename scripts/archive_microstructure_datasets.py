"""CLI for archiving microstructure datasets into tiered storage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import yaml

from src.data_foundation.persist import DatasetPolicy, TieredDatasetArchiver


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        nargs="?",
        default="config/operational/microstructure_archival.yaml",
        type=Path,
        help="Path to the archival policy YAML configuration.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/microstructure/archive_report.json"),
        help="Destination for a JSON summary report.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Calculate archival actions without copying or deleting files.",
    )
    return parser.parse_args()


def _load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> int:
    args = _parse_args()
    config = _load_config(args.config)

    metadata_dir = Path(config.get("metadata_dir", "artifacts/microstructure/metadata"))
    archiver = TieredDatasetArchiver(metadata_dir=metadata_dir)

    datasets = config.get("datasets", [])
    report: Dict[str, Any] = {
        "config": str(args.config),
        "dry_run": args.dry_run,
        "results": [],
    }

    for dataset_cfg in datasets:
        dataset_name = dataset_cfg["name"]
        source_dir = Path(dataset_cfg["source"])
        hot_dir = Path(dataset_cfg["hot"])
        cold_path = dataset_cfg.get("cold")
        cold_dir = Path(cold_path) if cold_path else None
        description = dataset_cfg.get("description")
        policy = DatasetPolicy(
            hot_retention_days=int(dataset_cfg["hot_retention_days"]),
            cold_retention_days=(
                int(dataset_cfg["cold_retention_days"])
                if dataset_cfg.get("cold_retention_days") is not None
                else None
            ),
            description=description,
        )

        result = archiver.archive_dataset(
            dataset=dataset_name,
            source_dir=source_dir,
            hot_dir=hot_dir,
            policy=policy,
            cold_dir=cold_dir,
            dry_run=args.dry_run,
        )
        report["results"].append(result.to_dict())

        status = "missing" if result.missing_source else "ok"
        print(f"[{status}] {dataset_name}: hot={len(result.hot)} cold={len(result.cold)} expired={len(result.expired)}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
