from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


def _load_module():
    module_path = Path(__file__).resolve().parents[2] / "scripts" / "data_bootstrap.py"
    spec = importlib.util.spec_from_file_location("scripts.data_bootstrap", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def test_data_bootstrap_generates_pricing_artifacts(tmp_path):
    module = _load_module()

    raw = pd.DataFrame(
        [
            {
                "timestamp": "2024-01-01T00:00:00Z",
                "symbol": "EURUSD",
                "open": 1.05,
                "high": 1.06,
                "low": 1.04,
                "close": 1.055,
                "adj_close": 1.055,
                "volume": 1234,
            },
            {
                "timestamp": "2024-01-02T00:00:00Z",
                "symbol": "EURUSD",
                "open": 1.06,
                "high": 1.07,
                "low": 1.05,
                "close": 1.065,
                "adj_close": 1.065,
                "volume": 1500,
            },
        ]
    )

    source_path = tmp_path / "input.csv"
    raw.to_csv(source_path, index=False)

    output_dir = tmp_path / "output"
    exit_code = module.main(
        [
            "--vendor",
            "file",
            "--source-path",
            str(source_path),
            "--symbols",
            "EURUSD",
            "--output",
            str(output_dir),
            "--format",
            "csv",
            "--minimum-coverage",
            "0.1",
        ]
    )

    assert exit_code == 0

    dataset_path = output_dir / "pricing_file_1d.csv"
    assert dataset_path.exists()
    dataset = pd.read_csv(dataset_path)
    assert list(dataset.columns) == [
        "timestamp",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "source",
    ]
    assert set(dataset["symbol"]) == {"EURUSD"}

    metadata_path = output_dir / "pricing_metadata.json"
    issues_path = output_dir / "pricing_issues.json"
    assert metadata_path.exists()
    assert issues_path.exists()

    metadata = json.loads(metadata_path.read_text())
    assert metadata["row_count"] == 2
    assert metadata["symbols"] == ["EURUSD"]

    issues = json.loads(issues_path.read_text())
    assert "issues" in issues


def test_fail_on_quality_surfaces_error(tmp_path):
    module = _load_module()

    empty_frame = pd.DataFrame(
        columns=[
            "timestamp",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
        ]
    )
    source_path = tmp_path / "empty.csv"
    empty_frame.to_csv(source_path, index=False)

    output_dir = tmp_path / "output"
    exit_code = module.main(
        [
            "--vendor",
            "file",
            "--source-path",
            str(source_path),
            "--symbols",
            "EURUSD",
            "--output",
            str(output_dir),
            "--fail-on-quality",
            "--minimum-coverage",
            "0.1",
            "--format",
            "csv",
        ]
    )

    assert exit_code == 2

    issues_path = output_dir / "pricing_issues.json"
    assert issues_path.exists()
    issues = json.loads(issues_path.read_text())["issues"]
    assert any(issue["code"] == "no_data" for issue in issues)
