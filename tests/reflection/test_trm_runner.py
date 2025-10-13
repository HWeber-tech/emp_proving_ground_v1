import json
from pathlib import Path

import jsonschema

from src.reflection.trm.adapter import RIMInputAdapter
from src.reflection.trm.config import ModelConfig, RIMRuntimeConfig, TelemetryConfig
from src.reflection.trm.encoder import RIMEncoder
from src.reflection.trm.model import TRMModel
from src.reflection.trm.runner import TRMRunner

SCHEMA_PATH = Path("interfaces/rim_types.json")
EXAMPLE_DIARIES = Path("docs/examples/trm_exit_drill_diaries.jsonl")


def _prepare_diaries(tmp_path: Path) -> Path:
    diaries_dir = tmp_path / "artifacts" / "diaries"
    diaries_dir.mkdir(parents=True, exist_ok=True)
    target = diaries_dir / "diaries-0001.jsonl"
    target.write_text(EXAMPLE_DIARIES.read_text(), encoding="utf-8")
    return diaries_dir


def test_encoder_emits_expected_features(tmp_path: Path) -> None:
    diaries_dir = _prepare_diaries(tmp_path)
    adapter = RIMInputAdapter(diaries_dir, "diaries-*.jsonl", window_minutes=1440)
    batch = adapter.load_batch()
    assert batch is not None, "expected batch to load"

    encoder = RIMEncoder()
    encodings = encoder.encode(batch.entries)
    assert encodings, "expected at least one strategy encoding"
    feature_names = set(encodings[0].features.keys())
    assert feature_names == {
        "count_log",
        "mean_pnl_scaled",
        "pnl_std_scaled",
        "risk_rate",
        "win_rate",
        "loss_rate",
        "volatility_mean",
        "spread_mean_pips",
        "belief_confidence_mean",
        "pnl_trend_scaled",
        "drawdown_ratio",
    }


def test_runner_emits_schema_compliant_suggestions(tmp_path: Path) -> None:
    diaries_dir = _prepare_diaries(tmp_path)
    publish_dir = tmp_path / "suggestions"
    telemetry_dir = tmp_path / "logs"
    lock_path = tmp_path / "locks" / "rim.lock"
    governance_dir = tmp_path / "governance"
    queue_path = governance_dir / "queue.jsonl"
    digest_path = governance_dir / "digest.json"
    markdown_path = governance_dir / "digest.md"

    config = RIMRuntimeConfig(
        diaries_dir=diaries_dir,
        diary_glob="diaries-*.jsonl",
        window_minutes=1440,
        min_entries=1,
        suggestion_cap=5,
        confidence_floor=0.5,
        publish_channel=f"file://{publish_dir}",
        telemetry=TelemetryConfig(log_dir=telemetry_dir),
        model=ModelConfig(path=None, temperature=1.0),
        lock_path=lock_path,
        governance_queue_path=queue_path,
        governance_digest_path=digest_path,
        governance_markdown_path=markdown_path,
    )

    model = TRMModel.load(config.model.path, temperature=config.model.temperature)
    runner = TRMRunner(config, model, config_hash="test-config")
    result = runner.run()

    assert result.skipped_reason is None, f"unexpected skip: {result.skipped_reason}"
    assert result.suggestions_count > 0, "expected suggestions to be emitted"
    assert result.suggestions_path is not None and result.suggestions_path.exists()
    assert result.run_id is not None and result.run_id.strip(), "expected run_id"

    schema_doc = json.loads(SCHEMA_PATH.read_text())
    validator = jsonschema.Draft7Validator(
        schema_doc["definitions"]["RIMSuggestion"],
        resolver=jsonschema.RefResolver.from_schema(schema_doc),
    )

    suggestions = [json.loads(line) for line in result.suggestions_path.read_text().splitlines() if line]
    assert suggestions, "suggestions file should not be empty"
    for suggestion in suggestions:
        validator.validate(suggestion)
        assert suggestion["confidence"] >= 0.5
        assert suggestion["type"] in {"WEIGHT_ADJUST", "STRATEGY_FLAG", "EXPERIMENT_PROPOSAL"}

    # Telemetry log should exist
    assert telemetry_dir.exists()
    assert any(telemetry_dir.iterdir()), "expected telemetry log file"

    # Governance artifacts should be emitted
    assert digest_path.exists(), "expected governance digest"
    digest = json.loads(digest_path.read_text())
    assert digest["run_id"] == result.run_id
    assert digest["suggestion_count"] == result.suggestions_count

    assert markdown_path.exists(), "expected governance markdown"
    markdown = markdown_path.read_text()
    assert result.run_id in markdown

    assert queue_path.exists(), "expected governance queue"
    queue_lines = [
        json.loads(line)
        for line in queue_path.read_text().splitlines()
        if line.strip()
    ]
    assert len(queue_lines) == result.suggestions_count
    for entry in queue_lines:
        governance_meta = entry.get("governance")
        assert governance_meta is not None
        assert governance_meta["run_id"] == result.run_id
