"""Tests for the performance baseline CLI helper."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.performance_baseline import BaselineOptions, _run, parse_args


def test_parse_args_defaults() -> None:
    options = parse_args([])
    assert options.trades == 3
    assert options.throttle_max_trades == 1
    assert options.throttle_window_seconds == pytest.approx(60.0)
    assert options.indent == 2
    assert options.output is None


def test_parse_args_custom(tmp_path: Path) -> None:
    output_path = tmp_path / "baseline.json"
    options = parse_args(
        [
            "--trades",
            "5",
            "--throttle-max-trades",
            "2",
            "--throttle-window-seconds",
            "45",
            "--throttle-cooldown-seconds",
            "15",
            "--throttle-min-spacing-seconds",
            "5",
            "--throttle-multiplier",
            "0.5",
            "--throttle-scope-field",
            "strategy_id",
            "--backlog-threshold-ms",
            "120",
            "--max-processing-ms",
            "150",
            "--max-lag-ms",
            "300",
            "--max-cpu-percent",
            "75",
            "--max-memory-mb",
            "512",
            "--max-memory-percent",
            "40",
            "--output",
            str(output_path),
            "--indent",
            "4",
        ]
    )

    assert options.trades == 5
    assert options.throttle_max_trades == 2
    assert options.throttle_window_seconds == pytest.approx(45.0)
    assert options.throttle_cooldown_seconds == pytest.approx(15.0)
    assert options.throttle_min_spacing_seconds == pytest.approx(5.0)
    assert options.throttle_multiplier == pytest.approx(0.5)
    assert options.throttle_scope_fields == ("strategy_id",)
    assert options.backlog_threshold_ms == pytest.approx(120.0)
    assert options.max_processing_ms == pytest.approx(150.0)
    assert options.max_lag_ms == pytest.approx(300.0)
    assert options.max_cpu_percent == pytest.approx(75.0)
    assert options.max_memory_mb == pytest.approx(512.0)
    assert options.max_memory_percent == pytest.approx(40.0)
    assert options.output == output_path.resolve()
    assert options.indent == 4


@pytest.mark.asyncio()
async def test_run_throttles_when_limit_exceeded() -> None:
    options = BaselineOptions(trades=4)
    baseline = await _run(options)

    throttle = baseline["throttle"]
    assert throttle["state"] in {"rate_limited", "cooldown"}
    assert isinstance(throttle.get("message"), str)
    assert "Throttled" in throttle.get("message", "")

    execution = baseline["execution"]
    throttle_snapshot = execution.get("trade_throttle", {})
    assert throttle_snapshot.get("state") in {"rate_limited", "cooldown"}
    assert throttle_snapshot.get("message", "").startswith("Throttled")

    metadata = baseline["options"]
    assert metadata["trades"] == 4
    assert metadata["trade_throttle"]["max_trades"] == 1


@pytest.mark.asyncio()
async def test_run_respects_throttle_configuration() -> None:
    options = BaselineOptions(trades=2, throttle_max_trades=5, throttle_window_seconds=30.0)
    baseline = await _run(options)

    throttle = baseline["throttle"]
    assert throttle["state"] == "open"
    assert throttle.get("active") is False
    assert throttle.get("message") is None

    execution = baseline["execution"]
    trade_throttle = execution.get("trade_throttle", {})
    assert trade_throttle.get("state") == "open"
    metadata = trade_throttle.get("metadata", {})
    assert metadata.get("remaining_trades") >= 3


@pytest.mark.asyncio()
async def test_metadata_reflects_limits() -> None:
    options = BaselineOptions(
        trades=3,
        throttle_max_trades=2,
        throttle_window_seconds=90.0,
        max_processing_ms=180.0,
        max_lag_ms=320.0,
        max_cpu_percent=70.0,
        max_memory_mb=256.0,
        max_memory_percent=25.0,
    )
    baseline = await _run(options)

    metadata = baseline["options"]
    assert metadata["max_processing_ms"] == pytest.approx(180.0)
    assert metadata["max_lag_ms"] == pytest.approx(320.0)
    assert metadata["max_cpu_percent"] == pytest.approx(70.0)
    assert metadata["max_memory_mb"] == pytest.approx(256.0)
    assert metadata["max_memory_percent"] == pytest.approx(25.0)
    throttle_config = metadata["trade_throttle"]
    assert throttle_config["max_trades"] == 2
    assert throttle_config["window_seconds"] == pytest.approx(90.0)
