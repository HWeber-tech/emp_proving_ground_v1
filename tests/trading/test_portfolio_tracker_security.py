from __future__ import annotations

import logging
from pathlib import Path

import pytest

from src.trading.monitoring import portfolio_tracker


def test_portfolio_tracker_logs_invalid_state(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    store_root = tmp_path / "portfolio"
    store_root.mkdir()
    state_path = store_root / "portfolio_state.json"
    state_path.write_text("{ invalid", encoding="utf-8")

    class DummyStore:
        def __init__(self, base_dir: str = "data/portfolio") -> None:
            self.base_dir = str(store_root)

    monkeypatch.setattr(portfolio_tracker, "JSONStateStore", DummyStore)

    caplog.set_level(logging.WARNING)
    tracker = portfolio_tracker.PortfolioTracker()

    assert tracker.cash == 0.0
    assert tracker.positions == {}
    assert any("Failed to load portfolio state" in record.message for record in caplog.records)
