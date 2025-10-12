from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_foundation.persist.timescale import TimescaleConnectionSettings
from src.data_integration.real_data_slice import RealDataSliceConfig, run_real_data_slice


def test_real_data_slice_ingest_and_belief(tmp_path) -> None:
    db_path = tmp_path / "timescale_slice.db"
    settings = TimescaleConnectionSettings(
        url=f"sqlite:///{db_path}",
        application_name="pytest-real-slice",
    )
    config = RealDataSliceConfig(
        csv_path=Path("tests/data/eurusd_daily_slice.csv"),
        symbol="EURUSD",
        belief_id="pytest-real-data",
    )

    outcome = run_real_data_slice(config=config, settings=settings)

    assert outcome.ingest_result.rows_written == len(outcome.market_data)
    assert "volatility" in outcome.market_data.columns
    assert outcome.market_data["timestamp"].dt.tz is not None

    snapshot = outcome.sensory_snapshot
    assert snapshot["symbol"] == "EURUSD"
    integrated = snapshot["integrated_signal"]
    assert np.isfinite(integrated.strength)
    assert np.isfinite(integrated.confidence)

    belief_state = outcome.belief_state
    assert belief_state.symbol == "EURUSD"
    assert belief_state.posterior.support == 1
    assert belief_state.posterior.strength == integrated.strength
    assert belief_state.posterior.confidence == integrated.confidence
    assert len(belief_state.features) >= 5


def test_real_data_slice_ingest_from_provider(tmp_path) -> None:
    db_path = tmp_path / "timescale_provider.db"
    settings = TimescaleConnectionSettings(
        url=f"sqlite:///{db_path}",
        application_name="pytest-provider-slice",
    )
    config = RealDataSliceConfig(
        symbol="EURUSD",
        provider="yahoo",
        source="yahoo",
        lookback_days=3,
        belief_id="pytest-provider-data",
    )

    def _fake_fetch(symbols: list[str], lookback: int) -> pd.DataFrame:
        assert symbols == ["EURUSD"]
        assert lookback == 3
        dates = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
        return pd.DataFrame(
            {
                "date": dates,
                "symbol": ["EURUSD"] * 3,
                "open": [1.1, 1.11, 1.12],
                "high": [1.12, 1.13, 1.14],
                "low": [1.09, 1.10, 1.11],
                "close": [1.105, 1.12, 1.135],
                "adj_close": [1.105, 1.12, 1.135],
                "volume": [1_000, 1_100, 1_200],
            }
        )

    outcome = run_real_data_slice(
        config=config,
        settings=settings,
        fetch_daily=_fake_fetch,
    )

    assert outcome.ingest_result.rows_written == 3
    assert outcome.market_data["symbol"].str.upper().unique().tolist() == ["EURUSD"]
    assert "volatility" in outcome.market_data.columns

    snapshot = outcome.sensory_snapshot
    assert snapshot["symbol"] == "EURUSD"
    integrated = snapshot["integrated_signal"]
    assert np.isfinite(integrated.strength)
    assert np.isfinite(integrated.confidence)
