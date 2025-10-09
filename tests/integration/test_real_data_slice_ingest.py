from __future__ import annotations

from pathlib import Path

import numpy as np

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
