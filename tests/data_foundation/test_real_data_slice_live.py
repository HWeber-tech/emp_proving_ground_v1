from __future__ import annotations

import numpy as np
import pytest

from src.data_foundation.persist.timescale import TimescaleConnectionSettings
from src.data_integration.real_data_slice import RealDataSliceConfig, run_real_data_slice

pytest.importorskip("yfinance")


@pytest.mark.integration
@pytest.mark.slow
def test_run_real_data_slice_with_yahoo_provider(tmp_path) -> None:
    db_path = tmp_path / "real_slice_live.db"
    settings = TimescaleConnectionSettings(
        url=f"sqlite:///{db_path}",
        application_name="pytest-live-data",
    )
    config = RealDataSliceConfig(
        symbol="AAPL",
        provider="yahoo",
        source="yahoo",
        lookback_days=5,
        belief_id="pytest-live-bundle",
    )

    try:
        outcome = run_real_data_slice(config=config, settings=settings)
    except RuntimeError as exc:
        pytest.skip(f"Live provider returned no data: {exc}")

    assert outcome.ingest_result.rows_written > 0
    assert not outcome.market_data.empty
    assert outcome.market_data["symbol"].str.upper().eq("AAPL").all()
    assert outcome.market_data["timestamp"].dt.tz is not None

    snapshot = outcome.sensory_snapshot
    integrated = snapshot["integrated_signal"]
    assert snapshot["symbol"] == "AAPL"
    assert np.isfinite(integrated.strength)
    assert np.isfinite(integrated.confidence)

    belief_state = outcome.belief_state
    assert belief_state.belief_id == "pytest-live-bundle"
    assert belief_state.posterior.support == 1
    assert np.isfinite(belief_state.posterior.strength)
    assert np.isfinite(belief_state.posterior.confidence)

    regime_signal = outcome.regime_signal
    assert regime_signal.signal_id.endswith("regime")
    assert np.isfinite(float(regime_signal.metadata.get("volatility", 0.0)))

    if outcome.calibration is not None:
        assert outcome.calibration.learning_rate > 0
