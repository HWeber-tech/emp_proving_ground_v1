from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data_foundation.persist.timescale import TimescaleConnectionSettings
from src.data_integration.real_data_slice import (
    RealDataSliceConfig,
    build_belief_sequence_from_market_data,
    run_real_data_slice,
)


def test_real_data_slice_generates_belief_sequence(tmp_path) -> None:
    db_path = tmp_path / "timescale_sequence.db"
    settings = TimescaleConnectionSettings(
        url=f"sqlite:///{db_path}",
        application_name="pytest-belief-sequence",
    )
    config = RealDataSliceConfig(
        csv_path=Path("tests/data/eurusd_daily_slice.csv"),
        symbol="EURUSD",
        belief_id="sequence-real-data",
    )

    outcome = run_real_data_slice(config=config, settings=settings)

    sequence = build_belief_sequence_from_market_data(
        market_data=outcome.market_data,
        symbol=config.symbol,
        belief_id=config.belief_id,
    )

    assert len(sequence.belief_states) == len(outcome.market_data)
    assert len(sequence.regime_signals) == len(outcome.market_data)
    assert sequence.snapshots[-1]["symbol"] == config.symbol

    final_state = sequence.belief_states[-1]
    assert final_state.posterior.support == len(outcome.market_data)
    assert np.isfinite(final_state.posterior.strength)
    assert np.isfinite(final_state.posterior.confidence)

    final_snapshot = sequence.snapshots[-1]
    expected_snapshot = outcome.sensory_snapshot
    assert abs(
        final_snapshot["integrated_signal"].strength
        - expected_snapshot["integrated_signal"].strength
    ) <= 0.05
    assert abs(
        final_snapshot["integrated_signal"].confidence
        - expected_snapshot["integrated_signal"].confidence
    ) <= 0.05
    assert final_state.generated_at == final_snapshot["generated_at"]

    # Ensure covariance stays PSD and bounded during sequential updates.
    for state in sequence.belief_states:
        covariance = np.array(state.posterior.covariance)
        eigenvalues = np.linalg.eigvalsh(covariance)
        assert np.all(eigenvalues >= -1e-9)
        assert np.all(np.isfinite(eigenvalues))

    # Regime signals should publish telemetry to the synthetic bus.
    regime_events = [event for event in sequence.events if event.type == "telemetry.understanding.regime"]
    assert regime_events, "Regime FSM should emit telemetry events"
    assert sequence.regime_signals[-1].regime_state.regime in {"bullish", "bearish", "balanced", "uncertain"}

    supports = [state.posterior.support for state in sequence.belief_states]
    assert supports == list(range(1, len(outcome.market_data) + 1))
