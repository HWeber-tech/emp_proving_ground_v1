from src.core.sensory_organ import create_sensory_organ


def test_create_sensory_organ_accepts_sequence_drift_config() -> None:
    organ = create_sensory_organ(
        drift_config=[
            ("baseline_window", 10),
            ("evaluation_window", 5),
        ]
    )

    config = organ._drift_config

    assert config.baseline_window == 10
    assert config.evaluation_window == 5
