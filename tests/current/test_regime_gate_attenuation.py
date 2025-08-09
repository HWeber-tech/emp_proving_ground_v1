from src.data_foundation.config.vol_config import load_vol_config


def test_vol_config_attenuation_defaults():
    cfg = load_vol_config()  # defaults if file missing are ok
    # Attributes exist
    assert hasattr(cfg, 'gate_mode')
    assert hasattr(cfg, 'attenuation_factor')
    # Values are sane types
    assert isinstance(cfg.gate_mode, str)
    assert isinstance(cfg.attenuation_factor, float)

