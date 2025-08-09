from src.data_foundation.config.vol_config import load_vol_config


def test_brake_scale_loaded():
    cfg = load_vol_config()
    assert hasattr(cfg, 'brake_scale')
    assert isinstance(cfg.brake_scale, float)

