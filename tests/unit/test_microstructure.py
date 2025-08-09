from src.sensory.dimensions.microstructure import compute_features


def test_compute_features_spread_mid_imbalance():
    bids = [(1.1000, 1000000), (1.0999, 900000), (1.0998, 800000)]
    asks = [(1.1002, 1200000), (1.1003, 1100000), (1.1004, 1000000)]
    f = compute_features(bids, asks, levels=2)
    assert f["spread"] == 0.0002
    assert round(f["mid"], 5) == 1.1001
    assert "microprice" in f
    assert f["bid_depth_l2"] == 1900000
    assert f["ask_depth_l2"] == 2300000
    assert -1.0 <= f["top_imbalance"] <= 1.0
    assert -1.0 <= f["depth_imbalance_l2"] <= 1.0

