from src.sensory.dimensions.microstructure import (
    compute_features,
    compute_liquidity_pockets,
    compute_volatility_seeds,
)


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


def test_liquidity_pockets_detects_large_sizes():
    bids = [(1.1000, 1000000), (1.0999, 3000000), (1.0998, 900000)]
    asks = [(1.1002, 1200000), (1.1003, 1100000), (1.1004, 5000000)]
    p = compute_liquidity_pockets(bids, asks, levels=3, pocket_factor=1.5)
    assert p["bid_pocket_count_l3"] >= 1
    assert p["ask_pocket_count_l3"] >= 1


def test_volatility_seeds_computation():
    mids = [1.1000, 1.1001, 1.1003, 1.1002, 1.1004, 1.1001]
    v = compute_volatility_seeds(mids)
    assert set(["std5", "std10", "std20", "mean_abs_change"]).issubset(v.keys())

