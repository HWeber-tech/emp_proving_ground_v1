from src.core.performance.market_data_cache import get_global_cache


def test_market_data_cache_smoke():
    cache = get_global_cache()

    # Put a snapshot
    cache.put_snapshot("EURUSD", bid=1.1000, ask=1.2000, ts=1234567.0)

    # Get snapshot fields
    snap = cache.get_snapshot("EURUSD")
    assert snap is not None
    assert snap["symbol"] == "EURUSD"
    assert snap["bid"] == 1.1000
    assert snap["ask"] == 1.2000
    assert snap["ts"] == 1234567.0

    # Mid price
    mid = cache.maybe_get_mid("EURUSD")
    assert mid == (1.1000 + 1.2000) / 2.0