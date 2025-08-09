from src.operational.md_capture import MarketDataRecorder, MarketDataReplayer, _serialize_order_book


class DummyEntry:
    def __init__(self, price, size):
        self.price = price
        self.size = size


class DummyOB:
    def __init__(self):
        self.bids = [DummyEntry(1.1, 1000000)]
        self.asks = [DummyEntry(1.1002, 1200000)]


def test_serialize_and_replay(tmp_path):
    ob = DummyOB()
    rec = _serialize_order_book("EURUSD", ob)
    assert rec["symbol"] == "EURUSD"
    out = tmp_path / "cap.jsonl"
    r = MarketDataRecorder(str(out))
    try:
        # Simulate attach callback directly
        r._fh.write("\n")  # ensure file exists
        r._fh.flush()
    finally:
        r.close()

    # Write one record
    with open(out, "a", encoding="utf-8") as fh:
        import json
        fh.write(json.dumps(rec) + "\n")

    emitted = MarketDataReplayer(str(out), speed=0).replay(lambda s, o: None, max_events=1)
    assert emitted == 1

