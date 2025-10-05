import json
import logging

from src.operational.md_capture import MarketDataRecorder, MarketDataReplayer


class _BrokenHandle:
    def close(self) -> None:
        raise OSError("boom")


def test_market_data_recorder_logs_close_errors(tmp_path, caplog):
    capture_path = tmp_path / "capture.jsonl"
    recorder = MarketDataRecorder(str(capture_path))

    # Release the real handle to avoid leaking descriptors before swapping in the stub
    recorder._fh.close()  # type: ignore[attr-defined]
    recorder._fh = _BrokenHandle()  # type: ignore[attr-defined]

    caplog.set_level(logging.WARNING, logger="src.operational.md_capture")

    recorder.close()

    assert any(
        "Failed to close market data capture file" in record.message for record in caplog.records
    )


def test_market_data_replayer_logs_invalid_records_and_feature_writer_errors(tmp_path, caplog):
    capture_path = tmp_path / "capture.jsonl"
    valid_record = {
        "symbol": "EURUSD",
        "t": "2024-01-01T00:00:00",
        "bids": [],
        "asks": [],
    }
    invalid_iso_record = {"symbol": "EURUSD", "t": "bad", "bids": [], "asks": []}
    capture_path.write_text(
        json.dumps(valid_record)
        + "\n"
        + json.dumps(invalid_iso_record)
        + "\n"
        + "not-json\n"
    )

    replayer = MarketDataReplayer(str(capture_path))
    collected: list[tuple[str, dict[str, object]]] = []

    def callback(symbol: str, payload: dict[str, object]) -> None:
        collected.append((symbol, payload))

    def feature_writer(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("feature failure")

    caplog.set_level(logging.WARNING, logger="src.operational.md_capture")

    emitted = replayer.replay(callback, feature_writer=feature_writer)

    assert emitted == 1
    assert len(collected) == 1
    assert collected[0][0] == "EURUSD"
    assert any(
        "Feature writer failed" in record.message for record in caplog.records
    )
    assert any(
        "Skipping invalid market data record" in record.message for record in caplog.records
    )
