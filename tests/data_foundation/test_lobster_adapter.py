from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from src.data_foundation.ingest.lobster_adapter import (
    LobsterDataset,
    load_lobster_dataset,
    load_lobster_messages,
    load_lobster_order_book,
)


def _write_csv(path: Path, rows: list[list[object]]) -> None:
    data = "\n".join(
        ",".join(str(value) for value in row)
        for row in rows
    )
    path.write_text(data)


def test_load_lobster_messages_scaling_and_timestamp(tmp_path: Path) -> None:
    message_path = tmp_path / "messages.csv"
    _write_csv(
        message_path,
        [
            [0.0, 1, 101, 10, 250000, 1],
            [0.25, 2, 102, 5, 250500, -1],
        ],
    )

    frame = load_lobster_messages(
        message_path,
        price_scale=0.0001,
        timestamp_unit="s",
        base_date=datetime(2024, 1, 2),
    )

    assert list(frame.columns) == [
        "timestamp",
        "event_type",
        "order_id",
        "size",
        "price",
        "direction",
    ]
    assert float(frame.loc[0, "price"]) == 25.0
    assert float(frame.loc[1, "price"]) == 25.05
    assert frame["timestamp"].iloc[0] == pd.Timestamp("2024-01-02T00:00:00+0000", tz="UTC")
    assert frame["timestamp"].iloc[1] == pd.Timestamp("2024-01-02T00:00:00.250000+0000", tz="UTC")


def test_load_lobster_order_book_infers_levels(tmp_path: Path) -> None:
    order_book_path = tmp_path / "orderbook.csv"
    _write_csv(
        order_book_path,
        [
            [250000, 3, 250100, 2, 249900, 4, 249800, 5],
            [250500, 2, 250600, 1, 249500, 3, 249400, 2],
        ],
    )

    frame = load_lobster_order_book(order_book_path, price_scale=0.0001)

    assert list(frame.columns) == [
        "ask_price_1",
        "ask_size_1",
        "ask_price_2",
        "ask_size_2",
        "bid_price_1",
        "bid_size_1",
        "bid_price_2",
        "bid_size_2",
    ]
    assert float(frame.loc[0, "ask_price_1"]) == 25.0
    assert float(frame.loc[1, "bid_price_2"]) == 24.94


def test_load_lobster_dataset_validates_alignment(tmp_path: Path) -> None:
    messages = tmp_path / "messages.csv"
    order_book = tmp_path / "orderbook.csv"

    _write_csv(messages, [[0.0, 1, 101, 10, 250000, 1]])
    _write_csv(order_book, [[250000, 1, 249900, 1, 249800, 1, 249700, 1]])

    dataset = load_lobster_dataset(
        messages,
        order_book,
        price_scale=0.0001,
    )

    assert isinstance(dataset, LobsterDataset)
    assert len(dataset.messages) == len(dataset.order_book) == 1

    _write_csv(order_book, [
        [250000, 1, 249900, 1, 249800, 1, 249700, 1],
        [250100, 1, 249800, 1, 249700, 1, 249600, 1],
    ])

    try:
        load_lobster_dataset(messages, order_book, price_scale=0.0001)
    except ValueError as exc:
        assert "misalignment" in str(exc)
    else:  # pragma: no cover - defensive guard
        raise AssertionError("Expected misalignment validation to raise")

