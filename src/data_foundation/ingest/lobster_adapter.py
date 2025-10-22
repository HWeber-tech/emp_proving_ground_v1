"""Utilities for parsing LOBSTER order book datasets.

The LOBSTER (Limit Order Book System – The Efficient Reconstructor) dataset
provides limit order book event streams as a pair of CSV files: one containing
the event *messages* and another providing synchronised depth snapshots.  The
roadmap calls out building a parser that can normalise these files so the
HOW sensory cortex can consume rich microstructure signals.  This module keeps
the implementation light-weight and dependency free whilst still handling the
quirks of the public dataset (no headers, scaled integer prices, optional
compression).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, MutableMapping

import pandas as pd

__all__ = [
    "LobsterDataset",
    "load_lobster_messages",
    "load_lobster_order_book",
    "load_lobster_dataset",
]


_MESSAGE_COLUMNS = ("timestamp", "event_type", "order_id", "size", "price", "direction")


@dataclass(slots=True)
class LobsterDataset:
    """Container holding the parsed LOBSTER event stream."""

    messages: pd.DataFrame
    order_book: pd.DataFrame

    def validate_alignment(self) -> None:
        """Ensure the two frames are synchronised on row counts."""

        if len(self.messages) != len(self.order_book):
            raise ValueError(
                "LOBSTER dataset misalignment: "
                f"{len(self.messages)} messages vs {len(self.order_book)} order book rows"
            )


def _normalise_timestamp(
    series: pd.Series,
    *,
    timestamp_unit: str,
    base_date: datetime | str | None,
) -> pd.Series:
    """Convert LOBSTER timestamps to either timedeltas or datetimes."""

    unit = timestamp_unit
    try:
        deltas = pd.to_timedelta(series.astype("float64"), unit=unit)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"invalid timestamp unit {timestamp_unit!r}") from exc

    if base_date is None:
        return deltas

    base = pd.Timestamp(base_date)
    if base.tzinfo is None:
        base = base.tz_localize("UTC")
    return base + deltas


def load_lobster_messages(
    path: str | Path,
    *,
    price_scale: float = 0.01,
    timestamp_unit: str = "s",
    base_date: datetime | str | None = None,
    dtype_overrides: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Load a LOBSTER message file into a typed :class:`~pandas.DataFrame`.

    Parameters
    ----------
    path:
        Location of the `.csv` or compressed `.csv.gz` file.
    price_scale:
        Factor used to convert integer stored prices into tradeable units.  The
        public LOBSTER files use prices that are scaled by 100 (i.e. cents).
    timestamp_unit:
        Unit describing the granularity of the timestamp column.
    base_date:
        Optional anchor date.  When supplied the timestamp column is converted
        to timezone-aware :class:`~pandas.Timestamp` values.
    dtype_overrides:
        Optional mapping for overriding inferred dtypes.
    """

    path = Path(path)
    compression = "infer" if path.suffixes else None

    dtype: MutableMapping[str, str] = {
        "timestamp": "float64",
        "event_type": "int8",
        "order_id": "int64",
        "size": "int32",
        "price": "float64",
        "direction": "int8",
    }
    if dtype_overrides:
        dtype.update(dtype_overrides)

    frame = pd.read_csv(
        path,
        header=None,
        names=_MESSAGE_COLUMNS,
        dtype=dtype,
        compression=compression,
    )

    if frame.empty:
        return frame

    frame["price"] = frame["price"].astype("float64") * float(price_scale)
    frame["timestamp"] = _normalise_timestamp(
        frame["timestamp"], timestamp_unit=timestamp_unit, base_date=base_date
    )

    return frame


def load_lobster_order_book(
    path: str | Path,
    *,
    price_scale: float = 0.01,
    levels: int | None = None,
    dtype_overrides: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Load a LOBSTER order book file.

    The order book snapshot file does not include headers; column order is
    ``ask_price_1, ask_size_1, ..., bid_price_1, bid_size_1, ...``.  The number
    of levels is either inferred from the column count or provided explicitly.
    """

    path = Path(path)
    compression = "infer" if path.suffixes else None

    frame = pd.read_csv(path, header=None, compression=compression)

    if frame.empty:
        return frame

    column_count = frame.shape[1]
    inferred_levels = column_count // 4
    if column_count % 2 != 0:
        raise ValueError(
            "LOBSTER order book file should contain an even number of columns"
        )
    if levels is None:
        levels = inferred_levels
    if levels <= 0 or levels > inferred_levels:
        raise ValueError(
            f"invalid depth level count {levels}; file supports {inferred_levels}"
        )

    names: list[str] = []
    for depth in range(1, levels + 1):
        names.append(f"ask_price_{depth}")
        names.append(f"ask_size_{depth}")
    for depth in range(1, levels + 1):
        names.append(f"bid_price_{depth}")
        names.append(f"bid_size_{depth}")

    frame = frame.iloc[:, : 4 * levels]
    frame.columns = names

    dtype: MutableMapping[str, str] = {}
    if dtype_overrides:
        dtype.update(dtype_overrides)
    if dtype:
        frame = frame.astype(dtype)

    price_columns = [
        col
        for col in frame.columns
        if col.startswith("ask_price_") or col.startswith("bid_price_")
    ]
    frame[price_columns] = frame[price_columns].astype("float64") * float(price_scale)

    return frame


def load_lobster_dataset(
    message_path: str | Path,
    order_book_path: str | Path,
    *,
    price_scale: float = 0.01,
    timestamp_unit: str = "s",
    base_date: datetime | str | None = None,
    depth_levels: int | None = None,
) -> LobsterDataset:
    """Load the paired LOBSTER files and validate alignment."""

    messages = load_lobster_messages(
        message_path,
        price_scale=price_scale,
        timestamp_unit=timestamp_unit,
        base_date=base_date,
    )
    order_book = load_lobster_order_book(
        order_book_path,
        price_scale=price_scale,
        levels=depth_levels,
    )

    dataset = LobsterDataset(messages=messages, order_book=order_book)
    if not messages.empty or not order_book.empty:
        dataset.validate_alignment()
    return dataset

