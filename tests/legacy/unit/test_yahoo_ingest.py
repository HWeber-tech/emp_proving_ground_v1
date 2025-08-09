#!/usr/bin/env python3

import pandas as pd
from src.data_foundation.ingest.yahoo_ingest import fetch_daily_bars


def test_fetch_daily_bars_handles_empty_symbols():
    df = fetch_daily_bars([])
    assert isinstance(df, pd.DataFrame)
    assert df.empty


