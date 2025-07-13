import pytest
import pandas as pd
from datetime import datetime
from emp.data_pipeline import DukascopyIngestor, TickDataCleaner, TickDataStorage, MarketRegimeIdentifier

def test_dukas_copy_ingestor_stub():
    ingestor = DukascopyIngestor('raw', 'processed')
    with pytest.raises(NotImplementedError):
        ingestor.ingest_range('EURUSD', datetime.now(), datetime.now())

def test_tick_data_cleaner_stub():
    cleaner = TickDataCleaner('EURUSD')
    with pytest.raises(NotImplementedError):
        cleaner.clean(pd.DataFrame())

def test_tick_data_storage_stub():
    storage = TickDataStorage('processed')
    with pytest.raises(NotImplementedError):
        storage.load_tick_data('EURUSD', datetime.now(), datetime.now())
    with pytest.raises(NotImplementedError):
        storage.save_tick_data('EURUSD', 2020, 1, pd.DataFrame())

def test_market_regime_identifier_stub():
    identifier = MarketRegimeIdentifier('processed', 'EURUSD')
    with pytest.raises(NotImplementedError):
        identifier.identify_regimes(2020, 2021) 