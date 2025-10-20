import pandas as pd

from src.understanding.belief_real_data_utils import price_series_from_frame


def test_price_series_from_frame_accepts_uppercase_column() -> None:
    frame = pd.DataFrame({"Close": [1, 2, 3, 4]})

    result = price_series_from_frame(frame)

    assert result == [1.0, 2.0, 3.0, 4.0]
