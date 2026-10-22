import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def clean_df() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=48, freq="1h", tz="UTC")
    close = 100 + np.arange(48, dtype=float)
    return pd.DataFrame({"date": dates, "open": close - 0.5, "high": close + 1, "low": close - 1,
                         "close": close, "volume": np.full(48, 10.0)})
