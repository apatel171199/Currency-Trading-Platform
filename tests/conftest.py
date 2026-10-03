import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def make_candles(n: int = 3000, seed: int = 0, drift_period: float = 0.0, start: float = 1.1,
                 freq: str = "h") -> pd.DataFrame:
    """Synthetic candles. Each open equals the previous close (no gaps)."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.0012, n)
    if drift_period:
        returns += 0.0001 * np.sin(np.arange(n) / drift_period)

    close = start * np.exp(np.cumsum(returns))
    open_ = np.r_[start, close[:-1]]
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.0005, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.0005, n)))

    index = pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC", name="Datetime")
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close}, index=index)


def make_pattern_candles(n: int = 20000, seed: int = 7, strength: float = 0.5) -> pd.DataFrame:
    """Random walk with a hidden rule a model can learn: from 07:00 to 11:00
    UTC price drifts in the direction of the previous 6 hours."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC", name="Datetime")
    returns = rng.normal(0, 0.0012, n)
    hours = index.hour.to_numpy()

    for i in range(6, n):
        if 7 <= hours[i] <= 11:
            returns[i] += strength * np.sign(returns[i - 6:i].sum()) * 0.0012

    close = 1.1 * np.exp(np.cumsum(returns))
    open_ = np.r_[1.1, close[:-1]]
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.0005, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.0005, n)))
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close}, index=index)


@pytest.fixture
def candles() -> pd.DataFrame:
    return make_candles()
