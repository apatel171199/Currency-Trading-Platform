import numpy as np

from indicators import IndicatorCalculator


def test_donchian_excludes_current_candle(candles):
    result = IndicatorCalculator().add_donchian_channel(candles, period=5)
    expected = candles["High"].iloc[10:15].max()
    assert result["DC_UPPER_5"].iloc[15] == expected


def test_macd_columns_and_warmup(candles):
    result = IndicatorCalculator().add_macd(candles, 12, 26, 9)
    assert np.isnan(result["MACD_12_26_9"].iloc[24])
    assert not np.isnan(result["MACD_12_26_9"].iloc[25])
    hist = result["MACD_HIST_12_26_9"].dropna()
    line = result["MACD_12_26_9"] - result["MACD_SIGNAL_12_26_9"]
    assert np.allclose(hist, line.dropna())
