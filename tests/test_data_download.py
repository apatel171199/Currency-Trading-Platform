import lzma
from datetime import date, datetime, timezone

import numpy as np
import pytest

from data_download import CANDLE_RECORD, decode_candles, download_hourly, hourly_url


def bi5(rows):
    """Builds a file in Dukascopy's format: LZMA-compressed big-endian records."""
    records = np.array(rows, dtype=CANDLE_RECORD)
    return lzma.compress(records.tobytes(), format=lzma.FORMAT_ALONE)


def test_url_uses_zero_based_months():
    assert hourly_url("eurusd", 2020, 1).endswith("/EURUSD/2020/00/BID_candles_hour_1.bi5")
    assert "/2020/11/" in hourly_url("EURUSD", 2020, 12)


def test_decode_prices_times_and_weekend_filter():
    # seconds, open, close, low, high, volume  (Dukascopy's order)
    raw = bi5([(0, 110000, 110050, 109900, 110100, 12.5),
               (3600, 110050, 110020, 110000, 110080, 0.0),   # zero volume -> dropped
               (7200, 110020, 110010, 109990, 110030, 3.0)])
    start = datetime(2020, 3, 1, tzinfo=timezone.utc)
    data = decode_candles(raw, start, "EURUSD")

    assert list(data.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(data) == 2
    first = data.iloc[0]
    assert (first.Open, first.High, first.Low, first.Close) == pytest.approx((1.1, 1.101, 1.099, 1.1005))
    assert str(data.index[1]) == "2020-03-01 02:00:00+00:00"


def test_jpy_pairs_use_three_decimals():
    data = decode_candles(bi5([(0, 130123, 130200, 130000, 130300, 1.0)]),
                          datetime(2020, 1, 1, tzinfo=timezone.utc), "USDJPY")
    assert data.iloc[0].Open == pytest.approx(130.123)


def test_empty_file_means_no_data():
    assert decode_candles(b"", datetime(2020, 1, 1, tzinfo=timezone.utc), "EURUSD").empty


def test_download_caches_finished_months(tmp_path):
    calls = []

    def fake_fetch(url):
        calls.append(url)
        return bi5([(0, 110000, 110000, 110000, 110000, 1.0)])

    first = download_hourly("EURUSD", date(2020, 1, 1), date(2020, 3, 31),
                            cache_dir=tmp_path, fetch=fake_fetch, progress=False)
    assert len(first) == 3 and len(calls) == 3

    second = download_hourly("EURUSD", date(2020, 1, 1), date(2020, 3, 31),
                             cache_dir=tmp_path, fetch=fake_fetch, progress=False)
    assert len(calls) == 3  # served from the cache
    assert second.equals(first)
