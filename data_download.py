"""Downloads years of free historical forex prices from Dukascopy.

Dukascopy (a Swiss bank) publishes hourly candles back to 2003. Each month
is one small compressed file, so 10 years of one pair is ~120 downloads.
Downloaded months are cached in data/cache/, so running it again only
fetches what is new.

    python app.py download --symbols EURUSD GBPUSD USDJPY --years 10

The result is saved as data/EURUSD_H1.csv etc., which the rest of the
program reads with ``--source csv``.
"""
from __future__ import annotations

import lzma
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BASE_URL = "https://datafeed.dukascopy.com/datafeed"
DATA_DIR = Path("data")

# Each candle record: seconds since the start of the file's period, then
# open, close, low, high as integers in "points", then volume. Big-endian.
CANDLE_RECORD = np.dtype([
    ("seconds", ">i4"),
    ("open", ">i4"),
    ("close", ">i4"),
    ("low", ">i4"),
    ("high", ">i4"),
    ("volume", ">f4"),
])


def price_point(symbol: str) -> float:
    """Dukascopy stores JPY pairs with 3 decimals and other pairs with 5."""
    return 0.001 if "JPY" in symbol.upper() else 0.00001


def hourly_url(symbol: str, year: int, month: int) -> str:
    # Dukascopy months are zero-based: January is "00".
    return f"{BASE_URL}/{symbol.upper()}/{year}/{month - 1:02d}/BID_candles_hour_1.bi5"


def decode_candles(raw: bytes, period_start: datetime, symbol: str) -> pd.DataFrame:
    """Turns one downloaded .bi5 file into a DataFrame of candles."""
    columns = ["Open", "High", "Low", "Close", "Volume"]

    if not raw:
        return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], tz="UTC", name="Datetime"))

    records = np.frombuffer(lzma.decompress(raw), dtype=CANDLE_RECORD)
    point = price_point(symbol)

    index = pd.DatetimeIndex(
        pd.Timestamp(period_start) + pd.to_timedelta(records["seconds"].astype(np.int64), unit="s"),
        name="Datetime",
    )
    data = pd.DataFrame(
        {
            "Open": records["open"] * point,
            "High": records["high"] * point,
            "Low": records["low"] * point,
            "Close": records["close"] * point,
            "Volume": records["volume"].astype(float),
        },
        index=index,
    )

    # Dukascopy fills weekends/holidays with flat zero-volume candles.
    return data[data["Volume"] > 0][columns]


def _fetch(url: str, attempts: int = 4) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return b""  # no data for that period
            if attempt == attempts - 1:
                raise RuntimeError(f"Download failed ({error.code}): {url}") from error
        except urllib.error.URLError as error:
            if attempt == attempts - 1:
                raise RuntimeError(f"Could not reach Dukascopy: {error.reason}") from error
        time.sleep(2 ** attempt)

    return b""


def download_hourly(
    symbol: str,
    start: date,
    end: date,
    cache_dir: Path = DATA_DIR / "cache",
    fetch=_fetch,
    progress: bool = True,
) -> pd.DataFrame:
    """Hourly candles for [start, end]. Completed months are cached on disk."""
    symbol = symbol.upper().replace("/", "")
    today = datetime.now(timezone.utc).date()
    frames = []

    months = pd.period_range(start=start, end=end, freq="M")

    for number, period in enumerate(months, start=1):
        cache_file = Path(cache_dir) / symbol / f"{period.year}-{period.month:02d}.bi5"
        finished = (period.year, period.month) < (today.year, today.month)

        if finished and cache_file.exists():
            raw = cache_file.read_bytes()
        else:
            raw = fetch(hourly_url(symbol, period.year, period.month))
            if finished:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_bytes(raw)
            if fetch is _fetch:
                time.sleep(0.2)  # be polite to the server

        period_start = datetime(period.year, period.month, 1, tzinfo=timezone.utc)
        frames.append(decode_candles(raw, period_start, symbol))

        if progress and (number % 12 == 0 or number == len(months)):
            print(f"  {symbol}: {number}/{len(months)} months")

    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise RuntimeError(f"Dukascopy returned no data for {symbol}. Check the symbol name.")

    data = pd.concat(frames).sort_index()
    data = data[~data.index.duplicated(keep="last")]

    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    return data[(data.index >= start_ts) & (data.index < end_ts)]


def csv_path(symbol: str, timeframe: str = "H1") -> Path:
    return DATA_DIR / f"{symbol.upper()}_{timeframe.upper()}.csv"


def download_to_csv(symbol: str, years: float, progress: bool = True) -> Path:
    end = datetime.now(timezone.utc).date()
    start = (pd.Timestamp(end) - pd.DateOffset(days=int(round(years * 365.25)))).date()

    if progress:
        print(f"{symbol}: downloading hourly candles {start} -> {end} from Dukascopy")

    data = download_hourly(symbol, start, end, progress=progress)
    path = csv_path(symbol, "H1")
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index_label="Datetime")

    if progress:
        print(f"{symbol}: saved {len(data):,} candles to {path}")
    return path
