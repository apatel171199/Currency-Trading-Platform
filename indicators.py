from __future__ import annotations

import numpy as np
import pandas as pd


class IndicatorCalculator: #Calculates technical indicators from market price data.

    REQUIRED_COLUMNS = {"High", "Low", "Close"}

    def add_ema(
        self,
        data: pd.DataFrame,
        period: int,
        column_name: str | None = None,
    ) -> pd.DataFrame:
        self._validate_data(data)
        self._validate_period(period)

        result = data.copy()
        ema_name = column_name or f"EMA_{period}"

        result[ema_name] = result["Close"].ewm(
            span=period,
            adjust=False,).mean()

        return result

    def add_rsi(
        self,
        data: pd.DataFrame,
        period: int = 14,
        column_name: str | None = None,
    ) -> pd.DataFrame:
        self._validate_data(data)
        self._validate_period(period)

        result = data.copy()
        rsi_name = column_name or f"RSI_{period}"

        price_change = result["Close"].diff()

        gains = price_change.clip(lower=0)
        losses = -price_change.clip(upper=0)

        average_gain = gains.ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period,).mean()

        average_loss = losses.ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period,).mean()

        relative_strength = average_gain / average_loss.replace(0, np.nan)

        result[rsi_name] = 100 - (100 / (1 + relative_strength))

        return result

    def add_atr(
        self,
        data: pd.DataFrame,
        period: int = 14,
        column_name: str | None = None,
    ) -> pd.DataFrame:
        self._validate_data(data)
        self._validate_period(period)

        result = data.copy()
        atr_name = column_name or f"ATR_{period}"

        previous_close = result["Close"].shift(1)

        high_low = result["High"] - result["Low"]
        high_previous_close = (
            result["High"] - previous_close ).abs()
        low_previous_close = (
            result["Low"] - previous_close ).abs()

        true_range = pd.concat(
            [
                high_low,
                high_previous_close,
                low_previous_close,
            ],
            axis=1, ).max(axis=1)

        result[atr_name] = true_range.ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period, ).mean()

        return result

    def add_bollinger_bands(
        self,
        data: pd.DataFrame,
        period: int = 20,
        standard_deviations: float = 2.0,
    ) -> pd.DataFrame:
        self._validate_data(data)
        self._validate_period(period)

        if standard_deviations <= 0:
            raise ValueError("Standard deviations must be greater than zero.")

        result = data.copy()

        middle_band = result["Close"].rolling(window=period,).mean()

        rolling_std = result["Close"].rolling(window=period,).std()

        result[f"BB_MIDDLE_{period}"] = middle_band
        result[f"BB_UPPER_{period}"] = (middle_band + standard_deviations * rolling_std)
        result[f"BB_LOWER_{period}"] = (middle_band - standard_deviations * rolling_std)

        return result

    def add_macd(
        self,
        data: pd.DataFrame,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> pd.DataFrame:
        self._validate_data(data)

        for period in (fast_period, slow_period, signal_period):
            self._validate_period(period)

        if fast_period >= slow_period:
            raise ValueError("MACD fast period must be shorter than the slow period.")

        result = data.copy()
        suffix = f"{fast_period}_{slow_period}_{signal_period}"

        fast_ema = result["Close"].ewm(span=fast_period, adjust=False).mean()
        slow_ema = result["Close"].ewm(span=slow_period, adjust=False).mean()

        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

        # The first slow_period candles do not have enough history yet.
        macd_line.iloc[: slow_period - 1] = np.nan
        signal_line.iloc[: slow_period + signal_period - 2] = np.nan

        result[f"MACD_{suffix}"] = macd_line
        result[f"MACD_SIGNAL_{suffix}"] = signal_line
        result[f"MACD_HIST_{suffix}"] = macd_line - signal_line

        return result

    def add_donchian_channel(
        self,
        data: pd.DataFrame,
        period: int = 20,
    ) -> pd.DataFrame:
        """Highest high / lowest low of the PREVIOUS `period` candles.

        The current candle is excluded so that "close above the channel"
        is a real breakout and not compared against itself.
        """
        self._validate_data(data)
        self._validate_period(period)

        result = data.copy()

        result[f"DC_UPPER_{period}"] = (
            result["High"].rolling(window=period).max().shift(1))
        result[f"DC_LOWER_{period}"] = (
            result["Low"].rolling(window=period).min().shift(1))

        return result

    def _validate_data(self, data: pd.DataFrame) -> None:
        if data.empty:
            raise ValueError("Cannot calculate indicators on an empty DataFrame.")

        missing_columns = self.REQUIRED_COLUMNS.difference(
            data.columns)

        if missing_columns:
            missing = ", ".join(sorted(missing_columns))

            raise ValueError(f"Market data is missing required columns: {missing}")

    def _validate_period(self, period: int) -> None:
        if not isinstance(period, int):
            raise TypeError("Indicator period must be an integer.")

        if period <= 0:
            raise ValueError("Indicator period must be greater than zero.")