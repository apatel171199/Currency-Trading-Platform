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