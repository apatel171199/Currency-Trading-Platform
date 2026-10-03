"""Trading strategies.

Every strategy follows the same two-step contract:

1. ``prepare(data)`` adds the indicator columns the strategy needs.
2. ``signals(data)`` returns one signal per candle:
   +1 = BUY, -1 = SELL, 0 = WAIT.

A signal on candle ``i`` may only use information available when candle
``i`` closed. The backtester and the live trader both act on the NEXT
candle, so there is no look-ahead bias.

Signals are calculated for the whole DataFrame at once (vectorised),
which makes it fast enough to backtest hundreds of parameter
combinations.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from decision import Decision
from indicators import IndicatorCalculator

BUY = 1
SELL = -1
WAIT = 0


@dataclass(frozen=True)
class StrategyAnalysis:  # The strategy's decision and the evidence behind it

    decision: Decision
    score: int
    reasons: tuple[str, ...]


class Strategy(ABC):  # Base class shared by every strategy

    name: str = "Strategy"

    # Each strategy lists the values the optimizer should try.
    parameter_grid: dict[str, list[Any]] = {}

    # Every strategy needs ATR because stops and targets are ATR based.
    atr_period: int = 14

    def __init__(self, **parameters: Any) -> None:
        self.parameters = parameters
        self.calculator = IndicatorCalculator()
        self._validate_parameters()

    @property
    def label(self) -> str:
        if not self.parameters:
            return self.name

        settings = ", ".join(
            f"{key}={value}" for key, value in self.parameters.items())
        return f"{self.name}({settings})"

    @property
    def warmup(self) -> int:
        """Number of candles needed before the indicators are reliable."""
        numbers = [
            value for value in self.parameters.values()
            if isinstance(value, int) and not isinstance(value, bool)]
        return max([self.atr_period, *numbers]) * 3

    def prepare(self, data: pd.DataFrame) -> pd.DataFrame:
        result = self.calculator.add_atr(
            data, period=self.atr_period, column_name="ATR")
        return self._add_indicators(result)

    def signals(self, data: pd.DataFrame) -> pd.Series:
        raw = self._raw_signals(data)

        signal = pd.Series(raw, index=data.index).fillna(WAIT).astype(int)

        # Never trade before the indicators have warmed up.
        signal.iloc[: self.warmup] = WAIT

        return signal

    def analyze(self, data: pd.DataFrame) -> StrategyAnalysis:
        """Decision for the latest candle (used by the live trader)."""
        if data.empty:
            raise ValueError("Cannot analyze an empty DataFrame.")

        latest_signal = int(self.signals(data).iloc[-1])
        decision = {
            BUY: Decision.BUY,
            SELL: Decision.SELL,
        }.get(latest_signal, Decision.WAIT)

        return StrategyAnalysis(
            decision=decision,
            score=latest_signal,
            reasons=(f"{self.label} signalled {decision.name}.",),
        )

    @classmethod
    def parameter_combinations(cls) -> list[dict[str, Any]]:
        if not cls.parameter_grid:
            return [{}]

        keys = list(cls.parameter_grid)
        combinations = []

        for values in product(*(cls.parameter_grid[key] for key in keys)):
            parameters = dict(zip(keys, values))

            try:
                cls(**parameters)
            except ValueError:
                # Skip impossible combinations, e.g. fast EMA >= slow EMA.
                continue

            combinations.append(parameters)

        return combinations

    @abstractmethod
    def _add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        ...

    @abstractmethod
    def _raw_signals(self, data: pd.DataFrame) -> pd.Series | np.ndarray:
        ...

    def _validate_parameters(self) -> None:
        pass


def _crossed_above(fast: pd.Series, slow: pd.Series) -> pd.Series:
    return (fast > slow) & (fast.shift(1) <= slow.shift(1))


def _crossed_below(fast: pd.Series, slow: pd.Series) -> pd.Series:
    return (fast < slow) & (fast.shift(1) >= slow.shift(1))


class EmaCrossStrategy(Strategy):  # Trend following: fast EMA crosses slow EMA

    name = "EMA Cross"
    parameter_grid = {
        "fast": [5, 10, 20],
        "slow": [30, 50, 100],
    }

    def __init__(self, fast: int = 10, slow: int = 30) -> None:
        super().__init__(fast=fast, slow=slow)

    def _validate_parameters(self) -> None:
        if self.parameters["fast"] >= self.parameters["slow"]:
            raise ValueError("Fast EMA must be shorter than slow EMA.")

    def _add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        result = self.calculator.add_ema(
            data, self.parameters["fast"], column_name="EMA_FAST")
        return self.calculator.add_ema(
            result, self.parameters["slow"], column_name="EMA_SLOW")

    def _raw_signals(self, data: pd.DataFrame) -> np.ndarray:
        fast = data["EMA_FAST"]
        slow = data["EMA_SLOW"]

        return np.select(
            [_crossed_above(fast, slow), _crossed_below(fast, slow)],
            [BUY, SELL],
            default=WAIT)


class RsiReversionStrategy(Strategy):  # Mean reversion: buy oversold, sell overbought

    name = "RSI Reversion"
    parameter_grid = {
        "period": [7, 14],
        "oversold": [20, 30],
        "overbought": [70, 80],
        "trend_filter": [0, 200],
    }

    def __init__(
        self,
        period: int = 14,
        oversold: int = 30,
        overbought: int = 70,
        trend_filter: int = 0,
    ) -> None:
        super().__init__(
            period=period,
            oversold=oversold,
            overbought=overbought,
            trend_filter=trend_filter,
        )

    def _validate_parameters(self) -> None:
        if not 0 < self.parameters["oversold"] < self.parameters["overbought"] < 100:
            raise ValueError("RSI levels must satisfy 0 < oversold < overbought < 100.")

    def _add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        result = self.calculator.add_rsi(
            data, self.parameters["period"], column_name="RSI")

        if self.parameters["trend_filter"]:
            result = self.calculator.add_ema(
                result, self.parameters["trend_filter"], column_name="EMA_TREND")

        return result

    def _raw_signals(self, data: pd.DataFrame) -> np.ndarray:
        rsi = data["RSI"]
        oversold = pd.Series(self.parameters["oversold"], index=data.index)
        overbought = pd.Series(self.parameters["overbought"], index=data.index)

        # Signal when RSI comes BACK out of the extreme zone.
        buy = _crossed_above(rsi, oversold)
        sell = _crossed_below(rsi, overbought)

        if self.parameters["trend_filter"]:
            buy &= data["Close"] > data["EMA_TREND"]
            sell &= data["Close"] < data["EMA_TREND"]

        return np.select([buy, sell], [BUY, SELL], default=WAIT)


class BollingerReversionStrategy(Strategy):  # Fade moves outside the Bollinger Bands

    name = "Bollinger Reversion"
    parameter_grid = {
        "period": [20, 50],
        "deviations": [2.0, 2.5],
    }

    def __init__(self, period: int = 20, deviations: float = 2.0) -> None:
        super().__init__(period=period, deviations=deviations)

    def _validate_parameters(self) -> None:
        if self.parameters["deviations"] <= 0:
            raise ValueError("Standard deviations must be positive.")

    def _add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        period = self.parameters["period"]
        result = self.calculator.add_bollinger_bands(
            data, period=period, standard_deviations=self.parameters["deviations"])

        return result.rename(columns={
            f"BB_LOWER_{period}": "BB_LOWER",
            f"BB_MIDDLE_{period}": "BB_MIDDLE",
            f"BB_UPPER_{period}": "BB_UPPER",
        })

    def _raw_signals(self, data: pd.DataFrame) -> np.ndarray:
        close = data["Close"]

        # Price closes back inside the band after being outside it.
        buy = _crossed_above(close, data["BB_LOWER"])
        sell = _crossed_below(close, data["BB_UPPER"])

        return np.select([buy, sell], [BUY, SELL], default=WAIT)


class DonchianBreakoutStrategy(Strategy):  # Breakout: close above/below the N-candle range

    name = "Donchian Breakout"
    parameter_grid = {
        "period": [20, 55, 100],
    }

    def __init__(self, period: int = 20) -> None:
        super().__init__(period=period)

    def _add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        period = self.parameters["period"]
        result = self.calculator.add_donchian_channel(data, period=period)

        return result.rename(columns={
            f"DC_UPPER_{period}": "DC_UPPER",
            f"DC_LOWER_{period}": "DC_LOWER",
        })

    def _raw_signals(self, data: pd.DataFrame) -> np.ndarray:
        close = data["Close"]
        previous_close = close.shift(1)

        # Only the first close outside the channel, not every candle after.
        buy = (close > data["DC_UPPER"]) & (previous_close <= data["DC_UPPER"].shift(1))
        sell = (close < data["DC_LOWER"]) & (previous_close >= data["DC_LOWER"].shift(1))

        return np.select([buy, sell], [BUY, SELL], default=WAIT)


class MacdCrossStrategy(Strategy):  # Momentum: MACD line crosses its signal line

    name = "MACD Cross"
    parameter_grid = {
        "fast": [8, 12],
        "slow": [21, 26],
        "signal": [5, 9],
        "trend_filter": [0, 200],
    }

    def __init__(
        self,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        trend_filter: int = 0,
    ) -> None:
        super().__init__(
            fast=fast, slow=slow, signal=signal, trend_filter=trend_filter)

    def _validate_parameters(self) -> None:
        if self.parameters["fast"] >= self.parameters["slow"]:
            raise ValueError("MACD fast period must be shorter than slow period.")

    def _add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        fast = self.parameters["fast"]
        slow = self.parameters["slow"]
        signal = self.parameters["signal"]
        suffix = f"{fast}_{slow}_{signal}"

        result = self.calculator.add_macd(data, fast, slow, signal)
        result = result.rename(columns={
            f"MACD_{suffix}": "MACD",
            f"MACD_SIGNAL_{suffix}": "MACD_SIGNAL",
            f"MACD_HIST_{suffix}": "MACD_HIST",
        })

        if self.parameters["trend_filter"]:
            result = self.calculator.add_ema(
                result, self.parameters["trend_filter"], column_name="EMA_TREND")

        return result

    def _raw_signals(self, data: pd.DataFrame) -> np.ndarray:
        buy = _crossed_above(data["MACD"], data["MACD_SIGNAL"])
        sell = _crossed_below(data["MACD"], data["MACD_SIGNAL"])

        if self.parameters["trend_filter"]:
            buy &= data["Close"] > data["EMA_TREND"]
            sell &= data["Close"] < data["EMA_TREND"]

        return np.select([buy, sell], [BUY, SELL], default=WAIT)


class IndicatorStrategy(Strategy):  # Scores EMA trend, RSI and Bollinger Bands together

    name = "Indicator Score"
    parameter_grid = {
        "threshold": [3, 4],
    }

    def __init__(self, threshold: int = 3) -> None:
        super().__init__(threshold=threshold)

    @property
    def warmup(self) -> int:
        return 90

    def _validate_parameters(self) -> None:
        if not 1 <= self.parameters["threshold"] <= 4:
            raise ValueError("Score threshold must be between 1 and 4.")

    def _add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        result = self.calculator.add_ema(data, period=10)
        result = self.calculator.add_ema(result, period=30)
        result = self.calculator.add_rsi(result, period=14)
        return self.calculator.add_bollinger_bands(
            result, period=20, standard_deviations=2.0)

    def _scores(self, data: pd.DataFrame) -> pd.Series:
        ema_fast = data["EMA_10"]
        ema_slow = data["EMA_30"]
        rsi = data["RSI_14"]
        close = data["Close"]
        lower = data["BB_LOWER_20"]
        middle = data["BB_MIDDLE_20"]
        upper = data["BB_UPPER_20"]

        trend = np.select(
            [ema_fast > ema_slow, ema_fast < ema_slow], [2, -2], default=0)

        momentum = np.select(
            [
                (rsi >= 55) & (rsi < 70),
                (rsi > 30) & (rsi <= 45),
                rsi >= 70,
                rsi <= 30,
            ],
            [1, -1, -1, 1],
            default=0)

        location = np.select(
            [close > upper, close < lower, close > middle, close < middle],
            [-1, 1, 1, -1],
            default=0)

        return pd.Series(trend + momentum + location, index=data.index)

    def _raw_signals(self, data: pd.DataFrame) -> np.ndarray:
        scores = self._scores(data)
        threshold = self.parameters["threshold"]

        return np.select(
            [scores >= threshold, scores <= -threshold],
            [BUY, SELL],
            default=WAIT)

    def analyze(self, data: pd.DataFrame) -> StrategyAnalysis:
        """Explains the latest score in plain English."""
        if data.empty:
            raise ValueError("Cannot analyze an empty DataFrame.")

        latest = data.iloc[-1]
        needed = ["EMA_10", "EMA_30", "RSI_14", "BB_LOWER_20", "BB_MIDDLE_20", "BB_UPPER_20"]

        if latest[needed].isna().any():
            raise ValueError("The latest candle contains incomplete indicator values.")

        reasons: list[str] = []

        if latest["EMA_10"] > latest["EMA_30"]:
            reasons.append("EMA trend is bullish: EMA_10 is above EMA_30.")
        elif latest["EMA_10"] < latest["EMA_30"]:
            reasons.append("EMA trend is bearish: EMA_10 is below EMA_30.")
        else:
            reasons.append("EMA trend is neutral.")

        rsi = float(latest["RSI_14"])
        if 55 <= rsi < 70:
            reasons.append("RSI shows bullish momentum without being overbought.")
        elif 30 < rsi <= 45:
            reasons.append("RSI shows bearish momentum without being oversold.")
        elif rsi >= 70:
            reasons.append("RSI is overbought, increasing pullback risk.")
        elif rsi <= 30:
            reasons.append("RSI is oversold, increasing rebound potential.")
        else:
            reasons.append("RSI is neutral.")

        close = float(latest["Close"])
        if close > latest["BB_UPPER_20"]:
            reasons.append("Price is above the upper Bollinger Band.")
        elif close < latest["BB_LOWER_20"]:
            reasons.append("Price is below the lower Bollinger Band.")
        elif close > latest["BB_MIDDLE_20"]:
            reasons.append("Price is above the Bollinger middle band.")
        elif close < latest["BB_MIDDLE_20"]:
            reasons.append("Price is below the Bollinger middle band.")
        else:
            reasons.append("Price is near the Bollinger middle band.")

        score = int(self._scores(data.iloc[-1:]).iloc[-1])
        threshold = self.parameters["threshold"]

        if score >= threshold:
            decision = Decision.BUY
        elif score <= -threshold:
            decision = Decision.SELL
        else:
            decision = Decision.WAIT

        return StrategyAnalysis(decision=decision, score=score, reasons=tuple(reasons))


STRATEGY_CLASSES: dict[str, type[Strategy]] = {
    cls.__name__: cls
    for cls in (
        EmaCrossStrategy,
        RsiReversionStrategy,
        BollingerReversionStrategy,
        DonchianBreakoutStrategy,
        MacdCrossStrategy,
        IndicatorStrategy,
    )
}


def build_strategy(class_name: str, parameters: dict[str, Any]) -> Strategy:
    """Re-creates a strategy from the name/parameters saved by the optimizer."""
    try:
        cls = STRATEGY_CLASSES[class_name]
    except KeyError as error:
        known = ", ".join(STRATEGY_CLASSES)
        raise ValueError(f"Unknown strategy '{class_name}'. Known: {known}") from error

    return cls(**parameters)
