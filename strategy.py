from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from decision import Decision

@dataclass(frozen=True)
class StrategyAnalysis: #The strategy's decision and the evidence behind it

    decision: Decision
    score: int
    reasons: tuple[str, ...]


class IndicatorStrategy: #Scores the latest indicators and produces a trading decision

    REQUIRED_COLUMNS = {
        "Close",
        "EMA_10",
        "EMA_30",
        "RSI_14",
        "ATR_14",
        "BB_LOWER_20",
        "BB_MIDDLE_20",
        "BB_UPPER_20",}

    def analyze(self, data: pd.DataFrame) -> StrategyAnalysis:
        self._validate_data(data)

        latest = data.iloc[-1]

        score = 0
        reasons: list[str] = []#Trend: fast EMA compared with slow EMA
        
        if latest["EMA_10"] > latest["EMA_30"]:
            score += 2
            reasons.append("EMA trend is bullish: EMA_10 is above EMA_30.")
        elif latest["EMA_10"] < latest["EMA_30"]:
            score -= 2
            reasons.append("EMA trend is bearish: EMA_10 is below EMA_30.")
        else:
            reasons.append("EMA trend is neutral.")

        # Momentum: RSI
        rsi = float(latest["RSI_14"])

        if 55 <= rsi < 70:
            score += 1
            reasons.append("RSI shows bullish momentum without being overbought.")
        elif 30 < rsi <= 45:
            score -= 1
            reasons.append("RSI shows bearish momentum without being oversold.")
        elif rsi >= 70:
            score -= 1
            reasons.append("RSI is overbought, increasing pullback risk.")
        elif rsi <= 30:
            score += 1
            reasons.append("RSI is oversold, increasing rebound potential.")
        else:
            reasons.append("RSI is neutral.")

        # Bollinger Bands: price location
        close = float(latest["Close"])
        lower_band = float(latest["BB_LOWER_20"])
        middle_band = float(latest["BB_MIDDLE_20"])
        upper_band = float(latest["BB_UPPER_20"])

        if close > upper_band:
            score -= 1
            reasons.append("Price is above the upper Bollinger Band.")
        elif close < lower_band:
            score += 1
            reasons.append("Price is below the lower Bollinger Band.")
        elif close > middle_band:
            score += 1
            reasons.append("Price is above the Bollinger middle band.")
        elif close < middle_band:
            score -= 1
            reasons.append("Price is below the Bollinger middle band.")
        else:
            reasons.append("Price is near the Bollinger middle band.")

        # ATR measures volatility, not direction.
        atr = float(latest["ATR_14"])
        atr_percent = atr / close

        if atr_percent > 0.001:
            reasons.append("Current volatility is relatively high.")
        else:
            reasons.append("Current volatility is relatively moderate.")

        if score >= 3:
            decision = Decision.BUY
        elif score <= -3:
            decision = Decision.SELL
        else:
            decision = Decision.WAIT

        return StrategyAnalysis(
            decision=decision,
            score=score,
            reasons=tuple(reasons),
        )

    def _validate_data(self, data: pd.DataFrame) -> None:
        if data.empty:
            raise ValueError("Cannot analyze an empty DataFrame.")

        missing_columns = self.REQUIRED_COLUMNS.difference(data.columns)

        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Strategy data is missing required columns: {missing}")

        latest = data.iloc[-1]

        if latest[list(self.REQUIRED_COLUMNS)].isna().any():
            raise ValueError("The latest candle contains incomplete indicator values.")