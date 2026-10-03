"""Machine-learning strategy.

The idea in plain English
-------------------------
1. Describe every candle with numbers ("features"): recent returns, RSI,
   distance from moving averages, volatility, time of day, ...
2. Label every candle with what happened NEXT: starting from its close,
   did price move ``barrier_atr`` x ATR UP before it moved the same
   distance DOWN (within ``horizon`` candles)? 1 = up first, 0 = down first.
3. Train a gradient-boosting model to predict that label from the features.
4. Trade only when the model is confident: BUY when P(up first) is above
   ``threshold``, SELL when it is below ``1 - threshold``.

Avoiding the classic machine-learning trading mistakes
------------------------------------------------------
* A label needs ``horizon`` FUTURE candles. When the model is trained at
  candle t it only uses candles whose label was already known at t
  (index + horizon < t). Using later labels would let the model peek at
  the future and make the backtest look far better than reality.
* Walk-forward training: the model is retrained every ``retrain_every``
  candles on the most recent ``train_bars`` candles, and every prediction
  is made by a model that has never seen that candle. Markets change, so
  the model keeps learning from recent data.
* The model is deliberately small and regularised so it cannot simply
  memorise the training data.
* It still has to pass the optimizer's out-of-sample test like every
  other strategy. On random data it is rejected (see the tests).
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from strategy import BUY, SELL, STRATEGY_CLASSES, WAIT, Strategy

FEATURE_COLUMNS = [
    "ret_1", "ret_3", "ret_6", "ret_12", "ret_24",
    "rsi_14", "rsi_7",
    "dist_ema_20", "dist_ema_50", "dist_ema_100",
    "slope_ema_20", "slope_ema_50",
    "bb_position", "bb_width",
    "atr_ratio", "atr_rank",
    "macd_hist",
    "range_position",
    "candle_body", "upper_wick", "lower_wick",
    "hour_sin", "hour_cos", "weekday",
]

# Walk-forward predictions are expensive; the optimizer tries several
# thresholds that share the same model, so keep the last few results.
_PROBABILITY_CACHE: OrderedDict[tuple, np.ndarray] = OrderedDict()
_CACHE_SIZE = 8


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """Every feature at candle i uses only candles up to and including i."""
    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    open_ = data["Open"]
    atr = data["ATR"].replace(0, np.nan)

    features = pd.DataFrame(index=data.index)

    # Returns measured in ATRs, so they mean the same thing on every pair.
    for period in (1, 3, 6, 12, 24):
        features[f"ret_{period}"] = (close - close.shift(period)) / atr

    for period in (14, 7):
        change = close.diff()
        gain = change.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        loss = (-change.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        features[f"rsi_{period}"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    for period in (20, 50, 100):
        ema = close.ewm(span=period, adjust=False).mean()
        features[f"dist_ema_{period}"] = (close - ema) / atr
        if period in (20, 50):
            features[f"slope_ema_{period}"] = (ema - ema.shift(5)) / atr

    middle = close.rolling(20).mean()
    std = close.rolling(20).std()
    features["bb_position"] = (close - middle) / (2 * std).replace(0, np.nan)
    features["bb_width"] = 4 * std / atr

    features["atr_ratio"] = atr / close
    features["atr_rank"] = atr.rolling(100).rank(pct=True)

    fast = close.ewm(span=12, adjust=False).mean()
    slow = close.ewm(span=26, adjust=False).mean()
    macd = fast - slow
    features["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / atr

    highest = high.rolling(50).max()
    lowest = low.rolling(50).min()
    features["range_position"] = (close - lowest) / (highest - lowest).replace(0, np.nan)

    candle_range = (high - low).replace(0, np.nan)
    features["candle_body"] = (close - open_) / candle_range
    features["upper_wick"] = (high - np.maximum(open_, close)) / candle_range
    features["lower_wick"] = (np.minimum(open_, close) - low) / candle_range

    if isinstance(data.index, pd.DatetimeIndex):
        hours = data.index.hour + data.index.minute / 60
        features["hour_sin"] = np.sin(2 * np.pi * hours / 24)
        features["hour_cos"] = np.cos(2 * np.pi * hours / 24)
        features["weekday"] = data.index.dayofweek
    else:
        features["hour_sin"] = 0.0
        features["hour_cos"] = 0.0
        features["weekday"] = 0

    return features[FEATURE_COLUMNS]


def barrier_labels(data: pd.DataFrame, horizon: int, barrier_atr: float) -> np.ndarray:
    """1.0 = price rose barrier_atr x ATR before falling the same distance,
    0.0 = it fell first, NaN = neither within ``horizon`` candles, both in
    the same candle (unknowable order), or not enough future candles.

    The label of candle i uses candles i+1 ... i+horizon only.
    """
    close = data["Close"].to_numpy(dtype=float)
    high = data["High"].to_numpy(dtype=float)
    low = data["Low"].to_numpy(dtype=float)
    atr = data["ATR"].to_numpy(dtype=float)

    count = len(close)
    labels = np.full(count, np.nan)
    decided = np.zeros(count, dtype=bool)

    upper = close + barrier_atr * atr
    lower = close - barrier_atr * atr

    for step in range(1, horizon + 1):
        future_high = np.full(count, np.nan)
        future_low = np.full(count, np.nan)
        future_high[:-step] = high[step:]
        future_low[:-step] = low[step:]

        up = future_high >= upper
        down = future_low <= lower
        undecided = ~decided

        labels[undecided & up & ~down] = 1.0
        labels[undecided & down & ~up] = 0.0
        decided |= up | down  # "both" stays NaN but counts as decided

    labels[~np.isfinite(atr)] = np.nan
    return labels


def _new_model() -> HistGradientBoostingClassifier:
    # Small, strongly regularised model: shallow trees, at least 100
    # examples per leaf. On pure noise it stays close to 50/50 instead of
    # becoming over-confident.
    return HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.03,
        max_depth=3,
        min_samples_leaf=100,
        l2_regularization=5.0,
        early_stopping=False,
        random_state=0,
    )


class MLStrategy(Strategy):  # Gradient-boosting model retrained walk-forward

    name = "ML Gradient Boosting"
    parameter_grid = {
        "horizon": [6, 12],
        "threshold": [0.55, 0.6],
    }

    def __init__(
        self,
        horizon: int = 12,
        threshold: float = 0.55,
        barrier_atr: float = 1.0,
        train_bars: int = 4000,
        retrain_every: int = 500,
    ) -> None:
        super().__init__(
            horizon=horizon,
            threshold=threshold,
            barrier_atr=barrier_atr,
            train_bars=train_bars,
            retrain_every=retrain_every,
        )
        self._live_model: HistGradientBoostingClassifier | None = None
        self._live_trained_on: object = None
        self._live_bars_since_fit = 0

    @property
    def warmup(self) -> int:
        return 150  # longest feature look-back (100-candle ATR rank) + margin

    @property
    def min_train(self) -> int:
        """Candles needed before the first model is trained."""
        return max(self.parameters["train_bars"] // 2, 500)

    @property
    def history_needed(self) -> int:
        return self.parameters["train_bars"] + self.parameters["horizon"] + self.warmup + 50

    def _validate_parameters(self) -> None:
        p = self.parameters
        if not 0.5 < p["threshold"] < 1:
            raise ValueError("Threshold must be between 0.5 and 1.")
        if p["horizon"] < 1 or p["retrain_every"] < 1 or p["train_bars"] < 200:
            raise ValueError("horizon, retrain_every must be >= 1 and train_bars >= 200.")
        if p["barrier_atr"] <= 0:
            raise ValueError("barrier_atr must be positive.")

    def _add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        return pd.concat([data, build_features(data)], axis=1)

    # --- backtesting: walk-forward over the whole history ----------------

    def training_schedule(self, count: int) -> list[tuple[int, int, int, int]]:
        """(train_start, train_end, predict_start, predict_end) windows.

        train_end is exclusive and chosen so that every training label was
        already known at predict_start: index + horizon < predict_start.
        """
        horizon = self.parameters["horizon"]
        train_bars = self.parameters["train_bars"]
        step = self.parameters["retrain_every"]

        schedule = []
        for predict_start in range(self.min_train, count, step):
            train_end = predict_start - horizon  # labels of indexes < this are known
            train_start = max(0, train_end - train_bars)
            schedule.append((train_start, train_end, predict_start, min(predict_start + step, count)))
        return schedule

    def probabilities(self, data: pd.DataFrame) -> np.ndarray:
        """Walk-forward P(up first) for every candle (NaN before the first model)."""
        p = self.parameters
        key = (
            len(data), str(data.index[0]), str(data.index[-1]),
            float(data["Close"].sum()),
            p["horizon"], p["barrier_atr"], p["train_bars"], p["retrain_every"],
        )
        if key in _PROBABILITY_CACHE:
            _PROBABILITY_CACHE.move_to_end(key)
            return _PROBABILITY_CACHE[key]

        features = data[FEATURE_COLUMNS].to_numpy(dtype=float)
        labels = barrier_labels(data, p["horizon"], p["barrier_atr"])
        result = np.full(len(data), np.nan)

        for train_start, train_end, predict_start, predict_end in self.training_schedule(len(data)):
            model = self._fit(features[train_start:train_end], labels[train_start:train_end])
            if model is not None:
                result[predict_start:predict_end] = self._predict(
                    model, features[predict_start:predict_end])

        _PROBABILITY_CACHE[key] = result
        if len(_PROBABILITY_CACHE) > _CACHE_SIZE:
            _PROBABILITY_CACHE.popitem(last=False)
        return result

    def _raw_signals(self, data: pd.DataFrame) -> np.ndarray:
        probability = self.probabilities(data)
        threshold = self.parameters["threshold"]

        with np.errstate(invalid="ignore"):
            return np.select(
                [probability >= threshold, probability <= 1 - threshold],
                [BUY, SELL],
                default=WAIT)

    # --- live trading: only the newest candle is needed -------------------

    def latest_signal(self, data: pd.DataFrame) -> int:
        """Signal for the last candle; retrains every ``retrain_every`` new candles."""
        p = self.parameters
        latest_time = data.index[-1]

        if latest_time != self._live_trained_on:
            self._live_bars_since_fit += 1

        if self._live_model is None or self._live_bars_since_fit >= p["retrain_every"]:
            features = data[FEATURE_COLUMNS].to_numpy(dtype=float)
            labels = barrier_labels(data, p["horizon"], p["barrier_atr"])
            train_end = len(data) - p["horizon"]
            train_start = max(0, train_end - p["train_bars"])
            self._live_model = self._fit(features[train_start:train_end],
                                         labels[train_start:train_end])
            self._live_bars_since_fit = 0

        self._live_trained_on = latest_time

        if self._live_model is None or len(data) <= self.warmup:
            return WAIT

        probability = self._predict(self._live_model, data[FEATURE_COLUMNS].to_numpy(dtype=float)[-1:])[0]
        if probability >= p["threshold"]:
            return BUY
        if probability <= 1 - p["threshold"]:
            return SELL
        return WAIT

    # --- helpers ------------------------------------------------------------

    @staticmethod
    def _fit(features: np.ndarray, labels: np.ndarray) -> HistGradientBoostingClassifier | None:
        usable = np.isfinite(labels) & np.isfinite(features).all(axis=1)
        x, y = features[usable], labels[usable]

        # Need enough examples of BOTH outcomes to learn anything.
        if len(y) < 200 or min((y == 1).sum(), (y == 0).sum()) < 50:
            return None

        model = _new_model()
        model.fit(x, y.astype(int))
        return model

    @staticmethod
    def _predict(model: HistGradientBoostingClassifier, features: np.ndarray) -> np.ndarray:
        result = np.full(len(features), np.nan)
        usable = np.isfinite(features).all(axis=1)
        if usable.any():
            result[usable] = model.predict_proba(features[usable])[:, 1]
        return result


STRATEGY_CLASSES[MLStrategy.__name__] = MLStrategy
