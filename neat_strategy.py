"""A strategy whose "brain" is a neural network evolved by NEAT (evolution.py).

The evolved network is saved as a small JSON file. This module loads it and
runs it over every candle at once with numpy, which is what makes it fast
enough to backtest thousands of networks during evolution.

Inputs: the same 24 features the ML strategy uses (ml_strategy.py),
standardised with the mean/std of the TRAINING years only.
Output: one number per candle. Above +threshold = BUY, below -threshold =
SELL, otherwise WAIT.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from indicators import IndicatorCalculator
from ml_strategy import FEATURE_COLUMNS, build_features
from strategy import BUY, SELL, STRATEGY_CLASSES, WAIT, Strategy


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(5.0 * z, -60.0, 60.0)))


# numpy versions of neat-python's activation functions (same formulas).
ACTIVATIONS = {
    "sigmoid": _sigmoid,
    "tanh": lambda z: np.tanh(np.clip(2.5 * z, -60.0, 60.0)),
    "relu": lambda z: np.maximum(z, 0.0),
    "identity": lambda z: z,
    "clamped": lambda z: np.clip(z, -1.0, 1.0),
}


def standardise(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Scale features with training statistics; missing values become 0."""
    scaled = (features - mean) / np.where(std > 0, std, 1.0)
    scaled = np.clip(scaled, -5.0, 5.0)
    return np.nan_to_num(scaled, nan=0.0)


def run_network(network: dict[str, Any], inputs: np.ndarray) -> np.ndarray:
    """Evaluates a saved network on many rows at once. inputs: (rows, features)."""
    values: dict[int, np.ndarray] = {
        key: inputs[:, position] for position, key in enumerate(network["input_keys"])
    }
    zeros = np.zeros(len(inputs))

    for node in network["nodes"]:
        total = zeros.copy()
        for source, weight in node["links"]:
            total += weight * values.get(source, zeros)
        values[node["key"]] = ACTIVATIONS[node["activation"]](node["bias"] + node["response"] * total)

    return values.get(network["output_key"], zeros)


def network_inputs(network: dict[str, Any], features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(standardised inputs, rows with missing indicators)."""
    raw = features[FEATURE_COLUMNS].to_numpy(dtype=float)
    missing = ~np.isfinite(raw).all(axis=1)
    inputs = standardise(raw, np.asarray(network["feature_mean"]), np.asarray(network["feature_std"]))
    return inputs, missing


def signals_from_inputs(network: dict[str, Any], inputs: np.ndarray, missing: np.ndarray) -> np.ndarray:
    output = run_network(network, inputs)
    threshold = network["threshold"]

    signals = np.select([output > threshold, output < -threshold], [BUY, SELL], default=WAIT)
    signals[missing] = WAIT  # never trade on incomplete indicators
    return signals


def network_signals(network: dict[str, Any], features: pd.DataFrame) -> np.ndarray:
    return signals_from_inputs(network, *network_inputs(network, features))


def prepare_features(candles: pd.DataFrame, atr_period: int = 14) -> pd.DataFrame:
    """Candles + ATR + the 24 network inputs (used by evolution and trading)."""
    with_atr = IndicatorCalculator().add_atr(candles, period=atr_period, column_name="ATR")
    return pd.concat([with_atr, build_features(with_atr)], axis=1)


def save_network(network: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(network, indent=2), encoding="utf-8")
    return path


def load_network(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise ValueError(f"Evolved network file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


class NeatStrategy(Strategy):  # Trades with a neural network evolved by NEAT

    name = "NEAT Evolved"

    def __init__(self, network_file: str) -> None:
        super().__init__(network_file=network_file)
        self.network = load_network(network_file)

    @property
    def label(self) -> str:
        return f"{self.name}({Path(self.parameters['network_file']).stem})"

    @property
    def warmup(self) -> int:
        return 150

    @classmethod
    def parameter_combinations(cls) -> list[dict[str, Any]]:
        # Networks come from `python app.py evolve`, not from a grid search.
        return []

    def _add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        return pd.concat([data, build_features(data)], axis=1)

    def _raw_signals(self, data: pd.DataFrame) -> np.ndarray:
        return network_signals(self.network, data)


STRATEGY_CLASSES[NeatStrategy.__name__] = NeatStrategy
