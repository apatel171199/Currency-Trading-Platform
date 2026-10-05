import pandas as pd
import pytest

from decision import Decision
from strategy import STRATEGY_CLASSES, IndicatorStrategy, build_strategy


@pytest.mark.parametrize("cls", STRATEGY_CLASSES.values(), ids=list(STRATEGY_CLASSES))
def test_signals_do_not_look_ahead(cls, candles):
    """A signal must not change when FUTURE candles are added or changed."""
    for parameters in cls.parameter_combinations():
        strategy = cls(**parameters)
        full = strategy.signals(strategy.prepare(candles))

        cut = 2000
        past_only = strategy.signals(strategy.prepare(candles.iloc[:cut]))

        pd.testing.assert_series_equal(full.iloc[:cut], past_only, check_names=False)


GRID_STRATEGIES = {name: cls for name, cls in STRATEGY_CLASSES.items() if cls.parameter_combinations()}


@pytest.mark.parametrize("cls", GRID_STRATEGIES.values(), ids=list(GRID_STRATEGIES))
def test_strategies_produce_both_directions(cls, candles):
    strategy = cls()
    signals = strategy.signals(strategy.prepare(candles))
    assert set(signals.unique()) <= {-1, 0, 1}
    assert (signals == 1).any() and (signals == -1).any()
    assert (signals.iloc[: strategy.warmup] == 0).all()


def test_invalid_combinations_are_skipped():
    combos = STRATEGY_CLASSES["EmaCrossStrategy"].parameter_combinations()
    assert all(c["fast"] < c["slow"] for c in combos)


def test_build_strategy_round_trip():
    strategy = build_strategy("MacdCrossStrategy", {"fast": 8, "slow": 21, "signal": 5, "trend_filter": 0})
    assert strategy.label == "MACD Cross(fast=8, slow=21, signal=5, trend_filter=0)"

    with pytest.raises(ValueError):
        build_strategy("Nope", {})


def test_indicator_strategy_explains_itself(candles):
    strategy = IndicatorStrategy()
    analysis = strategy.analyze(strategy.prepare(candles))
    assert analysis.decision in Decision
    assert len(analysis.reasons) == 3
