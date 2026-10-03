import numpy as np
import pandas as pd
import pytest

from broker import PaperBroker
from conftest import make_candles, make_pattern_candles
from live_trader import LiveTrader, TradingSettings
from ml_strategy import FEATURE_COLUMNS, MLStrategy, barrier_labels, build_features
from optimizer import Candidate, Selection, StrategyOptimizer


def test_barrier_labels():
    rows = [
        # Close 1.0, ATR 0.1 -> barriers at 1.1 / 0.9 (barrier_atr = 1)
        (1.00, 1.00, 1.00, 1.00),
        (1.00, 1.05, 0.95, 1.00),  # nothing
        (1.00, 1.11, 0.98, 1.10),  # up barrier first -> label of row 0 is 1
        (1.10, 1.10, 1.10, 1.10),
    ]
    data = pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"])
    data["ATR"] = 0.1

    labels = barrier_labels(data, horizon=2, barrier_atr=1.0)
    assert labels[0] == 1.0
    assert np.isnan(labels[-1])  # no future candles to know the answer

    short = barrier_labels(data, horizon=1, barrier_atr=1.0)
    assert np.isnan(short[0])  # barrier not reached within 1 candle


def test_barrier_label_both_hit_in_one_candle_is_unknown():
    rows = [(1.0, 1.0, 1.0, 1.0), (1.0, 1.2, 0.8, 1.0)]
    data = pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"])
    data["ATR"] = 0.1
    assert np.isnan(barrier_labels(data, horizon=1, barrier_atr=1.0)[0])


def test_training_only_uses_labels_known_at_that_time():
    strategy = MLStrategy(horizon=12, train_bars=1000, retrain_every=250)
    schedule = strategy.training_schedule(5000)

    assert schedule[0][2] == strategy.min_train
    for train_start, train_end, predict_start, predict_end in schedule:
        # The newest training row's label looks at rows up to
        # train_end - 1 + horizon, which must be BEFORE predict_start.
        assert (train_end - 1) + 12 < predict_start
        assert train_end - train_start <= 1000
        assert predict_start < predict_end


def test_features_do_not_look_ahead(candles):
    prepared = MLStrategy().prepare(candles)
    full = build_features(prepared)
    past_only = build_features(prepared.iloc[:1500])
    pd.testing.assert_frame_equal(full.iloc[:1500], past_only)
    assert list(full.columns) == FEATURE_COLUMNS


def test_walk_forward_signals_do_not_look_ahead():
    data = make_candles(n=3000, seed=11)
    strategy = MLStrategy(horizon=6, train_bars=600, retrain_every=100)

    full = strategy.signals(strategy.prepare(data))
    past_only = strategy.signals(strategy.prepare(data.iloc[:2000]))

    assert (full.iloc[:2000] != 0).sum() > 50  # the model really traded
    pd.testing.assert_series_equal(full.iloc[:2000], past_only, check_names=False)


def test_model_learns_a_real_pattern():
    data = make_pattern_candles(strength=0.5)
    optimizer = StrategyOptimizer(strategy_classes=[MLStrategy])
    selection, _ = optimizer.optimize(data, "PATTERN", "H1")

    assert selection.approved
    assert selection.out_of_sample.t_stat >= 2
    assert selection.out_of_sample.expectancy_r > 0.2


def test_model_rejected_on_random_data():
    data = make_candles(n=20000, seed=104)
    selection, _ = StrategyOptimizer(strategy_classes=[MLStrategy]).optimize(data, "RW", "H1")
    assert not selection.approved


def test_latest_signal_retrains_only_every_n_candles(monkeypatch):
    data = MLStrategy().prepare(make_pattern_candles(n=6000))
    strategy = MLStrategy(retrain_every=5)
    fits = []
    original = MLStrategy._fit
    monkeypatch.setattr(MLStrategy, "_fit", staticmethod(lambda x, y: fits.append(1) or original(x, y)))

    signals = [strategy.latest_signal(data.iloc[: 5000 + i]) for i in range(11)]

    assert set(signals) <= {-1, 0, 1}
    assert len(fits) == 3  # first candle, then after 5 and 10 new candles


def test_live_trader_runs_ml_strategy(tmp_path):
    data = make_pattern_candles(n=6000)
    requested = []

    def feed(symbol, timeframe, count):
        requested.append(count)
        return data.iloc[: feed.cursor].tail(count)

    feed.cursor = 5000
    broker = PaperBroker(feed, starting_balance=100_000)
    selection = Selection(
        symbol="EURUSD", timeframe="H1", approved=True,
        candidate=Candidate("MLStrategy", {"horizon": 12, "threshold": 0.55, "barrier_atr": 1.0,
                                           "train_bars": 4000, "retrain_every": 500}, 1.5, 3.0),
        in_sample=None, out_of_sample=None, candles=0, data_start="", data_end="",
    )
    trader = LiveTrader(broker, {"EURUSD": selection}, TradingSettings(
        max_daily_loss_pct=100, max_drawdown_pct=99, max_spread_pips=5,
        journal_path=str(tmp_path / "journal.csv")))

    for feed.cursor in range(5000, 5300):
        trader.run_once()

    assert min(requested) >= MLStrategy().history_needed  # enough candles to train
    assert len(broker.closed_trades) > 0
