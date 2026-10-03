import numpy as np
import pandas as pd
import pytest

from backtester import Backtester, BacktestSettings, pip_size_for


def frame(rows):
    index = pd.date_range("2024-01-01", periods=len(rows), freq="h")
    data = pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=index)
    data["ATR"] = 0.0010
    return data


def run(rows, signals, **settings):
    defaults = dict(spread_pips=0.0, stop_atr_multiple=1.0, target_atr_multiple=2.0)
    defaults.update(settings)
    tester = Backtester(None, BacktestSettings(**defaults))
    return tester.run_signals(frame(rows), np.array(signals))


def test_buy_hits_target_for_plus_two_r():
    rows = [
        (1.1000, 1.1005, 1.0995, 1.1000),  # signal candle
        (1.1000, 1.1010, 1.0995, 1.1008),  # entry at open 1.1000
        (1.1008, 1.1025, 1.1005, 1.1020),  # target 1.1020 hit
        (1.1020, 1.1022, 1.1018, 1.1020),
    ]
    result = run(rows, [1, 0, 0, 0])
    trade = result.trades[0]
    assert trade.exit_reason == "TAKE_PROFIT"
    assert trade.r_multiple == pytest.approx(2.0)
    assert result.ending_balance == pytest.approx(102.0)  # +2% on 100


def test_stop_assumed_first_when_both_hit():
    rows = [
        (1.1000, 1.1001, 1.0999, 1.1000),
        (1.1000, 1.1030, 1.0980, 1.1000),  # both stop and target inside one candle
        (1.1000, 1.1001, 1.0999, 1.1000),
    ]
    trade = run(rows, [1, 0, 0]).trades[0]
    assert trade.exit_reason == "STOP_LOSS"
    assert trade.r_multiple == pytest.approx(-1.0)


def test_spread_costs_are_charged():
    rows = [
        (1.1000, 1.1001, 1.0999, 1.1000),
        (1.1000, 1.1001, 1.0999, 1.1000),
        (1.1000, 1.1001, 1.0999, 1.1000),
    ]
    trade = run(rows, [1, 0, 0], spread_pips=1.0).trades[0]
    assert trade.exit_reason == "END_OF_DATA"
    # Paid one full pip of spread on a 10-pip stop.
    assert trade.r_multiple == pytest.approx(-0.1)


def test_opposite_signal_reverses_position():
    rows = [(1.1000, 1.1001, 1.0999, 1.1000)] * 5
    result = run(rows, [1, 0, -1, 0, 0])
    assert [t.direction.name for t in result.trades] == ["BUY", "SELL"]
    assert result.trades[0].exit_reason == "OPPOSITE_SIGNAL"
    assert result.trades[1].entry_time == result.trades[0].exit_time


def test_sell_gap_through_stop_fills_at_open():
    rows = [
        (1.1000, 1.1001, 1.0999, 1.1000),
        (1.1000, 1.1002, 1.0998, 1.1000),  # entry, stop at 1.1010
        (1.1030, 1.1035, 1.1025, 1.1030),  # gaps above the stop
    ]
    trade = run(rows, [-1, 0, 0]).trades[0]
    assert trade.exit_reason == "STOP_LOSS"
    assert trade.r_multiple == pytest.approx(-3.0)


def test_pip_size():
    assert pip_size_for("USDJPY") == 0.01
    assert pip_size_for("EURUSD") == 0.0001


def test_full_run_with_strategy(candles):
    from strategy import EmaCrossStrategy

    result = Backtester(EmaCrossStrategy()).run(candles)
    assert result.trade_count > 10
    assert len(result.equity_curve) == result.trade_count + 1
    assert set(result.trades_dataframe()["exit_reason"]) <= {
        "STOP_LOSS", "TAKE_PROFIT", "OPPOSITE_SIGNAL", "END_OF_DATA"}
