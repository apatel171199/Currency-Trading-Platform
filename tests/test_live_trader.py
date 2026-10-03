import pytest

from backtester import Backtester, BacktestSettings
from broker import AccountInfo, PaperBroker
from conftest import make_candles
from live_trader import LiveTrader, TradingHalted, TradingSettings
from optimizer import Candidate, Selection


def selection(approved=True, stop=1.5, target=3.0):
    return Selection(
        symbol="EURUSD", timeframe="H1", approved=approved,
        candidate=Candidate("EmaCrossStrategy", {"fast": 10, "slow": 30}, stop, target),
        in_sample=None, out_of_sample=None, candles=0, data_start="", data_end="",
    )


class Replay:
    """Feeds candles one at a time, like a live market."""

    def __init__(self, data, start):
        self.data = data
        self.cursor = start

    def __call__(self, symbol, timeframe, count):
        return self.data.iloc[: self.cursor]


def settings(tmp_path, **overrides):
    values = dict(max_daily_loss_pct=100, max_drawdown_pct=99, max_spread_pips=5,
                  journal_path=str(tmp_path / "journal.csv"))
    values.update(overrides)
    return TradingSettings(**values)


def test_live_trades_match_the_backtest(tmp_path):
    data = make_candles(n=1500, seed=3)
    replay = Replay(data, start=200)
    broker = PaperBroker(replay, starting_balance=100_000, spread_pips=1.0)
    trader = LiveTrader(broker, {"EURUSD": selection()}, settings(tmp_path))
    trader.check_account()

    while replay.cursor <= len(data):
        trader.run_once()
        replay.cursor += 1

    candidate = selection().candidate
    backtest = Backtester(candidate.build(), BacktestSettings(
        spread_pips=1.0, stop_atr_multiple=1.5, target_atr_multiple=3.0)).run(data)

    expected = [(t.entry_time, t.direction.name, t.exit_reason) for t in backtest.trades
                if t.exit_reason != "END_OF_DATA" and t.entry_time > data.index[200]]
    reason = {"CLOSED": "OPPOSITE_SIGNAL", "STOP_LOSS": "STOP_LOSS", "TAKE_PROFIT": "TAKE_PROFIT"}
    live = [
        (data.index[data.index.get_loc(t["signal_candle"]) + 1],
         "BUY" if t["direction"] > 0 else "SELL",
         reason[t["reason"]])
        for t in broker.closed_trades
    ]

    assert len(live) > 10
    assert live == expected


def test_refuses_real_money_account_by_default(tmp_path):
    broker = PaperBroker(Replay(make_candles(500), 500))
    broker.account = lambda: AccountInfo(1, 1000, 1000, "USD", is_demo=False, trade_allowed=True)
    trader = LiveTrader(broker, {"EURUSD": selection()}, settings(tmp_path))

    with pytest.raises(TradingHalted, match="REAL-money"):
        trader.check_account()

    allowed = LiveTrader(broker, {"EURUSD": selection()},
                         settings(tmp_path, allow_real_account=True))
    allowed.check_account()


def test_unapproved_strategy_is_never_traded(tmp_path):
    data = make_candles(n=1500, seed=3)
    replay = Replay(data, start=200)
    broker = PaperBroker(replay, starting_balance=100_000)
    trader = LiveTrader(broker, {"EURUSD": selection(approved=False)}, settings(tmp_path))

    with pytest.raises(TradingHalted, match="No market has an approved strategy"):
        trader.check_account()

    for replay.cursor in range(200, 1500):
        trader.run_once()
    assert broker.closed_trades == [] and broker.positions("EURUSD", trader.settings.magic) == []


def test_dry_run_sends_no_orders(tmp_path):
    data = make_candles(n=1500, seed=3)
    replay = Replay(data, start=200)
    broker = PaperBroker(replay, starting_balance=100_000)
    trader = LiveTrader(broker, {"EURUSD": selection()}, settings(tmp_path, dry_run=True))

    for replay.cursor in range(200, 1500):
        trader.run_once()

    assert broker.closed_trades == []
    journal = (tmp_path / "journal.csv").read_text()
    assert "DRY RUN" in journal


def test_small_account_skips_instead_of_over_risking(tmp_path, caplog):
    data = make_candles(n=1500, seed=3)
    replay = Replay(data, start=200)
    broker = PaperBroker(replay, starting_balance=100)  # 1% = $1, below 0.01 lot risk
    trader = LiveTrader(broker, {"EURUSD": selection()}, settings(tmp_path))

    for replay.cursor in range(200, 1500):
        trader.run_once()

    assert broker.closed_trades == []
    assert "minimum lot" in caplog.text


def test_drawdown_limit_halts_trading(tmp_path):
    data = make_candles(n=1500, seed=3)
    replay = Replay(data, start=200)
    broker = PaperBroker(replay, starting_balance=100_000)
    trader = LiveTrader(broker, {"EURUSD": selection()},
                        settings(tmp_path, risk_per_trade=0.05, max_drawdown_pct=5))

    with pytest.raises(TradingHalted, match="below its peak"):
        for replay.cursor in range(200, 1500):
            trader.run_once()
