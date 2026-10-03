"""MT5Broker tested against a fake MetaTrader5 module (the real one is Windows-only)."""
from types import SimpleNamespace

import numpy as np
import pytest

from broker import calculate_volume, SymbolSpec
from mt5_broker import MT5Broker


class FakeMT5:
    TIMEFRAME_H1 = 16385
    ACCOUNT_TRADE_MODE_DEMO = 0
    POSITION_TYPE_BUY = 0
    TRADE_ACTION_DEAL = 1
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    ORDER_TIME_GTC = 0
    ORDER_FILLING_FOK = 0
    ORDER_FILLING_IOC = 1
    ORDER_FILLING_RETURN = 2
    TRADE_RETCODE_DONE = 10009
    TRADE_RETCODE_PLACED = 10008

    def __init__(self, check_retcode=0, filling_mode=2):
        self.sent = []
        self.check_retcode = check_retcode
        self.filling_mode = filling_mode
        self.rates_call = None

    def initialize(self, **kwargs):
        self.init_kwargs = kwargs
        return True

    def shutdown(self):
        pass

    def last_error(self):
        return (1, "fake")

    def symbol_select(self, symbol, enable):
        return True

    def symbol_info(self, symbol):
        return SimpleNamespace(digits=5, point=0.00001, trade_tick_size=0.00001,
                               trade_tick_value=1.0, volume_min=0.01, volume_max=50.0,
                               volume_step=0.01, trade_stops_level=10,
                               filling_mode=self.filling_mode)

    def symbol_info_tick(self, symbol):
        return SimpleNamespace(bid=1.10000, ask=1.10010)

    def copy_rates_from_pos(self, symbol, timeframe, start, count):
        self.rates_call = (symbol, timeframe, start, count)
        dtype = [("time", "i8"), ("open", "f8"), ("high", "f8"), ("low", "f8"),
                 ("close", "f8"), ("tick_volume", "i8"), ("spread", "i4"), ("real_volume", "i8")]
        return np.array([(1_700_000_000 + 3600 * i, 1.1, 1.2, 1.0, 1.15, 10, 12, 0)
                         for i in range(count)], dtype=dtype)

    def account_info(self):
        return SimpleNamespace(login=123, balance=1000.0, equity=1000.0, currency="USD",
                               trade_mode=0, trade_allowed=True)

    def terminal_info(self):
        return SimpleNamespace(trade_allowed=True)

    def positions_get(self, symbol):
        return (
            SimpleNamespace(ticket=1, symbol=symbol, type=0, volume=0.1, price_open=1.1,
                            sl=1.09, tp=1.12, profit=5.0, magic=42),
            SimpleNamespace(ticket=2, symbol=symbol, type=1, volume=0.1, price_open=1.1,
                            sl=1.11, tp=1.08, profit=-1.0, magic=999),  # someone else's trade
        )

    def order_check(self, request):
        return SimpleNamespace(retcode=self.check_retcode, comment="check")

    def order_send(self, request):
        self.sent.append(request)
        return SimpleNamespace(retcode=self.TRADE_RETCODE_DONE, order=77, price=request["price"],
                               comment="done")


def test_candles_skip_forming_bar_and_rename():
    fake = FakeMT5()
    broker = MT5Broker(mt5_module=fake)
    data = broker.candles("EURUSD", "H1", 5)
    assert fake.rates_call == ("EURUSD", FakeMT5.TIMEFRAME_H1, 1, 5)
    assert list(data.columns) == ["Open", "High", "Low", "Close", "Volume", "Spread"]
    assert str(data.index.tz) == "UTC"


def test_open_position_builds_correct_request():
    fake = FakeMT5()
    broker = MT5Broker(mt5_module=fake)
    result = broker.open_position("EURUSD", 1, 0.05, 1.0985012345, 1.1030098765, 42, "x" * 40)

    assert result.ok and result.ticket == 77
    request = fake.sent[0]
    assert request["type"] == FakeMT5.ORDER_TYPE_BUY
    assert request["price"] == 1.10010  # buys at the ask
    assert request["sl"] == 1.0985 and request["tp"] == 1.10301
    assert request["type_filling"] == FakeMT5.ORDER_FILLING_IOC
    assert len(request["comment"]) == 31


def test_rejected_order_check_sends_nothing():
    fake = FakeMT5(check_retcode=10019)  # not enough money
    broker = MT5Broker(mt5_module=fake)
    result = broker.open_position("EURUSD", -1, 0.05, 1.11, 1.09, 42, "t")
    assert not result.ok and fake.sent == []


def test_positions_filtered_by_magic_and_close_uses_opposite_side():
    fake = FakeMT5()
    broker = MT5Broker(mt5_module=fake)
    positions = broker.positions("EURUSD", 42)
    assert [p.ticket for p in positions] == [1]
    broker.close_position(positions[0])
    assert fake.sent[0]["type"] == FakeMT5.ORDER_TYPE_SELL
    assert fake.sent[0]["position"] == 1
    assert fake.sent[0]["price"] == 1.10000  # closes a buy at the bid


def test_account_detects_demo():
    account = MT5Broker(mt5_module=FakeMT5()).account()
    assert account.is_demo and account.trade_allowed


SPEC = SymbolSpec("EURUSD", 5, 0.00001, 0.00001, 1.0, 0.01, 50.0, 0.01)


@pytest.mark.parametrize("risk, stop, expected", [
    (100.0, 0.0020, 0.50),   # $100 risk / 20 pips ($200 per lot) = 0.5 lots
    (100.0, 0.0030, 0.33),   # rounds DOWN, never up
    (1.0, 0.0020, 0.0),      # $1 risk is below the minimum lot -> skip
    (1e9, 0.0010, 50.0),     # capped at volume_max
])
def test_calculate_volume(risk, stop, expected):
    assert calculate_volume(risk, stop, SPEC) == pytest.approx(expected)
