"""Broker interface used by the live trader.

``MT5Broker`` (mt5_broker.py) talks to a real MetaTrader 5 terminal.
``PaperBroker`` (below) simulates fills in memory so the whole trading
loop can be tested — on any operating system — without risking money.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from backtester import pip_size_for

CONTRACT_SIZE = 100_000  # 1 standard forex lot


@dataclass(frozen=True)
class AccountInfo:
    login: int
    balance: float
    equity: float
    currency: str
    is_demo: bool
    trade_allowed: bool


@dataclass(frozen=True)
class SymbolSpec:
    name: str
    digits: int
    point: float
    tick_size: float
    tick_value: float  # account-currency value of one tick for 1.0 lot
    volume_min: float
    volume_max: float
    volume_step: float
    stops_level_points: int = 0

    @property
    def pip_size(self) -> float:
        return pip_size_for(self.name)

    @property
    def min_stop_distance(self) -> float:
        return self.stops_level_points * self.point


@dataclass(frozen=True)
class Quote:
    bid: float
    ask: float

    @property
    def spread(self) -> float:
        return self.ask - self.bid


@dataclass(frozen=True)
class Position:
    ticket: int
    symbol: str
    direction: int  # +1 buy, -1 sell
    volume: float
    open_price: float
    stop_loss: float
    take_profit: float
    profit: float
    magic: int


@dataclass(frozen=True)
class OrderResult:
    ok: bool
    message: str
    ticket: int | None = None
    price: float | None = None


class Broker(ABC):

    @abstractmethod
    def account(self) -> AccountInfo: ...

    @abstractmethod
    def symbol_spec(self, symbol: str) -> SymbolSpec: ...

    @abstractmethod
    def candles(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        """Most recent CLOSED candles (the still-forming candle is excluded)."""

    @abstractmethod
    def quote(self, symbol: str) -> Quote: ...

    @abstractmethod
    def positions(self, symbol: str, magic: int) -> list[Position]: ...

    @abstractmethod
    def open_position(
        self,
        symbol: str,
        direction: int,
        volume: float,
        stop_loss: float,
        take_profit: float,
        magic: int,
        comment: str,
    ) -> OrderResult: ...

    @abstractmethod
    def close_position(self, position: Position) -> OrderResult: ...

    def shutdown(self) -> None:
        pass


def calculate_volume(risk_amount: float, stop_distance: float, spec: SymbolSpec) -> float:
    """Lot size so that hitting the stop loses about ``risk_amount``.

    Rounds DOWN to the broker's volume step. Returns 0.0 when even the
    smallest allowed lot would risk more than ``risk_amount`` — the trade
    is then skipped instead of quietly risking more than intended.
    """
    if risk_amount <= 0 or stop_distance <= 0 or spec.tick_size <= 0 or spec.tick_value <= 0:
        return 0.0

    loss_per_lot = stop_distance / spec.tick_size * spec.tick_value
    raw_volume = risk_amount / loss_per_lot

    steps = math.floor(raw_volume / spec.volume_step + 1e-9)
    volume = round(steps * spec.volume_step, 8)

    if volume < spec.volume_min:
        return 0.0

    return min(volume, spec.volume_max)


class PaperBroker(Broker):
    """Simulated broker that fills orders at the latest candle close.

    ``feed(symbol, timeframe, count)`` must return closed candles. It can be
    Twelve Data, a CSV replay, or even a real MT5 connection (paper trading
    on live prices). Profit is converted to a USD account; for crosses
    without USD (e.g. EURGBP) the conversion is approximate.
    """

    def __init__(
        self,
        feed: Callable[[str, str, int], pd.DataFrame],
        starting_balance: float = 10_000.0,
        spread_pips: float = 1.0,
    ) -> None:
        self.feed = feed
        self.balance = starting_balance
        self.spread_pips = spread_pips
        self.closed_trades: list[dict] = []

        self._positions: dict[int, dict] = {}
        self._last_candles: dict[str, pd.DataFrame] = {}
        self._next_ticket = 1

    # --- market data -------------------------------------------------

    def candles(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        data = self.feed(symbol, timeframe, count)
        self._last_candles[symbol] = data
        self._update_stops(symbol, data)
        return data

    def quote(self, symbol: str) -> Quote:
        data = self._require_candles(symbol)
        mid = float(data["Close"].iloc[-1])
        half = self.spread_pips * pip_size_for(symbol) / 2
        return Quote(bid=mid - half, ask=mid + half)

    def symbol_spec(self, symbol: str) -> SymbolSpec:
        pip = pip_size_for(symbol)
        tick_size = pip / 10
        price = float(self._require_candles(symbol)["Close"].iloc[-1])

        return SymbolSpec(
            name=symbol,
            digits=3 if pip == 0.01 else 5,
            point=tick_size,
            tick_size=tick_size,
            tick_value=self._to_account_currency(symbol, tick_size * CONTRACT_SIZE, price),
            volume_min=0.01,
            volume_max=100.0,
            volume_step=0.01,
        )

    # --- account & positions -----------------------------------------

    def account(self) -> AccountInfo:
        floating = sum(self._floating_profit(p) for p in self._positions.values())
        return AccountInfo(
            login=0,
            balance=self.balance,
            equity=self.balance + floating,
            currency="USD",
            is_demo=True,
            trade_allowed=True,
        )

    def positions(self, symbol: str, magic: int) -> list[Position]:
        return [
            self._to_position(ticket, p)
            for ticket, p in self._positions.items()
            if p["symbol"] == symbol and p["magic"] == magic
        ]

    def open_position(self, symbol, direction, volume, stop_loss, take_profit, magic, comment):
        quote = self.quote(symbol)
        price = quote.ask if direction > 0 else quote.bid
        ticket = self._next_ticket
        self._next_ticket += 1

        self._positions[ticket] = dict(
            symbol=symbol,
            direction=direction,
            volume=volume,
            open_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            magic=magic,
            comment=comment,
            # Filled at this candle's close, i.e. the next candle's open.
            signal_candle=self._require_candles(symbol).index[-1],
            open_time=self._require_candles(symbol).index[-1],
        )
        return OrderResult(True, "paper fill", ticket=ticket, price=price)

    def close_position(self, position: Position) -> OrderResult:
        quote = self.quote(position.symbol)
        price = quote.bid if position.direction > 0 else quote.ask
        self._close(position.ticket, price, "CLOSED")
        return OrderResult(True, "paper close", ticket=position.ticket, price=price)

    # --- internals ---------------------------------------------------

    def _require_candles(self, symbol: str) -> pd.DataFrame:
        if symbol not in self._last_candles:
            raise RuntimeError(f"No candles loaded for {symbol} yet.")
        return self._last_candles[symbol]

    def _update_stops(self, symbol: str, data: pd.DataFrame) -> None:
        for ticket, p in list(self._positions.items()):
            if p["symbol"] != symbol:
                continue

            new_candles = data[data.index > p["open_time"]]

            for _, candle in new_candles.iterrows():
                if p["direction"] > 0:
                    stop_hit = candle["Low"] <= p["stop_loss"]
                    target_hit = candle["High"] >= p["take_profit"]
                else:
                    stop_hit = candle["High"] >= p["stop_loss"]
                    target_hit = candle["Low"] <= p["take_profit"]

                if stop_hit:
                    self._close(ticket, p["stop_loss"], "STOP_LOSS")
                    break
                if target_hit:
                    self._close(ticket, p["take_profit"], "TAKE_PROFIT")
                    break
            else:
                if len(new_candles):
                    p["open_time"] = new_candles.index[-1]

    def _close(self, ticket: int, price: float, reason: str) -> None:
        p = self._positions.pop(ticket)
        profit = self._profit(p, price)
        self.balance += profit
        self.closed_trades.append({**p, "close_price": price, "profit": profit, "reason": reason})

    def _profit(self, p: dict, price: float) -> float:
        quote_profit = p["direction"] * (price - p["open_price"]) * p["volume"] * CONTRACT_SIZE
        return self._to_account_currency(p["symbol"], quote_profit, price)

    def _floating_profit(self, p: dict) -> float:
        quote = self.quote(p["symbol"])
        price = quote.bid if p["direction"] > 0 else quote.ask
        return self._profit(p, price)

    @staticmethod
    def _to_account_currency(symbol: str, amount: float, price: float) -> float:
        cleaned = symbol.replace("/", "").upper()
        if cleaned[3:6] == "USD":
            return amount
        if cleaned[:3] == "USD" and price > 0:
            return amount / price
        return amount  # approximation for non-USD crosses

    def _to_position(self, ticket: int, p: dict) -> Position:
        return Position(
            ticket=ticket,
            symbol=p["symbol"],
            direction=p["direction"],
            volume=p["volume"],
            open_price=p["open_price"],
            stop_loss=p["stop_loss"],
            take_profit=p["take_profit"],
            profit=self._floating_profit(p),
            magic=p["magic"],
        )
