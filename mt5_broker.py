"""MetaTrader 5 connection.

Requires Windows, the MetaTrader 5 terminal installed and logged in, and
``pip install MetaTrader5``. In the terminal enable:
Tools -> Options -> Expert Advisors -> "Allow algorithmic trading".
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from broker import AccountInfo, Broker, OrderResult, Position, Quote, SymbolSpec
from market import normalize_timeframe


class MT5Broker(Broker):

    def __init__(
        self,
        login: int | None = None,
        password: str | None = None,
        server: str | None = None,
        path: str | None = None,
        deviation_points: int = 20,
        mt5_module: Any = None,
    ) -> None:
        if mt5_module is None:
            try:
                import MetaTrader5 as mt5_module  # type: ignore[no-redef]
            except ImportError as error:
                raise RuntimeError(
                    "The MetaTrader5 package is not installed. It only works on "
                    "Windows: pip install MetaTrader5"
                ) from error

        self.mt5 = mt5_module
        self.deviation_points = deviation_points

        kwargs: dict[str, Any] = {}
        if path:
            kwargs["path"] = path
        if login:
            kwargs.update(login=int(login), password=password or "", server=server or "")

        if not self.mt5.initialize(**kwargs):
            error = self.mt5.last_error()
            self.mt5.shutdown()
            raise RuntimeError(f"Could not connect to MetaTrader 5: {error}")

    # --- market data -------------------------------------------------

    def candles(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        self._select(symbol)
        mt5_timeframe = getattr(self.mt5, f"TIMEFRAME_{normalize_timeframe(timeframe)}")

        # Start at position 1 to skip the candle that is still forming.
        rates = self.mt5.copy_rates_from_pos(symbol, mt5_timeframe, 1, count)

        if rates is None or len(rates) == 0:
            raise RuntimeError(f"MT5 returned no candles for {symbol}: {self.mt5.last_error()}")

        data = pd.DataFrame(rates)
        data["Datetime"] = pd.to_datetime(data["time"], unit="s", utc=True)
        data = data.rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "tick_volume": "Volume",
            "spread": "Spread",
        })

        columns = [c for c in ("Open", "High", "Low", "Close", "Volume", "Spread") if c in data]
        return data.set_index("Datetime")[columns].sort_index()

    def quote(self, symbol: str) -> Quote:
        self._select(symbol)
        tick = self.mt5.symbol_info_tick(symbol)

        if tick is None:
            raise RuntimeError(f"No price for {symbol}: {self.mt5.last_error()}")

        return Quote(bid=float(tick.bid), ask=float(tick.ask))

    def symbol_spec(self, symbol: str) -> SymbolSpec:
        info = self._select(symbol)

        return SymbolSpec(
            name=symbol,
            digits=int(info.digits),
            point=float(info.point),
            tick_size=float(info.trade_tick_size),
            tick_value=float(info.trade_tick_value),
            volume_min=float(info.volume_min),
            volume_max=float(info.volume_max),
            volume_step=float(info.volume_step),
            stops_level_points=int(info.trade_stops_level),
        )

    # --- account & positions -----------------------------------------

    def account(self) -> AccountInfo:
        info = self.mt5.account_info()
        terminal = self.mt5.terminal_info()

        if info is None:
            raise RuntimeError(f"Could not read the MT5 account: {self.mt5.last_error()}")

        return AccountInfo(
            login=int(info.login),
            balance=float(info.balance),
            equity=float(info.equity),
            currency=str(info.currency),
            is_demo=info.trade_mode == self.mt5.ACCOUNT_TRADE_MODE_DEMO,
            trade_allowed=bool(info.trade_allowed) and bool(terminal and terminal.trade_allowed),
        )

    def positions(self, symbol: str, magic: int) -> list[Position]:
        raw = self.mt5.positions_get(symbol=symbol) or ()

        return [
            Position(
                ticket=int(p.ticket),
                symbol=p.symbol,
                direction=1 if p.type == self.mt5.POSITION_TYPE_BUY else -1,
                volume=float(p.volume),
                open_price=float(p.price_open),
                stop_loss=float(p.sl),
                take_profit=float(p.tp),
                profit=float(p.profit),
                magic=int(p.magic),
            )
            for p in raw
            if int(p.magic) == magic
        ]

    def open_position(
        self,
        symbol: str,
        direction: int,
        volume: float,
        stop_loss: float,
        take_profit: float,
        magic: int,
        comment: str,
    ) -> OrderResult:
        info = self._select(symbol)
        quote = self.quote(symbol)

        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": self.mt5.ORDER_TYPE_BUY if direction > 0 else self.mt5.ORDER_TYPE_SELL,
            "price": quote.ask if direction > 0 else quote.bid,
            "sl": round(stop_loss, info.digits),
            "tp": round(take_profit, info.digits),
            "deviation": self.deviation_points,
            "magic": magic,
            "comment": comment[:31],  # MT5 limits comments to 31 characters
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self._filling_mode(info),
        }
        return self._send(request)

    def close_position(self, position: Position) -> OrderResult:
        info = self._select(position.symbol)
        quote = self.quote(position.symbol)
        closing_buy = position.direction > 0

        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": self.mt5.ORDER_TYPE_SELL if closing_buy else self.mt5.ORDER_TYPE_BUY,
            "position": position.ticket,
            "price": quote.bid if closing_buy else quote.ask,
            "deviation": self.deviation_points,
            "magic": position.magic,
            "comment": "close",
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self._filling_mode(info),
        }
        return self._send(request)

    def shutdown(self) -> None:
        self.mt5.shutdown()

    # --- internals ---------------------------------------------------

    def _select(self, symbol: str) -> Any:
        if not self.mt5.symbol_select(symbol, True):
            raise RuntimeError(
                f"Symbol {symbol} is not available at this broker "
                f"(some brokers add a suffix, e.g. EURUSD.m): {self.mt5.last_error()}")

        info = self.mt5.symbol_info(symbol)
        if info is None:
            raise RuntimeError(f"No symbol info for {symbol}: {self.mt5.last_error()}")
        return info

    def _filling_mode(self, info: Any) -> int:
        # symbol_info.filling_mode is a bit mask: 1 = FOK allowed, 2 = IOC allowed.
        if info.filling_mode & 1:
            return self.mt5.ORDER_FILLING_FOK
        if info.filling_mode & 2:
            return self.mt5.ORDER_FILLING_IOC
        return self.mt5.ORDER_FILLING_RETURN

    def _send(self, request: dict[str, Any]) -> OrderResult:
        check = self.mt5.order_check(request)

        if check is None:
            return OrderResult(False, f"order_check failed: {self.mt5.last_error()}")

        if check.retcode != 0:
            return OrderResult(False, f"order_check rejected ({check.retcode}): {check.comment}")

        result = self.mt5.order_send(request)

        if result is None:
            return OrderResult(False, f"order_send failed: {self.mt5.last_error()}")

        accepted = {self.mt5.TRADE_RETCODE_DONE, self.mt5.TRADE_RETCODE_PLACED}

        if result.retcode not in accepted:
            return OrderResult(False, f"order rejected ({result.retcode}): {result.comment}")

        return OrderResult(
            True,
            "filled",
            ticket=int(result.order),
            price=float(result.price),
        )
