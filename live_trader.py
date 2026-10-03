"""Runs the selected strategies on a broker, one closed candle at a time.

It follows exactly the same rules as the backtester so live results can be
compared with backtest results:

* act only once per CLOSED candle,
* stop-loss / take-profit at ATR multiples set when the order is placed,
* an opposite signal closes the trade and opens the reverse one.

Safety rails (all on by default):

* Refuses to trade a REAL-money account unless ``allow_real_account`` is set.
* Never trades a market whose strategy failed the out-of-sample test.
* Skips a trade if even the minimum lot size would risk too much.
* Stops opening trades for the day after ``max_daily_loss_pct``.
* Stops trading completely after ``max_drawdown_pct`` from the equity peak.
* Skips entries while the spread is unusually wide (news, rollover).
* ``dry_run`` logs what it would do without sending any order.
"""
from __future__ import annotations

import csv
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from broker import Broker, calculate_volume
from optimizer import Selection
from strategy import Strategy

logger = logging.getLogger("trader")


@dataclass(frozen=True)
class TradingSettings:
    risk_per_trade: float = 0.01
    max_daily_loss_pct: float = 3.0
    max_drawdown_pct: float = 15.0
    max_spread_pips: float = 3.0
    candles: int = 1000
    poll_seconds: int = 15
    magic: int = 271_003
    dry_run: bool = False
    allow_real_account: bool = False
    journal_path: str = "reports/live_trades.csv"


class TradingHalted(RuntimeError):
    pass


@dataclass
class _SymbolState:
    selection: Selection
    last_candle: object = None
    strategy: Strategy | None = None


@dataclass
class LiveTrader:
    broker: Broker
    selections: dict[str, Selection]
    settings: TradingSettings = field(default_factory=TradingSettings)
    reoptimize: Callable[[str], Selection] | None = None
    reoptimize_every_hours: float = 0.0

    def __post_init__(self) -> None:
        if not 0 < self.settings.risk_per_trade <= 0.05:
            raise ValueError("risk_per_trade must be between 0 and 5% (0.05).")

        self._states = {
            symbol: _SymbolState(selection)
            for symbol, selection in self.selections.items()
        }
        self._day: str | None = None
        self._day_start_equity = 0.0
        self._peak_equity = 0.0
        self._last_reoptimize = time.monotonic()

    # --- main loop ---------------------------------------------------

    def check_account(self) -> None:
        account = self.broker.account()

        logger.info(
            "Account %s | balance %.2f %s | %s",
            account.login, account.balance, account.currency,
            "DEMO" if account.is_demo else "REAL MONEY",
        )

        if not account.is_demo and not self.settings.allow_real_account:
            raise TradingHalted(
                "This is a REAL-money account. Practise on a demo account first. "
                "If you really mean it, restart with --allow-real-account."
            )

        if not account.trade_allowed and not self.settings.dry_run:
            raise TradingHalted(
                "Trading is disabled in the terminal/account. In MT5 enable "
                "'Algo Trading' (toolbar button) and try again."
            )

        tradable = [s for s, st in self._states.items() if st.selection.approved]
        if not tradable:
            raise TradingHalted(
                "No market has an approved strategy (none passed the out-of-sample "
                "test). Not trading is the right call here.")

        for symbol, state in self._states.items():
            if state.selection.approved:
                logger.info("%s: trading %s", symbol, state.selection.candidate.label)
            else:
                logger.warning("%s: no approved strategy - will not trade it.", symbol)

    def run_forever(self) -> None:
        self.check_account()
        logger.info("Trader started. Press Ctrl+C to stop (open trades keep their SL/TP).")

        try:
            while True:
                self.run_once()
                time.sleep(self.settings.poll_seconds)
        except KeyboardInterrupt:
            logger.info("Stopped by user.")
        finally:
            self.broker.shutdown()

    def run_once(self) -> None:
        self._maybe_reoptimize()

        for symbol in self._states:
            try:
                self.step(symbol)
            except TradingHalted:
                raise
            except Exception as error:  # keep the other symbols running
                logger.exception("%s: error during step: %s", symbol, error)

    # --- one symbol, one candle --------------------------------------

    def step(self, symbol: str) -> None:
        state = self._states[symbol]
        selection = state.selection

        if not selection.approved or selection.candidate is None:
            return

        # Keep one strategy object per symbol so models are not retrained
        # on every candle (rebuilt only when the selection changes).
        if state.strategy is None:
            state.strategy = selection.candidate.build()
        strategy = state.strategy

        count = max(self.settings.candles, strategy.history_needed)
        data = self.broker.candles(symbol, selection.timeframe, count)

        latest_time = data.index[-1]
        if latest_time == state.last_candle:
            return  # this candle was already handled
        state.last_candle = latest_time

        prepared = strategy.prepare(data)
        signal = strategy.latest_signal(prepared)
        atr = float(prepared["ATR"].iloc[-1])

        positions = self.broker.positions(symbol, self.settings.magic)

        # 1) Opposite signal: close (same as the backtester).
        for position in positions:
            if signal == -position.direction:
                self._close(position, "opposite signal")

        positions = self.broker.positions(symbol, self.settings.magic)

        if signal == 0 or positions:
            return

        # 2) Risk checks before opening anything new.
        if not self._risk_allows_new_trade():
            return

        self._open(symbol, signal, atr, selection, strategy.label)

    # --- helpers -----------------------------------------------------

    def _risk_allows_new_trade(self) -> bool:
        account = self.broker.account()
        today = datetime.now(timezone.utc).date().isoformat()

        if self._day != today:
            self._day = today
            self._day_start_equity = account.equity

        self._peak_equity = max(self._peak_equity, account.equity)

        drawdown = (1 - account.equity / self._peak_equity) * 100 if self._peak_equity else 0
        if drawdown >= self.settings.max_drawdown_pct:
            raise TradingHalted(
                f"Equity is {drawdown:.1f}% below its peak (limit "
                f"{self.settings.max_drawdown_pct}%). Trading stopped. Review the "
                f"strategy before restarting.")

        if self._day_start_equity <= 0 or account.equity <= 0:
            raise TradingHalted("Account equity is zero.")

        daily_loss = (1 - account.equity / self._day_start_equity) * 100
        if daily_loss >= self.settings.max_daily_loss_pct:
            logger.warning(
                "Daily loss %.2f%% reached the %.1f%% limit - no new trades today.",
                daily_loss, self.settings.max_daily_loss_pct)
            return False

        return True

    def _open(self, symbol, direction, atr, selection, label) -> None:
        spec = self.broker.symbol_spec(symbol)
        quote = self.broker.quote(symbol)
        account = self.broker.account()
        candidate = selection.candidate

        spread_pips = quote.spread / spec.pip_size
        if spread_pips > self.settings.max_spread_pips:
            logger.info("%s: spread %.1f pips is too wide - skipping.", symbol, spread_pips)
            return

        if not atr > 0:
            logger.info("%s: ATR not ready - skipping.", symbol)
            return

        entry = quote.ask if direction > 0 else quote.bid
        stop_distance = atr * candidate.stop_atr_multiple
        target_distance = atr * candidate.target_atr_multiple

        if min(stop_distance, target_distance) <= spec.min_stop_distance:
            logger.info("%s: stop is closer than the broker allows - skipping.", symbol)
            return

        stop_loss = entry - direction * stop_distance
        take_profit = entry + direction * target_distance

        risk_amount = account.equity * self.settings.risk_per_trade
        volume = calculate_volume(risk_amount, stop_distance, spec)
        side = "BUY" if direction > 0 else "SELL"

        if volume <= 0:
            logger.warning(
                "%s: %s skipped - risking %.2f %s on a %.1f-pip stop needs less than the "
                "minimum lot (%.2f). Deposit more or lower the timeframe/ATR multiple.",
                symbol, side, risk_amount, account.currency,
                stop_distance / spec.pip_size, spec.volume_min)
            return

        logger.info(
            "%s: %s %.2f lots @ %.5f  SL %.5f  TP %.5f  (risk %.2f %s, %s)",
            symbol, side, volume, entry, stop_loss, take_profit,
            risk_amount, account.currency, label)

        if self.settings.dry_run:
            self._journal(symbol, side, volume, entry, stop_loss, take_profit, "DRY RUN", label)
            return

        result = self.broker.open_position(
            symbol, direction, volume, stop_loss, take_profit,
            self.settings.magic, comment=label[:31])

        status = "OPENED" if result.ok else f"FAILED: {result.message}"
        (logger.info if result.ok else logger.error)("%s: %s", symbol, status)
        self._journal(symbol, side, volume, result.price or entry, stop_loss, take_profit,
                      status, label)

    def _close(self, position, reason: str) -> None:
        side = "CLOSE BUY" if position.direction > 0 else "CLOSE SELL"
        logger.info("%s: %s #%s (%s), floating P/L %.2f",
                    position.symbol, side, position.ticket, reason, position.profit)

        if self.settings.dry_run:
            self._journal(position.symbol, side, position.volume, position.open_price,
                          position.stop_loss, position.take_profit, "DRY RUN", reason)
            return

        result = self.broker.close_position(position)
        status = "CLOSED" if result.ok else f"FAILED: {result.message}"
        (logger.info if result.ok else logger.error)("%s: %s", position.symbol, status)
        self._journal(position.symbol, side, position.volume, result.price or 0.0,
                      position.stop_loss, position.take_profit, status, reason)

    def _maybe_reoptimize(self) -> None:
        if not self.reoptimize or self.reoptimize_every_hours <= 0:
            return

        if time.monotonic() - self._last_reoptimize < self.reoptimize_every_hours * 3600:
            return

        self._last_reoptimize = time.monotonic()

        for symbol, state in self._states.items():
            try:
                selection = self.reoptimize(symbol)
            except Exception as error:
                logger.exception("%s: re-optimisation failed, keeping old strategy: %s",
                                 symbol, error)
                continue

            state.selection = selection
            state.strategy = None
            if selection.approved:
                logger.info("%s: re-optimised -> %s", symbol, selection.candidate.label)
            else:
                logger.warning("%s: re-optimised -> no approved strategy, pausing it.", symbol)

    def _journal(self, symbol, action, volume, price, stop_loss, take_profit, status, note):
        path = Path(self.settings.journal_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not path.exists()

        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if new_file:
                writer.writerow(["time_utc", "symbol", "action", "volume", "price",
                                 "stop_loss", "take_profit", "status", "note"])
            writer.writerow([
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                symbol, action, volume, round(price, 6), round(stop_loss, 6),
                round(take_profit, 6), status, note,
            ])
