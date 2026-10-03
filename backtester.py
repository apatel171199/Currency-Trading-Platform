from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from decision import Decision
from strategy import BUY, SELL, Strategy


def pip_size_for(symbol: str) -> float:
    """JPY pairs are quoted with 2/3 decimals, most other pairs with 4/5."""
    return 0.01 if "JPY" in symbol.upper() else 0.0001


@dataclass(frozen=True)
class BacktestSettings:
    starting_balance: float = 100.0

    # Risk 1% of the current balance per trade.
    risk_per_trade: float = 0.01

    # Stop and target are based on ATR.
    stop_atr_multiple: float = 1.5
    target_atr_multiple: float = 3.0

    # Approximate EUR/USD spread.
    spread_pips: float = 1.0
    pip_size: float = 0.0001

    # Round-trip commission, expressed in pips (0 for most "standard" accounts).
    commission_pips: float = 0.0


@dataclass(frozen=True)
class Trade:
    direction: Decision
    entry_time: object
    exit_time: object

    entry_price: float
    exit_price: float

    stop_price: float
    target_price: float

    # How many "R" (multiples of the amount risked) the trade made or lost.
    r_multiple: float
    profit_loss: float
    return_pct: float

    exit_reason: str
    balance_after: float


@dataclass(frozen=True)
class BacktestResult:
    starting_balance: float
    ending_balance: float
    trades: tuple[Trade, ...]
    equity_curve: tuple[float, ...] = field(repr=False)

    @property
    def total_return_pct(self) -> float:
        return (self.ending_balance / self.starting_balance - 1) * 100

    @property
    def trade_count(self) -> int:
        return len(self.trades)

    @property
    def winning_trades(self) -> int:
        return sum(trade.profit_loss > 0 for trade in self.trades)

    @property
    def losing_trades(self) -> int:
        return sum(trade.profit_loss < 0 for trade in self.trades)

    @property
    def win_rate_pct(self) -> float:
        if not self.trades:
            return 0.0

        return (self.winning_trades / len(self.trades)) * 100

    @property
    def average_win(self) -> float:
        wins = [trade.profit_loss for trade in self.trades if trade.profit_loss > 0]
        return sum(wins) / len(wins) if wins else 0.0

    @property
    def average_loss(self) -> float:
        losses = [trade.profit_loss for trade in self.trades if trade.profit_loss < 0]
        return sum(losses) / len(losses) if losses else 0.0

    @property
    def expectancy_r(self) -> float:
        """Average R per trade. Above 0 means the edge paid for the costs."""
        if not self.trades:
            return 0.0

        return sum(trade.r_multiple for trade in self.trades) / len(self.trades)

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(t.profit_loss for t in self.trades if t.profit_loss > 0)
        gross_loss = abs(sum(t.profit_loss for t in self.trades if t.profit_loss < 0))

        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

    @property
    def maximum_drawdown_pct(self) -> float:
        if not self.equity_curve:
            return 0.0

        peak = self.equity_curve[0]
        maximum_drawdown = 0.0

        for equity in self.equity_curve:
            peak = max(peak, equity)

            if peak > 0:
                maximum_drawdown = max(maximum_drawdown, (peak - equity) / peak)

        return maximum_drawdown * 100

    def trades_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "direction": trade.direction.name,
                    "entry_time": trade.entry_time,
                    "exit_time": trade.exit_time,
                    "entry_price": trade.entry_price,
                    "exit_price": trade.exit_price,
                    "stop_price": trade.stop_price,
                    "target_price": trade.target_price,
                    "r_multiple": trade.r_multiple,
                    "profit_loss": trade.profit_loss,
                    "return_pct": trade.return_pct,
                    "exit_reason": trade.exit_reason,
                    "balance_after": trade.balance_after,
                }
                for trade in self.trades
            ]
        )


class Backtester:  # Simulates one risk-managed position at a time
    """Rules (identical to what the live trader does):

    * A signal on candle i is acted on at the OPEN of candle i + 1.
    * Stop-loss and take-profit are placed ATR * multiple away from entry.
    * If the stop and the target are both touched inside one candle we
      assume the stop happened first (the pessimistic choice).
    * An opposite signal closes the trade at the next open and opens a
      trade in the new direction ("stop and reverse").
    * Profit is measured in R: losing the full stop distance = -1R,
      which is ``risk_per_trade`` of the balance. This keeps the maths
      correct for every currency pair, including JPY pairs.
    """

    def __init__(
        self,
        strategy: Strategy | None,
        settings: BacktestSettings | None = None,
    ) -> None:
        self.strategy = strategy
        self.settings = settings or BacktestSettings()

        self._validate_settings()

    def run(self, data: pd.DataFrame, prepared: bool = False) -> BacktestResult:
        """Backtest on raw OHLC data (or on data already passed through
        ``strategy.prepare`` when ``prepared=True``)."""
        self._validate_data(data)

        if self.strategy is None:
            raise ValueError("Backtester.run() needs a strategy.")

        if not prepared:
            data = self.strategy.prepare(data)

        signals = self.strategy.signals(data).to_numpy()
        return self.run_signals(data, signals)

    def run_signals(self, data: pd.DataFrame, signals: np.ndarray) -> BacktestResult:
        """Fast path used by the optimizer: signals are pre-computed."""
        opens = data["Open"].to_numpy(dtype=float)
        highs = data["High"].to_numpy(dtype=float)
        lows = data["Low"].to_numpy(dtype=float)
        closes = data["Close"].to_numpy(dtype=float)
        atrs = data["ATR"].to_numpy(dtype=float)
        times = data.index

        settings = self.settings
        half_cost = (settings.spread_pips + settings.commission_pips) * settings.pip_size / 2

        balance = settings.starting_balance
        trades: list[Trade] = []
        equity_curve = [balance]

        candle_count = len(data)
        signal_positions = np.flatnonzero(signals[:-1])  # last candle has no next open

        # Next candle index on which a new signal may be acted upon.
        index = 0
        pointer = 0
        pending_reverse: int | None = None

        while True:
            if pending_reverse is not None:
                signal_index = pending_reverse
                pending_reverse = None
            else:
                while pointer < len(signal_positions) and signal_positions[pointer] < index:
                    pointer += 1

                if pointer >= len(signal_positions):
                    break

                signal_index = int(signal_positions[pointer])
                pointer += 1

            direction = int(signals[signal_index])
            atr = atrs[signal_index]

            if not np.isfinite(atr) or atr <= 0 or balance <= 0:
                index = signal_index + 1
                continue

            entry_index = signal_index + 1
            entry_price = opens[entry_index] + direction * half_cost

            stop_distance = atr * settings.stop_atr_multiple
            stop_price = entry_price - direction * stop_distance
            target_price = entry_price + direction * atr * settings.target_atr_multiple

            exit_index, raw_exit_price, reason = self._find_exit(
                signals, highs, lows, opens, closes,
                entry_index, direction, stop_price, target_price,
            )

            exit_price = raw_exit_price - direction * half_cost
            r_multiple = direction * (exit_price - entry_price) / stop_distance

            risk_amount = balance * settings.risk_per_trade
            profit_loss = r_multiple * risk_amount
            balance_before = balance
            balance = max(balance + profit_loss, 0.0)

            trades.append(
                Trade(
                    direction=Decision.BUY if direction == BUY else Decision.SELL,
                    entry_time=times[entry_index],
                    exit_time=times[exit_index],
                    entry_price=entry_price,
                    exit_price=exit_price,
                    stop_price=stop_price,
                    target_price=target_price,
                    r_multiple=r_multiple,
                    profit_loss=profit_loss,
                    return_pct=profit_loss / balance_before * 100,
                    exit_reason=reason,
                    balance_after=balance,
                )
            )
            equity_curve.append(balance)

            if reason == "OPPOSITE_SIGNAL":
                # The opposite signal was on exit_index - 1; it opens the
                # reverse trade at the same open we exited on.
                pending_reverse = exit_index - 1
            else:
                # A signal on the exit candle itself is fine: it is acted
                # on at the next open, after this trade is already closed.
                index = exit_index

        return BacktestResult(
            starting_balance=settings.starting_balance,
            ending_balance=balance,
            trades=tuple(trades),
            equity_curve=tuple(equity_curve),
        )

    @staticmethod
    def _find_exit(
        signals: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        opens: np.ndarray,
        closes: np.ndarray,
        entry_index: int,
        direction: int,
        stop_price: float,
        target_price: float,
    ) -> tuple[int, float, str]:
        last = len(opens) - 1

        for index in range(entry_index, last + 1):
            if direction == BUY:
                stop_hit = lows[index] <= stop_price
                target_hit = highs[index] >= target_price
            else:
                stop_hit = highs[index] >= stop_price
                target_hit = lows[index] <= target_price

            if stop_hit:
                # A gap through the stop fills at the (worse) open price.
                gap_price = opens[index]
                gapped = (gap_price < stop_price) if direction == BUY else (gap_price > stop_price)
                return index, gap_price if gapped else stop_price, "STOP_LOSS"

            if target_hit:
                return index, target_price, "TAKE_PROFIT"

            if index < last and signals[index] == -direction:
                return index + 1, opens[index + 1], "OPPOSITE_SIGNAL"

        return last, closes[last], "END_OF_DATA"

    def _validate_data(self, data: pd.DataFrame) -> None:
        required_columns = {"Open", "High", "Low", "Close"}

        if data.empty:
            raise ValueError("Cannot backtest empty market data.")

        missing = required_columns.difference(data.columns)

        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"Backtest data is missing columns: {names}")

    def _validate_settings(self) -> None:
        if self.settings.starting_balance <= 0:
            raise ValueError("Starting balance must be positive.")

        if not 0 < self.settings.risk_per_trade <= 1:
            raise ValueError("Risk per trade must be between 0 and 1.")

        if self.settings.stop_atr_multiple <= 0:
            raise ValueError("Stop ATR multiple must be positive.")

        if self.settings.target_atr_multiple <= 0:
            raise ValueError("Target ATR multiple must be positive.")

        if self.settings.spread_pips < 0 or self.settings.commission_pips < 0:
            raise ValueError("Spread and commission cannot be negative.")
