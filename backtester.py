from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from decision import Decision
from strategy import IndicatorStrategy


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


@dataclass(frozen=True)
class Trade:
    direction: Decision
    entry_time: object
    exit_time: object

    entry_price: float
    exit_price: float

    stop_price: float
    target_price: float

    position_size: float
    profit_loss: float
    return_pct: float

    exit_reason: str
    balance_after: float


@dataclass(frozen=True)
class BacktestResult:
    starting_balance: float
    ending_balance: float
    trades: tuple[Trade, ...]
    equity_curve: tuple[float, ...]

    @property
    def total_return_pct(self) -> float:
        return (self.ending_balance / self.starting_balance - 1) * 100

    @property
    def winning_trades(self) -> int:
        return sum(
            trade.profit_loss > 0
            for trade in self.trades)

    @property
    def losing_trades(self) -> int:
        return sum(
            trade.profit_loss < 0
            for trade in self.trades)

    @property
    def win_rate_pct(self) -> float:
        if not self.trades:
            return 0.0

        return (self.winning_trades / len(self.trades)) * 100

    @property
    def average_win(self) -> float:
        wins = [
            trade.profit_loss
            for trade in self.trades
            if trade.profit_loss > 0]

        return sum(wins) / len(wins) if wins else 0.0

    @property
    def average_loss(self) -> float:
        losses = [
            trade.profit_loss
            for trade in self.trades
            if trade.profit_loss < 0]

        return(sum(losses) / len(losses)
            if losses
            else 0.0 )

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(
            trade.profit_loss
            for trade in self.trades
            if trade.profit_loss > 0)

        gross_loss = abs(
            sum(
                trade.profit_loss
                for trade in self.trades
                if trade.profit_loss < 0 ))

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

            drawdown = ( peak - equity ) / peak

            maximum_drawdown = max( maximum_drawdown, drawdown, )

        return maximum_drawdown * 100

    def trades_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{      "direction": trade.direction.name,
                    "entry_time": trade.entry_time,
                    "exit_time": trade.exit_time,
                    "entry_price": trade.entry_price,
                    "exit_price": trade.exit_price,
                    "stop_price": trade.stop_price,
                    "target_price": trade.target_price,
                    "position_size": trade.position_size,
                    "profit_loss": trade.profit_loss,
                    "return_pct": trade.return_pct,
                    "exit_reason": trade.exit_reason,
                    "balance_after": trade.balance_after,}
                for trade in self.trades])


class Backtester: #Simulates one risk-managed position at a time

    def __init__(
        self,
        strategy: IndicatorStrategy,
        settings: BacktestSettings | None = None, ) -> None:
        self.strategy = strategy
        self.settings = settings or BacktestSettings()

        self._validate_settings()

    def run(self, data: pd.DataFrame) -> BacktestResult:
        self._validate_data(data)

        balance = self.settings.starting_balance
        trades: list[Trade] = []
        equity_curve = [balance]

        index = 0

        while index < len(data) - 1:
            history = data.iloc[: index + 1]

            try:
                analysis = self.strategy.analyze(history)
            except ValueError:
                index += 1
                continue

            direction = analysis.decision

            if direction not in (
                Decision.BUY,
                Decision.SELL, ):
                index += 1
                continue

            signal_row = data.iloc[index]
            atr = float(signal_row["ATR_14"])

            if pd.isna(atr) or atr <= 0:
                index += 1
                continue

            # Enter on the following candle to prevent
            # look-ahead bias.
            entry_index = index + 1
            entry_row = data.iloc[entry_index]

            raw_entry_price = float(entry_row["Open"])
            entry_time = data.index[entry_index]

            entry_price = self._apply_entry_spread(
                direction,
                raw_entry_price,)

            stop_distance = (atr * self.settings.stop_atr_multiple)

            target_distance = (atr * self.settings.target_atr_multiple)

            if direction is Decision.BUY:
                stop_price = entry_price - stop_distance
                target_price = entry_price + target_distance
            else:
                stop_price = entry_price + stop_distance
                target_price = entry_price - target_distance

            risk_amount = ( balance * self.settings.risk_per_trade)

            position_size = (risk_amount / stop_distance)

            trade, exit_index = self._simulate_trade(
                data=data,
                entry_index=entry_index,
                direction=direction,
                entry_time=entry_time,
                entry_price=entry_price,
                stop_price=stop_price,
                target_price=target_price,
                position_size=position_size,
                balance_before=balance, )

            balance = trade.balance_after
            trades.append(trade)
            equity_curve.append(balance)

            # Continue after the candle on which the trade exited.
            index = max(exit_index + 1, index + 1)

        return BacktestResult(
            starting_balance=self.settings.starting_balance,
            ending_balance=balance,
            trades=tuple(trades),
            equity_curve=tuple(equity_curve), )

    def _simulate_trade(
        self,
        data: pd.DataFrame,
        entry_index: int,
        direction: Decision,
        entry_time: object,
        entry_price: float,
        stop_price: float,
        target_price: float,
        position_size: float,
        balance_before: float, ) -> tuple[Trade, int]:
        for index in range(
            entry_index,
            len(data), ):
            candle = data.iloc[index]

            high = float(candle["High"])
            low = float(candle["Low"])

            if direction is Decision.BUY:
                stop_hit = low <= stop_price
                target_hit = high >= target_price
            else:
                stop_hit = high >= stop_price
                target_hit = low <= target_price

            # We cannot know which happened first within a
            # candle, so use the conservative assumption:
            # the stop-loss happened first.
            if stop_hit:
                return self._create_trade(
                    direction=direction,
                    entry_time=entry_time,
                    exit_time=data.index[index],
                    entry_price=entry_price,
                    raw_exit_price=stop_price,
                    stop_price=stop_price,
                    target_price=target_price,
                    position_size=position_size,
                    balance_before=balance_before,
                    exit_reason="STOP_LOSS", ), index

            if target_hit:
                return self._create_trade(
                    direction=direction,
                    entry_time=entry_time,
                    exit_time=data.index[index],
                    entry_price=entry_price,
                    raw_exit_price=target_price,
                    stop_price=stop_price,
                    target_price=target_price,
                    position_size=position_size,
                    balance_before=balance_before,
                    exit_reason="TAKE_PROFIT", ), index

            # Exit at the next candle's open when the
            # strategy gives an opposite signal.
            if index < len(data) - 1:
                history = data.iloc[: index + 1]

                try:
                    analysis = self.strategy.analyze(history)
                except ValueError:
                    continue

                opposite_signal = (
                    direction is Decision.BUY
                    and analysis.decision is Decision.SELL
                ) or (
                    direction is Decision.SELL
                    and analysis.decision is Decision.BUY)

                if opposite_signal:
                    next_row = data.iloc[index + 1]

                    return self._create_trade(
                        direction=direction,
                        entry_time=entry_time,
                        exit_time=data.index[index + 1],
                        entry_price=entry_price,
                        raw_exit_price=float(next_row["Open"]),
                        stop_price=stop_price,
                        target_price=target_price,
                        position_size=position_size,
                        balance_before=balance_before,
                        exit_reason="OPPOSITE_SIGNAL", ), index + 1

        # Close any remaining trade on the final candle.
        final_index = len(data) - 1
        final_row = data.iloc[final_index]

        return self._create_trade(
            direction=direction,
            entry_time=entry_time,
            exit_time=data.index[final_index],
            entry_price=entry_price,
            raw_exit_price=float(final_row["Close"]),
            stop_price=stop_price,
            target_price=target_price,
            position_size=position_size,
            balance_before=balance_before,
            exit_reason="END_OF_DATA", ), final_index

    def _create_trade(
        self,
        direction: Decision,
        entry_time: object,
        exit_time: object,
        entry_price: float,
        raw_exit_price: float,
        stop_price: float,
        target_price: float,
        position_size: float,
        balance_before: float,
        exit_reason: str, ) -> Trade:
        exit_price = self._apply_exit_spread(
            direction,
            raw_exit_price,)

        if direction is Decision.BUY:
            price_change = (exit_price - entry_price)
        else:
            price_change = (entry_price - exit_price)

        profit_loss = price_change * position_size

        balance_after = max(balance_before + profit_loss,0.0,)

        return_pct = (
            profit_loss / balance_before * 100
            if balance_before > 0
            else 0.0)

        return Trade(
            direction=direction,
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=entry_price,
            exit_price=exit_price,
            stop_price=stop_price,
            target_price=target_price,
            position_size=position_size,
            profit_loss=profit_loss,
            return_pct=return_pct,
            exit_reason=exit_reason,
            balance_after=balance_after,)

    def _apply_entry_spread(
        self,
        direction: Decision,
        price: float,) -> float:
        half_spread = self._spread_price() / 2

        if direction is Decision.BUY:
            return price + half_spread

        return price - half_spread

    def _apply_exit_spread(
        self,
        direction: Decision,
        price: float, ) -> float:
        half_spread = self._spread_price() / 2

        if direction is Decision.BUY:
            return price - half_spread

        return price + half_spread

    def _spread_price(self) -> float:
        return (self.settings.spread_pips * self.settings.pip_size)

    def _validate_data(
        self,
        data: pd.DataFrame,) -> None:
        required_columns = {
            "Open",
            "High",
            "Low",
            "Close",
            "ATR_14",}

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

        if self.settings.spread_pips < 0:
            raise ValueError("Spread cannot be negative.")