"""Finds the best strategy for a market — honestly.

How it works
------------
1. The candles are split in two: the first part (in-sample, 70% by
   default) and the most recent part (out-of-sample, 30%).
2. Every strategy, every parameter combination and every stop/target
   setting is backtested on the in-sample part and ranked.
3. Only the top candidates are then tested on the out-of-sample candles,
   which they were NOT tuned on. A candidate is approved only if it is
   still profitable there, by a margin that is unlikely to be luck
   (t-statistic >= 2 over at least 30 trades).

Step 3 matters: with hundreds of combinations, something will always look
great on the data it was tuned on just by luck ("overfitting"). Checking
on unseen data is the cheapest protection against that — but only if the
bar is high enough. A weak bar (e.g. "profit factor above 1.1") is passed
by luck surprisingly often when several finalists each get a try. If
nothing passes, the optimizer says so — and the live trader will not
trade that market.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from backtester import Backtester, BacktestResult, BacktestSettings
from strategy import STRATEGY_CLASSES, Strategy, build_strategy


@dataclass(frozen=True)
class OptimizerSettings:
    in_sample_ratio: float = 0.7

    # Ignore results based on too few trades — they are mostly luck.
    min_trades_in_sample: int = 30
    min_trades_out_of_sample: int = 30

    # How many of the best in-sample candidates get the out-of-sample test.
    # Every extra finalist is another chance for luck to pass the test, so
    # keep this small.
    finalists: int = 5

    stop_atr_multiples: tuple[float, ...] = (1.5, 2.0)
    target_atr_multiples: tuple[float, ...] = (1.5, 3.0)

    # Out-of-sample requirements for a strategy to be approved.
    min_out_of_sample_profit_factor: float = 1.1
    # t-statistic of the average R on unseen data. About 2 means "less
    # than a ~2.5% chance this profit is pure luck" for a single test.
    min_out_of_sample_t_stat: float = 2.0
    max_out_of_sample_drawdown_pct: float = 25.0


@dataclass(frozen=True)
class Metrics:
    trades: int
    return_pct: float
    win_rate_pct: float
    profit_factor: float
    max_drawdown_pct: float
    expectancy_r: float
    sqn: float
    t_stat: float = 0.0

    @classmethod
    def from_result(cls, result: BacktestResult) -> "Metrics":
        return cls(
            trades=result.trade_count,
            return_pct=round(result.total_return_pct, 3),
            win_rate_pct=round(result.win_rate_pct, 2),
            profit_factor=round(min(result.profit_factor, 999.0), 3),
            max_drawdown_pct=round(result.maximum_drawdown_pct, 3),
            expectancy_r=round(result.expectancy_r, 4),
            sqn=round(system_quality_number(result), 4),
            t_stat=round(t_statistic(result), 4),
        )


@dataclass(frozen=True)
class Candidate:
    strategy: str  # class name, e.g. "EmaCrossStrategy"
    parameters: dict[str, Any]
    stop_atr_multiple: float
    target_atr_multiple: float

    def build(self) -> Strategy:
        return build_strategy(self.strategy, self.parameters)

    @property
    def label(self) -> str:
        return (
            f"{self.build().label} "
            f"SL={self.stop_atr_multiple}xATR TP={self.target_atr_multiple}xATR"
        )


@dataclass
class Evaluation:
    candidate: Candidate
    in_sample: Metrics
    out_of_sample: Metrics | None = None
    approved: bool = False
    rejection_reason: str = ""


@dataclass
class Selection:
    """The winner for one symbol. Saved to JSON and read by the live trader."""

    symbol: str
    timeframe: str
    approved: bool
    candidate: Candidate | None
    in_sample: Metrics | None
    out_of_sample: Metrics | None
    candles: int
    data_start: str
    data_end: str
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Selection":
        data = dict(data)
        if data.get("candidate"):
            data["candidate"] = Candidate(**data["candidate"])
        for key in ("in_sample", "out_of_sample"):
            if data.get(key):
                data[key] = Metrics(**data[key])
        return cls(**data)


def t_statistic(result: BacktestResult) -> float:
    """mean(R) / std(R) * sqrt(trades): how sure we can be the average R
    is above zero. Unlike SQN the trade count is not capped."""
    r_values = [trade.r_multiple for trade in result.trades]

    if len(r_values) < 2:
        return 0.0

    mean = sum(r_values) / len(r_values)
    variance = sum((r - mean) ** 2 for r in r_values) / (len(r_values) - 1)

    if variance == 0:
        return 0.0

    return mean / math.sqrt(variance) * math.sqrt(len(r_values))


def system_quality_number(result: BacktestResult) -> float:
    """Van Tharp's SQN: mean(R) / std(R) * sqrt(trades).

    Rewards strategies that make money consistently, not ones that got
    lucky on a single huge trade.
    """
    r_values = [trade.r_multiple for trade in result.trades]

    if len(r_values) < 2:
        return 0.0

    mean = sum(r_values) / len(r_values)
    variance = sum((r - mean) ** 2 for r in r_values) / (len(r_values) - 1)
    std = math.sqrt(variance)

    if std == 0:
        return 0.0

    # Cap the trade count so that very active strategies are not favoured
    # only because they trade a lot.
    return mean / std * math.sqrt(min(len(r_values), 100))


class StrategyOptimizer:

    def __init__(
        self,
        base_settings: BacktestSettings | None = None,
        settings: OptimizerSettings | None = None,
        strategy_classes: Iterable[type[Strategy]] | None = None,
    ) -> None:
        self.base_settings = base_settings or BacktestSettings()
        self.settings = settings or OptimizerSettings()
        self.strategy_classes = list(strategy_classes or STRATEGY_CLASSES.values())

        if not 0.3 <= self.settings.in_sample_ratio <= 0.9:
            raise ValueError("In-sample ratio must be between 0.3 and 0.9.")

    def candidates(self) -> list[Candidate]:
        result = []

        for cls in self.strategy_classes:
            for parameters in cls.parameter_combinations():
                for stop in self.settings.stop_atr_multiples:
                    for target in self.settings.target_atr_multiples:
                        result.append(Candidate(cls.__name__, parameters, stop, target))

        return result

    def optimize(
        self,
        data: pd.DataFrame,
        symbol: str,
        timeframe: str,
        progress: bool = False,
    ) -> tuple[Selection, list[Evaluation]]:
        if len(data) < 500:
            raise ValueError(
                f"{symbol}: only {len(data)} candles. Use at least 500 (ideally 5,000+).")

        split = int(len(data) * self.settings.in_sample_ratio)
        evaluations: list[Evaluation] = []

        # Signals are calculated once per strategy/parameter set and reused
        # for every stop/target combination. Indicators only look backwards,
        # so computing them on all the data does not leak the future into
        # the in-sample period.
        prepared_cache: dict[tuple[str, str], tuple[pd.DataFrame, Any]] = {}

        candidates = self.candidates()

        for number, candidate in enumerate(candidates, start=1):
            key = (candidate.strategy, json.dumps(candidate.parameters, sort_keys=True))

            if key not in prepared_cache:
                strategy = candidate.build()
                prepared = strategy.prepare(data)
                prepared_cache[key] = (prepared, strategy.signals(prepared).to_numpy())

            prepared, signals = prepared_cache[key]
            backtester = self._backtester(candidate)

            in_sample = backtester.run_signals(prepared.iloc[:split], signals[:split])
            evaluations.append(Evaluation(candidate, Metrics.from_result(in_sample)))

            if progress and number % 50 == 0:
                print(f"  {symbol}: tested {number}/{len(candidates)} combinations...")

        ranked = sorted(
            (e for e in evaluations if e.in_sample.trades >= self.settings.min_trades_in_sample),
            key=lambda e: e.in_sample.sqn,
            reverse=True,
        )

        finalists = [e for e in ranked if e.in_sample.expectancy_r > 0][: self.settings.finalists]

        for evaluation in finalists:
            prepared, signals = prepared_cache[
                (evaluation.candidate.strategy,
                 json.dumps(evaluation.candidate.parameters, sort_keys=True))]
            backtester = self._backtester(evaluation.candidate)

            out_of_sample = backtester.run_signals(prepared.iloc[split:], signals[split:])
            evaluation.out_of_sample = Metrics.from_result(out_of_sample)
            evaluation.approved, evaluation.rejection_reason = self._judge(
                evaluation.out_of_sample)

        approved = sorted(
            (e for e in finalists if e.approved),
            key=lambda e: e.out_of_sample.sqn,
            reverse=True,
        )

        common = dict(
            symbol=symbol,
            timeframe=timeframe,
            candles=len(data),
            data_start=str(data.index[0]),
            data_end=str(data.index[-1]),
        )

        if approved:
            best = approved[0]
            selection = Selection(
                approved=True,
                candidate=best.candidate,
                in_sample=best.in_sample,
                out_of_sample=best.out_of_sample,
                **common,
            )
        else:
            best = finalists[0] if finalists else None
            selection = Selection(
                approved=False,
                candidate=best.candidate if best else None,
                in_sample=best.in_sample if best else None,
                out_of_sample=best.out_of_sample if best else None,
                note=(
                    "No strategy stayed profitable on unseen data. "
                    "The live trader will NOT trade this market."
                ),
                **common,
            )

        return selection, evaluations

    def _backtester(self, candidate: Candidate) -> Backtester:
        settings = BacktestSettings(
            **{
                **asdict(self.base_settings),
                "stop_atr_multiple": candidate.stop_atr_multiple,
                "target_atr_multiple": candidate.target_atr_multiple,
            }
        )
        # The strategy object is not needed for run_signals().
        return Backtester(strategy=None, settings=settings)

    def _judge(self, metrics: Metrics) -> tuple[bool, str]:
        if metrics.trades < self.settings.min_trades_out_of_sample:
            return False, f"only {metrics.trades} out-of-sample trades"
        if metrics.expectancy_r <= 0:
            return False, "lost money on unseen data"
        if metrics.t_stat < self.settings.min_out_of_sample_t_stat:
            return False, (
                f"profit on unseen data could be luck (t-stat {metrics.t_stat:.2f} < "
                f"{self.settings.min_out_of_sample_t_stat})")
        if metrics.profit_factor < self.settings.min_out_of_sample_profit_factor:
            return False, f"out-of-sample profit factor {metrics.profit_factor:.2f} too low"
        if metrics.max_drawdown_pct > self.settings.max_out_of_sample_drawdown_pct:
            return False, f"out-of-sample drawdown {metrics.max_drawdown_pct:.1f}% too deep"
        return True, ""


def leaderboard(evaluations: list[Evaluation], symbol: str) -> pd.DataFrame:
    rows = []

    for evaluation in evaluations:
        row: dict[str, Any] = {
            "symbol": symbol,
            "strategy": evaluation.candidate.build().label,
            "stop_atr": evaluation.candidate.stop_atr_multiple,
            "target_atr": evaluation.candidate.target_atr_multiple,
        }
        row.update({f"is_{k}": v for k, v in asdict(evaluation.in_sample).items()})

        if evaluation.out_of_sample:
            row.update({f"oos_{k}": v for k, v in asdict(evaluation.out_of_sample).items()})

        row["approved"] = evaluation.approved
        row["rejection_reason"] = evaluation.rejection_reason
        rows.append(row)

    table = pd.DataFrame(rows)
    return table.sort_values("is_sqn", ascending=False).reset_index(drop=True)


def save_selections(selections: dict[str, Selection], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = load_selections(path) if path.exists() else {}
    existing.update(selections)

    path.write_text(
        json.dumps({s: sel.to_dict() for s, sel in existing.items()}, indent=2),
        encoding="utf-8",
    )


def load_selections(path: str | Path) -> dict[str, Selection]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run 'python app.py optimize' first to pick strategies.")

    raw = json.loads(path.read_text(encoding="utf-8"))
    return {symbol: Selection.from_dict(data) for symbol, data in raw.items()}
