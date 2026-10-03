"""Currency Trading Platform — command line.

    python app.py optimize --source mt5 --symbols EURUSD GBPUSD USDJPY --timeframe H1
    python app.py trade    --broker mt5 --symbols EURUSD GBPUSD USDJPY
    python app.py scan     --source twelvedata --symbols EURUSD

Run ``python app.py <command> --help`` for every option.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

from backtester import Backtester, BacktestSettings, pip_size_for
from market import MarketDataProvider, MarketRequest, load_csv, normalize_timeframe
from optimizer import (
    OptimizerSettings,
    Selection,
    StrategyOptimizer,
    leaderboard,
    load_selections,
    save_selections,
)
from strategy import IndicatorStrategy

REPORTS = Path("reports")
SELECTIONS_FILE = REPORTS / "best_strategies.json"


# --- setup -----------------------------------------------------------------

def load_dotenv(path: str = ".env") -> None:
    """Reads KEY=VALUE lines from .env into the environment (if the file exists)."""
    env_file = Path(path)
    if not env_file.exists():
        return

    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def setup_logging() -> None:
    REPORTS.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(REPORTS / "trading.log", encoding="utf-8"),
        ],
    )


def connect_mt5():
    from mt5_broker import MT5Broker

    return MT5Broker(
        login=int(os.environ["MT5_LOGIN"]) if os.getenv("MT5_LOGIN") else None,
        password=os.getenv("MT5_PASSWORD"),
        server=os.getenv("MT5_SERVER"),
        path=os.getenv("MT5_PATH"),
    )


class DataLoader:  # Gets candles from MT5, Twelve Data or CSV files

    def __init__(self, source: str, csv_files: dict[str, str] | None = None) -> None:
        self.source = source
        self.csv_files = csv_files or {}
        self._mt5 = None
        self._twelve_data = None

    def candles(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        if self.source == "mt5":
            return self.mt5.candles(symbol, timeframe, count)

        if self.source == "twelvedata":
            if self._twelve_data is None:
                self._twelve_data = MarketDataProvider()
            return self._twelve_data.download(
                MarketRequest(symbol=symbol, interval=timeframe, output_size=min(count, 5000)))

        if self.source == "csv":
            if symbol not in self.csv_files:
                raise ValueError(f"No CSV given for {symbol}. Use --csv {symbol}=path/to/file.csv")
            return load_csv(self.csv_files[symbol]).tail(count)

        raise ValueError(f"Unknown data source '{self.source}'.")

    def spread_pips(self, symbol: str, data: pd.DataFrame, default: float) -> float:
        """Uses the broker's real historical spread when MT5 provides it."""
        if self.source != "mt5" or "Spread" not in data:
            return default

        spec = self.mt5.symbol_spec(symbol)
        median_points = float(data["Spread"].median())
        return max(median_points * spec.point / spec.pip_size, 0.1)

    @property
    def mt5(self):
        if self._mt5 is None:
            self._mt5 = connect_mt5()
        return self._mt5

    def close(self) -> None:
        if self._mt5 is not None:
            self._mt5.shutdown()


def parse_csv_args(values: list[str] | None) -> dict[str, str]:
    result = {}
    for value in values or []:
        if "=" not in value:
            raise SystemExit(f"--csv expects SYMBOL=path, got '{value}'")
        symbol, path = value.split("=", 1)
        result[symbol.upper()] = path
    return result


# --- optimize --------------------------------------------------------------

def optimize_symbol(
    loader: DataLoader,
    symbol: str,
    timeframe: str,
    bars: int,
    spread: float | None,
    verbose: bool = True,
) -> tuple[Selection, pd.DataFrame]:
    data = loader.candles(symbol, timeframe, bars)
    default_spread = spread if spread is not None else 1.5
    spread_pips = spread if spread is not None else loader.spread_pips(symbol, data, default_spread)

    base = BacktestSettings(
        starting_balance=10_000.0,
        risk_per_trade=0.01,
        spread_pips=spread_pips,
        pip_size=pip_size_for(symbol),
    )

    if verbose:
        print(f"\n{symbol} {timeframe}: {len(data)} candles "
              f"({data.index[0]} -> {data.index[-1]}), spread {spread_pips:.1f} pips")

    optimizer = StrategyOptimizer(base_settings=base, settings=OptimizerSettings())
    selection, evaluations = optimizer.optimize(data, symbol, timeframe, progress=verbose)
    return selection, leaderboard(evaluations, symbol)


def print_selection(selection: Selection) -> None:
    print("-" * 72)
    if selection.candidate is None:
        print(f"{selection.symbol}: no strategy produced enough profitable trades.")
        print(f"  {selection.note}")
        return

    verdict = "APPROVED" if selection.approved else "REJECTED"
    print(f"{selection.symbol} [{verdict}]  {selection.candidate.label}")

    for title, m in (("tuning data ", selection.in_sample), ("unseen data ", selection.out_of_sample)):
        if m is None:
            continue
        print(f"  {title}: {m.trades:4d} trades | return {m.return_pct:7.2f}% | "
              f"win {m.win_rate_pct:5.1f}% | PF {m.profit_factor:5.2f} | "
              f"max DD {m.max_drawdown_pct:5.1f}% | {m.expectancy_r:+.3f}R/trade")

    if selection.note:
        print(f"  {selection.note}")


def command_optimize(args: argparse.Namespace) -> int:
    timeframe = normalize_timeframe(args.timeframe)
    loader = DataLoader(args.source, parse_csv_args(args.csv))
    selections: dict[str, Selection] = {}
    tables = []

    try:
        for symbol in args.symbols:
            try:
                selection, table = optimize_symbol(loader, symbol, timeframe, args.bars, args.spread)
            except (RuntimeError, ValueError) as error:
                print(f"\n{symbol}: skipped - {error}")
                continue
            selections[symbol] = selection
            tables.append(table)
    finally:
        loader.close()

    if not selections:
        print("\nNothing was optimised.")
        return 1

    REPORTS.mkdir(exist_ok=True)
    pd.concat(tables).to_csv(REPORTS / "leaderboard.csv", index=False)
    save_selections(selections, SELECTIONS_FILE)

    print("\n" + "=" * 72)
    print("BEST STRATEGY PER MARKET")
    print("=" * 72)
    for selection in selections.values():
        print_selection(selection)

    print("\nFull results: reports/leaderboard.csv")
    print(f"Saved for the trader: {SELECTIONS_FILE}")
    print("\nReminder: past results do not guarantee future profits. "
          "Run on a demo account first.")
    return 0


# --- trade -----------------------------------------------------------------

def command_trade(args: argparse.Namespace) -> int:
    from broker import PaperBroker
    from live_trader import LiveTrader, TradingHalted, TradingSettings

    setup_logging()
    log = logging.getLogger("trader")

    try:
        all_selections = load_selections(SELECTIONS_FILE)
    except FileNotFoundError as error:
        log.error(str(error))
        return 1

    symbols = args.symbols or list(all_selections)
    missing = [s for s in symbols if s not in all_selections]
    if missing:
        log.error("No optimisation results for %s. Run 'python app.py optimize' first.",
                  ", ".join(missing))
        return 1

    selections = {s: all_selections[s] for s in symbols}

    if args.broker == "mt5":
        broker = connect_mt5()
        loader = DataLoader("mt5")
        loader._mt5 = broker
    else:
        loader = DataLoader(args.paper_source, parse_csv_args(args.csv))
        broker = PaperBroker(loader.candles, starting_balance=args.paper_balance)

    settings = TradingSettings(
        risk_per_trade=args.risk,
        max_daily_loss_pct=args.max_daily_loss,
        max_drawdown_pct=args.max_drawdown,
        max_spread_pips=args.max_spread,
        poll_seconds=args.poll_seconds,
        dry_run=args.dry_run,
        allow_real_account=args.allow_real_account,
    )

    def reoptimize(symbol: str) -> Selection:
        old = selections[symbol]
        selection, _ = optimize_symbol(loader, symbol, old.timeframe, args.bars, None, verbose=False)
        save_selections({symbol: selection}, SELECTIONS_FILE)
        return selection

    trader = LiveTrader(
        broker=broker,
        selections=selections,
        settings=settings,
        reoptimize=reoptimize,
        reoptimize_every_hours=args.reoptimize_hours,
    )

    try:
        trader.run_forever()
    except TradingHalted as error:
        log.error("TRADING HALTED: %s", error)
        return 2
    return 0


# --- scan (quick look, the original feature) --------------------------------

def command_scan(args: argparse.Namespace) -> int:
    loader = DataLoader(args.source, parse_csv_args(args.csv))
    timeframe = normalize_timeframe(args.timeframe)
    strategy = IndicatorStrategy()

    try:
        for symbol in args.symbols:
            data = strategy.prepare(loader.candles(symbol, timeframe, args.bars))
            analysis = strategy.analyze(data)
            result = Backtester(strategy, BacktestSettings(pip_size=pip_size_for(symbol))).run(
                data, prepared=True)

            print(f"\n{symbol} {timeframe} - {analysis.decision.name} (score {analysis.score})")
            for reason in analysis.reasons:
                print(f"  - {reason}")
            print(f"  Backtest on {len(data)} candles: {result.trade_count} trades, "
                  f"{result.total_return_pct:.2f}% return, win rate {result.win_rate_pct:.1f}%")
    finally:
        loader.close()
    return 0


# --- argument parsing --------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Currency Trading Platform for MetaTrader 5")
    commands = parser.add_subparsers(dest="command", required=True)

    def add_data_arguments(sub, default_bars: int) -> None:
        sub.add_argument("--source", choices=["mt5", "twelvedata", "csv"], default="mt5",
                         help="where candles come from (default: mt5)")
        sub.add_argument("--symbols", nargs="+", default=["EURUSD"], type=str.upper)
        sub.add_argument("--timeframe", default="H1", help="M1 M5 M15 M30 H1 H4 D1")
        sub.add_argument("--bars", type=int, default=default_bars)
        sub.add_argument("--csv", nargs="*", metavar="SYMBOL=PATH",
                         help="CSV file per symbol when --source csv")

    optimize = commands.add_parser("optimize", help="backtest all strategies and pick the best")
    add_data_arguments(optimize, default_bars=20_000)
    optimize.add_argument("--spread", type=float, default=None,
                          help="spread in pips (default: broker's real spread on MT5, else 1.5)")
    optimize.set_defaults(func=command_optimize)

    trade = commands.add_parser("trade", help="trade the approved strategies")
    trade.add_argument("--broker", choices=["mt5", "paper"], default="mt5")
    trade.add_argument("--symbols", nargs="*", type=str.upper,
                       help="default: every symbol in reports/best_strategies.json")
    trade.add_argument("--risk", type=float, default=0.01, help="fraction risked per trade")
    trade.add_argument("--max-daily-loss", type=float, default=3.0, help="percent")
    trade.add_argument("--max-drawdown", type=float, default=15.0, help="percent")
    trade.add_argument("--max-spread", type=float, default=3.0, help="pips")
    trade.add_argument("--poll-seconds", type=int, default=15)
    trade.add_argument("--dry-run", action="store_true", help="log orders but never send them")
    trade.add_argument("--allow-real-account", action="store_true",
                       help="permit trading a REAL-money account (default: demo only)")
    trade.add_argument("--reoptimize-hours", type=float, default=0.0,
                       help="re-run the optimizer every N hours (0 = never)")
    trade.add_argument("--bars", type=int, default=20_000, help="candles used when re-optimising")
    trade.add_argument("--paper-source", choices=["twelvedata", "csv"], default="twelvedata")
    trade.add_argument("--paper-balance", type=float, default=10_000.0)
    trade.add_argument("--csv", nargs="*", metavar="SYMBOL=PATH")
    trade.set_defaults(func=command_trade)

    scan = commands.add_parser("scan", help="explain the current indicator score")
    add_data_arguments(scan, default_bars=500)
    scan.set_defaults(func=command_scan)

    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
