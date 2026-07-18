import backtester
from indicators import IndicatorCalculator
from market import MarketDataProvider, MarketRequest
from strategy import IndicatorStrategy
from backtester import Backtester, BacktestSettings
import strategy
import os

def main() -> None:
    print("=" * 60)
    print("              TradingAI Market Scanner")
    print("=" * 60)

    request = MarketRequest(
    symbol="EUR/USD",
    interval="15min",
    output_size=500,
    timezone="UTC",)

    provider = MarketDataProvider()
    calculator = IndicatorCalculator()
    strategy = IndicatorStrategy()

    backtest_settings = BacktestSettings(
    starting_balance=100.0,
    risk_per_trade=0.01,
    stop_atr_multiple=1.5,
    target_atr_multiple=3.0,
    spread_pips=1.0,)

    backtester = Backtester(strategy=strategy, settings=backtest_settings,)

    try:
        market_data = provider.download(request)

        market_data = calculator.add_ema(
            market_data,
            period=10,)

        market_data = calculator.add_ema(
            market_data,
            period=30,)
        
        market_data = calculator.add_rsi(
            market_data,
            period=14,)

        market_data = calculator.add_atr(
            market_data,
            period=14,)

        market_data = calculator.add_bollinger_bands(
            market_data,
            period=20,
            standard_deviations=2.0,)
        
        analysis = strategy.analyze(market_data)

        backtest_result = backtester.run(market_data)

    except (RuntimeError, ValueError, TypeError, ) as error:
        print(f"\nProgram failed: {error}")
        return

    print(f"\nSymbol: {request.symbol}")
    print(f"Candles downloaded: {len(market_data)}")

    columns_to_display = [
        "Close",
        "EMA_10",
        "EMA_30",
        "RSI_14",
        "ATR_14",
        "BB_LOWER_20",
        "BB_MIDDLE_20",
        "BB_UPPER_20",]

    print("\nLatest five candles:")
    print(market_data[columns_to_display].tail())

    print("\nStrategy analysis:")
    print(f"Score: {analysis.score}")
    print(f"Decision: {analysis.decision.name}")

    print("\nReasons:")
    for reason in analysis.reasons:
        print(f"- {reason}")

    print("\n" + "=" * 51)
    print(".              BACKTEST REPORT")
    print("=" * 51)

    print(
    f"Starting balance:  "
    f"${backtest_result.starting_balance:.2f}")
    print(
    f"Ending balance:    "
    f"${backtest_result.ending_balance:.2f}")
    print(
    f"Net return:        "
    f"{backtest_result.total_return_pct:.2f}%")
    print(
    f"Total trades:      "
    f"{len(backtest_result.trades)}")
    print(
    f"Winning trades:    "
    f"{backtest_result.winning_trades}")
    print(
    f"Losing trades:     "
    f"{backtest_result.losing_trades}")
    print(
    f"Win rate:          "
    f"{backtest_result.win_rate_pct:.2f}%")
    print(
    f"Average win:       "
    f"${backtest_result.average_win:.2f}")
    print(
    f"Average loss:      "
    f"${backtest_result.average_loss:.2f}")

    profit_factor = backtest_result.profit_factor

    if profit_factor == float("inf"):
        profit_factor_text = "Infinity"
    else:
        profit_factor_text = f"{profit_factor:.2f}"

    print(
    f"Profit factor:     "
    f"{profit_factor_text}")
    print(
    f"Maximum drawdown:  "
    f"{backtest_result.maximum_drawdown_pct:.2f}%")

    trade_log = backtest_result.trades_dataframe()

    os.makedirs("reports", exist_ok=True)

    trade_log.to_csv(
    "reports/trade_log.csv",
    index=False)

    print(
    "\nTrade log saved to "
    "reports/trade_log.csv")

if __name__ == "__main__":
    main()