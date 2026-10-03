## Currency Trading Platform

This is a personal project that I'm building to learn Python, programming, algorithms, and trading.

The program connects to **MetaTrader 5**, downloads historical Forex data, backtests several trading strategies with hundreds of parameter combinations, picks the best one for each currency pair, and can then trade it automatically. It has risk limits built in.

> **Important:** backtests show how a strategy *would have* done in the past. They don't guarantee future profits, and most retail Forex traders lose money. Always run this on a **demo account** first. By default the program refuses to trade a real-money account.

---

## Features

- Download candles from **MetaTrader 5**, Twelve Data, or a CSV file
- Indicators: EMA, RSI, ATR, Bollinger Bands, MACD, Donchian Channel
- **6 strategies**, each with a grid of settings to try:
  | Strategy | Idea |
  |---|---|
  | EMA Cross | Trend: fast EMA crosses the slow EMA |
  | RSI Reversion | Buy when RSI leaves oversold, sell when it leaves overbought (optional trend filter) |
  | Bollinger Reversion | Price closes back inside the bands |
  | Donchian Breakout | Close above/below the last N candles' range |
  | MACD Cross | MACD crosses its signal line (optional trend filter) |
  | Indicator Score | My original EMA + RSI + Bollinger scoring system |
- Fast backtester: ATR stop-loss and take-profit, spread costs, no look-ahead bias
- **Optimizer** that tests ~200 combinations per currency pair and approves a winner only if it also makes money on data it was never tuned on
- **Live trader** for MT5 with these safety rails:
  - Demo accounts only, unless you pass `--allow-real-account`
  - Fixed % risk per trade (default 1%), with lot size calculated from the stop distance
  - Skips a trade rather than over-risk when the account is too small for the minimum lot
  - Stops opening new trades for the day after a daily loss limit (default 3%)
  - Stops completely after a drawdown limit (default 15%)
  - Skips entries when the spread is unusually wide
  - `--dry-run` mode logs orders without sending them
  - Every order has a stop-loss and take-profit set at the broker, so trades stay protected if the program or PC stops
- Paper trading (simulated broker) that works without MT5, so it also runs on Mac and Linux
- 41 automated tests

---

## Quick start

### 1. Install (Windows, because the MetaTrader5 Python package only exists for Windows)

```bash
pip install -r requirements.txt
```

Install the MetaTrader 5 terminal from your broker, open a **demo account**, and in MT5 go to
*Tools → Options → Expert Advisors* and tick **Allow algorithmic trading**. Then click the **Algo Trading** button in the toolbar so it turns green.

Optional: copy `.env.example` to `.env` and fill in your MT5 login. If you leave it empty, the program uses the account already logged in to the terminal.

### 2. Find the best strategy for each pair

```bash
python app.py optimize --symbols EURUSD GBPUSD USDJPY AUDUSD --timeframe H1 --bars 20000
```

Example output:

```
EURUSD [APPROVED]  MACD Cross(fast=12, slow=26, signal=5, trend_filter=200) SL=2.0xATR TP=3.0xATR
  tuning data :  432 trades | return  151.94% | win  47.9% | PF  1.46 | max DD  11.0% | +0.222R/trade
  unseen data :  213 trades | return   61.11% | win  48.4% | PF  1.46 | max DD  10.8% | +0.231R/trade
```

(These numbers come from synthetic test data, not a real market.)

- `reports/leaderboard.csv` lists every combination that was tested
- `reports/best_strategies.json` holds the winners; the trader reads this file

### 3. Trade it

```bash
# Watch what it would do, without sending orders:
python app.py trade --dry-run

# Trade the demo account:
python app.py trade

# Re-run the optimizer every 7 days while trading:
python app.py trade --reoptimize-hours 168
```

The program checks for a newly closed candle every 15 seconds. Stop it with `Ctrl+C`; open trades keep their stop-loss and take-profit. Every order is logged to `reports/live_trades.csv` and `reports/trading.log`.

### Without MetaTrader 5 (Mac/Linux, or just practising)

```bash
export TWELVE_DATA_API_KEY=your_key
python app.py optimize --source twelvedata --symbols EURUSD --timeframe H1 --bars 5000
python app.py trade --broker paper
```

Or use a CSV file (for example one exported from MT5 with *View → Symbols → Bars → Export*):

```bash
python app.py optimize --source csv --symbols EURUSD --csv EURUSD=data/EURUSD_H1.csv
```

---

## How the "best strategy" is chosen

1. The candles are split: the first **70%** is used for tuning and the most recent **30%** is kept hidden.
2. Every strategy, setting, and stop/target combination is backtested on the first 70%.
3. Results are ranked by **SQN** (System Quality Number), which rewards *consistent* profits over one lucky trade. Results with fewer than 30 trades are ignored.
4. The top 10 are tested on the hidden 30%. A strategy is **approved** only if it's still profitable there, with a profit factor of at least 1.1, a drawdown under 25%, and at least 10 trades.
5. If nothing passes, nothing gets traded. That's on purpose.

Step 4 is what stops the program from fooling itself. To check it, I ran the optimizer on 5 sets of **pure random data**, where no strategy can have a real edge. Some combinations looked amazing on the tuning data (up to +81%), but all 5 were **rejected** because they lost money on the hidden part. There's an automated test for this too.

---

## Account size and risk

With 1% risk per trade, a 20-pip stop on EURUSD and the minimum 0.01 lot, a trade loses about $2 if the stop is hit. That means you need roughly **$200 or more** before the program can place trades at 1% risk. On a smaller account it logs a warning and skips the trade rather than risking more than you asked for.

---

## Project Structure

```
Currency-Trading-Platform
├── app.py            command line: optimize / trade / scan
├── market.py         Twelve Data download, CSV loader, timeframe helpers
├── indicators.py     EMA, RSI, ATR, Bollinger Bands, MACD, Donchian
├── strategy.py       the 6 strategies and their parameter grids
├── backtester.py     simulates trades with ATR stops, targets and spread
├── optimizer.py      tests every combination, checks on unseen data, saves the winner
├── broker.py         broker interface, lot-size calculation, paper broker
├── mt5_broker.py     MetaTrader 5 connection and order sending
├── live_trader.py    the trading loop and its safety rails
├── decision.py       BUY / SELL / WAIT
├── tests/            automated tests (run: python -m pytest)
└── reports/          leaderboards, chosen strategies, trade logs
```

Each file has its own responsibility, which makes the project easier to understand and update.

## What I'm Learning

This project has helped me learn about:

- Python
- Object-Oriented Programming (OOP) and abstract base classes
- Git & GitHub
- APIs (Twelve Data, MetaTrader 5)
- Pandas and NumPy (vectorised calculations)
- Technical indicators
- Backtesting, overfitting and out-of-sample testing
- Risk management and position sizing
- Automated testing with pytest
- Writing cleaner, more organized code

## Current Goals

- [x] Download market data
- [x] Calculate technical indicators
- [x] Create a rule-based trading strategy
- [x] Build a backtesting engine
- [x] Finish the strategy optimizer
- [x] Improve risk management
- [x] Add more technical indicators
- [x] Connect to MetaTrader 5 and trade automatically
- [ ] Run on a demo account for at least 1–3 months and compare live results with the backtest
- [ ] Walk-forward optimization (repeated rolling train/test windows)
- [ ] Learn machine learning and experiment with AI strategies

---

## Why I Started This Project

I'm not a Computer Science major, but I wanted to challenge myself by building something larger than the small programs I'd written before.

This project gives me a chance to practice programming while also learning about quantitative trading and software design.

I'm treating it as a long-term learning project, so I'll keep improving it as my skills grow.
