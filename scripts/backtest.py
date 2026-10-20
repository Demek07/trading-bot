import os
from collections import Counter

import pandas as pd
from loguru import logger

from src.analysis.strategy import FEATURES, MLStrategy
from src.core.config import Config


def backtest(
    initial_balance: float = 1000.0,
    fee_pct: float = 0.002,        # 0.2% комиссия (консервативно)
    slippage_pct: float = 0.001,   # 0.1% slippage
    start_idx: int = None,         # None → 80% (out-of-sample)
    buy_threshold: float = 0.60,
    sell_threshold: float = 0.60,
    use_trend_filter: bool = True,
):
    filepath = os.path.join(
        Config.DATA_DIR, f"{Config.SYMBOL.replace('/', '_')}_{Config.TIMEFRAME}.csv"
    )
    if not os.path.exists(filepath):
        logger.error(f"Data file not found: {filepath}")
        return

    df = pd.read_csv(filepath, parse_dates=["timestamp"])

    strategy = MLStrategy()
    if not strategy.load_model():
        return

    df = strategy._calculate_indicators(df)
    logger.info(f"Indicators computed. Rows after dropna: {len(df)}")

    if len(df) < 100:
        logger.error("Not enough data after indicator calculation.")
        return

    if start_idx is None:
        start_idx = int(len(df) * 0.8)
        logger.info(f"Running OUT-OF-SAMPLE backtest from index {start_idx}/{len(df)}")
    else:
        logger.info(f"Running backtest from index {start_idx}/{len(df)}")

    logger.info(
        f"Thresholds: BUY>{buy_threshold}, SELL>{sell_threshold}, "
        f"trend_filter={use_trend_filter}"
    )

    # Батчевый инференс
    X = df[FEATURES]
    predictions = strategy.model.predict(X)
    probabilities = strategy.model.predict_proba(X)

    balance = initial_balance
    position = 0.0
    entry_price = 0.0
    trades = []
    closed_pnls = []

    for i in range(start_idx, len(df) - 1):
        pred = predictions[i]
        proba = probabilities[i]
        current_price = df.iloc[i]["close"]

        trend_up = df.iloc[i]["close"] > df.iloc[i]["sma_trend"]
        trend_down = df.iloc[i]["close"] < df.iloc[i]["sma_trend"]

        signal = "HOLD"
        if pred == 1 and proba[1] > buy_threshold:
            if not use_trend_filter or trend_up:
                signal = "BUY"
        elif pred == 0 and proba[0] > sell_threshold:
            if not use_trend_filter or trend_down:
                signal = "SELL"

        if signal == "BUY" and position == 0:
            exec_price = current_price * (1 + slippage_pct)
            position = (balance * (1 - fee_pct)) / exec_price
            balance = 0.0
            entry_price = exec_price
            trades.append({"type": "BUY", "price": exec_price, "idx": i,
                           "time": df.iloc[i]["timestamp"]})

        elif signal == "SELL" and position > 0:
            exec_price = current_price * (1 - slippage_pct)
            balance = position * exec_price * (1 - fee_pct)
            pnl = (exec_price - entry_price) * position
            closed_pnls.append(pnl)
            trades.append({"type": "SELL", "price": exec_price, "idx": i,
                           "time": df.iloc[i]["timestamp"]})
            position = 0.0
            entry_price = 0.0

    # Закрываем позицию в конце
    final_price = df.iloc[-1]["close"]
    if position > 0:
        exec_price = final_price * (1 - slippage_pct)
        balance = position * exec_price * (1 - fee_pct)
        closed_pnls.append((exec_price - entry_price) * position)
        position = 0.0

    final_equity = balance + position * final_price
    profit_pct = (final_equity - initial_balance) / initial_balance * 100

    logger.info("=" * 50)
    logger.info(f"Initial: ${initial_balance:.2f}")
    logger.info(f"Final Equity: ${final_equity:.2f}")
    logger.info(f"Profit: {profit_pct:.2f}%")
    logger.info(f"Total Trades: {len(trades)} (closed: {len(closed_pnls)})")

    if closed_pnls:
        wins = [p for p in closed_pnls if p > 0]
        losses = [p for p in closed_pnls if p <= 0]
        logger.info(
            f"Win rate: {len(wins)/len(closed_pnls)*100:.1f}% "
            f"({len(wins)} wins / {len(losses)} losses)"
        )
        logger.info(f"Avg PnL per closed trade: ${sum(closed_pnls)/len(closed_pnls):.2f}")
        logger.info(f"Best trade: ${max(closed_pnls):.2f}")
        logger.info(f"Worst trade: ${min(closed_pnls):.2f}")

    if trades:
        trade_days = Counter(pd.to_datetime(t["time"]).date() for t in trades)
        logger.info(f"Active trading days: {len(trade_days)}")
        logger.info(
            f"Avg trades/active day: {len(trades) / max(1, len(trade_days)):.1f}"
        )
        logger.info(f"Top-3 busiest days: {trade_days.most_common(3)}")

    buy_hold = (final_price / df.iloc[start_idx]["close"] - 1) * 100
    logger.info(f"Buy & Hold: {buy_hold:.2f}%")

    if profit_pct > buy_hold:
        logger.success("Strategy BEAT Buy & Hold!")
    else:
        logger.warning("Strategy UNDERPERFORMED Buy & Hold.")


if __name__ == "__main__":
    # Out-of-sample, строгие пороги, реалистичные издержки, фильтр тренда
    backtest()