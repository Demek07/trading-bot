import os

import joblib
import pandas as pd
from loguru import logger

from src.analysis.mtf_features import FEATURES_MTF, merge_timeframes
from src.core.config import Config


def backtest_mtf(
    initial_balance: float = 1000.0,
    fee_pct: float = 0.002,
    slippage_pct: float = 0.001,
    stop_loss_pct: float = 0.03,
    take_profit_pct: float = 0.06,
    trailing_stop_pct: float = 0.02,
    min_hold_bars: int = 4,
    use_regime_filter: bool = True,   # НОВОЕ: фильтр режима рынка (1d SMA200)
    start_idx: int = None,
    buy_threshold: float = 0.58,
    sell_threshold: float = 0.58,
):
    path_1h = os.path.join(Config.DATA_DIR, f"{Config.SYMBOL.replace('/', '_')}_1h.csv")
    path_4h = os.path.join(Config.DATA_DIR, f"{Config.SYMBOL.replace('/', '_')}_4h.csv")
    path_1d = os.path.join(Config.DATA_DIR, f"{Config.SYMBOL.replace('/', '_')}_1d.csv")

    df = merge_timeframes(
        pd.read_csv(path_1h, parse_dates=["timestamp"]),
        pd.read_csv(path_4h, parse_dates=["timestamp"]),
    )

    # --- Фильтр режима рынка (1d SMA200) ---
    if use_regime_filter and os.path.exists(path_1d):
        df_1d = pd.read_csv(path_1d, parse_dates=["timestamp"])
        df_1d["sma200_1d"] = df_1d["close"].rolling(window=200).mean()
        df_1d["regime_bull"] = (df_1d["close"] > df_1d["sma200_1d"]).astype(int)
        df_1d = df_1d.dropna(subset=["sma200_1d"])

        df = pd.merge_asof(
            df.sort_values("timestamp"),
            df_1d[["timestamp", "regime_bull"]].sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )
        df["regime_bull"] = df["regime_bull"].fillna(1).astype(int)
        logger.info(
            f"1d regime filter loaded. Bull bars: "
            f"{df['regime_bull'].sum()}/{len(df)} "
            f"({df['regime_bull'].mean():.1%})"
        )
    else:
        df["regime_bull"] = 1
        if use_regime_filter:
            logger.warning("1d data not found — trading without regime filter.")
        else:
            logger.info("Regime filter disabled.")

    # --- Модель ---
    model_path = os.path.join(Config.MODEL_DIR, "model_mtf.joblib")
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}. Train MTF model first.")
        return
    model = joblib.load(model_path)
    logger.info("MTF model loaded.")

    if start_idx is None:
        start_idx = int(len(df) * 0.8)
        logger.info(f"OUT-OF-SAMPLE from index {start_idx}/{len(df)}")
    else:
        logger.info(f"Backtest from index {start_idx}/{len(df)}")

    logger.info(
        f"Params: SL={stop_loss_pct:.2%}, TP={take_profit_pct:.2%}, "
        f"trailing={trailing_stop_pct:.2%}, min_hold={min_hold_bars}, "
        f"regime_filter={use_regime_filter}, "
        f"buy>{buy_threshold}, sell>{sell_threshold}"
    )

    X = df[FEATURES_MTF]
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    balance = initial_balance
    position = 0.0
    entry_price = 0.0
    entry_bar = 0
    stop_loss = 0.0
    take_profit = 0.0
    highest_since_entry = 0.0
    trades = []
    closed_pnls = []

    for i in range(start_idx, len(df) - 1):
        row = df.iloc[i]
        current_price = row["close"]
        pred = predictions[i]
        proba = probabilities[i]
        trend_4h = row["trend_4h"]
        regime_bull = row["regime_bull"]

        if position > 0:
            highest_since_entry = max(highest_since_entry, current_price)

        # --- Проверка выхода: SL / TP / Trailing (работают всегда) ---
        if position > 0:
            exit_price = None
            exit_reason = None

            if current_price <= stop_loss:
                exit_price = stop_loss * (1 - slippage_pct)
                exit_reason = "SL"
            elif current_price >= take_profit:
                exit_price = take_profit * (1 - slippage_pct)
                exit_reason = "TP"
            else:
                trailing_level = highest_since_entry * (1 - trailing_stop_pct)
                if (
                    trailing_level > entry_price * (1 + fee_pct * 2)
                    and current_price <= trailing_level
                ):
                    exit_price = trailing_level * (1 - slippage_pct)
                    exit_reason = "TRAIL"

            if exit_price is not None:
                balance = position * exit_price * (1 - fee_pct)
                pnl = (exit_price - entry_price) * position
                closed_pnls.append(pnl)
                trades.append(
                    {"type": exit_reason, "price": exit_price, "idx": i, "pnl": pnl}
                )
                position = 0.0
                entry_price = 0.0
                highest_since_entry = 0.0
                continue

        # --- Генерация сигнала ---
        signal = "HOLD"

        # Фильтр режима: BUY только если 1d-тренд вверх
        if (
            pred == 1
            and proba[1] > buy_threshold
            and trend_4h == 1
            and regime_bull == 1
        ):
            signal = "BUY"
        elif pred == 0 and proba[0] > sell_threshold:
            signal = "SELL"

        # Минимальное удержание: не закрываем по SELL слишком рано
        if position > 0 and signal == "SELL" and (i - entry_bar) < min_hold_bars:
            signal = "HOLD"

        if signal == "BUY" and position == 0:
            exec_price = current_price * (1 + slippage_pct)
            position = (balance * (1 - fee_pct)) / exec_price
            balance = 0.0
            entry_price = exec_price
            entry_bar = i
            stop_loss = exec_price * (1 - stop_loss_pct)
            take_profit = exec_price * (1 + take_profit_pct)
            highest_since_entry = exec_price
            trades.append({"type": "BUY", "price": exec_price, "idx": i})

        elif signal == "SELL" and position > 0:
            exec_price = current_price * (1 - slippage_pct)
            balance = position * exec_price * (1 - fee_pct)
            pnl = (exec_price - entry_price) * position
            closed_pnls.append(pnl)
            trades.append(
                {"type": "SELL", "price": exec_price, "idx": i, "pnl": pnl}
            )
            position = 0.0
            entry_price = 0.0
            highest_since_entry = 0.0

    # --- Закрываем позицию в конце ---
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
        logger.info(
            f"Avg PnL per closed trade: ${sum(closed_pnls)/len(closed_pnls):.2f}"
        )
        logger.info(f"Best trade: ${max(closed_pnls):.2f}")
        logger.info(f"Worst trade: ${min(closed_pnls):.2f}")

        exit_types = {}
        for t in trades:
            if t["type"] in ("SL", "TP", "TRAIL", "SELL"):
                exit_types[t["type"]] = exit_types.get(t["type"], 0) + 1
        logger.info(f"Exit breakdown: {exit_types}")

    buy_hold = (final_price / df.iloc[start_idx]["close"] - 1) * 100
    logger.info(f"Buy & Hold: {buy_hold:.2f}%")

    if profit_pct > buy_hold:
        logger.success("Strategy BEAT Buy & Hold!")
    else:
        logger.warning("Strategy UNDERPERFORMED Buy & Hold.")


if __name__ == "__main__":
    # По умолчанию: OOS, реалистичные издержки, фильтр режима
    backtest_mtf()