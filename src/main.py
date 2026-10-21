import asyncio
import os
import sys
import warnings
from pathlib import Path

import ccxt.async_support as ccxt
import joblib
import pandas as pd
from loguru import logger

from src.analysis.mtf_features import FEATURES_MTF, merge_timeframes
from src.core.config import Config
from src.database.models import SessionLocal, Trade, init_db
from src.risk.manager import RiskManager

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ============================================================
# НАСТРОЙКА ЛОГИРОВАНИЯ
# ============================================================
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Убираем дефолтный sink (stderr), чтобы контролировать формат
logger.remove()

# 1. Консоль — только INFO и выше, компактный формат
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
           "<level>{level: <8}</level> | <level>{message}</level>",
    colorize=True,
)

# 2. Основной файл — всё (DEBUG+), ротация по дням, храним 30 дней
logger.add(
    LOG_DIR / "bot_{time:YYYY-MM-DD}.log",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
           "{name}:{function}:{line} | {message}",
    rotation="00:00",       # новый файл в полночь
    retention="30 days",    # храним 30 дней
    compression="zip",      # старые сжимаем в zip
    encoding="utf-8",
    enqueue=True,           # безопасно для async
)

# 3. Файл только для сделок — для быстрого анализа PnL
logger.add(
    LOG_DIR / "trades_{time:YYYY-MM-DD}.log",
    level="SUCCESS",
    format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
    rotation="00:00",
    retention="90 days",    # сделки храним дольше
    compression="zip",
    encoding="utf-8",
    filter=lambda record: "[TRADE]" in record["message"],
    enqueue=True,
)

# ============================================================
# РЕЖИМ ТЕСТИРОВАНИЯ
# ============================================================
TEST_MODE = False  # A — строгие условия (реальная торговля)


async def load_regime_filter(exchange, symbol: str) -> pd.DataFrame:
    try:
        ohlcv = await exchange.fetch_ohlcv(symbol, "1d", limit=250)
        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms").astype(
            "datetime64[ns]"
        )
        df["sma200_1d"] = df["close"].rolling(window=200).mean()
        df["regime_bull"] = (df["close"] > df["sma200_1d"]).astype(int)
        return df.dropna(subset=["sma200_1d"])
    except Exception as e:
        logger.error(f"Failed to load 1d regime: {e}")
        return pd.DataFrame()


async def main():
    init_db()
    logger.info(f"Starting Bot (MTF + regime). DRY_RUN={Config.DRY_RUN}")
    logger.info(
        f"MODE: {'B — TEST (loose thresholds)' if TEST_MODE else 'A — STRICT'}"
    )
    logger.info(f"Logs dir: {LOG_DIR.resolve()}")

    exchange = getattr(ccxt, Config.EXCHANGE_ID)({"enableRateLimit": True})

    model_path = os.path.join(Config.MODEL_DIR, "model_mtf.joblib")
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        return
    model = joblib.load(model_path)
    logger.info("MTF model loaded.")

    risk_manager = RiskManager(balance=1000)
    position = 0.0
    entry_price = 0.0

    last_regime_update = 0
    regime_bull = 1

    while True:
        try:
            now = asyncio.get_event_loop().time()
            if now - last_regime_update > 3600:
                df_1d = await load_regime_filter(exchange, Config.SYMBOL)
                if not df_1d.empty:
                    regime_bull = int(df_1d.iloc[-1]["regime_bull"])
                    logger.info(f"Regime (1d): {'BULL' if regime_bull else 'BEAR'}")
                else:
                    logger.warning("Regime data unavailable, defaulting to BULL")
                last_regime_update = now

            ohlcv_1h = await exchange.fetch_ohlcv(Config.SYMBOL, "1h", limit=300)
            ohlcv_4h = await exchange.fetch_ohlcv(Config.SYMBOL, "4h", limit=250)

            df_1h = pd.DataFrame(
                ohlcv_1h,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            df_4h = pd.DataFrame(
                ohlcv_4h,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            df_1h["timestamp"] = pd.to_datetime(df_1h["timestamp"], unit="ms").astype(
                "datetime64[ns]"
            )
            df_4h["timestamp"] = pd.to_datetime(df_4h["timestamp"], unit="ms").astype(
                "datetime64[ns]"
            )

            df = merge_timeframes(df_1h, df_4h)
            if len(df) < 1:
                logger.warning("Not enough data after MTF merge.")
                await asyncio.sleep(60)
                continue

            last = df.iloc[-1]
            X = last[FEATURES_MTF].to_frame().T
            pred = model.predict(X)[0]
            proba = model.predict_proba(X)[0]
            price = last["close"]

            # ---- Генерация сигнала ----
            signal = "HOLD"

            if TEST_MODE:
                if pred == 1 and proba[1] > 0.50 and regime_bull == 1:
                    signal = "BUY"
                elif pred == 0 and proba[0] > 0.50:
                    signal = "SELL"
            else:
                if (
                    pred == 1
                    and proba[1] > 0.58
                    and last["trend_4h"] == 1
                    and regime_bull == 1
                ):
                    signal = "BUY"
                elif pred == 0 and proba[0] > 0.58:
                    signal = "SELL"

            # Не спамим SELL, если позиции нет
            log_signal = signal
            if signal == "SELL" and position == 0:
                log_signal = "HOLD"

            logger.info(
                f"Price: {price:.2f} | Signal: {log_signal} | "
                f"P(up)={proba[1]:.2f} | trend_4h={last['trend_4h']} | "
                f"regime={'BULL' if regime_bull else 'BEAR'} | Pos: {position}"
            )

            # ---- Исполнение ----
            if signal == "BUY" and position == 0:
                order = risk_manager.calculate_position_size(price, "BUY")
                if order:
                    position = order["amount"]
                    entry_price = price
                    logger.success(
                        f"[TRADE] BUY {position:.4f} {Config.SYMBOL} @ {price:.2f} "
                        f"(P(up)={proba[1]:.2f})"
                    )
                    db = SessionLocal()
                    db.add(
                        Trade(
                            symbol=Config.SYMBOL,
                            side="buy",
                            price=price,
                            amount=position,
                        )
                    )
                    db.commit()
                    db.close()

            elif signal == "SELL" and position > 0:
                pnl = (price - entry_price) * position
                pnl_pct = (price / entry_price - 1) * 100
                logger.success(
                    f"[TRADE] SELL {position:.4f} {Config.SYMBOL} @ {price:.2f} | "
                    f"PnL=${pnl:+.2f} ({pnl_pct:+.2f}%)"
                )
                db = SessionLocal()
                db.add(
                    Trade(
                        symbol=Config.SYMBOL,
                        side="sell",
                        price=price,
                        amount=position,
                        pnl=pnl,
                    )
                )
                db.commit()
                db.close()
                position = 0
                entry_price = 0

            await asyncio.sleep(60)

        except KeyboardInterrupt:
            logger.info("Stopped by user.")
            break
        except Exception as e:
            logger.error(f"Loop error: {e}")
            await asyncio.sleep(10)

    await exchange.close()


if __name__ == "__main__":
    asyncio.run(main())