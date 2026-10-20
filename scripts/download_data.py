import asyncio
import os
import sys

import ccxt.async_support as ccxt
import pandas as pd
from loguru import logger

from src.core.config import Config


async def download_historical_data(symbol: str, timeframe: str, since_days: int = 180):
    exchange = getattr(ccxt, Config.EXCHANGE_ID)({"enableRateLimit": True})

    since = exchange.parse8601(
        (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=since_days)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    )

    all_ohlcv = []
    logger.info(f"Starting download {symbol} {timeframe} for last {since_days} days...")

    while True:
        try:
            ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not ohlcv:
                break

            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1

            logger.info(
                f"Downloaded {len(all_ohlcv)} candles. "
                f"Last: {pd.to_datetime(ohlcv[-1][0], unit='ms')}"
            )

            if since >= exchange.milliseconds():
                break

            await asyncio.sleep(exchange.rateLimit / 1000)
        except Exception as e:
            logger.error(f"Error: {e}")
            break

    await exchange.close()

    df = pd.DataFrame(
        all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = (
        df.drop_duplicates(subset="timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    os.makedirs(Config.DATA_DIR, exist_ok=True)
    filepath = os.path.join(Config.DATA_DIR, f"{symbol.replace('/', '_')}_{timeframe}.csv")
    df.to_csv(filepath, index=False)
    logger.success(f"Saved {len(df)} rows to {filepath}")
    return df


if __name__ == "__main__":
    asyncio.run(
        download_historical_data(Config.SYMBOL, Config.TIMEFRAME, since_days=180)
    )

if __name__ == "__main__":
    timeframe = sys.argv[1] if len(sys.argv) > 1 else Config.TIMEFRAME
    asyncio.run(download_historical_data(Config.SYMBOL, timeframe, since_days=365))