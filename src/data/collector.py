import ccxt.async_support as ccxt
import pandas as pd
from src.core.config import Config
from loguru import logger

class DataCollector:
    def __init__(self):
        self.exchange = getattr(ccxt, Config.EXCHANGE_ID)({
            'apiKey': Config.API_KEY,
            'secret': Config.API_SECRET,
            'enableRateLimit': True,
        })

    async def fetch_ohlcv(self, limit=100) -> pd.DataFrame:
        """Получает исторические свечи."""
        try:
            ohlcv = await self.exchange.fetch_ohlcv(Config.SYMBOL, Config.TIMEFRAME, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return pd.DataFrame()

    async def close(self):
        await self.exchange.close()