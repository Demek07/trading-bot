import ccxt.async_support as ccxt
from loguru import logger

from src.core.config import Config


class TradeExecutor:
    def __init__(self):
        self.exchange = getattr(ccxt, Config.EXCHANGE_ID)(
            {
                "apiKey": Config.API_KEY,
                "secret": Config.API_SECRET,
                "enableRateLimit": True,
            }
        )

    async def place_order(self, order_params: dict):
        if Config.DRY_RUN:
            logger.info(f"[DRY RUN] Order placed: {order_params}")
            return {"status": "simulated", "id": "12345"}

        try:
            order = await self.exchange.create_order(
                symbol=order_params["symbol"],
                type="market",
                side=order_params["side"],
                amount=order_params["amount"],
            )
            logger.info(f"Real order placed: {order}")
            return order
        except Exception as e:
            logger.error(f"Order execution failed: {e}")
            return None

    async def close(self):
        await self.exchange.close()