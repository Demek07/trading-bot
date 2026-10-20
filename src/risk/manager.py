from loguru import logger

from src.core.config import Config


class RiskManager:
    def __init__(self, balance: float = 1000.0):
        self.balance = balance

    def calculate_position_size(self, current_price: float, signal: str):
        if signal == "HOLD":
            return None

        risk_amount = self.balance * Config.MAX_POSITION_SIZE_PCT
        position_size = risk_amount / current_price

        if signal == "BUY":
            stop_loss = current_price * (1 - Config.STOP_LOSS_PCT)
            take_profit = current_price * (1 + Config.TAKE_PROFIT_PCT)
        elif signal == "SELL":
            stop_loss = current_price * (1 + Config.STOP_LOSS_PCT)
            take_profit = current_price * (1 - Config.TAKE_PROFIT_PCT)
        else:
            return None

        return {
            "symbol": Config.SYMBOL,
            "side": signal.lower(),
            "amount": round(position_size, 4),
            "price": current_price,
            "sl": stop_loss,
            "tp": take_profit,
        }