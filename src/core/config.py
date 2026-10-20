import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    EXCHANGE_ID = os.getenv("EXCHANGE_ID", "binance")
    API_KEY = os.getenv("API_KEY", "")
    API_SECRET = os.getenv("API_SECRET", "")
    SYMBOL = os.getenv("SYMBOL", "BTC/USDT")
    TIMEFRAME = os.getenv("TIMEFRAME", "1h")

    # Risk Management
    MAX_POSITION_SIZE_PCT = float(os.getenv("MAX_POSITION_SIZE_PCT", 0.05))
    STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", 0.02))
    TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", 0.05))

    # Database
    DB_URL = os.getenv("DB_URL", "sqlite:///trading.db")

    # Directories
    DATA_DIR = os.getenv("DATA_DIR", "./data")
    MODEL_DIR = os.getenv("MODEL_DIR", "./models")

    # Mode
    DRY_RUN = os.getenv("DRY_RUN", "True").lower() == "true"