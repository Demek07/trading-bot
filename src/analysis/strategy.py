import os

import joblib
import pandas as pd
from loguru import logger

from src.core.config import Config

FEATURES = ["rsi", "macd", "price_vs_sma", "rsi_change"]


class MLStrategy:
    def __init__(self):
        self.model = None
        self.features = FEATURES
        self.model_path = os.path.join(Config.MODEL_DIR, "model.joblib")

    def load_model(self) -> bool:
        if not os.path.exists(self.model_path):
            logger.error(f"Model not found at {self.model_path}. Train it first.")
            return False
        self.model = joblib.load(self.model_path)
        logger.info("Model loaded successfully.")
        return True

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        df["sma_fast"] = df["close"].rolling(window=9).mean()
        df["sma_slow"] = df["close"].rolling(window=21).mean()
        df["sma_trend"] = df["close"].rolling(window=200).mean()

        df["ema12"] = df["close"].ewm(span=12).mean()
        df["ema26"] = df["close"].ewm(span=26).mean()
        df["macd"] = df["ema12"] - df["ema26"]

        df["volatility"] = (df["high"] - df["low"]) / df["close"]
        df["volume_change"] = df["volume"].pct_change()
        df["price_vs_sma"] = (df["close"] - df["sma_slow"]) / df["sma_slow"]
        df["rsi_change"] = df["rsi"].diff()

        return df.dropna()

    def generate_signal(
        self,
        df: pd.DataFrame,
        buy_threshold: float = 0.60,
        sell_threshold: float = 0.60,
        use_trend_filter: bool = True,
    ) -> dict:
        if self.model is None:
            self.load_model()
        if self.model is None:
            return {"signal": "HOLD", "confidence": 0.0}

        df = self._calculate_indicators(df)
        if len(df) < 1:
            return {"signal": "HOLD", "confidence": 0.0}

        last = df.iloc[-1]
        X = last[self.features].to_frame().T

        pred = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]

        # Фильтр тренда: торгуем только по направлению SMA200
        trend_up = last["close"] > last["sma_trend"]
        trend_down = last["close"] < last["sma_trend"]

        if pred == 1 and proba[1] > buy_threshold:
            if not use_trend_filter or trend_up:
                return {"signal": "BUY", "confidence": float(proba[1])}
        elif pred == 0 and proba[0] > sell_threshold:
            if not use_trend_filter or trend_down:
                return {"signal": "SELL", "confidence": float(proba[0])}

        return {"signal": "HOLD", "confidence": float(max(proba))}