import os

import joblib
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from src.core.config import Config

# Упрощенный набор фич — меньше шума, лучше обобщение
FEATURES = ["rsi", "macd", "price_vs_sma", "rsi_change"]

# Порог значимого движения для таргета
TARGET_THRESHOLD = 0.005  # 0.5%


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # RSI
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # SMA
    df["sma_fast"] = df["close"].rolling(window=9).mean()
    df["sma_slow"] = df["close"].rolling(window=21).mean()
    df["sma_trend"] = df["close"].rolling(window=200).mean()  # фильтр тренда

    # MACD
    df["ema12"] = df["close"].ewm(span=12).mean()
    df["ema26"] = df["close"].ewm(span=26).mean()
    df["macd"] = df["ema12"] - df["ema26"]

    # Дополнительные фичи
    df["volatility"] = (df["high"] - df["low"]) / df["close"]
    df["volume_change"] = df["volume"].pct_change()
    df["price_vs_sma"] = (df["close"] - df["sma_slow"]) / df["sma_slow"]
    df["rsi_change"] = df["rsi"].diff()

    # Таргет: значимый рост > TARGET_THRESHOLD
    future_return = df["close"].shift(-1) / df["close"] - 1
    df["target"] = (future_return > TARGET_THRESHOLD).astype(int)

    # Убираем нейтральные движения из обучения (шум)
    mask = future_return.abs() > TARGET_THRESHOLD
    df = df[mask]

    return df.dropna()


def train():
    filepath = os.path.join(
        Config.DATA_DIR, f"{Config.SYMBOL.replace('/', '_')}_{Config.TIMEFRAME}.csv"
    )
    if not os.path.exists(filepath):
        logger.error(f"Data file not found: {filepath}. Run download_data.py first.")
        return

    df = pd.read_csv(filepath, parse_dates=["timestamp"])
    logger.info(f"Loaded {len(df)} rows")

    df = add_features(df)
    logger.info(f"After feature engineering + target filter: {len(df)} rows")

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train, y_train = train_df[FEATURES], train_df["target"]
    X_test, y_test = test_df[FEATURES], test_df["target"]

    logger.info(
        f"Train: {len(train_df)} rows, Test: {len(test_df)} rows, "
        f"Target balance (train): {y_train.mean():.2%} positive"
    )

    # Упрощенная модель — меньше переобучения
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=50,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"Test Accuracy: {acc:.4f}")
    logger.info(f"\n{classification_report(y_test, y_pred)}")

    for name, imp in zip(FEATURES, model.feature_importances_):
        logger.info(f"Feature '{name}': {imp:.4f}")

    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    model_path = os.path.join(Config.MODEL_DIR, "model.joblib")
    joblib.dump(model, model_path)
    logger.success(f"Model saved to {model_path}")


if __name__ == "__main__":
    train()