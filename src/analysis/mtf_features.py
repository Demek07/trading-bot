import pandas as pd


def calculate_indicators_1h(df: pd.DataFrame) -> pd.DataFrame:
    """Индикаторы локального таймфрейма (1h)."""
    df = df.copy()
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["rsi_1h"] = 100 - (100 / (1 + rs))

    df["ema12_1h"] = df["close"].ewm(span=12).mean()
    df["ema26_1h"] = df["close"].ewm(span=26).mean()
    df["macd_1h"] = df["ema12_1h"] - df["ema26_1h"]
    df["macd_signal_1h"] = df["macd_1h"].ewm(span=9).mean()
    df["macd_hist_1h"] = df["macd_1h"] - df["macd_signal_1h"]

    df["sma20_1h"] = df["close"].rolling(window=20).mean()
    df["price_vs_sma20_1h"] = (df["close"] - df["sma20_1h"]) / df["sma20_1h"]

    df["volatility_1h"] = (df["high"] - df["low"]) / df["close"]
    df["volume_change_1h"] = df["volume"].pct_change()
    df["rsi_change_1h"] = df["rsi_1h"].diff()

    return df


def calculate_indicators_4h(df: pd.DataFrame) -> pd.DataFrame:
    """Индикаторы старшего таймфрейма (4h)."""
    df = df.copy()
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["rsi_4h"] = 100 - (100 / (1 + rs))

    df["sma50_4h"] = df["close"].rolling(window=50).mean()
    df["sma200_4h"] = df["close"].rolling(window=200).mean()
    df["price_vs_sma200_4h"] = (df["close"] - df["sma200_4h"]) / df["sma200_4h"]
    df["trend_4h"] = (df["close"] > df["sma200_4h"]).astype(int)

    df["ema12_4h"] = df["close"].ewm(span=12).mean()
    df["ema26_4h"] = df["close"].ewm(span=26).mean()
    df["macd_4h"] = df["ema12_4h"] - df["ema26_4h"]

    return df


def merge_timeframes(df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> pd.DataFrame:
    """
    Объединяет 1h и 4h фичи.
    Важно: используем merge_asof с direction='backward',
    чтобы на 1h-свече была доступна только УЖЕ ЗАКРЫТАЯ 4h-свеча (без look-ahead).
    """

    # Приводим timestamp к единому типу datetime64[ns]
    df_1h = df_1h.copy()
    df_4h = df_4h.copy()
    df_1h["timestamp"] = pd.to_datetime(df_1h["timestamp"]).astype("datetime64[ns]")
    df_4h["timestamp"] = pd.to_datetime(df_4h["timestamp"]).astype("datetime64[ns]")

    df_1h = calculate_indicators_1h(df_1h)
    df_4h = calculate_indicators_4h(df_4h)

    df_4h_features = df_4h[
        ["timestamp", "rsi_4h", "sma50_4h", "sma200_4h",
         "price_vs_sma200_4h", "trend_4h", "macd_4h"]
    ].copy()

    # Закрытие 4h-свечи = timestamp + 4h. Сдвигаем, чтобы не заглядывать в будущее.
    df_4h_features["available_at"] = df_4h_features["timestamp"] + pd.Timedelta(hours=4)

    df_merged = pd.merge_asof(
        df_1h.sort_values("timestamp"),
        df_4h_features.drop(columns=["timestamp"]).sort_values("available_at"),
        left_on="timestamp",
        right_on="available_at",
        direction="backward",
    )

    return df_merged.dropna()


FEATURES_MTF = [
    # 1h
    "rsi_1h", "macd_1h", "macd_hist_1h", "price_vs_sma20_1h",
    "volatility_1h", "volume_change_1h", "rsi_change_1h",
    # 4h
    "rsi_4h", "price_vs_sma200_4h", "trend_4h", "macd_4h",
]