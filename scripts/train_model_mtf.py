import os

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from src.analysis.mtf_features import FEATURES_MTF, merge_timeframes
from src.core.config import Config

TARGET_THRESHOLD = 0.003  # 0.3% — больше обучающих примеров


def build_dataset() -> pd.DataFrame:
    path_1h = os.path.join(Config.DATA_DIR, f"{Config.SYMBOL.replace('/', '_')}_1h.csv")
    path_4h = os.path.join(Config.DATA_DIR, f"{Config.SYMBOL.replace('/', '_')}_4h.csv")
    if not os.path.exists(path_1h) or not os.path.exists(path_4h):
        raise FileNotFoundError("Run download_data.py for 1h and 4h first.")

    df_1h = pd.read_csv(path_1h, parse_dates=["timestamp"])
    df_4h = pd.read_csv(path_4h, parse_dates=["timestamp"])

    df = merge_timeframes(df_1h, df_4h)
    logger.info(f"Merged MTF dataset: {len(df)} rows")

    future_return = df["close"].shift(-1) / df["close"] - 1
    df["target"] = (future_return > TARGET_THRESHOLD).astype(int)
    mask = future_return.abs() > TARGET_THRESHOLD
    df = df[mask].dropna(subset=FEATURES_MTF + ["target"])
    logger.info(f"After target filter: {len(df)} rows")

    return df


def walk_forward_validate(df: pd.DataFrame, n_splits: int = 5):
    n = len(df)
    train_pct = 0.6
    step_pct = (1 - train_pct) / n_splits

    results = []
    for i in range(n_splits):
        train_end = int(n * (train_pct + i * step_pct))
        test_start = train_end
        test_end = int(n * (train_pct + (i + 1) * step_pct))

        if test_end > n:
            break

        train_df = df.iloc[:train_end]
        test_df = df.iloc[test_start:test_end]

        model = RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=15,
            random_state=42, class_weight="balanced",
        )
        model.fit(train_df[FEATURES_MTF], train_df["target"])
        preds = model.predict(test_df[FEATURES_MTF])
        acc = accuracy_score(test_df["target"], preds)
        results.append(acc)
        logger.info(
            f"Fold {i+1}: train_end={train_end}, test=[{test_start}:{test_end}], "
            f"acc={acc:.4f}"
        )

    if results:
        logger.info(
            f"Walk-forward mean accuracy: {np.mean(results):.4f} "
            f"(std {np.std(results):.4f})"
        )
    return results


def train_final():
    df = build_dataset()
    logger.info("Running walk-forward validation...")
    walk_forward_validate(df)

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    logger.info(f"Final train: {len(train_df)}, test: {len(test_df)}")

    model = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=15,
        random_state=42, class_weight="balanced",
    )
    model.fit(train_df[FEATURES_MTF], train_df["target"])

    y_pred = model.predict(test_df[FEATURES_MTF])
    logger.info(f"Holdout accuracy: {accuracy_score(test_df['target'], y_pred):.4f}")
    logger.info(f"\n{classification_report(test_df['target'], y_pred)}")

    for name, imp in zip(FEATURES_MTF, model.feature_importances_):
        logger.info(f"Feature '{name}': {imp:.4f}")

    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    model_path = os.path.join(Config.MODEL_DIR, "model_mtf.joblib")
    joblib.dump(model, model_path)
    logger.success(f"MTF model saved to {model_path}")


if __name__ == "__main__":
    train_final()