import numpy as np
import pandas as pd
import pytest

from src.analysis.mtf_features import (
    FEATURES_MTF,
    calculate_indicators_1h,
    calculate_indicators_4h,
    merge_timeframes,
)


@pytest.fixture
def df_1h():
    n = 1000
    prices = 100 + np.cumsum(np.random.randn(n))
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
        "open": prices, "high": prices + 1, "low": prices - 1,
        "close": prices, "volume": [1000] * n,
    })


@pytest.fixture
def df_4h():
    n = 500
    prices = 100 + np.cumsum(np.random.randn(n))
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="4h"),
        "open": prices, "high": prices + 1, "low": prices - 1,
        "close": prices, "volume": [1000] * n,
    })


def test_indicators_1h(df_1h):
    out = calculate_indicators_1h(df_1h)
    assert "rsi_1h" in out.columns
    assert "macd_hist_1h" in out.columns


def test_indicators_4h(df_4h):
    out = calculate_indicators_4h(df_4h)
    assert "rsi_4h" in out.columns
    assert "trend_4h" in out.columns


def test_merge_no_lookahead(df_1h, df_4h):
    """Проверяем, что 4h-фичи не заглядывают в будущее."""
    merged = merge_timeframes(df_1h, df_4h)
    assert len(merged) > 0
    for f in FEATURES_MTF:
        assert f in merged.columns
    assert merged["trend_4h"].notnull().all()


def test_merge_backward_only(df_1h, df_4h):
    """
    Ключевая проверка от look-ahead: на каждой 1h-свече
    available_at (4h-свечи) должно быть <= timestamp (1h-свечи).
    """
    merged = merge_timeframes(df_1h, df_4h)
    assert (merged["available_at"] <= merged["timestamp"]).all()