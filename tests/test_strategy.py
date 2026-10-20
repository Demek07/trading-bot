import numpy as np
import pandas as pd
import pytest

from src.analysis.strategy import MLStrategy


@pytest.fixture
def sample_df():
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(200))
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=200, freq="h"),
            "open": prices,
            "high": prices + 1,
            "low": prices - 1,
            "close": prices,
            "volume": [1000] * 200,
        }
    )


def test_indicators(sample_df):
    s = MLStrategy()
    out = s._calculate_indicators(sample_df)
    assert "rsi" in out.columns
    assert "macd" in out.columns
    assert "volatility" in out.columns
    assert "price_vs_sma" in out.columns
    assert not out.isnull().any().any()


def test_signal_returns_dict(sample_df):
    s = MLStrategy()
    result = s.generate_signal(sample_df)
    assert isinstance(result, dict)
    assert result["signal"] in ["BUY", "SELL", "HOLD"]
    assert 0.0 <= result["confidence"] <= 1.0