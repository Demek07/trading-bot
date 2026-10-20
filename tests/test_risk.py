from src.risk.manager import RiskManager


def test_position_sizing():
    rm = RiskManager(balance=1000)
    order = rm.calculate_position_size(current_price=100, signal="BUY")
    assert order is not None
    assert order["side"] == "buy"
    assert order["amount"] > 0
    assert order["sl"] < 100
    assert order["tp"] > 100


def test_no_order_on_hold():
    rm = RiskManager(balance=1000)
    assert rm.calculate_position_size(100, "HOLD") is None