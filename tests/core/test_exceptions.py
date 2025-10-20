from src.core.exceptions import OrderExecutionException


def test_order_execution_exception_prunes_null_context():
    exc = OrderExecutionException("failed", order_id=None, symbol="AAPL", details=None)

    assert exc.context == {"symbol": "AAPL"}
    assert exc.error_code == "OrderExecutionException"


def test_emp_exception_str_includes_error_code():
    exc = OrderExecutionException("failed", order_id="order-1", symbol=None)

    assert str(exc) == "[OrderExecutionException] failed"
