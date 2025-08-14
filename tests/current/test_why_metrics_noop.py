from src.operational.metrics import set_why_conf, set_why_feature, set_why_signal


def test_why_metrics_noop():
    # Should not raise even if prometheus_client is missing
    set_why_signal("EURUSD", 0.1)
    set_why_conf("EURUSD", 0.9)
    set_why_feature("yields", True)

