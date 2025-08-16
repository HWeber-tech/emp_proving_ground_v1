import sys
import types


def test_start_metrics_server_idempotent() -> None:
    # Lazy import; function resolves prometheus_client inside the call
    import src.operational.metrics as metrics  # type: ignore

    # Stub prometheus_client.start_http_server
    orig = sys.modules.get("prometheus_client")
    calls = {"n": 0}

    stub = types.ModuleType("prometheus_client")
    def start_http_server(_port: int) -> None:  # noqa: N802 (3rd-party style)
        calls["n"] += 1
    stub.start_http_server = start_http_server  # type: ignore[attr-defined]

    sys.modules["prometheus_client"] = stub
    try:
        metrics._started = False  # reset guarded state
        metrics.start_metrics_server(42424)
        metrics.start_metrics_server(42424)
        assert calls["n"] == 1  # only started once
    finally:
        if orig is not None:
            sys.modules["prometheus_client"] = orig
        else:
            sys.modules.pop("prometheus_client", None)
        metrics._started = False