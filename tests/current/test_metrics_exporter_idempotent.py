import sys
import types
from pathlib import Path


def test_start_metrics_server_idempotent() -> None:
    # Lazy import; function resolves prometheus_client inside the call
    import src.operational.metrics as metrics  # type: ignore

    cert = Path(__file__).resolve().parents[1] / "runtime" / "certs" / "server.pem"
    key = Path(__file__).resolve().parents[1] / "runtime" / "certs" / "server.key"

    # Stub prometheus_client exports used by metrics module
    orig = sys.modules.get("prometheus_client")
    stub = types.ModuleType("prometheus_client")
    stub.CONTENT_TYPE_LATEST = "text/plain"  # type: ignore[attr-defined]
    stub.REGISTRY = object()  # type: ignore[attr-defined]
    stub.generate_latest = lambda _registry: b"metric 1\n"  # type: ignore[attr-defined]

    ports: list[int] = []

    class _Server:
        def __init__(self, port: int) -> None:
            self.server_address = ("", port)

        def serve_forever(self) -> None:
            ports.append(self.server_address[1])

    class _Thread:
        def __init__(self, target, **_kwargs) -> None:
            self._target = target

        def start(self) -> None:
            self._target()

    orig_factory = metrics._make_metrics_server
    orig_thread = metrics.threading.Thread
    metrics._make_metrics_server = lambda port, _handler, _context: _Server(port)  # type: ignore[assignment]
    metrics.threading.Thread = _Thread  # type: ignore[assignment]

    sys.modules["prometheus_client"] = stub
    try:
        metrics._started = False  # reset guarded state
        metrics.start_metrics_server(42424, cert_path=str(cert), key_path=str(key))
        metrics.start_metrics_server(42424, cert_path=str(cert), key_path=str(key))
        assert ports == [42424]  # only started once
    finally:
        metrics._make_metrics_server = orig_factory  # type: ignore[assignment]
        metrics.threading.Thread = orig_thread  # type: ignore[assignment]
        if orig is not None:
            sys.modules["prometheus_client"] = orig
        else:
            sys.modules.pop("prometheus_client", None)
        metrics._started = False
