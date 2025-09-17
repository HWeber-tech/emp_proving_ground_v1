import pytest

from src.operational.metrics_registry import MetricsRegistry


def test_concurrent_labels_and_ops_no_race():
    # This test runs regardless of prometheus availability to ensure no exceptions under concurrency.
    reg = MetricsRegistry()
    c = reg.get_counter("m4_threads_counter", "desc", ["a", "b"])
    g = reg.get_gauge("m4_threads_gauge", "desc", ["k"])

    import threading

    errs = []

    def do_counter(i: int):
        try:
            child = c.labels(a="A", b=str(i % 3))
            child.inc()
            child.inc(0.5)
        except Exception as e:  # pragma: no cover - should never happen
            errs.append(e)

    def do_gauge(i: int):
        try:
            child = g.labels(k=str(i % 5))
            child.set(float(i % 7))
        except Exception as e:  # pragma: no cover - should never happen
            errs.append(e)

    threads = []
    for i in range(50):
        threads.append(threading.Thread(target=do_counter, args=(i,)))
        threads.append(threading.Thread(target=do_gauge, args=(i,)))

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errs, f"Unexpected exceptions in concurrent operations: {errs!r}"


def test_concurrent_get_counter_same_identity_when_prometheus_available():
    # Only validate identity semantics when real prometheus is present.
    pytest.importorskip("prometheus_client")

    reg = MetricsRegistry()

    import concurrent.futures

    def get_obj(_):
        return reg.get_counter("m4_identity_counter", "desc", ["x", "y"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as ex:
        objs = list(ex.map(get_obj, range(200)))

    assert objs, "Expected some results from concurrent get_counter()"
    first = objs[0]
    assert all(o is first for o in objs), "Expected the same memoized metric object across threads"
