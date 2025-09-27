from __future__ import annotations

import pytest

from src.trading.order_management import (
    InMemoryOrderEventJournal,
    OrderLifecycleProcessor,
    OrderMetadata,
    OrderStateError,
    OrderStatus,
)


class DummyBroker:
    def __init__(self) -> None:
        self.callbacks: dict[str, list] = {}

    def add_event_listener(self, event_type: str, callback) -> bool:
        self.callbacks.setdefault(event_type, []).append(callback)
        return True

    def remove_event_listener(self, event_type: str, callback) -> bool:
        callbacks = self.callbacks.get(event_type, [])
        if callback in callbacks:
            callbacks.remove(callback)
            return True
        return False

    def emit(self, event_type: str, order_id: str, payload: dict) -> None:
        for callback in list(self.callbacks.get(event_type, [])):
            callback(order_id, payload)


@pytest.fixture
def lifecycle_processor() -> tuple[OrderLifecycleProcessor, InMemoryOrderEventJournal]:
    journal = InMemoryOrderEventJournal()
    processor = OrderLifecycleProcessor(journal=journal)
    return processor, journal


def _register(processor: OrderLifecycleProcessor) -> str:
    metadata = OrderMetadata(
        order_id="ABC-123",
        symbol="EUR/USD",
        side="BUY",
        quantity=100,
    )
    processor.register_order(metadata)
    return metadata.order_id


def test_ack_and_fill_flow(lifecycle_processor: tuple[OrderLifecycleProcessor, InMemoryOrderEventJournal]) -> None:
    processor, journal = lifecycle_processor
    order_id = _register(processor)

    snap = processor.apply_acknowledgement(order_id)
    assert snap.status is OrderStatus.ACKNOWLEDGED

    snap = processor.apply_partial_fill(
        order_id,
        {"last_qty": "25", "last_px": "1.001"},
    )
    assert snap.status is OrderStatus.PARTIALLY_FILLED
    assert snap.filled_quantity == pytest.approx(25)

    snap = processor.apply_fill(order_id, {"cum_qty": "100", "last_px": "1.002"})
    assert snap.status is OrderStatus.FILLED
    assert snap.filled_quantity == pytest.approx(100)

    assert [record["event"]["event_type"] for record in journal.records] == [
        "acknowledged",
        "partial_fill",
        "filled",
    ]


def test_cancel_flow(lifecycle_processor: tuple[OrderLifecycleProcessor, InMemoryOrderEventJournal]) -> None:
    processor, journal = lifecycle_processor
    order_id = _register(processor)

    processor.apply_acknowledgement(order_id)
    snap = processor.apply_cancel(order_id)
    assert snap.status is OrderStatus.CANCELLED
    assert [record["event"]["event_type"] for record in journal.records][-1] == "cancelled"


def test_reject_from_pending(lifecycle_processor: tuple[OrderLifecycleProcessor, InMemoryOrderEventJournal]) -> None:
    processor, journal = lifecycle_processor
    order_id = _register(processor)

    snap = processor.apply_reject(order_id, {"text": "Invalid price"})
    assert snap.status is OrderStatus.REJECTED
    assert journal.records[-1]["event"]["event_type"] == "rejected"


def test_unknown_order_raises_error(lifecycle_processor: tuple[OrderLifecycleProcessor, InMemoryOrderEventJournal]) -> None:
    processor, _ = lifecycle_processor
    with pytest.raises(OrderStateError):
        processor.apply_acknowledgement("does-not-exist")


def test_broker_attachment_dispatches_events(lifecycle_processor: tuple[OrderLifecycleProcessor, InMemoryOrderEventJournal]) -> None:
    processor, journal = lifecycle_processor
    order_id = _register(processor)
    broker = DummyBroker()

    processor.attach_broker(broker)
    broker.emit("acknowledged", order_id, {"exec_type": "0"})
    broker.emit("partial_fill", order_id, {"exec_type": "1", "last_qty": "10", "last_px": "1.0"})
    broker.emit("filled", order_id, {"exec_type": "2", "cum_qty": "100", "last_px": "1.01"})

    assert len(journal.records) == 3
    processor.detach_broker()
    assert all(not callbacks for callbacks in broker.callbacks.values())
