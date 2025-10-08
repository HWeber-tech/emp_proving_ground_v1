from __future__ import annotations

import json
from typing import Dict, List, Optional

import pytest

from src.thinking.prediction.predictive_market_modeler import PredictiveMarketModeler


class _FakeStateStore:
    def __init__(self) -> None:
        self.storage: Dict[str, str] = {}

    async def set(self, key: str, value: str, expire: Optional[int] = None) -> bool:
        self.storage[key] = value
        return True

    async def get(self, key: str) -> Optional[str]:
        return self.storage.get(key)

    async def delete(self, key: str) -> bool:  # pragma: no cover - helper parity
        return self.storage.pop(key, None) is not None

    async def keys(self, pattern: str) -> List[str]:  # pragma: no cover - unused
        return [key for key in self.storage if key.startswith(pattern.rstrip("*"))]

    async def clear(self) -> bool:  # pragma: no cover - unused
        self.storage.clear()
        return True


@pytest.mark.asyncio
async def test_prediction_history_discards_malicious_payload() -> None:
    store = _FakeStateStore()
    modeler = PredictiveMarketModeler(store)  # type: ignore[arg-type]

    malicious_blob = '__import__("os").system("echo owned")'
    await store.set(modeler._prediction_history_key, malicious_blob)

    history = await modeler._get_historical_data()

    assert history == {
        "accuracy": 0.75,
        "total_predictions": 0,
        "successful_predictions": 0,
    }


@pytest.mark.asyncio
async def test_store_predictions_serializes_to_json() -> None:
    store = _FakeStateStore()
    modeler = PredictiveMarketModeler(store)  # type: ignore[arg-type]

    predictions = [{"confidence": 0.6, "probability": 0.55}]

    await modeler._store_predictions(predictions)

    assert len(store.storage) == 1
    payload = next(iter(store.storage.values()))
    decoded = json.loads(payload)
    assert decoded == predictions
