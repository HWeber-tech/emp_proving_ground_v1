from datetime import datetime, timezone
from decimal import Decimal

from src.sensory.lineage import build_lineage_record


def test_build_lineage_record_sanitises_inputs() -> None:
    record = build_lineage_record(
        "HOW",
        "sensory.how",
        inputs={
            "timestamp": datetime(2024, 1, 1, tzinfo=timezone.utc),
            "volume": Decimal("123.45"),
        },
        outputs={"signal": 0.5, "confidence": 0.75},
        telemetry={"liquidity": Decimal("0.9")},
        metadata={"mode": "market_data"},
    )

    payload = record.as_dict()
    assert payload["dimension"] == "HOW"
    assert payload["source"] == "sensory.how"
    assert payload["inputs"]["timestamp"].endswith("+00:00")
    assert payload["inputs"]["volume"] == 123.45
    assert payload["outputs"] == {"signal": 0.5, "confidence": 0.75}
    assert payload["telemetry"]["liquidity"] == 0.9
    assert payload["metadata"]["mode"] == "market_data"
    assert "generated_at" in payload
