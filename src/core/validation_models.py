from __future__ import annotations

from datetime import datetime
from typing import Optional


class ValidationResult:
    """Canonical validation result model."""

    def __init__(
        self,
        test_name: str,
        passed: bool,
        value: float,
        threshold: float,
        unit: str,
        details: str = "",
        metadata: Optional[dict[str, object]] = None,
    ):
        self.test_name = test_name
        self.passed = passed
        self.value = value
        self.threshold = threshold
        self.unit = unit
        self.details = details
        self.metadata = metadata or {}
        self.timestamp = datetime.now()

    def to_dict(self) -> dict[str, object]:
        return {
            "test_name": self.test_name,
            "passed": self.passed,
            "value": self.value,
            "threshold": self.threshold,
            "unit": self.unit,
            "details": self.details,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


__all__ = ["ValidationResult"]