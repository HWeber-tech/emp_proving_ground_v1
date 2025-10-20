from __future__ import annotations

import logging
from pathlib import Path

import pytest

from src.governance.promotion_integrity import PromotionGuard, PromotionIntegrityError


def test_promotion_guard_logs_rbac_rejection(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    guard = PromotionGuard(
        ledger_path=tmp_path / "ledger.json",
        diary_path=tmp_path / "diary.json",
    )

    with caplog.at_level(logging.WARNING):
        with pytest.raises(PromotionIntegrityError):
            guard.validate("policy-123", "approved")

    assert any(
        record.message.startswith("Promotion guard rejection") for record in caplog.records
    )
