from __future__ import annotations

from datetime import datetime, timedelta, timezone
try:  # Python < 3.11 compatibility
    from datetime import UTC  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    UTC = timezone.utc  # type: ignore[assignment]
from pathlib import Path
from typing import Callable

from src.governance.policy_changelog import render_policy_ledger_changelog
from src.governance.policy_ledger import (
    LedgerReleaseManager,
    PolicyLedgerFeatureFlags,
    PolicyLedgerStage,
    PolicyLedgerStore,
)


def _incrementing_now_factory(start: datetime) -> Callable[[], datetime]:
    counter = {"offset": 0}

    def _now() -> datetime:
        current = start + timedelta(minutes=counter["offset"])
        counter["offset"] += 1
        return current

    return _now


def test_render_policy_ledger_changelog_includes_history(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.json"
    start = datetime(2024, 5, 1, 12, 0, tzinfo=UTC)
    store = PolicyLedgerStore(ledger_path, now=_incrementing_now_factory(start))
    manager = LedgerReleaseManager(store)

    manager.promote(
        policy_id="alpha.policy",
        tactic_id="alpha-tactic",
        stage=PolicyLedgerStage.PAPER,
        approvals=("risk", "compliance"),
        evidence_id="diary-001",
        threshold_overrides={"warn_confidence_floor": 0.82},
    )
    manager.promote(
        policy_id="alpha.policy",
        tactic_id="alpha-tactic",
        stage=PolicyLedgerStage.PILOT,
        approvals=("risk", "ops"),
        evidence_id="diary-002",
    )

    markdown = render_policy_ledger_changelog(
        store,
        runbook_url="https://runbooks/policy",
        generated_at=datetime(2024, 5, 2, 9, 30, tzinfo=UTC),
    )

    assert "Policy Ledger Governance Summary" in markdown
    assert "alpha.policy" in markdown
    assert "diary-002" in markdown
    assert "warn_confidence_floor" in markdown
    assert "https://runbooks/policy" in markdown
    assert "PILOT" in markdown
    assert "|" in markdown  # history table present


def test_render_policy_ledger_changelog_flags_missing_evidence(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.json"
    store = PolicyLedgerStore(ledger_path)
    manager = LedgerReleaseManager(
        store,
        feature_flags=PolicyLedgerFeatureFlags(require_diary_evidence=False),
    )

    manager.promote(
        policy_id="beta.policy",
        tactic_id="beta-tactic",
        stage=PolicyLedgerStage.EXPERIMENT,
        approvals=("risk",),
    )

    markdown = render_policy_ledger_changelog(store, runbook_url="http://runbook")
    assert "beta.policy" in markdown
    assert "missing (see runbook)" in markdown
    assert "none (see runbook)" not in markdown.splitlines()[0]  # header unaffected

