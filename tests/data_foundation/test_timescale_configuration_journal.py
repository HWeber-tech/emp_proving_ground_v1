"""Round-trip tests for Timescale configuration audit journal."""

from pathlib import Path

from src.data_foundation.persist.timescale import (
    TimescaleConfigurationAuditJournal,
    TimescaleConnectionSettings,
    TimescaleMigrator,
)
from src.governance.system_config import DataBackboneMode, EmpTier, RunMode, SystemConfig
from src.operations.configuration_audit import evaluate_configuration_audit


def _create_engine(db_path: Path):
    settings = TimescaleConnectionSettings(url=f"sqlite:///{db_path}")
    engine = settings.create_engine()
    TimescaleMigrator(engine).apply()
    return engine


def test_configuration_audit_journal_roundtrip(tmp_path) -> None:
    engine = _create_engine(tmp_path / "config-audit.db")
    journal = TimescaleConfigurationAuditJournal(engine)

    baseline = evaluate_configuration_audit(SystemConfig())
    updated = SystemConfig().with_updated(
        run_mode=RunMode.live,
        tier=EmpTier.tier_1,
        data_backbone_mode=DataBackboneMode.institutional,
    )
    snapshot = evaluate_configuration_audit(
        updated,
        previous=baseline.current_config,
    )

    stored = journal.record_snapshot(snapshot.as_dict())
    assert stored["snapshot_id"] == snapshot.snapshot_id

    latest = journal.fetch_latest()
    assert latest is not None
    assert latest.snapshot_id == snapshot.snapshot_id
    assert latest.current_config["run_mode"] == RunMode.live.value

    recent = journal.fetch_recent(limit=2)
    assert recent
    assert recent[0].snapshot_id == snapshot.snapshot_id

    journal.close()
