import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.reflection.self_audit import SelfAuditLog


def _log_path(root: Path, moment: datetime) -> Path:
    return root / f"{moment.year:04d}" / f"{moment.month:02d}" / f"{moment.day:02d}" / "self_audit.json"


def test_self_audit_creates_daily_log(tmp_path) -> None:
    reporter = SelfAuditLog(root=tmp_path)
    moment = datetime(2024, 5, 1, 9, 30, tzinfo=timezone.utc)

    entry = reporter.record(
        knows=["AlphaTrade can replay decision diaries"],
        doubts=["Need latency measurements for new feed"],
        changes=["Enabled structured logging transport"],
        metadata={"owner": "operations", 1: "converted"},
        when=moment,
    )

    path = _log_path(tmp_path, moment)
    assert path.exists(), "expected daily self-audit log"
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["date"] == "2024-05-01"
    stored_entry = payload["entries"][0]
    assert stored_entry["knows"] == ["AlphaTrade can replay decision diaries"]
    assert stored_entry["doubts"] == ["Need latency measurements for new feed"]
    assert stored_entry["changes"] == ["Enabled structured logging transport"]
    assert stored_entry["signature"] == entry.signature
    assert stored_entry["metadata"] == {"owner": "operations", "1": "converted"}


def test_self_audit_chains_signatures(tmp_path) -> None:
    reporter = SelfAuditLog(root=tmp_path)
    base = datetime(2024, 6, 2, 8, 0, tzinfo=timezone.utc)

    first = reporter.record(
        knows="Governance packet is complete",
        changes="Applied RIM suggestion for EURUSD",
        when=base,
    )
    second = reporter.record(
        knows=["Governance packet is complete"],
        doubts=["Need regression run for AUDUSD"],
        changes=["Captured compliance reconciliation"],
        when=base + timedelta(hours=3),
    )

    assert second.previous_signature == first.signature

    path = _log_path(tmp_path, base)
    payload = json.loads(path.read_text(encoding="utf-8"))
    stored_entries = payload["entries"]
    assert len(stored_entries) == 2
    assert stored_entries[1]["previous_signature"] == first.signature


def test_self_audit_normalises_buckets(tmp_path) -> None:
    reporter = SelfAuditLog(root=tmp_path)
    moment = datetime(2024, 7, 15, 21, 5, tzinfo=timezone.utc)

    entry = reporter.record(
        knows=["  New market regime classifier  ", "New market regime classifier", ""],
        doubts="   ",
        changes=["Expanded feature lattice", "Expanded feature lattice", "Risk thresholds refreshed"],
        when=moment,
    )

    assert entry.knows == ("New market regime classifier",)
    assert entry.doubts == tuple()
    assert entry.changes == ("Expanded feature lattice", "Risk thresholds refreshed")

    path = _log_path(tmp_path, moment)
    payload = json.loads(path.read_text(encoding="utf-8"))
    stored_entry = payload["entries"][0]
    assert stored_entry["knows"] == ["New market regime classifier"]
    assert stored_entry["doubts"] == []
    assert stored_entry["changes"] == [
        "Expanded feature lattice",
        "Risk thresholds refreshed",
    ]
