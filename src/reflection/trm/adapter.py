"""Input adapter that reads decision diaries and prepares TRM batches."""

from __future__ import annotations

import datetime as dt
import json
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable

from .types import DecisionDiaryEntry, RIMInputBatch, RIMWindow


class RIMInputAdapter:
    """Loads decision diary slices for TRM inference."""

    def __init__(self, diaries_dir: Path, diary_glob: str, window_minutes: int) -> None:
        self._diaries_dir = diaries_dir
        self._diary_glob = diary_glob
        self._window_minutes = max(1, window_minutes)

    def latest_path(self) -> Path | None:
        if not self._diaries_dir.exists():
            return None
        candidates = sorted(self._diaries_dir.glob(self._diary_glob))
        return candidates[-1] if candidates else None

    def load_batch(self) -> RIMInputBatch | None:
        path = self.latest_path()
        if path is None:
            return None

        entries = list(self._load_entries(path))
        if not entries:
            return None

        entries.sort(key=lambda entry: entry.timestamp)
        end_ts = entries[-1].timestamp
        window_delta = dt.timedelta(minutes=self._window_minutes)
        window_start = end_ts - window_delta
        filtered_entries = tuple(entry for entry in entries if entry.timestamp >= window_start)
        if not filtered_entries:
            filtered_entries = tuple(entries[-1:])
        window_minutes = int((filtered_entries[-1].timestamp - filtered_entries[0].timestamp).total_seconds() / 60) or 1
        rim_window = RIMWindow(start=filtered_entries[0].timestamp, end=filtered_entries[-1].timestamp, minutes=window_minutes)

        aggregates = self._build_aggregates(filtered_entries)
        input_hash = self._compute_input_hash(filtered_entries)
        return RIMInputBatch(
            entries=filtered_entries,
            input_hash=input_hash,
            window=rim_window,
            aggregates=aggregates,
            source_path=path,
        )

    def _load_entries(self, path: Path) -> Iterable[DecisionDiaryEntry]:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                try:
                    yield self._normalise_entry(payload)
                except (KeyError, ValueError, TypeError):
                    continue

    def _normalise_entry(self, payload: dict[str, Any]) -> DecisionDiaryEntry:
        timestamp_raw = str(payload["timestamp"])
        timestamp = _parse_timestamp(timestamp_raw)
        risk_flags = tuple(str(flag) for flag in payload.get("risk_flags", ()) if flag)
        outcome_labels = tuple(str(label) for label in payload.get("outcome_labels", ()) if label)
        features_digest = {
            key: float(value)
            for key, value in (payload.get("features_digest") or {}).items()
            if isinstance(value, (int, float))
        }
        belief_summary = payload.get("belief_state_summary") or {}
        belief_confidence = belief_summary.get("confidence")
        belief_confidence_value = float(belief_confidence) if isinstance(belief_confidence, (int, float)) else None
        return DecisionDiaryEntry(
            raw=payload,
            timestamp=timestamp,
            strategy_id=str(payload.get("strategy_id") or "unknown"),
            instrument=str(payload.get("instrument") or "unknown"),
            pnl=float(payload.get("pnl", 0.0)),
            risk_flags=risk_flags,
            outcome_labels=outcome_labels,
            features_digest=features_digest,
            belief_confidence=belief_confidence_value,
            action=str(payload.get("action") or "unknown"),
            input_hash=str(payload.get("input_hash") or ""),
        )

    @staticmethod
    def _compute_input_hash(entries: Iterable[DecisionDiaryEntry]) -> str:
        digest_source = ",".join(
            f"{entry.timestamp.isoformat()}|{entry.strategy_id}|{entry.input_hash}|{entry.pnl:.6f}"
            for entry in entries
        )
        return sha256(digest_source.encode("utf-8")).hexdigest()

    @staticmethod
    def _build_aggregates(entries: Iterable[DecisionDiaryEntry]) -> dict[str, Any]:
        total = 0
        pnl = 0.0
        strategies: dict[str, int] = {}
        risk_counts: dict[str, int] = {}
        for entry in entries:
            total += 1
            pnl += entry.pnl
            strategies[entry.strategy_id] = strategies.get(entry.strategy_id, 0) + 1
            for flag in entry.risk_flags:
                risk_counts[flag] = risk_counts.get(flag, 0) + 1
        return {
            "total_entries": total,
            "mean_pnl": pnl / total if total else 0.0,
            "strategies": strategies,
            "risk_flags": risk_counts,
        }


def _parse_timestamp(value: str) -> dt.datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    parsed = dt.datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


__all__ = ["RIMInputAdapter"]
