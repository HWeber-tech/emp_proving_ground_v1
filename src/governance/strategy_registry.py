"""
EMP Strategy Registry v1.2 - SQLite Implementation

Persistent strategy registry using SQLite for champion genome storage.
Implements GOV-02 ticket requirements for database-backed strategy management.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, MutableMapping, cast

try:  # PyYAML is optional in some runtime profiles
    import yaml
except Exception:  # pragma: no cover - graceful fallback when PyYAML absent
    yaml = None

from src.governance.policy_ledger import PolicyLedgerStage
from src.governance.promotion_integrity import PromotionGuard, PromotionIntegrityError


_DEFAULT_PROMOTION_GUARD_CONFIG = Path("config/governance/promotion_guard.yaml")
_DEFAULT_LEDGER_PATH = Path("artifacts/governance/policy_ledger.json")
_DEFAULT_DIARY_PATH = Path("artifacts/governance/decision_diary.json")


class StrategyStatus(Enum):
    EVOLVED = "evolved"
    APPROVED = "approved"
    APPROVED_DEFAULT = "approved_default"
    APPROVED_FALLBACK = "approved_fallback"
    ACTIVE = "active"
    INACTIVE = "inactive"
    REJECTED = "rejected"


class StrategyRegistryError(RuntimeError):
    """Raised when the registry cannot persist or retrieve strategy data."""


class StrategyRegistry:
    """Persistent strategy registry using SQLite database."""

    def __init__(
        self,
        db_path: str = "governance.db",
        *,
        promotion_guard: PromotionGuard | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger(f"{__name__}.StrategyRegistry")
        self._initialize_database()
        if promotion_guard is None:
            promotion_guard = self._build_default_promotion_guard()
        self._promotion_guard = promotion_guard

    @contextmanager
    def _managed_connection(self) -> Iterator[sqlite3.Connection]:
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
        except sqlite3.Error as exc:  # pragma: no cover - sqlite connect failure
            raise StrategyRegistryError("Unable to open strategy registry database") from exc

        try:
            yield conn
            conn.commit()
        except sqlite3.Error as exc:
            conn.rollback()
            raise StrategyRegistryError("Strategy registry database operation failed") from exc
        finally:
            conn.close()

    def close(self) -> None:
        """Release any registry resources.

        The registry opens short-lived SQLite connections via
        :meth:`_managed_connection`, so there is no persistent handle to close.
        Exposing an explicit ``close`` method keeps the API aligned with the
        legacy shim while letting callers treat the registry like other managed
        resources.
        """

        self._logger.debug("Strategy Registry close requested; no persistent connections to close")

    def _initialize_database(self) -> None:
        governance_columns = {
            "seed_source": "TEXT",
            "catalogue_name": "TEXT",
            "catalogue_version": "TEXT",
            "catalogue_seeded_at": "REAL",
            "catalogue_metadata": "TEXT",
            "catalogue_entry_id": "TEXT",
            "catalogue_entry_name": "TEXT",
            "catalogue_entry_metadata": "TEXT",
        }

        with self._managed_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS strategies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    genome_id TEXT UNIQUE NOT NULL,
                    created_at TEXT NOT NULL,
                    dna TEXT NOT NULL,
                    fitness_report TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'evolved',
                    strategy_name TEXT,
                    generation INTEGER,
                    fitness_score REAL,
                    max_drawdown REAL,
                    sharpe_ratio REAL,
                    total_return REAL,
                    volatility REAL,
                    seed_source TEXT,
                    catalogue_name TEXT,
                    catalogue_version TEXT,
                    catalogue_seeded_at REAL,
                    catalogue_metadata TEXT,
                    catalogue_entry_id TEXT,
                    catalogue_entry_name TEXT,
                    catalogue_entry_metadata TEXT
                )
            """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_genome_id ON strategies(genome_id)
            """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_status ON strategies(status)
            """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_fitness_score ON strategies(fitness_score DESC)
            """
            )

            cursor.execute("PRAGMA table_info(strategies)")
            existing_columns = {row[1] for row in cursor.fetchall()}
            for column, definition in governance_columns.items():
                if column not in existing_columns:
                    cursor.execute(
                        f"ALTER TABLE strategies ADD COLUMN {column} {definition}"
                    )

        self._logger.info("Strategy Registry initialised with database: %s", self.db_path)

    def _build_default_promotion_guard(self) -> PromotionGuard:
        guard = self._promotion_guard_from_config()
        if guard is not None:
            return guard

        ledger_path = Path(
            os.getenv("POLICY_LEDGER_PATH", str(_DEFAULT_LEDGER_PATH))
        )
        diary_path = Path(
            os.getenv("DECISION_DIARY_PATH", str(_DEFAULT_DIARY_PATH))
        )
        regimes_env = os.getenv("PROMOTION_REQUIRED_REGIMES")
        required_regimes: Iterable[str] | None
        if regimes_env:
            required_regimes = [value.strip() for value in regimes_env.split(",") if value.strip()]
        else:
            required_regimes = None
        try:
            min_regime = int(os.getenv("PROMOTION_MIN_REGIME_COUNT", "3"))
        except ValueError:
            min_regime = 3
        return PromotionGuard(
            ledger_path=ledger_path,
            diary_path=diary_path,
            required_regimes=required_regimes,
            min_decisions_per_regime=min_regime,
        )

    def _promotion_guard_from_config(self) -> PromotionGuard | None:
        config_path_env = os.getenv("PROMOTION_GUARD_CONFIG")
        if config_path_env:
            config_path = Path(config_path_env).expanduser()
        else:
            config_path = _DEFAULT_PROMOTION_GUARD_CONFIG
        if not config_path.exists():
            return None
        if yaml is None:
            self._logger.warning(
                "PyYAML unavailable; skipping promotion guard config at %s", config_path
            )
            return None
        try:
            raw_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - defensive guard for malformed YAML
            self._logger.warning(
                "Failed to load promotion guard config from %s: %s", config_path, exc
            )
            return None
        if not isinstance(raw_payload, Mapping):
            self._logger.warning(
                "Promotion guard config at %s must be a mapping", config_path
            )
            return None
        guard_payload = raw_payload.get("promotion_guard", raw_payload)
        if not isinstance(guard_payload, Mapping):
            self._logger.warning(
                "Promotion guard payload at %s missing 'promotion_guard' mapping", config_path
            )
            return None

        ledger_path = Path(
            str(guard_payload.get("ledger_path", _DEFAULT_LEDGER_PATH))
        ).expanduser()
        diary_path = Path(
            str(guard_payload.get("diary_path", _DEFAULT_DIARY_PATH))
        ).expanduser()

        stage_requirements_payload = guard_payload.get("stage_requirements")
        stage_requirements: MutableMapping[str, str | PolicyLedgerStage] | None = None
        if isinstance(stage_requirements_payload, Mapping):
            stage_requirements = {}
            for status_key, stage_value in stage_requirements_payload.items():
                key = str(status_key).strip()
                if not key:
                    continue
                try:
                    stage = PolicyLedgerStage.from_value(stage_value)
                except ValueError:
                    self._logger.warning(
                        "Skipping unknown stage requirement '%s' -> %r in %s",
                        key,
                        stage_value,
                        config_path,
                    )
                    continue
                stage_requirements[key] = stage

        regimes_payload = guard_payload.get("required_regimes")
        required_regimes: tuple[str, ...] | None = None
        if isinstance(regimes_payload, Iterable) and not isinstance(regimes_payload, (str, bytes)):
            cleaned: list[str] = []
            seen: dict[str, None] = {}
            for value in regimes_payload:
                text = str(value).strip().lower()
                if text and text not in seen:
                    seen[text] = None
                    cleaned.append(text)
            required_regimes = tuple(cleaned) if cleaned else None

        min_decisions_raw = guard_payload.get("min_decisions_per_regime")
        if min_decisions_raw is None:
            min_decisions_raw = guard_payload.get("min_decisions")
        try:
            min_decisions = int(min_decisions_raw) if min_decisions_raw is not None else 3
        except (TypeError, ValueError):
            min_decisions = 3

        gate_statuses_payload = guard_payload.get("regime_gate_statuses")
        regime_gate_statuses: tuple[str, ...] | None = None
        if isinstance(gate_statuses_payload, Iterable) and not isinstance(
            gate_statuses_payload, (str, bytes)
        ):
            cleaned_statuses = {
                str(value).strip().lower()
                for value in gate_statuses_payload
                if value and str(value).strip()
            }
            regime_gate_statuses = tuple(sorted(cleaned_statuses)) if cleaned_statuses else None

        guard_kwargs: dict[str, Any] = {
            "ledger_path": ledger_path,
            "diary_path": diary_path,
            "min_decisions_per_regime": max(0, min_decisions),
        }
        if stage_requirements:
            guard_kwargs["stage_requirements"] = stage_requirements
        if required_regimes:
            guard_kwargs["required_regimes"] = required_regimes
        if regime_gate_statuses:
            guard_kwargs["regime_gate_statuses"] = regime_gate_statuses

        return PromotionGuard(**guard_kwargs)

    def _json_loads(self, payload: str | None, context: str) -> Any | None:
        if not payload:
            return None
        try:
            return json.loads(payload)
        except json.JSONDecodeError as exc:
            raise StrategyRegistryError(f"{context} contains invalid JSON") from exc

    def _json_mapping(self, payload: str | None, context: str) -> dict[str, Any] | None:
        data = self._json_loads(payload, context)
        if data is None:
            return None
        if not isinstance(data, dict):
            raise StrategyRegistryError(f"{context} must be a JSON object")
        return cast(dict[str, Any], data)

    def _row_to_strategy(self, row: sqlite3.Row) -> dict[str, Any]:
        catalogue_metadata = self._json_mapping(row["catalogue_metadata"], "catalogue_metadata")
        catalogue_entry_metadata = self._json_mapping(
            row["catalogue_entry_metadata"], "catalogue_entry_metadata"
        )
        return {
            "id": row["id"],
            "genome_id": row["genome_id"],
            "created_at": row["created_at"],
            "dna": self._json_loads(row["dna"], "dna"),
            "fitness_report": self._json_mapping(row["fitness_report"], "fitness_report") or {},
            "status": row["status"],
            "strategy_name": row["strategy_name"],
            "generation": row["generation"],
            "fitness_score": row["fitness_score"],
            "max_drawdown": row["max_drawdown"],
            "sharpe_ratio": row["sharpe_ratio"],
            "total_return": row["total_return"],
            "volatility": row["volatility"],
            "seed_source": row["seed_source"],
            "catalogue_name": row["catalogue_name"],
            "catalogue_version": row["catalogue_version"],
            "catalogue_seeded_at": row["catalogue_seeded_at"],
            "catalogue_metadata": catalogue_metadata,
            "catalogue_entry_id": row["catalogue_entry_id"],
            "catalogue_entry_name": row["catalogue_entry_name"],
            "catalogue_entry_metadata": catalogue_entry_metadata,
        }

    def _normalise_status(
        self, status: StrategyStatus | str | None, *, default: StrategyStatus
    ) -> str:
        if status is None:
            return default.value
        if isinstance(status, StrategyStatus):
            return status.value
        normalised = str(status).strip().lower()
        if normalised in {member.value for member in StrategyStatus}:
            return normalised
        return default.value

    def register_champion(
        self,
        genome: Any,
        fitness_report: dict[str, Any],
        *,
        provenance: Mapping[str, Any] | None = None,
        status: StrategyStatus | str | None = None,
    ) -> bool:
        genome_id = str(getattr(genome, "id", genome))
        status_value = self._normalise_status(status, default=StrategyStatus.EVOLVED)
        status_enum: StrategyStatus | None
        try:
            status_enum = StrategyStatus(status_value)
        except ValueError:
            status_enum = None
        if self._promotion_guard is not None:
            try:
                self._promotion_guard.validate(genome_id, status_enum or status_value)
            except PromotionIntegrityError as exc:
                raise StrategyRegistryError(str(exc)) from exc

        try:
            dna_source = getattr(genome, "decision_tree", None)
            if dna_source is None:
                dna_source = str(genome)
            dna_json = json.dumps(dna_source, default=str)
        except (TypeError, ValueError) as exc:
            raise StrategyRegistryError("Unable to serialise genome DNA") from exc

        strategy_name = getattr(genome, "name", f"genome_{genome_id}")
        generation = getattr(genome, "generation", 0)
        fitness_score = fitness_report.get("fitness_score", 0.0)
        max_drawdown = fitness_report.get("max_drawdown", 0.0)
        sharpe_ratio = fitness_report.get("sharpe_ratio", 0.0)
        total_return = fitness_report.get("total_return", 0.0)
        volatility = fitness_report.get("volatility", 0.0)

        seed_source: str | None = None
        catalogue_name: str | None = None
        catalogue_version: str | None = None
        catalogue_seeded_at: float | None = None
        catalogue_metadata_json: str | None = None
        catalogue_entry_id: str | None = None
        catalogue_entry_name: str | None = None
        catalogue_entry_metadata_json: str | None = None

        report_payload = dict(fitness_report)
        metadata = report_payload.setdefault("metadata", {})
        if provenance:
            seed_source_value = provenance.get("seed_source")
            if seed_source_value:
                seed_source = str(seed_source_value)

            catalogue_payload = provenance.get("catalogue")
            if isinstance(catalogue_payload, Mapping):
                catalogue_name = str(catalogue_payload.get("name") or "") or None
                catalogue_version = str(catalogue_payload.get("version") or "") or None
                seeded_at_value = catalogue_payload.get("seeded_at")
                try:
                    catalogue_seeded_at = (
                        float(seeded_at_value) if seeded_at_value is not None else None
                    )
                except (TypeError, ValueError):
                    catalogue_seeded_at = None
                catalogue_metadata_json = json.dumps(dict(catalogue_payload), default=str)

            entry_payload = provenance.get("entry")
            if isinstance(entry_payload, Mapping):
                catalogue_entry_id = str(entry_payload.get("id") or "") or None
                catalogue_entry_name = str(entry_payload.get("name") or "") or None
                catalogue_entry_metadata_json = json.dumps(dict(entry_payload), default=str)

            if isinstance(metadata, dict):
                metadata.setdefault("catalogue_provenance", dict(provenance))
            else:
                report_payload["metadata"] = {"catalogue_provenance": dict(provenance)}

        report_json = json.dumps(report_payload, default=str)

        created_at = datetime.now(tz=UTC).isoformat()

        with self._managed_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO strategies
                (genome_id, created_at, dna, fitness_report, status, strategy_name,
                 generation, fitness_score, max_drawdown, sharpe_ratio, total_return, volatility,
                 seed_source, catalogue_name, catalogue_version, catalogue_seeded_at,
                 catalogue_metadata, catalogue_entry_id, catalogue_entry_name,
                 catalogue_entry_metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    genome_id,
                    created_at,
                    dna_json,
                    report_json,
                    status_value,
                    strategy_name,
                    generation,
                    fitness_score,
                    max_drawdown,
                    sharpe_ratio,
                    total_return,
                    volatility,
                    seed_source,
                    catalogue_name,
                    catalogue_version,
                    catalogue_seeded_at,
                    catalogue_metadata_json,
                    catalogue_entry_id,
                    catalogue_entry_name,
                    catalogue_entry_metadata_json,
                ),
            )

        self._logger.info("Registered champion genome: %s", genome_id)
        return True

    def get_strategy(self, strategy_id: str) -> dict[str, Any] | None:
        with self._managed_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM strategies WHERE genome_id = ?", (strategy_id,))
            row = cursor.fetchone()

        if row is None:
            return None
        return self._row_to_strategy(row)

    def update_strategy_status(
        self, strategy_id: str, new_status: StrategyStatus | str
    ) -> bool:
        status_value = self._normalise_status(new_status, default=StrategyStatus.EVOLVED)
        status_enum: StrategyStatus | None
        try:
            status_enum = StrategyStatus(status_value)
        except ValueError:
            status_enum = None
        if self._promotion_guard is not None:
            try:
                self._promotion_guard.validate(strategy_id, status_enum or status_value)
            except PromotionIntegrityError as exc:
                raise StrategyRegistryError(str(exc)) from exc
        with self._managed_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE strategies SET status = ? WHERE genome_id = ?",
                (status_value, strategy_id),
            )
            updated = cursor.rowcount > 0

        if updated:
            self._logger.info(
                "Updated strategy %s status to %s", strategy_id, status_value
            )
        else:
            self._logger.warning("Strategy %s not found", strategy_id)
        return updated

    def get_champion_strategies(self, limit: int = 10) -> list[dict[str, Any]]:
        with self._managed_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM strategies
                WHERE status IN (
                    'evolved',
                    'approved',
                    'approved_default',
                    'approved_fallback',
                    'active'
                )
                ORDER BY fitness_score DESC
                LIMIT ?
            """,
                (limit,),
            )
            rows = cursor.fetchall()

        return [self._row_to_strategy(row) for row in rows]

    def get_strategies_by_status(self, status: str) -> list[dict[str, Any]]:
        with self._managed_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM strategies WHERE status = ?", (status,))
            rows = cursor.fetchall()

        return [self._row_to_strategy(row) for row in rows]

    def get_registry_summary(self) -> dict[str, Any]:
        with self._managed_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    COUNT(*) as total_strategies,
                    COUNT(CASE WHEN status = 'evolved' THEN 1 END) as evolved_count,
                    COUNT(CASE WHEN status = 'approved' THEN 1 END) as approved_count,
                    COUNT(CASE WHEN status = 'active' THEN 1 END) as active_count,
                    AVG(fitness_score) as avg_fitness_score,
                    MAX(fitness_score) as max_fitness_score,
                    MIN(fitness_score) as min_fitness_score,
                    COUNT(CASE WHEN seed_source = 'catalogue' THEN 1 END) as catalogue_seeded,
                    COUNT(
                        CASE
                            WHEN seed_source = 'catalogue'
                                 AND (catalogue_entry_id IS NULL OR catalogue_entry_id = '')
                            THEN 1
                        END
                    ) as catalogue_missing_provenance,
                    COUNT(
                        CASE
                            WHEN catalogue_entry_id IS NOT NULL AND catalogue_entry_id <> ''
                            THEN 1
                        END
                    ) as catalogue_entry_count,
                    MAX(catalogue_seeded_at) as latest_catalogue_seeded_at
                FROM strategies
            """
            )
            summary_row = cursor.fetchone()
            if summary_row is None:
                raise StrategyRegistryError("Failed to read registry summary")

            cursor.execute(
                """
                SELECT seed_source, COUNT(*) as count
                FROM strategies
                GROUP BY seed_source
            """
            )
            seed_source_counts: dict[str, int] = {}
            for source_row in cursor.fetchall():
                key = source_row["seed_source"] or ""
                seed_source_counts[str(key)] = source_row["count"]

            cursor.execute(
                """
                SELECT catalogue_name, catalogue_version
                FROM strategies
                WHERE catalogue_name IS NOT NULL AND catalogue_name <> ''
                GROUP BY catalogue_name, catalogue_version
            """
            )
            catalogue_names: set[str] = set()
            catalogue_versions: set[str] = set()
            for catalogue_row in cursor.fetchall():
                name = catalogue_row["catalogue_name"]
                version = catalogue_row["catalogue_version"]
                if name:
                    catalogue_names.add(str(name))
                if version:
                    catalogue_versions.add(str(version))

            cursor.execute(
                """
                SELECT catalogue_metadata
                FROM strategies
                WHERE catalogue_metadata IS NOT NULL AND catalogue_metadata <> ''
                ORDER BY catalogue_seeded_at DESC
                LIMIT 1
            """
            )
            latest_catalogue_row = cursor.fetchone()

        summary: dict[str, Any] = {
            "total_strategies": summary_row["total_strategies"],
            "evolved_count": summary_row["evolved_count"],
            "approved_count": summary_row["approved_count"],
            "active_count": summary_row["active_count"],
            "avg_fitness_score": summary_row["avg_fitness_score"] or 0.0,
            "max_fitness_score": summary_row["max_fitness_score"] or 0.0,
            "min_fitness_score": summary_row["min_fitness_score"] or 0.0,
            "catalogue_seeded": summary_row["catalogue_seeded"],
            "catalogue_entry_count": summary_row["catalogue_entry_count"],
            "catalogue_missing_provenance": summary_row["catalogue_missing_provenance"],
            "latest_catalogue_seeded_at": summary_row["latest_catalogue_seeded_at"],
            "database_path": str(self.db_path),
            "seed_source_counts": seed_source_counts,
            "catalogue_names": sorted(catalogue_names),
            "catalogue_versions": sorted(catalogue_versions),
        }

        if latest_catalogue_row and latest_catalogue_row["catalogue_metadata"]:
            summary["latest_catalogue_metadata"] = self._json_mapping(
                latest_catalogue_row["catalogue_metadata"], "latest_catalogue_metadata"
            )

        return summary


__all__ = [
    "StrategyRegistry",
    "StrategyRegistryError",
    "StrategyStatus",
]
