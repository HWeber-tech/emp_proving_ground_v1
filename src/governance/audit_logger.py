"""
EMP Audit Logger v1.1

Audit logging for the governance layer.
Tracks all governance decisions and actions for compliance and transparency.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AuditLogger:
    """Audit logger for governance layer actions."""

    def __init__(self, log_file: Optional[str] = None) -> None:
        self.log_file = log_file or "data/audit_log.jsonl"
        self.log_path = Path(self.log_file)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Audit Logger initialized with log file: {self.log_file}")

    def log_decision(
        self,
        decision_type: str,
        strategy_id: str,
        genome_id: str,
        approver: Optional[str] = None,
        reason: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Log a governance decision."""
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "governance_decision",
            "decision_type": decision_type,
            "strategy_id": strategy_id,
            "genome_id": genome_id,
            "approver": approver,
            "reason": reason,
            "metadata": metadata or {},
        }

        self._write_log_entry(audit_entry)
        logger.info(f"Governance decision logged: {decision_type} for {strategy_id}")

    def log_fitness_evaluation(
        self,
        genome_id: str,
        strategy_id: str,
        fitness_score: float,
        generation: int,
        performance_metrics: dict[str, Any],
        risk_metrics: dict[str, Any],
    ) -> None:
        """Log a fitness evaluation."""
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "fitness_evaluation",
            "genome_id": genome_id,
            "strategy_id": strategy_id,
            "fitness_score": fitness_score,
            "generation": generation,
            "performance_metrics": performance_metrics,
            "risk_metrics": risk_metrics,
        }

        self._write_log_entry(audit_entry)
        logger.debug(f"Fitness evaluation logged: {fitness_score:.4f} for {genome_id}")

    def log_strategy_registration(
        self,
        strategy_id: str,
        genome_id: str,
        registration_type: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Log a strategy registration event."""
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "strategy_registration",
            "strategy_id": strategy_id,
            "genome_id": genome_id,
            "registration_type": registration_type,
            "metadata": metadata or {},
        }

        self._write_log_entry(audit_entry)
        logger.info(f"Strategy registration logged: {registration_type} for {strategy_id}")

    def log_approval_workflow(
        self,
        workflow_id: str,
        strategy_id: str,
        workflow_type: str,
        status: str,
        approver: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Log an approval workflow event."""
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "approval_workflow",
            "workflow_id": workflow_id,
            "strategy_id": strategy_id,
            "workflow_type": workflow_type,
            "status": status,
            "approver": approver,
            "metadata": metadata or {},
        }

        self._write_log_entry(audit_entry)
        logger.info(f"Approval workflow logged: {workflow_type} - {status} for {strategy_id}")

    def log_system_event(
        self,
        event_type: str,
        component: str,
        severity: str,
        message: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Log a system event."""
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "system_event",
            "system_event_type": event_type,
            "component": component,
            "severity": severity,
            "message": message,
            "metadata": metadata or {},
        }

        self._write_log_entry(audit_entry)

        # Log to console based on severity
        if severity == "critical":
            logger.critical(f"System event: {message}")
        elif severity == "error":
            logger.error(f"System event: {message}")
        elif severity == "warning":
            logger.warning(f"System event: {message}")
        else:
            logger.info(f"System event: {message}")

    def log_compliance_check(
        self,
        check_type: str,
        strategy_id: str,
        passed: bool,
        violations: list[str],
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Log a compliance check."""
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "compliance_check",
            "check_type": check_type,
            "strategy_id": strategy_id,
            "passed": passed,
            "violations": violations,
            "metadata": metadata or {},
        }

        self._write_log_entry(audit_entry)

        if passed:
            logger.info(f"Compliance check passed: {check_type} for {strategy_id}")
        else:
            logger.warning(
                f"Compliance check failed: {check_type} for {strategy_id} - violations: {violations}"
            )

    def _write_log_entry(self, entry: dict[str, Any]) -> None:
        """Write a log entry to the audit log file."""
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Error writing to audit log: {e}")

    def get_audit_history(
        self,
        strategy_id: Optional[str] = None,
        event_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Get audit history with optional filtering."""
        try:
            if not self.log_path.exists():
                return []

            entries = []
            with open(self.log_path, "r") as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())

                        # Apply filters
                        if strategy_id and entry.get("strategy_id") != strategy_id:
                            continue
                        if event_type and entry.get("event_type") != event_type:
                            continue
                        if start_time:
                            entry_time = datetime.fromisoformat(entry["timestamp"])
                            if entry_time < start_time:
                                continue
                        if end_time:
                            entry_time = datetime.fromisoformat(entry["timestamp"])
                            if entry_time > end_time:
                                continue

                        entries.append(entry)

                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON in audit log: {line.strip()}")
                        continue

            # Apply limit
            if limit:
                entries = entries[-limit:]

            return entries

        except Exception as e:
            logger.error(f"Error reading audit log: {e}")
            return []

    def get_audit_statistics(self) -> dict[str, Any]:
        """Get audit log statistics."""
        try:
            if not self.log_path.exists():
                return {"total_entries": 0, "event_types": {}, "strategies": {}, "date_range": None}

            entries = []
            with open(self.log_path, "r") as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        entries.append(entry)
                    except json.JSONDecodeError:
                        continue

            if not entries:
                return {"total_entries": 0, "event_types": {}, "strategies": {}, "date_range": None}

            # Calculate statistics
            event_types: dict[str, int] = {}
            strategies: dict[str, int] = {}
            timestamps = []

            for entry in entries:
                event_type = entry.get("event_type", "unknown")
                event_types[event_type] = event_types.get(event_type, 0) + 1

                strategy_id = entry.get("strategy_id")
                if strategy_id:
                    strategies[strategy_id] = strategies.get(strategy_id, 0) + 1

                timestamps.append(datetime.fromisoformat(entry["timestamp"]))

            # Calculate date range
            if timestamps:
                start_time = min(timestamps)
                end_time = max(timestamps)
                date_range = {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                    "duration_days": (end_time - start_time).days,
                }
            else:
                date_range = None

            return {
                "total_entries": len(entries),
                "event_types": event_types,
                "strategies": strategies,
                "date_range": date_range,
            }

        except Exception as e:
            logger.error(f"Error calculating audit statistics: {e}")
            return {
                "total_entries": 0,
                "event_types": {},
                "strategies": {},
                "date_range": None,
                "error": str(e),
            }

    def clear_audit_log(self) -> None:
        """Clear the audit log file."""
        try:
            if self.log_path.exists():
                self.log_path.unlink()
            logger.info("Audit log cleared")
        except Exception as e:
            logger.error(f"Error clearing audit log: {e}")

    def export_audit_log(self, export_file: str) -> None:
        """Export audit log to a different file."""
        try:
            if not self.log_path.exists():
                logger.warning("No audit log to export")
                return

            import shutil

            shutil.copy2(self.log_path, export_file)
            logger.info(f"Audit log exported to: {export_file}")
        except Exception as e:
            logger.error(f"Error exporting audit log: {e}")
