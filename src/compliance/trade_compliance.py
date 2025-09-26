"""Trade compliance monitor that enforces policy thresholds and emits telemetry."""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from types import SimpleNamespace
from typing import Any, Deque, Iterable, Mapping, MutableMapping, Protocol
from uuid import uuid4

from src.core.event_bus import Event, EventBus, SubscriptionHandle, get_global_bus

logger = logging.getLogger(__name__)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _as_int(value: Any, default: int) -> int:
    try:
        return int(str(value))
    except Exception:
        return default


def _as_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    try:
        text = str(value).strip()
    except Exception:
        return default
    return text or default


def _parse_csv(value: Any) -> frozenset[str]:
    if value is None:
        return frozenset()
    if isinstance(value, (set, frozenset)):
        return frozenset(str(item).strip().upper() for item in value if str(item).strip())
    items: Iterable[str]
    if isinstance(value, (list, tuple)):
        items = [str(item) for item in value]
    else:
        items = str(value).replace(";", ",").split(",")
    return frozenset(symbol.strip().upper() for symbol in items if symbol.strip())


def _parse_allowed_sides(value: Any) -> frozenset[str]:
    sides = _parse_csv(value)
    if not sides:
        return frozenset({"BUY", "SELL"})
    return frozenset(side.upper() for side in sides)


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=UTC)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return datetime.now(tz=UTC)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    return datetime.now(tz=UTC)


@dataclass(frozen=True)
class TradeCompliancePolicy:
    """Policy thresholds that the trade compliance monitor enforces."""

    policy_name: str = "default"
    max_single_trade_notional: float = 500_000.0
    max_daily_symbol_notional: float = 2_000_000.0
    max_trades_per_symbol_per_day: int = 200
    restricted_symbols: frozenset[str] = field(default_factory=frozenset)
    allowed_sides: frozenset[str] = field(default_factory=lambda: frozenset({"BUY", "SELL"}))
    report_channel: str = "telemetry.compliance.trade"

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "TradeCompliancePolicy":
        mapping = mapping or {}

        name = _as_str(mapping.get("COMPLIANCE_POLICY_NAME"), default="default")
        single_notional = _as_float(
            mapping.get("COMPLIANCE_MAX_SINGLE_NOTIONAL"),
            default=cls.max_single_trade_notional,
        )
        daily_notional = _as_float(
            mapping.get("COMPLIANCE_MAX_DAILY_NOTIONAL"),
            default=cls.max_daily_symbol_notional,
        )
        trades_per_symbol = _as_int(
            mapping.get("COMPLIANCE_MAX_TRADES_PER_SYMBOL"),
            default=cls.max_trades_per_symbol_per_day,
        )
        restricted = _parse_csv(mapping.get("COMPLIANCE_RESTRICTED_SYMBOLS"))
        allowed = _parse_allowed_sides(mapping.get("COMPLIANCE_ALLOWED_SIDES"))
        channel = (
            _as_str(
                mapping.get("COMPLIANCE_REPORT_CHANNEL"),
                default=cls.report_channel,
            )
            or cls.report_channel
        )

        return cls(
            policy_name=name or "default",
            max_single_trade_notional=single_notional,
            max_daily_symbol_notional=daily_notional,
            max_trades_per_symbol_per_day=trades_per_symbol,
            restricted_symbols=restricted,
            allowed_sides=allowed,
            report_channel=channel,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "max_single_trade_notional": self.max_single_trade_notional,
            "max_daily_symbol_notional": self.max_daily_symbol_notional,
            "max_trades_per_symbol_per_day": self.max_trades_per_symbol_per_day,
            "restricted_symbols": sorted(self.restricted_symbols),
            "allowed_sides": sorted(self.allowed_sides),
            "report_channel": self.report_channel,
        }


@dataclass(frozen=True)
class ComplianceCheckResult:
    """Outcome of a single compliance rule evaluation."""

    rule_id: str
    name: str
    passed: bool
    severity: str
    message: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "rule_id": self.rule_id,
            "name": self.name,
            "passed": self.passed,
            "severity": self.severity,
            "message": self.message,
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class TradeComplianceSnapshot:
    """Compliance snapshot generated for a single execution report."""

    trade_id: str
    intent_id: str | None
    symbol: str
    side: str
    quantity: float
    price: float
    notional: float
    timestamp: datetime
    status: str
    checks: tuple[ComplianceCheckResult, ...]
    totals: Mapping[str, Any]
    policy_name: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "intent_id": self.intent_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "price": self.price,
            "notional": self.notional,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status,
            "checks": [check.as_dict() for check in self.checks],
            "totals": dict(self.totals),
            "policy_name": self.policy_name,
        }


class ComplianceSnapshotJournal(Protocol):
    """Interface for persistence layers that store compliance snapshots."""

    def record_snapshot(
        self, snapshot: Mapping[str, Any], *, strategy_id: str
    ) -> Mapping[str, Any] | None:
        """Persist a compliance snapshot and return a serialisable summary."""

    def fetch_recent(
        self, *, limit: int = 5
    ) -> Iterable[Mapping[str, Any]]:  # pragma: no cover - protocol
        """Return recent journal entries if supported."""

    def close(self) -> None:  # pragma: no cover - protocol
        """Release any underlying resources."""


class TradeComplianceMonitor:
    """Monitors execution reports, enforces policy thresholds, and emits telemetry."""

    def __init__(
        self,
        *,
        event_bus: EventBus,
        policy: TradeCompliancePolicy | None = None,
        audit_logger: Any | None = None,
        strategy_id: str | None = None,
        snapshot_journal: ComplianceSnapshotJournal | None = None,
    ) -> None:
        self.event_bus = event_bus
        self.policy = policy or TradeCompliancePolicy()
        self.audit_logger = audit_logger
        self.strategy_id = strategy_id or "unknown"
        self.snapshot_journal = snapshot_journal

        self._daily_notional: MutableMapping[str, float] = defaultdict(float)
        self._daily_trades: MutableMapping[str, int] = defaultdict(int)
        self._current_day: date = datetime.now(tz=UTC).date()
        self._history: Deque[TradeComplianceSnapshot] = deque(maxlen=20)
        self.last_snapshot: TradeComplianceSnapshot | None = None
        self._last_journal_entry: Mapping[str, Any] | None = None

        self._subscription: SubscriptionHandle | None = None
        try:
            self._subscription = self.event_bus.subscribe("execution.report", self._handle_event)
        except Exception:  # pragma: no cover - defensive guard
            logger.exception("Failed to subscribe trade compliance monitor to execution.report")

    async def _handle_event(self, event: Event) -> None:
        await self.on_execution_report(event.payload)

    async def on_execution_report(self, payload: Any) -> None:
        report = self._normalise_payload(payload)
        if report is None:
            return

        snapshot = self._evaluate_report(report)
        self.last_snapshot = snapshot
        self._history.append(snapshot)
        self._emit_snapshot(snapshot)
        self._record_audit(snapshot)
        self._record_journal(snapshot)

    def _normalise_payload(self, payload: Any) -> Mapping[str, Any] | None:
        if payload is None:
            return None
        if isinstance(payload, Mapping):
            return payload
        if hasattr(payload, "__dict"):
            return payload.__dict__  # pragma: no cover - SimpleNamespace path
        if hasattr(payload, "__dict__"):
            return dict(payload.__dict__)
        if isinstance(payload, SimpleNamespace):  # pragma: no cover - compat
            return payload.__dict__
        return None

    def _evaluate_report(self, report: Mapping[str, Any]) -> TradeComplianceSnapshot:
        timestamp = _parse_timestamp(report.get("timestamp"))
        self._reset_if_new_day(timestamp.date())

        quantity = _as_float(report.get("quantity"), default=0.0)
        price = _as_float(report.get("price"), default=0.0)
        notional = abs(quantity * price)
        symbol = _as_str(report.get("symbol"), default="UNKNOWN").upper() or "UNKNOWN"
        side = _as_str(report.get("side"), default="BUY").upper() or "BUY"
        trade_id = _as_str(
            report.get("event_id") or report.get("order_id") or report.get("execution_id"),
            default=f"trade-{uuid4()}",
        )
        intent_id = report.get("trade_intent_id")
        status = _as_str(report.get("status"), default="unknown").upper()

        checks: list[ComplianceCheckResult] = [
            ComplianceCheckResult(
                rule_id="trade_recorded",
                name="Execution report recorded",
                passed=True,
                severity="info",
                message="Execution report processed",
                metadata={
                    "status": status,
                    "source": report.get("source"),
                },
            )
        ]

        symbol_notional = self._daily_notional[symbol]
        symbol_trades = self._daily_trades[symbol]

        valid_trade = quantity > 0 and price > 0
        if valid_trade:
            symbol_notional += notional
            symbol_trades += 1
            self._daily_notional[symbol] = symbol_notional
            self._daily_trades[symbol] = symbol_trades

        if quantity <= 0:
            checks.append(
                ComplianceCheckResult(
                    rule_id="non_positive_quantity",
                    name="Quantity must be positive",
                    passed=False,
                    severity="critical",
                    message="Execution quantity must be greater than zero",
                    metadata={"quantity": quantity},
                )
            )

        if price <= 0:
            checks.append(
                ComplianceCheckResult(
                    rule_id="non_positive_price",
                    name="Price must be positive",
                    passed=False,
                    severity="critical",
                    message="Execution price must be greater than zero",
                    metadata={"price": price},
                )
            )

        if symbol in self.policy.restricted_symbols:
            checks.append(
                ComplianceCheckResult(
                    rule_id="restricted_symbol",
                    name="Restricted instrument",
                    passed=False,
                    severity="critical",
                    message="Symbol is currently restricted for trading",
                    metadata={"symbol": symbol},
                )
            )

        if side not in self.policy.allowed_sides:
            checks.append(
                ComplianceCheckResult(
                    rule_id="side_not_allowed",
                    name="Unsupported side",
                    passed=False,
                    severity="critical",
                    message="Side is not permitted under the active policy",
                    metadata={"side": side},
                )
            )

        if (
            valid_trade
            and self.policy.max_single_trade_notional > 0
            and notional > self.policy.max_single_trade_notional
        ):
            checks.append(
                ComplianceCheckResult(
                    rule_id="single_trade_notional",
                    name="Single-trade notional limit",
                    passed=False,
                    severity="critical",
                    message="Trade notional exceeds the configured single-trade limit",
                    metadata={
                        "limit": self.policy.max_single_trade_notional,
                        "observed": notional,
                    },
                )
            )

        if (
            valid_trade
            and self.policy.max_daily_symbol_notional > 0
            and symbol_notional > self.policy.max_daily_symbol_notional
        ):
            checks.append(
                ComplianceCheckResult(
                    rule_id="daily_symbol_notional",
                    name="Daily symbol notional limit",
                    passed=False,
                    severity="critical",
                    message="Cumulative daily notional exceeds the configured limit",
                    metadata={
                        "limit": self.policy.max_daily_symbol_notional,
                        "observed": symbol_notional,
                        "trades": symbol_trades,
                    },
                )
            )

        if (
            valid_trade
            and self.policy.max_trades_per_symbol_per_day > 0
            and symbol_trades > self.policy.max_trades_per_symbol_per_day
        ):
            checks.append(
                ComplianceCheckResult(
                    rule_id="daily_symbol_trade_count",
                    name="Daily symbol trade count",
                    passed=False,
                    severity="warning",
                    message="Trade count for symbol exceeded policy threshold",
                    metadata={
                        "limit": self.policy.max_trades_per_symbol_per_day,
                        "observed": symbol_trades,
                    },
                )
            )

        has_critical = any(not check.passed and check.severity == "critical" for check in checks)
        has_warning = any(not check.passed and check.severity != "critical" for check in checks)
        status_label = "fail" if has_critical else ("warn" if has_warning else "pass")

        totals = {
            "symbol_daily_notional": symbol_notional,
            "symbol_daily_trades": symbol_trades,
        }

        return TradeComplianceSnapshot(
            trade_id=trade_id,
            intent_id=_as_str(intent_id) if intent_id is not None else None,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            notional=notional,
            timestamp=timestamp,
            status=status_label,
            checks=tuple(checks),
            totals=totals,
            policy_name=self.policy.policy_name,
        )

    def _emit_snapshot(self, snapshot: TradeComplianceSnapshot) -> None:
        payload = snapshot.as_dict()
        event = Event(
            type=self.policy.report_channel,
            payload=payload,
            source="trade_compliance_monitor",
        )

        published = False
        publish_from_sync = getattr(self.event_bus, "publish_from_sync", None)
        if callable(publish_from_sync):
            try:
                published = publish_from_sync(event) is not None
            except Exception:  # pragma: no cover - defensive logging
                logger.debug("Trade compliance publish_from_sync failed", exc_info=True)

        if not published:
            try:
                topic_bus = get_global_bus()
                topic_bus.publish_sync(
                    self.policy.report_channel,
                    payload,
                    source="trade_compliance_monitor",
                )
            except Exception:  # pragma: no cover - background bus optional
                logger.debug("Trade compliance telemetry publish failed", exc_info=True)

    def _record_audit(self, snapshot: TradeComplianceSnapshot) -> None:
        if self.audit_logger is None:
            return

        violations = [check.message for check in snapshot.checks if not check.passed]

        metadata = {
            "trade_id": snapshot.trade_id,
            "symbol": snapshot.symbol,
            "side": snapshot.side,
            "status": snapshot.status,
            "totals": dict(snapshot.totals),
            "checks": [check.as_dict() for check in snapshot.checks],
            "policy": self.policy.as_dict(),
        }

        try:
            self.audit_logger.log_compliance_check(
                check_type="trade_compliance",
                strategy_id=self.strategy_id,
                passed=not violations,
                violations=violations,
                metadata=metadata,
            )
        except Exception:  # pragma: no cover - audit logging should not break runtime
            logger.debug("Failed to record compliance audit entry", exc_info=True)

    def _record_journal(self, snapshot: TradeComplianceSnapshot) -> None:
        journal = self.snapshot_journal
        if journal is None:
            return

        try:
            entry = journal.record_snapshot(snapshot.as_dict(), strategy_id=self.strategy_id)
        except Exception:  # pragma: no cover - journal should not break runtime
            logger.debug("Failed to persist compliance snapshot", exc_info=True)
            return

        if entry is not None:
            try:
                self._last_journal_entry = dict(entry)
            except Exception:  # pragma: no cover - defensive copy
                self._last_journal_entry = None

    def _reset_if_new_day(self, event_day: date) -> None:
        if event_day == self._current_day:
            return
        self._current_day = event_day
        self._daily_notional.clear()
        self._daily_trades.clear()

    def summary(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "policy": self.policy.as_dict(),
            "last_snapshot": self.last_snapshot.as_dict() if self.last_snapshot else None,
            "history": [snapshot.as_dict() for snapshot in self._history],
            "daily_totals": {
                symbol: {
                    "notional": notional,
                    "trades": self._daily_trades.get(symbol, 0),
                }
                for symbol, notional in sorted(self._daily_notional.items())
            },
        }

        if self._last_journal_entry is not None:
            payload["journal"] = {"last_entry": dict(self._last_journal_entry)}
        elif self.snapshot_journal is not None:
            try:
                recent = [dict(entry) for entry in self.snapshot_journal.fetch_recent(limit=5)]
                if recent:
                    payload["journal"] = {"recent_entries": recent}
            except Exception:  # pragma: no cover - journal fetch should not break summary
                logger.debug("Failed to fetch compliance journal summary", exc_info=True)
        return payload

    def close(self) -> None:
        handle = self._subscription
        if handle is None:
            return
        try:
            self.event_bus.unsubscribe(handle)
        except Exception:  # pragma: no cover - defensive logging
            logger.debug("Failed to unsubscribe trade compliance monitor", exc_info=True)
        self._subscription = None

        if self.snapshot_journal is not None:
            try:
                self.snapshot_journal.close()
            except Exception:  # pragma: no cover - defensive cleanup
                logger.debug("Failed to close compliance snapshot journal", exc_info=True)


__all__ = [
    "ComplianceCheckResult",
    "TradeComplianceMonitor",
    "TradeCompliancePolicy",
    "TradeComplianceSnapshot",
]
