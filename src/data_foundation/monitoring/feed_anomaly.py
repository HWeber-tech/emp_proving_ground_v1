"""Data feed anomaly detection aligned with the high-impact roadmap.

The Phase 3 roadmap requires anomaly detection for data-feed breaks and
erroneous ticks.  This module keeps the implementation light-weight while
providing deterministic helpers that other layers (operations dashboards,
CI checks, CLI tooling) can reuse without needing external services.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from statistics import median
from typing import Iterable, Mapping, Sequence

__all__ = [
    "Tick",
    "FeedGap",
    "FalseTick",
    "DroppedTick",
    "FeedHealthStatus",
    "FeedAnomalyConfig",
    "FeedAnomalyReport",
    "analyse_feed",
]


@dataclass(frozen=True, slots=True)
class Tick:
    """Minimal tick structure used by the detector."""

    timestamp: datetime
    price: float
    volume: float | None = None
    seqno: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp, datetime):  # pragma: no cover - defensive
            raise TypeError("timestamp must be a datetime instance")
        if self.seqno is not None and not isinstance(self.seqno, int):  # pragma: no cover - defensive
            raise TypeError("seqno must be an int when provided")


@dataclass(frozen=True, slots=True)
class FeedGap:
    """Represents a gap in the incoming market data feed."""

    start: datetime
    end: datetime
    duration_seconds: float
    estimated_missing_ticks: int

    def as_dict(self) -> Mapping[str, object]:
        return {
            "start": self.start.astimezone(UTC).isoformat(),
            "end": self.end.astimezone(UTC).isoformat(),
            "duration_seconds": self.duration_seconds,
            "estimated_missing_ticks": self.estimated_missing_ticks,
        }


@dataclass(frozen=True, slots=True)
class FalseTick:
    """Represents a potential erroneous tick."""

    timestamp: datetime
    price: float
    previous_price: float
    deviation_pct: float
    volume: float | None

    def as_dict(self) -> Mapping[str, object]:
        return {
            "timestamp": self.timestamp.astimezone(UTC).isoformat(),
            "price": self.price,
            "previous_price": self.previous_price,
            "deviation_pct": self.deviation_pct,
            "volume": self.volume,
        }


@dataclass(frozen=True, slots=True)
class DroppedTick:
    """Captured metadata for ticks removed during normalisation."""

    timestamp: datetime
    seqno: int | None
    reason_code: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "timestamp": self.timestamp.astimezone(UTC).isoformat(),
            "seqno": self.seqno,
            "reason_code": self.reason_code,
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


class FeedHealthStatus(StrEnum):
    """Severity levels surfaced to operators."""

    ok = "ok"
    warn = "warn"
    fail = "fail"


@dataclass(frozen=True, slots=True)
class FeedAnomalyConfig:
    """Configuration controlling anomaly detection thresholds."""

    max_gap_seconds: float = 90.0
    fail_gap_multiplier: float = 3.0
    price_jump_pct: float = 5.0
    false_tick_limit: int = 2
    min_samples_for_ok: int = 5
    lookback_limit: int = 5000
    stale_grace_seconds: float = 120.0
    min_false_tick_reversion_pct: float = 2.5

    def __post_init__(self) -> None:  # pragma: no cover - defensive
        if self.max_gap_seconds <= 0:
            raise ValueError("max_gap_seconds must be positive")
        if self.fail_gap_multiplier < 1:
            raise ValueError("fail_gap_multiplier must be >= 1")
        if self.price_jump_pct <= 0:
            raise ValueError("price_jump_pct must be positive")
        if self.false_tick_limit < 0:
            raise ValueError("false_tick_limit must be >= 0")
        if self.min_samples_for_ok < 1:
            raise ValueError("min_samples_for_ok must be >= 1")
        if self.lookback_limit < 1:
            raise ValueError("lookback_limit must be >= 1")
        if self.stale_grace_seconds <= 0:
            raise ValueError("stale_grace_seconds must be positive")
        if self.min_false_tick_reversion_pct < 0:
            raise ValueError("min_false_tick_reversion_pct must be >= 0")


@dataclass(slots=True)
class FeedAnomalyReport:
    """Summary of detected anomalies for a symbol."""

    symbol: str
    generated_at: datetime
    status: FeedHealthStatus
    sample_count: int
    lookback_seconds: float | None
    median_interval_seconds: float | None
    max_gap_seconds: float | None
    stale: bool
    gaps: tuple[FeedGap, ...] = ()
    false_ticks: tuple[FalseTick, ...] = ()
    dropped_ticks: tuple[DroppedTick, ...] = ()
    issues: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "generated_at": self.generated_at.astimezone(UTC).isoformat(),
            "status": self.status.value,
            "sample_count": self.sample_count,
            "lookback_seconds": self.lookback_seconds,
            "median_interval_seconds": self.median_interval_seconds,
            "max_gap_seconds": self.max_gap_seconds,
            "stale": self.stale,
            "gaps": [gap.as_dict() for gap in self.gaps],
            "false_ticks": [tick.as_dict() for tick in self.false_ticks],
            "dropped_ticks": [tick.as_dict() for tick in self.dropped_ticks],
            "issues": list(self.issues),
            "metadata": dict(self.metadata),
        }

    def to_markdown(self) -> str:
        lines = [
            f"**Feed health** — {self.symbol}",
            f"- Status: {self.status.value}",
            f"- Generated: {self.generated_at.astimezone(UTC).isoformat()}",
            f"- Samples: {self.sample_count}",
        ]
        if self.median_interval_seconds is not None:
            lines.append(f"- Median interval: {self.median_interval_seconds:.2f}s")
        if self.max_gap_seconds is not None:
            lines.append(f"- Largest gap: {self.max_gap_seconds:.2f}s")
        if self.stale:
            lines.append("- ⚠️ Feed is stale")
        if self.issues:
            lines.append("\n**Issues:**")
            for issue in self.issues:
                lines.append(f"- {issue}")
        if self.gaps:
            lines.append("\n**Detected gaps:**")
            for gap in self.gaps:
                lines.append(
                    "- {start} → {end} ({duration:.1f}s, ~{missing} ticks)".format(
                        start=gap.start.astimezone(UTC).isoformat(),
                        end=gap.end.astimezone(UTC).isoformat(),
                        duration=gap.duration_seconds,
                        missing=gap.estimated_missing_ticks,
                    )
                )
        if self.false_ticks:
            lines.append("\n**Suspect ticks:**")
            for tick in self.false_ticks:
                lines.append(
                    "- {timestamp} price {price:.4f} dev {dev:.2f}% (prev {prev:.4f})".format(
                        timestamp=tick.timestamp.astimezone(UTC).isoformat(),
                        price=tick.price,
                        dev=tick.deviation_pct,
                        prev=tick.previous_price,
                    )
                )
        if self.dropped_ticks:
            lines.append("\n**Dropped ticks:**")
            for tick in self.dropped_ticks:
                lines.append(
                    "- {timestamp} seqno={seqno} reason={reason}".format(
                        timestamp=tick.timestamp.astimezone(UTC).isoformat(),
                        seqno=tick.seqno if tick.seqno is not None else "?",
                        reason=tick.reason_code,
                    )
                )
        return "\n".join(lines)


def _normalise_ticks(
    ticks: Sequence[Tick] | Iterable[Tick], limit: int
) -> tuple[list[Tick], list[DroppedTick]]:
    if isinstance(ticks, Sequence):
        selected = list(ticks)[-limit:]
    else:
        selected = list(ticks)
        if len(selected) > limit:
            selected = selected[-limit:]

    ordered: list[Tick] = []
    dropped: list[DroppedTick] = []
    last_ts: datetime | None = None
    last_seqno: int | None = None

    for tick in sorted(
        selected, key=lambda item: (item.timestamp, item.seqno if item.seqno is not None else -1)
    ):
        if last_ts is not None and tick.timestamp < last_ts:
            dropped.append(
                DroppedTick(
                    timestamp=tick.timestamp,
                    seqno=tick.seqno,
                    reason_code="out_of_order_timestamp",
                    metadata={"previous_timestamp": last_ts.isoformat()},
                )
            )
            continue
        if (
            last_ts is not None
            and tick.timestamp == last_ts
            and tick.seqno is not None
            and last_seqno is not None
            and tick.seqno <= last_seqno
        ):
            dropped.append(
                DroppedTick(
                    timestamp=tick.timestamp,
                    seqno=tick.seqno,
                    reason_code="out_of_order_seqno",
                    metadata={"previous_seqno": last_seqno},
                )
            )
            continue

        ordered.append(tick)
        last_ts = tick.timestamp
        last_seqno = tick.seqno if tick.seqno is not None else last_seqno

    return ordered, dropped


def _estimate_median_interval(ticks: Sequence[Tick]) -> float | None:
    if len(ticks) < 2:
        return None
    deltas = [
        (current.timestamp - previous.timestamp).total_seconds()
        for previous, current in zip(ticks, ticks[1:])
    ]
    deltas = [delta for delta in deltas if delta > 0]
    if not deltas:
        return None
    return float(median(deltas))


def _estimate_missing_ticks(delta: float, median_interval: float | None) -> int:
    if median_interval and median_interval > 0:
        missing = max(int(round(delta / median_interval)) - 1, 0)
        return missing
    return 0


def _detect_gaps(
    ticks: Sequence[Tick],
    *,
    config: FeedAnomalyConfig,
    median_interval: float | None,
) -> tuple[list[FeedGap], float | None]:
    largest_gap: float | None = None
    gaps: list[FeedGap] = []
    for previous, current in zip(ticks, ticks[1:]):
        delta = (current.timestamp - previous.timestamp).total_seconds()
        if delta > config.max_gap_seconds:
            largest_gap = max(largest_gap or 0.0, delta)
            gaps.append(
                FeedGap(
                    start=previous.timestamp,
                    end=current.timestamp,
                    duration_seconds=delta,
                    estimated_missing_ticks=_estimate_missing_ticks(delta, median_interval),
                )
            )
    return gaps, largest_gap


def _detect_false_ticks(
    ticks: Sequence[Tick],
    *,
    config: FeedAnomalyConfig,
) -> list[FalseTick]:
    suspects: list[FalseTick] = []
    for index in range(1, len(ticks)):
        previous = ticks[index - 1]
        current = ticks[index]
        if previous.price <= 0 or current.price <= 0:
            continue
        change = abs(current.price - previous.price)
        deviation_pct = (change / previous.price) * 100
        if deviation_pct < config.price_jump_pct:
            continue
        reverted = False
        if index + 1 < len(ticks):
            next_tick = ticks[index + 1]
            if previous.price > 0:
                reversion_pct = abs(next_tick.price - previous.price) / previous.price * 100
                if reversion_pct <= config.min_false_tick_reversion_pct:
                    reverted = True
        if reverted:
            suspects.append(
                FalseTick(
                    timestamp=current.timestamp,
                    price=current.price,
                    previous_price=previous.price,
                    deviation_pct=round(deviation_pct, 4),
                    volume=current.volume,
                )
            )
    return suspects


def analyse_feed(
    symbol: str,
    ticks: Sequence[Tick] | Iterable[Tick],
    *,
    config: FeedAnomalyConfig | None = None,
    now: datetime | None = None,
) -> FeedAnomalyReport:
    """Analyse tick history and surface feed anomalies."""

    resolved_config = config or FeedAnomalyConfig()
    sample_ticks, dropped_ticks = _normalise_ticks(ticks, resolved_config.lookback_limit)
    generated_at = now or datetime.now(tz=UTC)
    status = FeedHealthStatus.ok
    issues: list[str] = []

    if not sample_ticks:
        issues.append("No ticks available for analysis")
        return FeedAnomalyReport(
            symbol=symbol,
            generated_at=generated_at,
            status=FeedHealthStatus.fail,
            sample_count=0,
            lookback_seconds=None,
            median_interval_seconds=None,
            max_gap_seconds=None,
            stale=True,
            dropped_ticks=tuple(dropped_ticks),
            issues=tuple(issues),
        )

    median_interval = _estimate_median_interval(sample_ticks)
    lookback_seconds = (
        sample_ticks[-1].timestamp - sample_ticks[0].timestamp
    ).total_seconds() if len(sample_ticks) > 1 else None

    gaps, largest_gap = _detect_gaps(
        sample_ticks,
        config=resolved_config,
        median_interval=median_interval,
    )

    false_ticks = _detect_false_ticks(sample_ticks, config=resolved_config)

    def _escalate(candidate: FeedHealthStatus) -> None:
        nonlocal status
        if status is FeedHealthStatus.fail:
            return
        if candidate is FeedHealthStatus.fail:
            status = FeedHealthStatus.fail
        elif status is FeedHealthStatus.ok and candidate is FeedHealthStatus.warn:
            status = FeedHealthStatus.warn

    if len(sample_ticks) < resolved_config.min_samples_for_ok:
        issues.append(
            f"Only {len(sample_ticks)} ticks available (< {resolved_config.min_samples_for_ok})"
        )
        _escalate(FeedHealthStatus.warn)

    if gaps:
        gap_issue = f"Detected {len(gaps)} feed gaps"
        if largest_gap:
            gap_issue += f"; largest {largest_gap:.1f}s"
        issues.append(gap_issue)
        if largest_gap and largest_gap > resolved_config.max_gap_seconds * resolved_config.fail_gap_multiplier:
            _escalate(FeedHealthStatus.fail)
        else:
            _escalate(FeedHealthStatus.warn)

    if false_ticks:
        issues.append(f"Detected {len(false_ticks)} suspect ticks")
        if len(false_ticks) > resolved_config.false_tick_limit:
            _escalate(FeedHealthStatus.fail)
        else:
            _escalate(FeedHealthStatus.warn)

    if dropped_ticks:
        issues.append(f"Dropped {len(dropped_ticks)} out-of-order ticks")
        _escalate(FeedHealthStatus.warn)

    last_tick = sample_ticks[-1]
    stale_delta = (generated_at - last_tick.timestamp).total_seconds()
    stale = stale_delta > resolved_config.stale_grace_seconds
    if stale:
        issues.append(
            f"Feed stale by {stale_delta:.1f}s (grace {resolved_config.stale_grace_seconds:.1f}s)"
        )
        _escalate(FeedHealthStatus.fail)

    metadata: dict[str, object] = {}
    if median_interval is not None:
        metadata["median_interval_seconds"] = median_interval
    metadata["stale_delta_seconds"] = stale_delta
    if dropped_ticks:
        metadata["dropped_tick_count"] = len(dropped_ticks)
        metadata["dropped_reason_codes"] = [tick.reason_code for tick in dropped_ticks]

    return FeedAnomalyReport(
        symbol=symbol,
        generated_at=generated_at,
        status=status,
        sample_count=len(sample_ticks),
        lookback_seconds=lookback_seconds,
        median_interval_seconds=median_interval,
        max_gap_seconds=largest_gap,
        stale=stale,
        gaps=tuple(gaps),
        false_ticks=tuple(false_ticks),
        dropped_ticks=tuple(dropped_ticks),
        issues=tuple(issues),
        metadata=metadata,
    )
