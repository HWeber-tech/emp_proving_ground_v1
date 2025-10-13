"""Production runner orchestration for the Tiny Recursive Model."""

from __future__ import annotations

import datetime as dt
import json
import os
import platform
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .adapter import RIMInputAdapter
from .config import AutoApplySettings, RIMRuntimeConfig
from .encoder import RIMEncoder
from .model import TRMModel
from .governance import AutoApplyRuleConfig, ProposalEvaluation, publish_governance_artifacts
from .postprocess import build_suggestions
from .types import RIMInputBatch, StrategyEncoding


@dataclass(slots=True)
class TRMRunResult:
    suggestions_path: Path | None
    suggestions_count: int
    runtime_seconds: float
    skipped_reason: str | None = None
    run_id: str | None = None


class TRMRunner:
    """Coordinates diary loading, encoding, model inference, and publication."""

    def __init__(self, config: RIMRuntimeConfig, model: TRMModel, *, config_hash: str) -> None:
        self._config = config
        self._model = model
        self._config_hash = config_hash
        self._encoder = RIMEncoder()

    def run(self) -> TRMRunResult:
        start = time.perf_counter()

        if self._config.kill_switch:
            runtime = time.perf_counter() - start
            return TRMRunResult(None, 0, runtime, skipped_reason="kill_switch")

        adapter = RIMInputAdapter(
            self._config.diaries_dir,
            self._config.diary_glob,
            self._config.window_minutes,
        )

        with _FileLock(self._config.lock_path) as lock:
            if not lock.acquired:
                runtime = time.perf_counter() - start
                return TRMRunResult(None, 0, runtime, skipped_reason="lock_active")

            batch = adapter.load_batch()
            if batch is None:
                runtime = time.perf_counter() - start
                return TRMRunResult(None, 0, runtime, skipped_reason="no_diaries")

            if len(batch.entries) < self._config.min_entries:
                runtime = time.perf_counter() - start
                return TRMRunResult(None, 0, runtime, skipped_reason="insufficient_entries")

            result = self._execute(batch, start)

        return result

    def _execute(self, batch: RIMInputBatch, start: float) -> TRMRunResult:
        run_timestamp = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
        run_id = self._build_run_id(run_timestamp)

        encodings = self._encoder.encode(batch.entries)
        inferences = [self._model.infer(encoding) for encoding in encodings]
        suggestions = build_suggestions(
            batch,
            encodings,
            inferences,
            self._config,
            model_hash=self._model.model_hash,
            config_hash=self._config_hash,
        )

        auto_apply_config: AutoApplyRuleConfig | None = None
        proposal_evaluations: Sequence[ProposalEvaluation] | None = None
        settings = self._config.auto_apply
        if settings and settings.enabled:
            proposal_evaluations = _build_proposal_evaluations(
                suggestions,
                batch=batch,
                encodings=encodings,
                settings=settings,
            )
            if proposal_evaluations:
                auto_apply_config = AutoApplyRuleConfig(
                    uplift_threshold=settings.uplift_threshold,
                    max_risk_hits=settings.max_risk_hits,
                    min_budget_remaining=settings.min_budget_remaining,
                    max_budget_utilisation=settings.max_budget_utilisation,
                    require_budget_metrics=settings.require_budget_metrics,
                )

        suggestions_path = self._publish(
            suggestions,
            run_id=run_id,
            run_timestamp=run_timestamp,
        )
        runtime = time.perf_counter() - start
        self._log_metrics(batch, runtime, len(suggestions), suggestions_path)
        if self._config.enable_governance_gate:
            self._publish_governance(
                suggestions,
                run_id=run_id,
                run_timestamp=run_timestamp,
                batch=batch,
                suggestions_path=suggestions_path,
                proposal_evaluations=proposal_evaluations,
                auto_apply_config=auto_apply_config,
            )
        return TRMRunResult(
            suggestions_path,
            len(suggestions),
            runtime,
            run_id=run_id,
        )

    def _publish(
        self,
        suggestions: Sequence[dict[str, object]],
        *,
        run_id: str,
        run_timestamp: dt.datetime,
    ) -> Path | None:
        if not suggestions:
            return None
        publish_channel = self._config.publish_channel
        if publish_channel.startswith("file://"):
            target_dir = Path(publish_channel[len("file://") :])
        else:
            target_dir = Path("artifacts/rim_suggestions")
        target_dir.mkdir(parents=True, exist_ok=True)
        timestamp_slug = run_timestamp.strftime("%Y%m%dT%H%M%S")
        output_path = target_dir / f"rim-suggestions-UTC-{timestamp_slug}-{os.getpid()}.jsonl"
        with output_path.open("w", encoding="utf-8") as handle:
            for item in suggestions:
                payload = dict(item)
                payload.setdefault("run_id", run_id)
                handle.write(json.dumps(payload) + "\n")
        return output_path

    def _publish_governance(
        self,
        suggestions: Sequence[dict[str, object]],
        *,
        run_id: str,
        run_timestamp: dt.datetime,
        batch: RIMInputBatch,
        suggestions_path: Path | None,
        proposal_evaluations: Sequence[ProposalEvaluation] | None = None,
        auto_apply_config: AutoApplyRuleConfig | None = None,
    ) -> None:
        publish_governance_artifacts(
            suggestions,
            run_id=run_id,
            run_timestamp=run_timestamp,
            window=batch.window,
            input_hash=batch.input_hash,
            model_hash=self._model.model_hash,
            config_hash=self._config_hash,
            queue_path=self._config.governance_queue_path,
            digest_path=self._config.governance_digest_path,
            markdown_path=self._config.governance_markdown_path,
            artifact_path=suggestions_path,
            proposal_evaluations=proposal_evaluations,
            auto_apply_config=auto_apply_config,
        )

    def _log_metrics(
        self,
        batch: RIMInputBatch,
        runtime_seconds: float,
        suggestions_count: int,
        suggestions_path: Path | None,
    ) -> None:
        log_dir = self._config.telemetry.log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = dt.datetime.utcnow().replace(microsecond=0)
        line = (
            f"{timestamp.isoformat()}Z runtime_ms={runtime_seconds * 1000:.2f} "
            f"entries={len(batch.entries)} suggestions={suggestions_count} "
            f"model_hash={self._model.model_hash} suggestions_path={suggestions_path or 'none'}"
        )
        log_path = log_dir / f"rim-{timestamp:%Y%m%d}.log"
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def _build_run_id(self, timestamp: dt.datetime) -> str:
        node = platform.node() or "unknown"
        slug = timestamp.strftime("%Y%m%dT%H%M%S")
        return f"{slug}-{node}-{os.getpid()}"


def _build_proposal_evaluations(
    suggestions: Sequence[Mapping[str, object]],
    *,
    batch: RIMInputBatch,
    encodings: Sequence[StrategyEncoding],
    settings: AutoApplySettings,
) -> tuple[ProposalEvaluation, ...]:
    if not suggestions or not encodings:
        return tuple()

    stats_lookup = {encoding.strategy_id: encoding.stats for encoding in encodings}
    baseline_mean = 0.0
    try:
        baseline_mean = float(batch.aggregates.get("mean_pnl", 0.0) or 0.0)
    except Exception:
        baseline_mean = 0.0

    entry_counts: Counter[str] = Counter(entry.strategy_id for entry in batch.entries)
    risk_counts: Counter[str] = Counter()
    for entry in batch.entries:
        if entry.risk_flags:
            risk_counts[entry.strategy_id] += len(entry.risk_flags)

    evaluations: list[ProposalEvaluation] = []
    for suggestion in suggestions:
        suggestion_id = str(suggestion.get("suggestion_id", "")).strip()
        if not suggestion_id:
            continue
        strategy_id = _suggestion_strategy_id(suggestion)
        if strategy_id is None:
            continue
        stats = stats_lookup.get(strategy_id)
        if stats is None:
            continue

        uplift = float(stats.mean_pnl - baseline_mean)
        risk_hits = int(risk_counts.get(strategy_id, 0))

        limit = settings.budget_limit_for(strategy_id)
        if limit > 0:
            used = float(entry_counts.get(strategy_id, 0))
            budget_remaining = float(limit - used)
            budget_utilisation = used / float(limit)
        else:
            budget_remaining = None
            budget_utilisation = None

        metadata: dict[str, object] = {
            "baseline_mean_pnl": baseline_mean,
            "strategy_mean_pnl": stats.mean_pnl,
            "entry_count": stats.entry_count,
            "window_minutes": batch.window.minutes,
        }
        if limit > 0:
            metadata["budget_limit"] = float(limit)
            metadata["entries_observed"] = entry_counts.get(strategy_id, 0)

        evaluations.append(
            ProposalEvaluation(
                suggestion_id=suggestion_id,
                oos_uplift=uplift,
                risk_hits=risk_hits,
                budget_remaining=budget_remaining,
                budget_utilisation=budget_utilisation,
                metadata=metadata,
            )
        )

    return tuple(evaluations)


def _suggestion_strategy_id(suggestion: Mapping[str, object]) -> str | None:
    payload = suggestion.get("payload")
    if isinstance(payload, Mapping):
        raw_strategy = payload.get("strategy_id")
        if raw_strategy:
            return str(raw_strategy)
        candidates = payload.get("strategy_candidates")
        if isinstance(candidates, Sequence) and candidates:
            candidate = candidates[0]
            if candidate:
                return str(candidate)
    return None


class _FileLock:
    """Best-effort file lock with stale detection."""

    def __init__(self, path: Path, *, ttl_seconds: int = 7200) -> None:
        self._path = path
        self._ttl = ttl_seconds
        self._fd: int | None = None
        self._acquired = False

    def __enter__(self) -> "_FileLock":
        self._acquired = self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._acquired:
            self.release()

    @property
    def acquired(self) -> bool:
        return self._acquired

    def acquire(self) -> bool:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        now = time.time()
        try:
            fd = os.open(self._path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if self._is_stale(now):
                try:
                    self._path.unlink()
                except FileNotFoundError:
                    pass
                return self.acquire()
            return False
        os.write(fd, str(now).encode("utf-8"))
        self._fd = fd
        return True

    def release(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        try:
            self._path.unlink()
        except FileNotFoundError:
            pass

    def _is_stale(self, now: float) -> bool:
        try:
            stat = self._path.stat()
        except FileNotFoundError:
            return False
        return now - stat.st_mtime > self._ttl


__all__ = ["TRMRunResult", "TRMRunner"]
