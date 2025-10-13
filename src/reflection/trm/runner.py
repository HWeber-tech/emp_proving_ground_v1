"""Production runner orchestration for the Tiny Recursive Model."""

from __future__ import annotations

import datetime as dt
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .adapter import RIMInputAdapter
from .config import RIMRuntimeConfig
from .encoder import RIMEncoder
from .model import TRMModel
from .postprocess import build_suggestions
from .types import RIMInputBatch


@dataclass(slots=True)
class TRMRunResult:
    suggestions_path: Path | None
    suggestions_count: int
    runtime_seconds: float
    skipped_reason: str | None = None


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

        suggestions_path = self._publish(suggestions)
        runtime = time.perf_counter() - start
        self._log_metrics(batch, runtime, len(suggestions), suggestions_path)
        return TRMRunResult(suggestions_path, len(suggestions), runtime)

    def _publish(self, suggestions: Iterable[dict[str, object]]) -> Path | None:
        suggestions = list(suggestions)
        if not suggestions:
            return None
        publish_channel = self._config.publish_channel
        if publish_channel.startswith("file://"):
            target_dir = Path(publish_channel[len("file://") :])
        else:
            target_dir = Path("artifacts/rim_suggestions")
        target_dir.mkdir(parents=True, exist_ok=True)
        timestamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        run_id = f"{timestamp}-{os.uname().nodename}-{os.getpid()}"
        output_path = target_dir / f"rim-suggestions-UTC-{timestamp}-{os.getpid()}.jsonl"
        with output_path.open("w", encoding="utf-8") as handle:
            for item in suggestions:
                item.setdefault("run_id", run_id)
                handle.write(json.dumps(item) + "\n")
        return output_path

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
