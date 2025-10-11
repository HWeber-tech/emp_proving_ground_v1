"""Runtime guardrails for live trading and kill-switch enforcement."""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)


@dataclass
class SafetyContext:
    run_mode: str
    confirm_live: bool
    kill_switch_path: Optional[Path]


class SafetyManager:
    def __init__(
        self,
        run_mode: str,
        confirm_live: bool,
        kill_switch_path: str | Path | None,
    ) -> None:
        self._ctx = SafetyContext(
            run_mode=self._normalize_run_mode(run_mode),
            confirm_live=confirm_live,
            kill_switch_path=self._normalize_kill_switch_path(kill_switch_path),
        )

    @staticmethod
    def _normalize_run_mode(value: str) -> str:
        """Normalise run mode strings for consistent policy enforcement."""

        return value.strip().lower()

    @staticmethod
    def _normalize_kill_switch_path(path: str | Path | None) -> Optional[Path]:
        if path is None:
            return None
        raw = str(path).strip()
        if not raw or raw.lower() in {"none", "disabled", "off"}:
            return None
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = Path(tempfile.gettempdir()) / candidate
        return candidate

    @staticmethod
    def _coerce_confirmation(value: Any) -> bool:
        """Normalise truthy flags from configuration payloads.

        The configuration surface historically accepted environment derived
        strings (e.g. ``"true"``/``"false"``) in addition to native booleans.
        ``bool("false")`` would evaluate to ``True`` which defeats the
        explicit confirmation gate for live trading.  This helper interprets
        common string representations and raises for unrecognised payloads so
        misconfigurations fail fast during bootstrap.
        """

        if isinstance(value, bool):
            return value
        if value is None:
            return False
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            normalised = value.strip().lower()
            if not normalised:
                return False
            if normalised in {"true", "1", "yes", "y", "on"}:
                return True
            if normalised in {"false", "0", "no", "n", "off"}:
                return False
            raise ValueError(f"Unrecognised confirmation flag: {value!r}")
        raise TypeError(f"Unsupported confirmation flag type: {type(value)!r}")

    @classmethod
    def from_config(cls, config: Mapping[str, object] | object) -> "SafetyManager":
        def _lookup(name: str, default: object) -> object:
            if isinstance(config, Mapping):
                return config.get(name, default)
            return getattr(config, name, default)

        run_mode_raw = _lookup("run_mode", "paper")
        confirm_live_raw = _lookup("confirm_live", False)
        kill_switch_raw = _lookup("kill_switch_path", None)

        run_mode = (
            cls._normalize_run_mode(str(run_mode_raw))
            if run_mode_raw is not None
            else "paper"
        )
        try:
            confirm_live = cls._coerce_confirmation(confirm_live_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid confirm_live flag in safety configuration") from exc

        return cls(run_mode, confirm_live, kill_switch_raw)

    @property
    def kill_switch_path(self) -> Optional[Path]:
        return self._ctx.kill_switch_path

    def enforce(self) -> None:
        # Live mode requires explicit confirmation
        if self._ctx.run_mode == "live" and not self._ctx.confirm_live:
            raise RuntimeError("Live mode requires CONFIRM_LIVE=true. Aborting.")

        # Kill-switch file halts startup
        kill_switch = self._ctx.kill_switch_path
        if kill_switch:
            try:
                if kill_switch.exists():
                    raise RuntimeError(f"Kill-switch engaged at {kill_switch}. Aborting.")
            except OSError as exc:
                logger.warning(
                    "Unable to inspect kill-switch path %s: %s",
                    kill_switch,
                    exc,
                    exc_info=exc,
                )
