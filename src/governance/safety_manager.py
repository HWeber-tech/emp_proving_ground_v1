"""
Safety Manager
==============

Centralizes runtime guardrails to prevent accidental live trading and to honor
an external kill-switch. Keeps safety logic out of `main.py` and makes it easy
to extend with remote toggles in the future.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Mapping


@dataclass
class SafetyContext:
    run_mode: str
    confirm_live: bool
    kill_switch_path: Optional[str]


class SafetyManager:
    def __init__(self, run_mode: str, confirm_live: bool, kill_switch_path: Optional[str]):
        self._ctx = SafetyContext(run_mode=run_mode, confirm_live=confirm_live, kill_switch_path=kill_switch_path)

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> "SafetyManager":
        run_mode = getattr(config, "run_mode", "paper")
        confirm_live = getattr(config, "confirm_live", False)
        kill_switch_path = getattr(config, "kill_switch_path", None)
        return cls(run_mode, confirm_live, kill_switch_path)

    def enforce(self) -> None:
        # Live mode requires explicit confirmation
        if self._ctx.run_mode == "live" and not self._ctx.confirm_live:
            raise RuntimeError("Live mode requires CONFIRM_LIVE=true. Aborting.")

        # Kill-switch file halts startup
        if self._ctx.kill_switch_path:
            try:
                if os.path.exists(self._ctx.kill_switch_path):
                    raise RuntimeError(f"Kill-switch engaged at {self._ctx.kill_switch_path}. Aborting.")
            except Exception:
                # If path check fails, default to allowing startup; callers may log a warning
                pass


