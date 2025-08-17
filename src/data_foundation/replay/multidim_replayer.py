"""
Multi-dimension replayer: plays back MD (WHAT) and macro (WHY) streams with a unified clock.
"""
from __future__ import annotations

from datetime import datetime
import json
from typing import Callable, Optional


def _parse_jsonl(path: str):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    yield json.loads(line)
                except Exception:
                    continue
    except Exception:
        return


class MultiDimReplayer:
    def __init__(self, md_path: Optional[str] = None, macro_path: Optional[str] = None, yields_path: Optional[str] = None):
        self.md_path = md_path
        self.macro_path = macro_path
        self.yields_path = yields_path

    def replay(self,
               on_md: Optional[Callable[[dict], None]] = None,
               on_macro: Optional[Callable[[dict], None]] = None,
               on_yield: Optional[Callable[[dict], None]] = None,
               limit: Optional[int] = None) -> int:
        """Simple merge by timestamp ISO; no sleeping to keep offline fast."""
        events = []
        if self.md_path:
            for e in _parse_jsonl(self.md_path):
                e["_kind"] = "md"
                events.append(e)
        if self.macro_path:
            for e in _parse_jsonl(self.macro_path):
                e["_kind"] = "macro"
                events.append(e)
        if self.yields_path:
            for e in _parse_jsonl(self.yields_path):
                e["_kind"] = "yield"
                events.append(e)
        def ts(e):
            try:
                return datetime.fromisoformat(e.get("timestamp"))
            except Exception:
                return datetime.min
        events.sort(key=ts)
        emitted = 0
        for e in events:
            if limit and emitted >= limit:
                break
            if e.get("_kind") == "md" and on_md:
                on_md(e)
                emitted += 1
            elif e.get("_kind") == "macro" and on_macro:
                on_macro(e)
            elif e.get("_kind") == "yield" and on_yield:
                on_yield(e)
        return emitted


