"""
Lightweight JSON-based persistence for FIX session/order state.

- Default base directory: data/fix_state
- Atomic writes via temp file + rename
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, is_dataclass
from typing import Dict, Any, Optional

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover
    redis = None  # type: ignore


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _atomic_write(path: str, content: str) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


class JSONStateStore:
    """Thread-safe JSON state store."""

    def __init__(self, base_dir: str = "data/fix_state") -> None:
        self.base_dir = base_dir
        _ensure_dir(self.base_dir)
        self._lock = threading.Lock()

    def _path(self, name: str) -> str:
        return os.path.join(self.base_dir, f"{name}.json")

    def save_orders(self, orders: Dict[str, Any]) -> None:
        with self._lock:
            serializable: Dict[str, Any] = {}
            for cl_id, order in orders.items():
                if is_dataclass(order):
                    d = asdict(order)
                    # Convert datetime fields to isoformat strings
                    for k in ("created_time", "updated_time", "transact_time"):
                        if d.get(k) is not None:
                            d[k] = getattr(order, k).isoformat() if hasattr(order, k) else d[k]
                    serializable[cl_id] = d
                else:
                    serializable[cl_id] = order
            _ensure_dir(self.base_dir)
            _atomic_write(self._path("orders"), json.dumps(serializable, default=str))

    def load_orders(self) -> Dict[str, Any]:
        try:
            with open(self._path("orders"), "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except FileNotFoundError:
            return {}
        except Exception:
            return {}


class RedisStateStore:
    """Redis-backed state store. Falls back to JSONStateStore if Redis unavailable.
    Keys:
      emp:fix:orders -> JSON map of orders by clOrdID
      emp:fix:session:<session> -> JSON dict with seq state
    """

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0, password: Optional[str] = None,
                 fallback_dir: str = "data/fix_state") -> None:
        self.fallback = JSONStateStore(fallback_dir)
        self.client: Optional[Any] = None
        if redis is not None:
            try:
                pool = redis.ConnectionPool(host=host, port=port, db=db, password=password, decode_responses=True)
                self.client = redis.Redis(connection_pool=pool)
                self.client.ping()
            except Exception:
                self.client = None

    def save_orders(self, orders: Dict[str, Any]) -> None:
        if not self.client:
            return self.fallback.save_orders(orders)
        try:
            serializable: Dict[str, Any] = {}
            for cl_id, order in orders.items():
                if is_dataclass(order):
                    d = asdict(order)
                    for k in ("created_time", "updated_time", "transact_time"):
                        if d.get(k) is not None:
                            d[k] = getattr(order, k).isoformat() if hasattr(order, k) else d[k]
                    serializable[cl_id] = d
                else:
                    serializable[cl_id] = order
            self.client.set("emp:fix:orders", json.dumps(serializable, default=str))
        except Exception:
            self.fallback.save_orders(orders)

    def load_orders(self) -> Dict[str, Any]:
        if not self.client:
            return self.fallback.load_orders()
        try:
            raw = self.client.get("emp:fix:orders")
            return json.loads(raw) if raw else {}
        except Exception:
            return self.fallback.load_orders()

    def save_session_state(self, session: str, state: Dict[str, Any]) -> None:
        if not self.client:
            return self.fallback.save_session_state(session, state)
        try:
            self.client.set(f"emp:fix:session:{session}", json.dumps(state, default=str))
        except Exception:
            self.fallback.save_session_state(session, state)

    def load_session_state(self, session: str) -> Dict[str, Any]:
        if not self.client:
            return self.fallback.load_session_state(session)
        try:
            raw = self.client.get(f"emp:fix:session:{session}")
            return json.loads(raw) if raw else {}
        except Exception:
            return self.fallback.load_session_state(session)

    def save_session_state(self, session: str, state: Dict[str, Any]) -> None:
        with self._lock:
            _ensure_dir(self.base_dir)
            _atomic_write(self._path(f"session_{session}"), json.dumps(state, default=str))

    def load_session_state(self, session: str) -> Dict[str, Any]:
        try:
            with open(self._path(f"session_{session}"), "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except FileNotFoundError:
            return {}
        except Exception:
            return {}


