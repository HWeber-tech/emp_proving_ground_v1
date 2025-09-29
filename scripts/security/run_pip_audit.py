"""CLI wrapper around :mod:`src.security.pip_audit_runner`."""

from __future__ import annotations

from src.security.pip_audit_runner import main


if __name__ == "__main__":  # pragma: no cover - convenience wrapper
    raise SystemExit(main())
