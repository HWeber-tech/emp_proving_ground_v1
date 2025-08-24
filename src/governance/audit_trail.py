from __future__ import annotations


class AuditTrail:
    def __init__(self) -> None:
        """Typed shim for audit trail."""
        pass

    def record(self, event: str) -> None:
        """Record an audit event."""
        return None


__all__ = ["AuditTrail"]
