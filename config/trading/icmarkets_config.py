"""Legacy placeholder for the deprecated OpenAPI configuration helpers."""

from __future__ import annotations

import textwrap


def describe() -> str:
    """Return a short notice explaining the deprecation."""
    return textwrap.dedent(
        """
        The OpenAPI configuration helpers were removed when the project adopted
        a FIX-only connectivity policy. Use the FIX configuration utilities in
        src.operational and docs/fix_api/ instead.
        """
    )


__all__ = ["describe"]
