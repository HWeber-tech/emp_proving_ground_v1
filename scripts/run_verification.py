#!/usr/bin/env python3
"""Legacy placeholder for the retired OpenAPI verification orchestrator."""

from __future__ import annotations

import sys
from textwrap import dedent


DEPRECATION_NOTICE = dedent(
    """
    The verification orchestrator previously coordinated the OpenAPI-based
    tooling. Those integrations were removed during the FIX-only migration, so
    this wrapper intentionally aborts and documents the deprecation instead.

    Refer to docs/policies/integration_policy.md for the maintained
    verification flow.
    """
)


def main() -> int:
    sys.stderr.write(DEPRECATION_NOTICE)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
