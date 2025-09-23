#!/usr/bin/env python3
"""Legacy placeholder for the retired OpenAPI account lookup helper."""

from __future__ import annotations

import sys
from textwrap import dedent


DEPRECATION_NOTICE = dedent(
    """
    This script previously queried the deprecated OpenAPI stack to enumerate
    trading accounts. With the move to FIX-only connectivity the helper is no
    longer operational and simply documents the deprecation.

    Consult docs/policies/integration_policy.md for current credential and
    account management guidance.
    """
)


def main() -> int:
    sys.stderr.write(DEPRECATION_NOTICE)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
