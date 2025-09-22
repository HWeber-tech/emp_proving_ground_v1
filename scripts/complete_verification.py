#!/usr/bin/env python3
"""Legacy placeholder for the retired OpenAPI verification workflow."""

from __future__ import annotations

import sys
from textwrap import dedent


DEPRECATION_NOTICE = dedent(
    """
    The comprehensive verification routine previously automated the
    OpenAPI-based connectivity checks. That integration path has been
    decommissioned under the FIX-only policy, so this entry point now only
    documents the retirement.

    Refer to docs/policies/integration_policy.md for the supported
    verification strategy.
    """
)


def main() -> int:
    sys.stderr.write(DEPRECATION_NOTICE)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
