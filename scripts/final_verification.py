#!/usr/bin/env python3
"""Legacy placeholder for the retired OpenAPI final verification."""

from __future__ import annotations

import sys
from textwrap import dedent


DEPRECATION_NOTICE = dedent(
    """
    This helper previously wrapped the last-mile OpenAPI verification steps.
    The workflow has been retired alongside the OpenAPI connectivity stack and
    now serves only as documentation of that deprecation.

    Follow docs/policies/integration_policy.md for the supported FIX-focused
    validation options.
    """
)


def main() -> int:
    sys.stderr.write(DEPRECATION_NOTICE)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
