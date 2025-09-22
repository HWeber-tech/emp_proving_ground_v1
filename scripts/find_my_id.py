#!/usr/bin/env python3
"""Legacy placeholder for the retired OpenAPI account lookup helper."""

from __future__ import annotations

import sys
from textwrap import dedent


NOTICE = dedent(
    """
    This command previously queried the deprecated OpenAPI endpoint to list
    account identifiers. The workflow has been removed alongside the OpenAPI
    integration, so the script now exits immediately.

    Use the FIX onboarding guides in docs/fix_api/ for supported credential
    discovery steps.
    """
)


def main() -> int:
    sys.stderr.write(NOTICE)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
