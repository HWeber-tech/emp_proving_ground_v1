#!/usr/bin/env python3
"""Legacy placeholder for the retired OpenAPI credential flow."""

from __future__ import annotations

import sys
from textwrap import dedent


NOTICE = dedent(
    """
    The interactive credential helper previously automated the OpenAPI OAuth
    flow. That pathway has been removed and the script now simply informs users
    of the deprecation.

    Use the FIX onboarding procedures in docs/fix_api/ to configure supported
    credentials.
    """
)


def main() -> int:
    sys.stderr.write(NOTICE)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
