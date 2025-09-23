#!/usr/bin/env python3
"""Legacy placeholder for the retired OpenAPI symbol debugger."""

from __future__ import annotations

import sys
from textwrap import dedent


DEPRECATION_NOTICE = dedent(
    """
    Symbol debugging previously interacted with the deprecated OpenAPI stack.
    The helper now only records that retirement and exits immediately.

    Active tooling lives under the FIX workflows documented in
    docs/policies/integration_policy.md.
    """
)


def main() -> int:
    sys.stderr.write(DEPRECATION_NOTICE)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
