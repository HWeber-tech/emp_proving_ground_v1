#!/usr/bin/env python3
"""Legacy placeholder for the retired OpenAPI discovery routine."""

from __future__ import annotations

import sys
from textwrap import dedent


DEPRECATION_NOTICE = dedent(
    """
    The discovery workflow formerly relied on the OpenAPI connectivity path.
    That integration has been removed in favour of FIX-only execution, so this
    command no longer performs any actions.

    Review docs/policies/integration_policy.md for the maintained discovery and
    verification flows.
    """
)


def main() -> int:
    sys.stderr.write(DEPRECATION_NOTICE)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
