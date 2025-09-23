#!/usr/bin/env python3
"""Legacy placeholder for the retired OpenAPI credential probe."""

from __future__ import annotations

import sys
from textwrap import dedent


DEPRECATION_NOTICE = dedent(
    """
    This utility previously validated credentials against the deprecated
    OpenAPI stack. It now serves as a reminder that the flow has been removed in
    favour of FIX-only connectivity.

    See docs/policies/integration_policy.md for the supported credential
    validation guidance.
    """
)


def main() -> int:
    sys.stderr.write(DEPRECATION_NOTICE)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
