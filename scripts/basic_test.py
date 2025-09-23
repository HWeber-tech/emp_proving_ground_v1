#!/usr/bin/env python3
"""Legacy placeholder for the retired OpenAPI smoke test."""

from __future__ import annotations

import sys
from textwrap import dedent


DEPRECATION_NOTICE = dedent(
    """
    The original smoke test exercised the OpenAPI connectivity layer. That
    pathway has been retired, so this command now exists solely to document the
    deprecation and to redirect contributors to the supported flows.

    Review docs/policies/integration_policy.md for current smoke-test
    recommendations.
    """
)


def main() -> int:
    sys.stderr.write(DEPRECATION_NOTICE)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
