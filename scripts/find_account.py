#!/usr/bin/env python3
"""Legacy placeholder for the retired OpenAPI account discovery helper."""

from __future__ import annotations

import sys
from textwrap import dedent


DEPRECATION_NOTICE = dedent(
    """
    The interactive account discovery flow relied on the discontinued
    OpenAPI integration. In the FIX-only era this helper merely advertises the
    deprecation so contributors do not attempt to revive the legacy stack.

    Please follow docs/policies/integration_policy.md for supported
    configuration steps.
    """
)


def main() -> int:
    sys.stderr.write(DEPRECATION_NOTICE)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
