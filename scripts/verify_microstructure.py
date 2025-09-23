#!/usr/bin/env python3
"""Legacy placeholder for the retired OpenAPI microstructure verifier."""

from __future__ import annotations

import sys
from textwrap import dedent


DEPRECATION_NOTICE = dedent(
    """
    The previous implementation of this helper depended on the deprecated
    OpenAPI connectivity stack. The project now enforces a FIX-only policy, so
    the microstructure verifier is no longer available.

    Consult docs/policies/integration_policy.md for the current connectivity
    guidance and supported verification flows.
    """
)


def main() -> int:
    sys.stderr.write(DEPRECATION_NOTICE)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
