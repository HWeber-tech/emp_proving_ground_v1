#!/usr/bin/env python3
"""Legacy placeholder for the retired OpenAPI reporting helper."""

from __future__ import annotations

import sys
from textwrap import dedent


DEPRECATION_NOTICE = dedent(
    """
    The reporting helper used data collected from the deprecated OpenAPI
    integration. The project no longer maintains that pathway, so this
    executable simply documents the retirement and exits with a non-zero code.

    For supported workflows consult docs/policies/integration_policy.md.
    """
)


def main() -> int:
    sys.stderr.write(DEPRECATION_NOTICE)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
