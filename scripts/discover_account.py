#!/usr/bin/env python3
"""Legacy placeholder for the retired OpenAPI account discovery helper."""

from __future__ import annotations

import sys
from textwrap import dedent


DEPRECATION_NOTICE = dedent(
    """
    Account discovery previously piggybacked on the deprecated OpenAPI stack.
    That pathway has been removed, so this command intentionally exits without
    performing any remote calls.

    See docs/policies/integration_policy.md for information on provisioning
    credentials for the supported FIX connectivity.
    """
)


def main() -> int:
    sys.stderr.write(DEPRECATION_NOTICE)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
