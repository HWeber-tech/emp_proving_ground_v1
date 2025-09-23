#!/usr/bin/env python3
"""Stub notifier for the retired OpenAPI tooling quick check."""

from __future__ import annotations

import sys
from textwrap import dedent


MESSAGE = dedent(
    """
    The quick-test harness for the OpenAPI tooling has been removed. The project
    enforces a FIX-only policy and no longer bundles the legacy verifier or its
    documentation set.

    Use the FIX smoke tests and verification scripts described in
    docs/policies/integration_policy.md instead.
    """
)


def main() -> int:
    sys.stderr.write(MESSAGE)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
