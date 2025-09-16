#!/usr/bin/env python3

import os
import sys


def main() -> int:
    # Ensure project root on path
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, root)
    os.environ.setdefault("PYTHONPATH", root)

    try:
        import pytest  # type: ignore
    except Exception:
        print("pytest is required. Install with: pip install -r requirements/dev.txt or pip install pytest")
        return 2

    args = ["-q"]
    # Allow selective paths via CLI
    if len(sys.argv) > 1:
        args.extend(sys.argv[1:])
    else:
        args.append("tests")
    return pytest.main(args)


if __name__ == "__main__":
    raise SystemExit(main())


