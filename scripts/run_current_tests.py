#!/usr/bin/env python3

import os
import subprocess
import sys


def main() -> int:
    os.environ.setdefault("EMP_USE_MOCK_FIX", "1")
    cmd = [sys.executable, "-m", "pytest", "-q"]
    proc = subprocess.run(cmd)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())


