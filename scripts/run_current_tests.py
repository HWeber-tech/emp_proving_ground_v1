#!/usr/bin/env python3

import subprocess
import sys


def main() -> int:
    # Tests run offline by default; individual tests can set EMP_USE_MOCK_FIX=1 when needed
    cmd = [sys.executable, "-m", "pytest", "-q"]
    proc = subprocess.run(cmd)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())


