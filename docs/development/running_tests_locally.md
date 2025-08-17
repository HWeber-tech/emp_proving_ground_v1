# Running tests locally (no sudo)

Three no-sudo options to run pytest locally. Commands are copy-paste ready.

## A) uv (recommended; no sudo)
- Install uv and add it to PATH:
  - curl -LsSf https://astral.sh/uv/install.sh | sh
  - export PATH="$HOME/.local/bin:$PATH"
- Run tests:
  - PYTHONPATH=. EMP_USE_MOCK_FIX=1 uvx pytest -q [tests/current](tests/current)
- Or use the script:
  - bash [scripts/dev/test_uv.sh](scripts/dev/test_uv.sh)
- Targeted tests example:
  - PYTHONPATH=. EMP_USE_MOCK_FIX=1 uvx pytest -q [tests/current/test_operational_metrics_extra.py](tests/current/test_operational_metrics_extra.py) [tests/current/test_operational_metrics_hist.py](tests/current/test_operational_metrics_hist.py) [tests/current/test_position_model_extra.py](tests/current/test_position_model_extra.py) [tests/current/test_trade_model_extra.py](tests/current/test_trade_model_extra.py) [tests/current/test_yield_signal_extra.py](tests/current/test_yield_signal_extra.py)

## B) Bootstrap pip to user site (no sudo)
- Install pip to user site and add PATH:
  - curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
  - python3 /tmp/get-pip.py --user
  - export PATH="$HOME/.local/bin:$PATH"
- Install pytest and run:
  - PYTHONPATH=. EMP_USE_MOCK_FIX=1 python3 -m pip install --user pytest
  - PYTHONPATH=. EMP_USE_MOCK_FIX=1 python3 -m pytest -q [tests/current](tests/current)

## C) Dev Docker image (no host Python needed)
- Build the image:
  - docker build -f [dev.Dockerfile](dev.Dockerfile) -t emp-dev .
- Run tests inside the container (mounts your repo and sets env vars):
  - docker run --rm -e EMP_USE_MOCK_FIX=1 -e PYTHONPATH=/app -v "$PWD":/app -w /app emp-dev pytest -q [tests/current](tests/current)