FROM python:3.11-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Small set of tools for robust networking and optional VCS installs
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip tooling and install the typing toolchain only (pinned mypy)
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install "mypy==1.17.1" "mypy-extensions>=1.0.0" \
    "types-requests" "types-PyYAML" "types-python-dateutil" || true

# No project dependency installs; mypy.ini and sources are mounted by the runner.
# Workdir is provided by docker run (-w /workspace)
