# syntax=docker/dockerfile:1.4
FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install minimal OS deps commonly needed by pip builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install dev requirements
WORKDIR /workspace
COPY requirements/dev.txt /tmp/dev.txt
RUN python -m pip install --upgrade pip && \
    pip install -r /tmp/dev.txt

ENTRYPOINT [ "bash" ]