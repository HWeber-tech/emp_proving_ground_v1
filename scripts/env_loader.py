"""Helpers for locating the runtime secrets dotenv file."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

ENV_FILE_ENVVAR = "EMP_SECRETS_ENV_FILE"
_DEFAULT_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def resolve_env_file() -> Path:
    """Return the configured secrets env file path (may not exist)."""
    override = os.environ.get(ENV_FILE_ENVVAR)
    if override:
        return Path(override).expanduser()
    return _DEFAULT_ENV_PATH


def load_dotenv_if_available() -> Tuple[Path, bool]:
    """Attempt to load the secrets dotenv file; return path and success flag."""
    env_path = resolve_env_file()
    if env_path.exists():
        try:
            from dotenv import load_dotenv
        except ImportError:
            return env_path, False
        load_dotenv(dotenv_path=env_path)
        return env_path, True
    return env_path, False
