"""System configuration management for EMP Professional Predator (typed)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, Optional, Literal

import os as _os

try:
    # Prefer Pydantic v2 + pydantic-settings if available
    from pydantic_settings import BaseSettings  # type: ignore
    from pydantic import Field  # type: ignore
    _HAS_SETTINGS = True
except Exception:  # No pydantic-settings available
    _HAS_SETTINGS = False

    class BaseSettings:  # minimal shim
        def __init__(self, *args, **kwargs) -> None:
            pass

    def Field(default=None, env: str | None = None, default_factory=None):  # type: ignore
        return default


def _default_kill_switch() -> Path:
    tmp_dir = os.getenv("TMP") or os.getenv("TEMP") or "/tmp"
    return Path(tmp_dir) / "emp_pg.KILL"


class SystemConfig(BaseSettings):
    """Typed configuration with env and .env support."""

    # Modes and environment
    run_mode: Literal["mock", "paper", "live"] = Field("paper", env="RUN_MODE")
    environment: str = Field("demo", env="EMP_ENVIRONMENT")
    emp_tier: Literal["tier_0", "tier_1", "tier_2"] = Field("tier_0", env="EMP_TIER")
    confirm_live: bool = Field(False, env="CONFIRM_LIVE")

    # Protocol (FIX only)
    connection_protocol: Literal["fix"] = Field("fix", env="CONNECTION_PROTOCOL")

    # Credentials
    account_number: Optional[str] = Field(default=None, env="ICMARKETS_ACCOUNT")
    password: Optional[str] = Field(default=None, env="ICMARKETS_PASSWORD")

    # Safety
    kill_switch_path: Path = Field(default_factory=_default_kill_switch, env="EMP_KILL_SWITCH")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @property
    def CONNECTION_PROTOCOL(self) -> str:
        """Backward-compatible uppercase alias (read-only)."""
        return self.connection_protocol

    def validate_config(self) -> bool:
        placeholders = {None, "9533708", "WNSE5822", "your_account_id_here", "your_trade_password"}
        if self.account_number in placeholders or self.password in placeholders:
            raise ValueError("IC Markets credentials must be provided via environment variables")
        # FIX-only enforcement (defense-in-depth)
        if self.connection_protocol != "fix":
            raise ValueError("Only FIX is supported in this build. Set CONNECTION_PROTOCOL=fix.")
        return True

    def get_config(self) -> Dict[str, Any]:
        return {
            "connection_protocol": self.connection_protocol,
            "environment": self.environment,
            "account_number": self.account_number,
            "emp_tier": self.emp_tier,
            "run_mode": self.run_mode,
        }

    def get_icmarkets_config(self) -> Dict[str, Any]:
        host = "demo-uk-eqx-01.p.c-trader.com" if self.environment == "demo" else "live-uk-eqx-01.p.c-trader.com"
        return {
            "account_number": self.account_number,
            "password": self.password,
            "environment": self.environment,
            "host": host,
            "price_port": 5211,
            "trade_port": 5212,
        }

    # Minimal env loading if pydantic-settings is unavailable
    def __init__(self, *args, **kwargs) -> None:  # type: ignore[override]
        super().__init__(*args, **kwargs)
        if not _HAS_SETTINGS:
            # Populate from environment when pydantic-settings is not installed
            self.run_mode = _os.getenv("RUN_MODE", str(self.run_mode))
            self.environment = _os.getenv("EMP_ENVIRONMENT", str(self.environment))
            self.emp_tier = _os.getenv("EMP_TIER", str(self.emp_tier))
            cp = _os.getenv("CONNECTION_PROTOCOL", str(self.connection_protocol))
            self.connection_protocol = "fix" if cp != "fix" else cp  # force fix
            self.account_number = _os.getenv("ICMARKETS_ACCOUNT", self.account_number)
            self.password = _os.getenv("ICMARKETS_PASSWORD", self.password)
