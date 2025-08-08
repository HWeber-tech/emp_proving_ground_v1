"""System configuration management for EMP Professional Predator (typed)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, Optional, Literal

from pydantic import BaseSettings, Field


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

    # Protocol
    connection_protocol: Literal["fix", "openapi"] = Field("fix", env="CONNECTION_PROTOCOL")

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
