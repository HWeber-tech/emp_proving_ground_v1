from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_compose() -> dict[str, Any]:
    compose_path = REPO_ROOT / "docker-compose.yml"
    with compose_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _environment_to_dict(raw_env: Any) -> dict[str, str]:
    if isinstance(raw_env, dict):
        return {str(key): str(value) for key, value in raw_env.items()}

    env: dict[str, str] = {}
    if not isinstance(raw_env, (list, tuple)):
        return env

    for entry in raw_env:
        if not isinstance(entry, str):
            continue
        if "=" not in entry:
            continue
        key, value = entry.split("=", 1)
        env[key] = value
    return env


def test_engine_enables_metrics_exporter() -> None:
    compose = _load_compose()
    engine_env = compose["services"]["engine"].get("environment", [])
    env_map = _environment_to_dict(engine_env)

    assert env_map.get("METRICS_EXPORTER_ENABLED") in {"1", "true", "True"}
    assert env_map.get("EMP_METRICS_PORT") == "8081"


def test_prometheus_service_mounts_configs() -> None:
    compose = _load_compose()
    prometheus = compose["services"].get("prometheus")
    assert prometheus is not None, "Prometheus service must be defined"

    volumes = [str(volume) for volume in prometheus.get("volumes", [])]
    assert any("config/prometheus/prometheus.yml" in volume for volume in volumes)
    assert any("config/prometheus/emp_rules.yml" in volume for volume in volumes)

    depends_on = set(prometheus.get("depends_on", []))
    assert {"engine", "redis-exporter", "timescaledb-exporter", "kafka-exporter"} <= depends_on


def test_exporter_services_present() -> None:
    compose = _load_compose()
    services = compose["services"]

    for exporter in ("redis-exporter", "timescaledb-exporter", "kafka-exporter"):
        assert exporter in services, f"Missing {exporter} service"
        exporter_depends_on = services[exporter].get("depends_on", [])
        assert exporter_depends_on, f"{exporter} should depend on its backing service"
