"""Utility for provisioning and validating a local Elasticsearch cluster."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from uuid import uuid4

import requests
from requests import Session
from requests.exceptions import RequestException


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_COMPOSE_FILE = _REPO_ROOT / "docker" / "elasticsearch" / "docker-compose.yaml"
_DEFAULT_ELASTIC_URL = "http://localhost:9200"
_DEFAULT_INDEX = "logs-emp-bootstrap"


class DeploymentError(RuntimeError):
    """Raised when the deployment process fails."""


@dataclass(frozen=True)
class DeploymentConfig:
    """Configuration for orchestrating the Elasticsearch deployment."""

    compose_file: Path
    compose_command: Sequence[str]
    project_name: str | None
    elastic_url: str
    health_timeout: float
    health_interval: float
    ingest_index: str
    skip_compose: bool
    skip_ingest: bool


@dataclass(frozen=True)
class DeploymentResult:
    """Outcome of a deployment run."""

    cluster_health: dict[str, object]
    ingestion_document_id: str | None
    ingestion_marker: str | None


def _resolve_compose_command(preferred: str | None) -> Sequence[str]:
    if preferred:
        parts = shlex.split(preferred)
        if not parts:
            raise DeploymentError("compose command cannot be empty")
        executable = parts[0]
        if shutil.which(executable) is None:
            raise DeploymentError(f"compose executable '{executable}' not found on PATH")
        return parts

    docker = shutil.which("docker")
    docker_compose = shutil.which("docker-compose")

    if docker is not None:
        return [docker, "compose"]
    if docker_compose is not None:
        return [docker_compose]

    raise DeploymentError(
        "Neither 'docker' nor 'docker-compose' was found on PATH. "
        "Install Docker or provide --compose-command."
    )


def _run_compose_up(config: DeploymentConfig) -> None:
    command = list(config.compose_command)
    command.extend(["-f", str(config.compose_file)])
    if config.project_name:
        command.extend(["-p", config.project_name])
    command.extend(["up", "-d"])

    try:
        print("Invoking:", " ".join(command))
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:  # pragma: no cover - defensive guard
        raise DeploymentError(f"Failed to execute compose command: {exc}") from exc
    except subprocess.CalledProcessError as exc:
        raise DeploymentError(
            "docker compose up failed with exit code " f"{exc.returncode}"
        ) from exc


def _cluster_health(session: Session, *, url: str) -> dict[str, object]:
    response = session.get(f"{url.rstrip('/')}/_cluster/health", timeout=10)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise DeploymentError(
            "Unexpected response payload from _cluster/health: " f"{payload!r}"
        )
    return payload


def _wait_for_cluster(url: str, timeout: float, interval: float) -> dict[str, object]:
    deadline = time.time() + timeout
    with requests.Session() as session:
        last_error: Exception | None = None
        while time.time() < deadline:
            try:
                health = _cluster_health(session, url=url)
            except RequestException as exc:
                last_error = exc
            except DeploymentError as exc:
                last_error = exc
            else:
                status = str(health.get("status", "")).lower()
                if status in {"yellow", "green"}:
                    return health
            time.sleep(interval)
    if last_error:
        raise DeploymentError(f"Elasticsearch cluster did not become ready: {last_error}")
    raise DeploymentError("Timed out waiting for Elasticsearch cluster to become ready")


def _ingest_test_document(url: str, index: str) -> tuple[str, str]:
    marker = uuid4().hex
    document = {
        "message": "EMP bootstrap log entry",
        "deployment_marker": marker,
        "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    response = requests.post(
        f"{url.rstrip('/')}/{index}/_doc",
        json=document,
        timeout=10,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise DeploymentError(
            "Unexpected response payload when ingesting log: " f"{payload!r}"
        )
    document_id = payload.get("_id")
    if not isinstance(document_id, str):
        raise DeploymentError(
            "Elasticsearch did not return a document identifier for the bootstrap log"
        )
    return document_id, marker


def _verify_ingestion(url: str, index: str, marker: str) -> None:
    query = {"query": {"term": {"deployment_marker": {"value": marker}}}}
    response = requests.post(
        f"{url.rstrip('/')}/{index}/_search",
        params={"size": 1},
        json=query,
        timeout=10,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise DeploymentError("Unexpected payload returned from Elasticsearch search")
    hits = payload.get("hits")
    if not isinstance(hits, dict) or not hits.get("hits"):
        raise DeploymentError(
            "Bootstrap log document was not found in Elasticsearch after ingestion"
        )


def deploy(config: DeploymentConfig) -> DeploymentResult:
    if not config.compose_file.exists():
        raise DeploymentError(f"Compose file not found: {config.compose_file}")

    if not config.skip_compose:
        _run_compose_up(config)

    health = _wait_for_cluster(
        config.elastic_url, timeout=config.health_timeout, interval=config.health_interval
    )

    document_id: str | None = None
    marker: str | None = None

    if not config.skip_ingest:
        document_id, marker = _ingest_test_document(config.elastic_url, config.ingest_index)
        _verify_ingestion(config.elastic_url, config.ingest_index, marker)

    return DeploymentResult(
        cluster_health=health,
        ingestion_document_id=document_id,
        ingestion_marker=marker,
    )


def _parse_args(argv: Sequence[str]) -> DeploymentConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compose-file",
        type=Path,
        default=_DEFAULT_COMPOSE_FILE,
        help=f"Path to docker compose file (default: {_DEFAULT_COMPOSE_FILE})",
    )
    parser.add_argument(
        "--project-name",
        type=str,
        default=None,
        help="Optional project name to use for docker compose",
    )
    parser.add_argument(
        "--compose-command",
        type=str,
        default=None,
        help=(
            "Command used to invoke docker compose. Defaults to 'docker compose' when available "
            "otherwise falls back to 'docker-compose'."
        ),
    )
    parser.add_argument(
        "--elastic-url",
        type=str,
        default=_DEFAULT_ELASTIC_URL,
        help=f"Base URL for the Elasticsearch cluster (default: {_DEFAULT_ELASTIC_URL})",
    )
    parser.add_argument(
        "--health-timeout",
        type=float,
        default=300.0,
        help="Seconds to wait for the cluster to become ready (default: 300)",
    )
    parser.add_argument(
        "--health-interval",
        type=float,
        default=5.0,
        help="Seconds between readiness checks (default: 5)",
    )
    parser.add_argument(
        "--ingest-index",
        type=str,
        default=_DEFAULT_INDEX,
        help=f"Index used for the verification log event (default: {_DEFAULT_INDEX})",
    )
    parser.add_argument(
        "--skip-compose",
        action="store_true",
        help="Skip invoking docker compose (assume cluster is already running)",
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip ingesting the verification log document",
    )

    args = parser.parse_args(argv)

    compose_command = _resolve_compose_command(args.compose_command)

    return DeploymentConfig(
        compose_file=args.compose_file,
        compose_command=compose_command,
        project_name=args.project_name,
        elastic_url=args.elastic_url,
        health_timeout=args.health_timeout,
        health_interval=args.health_interval,
        ingest_index=args.ingest_index,
        skip_compose=args.skip_compose,
        skip_ingest=args.skip_ingest,
    )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        config = _parse_args(argv or sys.argv[1:])
        result = deploy(config)
    except DeploymentError as exc:
        print(f"Deployment failed: {exc}", file=sys.stderr)
        return 1
    except RequestException as exc:  # pragma: no cover - network guard
        print(f"HTTP request failed: {exc}", file=sys.stderr)
        return 1

    print("Elasticsearch cluster ready:")
    print(json.dumps(result.cluster_health, indent=2, sort_keys=True))

    if result.ingestion_document_id:
        print(
            "Verification log ingested:",
            json.dumps(
                {
                    "index": config.ingest_index,
                    "document_id": result.ingestion_document_id,
                    "marker": result.ingestion_marker,
                }
            ),
        )
    else:
        print("Ingestion skipped per configuration")

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
