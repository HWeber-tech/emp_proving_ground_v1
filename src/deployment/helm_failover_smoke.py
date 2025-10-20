"""Helm failover smoke test that validates pod readiness via historical replay."""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol

__all__ = [
    "HelmFailoverReplayConfig",
    "HelmFailoverReplayError",
    "HelmFailoverReplayResult",
    "HelmPodStatus",
    "run_helm_failover_replay_smoke",
]


class CommandRunner(Protocol):
    """Callable responsible for executing shell commands."""

    def __call__(
        self,
        command: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Execute ``command`` and return a completed process."""


@dataclass(slots=True)
class HelmFailoverReplayConfig:
    """Configuration for the Helm failover replay smoke test."""

    release: str
    namespace: str = "emp-system"
    selector: str | None = None
    max_wait_seconds: float = 120.0
    poll_interval_seconds: float = 5.0
    kubectl: str = "kubectl"
    replay_command: Sequence[str] | None = None
    replay_timeout: float | None = 180.0
    replay_environment: Mapping[str, str] | None = None
    kubectl_timeout: float = 30.0

    def __post_init__(self) -> None:
        if not self.release:
            raise ValueError("release must be provided")
        if self.max_wait_seconds <= 0:
            raise ValueError("max_wait_seconds must be positive")
        if self.poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        if self.kubectl_timeout <= 0:
            raise ValueError("kubectl_timeout must be positive")
        if self.selector is None:
            self.selector = f"app.kubernetes.io/instance={self.release}"

    def resolve_replay_command(self) -> tuple[str, ...]:
        """Return the replay command with defaults applied."""

        if self.replay_command is not None:
            return tuple(str(part) for part in self.replay_command)

        return (
            self.kubectl,
            "exec",
            f"deploy/{self.release}",
            "-n",
            self.namespace,
            "--",
            "python",
            "-m",
            "tools.runtime.run_simulation",
            "--timeout",
            "30",
            "--summary-path",
            "/tmp/helm_failover_replay_summary.json",
            "--symbols",
            "EURUSD",
        )

    def kubectl_get_pods_command(self) -> tuple[str, ...]:
        """Return the command tuple used to read pod readiness."""

        command = [self.kubectl, "get", "pods", "-n", self.namespace, "-o", "json"]
        if self.selector:
            command.extend(["-l", self.selector])
        return tuple(command)


@dataclass(slots=True)
class HelmPodStatus:
    """Ready status for a single pod belonging to the Helm release."""

    name: str
    ready: bool
    reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "ready": self.ready,
        }
        if self.reason:
            payload["reason"] = self.reason
        return payload


@dataclass(slots=True)
class HelmFailoverReplayResult:
    """Outcome from the Helm failover replay smoke test."""

    release: str
    namespace: str
    selector: str | None
    pods: tuple[HelmPodStatus, ...]
    replay_command: tuple[str, ...]
    replay_stdout: str
    replay_stderr: str
    replay_returncode: int
    started_at: datetime
    completed_at: datetime
    duration_seconds: float = field(init=False)

    def __post_init__(self) -> None:
        self.duration_seconds = max(0.0, (self.completed_at - self.started_at).total_seconds())

    def pods_ready(self) -> bool:
        """Return ``True`` when every recorded pod is ready."""

        return all(pod.ready for pod in self.pods)

    def as_dict(self) -> dict[str, object]:
        return {
            "release": self.release,
            "namespace": self.namespace,
            "selector": self.selector,
            "pods": [pod.as_dict() for pod in self.pods],
            "replay_command": list(self.replay_command),
            "replay_returncode": self.replay_returncode,
            "duration_seconds": self.duration_seconds,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
        }


class HelmFailoverReplayError(RuntimeError):
    """Raised when the Helm failover replay smoke test fails."""

    def __init__(
        self,
        message: str,
        *,
        pods: Sequence[HelmPodStatus] | None = None,
        returncode: int | None = None,
        stdout: str | None = None,
        stderr: str | None = None,
    ) -> None:
        super().__init__(message)
        self.pods = tuple(pods or ())
        self.returncode = returncode
        self.stdout = stdout or ""
        self.stderr = stderr or ""

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "error": str(self),
        }
        if self.pods:
            payload["pods"] = [pod.as_dict() for pod in self.pods]
        if self.returncode is not None:
            payload["returncode"] = self.returncode
        if self.stdout:
            payload["stdout"] = self.stdout
        if self.stderr:
            payload["stderr"] = self.stderr
        return payload


def _default_runner(
    command: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(  # noqa: PLW1510 - runner used synchronously
        list(command),
        env=merged_env,
        timeout=timeout,
        capture_output=True,
        text=True,
        check=False,
    )


def _parse_pod_statuses(payload: Mapping[str, object]) -> tuple[HelmPodStatus, ...]:
    items = payload.get("items", [])
    statuses: list[HelmPodStatus] = []
    if not isinstance(items, list):
        return tuple(statuses)

    for entry in items:
        if not isinstance(entry, Mapping):
            continue
        metadata = entry.get("metadata", {})
        status = entry.get("status", {})
        name = "<unknown>"
        if isinstance(metadata, Mapping):
            name = str(metadata.get("name") or name)
        ready = False
        reason: str | None = None

        if isinstance(status, Mapping):
            conditions = status.get("conditions", [])
            if isinstance(conditions, list):
                for condition in conditions:
                    if not isinstance(condition, Mapping):
                        continue
                    if condition.get("type") == "Ready":
                        ready = str(condition.get("status")).lower() == "true"
                        reason = (
                            str(condition.get("message"))
                            if condition.get("message")
                            else str(condition.get("reason")) if condition.get("reason") else reason
                        )
                        break
            if not ready:
                container_statuses = status.get("containerStatuses", [])
                waiting_reasons: list[str] = []
                if isinstance(container_statuses, list):
                    for container in container_statuses:
                        if not isinstance(container, Mapping):
                            continue
                        state = container.get("state", {})
                        if not isinstance(state, Mapping):
                            continue
                        waiting = state.get("waiting")
                        terminated = state.get("terminated")
                        if isinstance(waiting, Mapping):
                            reason_value = waiting.get("reason") or waiting.get("message")
                            if reason_value:
                                waiting_reasons.append(str(reason_value))
                        if isinstance(terminated, Mapping):
                            reason_value = terminated.get("reason") or terminated.get("message")
                            if reason_value:
                                waiting_reasons.append(str(reason_value))
                if waiting_reasons:
                    reason = ", ".join(waiting_reasons)
                elif reason is None:
                    phase = status.get("phase")
                    if phase:
                        reason = str(phase)

        statuses.append(HelmPodStatus(name=name, ready=ready, reason=reason))

    return tuple(statuses)


def _wait_for_pods_ready(
    config: HelmFailoverReplayConfig,
    *,
    runner: CommandRunner,
    sleep_fn: Callable[[float], None],
    monotonic_fn: Callable[[], float],
) -> tuple[HelmPodStatus, ...]:
    command = config.kubectl_get_pods_command()
    deadline = monotonic_fn() + config.max_wait_seconds
    last_stdout = ""
    last_stderr = ""

    while True:
        proc = runner(command, env=None, timeout=config.kubectl_timeout)
        last_stdout = proc.stdout or ""
        last_stderr = proc.stderr or ""

        if proc.returncode != 0:
            raise HelmFailoverReplayError(
                f"kubectl get pods failed with exit code {proc.returncode}",
                returncode=proc.returncode,
                stdout=last_stdout,
                stderr=last_stderr,
            )

        try:
            payload = json.loads(last_stdout or "{}")
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
            raise HelmFailoverReplayError(
                "Failed to parse JSON output from kubectl get pods",
                stdout=last_stdout,
                stderr=last_stderr,
            ) from exc

        pods = _parse_pod_statuses(payload)
        if not pods:
            raise HelmFailoverReplayError(
                "kubectl get pods did not return any pods",
                stdout=last_stdout,
                stderr=last_stderr,
            )

        not_ready = [pod for pod in pods if not pod.ready]
        if not not_ready:
            return pods

        if monotonic_fn() >= deadline:
            names = ", ".join(pod.name for pod in not_ready)
            raise HelmFailoverReplayError(
                f"Timed out waiting for pods to become ready: {names}",
                pods=pods,
                stdout=last_stdout,
                stderr=last_stderr,
            )

        sleep_fn(config.poll_interval_seconds)


def run_helm_failover_replay_smoke(
    config: HelmFailoverReplayConfig,
    *,
    runner: CommandRunner = _default_runner,
    sleep_fn: Callable[[float], None] = time.sleep,
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> HelmFailoverReplayResult:
    """Execute the Helm failover smoke test and return the result."""

    started_at = datetime.now(tz=UTC)
    pods = _wait_for_pods_ready(
        config,
        runner=runner,
        sleep_fn=sleep_fn,
        monotonic_fn=monotonic_fn,
    )

    replay_command = config.resolve_replay_command()
    proc = runner(
        replay_command,
        env=config.replay_environment,
        timeout=config.replay_timeout,
    )

    if proc.returncode != 0:
        raise HelmFailoverReplayError(
            f"Historical replay command failed with exit code {proc.returncode}",
            pods=pods,
            returncode=proc.returncode,
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
        )

    completed_at = datetime.now(tz=UTC)
    return HelmFailoverReplayResult(
        release=config.release,
        namespace=config.namespace,
        selector=config.selector,
        pods=pods,
        replay_command=replay_command,
        replay_stdout=proc.stdout or "",
        replay_stderr=proc.stderr or "",
        replay_returncode=proc.returncode,
        started_at=started_at,
        completed_at=completed_at,
    )
