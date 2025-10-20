from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Iterator

import pytest

from src.deployment import (
    HelmFailoverReplayConfig,
    HelmFailoverReplayError,
    HelmPodStatus,
    run_helm_failover_replay_smoke,
)


def _completed_process(
    command: Sequence[str],
    *,
    stdout: str = "",
    stderr: str = "",
    returncode: int = 0,
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(list(command), returncode, stdout, stderr)


class FakeRunner:
    def __init__(self, responses: dict[tuple[str, ...], list[subprocess.CompletedProcess[str]]]) -> None:
        self._responses = defaultdict(list)
        for command, items in responses.items():
            self._responses[command].extend(items)
        self.calls: list[tuple[tuple[str, ...], Mapping[str, str] | None, float | None]] = []

    def __call__(
        self,
        command: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        key = tuple(command)
        self.calls.append((key, env, timeout))
        bucket = self._responses.get(key)
        if not bucket:
            raise AssertionError(f"Unexpected command: {command}")
        response = bucket.pop(0)
        return response


def _monotonic(values: Sequence[float]) -> Iterator[float]:
    last = 0.0
    for value in values:
        last = value
        yield value
    while True:
        yield last


def _monotonic_fn(sequence: Sequence[float]) -> Callable[[], float]:
    iterator = _monotonic(sequence)

    def _next() -> float:
        return next(iterator)

    return _next


def test_run_helm_failover_smoke_success() -> None:
    config = HelmFailoverReplayConfig(release="emp", namespace="emp-system")
    kubectl_cmd = config.kubectl_get_pods_command()
    replay_cmd = config.resolve_replay_command()

    payload = {
        "items": [
            {
                "metadata": {"name": "emp-runtime-0"},
                "status": {
                    "conditions": [
                        {"type": "Ready", "status": "True"},
                    ],
                },
            }
        ]
    }

    runner = FakeRunner(
        {
            kubectl_cmd: [_completed_process(kubectl_cmd, stdout=json.dumps(payload))],
            replay_cmd: [_completed_process(replay_cmd, stdout="replay ok")],
        }
    )

    result = run_helm_failover_replay_smoke(
        config,
        runner=runner,
        sleep_fn=lambda _: None,
        monotonic_fn=_monotonic_fn([0.0]),
    )

    assert result.pods_ready()
    assert [pod.name for pod in result.pods] == ["emp-runtime-0"]
    assert result.replay_stdout == "replay ok"
    assert runner.calls[0][0] == kubectl_cmd
    assert runner.calls[1][0] == replay_cmd


def test_run_helm_failover_waits_for_readiness() -> None:
    config = HelmFailoverReplayConfig(
        release="emp",
        namespace="emp-system",
        poll_interval_seconds=2.5,
    )
    kubectl_cmd = config.kubectl_get_pods_command()
    replay_cmd = config.resolve_replay_command()

    not_ready_payload = {
        "items": [
            {
                "metadata": {"name": "emp-runtime-0"},
                "status": {
                    "conditions": [
                        {
                            "type": "Ready",
                            "status": "False",
                            "reason": "ContainersNotReady",
                        }
                    ],
                    "phase": "Running",
                },
            }
        ]
    }
    ready_payload = {
        "items": [
            {
                "metadata": {"name": "emp-runtime-0"},
                "status": {
                    "conditions": [
                        {
                            "type": "Ready",
                            "status": "True",
                        }
                    ],
                },
            }
        ]
    }

    runner = FakeRunner(
        {
            kubectl_cmd: [
                _completed_process(kubectl_cmd, stdout=json.dumps(not_ready_payload)),
                _completed_process(kubectl_cmd, stdout=json.dumps(ready_payload)),
            ],
            replay_cmd: [_completed_process(replay_cmd, stdout="done")],
        }
    )

    sleeps: list[float] = []

    result = run_helm_failover_replay_smoke(
        config,
        runner=runner,
        sleep_fn=lambda seconds: sleeps.append(seconds),
        monotonic_fn=_monotonic_fn([0.0, 1.0]),
    )

    assert result.pods_ready()
    assert len(result.pods) == 1
    assert sleeps == [config.poll_interval_seconds]
    assert runner.calls[0][0] == kubectl_cmd
    assert runner.calls[1][0] == kubectl_cmd
    assert runner.calls[2][0] == replay_cmd


def test_run_helm_failover_times_out_when_pods_never_ready() -> None:
    config = HelmFailoverReplayConfig(
        release="emp",
        namespace="emp-system",
        max_wait_seconds=5.0,
        poll_interval_seconds=1.0,
    )
    kubectl_cmd = config.kubectl_get_pods_command()

    not_ready_payload = {
        "items": [
            {
                "metadata": {"name": "emp-runtime-0"},
                "status": {
                    "conditions": [
                        {
                            "type": "Ready",
                            "status": "False",
                            "reason": "CrashLoopBackOff",
                        }
                    ],
                    "phase": "CrashLoopBackOff",
                },
            }
        ]
    }

    runner = FakeRunner(
        {
            kubectl_cmd: [_completed_process(kubectl_cmd, stdout=json.dumps(not_ready_payload))],
        }
    )

    with pytest.raises(HelmFailoverReplayError) as excinfo:
        run_helm_failover_replay_smoke(
            config,
            runner=runner,
            sleep_fn=lambda _: None,
            monotonic_fn=_monotonic_fn([0.0, 6.0]),
        )

    error = excinfo.value
    assert "Timed out" in str(error)
    assert error.pods
    assert error.pods[0] == HelmPodStatus(
        name="emp-runtime-0",
        ready=False,
        reason="CrashLoopBackOff",
    )


def test_run_helm_failover_raises_on_replay_failure() -> None:
    config = HelmFailoverReplayConfig(release="emp", namespace="emp-system")
    kubectl_cmd = config.kubectl_get_pods_command()
    replay_cmd = config.resolve_replay_command()

    ready_payload = {
        "items": [
            {
                "metadata": {"name": "emp-runtime-0"},
                "status": {
                    "conditions": [
                        {"type": "Ready", "status": "True"},
                    ],
                },
            }
        ]
    }

    runner = FakeRunner(
        {
            kubectl_cmd: [_completed_process(kubectl_cmd, stdout=json.dumps(ready_payload))],
            replay_cmd: [
                _completed_process(
                    replay_cmd,
                    stdout="starting",
                    stderr="boom",
                    returncode=2,
                )
            ],
        }
    )

    with pytest.raises(HelmFailoverReplayError) as excinfo:
        run_helm_failover_replay_smoke(
            config,
            runner=runner,
            sleep_fn=lambda _: None,
            monotonic_fn=_monotonic_fn([0.0]),
        )

    error = excinfo.value
    assert error.returncode == 2
    assert error.stdout == "starting"
    assert error.stderr == "boom"
    assert error.pods[0].ready is True


def test_config_sets_default_selector_and_command() -> None:
    config = HelmFailoverReplayConfig(release="predator", namespace="ops")
    assert config.selector == "app.kubernetes.io/instance=predator"

    replay_cmd = config.resolve_replay_command()
    assert replay_cmd[0] == config.kubectl
    assert f"deploy/{config.release}" in replay_cmd
    assert "--summary-path" in replay_cmd