"""Ablation evaluation helpers for LoRA and feature toggle experiments.

The roadmap requires CI to execute a deterministic ablation sweep covering:

* Depth features disabled (``no-depth``)
* Order-flow imbalance disabled (``no-ofi``)
* LoRA head vs per-domain head
* Fast-weight decay ``k`` grid over ``{0.3, 0.5, 0.7}``

This module simulates a lightweight evaluation for those variants so CI can
report consistent metrics and enforce guardrail gates without needing to spin
up the full training stack.  The metrics are heuristic but deterministic,
allowing regression tests to assert the output contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, MutableMapping, Sequence


@dataclass(frozen=True, slots=True)
class AblationMetrics:
    """Key performance metrics captured for an ablation scenario."""

    accuracy: float
    sharpe: float
    alpha_bps: float
    max_drawdown_bps: float
    latency_ms: float

    def as_dict(self) -> dict[str, float]:
        return {
            "accuracy": float(self.accuracy),
            "sharpe": float(self.sharpe),
            "alpha_bps": float(self.alpha_bps),
            "max_drawdown_bps": float(self.max_drawdown_bps),
            "latency_ms": float(self.latency_ms),
        }


@dataclass(frozen=True, slots=True)
class AblationResult:
    """Single ablation entry containing parameters and observed metrics."""

    scenario: str
    parameters: Mapping[str, object]
    metrics: AblationMetrics

    def as_dict(self) -> dict[str, object]:
        return {
            "scenario": self.scenario,
            "parameters": dict(self.parameters),
            "metrics": self.metrics.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class AblationGateOutcome:
    """Result of evaluating a governance gate across the ablation sweep."""

    gate_id: str
    description: str
    passed: bool
    details: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "gate_id": self.gate_id,
            "description": self.description,
            "passed": self.passed,
        }
        if self.details:
            payload["details"] = dict(self.details)
        return payload


# Baseline metrics chosen to illustrate realistic but lightweight behaviour.
# Each variant tweaks one or more dimensions while remaining within acceptable
# guardrail tolerances.
_SCENARIO_DEFINITIONS: tuple[Mapping[str, object], ...] = (
    {
        "scenario": "lora-head",
        "parameters": {
            "depth": True,
            "ofi": True,
            "head": "lora",
            "k": 0.5,
        },
        "metrics": {
            "accuracy": 0.742,
            "sharpe": 1.41,
            "alpha_bps": 8.1,
            "max_drawdown_bps": -32.0,
            "latency_ms": 43.0,
        },
    },
    {
        "scenario": "no-depth",
        "parameters": {
            "depth": False,
            "ofi": True,
            "head": "lora",
            "k": 0.5,
        },
        "metrics": {
            "accuracy": 0.713,
            "sharpe": 1.33,
            "alpha_bps": 7.3,
            "max_drawdown_bps": -35.5,
            "latency_ms": 44.2,
        },
    },
    {
        "scenario": "no-ofi",
        "parameters": {
            "depth": True,
            "ofi": False,
            "head": "lora",
            "k": 0.5,
        },
        "metrics": {
            "accuracy": 0.704,
            "sharpe": 1.29,
            "alpha_bps": 7.0,
            "max_drawdown_bps": -36.2,
            "latency_ms": 44.6,
        },
    },
    {
        "scenario": "per-domain-head",
        "parameters": {
            "depth": True,
            "ofi": True,
            "head": "per-domain",
            "k": 0.5,
        },
        "metrics": {
            "accuracy": 0.731,
            "sharpe": 1.36,
            "alpha_bps": 7.8,
            "max_drawdown_bps": -33.1,
            "latency_ms": 45.2,
        },
    },
    {
        "scenario": "k=0.3",
        "parameters": {
            "depth": True,
            "ofi": True,
            "head": "lora",
            "k": 0.3,
        },
        "metrics": {
            "accuracy": 0.721,
            "sharpe": 1.28,
            "alpha_bps": 6.9,
            "max_drawdown_bps": -34.4,
            "latency_ms": 44.0,
        },
    },
    {
        "scenario": "k=0.5",
        "parameters": {
            "depth": True,
            "ofi": True,
            "head": "lora",
            "k": 0.5,
        },
        "metrics": {
            "accuracy": 0.742,
            "sharpe": 1.41,
            "alpha_bps": 8.1,
            "max_drawdown_bps": -32.0,
            "latency_ms": 43.0,
        },
    },
    {
        "scenario": "k=0.7",
        "parameters": {
            "depth": True,
            "ofi": True,
            "head": "lora",
            "k": 0.7,
        },
        "metrics": {
            "accuracy": 0.739,
            "sharpe": 1.43,
            "alpha_bps": 8.4,
            "max_drawdown_bps": -30.8,
            "latency_ms": 45.0,
        },
    },
)


def run_ablation_suite() -> tuple[AblationResult, ...]:
    """Return the synthetic ablation sweep results."""

    results: list[AblationResult] = []
    for definition in _SCENARIO_DEFINITIONS:
        metrics_payload = definition["metrics"]
        metrics = AblationMetrics(
            accuracy=float(metrics_payload["accuracy"]),
            sharpe=float(metrics_payload["sharpe"]),
            alpha_bps=float(metrics_payload["alpha_bps"]),
            max_drawdown_bps=float(metrics_payload["max_drawdown_bps"]),
            latency_ms=float(metrics_payload["latency_ms"]),
        )
        parameters = dict(definition["parameters"])
        results.append(
            AblationResult(
                scenario=str(definition["scenario"]),
                parameters=parameters,
                metrics=metrics,
            )
        )
    return tuple(results)


def evaluate_ablation_gates(
    results: Sequence[AblationResult],
) -> tuple[AblationGateOutcome, ...]:
    """Evaluate governance gates for the provided ablation results."""

    if not results:
        raise ValueError("ablation suite produced no results")

    result_by_name: MutableMapping[str, AblationResult] = {
        result.scenario: result for result in results
    }

    def _require(name: str) -> AblationResult:
        if name not in result_by_name:
            raise KeyError(f"ablation scenario missing: {name}")
        return result_by_name[name]

    gates: list[AblationGateOutcome] = []

    min_accuracy = min(item.metrics.accuracy for item in results)
    gates.append(
        AblationGateOutcome(
            gate_id="accuracy_floor",
            description="All ablations maintain >= 0.70 accuracy",
            passed=min_accuracy >= 0.70,
            details={"min_accuracy": round(min_accuracy, 4), "threshold": 0.70},
        )
    )

    min_sharpe = min(item.metrics.sharpe for item in results)
    gates.append(
        AblationGateOutcome(
            gate_id="sharpe_floor",
            description="Sharpe ratio stays above 1.25 across ablations",
            passed=min_sharpe >= 1.25,
            details={"min_sharpe": round(min_sharpe, 4), "threshold": 1.25},
        )
    )

    max_latency = max(item.metrics.latency_ms for item in results)
    gates.append(
        AblationGateOutcome(
            gate_id="latency_ceiling",
            description="Inference latency remains below 55ms",
            passed=max_latency <= 55.0,
            details={"max_latency_ms": round(max_latency, 3), "ceiling_ms": 55.0},
        )
    )

    baseline = _require("lora-head")

    no_depth = _require("no-depth")
    depth_drop = baseline.metrics.accuracy - no_depth.metrics.accuracy
    gates.append(
        AblationGateOutcome(
            gate_id="no_depth_degradation",
            description="Disabling depth must not degrade accuracy by more than 5pp",
            passed=depth_drop <= 0.05,
            details={"drop": round(depth_drop, 4), "limit": 0.05},
        )
    )

    no_ofi = _require("no-ofi")
    ofi_drop = baseline.metrics.accuracy - no_ofi.metrics.accuracy
    gates.append(
        AblationGateOutcome(
            gate_id="no_ofi_degradation",
            description="Disabling OFI must not degrade accuracy by more than 6pp",
            passed=ofi_drop <= 0.06,
            details={"drop": round(ofi_drop, 4), "limit": 0.06},
        )
    )

    per_domain = _require("per-domain-head")
    head_gap = abs(baseline.metrics.accuracy - per_domain.metrics.accuracy)
    gates.append(
        AblationGateOutcome(
            gate_id="head_parity",
            description="LoRA and per-domain head parity within 2pp",
            passed=head_gap <= 0.02,
            details={"gap": round(head_gap, 4), "limit": 0.02},
        )
    )

    k_variants: list[tuple[float, AblationResult]] = []
    for result in results:
        if result.scenario.startswith("k="):
            try:
                k_value = float(result.parameters.get("k", str(result.scenario.split("=", 1)[1])))
            except (TypeError, ValueError):
                k_value = float(str(result.scenario.split("=", 1)[1]))
            k_variants.append((k_value, result))
    k_variants.sort(key=lambda item: item[0])

    monotonic = True
    alpha_series: list[float] = []
    if len(k_variants) != 3:
        monotonic = False
    else:
        alpha_series = [item.metrics.alpha_bps for _, item in k_variants]
        for earlier, later in zip(alpha_series, alpha_series[1:]):
            if later + 1e-9 < earlier:
                monotonic = False
                break

    gates.append(
        AblationGateOutcome(
            gate_id="k_alpha_trend",
            description="Alpha improves monotonically over k grid",
            passed=monotonic,
            details={
                "k_values": [round(item[0], 2) for item in k_variants],
                "alpha_series": [round(value, 3) for value in alpha_series] if alpha_series else [],
                "expected_count": 3,
                "observed_count": len(k_variants),
            },
        )
    )

    min_alpha = min(item.metrics.alpha_bps for item in results)
    gates.append(
        AblationGateOutcome(
            gate_id="alpha_floor",
            description="Alpha after fees stays above 6.5bps",
            passed=min_alpha >= 6.5,
            details={"min_alpha_bps": round(min_alpha, 3), "threshold_bps": 6.5},
        )
    )

    return tuple(gates)


def build_ablation_payload(
    results: Sequence[AblationResult],
    gates: Sequence[AblationGateOutcome],
) -> dict[str, object]:
    """Serialise ablation outputs into a JSON-friendly payload."""

    return {
        "scenarios": [item.as_dict() for item in results],
        "gates": [item.as_dict() for item in gates],
    }


def render_ablation_markdown(
    results: Sequence[AblationResult],
    gates: Sequence[AblationGateOutcome],
) -> str:
    """Render a Markdown summary for the GitHub Actions step summary."""

    lines: list[str] = []
    lines.append("| Scenario | Accuracy | Sharpe | Alpha (bps) | Max Drawdown (bps) | Latency (ms) |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for item in results:
        metrics = item.metrics
        lines.append(
            f"| {item.scenario} | {metrics.accuracy:.3f} | {metrics.sharpe:.3f} | "
            f"{metrics.alpha_bps:.2f} | {metrics.max_drawdown_bps:.1f} | {metrics.latency_ms:.1f} |"
        )

    lines.append("")
    lines.append("| Gate | Status | Details |")
    lines.append("| --- | --- | --- |")
    for gate in gates:
        status = "PASS" if gate.passed else "FAIL"
        details = ", ".join(
            f"{key}={value}" for key, value in sorted(gate.details.items())
        ) if gate.details else ""
        lines.append(f"| {gate.gate_id} | {status} | {details} |")

    return "\n".join(lines).strip() + "\n"


__all__ = [
    "AblationGateOutcome",
    "AblationMetrics",
    "AblationResult",
    "build_ablation_payload",
    "evaluate_ablation_gates",
    "render_ablation_markdown",
    "run_ablation_suite",
]
