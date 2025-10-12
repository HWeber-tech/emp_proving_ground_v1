"""Helpers for evaluating EMP mini-cycle experiment outcomes.

The tests exercise the high level logic that powers the pseudo code found in
internal run-books.  The real production system wires these helpers into a
larger orchestration layer but for the kata we only need the decision logic.

Each helper receives a raw ``comparison`` mapping that may already contain
ratios, deltas or deeply nested dictionaries depending on the caller.  The
functions therefore attempt to coerce the provided structure into a uniform
representation before checking the success criteria.

The module intentionally keeps the public return shape lightweight – a simple
``dict`` with ``ok`` (bool) plus human readable details.  This mirrors the
expectations of the CLI tools that surface the decision results to operators.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional

Number = float | int


@dataclass
class MetricComparison:
    """Container describing how a metric performed against the reference run."""

    reference: Optional[float]
    candidate: Optional[float]
    ratio: Optional[float]
    delta: Optional[float]
    pct_change: Optional[float]
    extras: Dict[str, Any] = field(default_factory=dict)

    def ensure_ratio(self) -> Optional[float]:
        if self.ratio is not None:
            return self.ratio
        if self.reference in (None, 0) or self.candidate is None:
            return None
        self.ratio = self.candidate / self.reference
        return self.ratio

    def ensure_pct_change(self) -> Optional[float]:
        if self.pct_change is not None:
            return self.pct_change
        if self.reference in (None, 0) or self.candidate is None:
            return None
        self.pct_change = (self.candidate - self.reference) / self.reference * 100.0
        return self.pct_change


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _to_float(value: Optional[Number]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    return float(value)


def _parse_metric(raw: Any) -> MetricComparison:
    """Normalise a metric description coming from comparison helpers."""

    reference: Optional[float] = None
    candidate: Optional[float] = None
    ratio: Optional[float] = None
    delta: Optional[float] = None
    pct_change: Optional[float] = None
    extras: Dict[str, Any] = {}

    if isinstance(raw, Mapping):
        mapping = dict(raw)
        for key in ("reference", "baseline", "control", "ref"):
            if key in mapping and _is_number(mapping[key]):
                reference = _to_float(mapping[key])
                break
        for key in ("candidate", "value", "treatment", "variant", "experiment"):
            if key in mapping and _is_number(mapping[key]):
                candidate = _to_float(mapping[key])
                break
        if "ratio" in mapping and _is_number(mapping["ratio"]):
            ratio = _to_float(mapping["ratio"])
        if "delta" in mapping and _is_number(mapping["delta"]):
            delta = _to_float(mapping["delta"])
        for key in ("pct_change", "percent_change", "percentage_change"):
            if key in mapping and _is_number(mapping[key]):
                pct_change = _to_float(mapping[key])
                break
        known = {
            "reference",
            "baseline",
            "control",
            "ref",
            "candidate",
            "value",
            "treatment",
            "variant",
            "experiment",
            "ratio",
            "delta",
            "pct_change",
            "percent_change",
            "percentage_change",
        }
        extras = {k: v for k, v in mapping.items() if k not in known}
    elif isinstance(raw, (list, tuple)):
        items = list(raw)
        if items:
            reference = _to_float(items[0]) if _is_number(items[0]) else None
        if len(items) >= 2 and _is_number(items[1]):
            candidate = _to_float(items[1])
        if len(items) >= 3 and _is_number(items[2]):
            ratio = _to_float(items[2])
    elif _is_number(raw):
        candidate = _to_float(raw)
    metric = MetricComparison(reference, candidate, ratio, delta, pct_change, extras)
    metric.ensure_ratio()
    if metric.delta is None and metric.reference is not None and metric.candidate is not None:
        metric.delta = metric.candidate - metric.reference
    metric.ensure_pct_change()
    return metric


def _flatten_metrics(comparison: Mapping[str, Any]) -> Dict[str, MetricComparison]:
    """Extract leaf metrics from arbitrarily nested comparison payloads."""

    flat: Dict[str, MetricComparison] = {}

    def _walk(prefix: str, node: Any) -> None:
        if isinstance(node, Mapping):
            # If this mapping looks like a metric description, parse it.
            if any(
                key in node
                for key in (
                    "reference",
                    "baseline",
                    "control",
                    "ref",
                    "candidate",
                    "value",
                    "treatment",
                    "variant",
                    "experiment",
                    "ratio",
                    "delta",
                    "pct_change",
                    "percent_change",
                    "percentage_change",
                )
            ):
                key_name = prefix.rstrip(".")
                flat[key_name] = _parse_metric(node)
                return
            for sub_key, sub_value in node.items():
                new_prefix = f"{prefix}.{sub_key}" if prefix else str(sub_key)
                _walk(new_prefix, sub_value)
        else:
            if prefix:
                flat[prefix] = _parse_metric(node)

    _walk("", comparison)
    return flat


def _lookup_metric(metrics: Mapping[str, MetricComparison], names: Iterable[str]) -> MetricComparison:
    for name in names:
        if name in metrics:
            return metrics[name]
    for name in names:
        simplified = name.replace(".", "_")
        for key, value in metrics.items():
            key_simple = key.replace(".", "_")
            if key_simple == simplified or key_simple.endswith(simplified) or simplified.endswith(key_simple):
                return value
    return MetricComparison(None, None, None, None, None, {})


def _result_template() -> Dict[str, Any]:
    return {"ok": True, "passed": {}, "failed": {}, "abort": False, "abort_reasons": []}


def evaluate_lion_success(
    comparison: Mapping[str, Any],
    criteria: Mapping[str, Number],
    abort_rules: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate the Lion optimizer pilot results against the acceptance rules."""

    abort_rules = abort_rules or {}
    metrics = _flatten_metrics(comparison)
    result = _result_template()

    def _check_ratio(metric_names: Iterable[str], threshold: float, label: str) -> None:
        metric = _lookup_metric(metrics, metric_names)
        ratio = metric.ensure_ratio()
        detail = {"ratio": ratio, "threshold": threshold}
        if ratio is not None and ratio <= threshold + 1e-12:
            result["passed"][label] = detail
        else:
            result["failed"][label] = detail
            result["ok"] = False

    _check_ratio(["val_loss", "validation_loss", "metrics.val_loss"], float(criteria.get("val_loss_ratio_vs_adam", float("inf"))), "val_loss_ratio_vs_adam")
    if "steps_ratio_vs_adam" in criteria:
        _check_ratio(["steps", "num_steps", "metrics.steps"], float(criteria["steps_ratio_vs_adam"]), "steps_ratio_vs_adam")
    if "gpu_mem_ratio_vs_adam" in criteria:
        _check_ratio(["gpu_mem", "gpu_mem_gb", "gpu_memory_gb", "memory.gpu"], float(criteria["gpu_mem_ratio_vs_adam"]), "gpu_mem_ratio_vs_adam")

    wall_clock_metric = _lookup_metric(metrics, ["wall_clock_s", "duration_s", "runtime.wall_clock"])
    if "time_overhead_vs_adam_pct" in abort_rules:
        pct_limit = float(abort_rules["time_overhead_vs_adam_pct"])
        ratio = wall_clock_metric.ensure_ratio()
        if ratio is not None and ratio - 1.0 > pct_limit / 100.0 + 1e-12:
            result["abort"] = True
            result["ok"] = False
            result["abort_reasons"].append(
                {
                    "rule": "time_overhead_vs_adam_pct",
                    "ratio": ratio,
                    "limit": pct_limit,
                }
            )

    if abort_rules.get("val_loss_worse_after_2x_steps"):
        val_metric = _lookup_metric(metrics, ["val_loss", "validation_loss", "metrics.val_loss"])
        triggered = False
        if val_metric.extras:
            for key, value in val_metric.extras.items():
                if isinstance(value, (int, float)) and "2x" in key:
                    reference = val_metric.reference
                    if reference is not None and value > reference + 1e-12:
                        triggered = True
                        break
        if triggered:
            result["abort"] = True
            result["ok"] = False
            result["abort_reasons"].append({"rule": "val_loss_worse_after_2x_steps"})

    return result


def evaluate_flash_success(
    comparison: Mapping[str, Any],
    throughput_gain_min: float,
    val_loss_degradation_max_pct: float,
) -> Dict[str, Any]:
    """Assess FlashAttention pilot results."""

    metrics = _flatten_metrics(comparison)
    result = _result_template()

    throughput_metric = _lookup_metric(
        metrics,
        [
            "throughput_samples_per_s",
            "throughput",
            "metrics.throughput_samples_per_s",
            "perf.throughput",
        ],
    )
    throughput_ratio = throughput_metric.ensure_ratio()
    if throughput_ratio is not None and throughput_ratio >= throughput_gain_min - 1e-12:
        result["passed"]["throughput_gain"] = {
            "ratio": throughput_ratio,
            "threshold": throughput_gain_min,
        }
    else:
        result["failed"]["throughput_gain"] = {
            "ratio": throughput_ratio,
            "threshold": throughput_gain_min,
        }
        result["ok"] = False

    val_metric = _lookup_metric(metrics, ["val_loss", "validation_loss", "metrics.val_loss"])
    val_pct = val_metric.ensure_pct_change()
    if val_pct is not None and val_pct <= val_loss_degradation_max_pct + 1e-12:
        result["passed"]["val_loss_degradation_pct"] = {
            "pct_change": val_pct,
            "threshold": val_loss_degradation_max_pct,
        }
    else:
        result["failed"]["val_loss_degradation_pct"] = {
            "pct_change": val_pct,
            "threshold": val_loss_degradation_max_pct,
        }
        result["ok"] = False

    return result


def evaluate_quant_success(
    comparison: Mapping[str, Any],
    max_val_loss_degradation_pct: float,
    min_latency_gain: float,
) -> Dict[str, Any]:
    """Evaluate a quantised inference experiment against degradation limits."""

    metrics = _flatten_metrics(comparison)
    result = _result_template()

    val_metric = _lookup_metric(metrics, ["val_loss", "validation_loss", "metrics.val_loss"])
    val_pct = val_metric.ensure_pct_change()
    if val_pct is not None and val_pct <= max_val_loss_degradation_pct + 1e-12:
        result["passed"]["val_loss_degradation_pct"] = {
            "pct_change": val_pct,
            "threshold": max_val_loss_degradation_pct,
        }
    else:
        result["failed"]["val_loss_degradation_pct"] = {
            "pct_change": val_pct,
            "threshold": max_val_loss_degradation_pct,
        }
        result["ok"] = False

    latency_metric = _lookup_metric(
        metrics,
        ["latency_ms_per_batch", "latency", "eval.latency_ms_per_batch"],
    )
    latency_ratio: Optional[float] = None
    latency_gain: Optional[float] = None
    ratio = latency_metric.ensure_ratio()
    if ratio is not None and ratio > 0:
        latency_ratio = ratio
        latency_gain = 1.0 / ratio
    elif latency_metric.reference in (None, 0) or latency_metric.candidate in (None, 0):
        latency_ratio = None
    else:
        latency_ratio = latency_metric.ensure_ratio()
        if latency_ratio not in (None, 0):
            latency_gain = 1.0 / latency_ratio

    if latency_gain is None:
        throughput_metric = _lookup_metric(
            metrics,
            ["throughput_samples_per_s", "throughput", "metrics.throughput_samples_per_s"],
        )
        throughput_ratio = throughput_metric.ensure_ratio()
        if throughput_ratio is not None and throughput_ratio > 0:
            latency_gain = throughput_ratio

    if latency_gain is not None and latency_gain >= min_latency_gain - 1e-12:
        result["passed"]["latency_gain"] = {
            "gain": latency_gain,
            "threshold": min_latency_gain,
            "source": "latency" if latency_metric.reference is not None else "throughput",
        }
    else:
        result["failed"]["latency_gain"] = {
            "gain": latency_gain,
            "threshold": min_latency_gain,
        }
        result["ok"] = False

    return result


__all__ = [
    "evaluate_lion_success",
    "evaluate_flash_success",
    "evaluate_quant_success",
]
