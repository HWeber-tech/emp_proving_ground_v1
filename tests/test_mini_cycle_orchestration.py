from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pytest

import emp.experiments.mini_cycle as mini_cycle
import emp.experiments.mini_cycle_orchestration as orchestration


class FakeModule:
    def __init__(self, name: str) -> None:
        self.name = name
        self.ensure_calls: List[Dict[str, Any]] = []

    def ensure_installed(self, deps: Iterable[str]) -> None:
        self.ensure_calls.append({"deps": list(deps)})


class FakeMemoryStore:
    def __init__(self) -> None:
        self.entries: List[Dict[str, Any]] = []
        self.add_calls: List[List[Dict[str, Any]]] = []
        self.search_calls: List[Dict[str, Any]] = []

    def add(self, batch: List[Dict[str, Any]]) -> None:
        copied: List[Dict[str, Any]] = []
        for item in batch:
            copied.append(
                {
                    "emb": list(item.get("emb", [])),
                    "outcome": float(item.get("outcome", 0.0)),
                    "note": str(item.get("note", "")),
                    "meta": dict(item.get("meta", {})),
                }
            )
        self.add_calls.append(copied)
        self.entries.extend(copied)

    def search(
        self,
        embedding: Any,
        topk: int,
        where: Optional[Dict[str, Any]] = None,
        recency_half_life_days: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        self.search_calls.append(
            {
                "embedding": embedding,
                "topk": topk,
                "where": dict(where or {}),
                "recency_half_life_days": recency_half_life_days,
            }
        )
        matches: List[Dict[str, Any]] = []
        for entry in self.entries:
            meta = entry.get("meta", {})
            if where and any(meta.get(key) != value for key, value in where.items()):
                continue
            matches.append(
                {
                    "outcome": entry.get("outcome", 0.0),
                    "note": entry.get("note", ""),
                    "meta": dict(meta),
                }
            )
        limit = max(0, int(topk)) if topk is not None else len(matches)
        return matches[:limit]


class FakeMemoryNamespace:
    def __init__(self, parent: "FakeEMP") -> None:
        self.parent = parent
        self.created: List[Dict[str, Any]] = []

    def create_store(self, **kwargs: Any) -> FakeMemoryStore:
        store = FakeMemoryStore()
        self.created.append(dict(kwargs))
        self.parent.memory_store = store
        return store


class FakeInferenceAPI:
    def __init__(self) -> None:
        self.feature_hooks: List[Any] = []

    def register_feature_hook(self, hook: Any) -> None:
        self.feature_hooks.append(hook)

    def run_feature_hooks(self, context: Dict[str, Any]) -> Dict[str, Any]:
        aggregated: Dict[str, Any] = {}
        for hook in self.feature_hooks:
            update = hook(context)
            if update:
                aggregated.update(update)
        return aggregated


class FakePlannerAPI:
    def __init__(self) -> None:
        self.context_hooks: List[Any] = []

    def register_context_hook(self, hook: Any) -> None:
        self.context_hooks.append(hook)

    def run_context_hooks(self, context: Dict[str, Any]) -> str:
        notes: List[str] = []
        for hook in self.context_hooks:
            note = hook(context)
            if note:
                notes.append(note)
        return "\n".join(notes)


class FakeStatsNamespace:
    def weighted(self, values: Sequence[float], weights: Sequence[float]) -> Dict[str, float]:
        pairs = [(float(v), float(w)) for v, w in zip(values, weights)]
        total_weight = sum(weight for _, weight in pairs)
        if total_weight == 0:
            mean = 0.0
        else:
            mean = sum(value * weight for value, weight in pairs) / total_weight
        return {"mean": mean}

    def hitrate(self, values: Sequence[int]) -> float:
        if not values:
            return 0.0
        return float(sum(values)) / float(len(values))

    def percentile(self, values: Sequence[float], pct: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(float(v) for v in values)
        if len(ordered) == 1:
            return ordered[0]
        pct = max(0.0, min(100.0, float(pct)))
        index = round(pct / 100.0 * (len(ordered) - 1))
        return ordered[int(index)]


class FakeTimeNamespace:
    def __init__(self) -> None:
        self.now = datetime(2024, 1, 10)

    def recency_weight(self, ts: Any, *, half_life_days: float) -> float:
        if isinstance(ts, datetime):
            age_days = (self.now - ts).days
        else:
            try:
                age_days = float(ts)
            except (TypeError, ValueError):
                age_days = 0.0
        if half_life_days <= 0:
            return 1.0
        return 0.5 ** (age_days / half_life_days)


class FakeDataNamespace:
    def __init__(self, parent: "FakeEMP") -> None:
        self.parent = parent

    def iter_windows(self, split: str, limit: int, fields: Sequence[str]):
        count = 0
        for window in self.parent.window_stream:
            if count >= limit:
                break
            payload: Dict[str, Any] = {}
            for field in fields:
                payload[field] = window.get(field)
            count += 1
            yield payload


class FakeLabelersNamespace:
    def future_outcome(self, metric: str, horizon: int, window: Dict[str, Any]) -> float:
        return float(window.get("future_outcome", 0.0))


class FakeRegimeModel:
    def __init__(self, parent: "FakeEMP") -> None:
        self.parent = parent

    def tag(self, context: Dict[str, Any]) -> int:
        return int(context.get("regime_tag", self.parent.default_regime_tag))


class FakeRegimeNamespace:
    def __init__(self, parent: "FakeEMP") -> None:
        self.parent = parent
        self.repeat_subset_results: List[Any] = []
        self.kmeans_calls: List[Dict[str, Any]] = []
        self.repeat_subset_calls: List[Dict[str, Any]] = []

    def kmeans(self, **kwargs: Any) -> FakeRegimeModel:
        self.kmeans_calls.append(dict(kwargs))
        return FakeRegimeModel(self.parent)

    def repeat_subset(self, **kwargs: Any) -> Any:
        self.repeat_subset_calls.append(dict(kwargs))
        if self.repeat_subset_results:
            return self.repeat_subset_results.pop(0)
        return []


class FakeEMP:
    def __init__(self) -> None:
        self.seed: Optional[int] = None
        self.project: Optional[str] = None
        self.run_group: Optional[str] = None
        self.env_vars: Dict[str, Any] = {}
        self.default_optimizer_name = "Adam"
        self.default_optimizer_config: Dict[str, Any] = {
            "type": "Adam",
            "lr": 3e-4,
            "betas": (0.9, 0.999),
            "weight_decay": 0.0,
        }
        self.default_inference_precision = 16
        self.kernel_state: Dict[str, bool] = {}
        self.kernel_calls: List[tuple[str, bool]] = []
        self.default_optimizer_calls: List[tuple[str, Dict[str, Any]]] = []
        self.default_inference_calls: List[int] = []
        self.compare_runs_payloads: List[Dict[str, Any]] = []
        self.compare_inference_payloads: List[Dict[str, Any]] = []
        self.tagged_runs: Dict[str, str] = {}
        self.logged_messages: List[tuple[str, str]] = []
        self.saved_summaries: List[str] = []
        self.exported_checkpoint: Optional[tuple[str, str]] = None
        self.run_history: List[Dict[str, Any]] = []
        self.modules: Dict[str, FakeModule] = {}
        self.memory = FakeMemoryNamespace(self)
        self.inference = FakeInferenceAPI()
        self.planner = FakePlannerAPI()
        self.stats = FakeStatsNamespace()
        self.time = FakeTimeNamespace()
        self.data = FakeDataNamespace(self)
        self.labelers = FakeLabelersNamespace()
        self.regime = FakeRegimeNamespace(self)
        self.memory_store: Optional[FakeMemoryStore] = None
        self.window_stream: List[Dict[str, Any]] = []
        self.evaluate_subset_results: List[Dict[str, Any]] = []
        self.evaluate_subset_calls: List[Dict[str, Any]] = []
        self.flags: Dict[str, bool] = {}
        self.default_regime_tag = 0

    def set_seed(self, value: int) -> None:
        self.seed = value

    def set_project(self, project: str) -> None:
        self.project = project

    def set_run_group(self, group: str) -> None:
        self.run_group = group

    def env(self, mapping: Dict[str, Any]) -> None:
        self.env_vars.update(mapping)

    def module(self, name: str) -> FakeModule:
        module = self.modules.get(name)
        if module is None:
            module = FakeModule(name)
            self.modules[name] = module
        return module

    def select_features(self, data: Dict[str, Any], names: Sequence[str]) -> List[float]:
        return [float(data.get(name, 0.0) or 0.0) for name in names]

    def embed(self, model: str, features: Sequence[Any]) -> List[float]:
        self.run_history.append({"embed_model": model, "feature_count": len(features)})
        return [float(value) for value in features]

    def run_experiment(self, label: str, **params: Any) -> str:
        record = {"label": label}
        record.update(params)
        self.run_history.append(record)
        return label

    def plot_curves(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - visual side effect
        pass

    def plot_bars(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - visual side effect
        pass

    def export_csv(self, run_id: str, out: str) -> None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(f"csv for {run_id}\n", encoding="utf-8")

    def save_summary(self, run_id: str, out: str) -> None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
        self.saved_summaries.append(out)

    def compare_runs(self, **_: Any) -> Dict[str, Any]:
        return self.compare_runs_payloads.pop(0)

    def evaluate_on_subset(
        self, run_id: str, *, subset: Any, metrics: Sequence[str]
    ) -> Dict[str, Any]:
        self.evaluate_subset_calls.append(
            {"run_id": run_id, "subset": subset, "metrics": list(metrics)}
        )
        if self.evaluate_subset_results:
            return self.evaluate_subset_results.pop(0)
        return {metric: 0.0 for metric in metrics}

    def tag_run(self, run_id: str, tag: str) -> None:
        self.tagged_runs[run_id] = tag

    def log(self, message: str, level: str = "info") -> None:
        self.logged_messages.append((level, message))

    def set_default_optimizer(self, name: str, config: Dict[str, Any]) -> None:
        self.default_optimizer_name = name
        self.default_optimizer_config = dict(config)
        self.default_optimizer_calls.append((name, dict(config)))

    def get_default_optimizer_config(self) -> Dict[str, Any]:
        return dict(self.default_optimizer_config)

    def render_table(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - visual side effect
        pass

    def generate_report(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - artifact
        pass

    def enable_kernel(self, name: str, enabled: bool) -> None:
        self.kernel_state[name] = enabled
        self.kernel_calls.append((name, enabled))

    def select_best_run(self, candidates: List[str], prefer_low_val_loss: bool = True) -> str:
        for run_id in candidates:
            if run_id:
                return run_id
        raise RuntimeError("No candidate runs supplied")

    def export_checkpoint(self, run_id: str, dst: str) -> str:
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        Path(dst).write_text(f"checkpoint:{run_id}", encoding="utf-8")
        self.exported_checkpoint = (run_id, dst)
        return dst

    def evaluate_inference(self, label: str, **_: Any) -> str:
        return label

    def quantize_and_evaluate(self, label: str, **_: Any) -> str:
        return label

    def compare_inference(self, **_: Any) -> Dict[str, Any]:
        return self.compare_inference_payloads.pop(0)

    def set_default_inference_precision(self, bits: int) -> None:
        self.default_inference_precision = bits
        self.default_inference_calls.append(bits)

    def get_default_inference_precision(self) -> int:
        return self.default_inference_precision

    def set_flag(self, name: str, value: bool) -> None:
        self.flags[name] = bool(value)

    def summarize_to_markdown(self, out: str, **_: Any) -> None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text("summary", encoding="utf-8")


def test_run_day1_day2_with_kernel_and_quant_fallbacks(monkeypatch, tmp_path):
    fake_emp = FakeEMP()
    fake_emp.compare_runs_payloads.extend(
        [
            {
                "val_loss": {"reference": 0.22, "candidate": 0.19},
                "throughput_samples_per_s": {"reference": 1000.0, "candidate": 1180.0},
                "gpu_mem_gb": {"reference": 8.0, "candidate": 7.2},
                "wall_clock_s": {"reference": 620.0, "candidate": 540.0},
            },
            {
                "val_loss": {"reference": 0.19, "candidate": 0.195},
                "throughput_samples_per_s": {"reference": 1180.0, "candidate": 1500.0},
                "gpu_mem_gb": {"reference": 7.2, "candidate": 7.0},
                "wall_clock_s": {"reference": 540.0, "candidate": 520.0},
            },
        ]
    )
    fake_emp.compare_inference_payloads.extend(
        [
            {
                "val_loss": {"reference": 0.11, "candidate": 0.1105},
                "latency_ms_per_batch": {"reference": 10.0, "candidate": 6.0},
            },
            {
                "val_loss": {"reference": 0.11, "candidate": 0.112},
                "latency_ms_per_batch": {"reference": 10.0, "candidate": 6.5},
            },
        ]
    )

    monkeypatch.setattr(orchestration, "ARTIFACTS_REPORT_DIR", tmp_path / "reports" / "mc_d1d2")
    monkeypatch.setattr(orchestration, "ARTIFACTS_CKPT_DIR", tmp_path / "ckpts" / "mc_d1d2")

    def fake_evaluate_lion_success(comparison: Dict[str, Any], **_: Any) -> Dict[str, Any]:
        assert comparison["val_loss"]["candidate"] < comparison["val_loss"]["reference"]
        return {"ok": True, "reason": "lion wins"}

    def fake_evaluate_flash_success(comparison: Dict[str, Any], **_: Any) -> Dict[str, Any]:
        assert comparison["throughput_samples_per_s"]["candidate"] > comparison["throughput_samples_per_s"]["reference"]
        return {"ok": False, "reason": "insufficient gain"}

    def fake_evaluate_quant_success(
        comparison: Dict[str, Any], *, min_latency_gain: float, **_: Any
    ) -> Dict[str, Any]:
        reference_latency = comparison["latency_ms_per_batch"]["reference"]
        candidate_latency = comparison["latency_ms_per_batch"]["candidate"]
        gain = reference_latency / candidate_latency
        return {"ok": gain >= min_latency_gain, "gain": gain}

    monkeypatch.setattr(orchestration, "evaluate_lion_success", fake_evaluate_lion_success)
    monkeypatch.setattr(orchestration, "evaluate_flash_success", fake_evaluate_flash_success)
    monkeypatch.setattr(orchestration, "evaluate_quant_success", fake_evaluate_quant_success)

    result = orchestration.run_day1_day2(emp_api=fake_emp)

    assert result["lion"]["ok"] is True
    assert fake_emp.default_optimizer_name == "Lion"
    assert fake_emp.kernel_state["flashattention2"] is False
    assert fake_emp.default_inference_precision == 8
    assert fake_emp.tagged_runs["day2_flashattn_lion"] == "REJECTED"
    assert fake_emp.tagged_runs["day2_infer_int8"] == "APPROVED_DEFAULT"
    assert fake_emp.tagged_runs["day2_infer_int4"] == "REJECTED"

    day1_summary = json.loads(
        (orchestration.ARTIFACTS_REPORT_DIR / "day1_summary.json").read_text(encoding="utf-8")
    )
    assert day1_summary["decision"]["ok"] is True

    assert fake_emp.logged_messages[-1][1].startswith("Mini-Cycle Days 1–2 complete")


def _sample_window(ts: datetime, symbol: str, future_outcome: float) -> Dict[str, Any]:
    return {
        "ts": ts,
        "symbol": symbol,
        "ret_1": 0.01,
        "ret_5": 0.02,
        "ret_20": 0.03,
        "atr_14": 1.1,
        "vol_14": 0.5,
        "mom_14": 0.04,
        "rsi_14": 55.0,
        "bb_pos": 0.1,
        "spread": 0.02,
        "depth_imbalance": 0.15,
        "roll_impact": 0.01,
        "corr_benchmark_20": 0.8,
        "skew_20": 0.2,
        "kurt_20": 3.5,
        "future_outcome": future_outcome,
    }


def test_run_day3_day4_retrieval_memory_success(monkeypatch, tmp_path):
    fake_emp = FakeEMP()
    fake_emp.window_stream = [
        _sample_window(datetime(2024, 1, 1), "ES", 0.12),
        _sample_window(datetime(2024, 1, 2), "ES", 0.18),
    ]
    fake_emp.regime.repeat_subset_results.append({"ids": [0, 1]})
    fake_emp.evaluate_subset_results.extend(
        [
            {"sharpe": 0.4, "max_dd": -0.10, "sortino": 0.6},
            {"sharpe": 0.58, "max_dd": -0.16, "sortino": 0.85},
        ]
    )

    report_dir = tmp_path / "reports" / "mc_d3d4"
    ckpt_dir = tmp_path / "ckpts" / "mc_d3d4"
    monkeypatch.setattr(orchestration, "ARTIFACTS_REPORT_DIR_D3D4", report_dir)
    monkeypatch.setattr(orchestration, "ARTIFACTS_CKPT_DIR_D3D4", ckpt_dir)

    result = orchestration.run_day3_day4(emp_api=fake_emp)

    assert result["decision"]["ok"] is True
    assert fake_emp.flags["RETRIEVAL_MEMORY_DEFAULT"] is True
    assert fake_emp.tagged_runs["day4_memON_backtest"] == "APPROVED_DEFAULT"
    assert fake_emp.memory_store is not None
    assert len(fake_emp.memory_store.entries) == len(fake_emp.window_stream)
    assert "emp.memory.retrieval" in fake_emp.modules
    assert fake_emp.modules["emp.memory.retrieval"].ensure_calls

    context = dict(fake_emp.window_stream[0])
    features = fake_emp.inference.run_feature_hooks(context)
    assert "mem_mean_outcome" in features
    notes = fake_emp.planner.run_context_hooks(context)
    assert notes.startswith("Similar past states")
    assert result["delta"]["sharpe_delta"] > 0.05
    assert result["delta"]["maxdd_delta"] < 0.0

    summary_path = report_dir / "summary_day3_day4.md"
    assert summary_path.exists()


def test_run_day3_day4_retrieval_memory_failure_rolls_back(monkeypatch, tmp_path):
    fake_emp = FakeEMP()
    fake_emp.window_stream = [
        _sample_window(datetime(2024, 1, 1), "NQ", 0.05),
        _sample_window(datetime(2024, 1, 2), "NQ", -0.02),
    ]
    fake_emp.regime.repeat_subset_results.append({"ids": [0, 1]})
    fake_emp.evaluate_subset_results.extend(
        [
            {"sharpe": 0.6, "max_dd": -0.12, "sortino": 0.9},
            {"sharpe": 0.45, "max_dd": -0.08, "sortino": 0.7},
        ]
    )

    report_dir = tmp_path / "reports" / "mc_d3d4_fail"
    ckpt_dir = tmp_path / "ckpts" / "mc_d3d4_fail"
    monkeypatch.setattr(orchestration, "ARTIFACTS_REPORT_DIR_D3D4", report_dir)
    monkeypatch.setattr(orchestration, "ARTIFACTS_CKPT_DIR_D3D4", ckpt_dir)

    result = orchestration.run_day3_day4(emp_api=fake_emp)

    assert result["decision"]["ok"] is False
    assert fake_emp.flags["RETRIEVAL_MEMORY_DEFAULT"] is False
    assert fake_emp.tagged_runs["day4_memON_backtest"] == "REJECTED"

    context = dict(fake_emp.window_stream[0])
    assert fake_emp.inference.run_feature_hooks(context) == {}
    assert fake_emp.planner.run_context_hooks(context) == ""
    assert any(level == "warn" and "REJECTED" in message for level, message in fake_emp.logged_messages)


def test_evaluate_lion_success_handles_control_treatment_payload():
    comparison = {
        "val_loss": {"control": 0.2, "treatment": 0.19},
        "gpu_mem_gb": {"control": 10.0, "treatment": 9.0},
    }
    criteria = {"val_loss_ratio_vs_adam": 1.0, "gpu_mem_ratio_vs_adam": 1.0}
    decision = mini_cycle.evaluate_lion_success(comparison, criteria, abort_rules={})
    assert decision["ok"] is True
    assert decision["passed"]["val_loss_ratio_vs_adam"]["ratio"] == pytest.approx(0.19 / 0.2)
    assert decision["passed"]["gpu_mem_ratio_vs_adam"]["ratio"] == pytest.approx(0.9)
