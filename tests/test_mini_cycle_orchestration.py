from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import types
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pytest
from click.testing import CliRunner

import emp.__main__ as emp_cli
import emp.experiments as experiments_pkg
import emp.experiments.mini_cycle as mini_cycle
import emp.experiments.mini_cycle_orchestration as orchestration


def test_cli_mini_cycle_days_option(monkeypatch):
    calls: List[str] = []

    monkeypatch.setattr(experiments_pkg, "run_day1_day2", lambda: calls.append("d1d2"))
    monkeypatch.setattr(experiments_pkg, "run_day3_day4", lambda: calls.append("d3d4"))

    runner = CliRunner()

    result = runner.invoke(emp_cli.cli, ["mini-cycle"])
    assert result.exit_code == 0
    assert calls == ["d1d2"]

    calls.clear()

    result = runner.invoke(emp_cli.cli, ["mini-cycle", "--days", "d3d4"])
    assert result.exit_code == 0
    assert calls == ["d3d4"]


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
        self.prune_calls: List[Dict[str, Any]] = []
        self.prune_ttl_calls: List[Dict[str, Any]] = []

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

    def prune(self, *args: Any, **kwargs: Any) -> None:
        payload: Dict[str, Any] = {}
        if args:
            payload["args"] = list(args)
        payload.update(kwargs)
        if "max_entries" not in payload and args:
            payload["max_entries"] = args[0]
        self.prune_calls.append(payload)
        max_entries = payload.get("max_entries")
        if isinstance(max_entries, int) and max_entries >= 0:
            while len(self.entries) > max_entries:
                self.entries.pop(0)

    def prune_older_than(self, *args: Any, **kwargs: Any) -> None:
        payload: Dict[str, Any] = {}
        if args:
            payload["args"] = list(args)
        payload.update(kwargs)
        ttl_days = payload.get("days")
        if ttl_days is None and payload.get("args"):
            ttl_days = payload["args"][0]
        self.prune_ttl_calls.append(payload)
        if ttl_days is None:
            return
        remaining: List[Dict[str, Any]] = []
        cutoff = FakeTimeNamespace().now
        for entry in self.entries:
            ts = entry.get("meta", {}).get("ts")
            if isinstance(ts, datetime):
                age = (cutoff - ts).days
                if age <= int(ttl_days):
                    remaining.append(entry)
            else:
                remaining.append(entry)
        self.entries = remaining


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
        self.saved_models: List[str] = []
        self.load_calls: List[str] = []
        self.persisted_model: Optional[FakeRegimeModel] = None

    def kmeans(self, **kwargs: Any) -> FakeRegimeModel:
        self.kmeans_calls.append(dict(kwargs))
        return FakeRegimeModel(self.parent)

    def save_model(self, model: FakeRegimeModel, path: str | Path, **_: Any) -> None:
        path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("saved", encoding="utf-8")
        self.saved_models.append(path)
        self.persisted_model = model

    def load_model(self, path: str | Path, **_: Any) -> FakeRegimeModel:
        path = str(path)
        self.load_calls.append(path)
        if self.persisted_model is None:
            raise FileNotFoundError(path)
        return self.persisted_model

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
        self.default_optimizer_name = "AdamW"
        self.default_optimizer_config: Dict[str, Any] = {
            "type": "AdamW",
            "lr": 2e-4,
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
        self.metric_store: Dict[str, Dict[str, float]] = {}

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

    def get_metric(self, run_id: str, key: str) -> float:
        return float(self.metric_store.get(run_id, {}).get(key, 0.0))


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
    assert fake_emp.tagged_runs["day1_lion_candidate"] == "APPROVED_DEFAULT"
    assert fake_emp.tagged_runs["day1_adamw_baseline"] == "APPROVED_FALLBACK"
    assert fake_emp.tagged_runs["day2_flashattn_lion"] == "REJECTED"
    assert fake_emp.tagged_runs["day2_infer_int8"] == "APPROVED_DEFAULT"
    assert fake_emp.tagged_runs["day2_infer_int4"] == "REJECTED"

    day1_summary = json.loads(
        (orchestration.ARTIFACTS_REPORT_DIR / "day1_summary.json").read_text(encoding="utf-8")
    )
    assert day1_summary["decision"]["ok"] is True

    assert fake_emp.logged_messages[-1][1].startswith("Mini-Cycle Days 1–2 complete")


def test_run_day1_day2_quant_failure_reverts_default(monkeypatch, tmp_path):
    fake_emp = FakeEMP()
    fake_emp.compare_runs_payloads.extend(
        [
            {
                "val_loss": {"reference": 0.22, "candidate": 0.19},
                "throughput_samples_per_s": {"reference": 1000.0, "candidate": 1100.0},
                "gpu_mem_gb": {"reference": 8.0, "candidate": 7.5},
                "wall_clock_s": {"reference": 620.0, "candidate": 580.0},
            },
            {
                "val_loss": {"reference": 0.19, "candidate": 0.188},
                "throughput_samples_per_s": {"reference": 1180.0, "candidate": 1300.0},
                "gpu_mem_gb": {"reference": 7.2, "candidate": 7.0},
                "wall_clock_s": {"reference": 540.0, "candidate": 500.0},
            },
        ]
    )
    fake_emp.compare_inference_payloads.extend(
        [
            {
                "val_loss": {"reference": 0.11, "candidate": 0.120},
                "latency_ms_per_batch": {"reference": 10.0, "candidate": 9.5},
            },
            {
                "val_loss": {"reference": 0.11, "candidate": 0.125},
                "latency_ms_per_batch": {"reference": 10.0, "candidate": 9.0},
            },
        ]
    )

    monkeypatch.setattr(orchestration, "ARTIFACTS_REPORT_DIR", tmp_path / "reports" / "mc_d1d2_fail")
    monkeypatch.setattr(orchestration, "ARTIFACTS_CKPT_DIR", tmp_path / "ckpts" / "mc_d1d2_fail")

    def fake_evaluate_lion_success(*_: Any, **__: Any) -> Dict[str, Any]:
        return {"ok": True, "reason": "lion wins"}

    def fake_evaluate_flash_success(*_: Any, **__: Any) -> Dict[str, Any]:
        return {"ok": True, "reason": "flash wins"}

    def fake_evaluate_quant_failure(*_: Any, **__: Any) -> Dict[str, Any]:
        return {"ok": False, "reason": "degrades"}

    monkeypatch.setattr(orchestration, "evaluate_lion_success", fake_evaluate_lion_success)
    monkeypatch.setattr(orchestration, "evaluate_flash_success", fake_evaluate_flash_success)
    monkeypatch.setattr(orchestration, "evaluate_quant_success", fake_evaluate_quant_failure)

    result = orchestration.run_day1_day2(emp_api=fake_emp)

    assert result["int8"]["ok"] is False
    assert result["int4"]["ok"] is False
    assert fake_emp.tagged_runs["day2_infer_fp16_ref"] == "APPROVED_DEFAULT"
    assert fake_emp.tagged_runs["day2_infer_int8"] == "REJECTED"
    assert fake_emp.tagged_runs["day2_infer_int4"] == "REJECTED"
    assert fake_emp.tagged_runs["day2_flashattn_lion"] == "APPROVED_DEFAULT"


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
    fake_emp.metric_store["day3_memOFF_backtest"] = {"latency_ms": 10.0}
    fake_emp.metric_store["day4_memON_backtest"] = {"latency_ms": 10.5}

    report_dir = tmp_path / "reports" / "mc_d3d4"
    ckpt_dir = tmp_path / "ckpts" / "mc_d3d4"
    monkeypatch.setattr(orchestration, "ARTIFACTS_REPORT_DIR_D3D4", report_dir)
    monkeypatch.setattr(orchestration, "ARTIFACTS_CKPT_DIR_D3D4", ckpt_dir)

    result = orchestration.run_day3_day4(emp_api=fake_emp)

    assert result["decision"]["ok"] is True
    assert fake_emp.flags["RETRIEVAL_MEMORY_DEFAULT"] is True
    assert fake_emp.tagged_runs["day4_memON_backtest"] == "APPROVED_DEFAULT"
    assert fake_emp.tagged_runs["day3_memOFF_backtest"] == "APPROVED_FALLBACK"
    assert fake_emp.memory_store is not None
    assert len(fake_emp.memory_store.entries) == len(fake_emp.window_stream)
    assert "emp.memory.retrieval" in fake_emp.modules
    assert fake_emp.modules["emp.memory.retrieval"].ensure_calls
    assert fake_emp.memory_store.prune_calls
    assert fake_emp.memory_store.prune_ttl_calls
    assert fake_emp.regime.saved_models

    context = dict(fake_emp.window_stream[0])
    features = fake_emp.inference.run_feature_hooks(context)
    assert "mem_mean_outcome" in features
    notes = fake_emp.planner.run_context_hooks(context)
    assert notes.startswith("Similar past states")
    assert result["delta"]["sharpe_delta"] > 0.05
    assert result["delta"]["maxdd_delta"] < 0.0

    summary_path = report_dir / "summary_day3_day4.md"
    assert summary_path.exists()
    assert result["delta"]["latency_increase_pct"] == pytest.approx(5.0)
    assert result["hook_latency_ms"]["count"] >= 0


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
    fake_emp.metric_store["day3_memOFF_backtest"] = {"latency_ms": 8.0}
    fake_emp.metric_store["day4_memON_backtest"] = {"latency_ms": 8.0}

    report_dir = tmp_path / "reports" / "mc_d3d4_fail"
    ckpt_dir = tmp_path / "ckpts" / "mc_d3d4_fail"
    monkeypatch.setattr(orchestration, "ARTIFACTS_REPORT_DIR_D3D4", report_dir)
    monkeypatch.setattr(orchestration, "ARTIFACTS_CKPT_DIR_D3D4", ckpt_dir)

    result = orchestration.run_day3_day4(emp_api=fake_emp)

    assert result["decision"]["ok"] is False
    assert fake_emp.flags["RETRIEVAL_MEMORY_DEFAULT"] is False
    assert fake_emp.tagged_runs["day4_memON_backtest"] == "REJECTED"
    assert fake_emp.tagged_runs["day3_memOFF_backtest"] == "APPROVED_DEFAULT"

    context = dict(fake_emp.window_stream[0])
    assert fake_emp.inference.run_feature_hooks(context) == {}
    assert fake_emp.planner.run_context_hooks(context) == ""
    assert any(level == "warn" and "REJECTED" in message for level, message in fake_emp.logged_messages)


def test_memory_hook_returns_empty_when_no_neighbors(monkeypatch, tmp_path):
    fake_emp = FakeEMP()
    fake_emp.window_stream = [
        _sample_window(datetime(2024, 1, 1), "ES", 0.1),
        _sample_window(datetime(2024, 1, 2), "ES", 0.2),
    ]
    fake_emp.regime.repeat_subset_results.append({"ids": [0, 1]})
    fake_emp.evaluate_subset_results.extend(
        [
            {"sharpe": 0.4, "max_dd": -0.1, "sortino": 0.6},
            {"sharpe": 0.5, "max_dd": -0.15, "sortino": 0.8},
        ]
    )
    fake_emp.metric_store["day3_memOFF_backtest"] = {"latency_ms": 10.0}
    fake_emp.metric_store["day4_memON_backtest"] = {"latency_ms": 10.2}

    report_dir = tmp_path / "reports" / "mc_d3d4_empty"
    ckpt_dir = tmp_path / "ckpts" / "mc_d3d4_empty"
    monkeypatch.setattr(orchestration, "ARTIFACTS_REPORT_DIR_D3D4", report_dir)
    monkeypatch.setattr(orchestration, "ARTIFACTS_CKPT_DIR_D3D4", ckpt_dir)

    orchestration.run_day3_day4(emp_api=fake_emp)

    context = dict(fake_emp.window_stream[0])
    context["symbol"] = "CL"
    assert fake_emp.inference.run_feature_hooks(context) == {}


def test_symbol_agnostic_toggle_allows_cross_symbol_neighbors(monkeypatch, tmp_path):
    fake_emp = FakeEMP()

    original_env = fake_emp.env

    def env_override(self: FakeEMP, mapping: Dict[str, Any]) -> None:
        mapping["SYMBOL_AWARE_RETRIEVAL"] = False
        original_env(mapping)

    fake_emp.env = types.MethodType(env_override, fake_emp)

    fake_emp.window_stream = [
        _sample_window(datetime(2024, 1, 1), "ES", 0.1),
        _sample_window(datetime(2024, 1, 2), "NQ", 0.15),
    ]
    fake_emp.regime.repeat_subset_results.append({"ids": [0, 1]})
    fake_emp.evaluate_subset_results.extend(
        [
            {"sharpe": 0.3, "max_dd": -0.09, "sortino": 0.55},
            {"sharpe": 0.37, "max_dd": -0.12, "sortino": 0.7},
        ]
    )
    fake_emp.metric_store["day3_memOFF_backtest"] = {"latency_ms": 9.0}
    fake_emp.metric_store["day4_memON_backtest"] = {"latency_ms": 9.2}

    report_dir = tmp_path / "reports" / "mc_d3d4_sym"
    ckpt_dir = tmp_path / "ckpts" / "mc_d3d4_sym"
    monkeypatch.setattr(orchestration, "ARTIFACTS_REPORT_DIR_D3D4", report_dir)
    monkeypatch.setattr(orchestration, "ARTIFACTS_CKPT_DIR_D3D4", ckpt_dir)

    orchestration.run_day3_day4(emp_api=fake_emp)

    context = dict(fake_emp.window_stream[0])
    context["symbol"] = "CL"
    features = fake_emp.inference.run_feature_hooks(context)
    assert "mem_density" in features and features["mem_density"] > 0


def test_empty_subset_evaluation_is_handled(monkeypatch, tmp_path):
    fake_emp = FakeEMP()
    fake_emp.window_stream = [
        _sample_window(datetime(2024, 1, 1), "ES", 0.05),
    ]
    fake_emp.metric_store["day3_memOFF_backtest"] = {"latency_ms": 7.5}
    fake_emp.metric_store["day4_memON_backtest"] = {"latency_ms": 7.7}

    report_dir = tmp_path / "reports" / "mc_d3d4_subset"
    ckpt_dir = tmp_path / "ckpts" / "mc_d3d4_subset"
    monkeypatch.setattr(orchestration, "ARTIFACTS_REPORT_DIR_D3D4", report_dir)
    monkeypatch.setattr(orchestration, "ARTIFACTS_CKPT_DIR_D3D4", ckpt_dir)

    result = orchestration.run_day3_day4(emp_api=fake_emp)

    assert "sharpe_delta" in result["delta"]
    assert result["subset"] == []


def test_notes_hook_respects_flag(monkeypatch, tmp_path):
    fake_emp = FakeEMP()

    original_env = fake_emp.env

    def env_override(self: FakeEMP, mapping: Dict[str, Any]) -> None:
        mapping["AUG_NOTES"] = False
        original_env(mapping)

    fake_emp.env = types.MethodType(env_override, fake_emp)

    fake_emp.window_stream = [
        _sample_window(datetime(2024, 1, 1), "ES", 0.15),
        _sample_window(datetime(2024, 1, 2), "ES", 0.2),
    ]
    fake_emp.regime.repeat_subset_results.append({"ids": [0, 1]})
    fake_emp.evaluate_subset_results.extend(
        [
            {"sharpe": 0.35, "max_dd": -0.1, "sortino": 0.6},
            {"sharpe": 0.45, "max_dd": -0.14, "sortino": 0.75},
        ]
    )
    fake_emp.metric_store["day3_memOFF_backtest"] = {"latency_ms": 11.0}
    fake_emp.metric_store["day4_memON_backtest"] = {"latency_ms": 11.2}

    report_dir = tmp_path / "reports" / "mc_d3d4_notes"
    ckpt_dir = tmp_path / "ckpts" / "mc_d3d4_notes"
    monkeypatch.setattr(orchestration, "ARTIFACTS_REPORT_DIR_D3D4", report_dir)
    monkeypatch.setattr(orchestration, "ARTIFACTS_CKPT_DIR_D3D4", ckpt_dir)

    orchestration.run_day3_day4(emp_api=fake_emp)

    context = dict(fake_emp.window_stream[0])
    assert fake_emp.planner.run_context_hooks(context) == ""


def test_latency_guard_blocks_approval(monkeypatch, tmp_path):
    fake_emp = FakeEMP()
    fake_emp.window_stream = [
        _sample_window(datetime(2024, 1, 1), "ES", 0.1),
        _sample_window(datetime(2024, 1, 2), "ES", 0.12),
    ]
    fake_emp.regime.repeat_subset_results.append({"ids": [0, 1]})
    fake_emp.evaluate_subset_results.extend(
        [
            {"sharpe": 0.35, "max_dd": -0.1, "sortino": 0.6},
            {"sharpe": 0.5, "max_dd": -0.15, "sortino": 0.82},
        ]
    )
    fake_emp.metric_store["day3_memOFF_backtest"] = {"latency_ms": 5.0}
    fake_emp.metric_store["day4_memON_backtest"] = {"latency_ms": 6.5}

    report_dir = tmp_path / "reports" / "mc_d3d4_latency"
    ckpt_dir = tmp_path / "ckpts" / "mc_d3d4_latency"
    monkeypatch.setattr(orchestration, "ARTIFACTS_REPORT_DIR_D3D4", report_dir)
    monkeypatch.setattr(orchestration, "ARTIFACTS_CKPT_DIR_D3D4", ckpt_dir)

    result = orchestration.run_day3_day4(emp_api=fake_emp)

    assert result["decision"]["ok"] is False
    assert "Latency overhead exceeded guard" in result["decision"]["reason"]
    assert result["latency_guard"]["increase_pct"] > result["latency_guard"]["threshold_pct"]


def test_regime_model_reload_skips_refit(monkeypatch, tmp_path):
    fake_emp = FakeEMP()
    fake_emp.window_stream = [
        _sample_window(datetime(2024, 1, 1), "ES", 0.1),
        _sample_window(datetime(2024, 1, 2), "ES", 0.12),
    ]
    fake_emp.regime.persisted_model = FakeRegimeModel(fake_emp)
    fake_emp.regime.repeat_subset_results.append({"ids": [0, 1]})
    fake_emp.evaluate_subset_results.extend(
        [
            {"sharpe": 0.3, "max_dd": -0.1, "sortino": 0.55},
            {"sharpe": 0.42, "max_dd": -0.14, "sortino": 0.7},
        ]
    )
    fake_emp.metric_store["day3_memOFF_backtest"] = {"latency_ms": 9.0}
    fake_emp.metric_store["day4_memON_backtest"] = {"latency_ms": 9.1}

    regime_path = tmp_path / "saved_regime.bin"
    regime_path.parent.mkdir(parents=True, exist_ok=True)
    regime_path.write_text("saved", encoding="utf-8")

    original_env = fake_emp.env

    def env_override(self: FakeEMP, mapping: Dict[str, Any]) -> None:
        mapping["REGIME_MODEL_PATH"] = str(regime_path)
        original_env(mapping)

    fake_emp.env = types.MethodType(env_override, fake_emp)

    report_dir = tmp_path / "reports" / "mc_d3d4_reload"
    ckpt_dir = tmp_path / "ckpts" / "mc_d3d4_reload"
    monkeypatch.setattr(orchestration, "ARTIFACTS_REPORT_DIR_D3D4", report_dir)
    monkeypatch.setattr(orchestration, "ARTIFACTS_CKPT_DIR_D3D4", ckpt_dir)

    orchestration.run_day3_day4(emp_api=fake_emp)

    assert fake_emp.regime.load_calls == [str(regime_path)]
    assert fake_emp.regime.kmeans_calls == []


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
