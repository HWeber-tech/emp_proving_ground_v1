from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import emp.experiments.mini_cycle_orchestration as orchestration


class FakeEMP:
    def __init__(self) -> None:
        self.seed: int | None = None
        self.project: str | None = None
        self.run_group: str | None = None
        self.env_vars: Dict[str, str] = {}
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
        self.exported_checkpoint: tuple[str, str] | None = None

    def set_seed(self, value: int) -> None:
        self.seed = value

    def set_project(self, project: str) -> None:
        self.project = project

    def set_run_group(self, group: str) -> None:
        self.run_group = group

    def env(self, mapping: Dict[str, str]) -> None:
        self.env_vars.update(mapping)

    def run_experiment(self, label: str, **_: Any) -> str:
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
