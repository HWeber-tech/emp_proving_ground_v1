"""End-to-end orchestration for the EMP mini-cycle days 1 and 2."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from emp.experiments.mini_cycle import (
    evaluate_flash_success,
    evaluate_lion_success,
    evaluate_quant_success,
)

# The real system injects an ``EMP`` facade that exposes the methods used below.
# Tests provide a lightweight stub and production wires the concrete runtime.
try:  # pragma: no cover - the real API is not available in the kata environment
    from emp.runtime import EMP  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    EMP = None  # type: ignore


ARTIFACTS_REPORT_DIR = Path("artifacts/reports/mc_d1d2")
ARTIFACTS_CKPT_DIR = Path("artifacts/ckpts/mc_d1d2")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _export_standard_artifacts(emp: Any, run_id: Any, label: str, artifacts_dir: str) -> None:
    base = Path(artifacts_dir)
    _ensure_dir(base)
    emp.plot_curves(run_id, x="step", ys=["train_loss", "val_loss"], title=f"{label}: Loss")
    emp.plot_curves(run_id, x="step", ys=["grad_norm"], title=f"{label}: Grad Norm")
    emp.plot_bars(
        run_id,
        names=["throughput_samples_per_s", "gpu_mem_gb"],
        title=f"{label}: Perf",
    )
    emp.export_csv(run_id, out=str(base / f"{label}_metrics.csv"))
    emp.save_summary(run_id, out=str(base / f"{label}_summary.json"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _resolve_emp(emp_api: Any | None) -> Any:
    if emp_api is not None:
        return emp_api
    if EMP is None:  # pragma: no cover - triggered only when runtime missing
        raise RuntimeError("EMP runtime is not available; provide `emp_api`.")
    return EMP


def run_day1_day2(emp_api: Any | None = None) -> Dict[str, Any]:
    """Orchestrates EMP mini-cycle Days 1–2.

    The routine executes the two day pilot described in the internal run-book:
    Day 1 compares Adam and Lion, while Day 2 validates FlashAttention and
    low-bit inference.  It produces the standard experiment artefacts and keeps
    rollbacks safe.
    """

    emp = _resolve_emp(emp_api)

    _ensure_dir(ARTIFACTS_REPORT_DIR)
    _ensure_dir(ARTIFACTS_CKPT_DIR)

    metrics: Iterable[str] = (
        "train_loss",
        "val_loss",
        "sharpe",
        "calmar",
        "max_dd",
        "grad_norm",
        "throughput_samples_per_s",
        "gpu_mem_gb",
        "wall_clock_s",
    )

    task = {
        "model_id": "emp_baseline_lstm_v3",
        "dataset_id": "market_window_4h_lookahead",
        "train_split": "train",
        "val_split": "val",
        "test_split": "test",
        "batch_size": 128,
        "num_workers": 4,
    }

    schedule = {
        "epochs": 12,
        "early_stop_patience": 3,
        "grad_clip": 1.0,
        "log_every": 50,
        "ckpt_dir": str(ARTIFACTS_CKPT_DIR),
        "artifacts_dir": str(ARTIFACTS_REPORT_DIR),
    }

    criteria = {
        "lion_success": {
            "val_loss_ratio_vs_adam": 0.95,
            "steps_ratio_vs_adam": 0.80,
            "gpu_mem_ratio_vs_adam": 0.90,
        },
        "lion_abort": {
            "val_loss_worse_after_2x_steps": True,
            "time_overhead_vs_adam_pct": 20,
        },
        "flash_success": {
            "throughput_gain_min": 1.8,
            "val_loss_degradation_max_pct": 1.0,
        },
        "int8_success": {
            "val_loss_degradation_max_pct": 0.5,
            "latency_gain_min": 1.5,
        },
        "int4_success": {
            "val_loss_degradation_max_pct": 1.5,
            "latency_gain_min": 2.0,
        },
    }

    emp.set_seed(1337)
    emp.set_project("emp_mini_cycles")
    emp.set_run_group("mc_day1_day2_lion_flashattn_q")
    emp.env(
        {
            "REPO": "/opt/emp/emp_proving_ground_v1",
            "CUDA_LAUNCH_BLOCKING": "0",
            "PYTORCH_ENABLE_SDPA": "1",
        }
    )

    adam_cfg = {"type": "Adam", "lr": 3e-4, "betas": (0.9, 0.999), "weight_decay": 0.0}
    lion_cfg = {"type": "Lion", "lr": 3e-4, "betas": (0.9, 0.99), "weight_decay": 0.0}

    run_adam = emp.run_experiment(
        label="day1_adam_baseline",
        model=task["model_id"],
        dataset=task["dataset_id"],
        optimizer=adam_cfg,
        schedule=schedule,
        metrics=metrics,
        notes="Day1 baseline with Adam",
    )
    _export_standard_artifacts(emp, run_adam, "day1_adam", schedule["artifacts_dir"])

    run_lion = emp.run_experiment(
        label="day1_lion_candidate",
        model=task["model_id"],
        dataset=task["dataset_id"],
        optimizer=lion_cfg,
        schedule=schedule,
        metrics=metrics,
        notes="Day1 Lion candidate",
    )
    _export_standard_artifacts(emp, run_lion, "day1_lion", schedule["artifacts_dir"])

    comparison_day1 = emp.compare_runs(
        reference=run_adam,
        candidate=run_lion,
        keys=["val_loss", "throughput_samples_per_s", "gpu_mem_gb", "wall_clock_s"],
    )

    lion_decision = evaluate_lion_success(
        comparison_day1,
        criteria=criteria["lion_success"],
        abort_rules=criteria["lion_abort"],
    )
    lion_ok = bool(lion_decision.get("ok"))

    if lion_ok:
        emp.tag_run(run_lion, "APPROVED")
        emp.log("Lion approved; setting Lion as default optimizer.", level="info")
        emp.set_default_optimizer("Lion", lion_cfg)
    else:
        emp.tag_run(run_lion, "REJECTED")
        emp.log("Lion failed criteria; keeping Adam as default optimizer.", level="warn")
        emp.set_default_optimizer("Adam", adam_cfg)

    ref_for_flash = run_lion if lion_ok else run_adam

    emp.render_table(comparison_day1, title="Day 1: Adam vs Lion Comparison")
    emp.generate_report(
        title="EMP Mini-Cycle Day 1 — Lion vs Adam",
        runs=[run_adam, run_lion],
        decision=lion_decision,
        out=str(ARTIFACTS_REPORT_DIR / "day1_report.html"),
    )

    _write_json(
        ARTIFACTS_REPORT_DIR / "day1_summary.json",
        {
            "adam_run": run_adam,
            "lion_run": run_lion,
            "decision": lion_decision,
            "comparison": comparison_day1,
        },
    )

    emp.enable_kernel("flashattention2", enabled=True)

    run_flash = emp.run_experiment(
        label="day2_flashattn_lion",
        model=task["model_id"],
        dataset=task["dataset_id"],
        optimizer=emp.get_default_optimizer_config(),
        schedule=schedule,
        metrics=metrics,
        notes="Day2: FlashAttention2 enabled",
    )
    _export_standard_artifacts(emp, run_flash, "day2_flashattn", schedule["artifacts_dir"])

    comparison_flash = emp.compare_runs(
        reference=ref_for_flash,
        candidate=run_flash,
        keys=["val_loss", "throughput_samples_per_s", "gpu_mem_gb", "wall_clock_s"],
    )

    flash_decision = evaluate_flash_success(
        comparison_flash,
        throughput_gain_min=criteria["flash_success"]["throughput_gain_min"],
        val_loss_degradation_max_pct=criteria["flash_success"]["val_loss_degradation_max_pct"],
    )
    flash_ok = bool(flash_decision.get("ok"))

    if flash_ok:
        emp.tag_run(run_flash, "APPROVED")
        emp.log("FlashAttention approved; kept enabled for next steps.", level="info")
    else:
        emp.tag_run(run_flash, "REJECTED")
        emp.enable_kernel("flashattention2", enabled=False)
        emp.log("FlashAttention failed criteria; rolled back to SDPA.", level="warn")

    emp.render_table(comparison_flash, title="Day 2A: FlashAttention vs Reference")
    emp.generate_report(
        title="EMP Mini-Cycle Day 2A — FlashAttention",
        runs=[ref_for_flash, run_flash],
        decision=flash_decision,
        out=str(ARTIFACTS_REPORT_DIR / "day2_flash_report.html"),
    )

    candidate_runs = [run_lion if lion_ok else run_adam]
    if flash_ok:
        candidate_runs.append(run_flash)
    best_train_run = emp.select_best_run(candidates=candidate_runs, prefer_low_val_loss=True)
    ckpt_path = emp.export_checkpoint(
        best_train_run,
        dst=str(ARTIFACTS_CKPT_DIR / "best_for_quant.pt"),
    )

    eval_ref = emp.evaluate_inference(
        label="day2_infer_fp16_ref",
        checkpoint=ckpt_path,
        dtype="fp16",
        metrics=["val_loss", "latency_ms_per_batch", "throughput_samples_per_s", "gpu_mem_gb"],
    )
    _export_standard_artifacts(emp, eval_ref, "day2_infer_fp16_ref", schedule["artifacts_dir"])

    eval_int8 = emp.quantize_and_evaluate(
        label="day2_infer_int8",
        checkpoint=ckpt_path,
        method="awq_or_gptq_auto",
        bits=8,
        metrics=["val_loss", "latency_ms_per_batch", "throughput_samples_per_s", "gpu_mem_gb"],
    )
    _export_standard_artifacts(emp, eval_int8, "day2_infer_int8", schedule["artifacts_dir"])

    eval_int4 = emp.quantize_and_evaluate(
        label="day2_infer_int4",
        checkpoint=ckpt_path,
        method="awq_or_gptq_auto",
        bits=4,
        metrics=["val_loss", "latency_ms_per_batch", "throughput_samples_per_s", "gpu_mem_gb"],
    )
    _export_standard_artifacts(emp, eval_int4, "day2_infer_int4", schedule["artifacts_dir"])

    comparison_int8 = emp.compare_inference(reference=eval_ref, candidate=eval_int8)
    comparison_int4 = emp.compare_inference(reference=eval_ref, candidate=eval_int4)

    int8_decision = evaluate_quant_success(
        comparison_int8,
        max_val_loss_degradation_pct=criteria["int8_success"]["val_loss_degradation_max_pct"],
        min_latency_gain=criteria["int8_success"]["latency_gain_min"],
    )
    int4_decision = evaluate_quant_success(
        comparison_int4,
        max_val_loss_degradation_pct=criteria["int4_success"]["val_loss_degradation_max_pct"],
        min_latency_gain=criteria["int4_success"]["latency_gain_min"],
    )

    int8_ok = bool(int8_decision.get("ok"))
    int4_ok = bool(int4_decision.get("ok"))

    if int4_ok:
        emp.set_default_inference_precision(bits=4)
        emp.tag_run(eval_int4, "APPROVED_DEFAULT")
        emp.tag_run(eval_int8, "APPROVED_FALLBACK")
        emp.log("INT4 approved as default inference precision; INT8 as fallback.", level="info")
    elif int8_ok:
        emp.set_default_inference_precision(bits=8)
        emp.tag_run(eval_int8, "APPROVED_DEFAULT")
        emp.tag_run(eval_int4, "REJECTED")
        emp.log("INT8 approved as default inference precision; INT4 rejected.", level="info")
    else:
        emp.set_default_inference_precision(bits=16)
        emp.tag_run(eval_int8, "REJECTED")
        emp.tag_run(eval_int4, "REJECTED")
        emp.log("Quantization failed thresholds; keeping FP16 as default.", level="warn")

    all_runs = [run for run in (ref_for_flash, run_flash, eval_ref, eval_int8, eval_int4) if run is not None]

    emp.generate_report(
        title="EMP Mini-Cycle Day 2 — Kernels & Quantization",
        runs=all_runs,
        decision={"flash": flash_decision, "int8": int8_decision, "int4": int4_decision},
        out=str(ARTIFACTS_REPORT_DIR / "day2_report.html"),
    )

    emp.summarize_to_markdown(
        out=str(ARTIFACTS_REPORT_DIR / "summary_day1_day2.md"),
        sections=[
            {
                "title": "Day 1 — Optimizer Pilot",
                "bullets": [
                    f"Adam run id: {run_adam}",
                    f"Lion run id: {run_lion}",
                    f"Decision: {'APPROVE' if lion_ok else 'REJECT'} Lion",
                    f"Key deltas (Lion vs Adam): {comparison_day1}",
                ],
            },
            {
                "title": "Day 2 — FlashAttention & Quantization",
                "bullets": [
                    f"FlashAttention decision: {'APPROVE' if flash_ok else 'REJECT'}",
                    f"Flash vs Ref deltas: {comparison_flash}",
                    f"Inference decisions: INT8={'OK' if int8_ok else 'NO'}, INT4={'OK' if int4_ok else 'NO'}",
                    f"Default inference precision set to: {emp.get_default_inference_precision()}-bit",
                ],
            },
        ],
    )

    _write_json(
        ARTIFACTS_REPORT_DIR / "day2_summary.json",
        {
            "flash": flash_decision,
            "int8": int8_decision,
            "int4": int4_decision,
            "comparison_flash": comparison_flash,
            "comparison_int8": comparison_int8,
            "comparison_int4": comparison_int4,
        },
    )

    if not lion_ok:
        emp.set_default_optimizer("Adam", adam_cfg)
    if not flash_ok:
        emp.enable_kernel("flashattention2", enabled=False)
    if not (int8_ok or int4_ok):
        emp.set_default_inference_precision(bits=16)

    emp.log("Mini-Cycle Days 1–2 complete. Artifacts in artifacts/reports/mc_d1d2", level="info")

    return {
        "lion": lion_decision,
        "flash": flash_decision,
        "int8": int8_decision,
        "int4": int4_decision,
    }
