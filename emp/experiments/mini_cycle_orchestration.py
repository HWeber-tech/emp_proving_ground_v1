"""End-to-end orchestration for the EMP mini-cycle pilots."""
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
ARTIFACTS_REPORT_DIR_D3D4 = Path("artifacts/reports/mc_d3d4")
ARTIFACTS_CKPT_DIR_D3D4 = Path("artifacts/ckpts/mc_d3d4")


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
    Day 1 compares AdamW and Lion, while Day 2 validates FlashAttention and
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
        "lr_scheduler": {"type": "cosine"},
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

    adam_cfg = {"type": "AdamW", "lr": 2e-4, "betas": (0.9, 0.999), "weight_decay": 0.0}
    lion_cfg = {"type": "Lion", "lr": 3e-4, "betas": (0.9, 0.99), "weight_decay": 0.0}

    run_adam = emp.run_experiment(
        label="day1_adamw_baseline",
        model=task["model_id"],
        dataset=task["dataset_id"],
        optimizer=adam_cfg,
        schedule=schedule,
        metrics=metrics,
        notes="Day1 baseline with AdamW",
    )
    _export_standard_artifacts(emp, run_adam, "day1_adamw", schedule["artifacts_dir"])

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
        emp.tag_run(run_lion, "APPROVED_DEFAULT")
        emp.tag_run(run_adam, "APPROVED_FALLBACK")
        emp.log("Lion approved; setting Lion as default optimizer.", level="info")
        emp.set_default_optimizer("Lion", lion_cfg)
    else:
        emp.tag_run(run_lion, "REJECTED")
        emp.tag_run(run_adam, "APPROVED_DEFAULT")
        emp.log("Lion failed criteria; keeping AdamW as default optimizer.", level="warn")
        emp.set_default_optimizer("AdamW", adam_cfg)

    ref_for_flash = run_lion if lion_ok else run_adam

    emp.render_table(comparison_day1, title="Day 1: AdamW vs Lion Comparison")
    emp.generate_report(
        title="EMP Mini-Cycle Day 1 — Lion vs AdamW",
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
        emp.tag_run(run_flash, "APPROVED_DEFAULT")
        emp.tag_run(ref_for_flash, "APPROVED_FALLBACK")
        emp.log("FlashAttention approved; kept enabled for next steps.", level="info")
    else:
        emp.tag_run(run_flash, "REJECTED")
        emp.enable_kernel("flashattention2", enabled=False)
        emp.tag_run(ref_for_flash, "APPROVED_DEFAULT")
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
        emp.tag_run(eval_ref, "APPROVED_DEFAULT")
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
                    f"AdamW run id: {run_adam}",
                    f"Lion run id: {run_lion}",
                    f"Decision: {'APPROVE' if lion_ok else 'REJECT'} Lion",
                    f"Key deltas (Lion vs AdamW): {comparison_day1}",
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
        emp.set_default_optimizer("AdamW", adam_cfg)
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


def run_day3_day4(emp_api: Any | None = None) -> Dict[str, Any]:
    """Execute the retrieval memory pilot for mini-cycle Days 3–4."""

    emp = _resolve_emp(emp_api)

    _ensure_dir(ARTIFACTS_REPORT_DIR_D3D4)
    _ensure_dir(ARTIFACTS_CKPT_DIR_D3D4)

    cfg: Dict[str, Any] = {
        "REPO": "/opt/emp/emp_proving_ground_v1",
        "ART_DIR": str(ARTIFACTS_REPORT_DIR_D3D4),
        "CKPT_DIR": str(ARTIFACTS_CKPT_DIR_D3D4),
        "MEMORY_ENABLED": True,
        "TOPK": 8,
        "EMBED_DIM": 384,
        "EMBED_MODEL": "all-MiniLM-L6-v2",
        "SIM_METRIC": "cosine",
        "DECAY_HALF_LIFE_DAYS": 90,
        "WRITE_EVERY_STEPS": 500,
        "AUG_FEATURES": True,
        "AUG_NOTES": True,
        "SYMBOL_AWARE_RETRIEVAL": True,
        "REGIME_K": 6,
        "REGIME_FEATURES": ["vol_14", "atr_14", "ret_5", "ret_20", "skew_20", "kurt_20"],
        "STATE_SNAPSHOT_FEATURES": [
            "ret_1",
            "ret_5",
            "ret_20",
            "atr_14",
            "vol_14",
            "mom_14",
            "rsi_14",
            "bb_pos",
            "spread",
            "depth_imbalance",
            "roll_impact",
            "corr_benchmark_20",
        ],
        "OUTCOME_HORIZON": 20,
        "OUTCOME_METRIC": "excess_ret",
        "BACKTEST_SPLIT": "val",
        "WINDOWS_FOR_EVAL": 1000,
        "MEMORY_MAX_ENTRIES": 50000,
        "MEMORY_TTL_DAYS": 365,
        "REGIME_MODEL_PATH": "artifacts/memory/emp_regime_model.bin",
        "RNG_SEED": 4242,
        "LATENCY_INCREASE_MAX_PCT": 10.0,
    }

    emp.set_seed(4242)
    emp.set_project("emp_mini_cycles")
    emp.set_run_group("mc_day3_day4_retrieval_memory")
    emp.env(cfg)

    module = emp.module("emp.memory.retrieval")
    module.ensure_installed(
        deps=["faiss-cpu", "sqlite3", "sentence-transformers"]
    )

    memory_store = emp.memory.create_store(
        name="emp_retrieval_memory",
        kind="faiss_sqlite",
        embedder=cfg["EMBED_MODEL"],
        embed_dim=cfg["EMBED_DIM"],
        sim_metric=cfg["SIM_METRIC"],
        path_index="artifacts/memory/emp_mem.index",
        path_meta="artifacts/memory/emp_mem.sqlite",
    )

    state_features = list(cfg["STATE_SNAPSHOT_FEATURES"])
    regime_features = list(cfg["REGIME_FEATURES"])

    def _state_embedding(batch_features: Dict[str, Any]) -> Any:
        selected = emp.select_features(batch_features, state_features)
        return emp.embed(cfg["EMBED_MODEL"], selected)

    regime_model_path = Path(cfg["REGIME_MODEL_PATH"])
    regime_model: Any | None = None
    loader = getattr(emp.regime, "load_model", None)
    if callable(loader) and regime_model_path.exists():
        try:
            regime_model = loader(path=str(regime_model_path))
            emp.log(
                f"Loaded persisted regime model from {regime_model_path}.",
                level="info",
            )
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            emp.log(
                f"Failed to load regime model ({exc}); refitting fresh model.",
                level="warn",
            )
            regime_model = None

    if regime_model is None:
        try:
            regime_model = emp.regime.kmeans(
                k=cfg["REGIME_K"],
                feature_names=regime_features,
                split="train",
                dataset="market_window_4h_lookahead",
                random_state=int(cfg["RNG_SEED"]),
            )
        except TypeError:
            regime_model = emp.regime.kmeans(
                k=cfg["REGIME_K"],
                feature_names=regime_features,
                split="train",
                dataset="market_window_4h_lookahead",
            )

        saver = getattr(emp.regime, "save_model", None)
        if callable(saver):
            try:
                saver(regime_model, path=str(regime_model_path))
                emp.log(
                    f"Persisted regime model to {regime_model_path}.",
                    level="info",
                )
            except TypeError:
                saver(regime_model, str(regime_model_path))
                emp.log(
                    f"Persisted regime model to {regime_model_path}.",
                    level="info",
                )
        else:
            _ensure_dir(regime_model_path.parent)
            with regime_model_path.open("w", encoding="utf-8") as handle:
                json.dump({"regime_k": cfg["REGIME_K"], "features": regime_features}, handle)
            emp.log(
                f"Saved regime model metadata to {regime_model_path} (fallback).",
                level="info",
            )

    hook_latency_stats = {"count": 0, "total_ms": 0.0}

    def _aggregate_neighbors(neighbors: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        records = list(neighbors)
        if not records:
            return {}
        weights = [
            emp.time.recency_weight(
                rec["meta"].get("ts"), half_life_days=cfg["DECAY_HALF_LIFE_DAYS"]
            )
            for rec in records
        ]
        outcomes = [rec.get("outcome", 0.0) for rec in records]
        weighted = emp.stats.weighted(values=outcomes, weights=weights)
        hitrate = emp.stats.hitrate([1 if value > 0 else 0 for value in outcomes])
        p90 = emp.stats.percentile(outcomes, 90)
        return {
            "mem_mean_outcome": weighted.get("mean", 0.0),
            "mem_p90_outcome": p90,
            "mem_hitrate": hitrate,
            "mem_density": len(records),
        }

    def _make_note(context: Dict[str, Any], regime_tag: Any) -> str:
        vol = float(context.get("vol_14", 0.0))
        ret20 = float(context.get("ret_20", 0.0))
        return f"regime={regime_tag} vol14={vol:.3f} ret20={ret20:.3f}"

    def _apply_memory_retention() -> None:
        if memory_store is None:
            return
        max_entries = int(cfg.get("MEMORY_MAX_ENTRIES", 0) or 0)
        ttl_days = float(cfg.get("MEMORY_TTL_DAYS", 0.0) or 0.0)
        if max_entries > 0:
            prune = getattr(memory_store, "prune", None)
            if callable(prune):
                try:
                    prune(max_entries=max_entries)
                except TypeError:
                    prune({"max_entries": max_entries})
                emp.log(
                    f"Applied memory prune to enforce max {max_entries} entries.",
                    level="info",
                )
        if ttl_days > 0:
            prune_ttl = getattr(memory_store, "prune_older_than", None)
            if callable(prune_ttl):
                try:
                    prune_ttl(days=ttl_days)
                except TypeError:
                    prune_ttl(ttl_days)
                emp.log(
                    f"Applied memory TTL prune at {ttl_days} days.",
                    level="info",
                )

    def write_memory_from_backtest(run_id: str) -> Dict[str, Any]:
        if not cfg.get("MEMORY_ENABLED") or memory_store is None:
            return {"run_id": run_id, "written": 0}
        stream = emp.data.iter_windows(
            split=cfg["BACKTEST_SPLIT"],
            limit=cfg["WINDOWS_FOR_EVAL"],
            fields=state_features + regime_features + ["symbol", "ts"],
        )
        batch: list[Dict[str, Any]] = []
        total_written = 0
        for index, window in enumerate(stream, start=1):
            regime_tag = regime_model.tag(window)
            embedding = _state_embedding(window)
            outcome = emp.labelers.future_outcome(
                metric=cfg["OUTCOME_METRIC"],
                horizon=cfg["OUTCOME_HORIZON"],
                window=window,
            )
            note = _make_note(window, regime_tag)
            meta = {
                "ts": window.get("ts"),
                "symbol": window.get("symbol"),
                "regime_tag": regime_tag,
            }
            batch.append(
                {
                    "emb": embedding,
                    "outcome": outcome,
                    "note": note,
                    "meta": meta,
                }
            )
            if index % int(cfg["WRITE_EVERY_STEPS"]) == 0:
                memory_store.add(batch)
                total_written += len(batch)
                batch = []
            if index >= int(cfg["WINDOWS_FOR_EVAL"]):
                break
        if batch:
            memory_store.add(batch)
            total_written += len(batch)
        _apply_memory_retention()
        emp.log(
            f"Memory write complete for {run_id}; stored {total_written} entries.",
            level="info",
        )
        return {"run_id": run_id, "written": total_written}

    def memory_augment_hook(context: Dict[str, Any]) -> Dict[str, Any]:
        if not (cfg.get("MEMORY_ENABLED") and cfg.get("AUG_FEATURES")):
            return {}
        if memory_store is None:
            return {}
        import time

        started = time.perf_counter()
        embedding = _state_embedding(context)
        regime_tag = regime_model.tag(context)
        filters = {"regime_tag": regime_tag}
        if cfg.get("SYMBOL_AWARE_RETRIEVAL", True) and "symbol" in context:
            filters["symbol"] = context["symbol"]
        neighbors = memory_store.search(
            embedding=embedding,
            topk=int(cfg["TOPK"]),
            where=filters,
            recency_half_life_days=cfg["DECAY_HALF_LIFE_DAYS"],
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        hook_latency_stats["count"] += 1
        hook_latency_stats["total_ms"] += elapsed_ms
        if not neighbors:
            return {}
        return _aggregate_neighbors(neighbors)

    def planner_notes_hook(context: Dict[str, Any]) -> str:
        if not (cfg.get("MEMORY_ENABLED") and cfg.get("AUG_NOTES")):
            return ""
        if memory_store is None:
            return ""
        embedding = _state_embedding(context)
        regime_tag = regime_model.tag(context)
        filters = {"regime_tag": regime_tag}
        if cfg.get("SYMBOL_AWARE_RETRIEVAL", True) and "symbol" in context:
            filters["symbol"] = context["symbol"]
        neighbors = memory_store.search(embedding, int(cfg["TOPK"]), where=filters)
        if not neighbors:
            return ""
        bullets = [
            "- {ts} {symbol} r{regime}: outcome={out:.4f} note={note}".format(
                ts=neighbor.get("meta", {}).get("ts"),
                symbol=neighbor.get("meta", {}).get("symbol"),
                regime=neighbor.get("meta", {}).get("regime_tag"),
                out=float(neighbor.get("outcome", 0.0)),
                note=neighbor.get("note", ""),
            )
            for neighbor in neighbors[:5]
        ]
        return "Similar past states:\n" + "\n".join(bullets)

    emp.inference.register_feature_hook(memory_augment_hook)
    emp.planner.register_context_hook(planner_notes_hook)

    write_memory_from_backtest("preload_mem_from_val")

    task = {
        "model_id": "emp_baseline_lstm_v3",
        "dataset_id": "market_window_4h_lookahead",
        "split": cfg["BACKTEST_SPLIT"],
        "batch_size": 128,
        "num_workers": 4,
    }

    schedule = {"epochs": 0, "eval_only": True, "log_every": 50}

    cfg["MEMORY_ENABLED"] = False
    run_off = emp.run_experiment(
        label="day3_memOFF_backtest",
        model=task["model_id"],
        dataset=task["dataset_id"],
        schedule=schedule,
        metrics=[
            "sharpe",
            "sortino",
            "max_dd",
            "calmar",
            "hit_rate",
            "avg_trade",
            "latency_ms",
            "wall_clock_s",
        ],
        notes="A/B baseline without retrieval memory",
    )

    cfg["MEMORY_ENABLED"] = True
    run_on = emp.run_experiment(
        label="day4_memON_backtest",
        model=task["model_id"],
        dataset=task["dataset_id"],
        schedule=schedule,
        metrics=[
            "sharpe",
            "sortino",
            "max_dd",
            "calmar",
            "hit_rate",
            "avg_trade",
            "latency_ms",
            "wall_clock_s",
        ],
        notes="A/B with retrieval memory augment",
    )

    for run_id, tag in ((run_off, "memOFF"), (run_on, "memON")):
        emp.export_csv(run_id, out=f"{cfg['ART_DIR']}/{tag}_metrics.csv")
        emp.save_summary(run_id, out=f"{cfg['ART_DIR']}/{tag}_summary.json")

    subset = emp.regime.repeat_subset(
        split=cfg["BACKTEST_SPLIT"],
        regime_model=regime_model,
        match_threshold=0.8,
    )

    res_off = emp.evaluate_on_subset(
        run_off,
        subset=subset,
        metrics=["sharpe", "max_dd", "sortino"],
    )
    res_on = emp.evaluate_on_subset(
        run_on,
        subset=subset,
        metrics=["sharpe", "max_dd", "sortino"],
    )

    sharpe_off = float(res_off.get("sharpe", 0.0) or 0.0)
    sharpe_on = float(res_on.get("sharpe", 0.0) or 0.0)
    maxdd_off = float(res_off.get("max_dd", 0.0) or 0.0)
    maxdd_on = float(res_on.get("max_dd", 0.0) or 0.0)
    sortino_off = float(res_off.get("sortino", 0.0) or 0.0)
    sortino_on = float(res_on.get("sortino", 0.0) or 0.0)

    def _lookup_metric(run_id: str, key: str) -> float:
        getters = [
            getattr(emp, "get_metric", None),
            getattr(emp, "metrics", None),
            getattr(emp, "get_run_metric", None),
        ]
        for getter in getters:
            if callable(getter):
                try:
                    value = getter(run_id, key)  # type: ignore[arg-type]
                except TypeError:
                    try:
                        payload = getter(run_id)  # type: ignore[misc]
                    except Exception:
                        continue
                    else:
                        if isinstance(payload, Mapping) and key in payload:
                            return float(payload[key] or 0.0)
                        continue
                except Exception:
                    continue
                else:
                    if isinstance(value, Mapping):
                        value = value.get(key, 0.0)
                    return float(value or 0.0)
        return 0.0

    latency_off = _lookup_metric(run_off, "latency_ms")
    latency_on = _lookup_metric(run_on, "latency_ms")
    latency_increase_pct = 0.0
    if latency_off > 0:
        latency_increase_pct = ((latency_on - latency_off) / latency_off) * 100.0

    delta = {
        "sharpe_delta": sharpe_on - sharpe_off,
        "maxdd_delta": maxdd_on - maxdd_off,
        "sortino_delta": sortino_on - sortino_off,
        "latency_increase_pct": latency_increase_pct,
    }

    success = delta["sharpe_delta"] > 0.05 and delta["maxdd_delta"] < 0.0
    abort = (delta["sharpe_delta"] < -0.05) or (
        delta["maxdd_delta"] > 0.0 and abs(delta["maxdd_delta"]) > 0.02
    )
    latency_guard = (
        latency_off > 0
        and latency_increase_pct > float(cfg.get("LATENCY_INCREASE_MAX_PCT", 0.0))
    )

    decision_ok = success and not abort and not latency_guard
    if latency_guard:
        reason = (
            "Latency overhead exceeded guard: "
            f"{latency_increase_pct:.2f}% > {cfg['LATENCY_INCREASE_MAX_PCT']}%"
        )
    elif success and not abort:
        reason = "Improved Sharpe and reduced MaxDD on regime repeats"
    else:
        reason = "Did not meet Sharpe/DD thresholds on regime repeats"

    decision = {
        "ok": decision_ok,
        "reason": reason,
    }

    if decision["ok"]:
        emp.tag_run(run_on, "APPROVED_DEFAULT")
        emp.tag_run(run_off, "APPROVED_FALLBACK")
        emp.set_flag("RETRIEVAL_MEMORY_DEFAULT", True)
        emp.log("Retrieval memory APPROVED as default (flag on).", level="info")
    else:
        emp.tag_run(run_on, "REJECTED")
        emp.tag_run(run_off, "APPROVED_DEFAULT")
        emp.set_flag("RETRIEVAL_MEMORY_DEFAULT", False)
        cfg["MEMORY_ENABLED"] = False
        emp.log("Retrieval memory REJECTED (kept off).", level="warn")

    emp.generate_report(
        title="EMP Mini-Cycle Days 3–4 — Retrieval Memory Pilot",
        runs=[run_off, run_on],
        decision=decision,
        extras={
            "subset_metrics_off": res_off,
            "subset_metrics_on": res_on,
            "deltas": delta,
            "config": dict(cfg),
            "hook_latency_ms": {
                "count": hook_latency_stats["count"],
                "avg_ms": (
                    hook_latency_stats["total_ms"] / hook_latency_stats["count"]
                    if hook_latency_stats["count"]
                    else 0.0
                ),
            },
            "latency_guard": {
                "baseline_ms": latency_off,
                "treatment_ms": latency_on,
                "increase_pct": latency_increase_pct,
                "threshold_pct": cfg["LATENCY_INCREASE_MAX_PCT"],
            },
        },
        out=f"{cfg['ART_DIR']}/day3_4_report.html",
    )

    emp.summarize_to_markdown(
        out=f"{cfg['ART_DIR']}/summary_day3_day4.md",
        sections=[
            {
                "title": "Config",
                "bullets": [
                    f"TOPK={cfg['TOPK']}",
                    f"Embedder={cfg['EMBED_MODEL']}",
                    f"Regime K={cfg['REGIME_K']}",
                ],
            },
            {"title": "Baseline (mem OFF)", "bullets": [f"run id: {run_off}"]},
            {"title": "Treatment (mem ON)", "bullets": [f"run id: {run_on}"]},
            {"title": "Regime-Repeat Deltas", "bullets": [str(delta)]},
            {
                "title": "Decision",
                "bullets": [
                    f"{'APPROVED' if decision['ok'] else 'REJECTED'} — {decision['reason']}"
                ],
            },
        ],
    )

    emp.log(
        "Mini-Cycle Days 3–4 complete. Artifacts at " + str(cfg["ART_DIR"]),
        level="info",
    )

    return {
        "baseline_run": run_off,
        "treatment_run": run_on,
        "delta": delta,
        "decision": decision,
        "subset": subset,
        "metrics_off": res_off,
        "metrics_on": res_on,
        "config": dict(cfg),
        "hook_latency_ms": {
            "count": hook_latency_stats["count"],
            "avg_ms": (
                hook_latency_stats["total_ms"] / hook_latency_stats["count"]
                if hook_latency_stats["count"]
                else 0.0
            ),
        },
        "latency_guard": {
            "baseline_ms": latency_off,
            "treatment_ms": latency_on,
            "increase_pct": latency_increase_pct,
            "threshold_pct": cfg["LATENCY_INCREASE_MAX_PCT"],
        },
    }
