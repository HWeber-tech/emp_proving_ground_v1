from __future__ import annotations

from src.governance.system_config import SystemConfig


def test_final_dry_run_diary_path_mirrors_decision_diary() -> None:
    env = {
        "FINAL_DRY_RUN_DIARY_PATH": "/tmp/dry-run/diary.jsonl",
    }
    cfg = SystemConfig.from_env(env)
    extras = cfg.extras

    assert extras["FINAL_DRY_RUN_DIARY_PATH"] == "/tmp/dry-run/diary.jsonl"
    assert extras["DECISION_DIARY_PATH"] == "/tmp/dry-run/diary.jsonl"


def test_final_dry_run_diary_path_does_not_override_existing() -> None:
    env = {
        "FINAL_DRY_RUN_DIARY_PATH": "/tmp/dry-run/diary.jsonl",
        "DECISION_DIARY_PATH": "/tmp/custom/diary.jsonl",
    }
    cfg = SystemConfig.from_env(env)
    extras = cfg.extras

    assert extras["DECISION_DIARY_PATH"] == "/tmp/custom/diary.jsonl"


def test_final_dry_run_performance_path_mirrors_metrics_path() -> None:
    env = {
        "FINAL_DRY_RUN_PERFORMANCE_PATH": "/tmp/dry-run/performance.json",
    }
    cfg = SystemConfig.from_env(env)
    extras = cfg.extras

    assert extras["FINAL_DRY_RUN_PERFORMANCE_PATH"] == "/tmp/dry-run/performance.json"
    assert extras["PERFORMANCE_METRICS_PATH"] == "/tmp/dry-run/performance.json"


def test_final_dry_run_performance_path_does_not_override_existing() -> None:
    env = {
        "FINAL_DRY_RUN_PERFORMANCE_PATH": "/tmp/dry-run/performance.json",
        "PERFORMANCE_METRICS_PATH": "/tmp/custom/perf.json",
    }
    cfg = SystemConfig.from_env(env)
    extras = cfg.extras

    assert extras["PERFORMANCE_METRICS_PATH"] == "/tmp/custom/perf.json"
