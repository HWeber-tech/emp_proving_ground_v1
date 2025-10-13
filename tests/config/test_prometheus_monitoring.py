from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_prometheus_config_references_emp_rules() -> None:
    config = _load_yaml(REPO_ROOT / "config" / "prometheus" / "prometheus.yml")
    assert "rule_files" in config, "Prometheus config must declare rule files"
    assert "emp_rules.yml" in config["rule_files"], "emp_rules.yml should be loaded"


def test_emp_rules_define_slo_alerts() -> None:
    rules = _load_yaml(REPO_ROOT / "config" / "prometheus" / "emp_rules.yml")
    assert isinstance(rules.get("groups"), list)

    groups = {group["name"]: group for group in rules["groups"]}
    slo_group = groups.get("understanding-loop-slos")
    assert slo_group, "Missing understanding-loop-slos rule group"

    expected_alerts = {
        "UnderstandingLoopLatencySLOWarn",
        "UnderstandingLoopLatencySLOBreach",
        "DriftAlertFreshnessSLOWarn",
        "DriftAlertFreshnessSLOBreach",
        "ReplayDeterminismSLOWarn",
        "ReplayDeterminismSLOBreach",
    }
    actual_alerts = {rule["alert"] for rule in slo_group.get("rules", [])}
    missing = expected_alerts - actual_alerts
    assert not missing, f"Missing alerts: {sorted(missing)}"

    for rule in slo_group.get("rules", []):
        expr = rule.get("expr", "")
        if rule["alert"].startswith("UnderstandingLoopLatency"):
            assert "understanding_loop_latency_status" in expr
        if rule["alert"].startswith("DriftAlertFreshness"):
            assert "drift_alert_freshness_status" in expr
        if rule["alert"].startswith("ReplayDeterminism"):
            assert "replay_determinism_status" in expr

    common_labels = slo_group.get("common_labels") or {}
    assert common_labels.get("service") == "understanding-loop"
