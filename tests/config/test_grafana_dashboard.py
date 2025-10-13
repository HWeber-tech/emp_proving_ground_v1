import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_PATH = (
    REPO_ROOT / "config" / "grafana" / "dashboards" / "json" / "emp_observability.json"
)


def test_dashboard_structure_and_datasource_uid() -> None:
    with DASHBOARD_PATH.open("r", encoding="utf-8") as handle:
        dashboard = json.load(handle)

    assert dashboard["title"] == "EMP Observability SLOs"
    assert dashboard["uid"] == "emp-observability"

    for panel in dashboard.get("panels", []):
        datasource = panel.get("datasource", {})
        if datasource:
            assert datasource.get("uid") == "emp-prometheus"
            assert datasource.get("type") == "prometheus"


def test_dashboard_targets_cover_core_metrics() -> None:
    with DASHBOARD_PATH.open("r", encoding="utf-8") as handle:
        dashboard = json.load(handle)

    expressions: set[str] = set()
    for panel in dashboard.get("panels", []):
        for target in panel.get("targets", []):
            expr = target.get("expr") or target.get("expression")
            if expr:
                expressions.add(expr)

    expected_snippets = {
        "understanding_loop_latency_status",
        "understanding_loop_latency_seconds",
        "drift_alert_freshness_status",
        "drift_alert_freshness_seconds",
        "replay_determinism_status",
        "replay_determinism_drift",
    }
    for snippet in expected_snippets:
        assert any(snippet in expr for expr in expressions), f"Missing metric: {snippet}"
