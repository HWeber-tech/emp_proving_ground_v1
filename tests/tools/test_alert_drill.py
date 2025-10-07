import json
from pathlib import Path

import pytest

from tools.telemetry.alert_drill import main as alert_drill_main


def test_alert_drill_cli_writes_timeline(tmp_path: Path) -> None:
    output = tmp_path / "timeline.json"

    exit_code = alert_drill_main(
        [
            "--incident-id",
            "ci-alert-2025-10-07",
            "--opened-at",
            "2025-10-07T12:00:00+00:00",
            "--ack-at",
            "2025-10-07T12:03:00+00:00",
            "--resolve-at",
            "2025-10-07T12:18:30+00:00",
            "--ack-actor",
            "oncall",
            "--resolve-actor",
            "maintainer",
            "--output",
            str(output),
            "--drill",
        ]
    )

    assert exit_code == 0

    payload = json.loads(output.read_text())
    assert payload["incident_id"] == "ci-alert-2025-10-07"
    assert payload["drill"] is True
    assert payload["events"][0]["channel"] == "github"
    assert payload["events"][1]["channel"] == "slack"
    assert payload["events"][1]["actor"] == "oncall"
    assert payload["events"][2]["actor"] == "maintainer"


def test_alert_drill_cli_rejects_invalid_order(tmp_path: Path) -> None:
    output = tmp_path / "timeline.json"

    with pytest.raises(SystemExit):
        alert_drill_main(
            [
                "--incident-id",
                "ci-alert-invalid",
                "--opened-at",
                "2025-10-07T12:00:00+00:00",
                "--ack-at",
                "2025-10-07T11:59:00+00:00",
                "--output",
                str(output),
            ]
        )
