from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from tools.observability.deploy_kibana_dashboards import deploy_dashboards


class _SuccessfulResponse:
    status_code = 200

    @staticmethod
    def json() -> dict[str, object]:
        return {"success": True}


class _ErrorResponse:
    status_code = 400

    @staticmethod
    def json() -> dict[str, object]:
        return {
            "success": False,
            "errors": [
                {
                    "id": "emp-operations-dashboard",
                    "error": {"statusCode": 400, "message": "invalid references"},
                }
            ],
        }


def test_deploy_dashboards_uploads_zipped_exports(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dashboard = tmp_path / "dashboard.ndjson"
    dashboard.write_text("{}\n", encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_post(url: str, *, headers: dict[str, str], files: dict[str, tuple[str, bytes, str]], auth, timeout: float):
        captured["url"] = url
        captured["headers"] = headers
        captured["files"] = files
        captured["auth"] = auth
        captured["timeout"] = timeout
        return _SuccessfulResponse()

    monkeypatch.setattr("tools.observability.deploy_kibana_dashboards.requests.post", fake_post)

    results = deploy_dashboards(
        dashboard.parent,
        kibana_url="http://kibana:5601",
        username="elastic",
        password="changeme",
    )

    assert captured["url"] == "http://kibana:5601/api/saved_objects/_import?overwrite=true"
    headers = captured["headers"]
    assert isinstance(headers, dict)
    assert headers["kbn-xsrf"] == "emp-automation"
    assert captured["auth"] == ("elastic", "changeme")

    files = captured["files"]
    assert isinstance(files, dict)
    name, payload, content_type = files["file"]
    assert name == "dashboard.ndjson.zip"
    assert content_type == "application/zip"

    archive = zipfile.ZipFile(io.BytesIO(payload))
    assert archive.namelist() == ["dashboard.ndjson"]
    assert archive.read("dashboard.ndjson") == dashboard.read_bytes()

    assert len(results) == 1
    assert results[0].success is True
    assert results[0].details is None


def test_deploy_dashboards_surfaces_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dashboard = tmp_path / "dashboard.ndjson"
    dashboard.write_text("{}\n", encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_post(url: str, *, headers: dict[str, str], files: dict[str, tuple[str, bytes, str]], auth, timeout: float):
        captured["url"] = url
        captured["headers"] = headers
        captured["files"] = files
        captured["auth"] = auth
        captured["timeout"] = timeout
        return _ErrorResponse()

    monkeypatch.setattr("tools.observability.deploy_kibana_dashboards.requests.post", fake_post)

    results = deploy_dashboards(
        dashboard.parent,
        kibana_url="http://kibana:5601",
        space="operations",
        api_key="abc123",
    )

    assert captured["url"] == "http://kibana:5601/s/operations/api/saved_objects/_import?overwrite=true"
    headers = captured["headers"]
    assert isinstance(headers, dict)
    assert headers["Authorization"] == "ApiKey abc123"
    assert captured["auth"] is None

    assert len(results) == 1
    result = results[0]
    assert result.success is False
    assert "invalid references" in (result.details or "")
    assert result.status_code == 400
