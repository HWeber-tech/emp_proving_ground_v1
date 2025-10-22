"""Utilities for importing Kibana dashboards via the Saved Objects API."""

from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import requests
from requests import Response
from requests.exceptions import RequestException


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DASHBOARD_DIR = _REPO_ROOT / "config" / "kibana" / "dashboards"


@dataclass(frozen=True)
class KibanaImportResult:
    """Result from attempting to import a dashboard bundle."""

    file: Path
    success: bool
    status_code: int
    details: str | None = None

    @property
    def filename(self) -> str:
        return self.file.name


@dataclass(frozen=True)
class SavedObjectSummary:
    """High-level view of a Kibana saved object contained in an export bundle."""

    file: Path
    object_type: str
    title: str
    description: str | None
    query: str | None
    data_source: str | None

    @property
    def display_type(self) -> str:
        return self.object_type.replace("_", " ").title()


def _collect_ndjson_files(directory: Path) -> list[Path]:
    files = [path for path in sorted(directory.glob("*.ndjson")) if path.is_file()]
    if not files:
        raise FileNotFoundError(
            f"No Kibana dashboard exports (.ndjson) found in {directory}"  # pragma: no cover - defensive
        )
    return files


def _build_import_url(base_url: str, space: str | None) -> str:
    base = base_url.rstrip("/")
    if space and space != "default":
        return f"{base}/s/{space}/api/saved_objects/_import?overwrite=true"
    return f"{base}/api/saved_objects/_import?overwrite=true"


def _zip_ndjson_file(path: Path) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(path, arcname=path.name)
    return buffer.getvalue()


def _load_json_lines(path: Path) -> Iterable[dict[str, object]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:  # pragma: no cover - defensive guard
            continue
        if isinstance(payload, dict):
            yield payload


def _extract_search_source(attributes: dict[str, object]) -> tuple[str | None, str | None]:
    meta = attributes.get("kibanaSavedObjectMeta")
    if not isinstance(meta, dict):
        return None, None
    raw_source = meta.get("searchSourceJSON")
    if not isinstance(raw_source, str):
        return None, None
    try:
        source = json.loads(raw_source)
    except json.JSONDecodeError:
        return None, None
    query = source.get("query")
    query_text = query.get("query") if isinstance(query, dict) else None
    index = source.get("index") if isinstance(source.get("index"), str) else None
    return query_text, index


def _summaries_for_file(path: Path) -> list[SavedObjectSummary]:
    summaries: list[SavedObjectSummary] = []
    for payload in _load_json_lines(path):
        object_type = payload.get("type")
        attributes = payload.get("attributes")
        if not isinstance(object_type, str) or not isinstance(attributes, dict):
            continue
        title = attributes.get("title") if isinstance(attributes.get("title"), str) else "(untitled)"
        description = (
            attributes.get("description")
            if isinstance(attributes.get("description"), str)
            else None
        )
        query_text, index = _extract_search_source(attributes)
        summaries.append(
            SavedObjectSummary(
                file=path,
                object_type=object_type,
                title=title,
                description=description,
                query=query_text,
                data_source=index,
            )
        )
    return summaries


def summarize_dashboards(directory: Path) -> dict[Path, list[SavedObjectSummary]]:
    files = _collect_ndjson_files(directory)
    return {path: _summaries_for_file(path) for path in files}


def _extract_details(response: Response) -> tuple[bool, str | None]:
    try:
        payload = response.json()
    except json.JSONDecodeError:
        return False, response.text

    success = bool(payload.get("success")) and not payload.get("errors")
    if success:
        return True, None

    errors = payload.get("errors")
    if isinstance(errors, Sequence):
        fragments: list[str] = []
        for error in errors:
            if not isinstance(error, dict):
                continue
            identifier = error.get("id")
            error_payload = error.get("error") if isinstance(error.get("error"), dict) else {}
            status_code = error_payload.get("statusCode") if isinstance(error_payload, dict) else None
            message = error_payload.get("message") if isinstance(error_payload, dict) else None
            fragment_parts = []
            if identifier:
                fragment_parts.append(f"id={identifier}")
            if status_code:
                fragment_parts.append(f"status={status_code}")
            if message:
                fragment_parts.append(f"message={message}")
            if fragment_parts:
                fragments.append("; ".join(fragment_parts))
        if fragments:
            return False, " | ".join(fragments)
    return False, json.dumps(payload, sort_keys=True)


def _post_import(
    path: Path,
    *,
    url: str,
    headers: dict[str, str],
    auth: tuple[str, str] | None,
    timeout: float,
) -> KibanaImportResult:
    zip_payload = _zip_ndjson_file(path)
    files = {"file": (f"{path.stem}.ndjson.zip", zip_payload, "application/zip")}
    try:
        response = requests.post(
            url,
            headers=headers,
            files=files,
            auth=auth,
            timeout=timeout,
        )
        success, details = _extract_details(response)
        return KibanaImportResult(
            file=path,
            success=success,
            status_code=response.status_code,
            details=details,
        )
    except RequestException as exc:  # pragma: no cover - defensive networking guard
        response = getattr(exc, "response", None)
        status_code = response.status_code if response is not None else 0
        return KibanaImportResult(file=path, success=False, status_code=status_code, details=str(exc))


def deploy_dashboards(
    directory: Path,
    *,
    kibana_url: str,
    space: str | None = "default",
    username: str | None = None,
    password: str | None = None,
    api_key: str | None = None,
    timeout: float = 30.0,
) -> list[KibanaImportResult]:
    files = _collect_ndjson_files(directory)
    url = _build_import_url(kibana_url, space)

    headers = {"kbn-xsrf": "emp-automation"}
    auth = None
    if api_key:
        headers["Authorization"] = f"ApiKey {api_key}"
    elif username and password:
        auth = (username, password)

    results: list[KibanaImportResult] = []
    for path in files:
        results.append(
            _post_import(
                path,
                url=url,
                headers=headers,
                auth=auth,
                timeout=timeout,
            )
        )
    return results


def _parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kibana-url",
        required=True,
        help="Base URL for the Kibana instance (e.g. http://localhost:5601)",
    )
    parser.add_argument(
        "--directory",
        type=Path,
        default=_DEFAULT_DASHBOARD_DIR,
        help="Directory containing Kibana dashboard exports (.ndjson)",
    )
    parser.add_argument(
        "--space",
        default="default",
        help="Kibana space to import dashboards into (default: default)",
    )
    parser.add_argument("--username", help="Username for basic authentication", default=None)
    parser.add_argument("--password", help="Password for basic authentication", default=None)
    parser.add_argument("--api-key", help="API key for authentication", default=None)
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Exit with status 0 even if some imports fail",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Suppress saved object summary output",
    )
    return parser.parse_args(args)


def main(args: Sequence[str] | None = None) -> int:
    options = _parse_args(args)
    if not options.no_summary:
        summary_map = summarize_dashboards(options.directory)
        for path, summaries in summary_map.items():
            relative_path = path.relative_to(_REPO_ROOT)
            print(f"=== {relative_path} ===")
            if not summaries:
                print("  (no saved objects detected)")
            for summary in summaries:
                parts = [f"- {summary.display_type}: {summary.title}"]
                if summary.data_source:
                    parts.append(f"index={summary.data_source}")
                if summary.query:
                    parts.append(f"query={summary.query}")
                if summary.description:
                    parts.append(f"description={summary.description}")
                print("  " + " | ".join(parts))
            print()
    results = deploy_dashboards(
        options.directory,
        kibana_url=options.kibana_url,
        space=options.space,
        username=options.username,
        password=options.password,
        api_key=options.api_key,
        timeout=options.timeout,
    )

    failures = [result for result in results if not result.success]

    for result in results:
        status = "SUCCESS" if result.success else "FAILED"
        message = f"[{status}] {result.filename} (HTTP {result.status_code})"
        if result.details:
            message += f": {result.details}"
        print(message)

    if failures and not options.allow_partial:
        print(
            f"Import failed for {len(failures)} dashboard bundle(s)",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
