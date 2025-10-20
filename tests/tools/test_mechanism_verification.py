from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import pytest
import yaml

from tools.roadmap import doc_tasks

REPO_ROOT = Path(__file__).resolve().parents[2]
ROADMAP_PATH = REPO_ROOT / "docs/roadmap.md"
MECHANISM_PATH = REPO_ROOT / "docs/status/mechanism_verification.yaml"


@dataclass(frozen=True)
class FeatureCase:
    """Representation of a roadmap feature and its mechanism metadata."""

    name: str
    slug: str
    section: str
    checked: bool


def _slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


def _load_roadmap_features() -> tuple[FeatureCase, ...]:
    tasks = doc_tasks.extract_document_tasks(ROADMAP_PATH)
    cases: list[FeatureCase] = []
    for task in tasks:
        if not task.section:
            continue
        if task.section[0] != "EMP Final Roadmap - Apex Predator Edition":
            continue
        match = re.search(r"\*\*(?P<name>[^*]+)\*\*", task.description)
        if not match:
            continue
        name = match.group("name").strip()
        slug = _slugify(name)
        section = task.section[-1]
        cases.append(FeatureCase(name=name, slug=slug, section=section, checked=task.checked))
    return tuple(cases)


def _load_mechanism_entries() -> dict[str, dict[str, object]]:
    with MECHANISM_PATH.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    entries: dict[str, dict[str, object]] = {}
    for entry in payload.get("features", ()):  # type: ignore[arg-type]
        slug = entry.get("slug")
        if not isinstance(slug, str):
            raise AssertionError("mechanism entry missing slug")
        if slug in entries:
            raise AssertionError(f"duplicate slug detected: {slug}")
        entries[slug] = entry
    return entries


ROADMAP_CASES = _load_roadmap_features()
MECHANISM_ENTRIES = _load_mechanism_entries()


def _ci_test_paths(entry: dict[str, object]) -> Iterable[str]:
    ci_tests = entry.get("ci_tests", [])
    if not isinstance(ci_tests, list):
        raise AssertionError("ci_tests must be a list")
    for item in ci_tests:
        if not isinstance(item, str):
            raise AssertionError("ci_test entries must be strings")
        yield item


def test_mechanism_entries_cover_all_features() -> None:
    roadmap_slugs = {case.slug for case in ROADMAP_CASES}
    entry_slugs = set(MECHANISM_ENTRIES)
    assert entry_slugs == roadmap_slugs


@pytest.mark.parametrize("case", ROADMAP_CASES, ids=lambda case: case.slug)
def test_feature_hypothesis(case: FeatureCase) -> None:
    entry = MECHANISM_ENTRIES[case.slug]

    assert entry["name"] == case.name
    assert entry["section"] == case.section

    hypothesis = entry.get("economic_hypothesis")
    assert isinstance(hypothesis, dict)

    null_hypothesis = hypothesis.get("null_hypothesis")
    alternative_hypothesis = hypothesis.get("alternative_hypothesis")
    validation_metric = hypothesis.get("validation_metric")

    assert isinstance(null_hypothesis, str) and null_hypothesis.startswith("Null hypothesis:")
    assert case.name in null_hypothesis

    assert isinstance(alternative_hypothesis, str) and alternative_hypothesis.startswith(
        "Alternative hypothesis:"
    )
    assert case.name in alternative_hypothesis

    assert isinstance(validation_metric, str) and validation_metric.startswith("Validation metric:")

    ci_tests = list(_ci_test_paths(entry))
    assert ci_tests, "expected at least one CI test reference"
    expected_test = f"tests/tools/test_mechanism_verification.py::test_feature_hypothesis[{case.slug}]"
    assert expected_test in ci_tests

    for reference in ci_tests:
        path_part, _, _ = reference.partition("::")
        assert (REPO_ROOT / path_part).exists(), f"missing CI test file for {reference}"
