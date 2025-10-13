import json
import textwrap
from pathlib import Path

from tools import rim_shadow_run

EXAMPLE_DIARIES = Path("docs/examples/trm_exit_drill_diaries.jsonl")


def _write_config(
    tmp_path: Path,
    *,
    diaries_dir: Path,
    publish_dir: Path,
    telemetry_dir: Path,
    governance_dir: Path,
    lock_path: Path,
) -> Path:
    config_text = textwrap.dedent(
        f"""
        diaries_dir: {diaries_dir.as_posix()}
        diary_glob: \"diaries-*.jsonl\"
        window_minutes: 1440
        min_entries: 5
        suggestion_cap: 5
        confidence_floor: 0.5
        enable_governance_gate: true
        publish_channel: file://{publish_dir.as_posix()}
        kill_switch: false
        telemetry:
          log_dir: {telemetry_dir.as_posix()}
        governance:
          queue_path: { (governance_dir / 'queue.jsonl').as_posix() }
          digest_path: { (governance_dir / 'digest.json').as_posix() }
          markdown_path: { (governance_dir / 'digest.md').as_posix() }
        lock_path: {lock_path.as_posix()}
        """
    ).strip()
    config_path = tmp_path / "rim.config.yml"
    config_path.write_text(config_text + "\n", encoding="utf-8")
    return config_path


def test_run_shadow_job_emits_governance_and_suggestions(tmp_path: Path) -> None:
    diaries_dir = tmp_path / "artifacts" / "diaries"
    diaries_dir.mkdir(parents=True, exist_ok=True)
    diary_path = diaries_dir / "diaries-0001.jsonl"
    diary_path.write_text(EXAMPLE_DIARIES.read_text(encoding="utf-8"), encoding="utf-8")

    publish_dir = tmp_path / "artifacts" / "rim_suggestions"
    telemetry_dir = tmp_path / "artifacts" / "rim_logs"
    governance_dir = tmp_path / "artifacts" / "governance"
    lock_path = tmp_path / "artifacts" / "locks" / "rim.lock"

    config_path = _write_config(
        tmp_path,
        diaries_dir=diaries_dir,
        publish_dir=publish_dir,
        telemetry_dir=telemetry_dir,
        governance_dir=governance_dir,
        lock_path=lock_path,
    )

    result = rim_shadow_run.run_shadow_job(config_path, min_entries=1, debug=False)

    assert result.skipped_reason is None
    assert result.suggestions_count > 0
    assert result.suggestions_path is not None and result.suggestions_path.exists()

    suggestions = [
        json.loads(line)
        for line in result.suggestions_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert suggestions, "expected suggestions file to contain entries"
    assert len(suggestions) == result.suggestions_count

    digest_path = governance_dir / "digest.json"
    assert digest_path.exists()
    digest = json.loads(digest_path.read_text(encoding="utf-8"))
    assert digest["run_id"] == result.run_id
    assert digest["suggestion_count"] == result.suggestions_count

    markdown_path = governance_dir / "digest.md"
    assert markdown_path.exists()

    queue_path = governance_dir / "queue.jsonl"
    assert queue_path.exists()
    queue_entries = [
        json.loads(line)
        for line in queue_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(queue_entries) == result.suggestions_count
    assert all(
        entry.get("governance", {}).get("run_id") == result.run_id
        for entry in queue_entries
    )


def test_run_shadow_job_writes_skip_digest_on_empty_diaries(tmp_path: Path) -> None:
    diaries_dir = tmp_path / "artifacts" / "diaries"
    diaries_dir.mkdir(parents=True, exist_ok=True)

    publish_dir = tmp_path / "artifacts" / "rim_suggestions"
    telemetry_dir = tmp_path / "artifacts" / "rim_logs"
    governance_dir = tmp_path / "artifacts" / "governance"
    lock_path = tmp_path / "artifacts" / "locks" / "rim.lock"

    config_path = _write_config(
        tmp_path,
        diaries_dir=diaries_dir,
        publish_dir=publish_dir,
        telemetry_dir=telemetry_dir,
        governance_dir=governance_dir,
        lock_path=lock_path,
    )

    result = rim_shadow_run.run_shadow_job(config_path, min_entries=1, debug=False)

    assert result.skipped_reason == "no_diaries"
    assert result.suggestions_count == 0
    assert result.suggestions_path is None
    assert result.run_id is not None

    digest_path = governance_dir / "digest.json"
    assert digest_path.exists()
    digest = json.loads(digest_path.read_text(encoding="utf-8"))
    assert digest["skip_reason"] == "no_diaries"
    assert digest["suggestion_count"] == 0
    assert digest["run_id"] == result.run_id

    markdown_path = governance_dir / "digest.md"
    assert markdown_path.exists()
    markdown = markdown_path.read_text(encoding="utf-8").lower()
    assert "skipped" in markdown

    queue_path = governance_dir / "queue.jsonl"
    assert not queue_path.exists()
