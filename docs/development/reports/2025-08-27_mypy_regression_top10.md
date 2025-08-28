# Mypy regression triage: Top 10 modules and error codes (2025-08-27)

Totals: Found 124 errors in 28 files (checked 343 source files)

Top 10 offender modules
- path | count
- src/thinking/patterns/trend_detector.py | 18
- src/thinking/patterns/cycle_detector.py | 11
- src/data_integration/data_fusion.py | 10
- src/intelligence/sentient_adaptation.py | 9
- src/sensory/organs/dimensions/institutional_tracker.py | 8
- src/trading/portfolio/real_portfolio_monitor.py | 7
- src/sensory/organs/dimensions/integration_orchestrator.py | 7
- src/evolution/mutation/gaussian_mutation.py | 7
- src/genome/models/genome_adapter.py | 6
- src/orchestration/enhanced_intelligence_engine.py | 5

Top 10 error codes
- error_code | count
- attr-defined | 53
- assignment | 23
- misc | 10
- object | 9
- arg-type | 9
- no-any-return | 6
- call-arg | 6
- operator | 5
- import-not-found | 5
- type-var | 3

Fix patterns by code
- attr-defined → see patterns in typing recipes
- arg-type → numeric normalization and guards
- call-arg → overload narrowing
- union-attr → isinstance + cast
- return-value → explicit return annotations
- assignment → typed locals
- index → key type guards
- misc → TYPE_CHECKING imports

References
- Base report: docs/development/reports/2025-08-27_mypy_regression_report.md
- Ranked offenders CSV: mypy_snapshots/mypy_ranked_offenders_2025-08-27T15-21-18Z.csv
- Error-code histogram CSV: mypy_snapshots/mypy_error_codes_2025-08-27T15-44-57Z.csv
- Summary (base): mypy_snapshots/mypy_summary_2025-08-27T15-21-18Z.txt
- Summary (detailed): mypy_snapshots/mypy_summary_detailed_2025-08-27T15-44-57Z.txt
- Typing recipes: docs/development/typing_recipes.md

Statement
This report is diagnostics-only; no code changes performed.

Repro/automation snippet (reads CSVs, writes this report and candidates list atomically)
```python
from __future__ import annotations
import csv
from pathlib import Path

def read_top_n(csv_path: str, n: int) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            # Handles either "path,count" or "code,count"
            key = row.get("path") or row.get("code")
            if key is None:
                continue
            cnt = int(row["count"])
            rows.append((key, cnt))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows[:n]

def main() -> None:
    base_dir = Path(".")
    offenders_csv = base_dir / "mypy_snapshots/mypy_ranked_offenders_2025-08-27T15-21-18Z.csv"
    codes_csv = base_dir / "mypy_snapshots/mypy_error_codes_2025-08-27T15-44-57Z.csv"
    summary_txt = base_dir / "mypy_snapshots/mypy_summary_2025-08-27T15-21-18Z.txt"

    top_offenders = read_top_n(str(offenders_csv), 10)
    top_codes = read_top_n(str(codes_csv), 10)

    totals_line = summary_txt.read_text().strip()

    report_path = base_dir / "docs/development/reports/2025-08-27_mypy_regression_top10.md"
    candidates_path = base_dir / "mypy_snapshots/candidates_regression_top10_2025-08-27T15-21-18Z.txt"

    # Write candidates (one path per line)
    candidates_path.parent.mkdir(parents=True, exist_ok=True)
    with open(candidates_path, "w") as f:
        for path, _ in top_offenders:
            f.write(f"{path}\n")

    # Build markdown content
    lines: list[str] = []
    lines.append("# Mypy regression triage: Top 10 modules and error codes (2025-08-27)")
    lines.append("")
    lines.append(f"Totals: {totals_line}")
    lines.append("")
    lines.append("Top 10 offender modules")
    lines.append("- path | count")
    for path, cnt in top_offenders:
        lines.append(f"- {path} | {cnt}")
    lines.append("")
    lines.append("Top 10 error codes")
    lines.append("- error_code | count")
    for code, cnt in top_codes:
        lines.append(f"- {code} | {cnt}")
    lines.append("")
    lines.append("Fix patterns by code")
    lines.append("- attr-defined → see patterns in typing recipes")
    lines.append("- arg-type → numeric normalization and guards")
    lines.append("- call-arg → overload narrowing")
    lines.append("- union-attr → isinstance + cast")
    lines.append("- return-value → explicit return annotations")
    lines.append("- assignment → typed locals")
    lines.append("- index → key type guards")
    lines.append("- misc → TYPE_CHECKING imports")
    lines.append("")
    lines.append("References")
    lines.append("- Base report: docs/development/reports/2025-08-27_mypy_regression_report.md")
    lines.append(f"- Ranked offenders CSV: {offenders_csv.as_posix()}")
    lines.append(f"- Error-code histogram CSV: {codes_csv.as_posix()}")
    lines.append(f"- Summary (base): {summary_txt.as_posix()}")
    lines.append("- Summary (detailed): mypy_snapshots/mypy_summary_detailed_2025-08-27T15-44-57Z.txt")
    lines.append("- Typing recipes: docs/development/typing_recipes.md")
    lines.append("")
    lines.append("Statement")
    lines.append("This report is diagnostics-only; no code changes performed.")
    lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write("\n".join(lines))

if __name__ == "__main__":
    main()
```