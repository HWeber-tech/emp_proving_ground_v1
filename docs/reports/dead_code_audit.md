# Dead code audit – 2025-10-08 05:44 UTC

*Generated via `vulture src tests --min-confidence 80 --sort-by-size`.*

## Summary

- **Total candidates**: 0
- **Command exit status**: vulture exited with status 0.
- **By symbol type**: n/a
- **Top modules**: n/a

## Observations

- All previously identified high-confidence candidates have been addressed. The audit now reports zero unused symbols at ≥80% confidence.
- Keep rerunning the audit after major refactors to ensure compatibility shims or unused protocol parameters do not reappear.

## Next steps

1. Add the 80% confidence vulture sweep to recurring hygiene checks so regressions are caught quickly.
2. Document expectations for new protocols/helpers so contributors avoid introducing unused parameters that trigger future audits.
3. Regenerate this report after major feature work to keep the shared context current.
