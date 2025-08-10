PY?=python

.PHONY: test current backtest

test current:
	EMP_USE_MOCK_FIX=1 $(PY) -m pytest -q

backtest:
	$(PY) scripts/backtest_report.py --file data/mock/md.jsonl --macro-file data/macro/calendar.jsonl --yields-file data/macro/yields.jsonl --parquet || true

.PHONY: fix-dry-run
fix-dry-run:
	EMP_USE_MOCK_FIX=1 $(PY) scripts/fix_dry_run.py

# Deprecated targets removed in cleanup


