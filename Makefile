PY?=python

.PHONY: test current backtest

test current:
	EMP_USE_MOCK_FIX=1 $(PY) -m pytest -q

backtest:
	$(PY) scripts/backtest_report.py --file data/mock/md.jsonl --macro-file data/macro/calendar.jsonl --yields-file data/macro/yields.jsonl --parquet || true


