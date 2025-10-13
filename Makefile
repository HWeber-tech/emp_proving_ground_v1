# MyPy workflow Makefile
# Standardizes quick runs, daemon usage, summaries, and the explicit-any rewriter.

MYPY=./.venv_mypy/bin/mypy
DMPY=./.venv_mypy/bin/dmypy
PY=./.venv_mypy/bin/python

SRC=src
CFG=mypy.ini

.PHONY: help
help:
	@echo "Targets:"
	@echo "  mypy                  - Run mypy with repo config"
	@echo "  type-summary          - Run reporter to emit JSON/CSV and console summary"
	@echo "  dmypy-start           - Start dmypy and warm caches"
	@echo "  dmypy-status          - Show dmypy status"
	@echo "  dmypy-check           - Run type check via dmypy"
	@echo "  dmypy-stop            - Stop dmypy"
	@echo "  type-explicit-any-dry - Dry-run explicit Any rewriter (prints diffs)"
	@echo "  type-explicit-any-apply - Apply explicit Any fixes in-place (types only)"
	@echo "  run-sim               - Run deterministic bootstrap simulation (summary + diary)"
	@echo "  nightly-replay        - Run nightly replay harness orchestrator"

.PHONY: mypy
mypy:
	@MYPYPATH="stubs:$(SRC)$${MYPYPATH:+:$$MYPYPATH}" \
	$(MYPY) $(SRC) --config-file $(CFG) --show-error-codes --no-color-output

.PHONY: type-summary
type-summary:
	$(PY) scripts/analysis/run_mypy_summary.py

.PHONY: dmypy-start
dmypy-start:
	@MYPYPATH="stubs:$(SRC)$${MYPYPATH:+:$$MYPYPATH}" \
	$(DMPY) run -- --config-file $(CFG) --show-error-codes --no-color-output $(SRC) || true

.PHONY: dmypy-status
dmypy-status:
	$(DMPY) status || true

.PHONY: dmypy-check
dmypy-check:
	@MYPYPATH="stubs:$(SRC)$${MYPYPATH:+:$$MYPYPATH}" \
	$(DMPY) run -- --config-file $(CFG) --show-error-codes --no-color-output $(SRC)

.PHONY: dmypy-stop
dmypy-stop:
	$(DMPY) stop || true

# ------------------------------------------------------------------------
# Bulk type cleanup wiring (Any->object etc.) via bulk driver script
# Defaults can be overridden on invocation:
#   make type-explicit-any-dry DIRS="src/core src/thinking" JOBS=4
# ------------------------------------------------------------------------
DIRS ?= src/core src/thinking src/trading src/ecosystem
INCLUDE ?=
EXCLUDE ?= stubs|tests|migrations|docs|__pycache__
JOBS ?= 8

.PHONY: type-explicit-any-dry
type-explicit-any-dry:
	bash scripts/cleanup/bulk_type_fix.sh --dirs "$(DIRS)" --include-regex "$(INCLUDE)" --exclude-regex "$(EXCLUDE)" --jobs "$(JOBS)"

.PHONY: type-explicit-any-apply
type-explicit-any-apply:
	bash scripts/cleanup/bulk_type_fix.sh --apply --dirs "$(DIRS)" --include-regex "$(INCLUDE)" --exclude-regex "$(EXCLUDE)" --jobs "$(JOBS)" --dmypy-check

# ------------------------------------------------------------------------
# Reflection Intelligence Module tooling
# ------------------------------------------------------------------------
RIM_CONFIG ?= config/reflection/rim.config.yml

.PHONY: rim-shadow
rim-shadow:
	python tools/rim_shadow_run.py --config $(RIM_CONFIG)

.PHONY: rim-validate
rim-validate:
	python tools/rim_validate.py

.PHONY: rim-prune
rim-prune:
	python tools/rim_prune.py

.PHONY: docs-check
docs-check:
	python tools/docs_check.py

# ------------------------------------------------------------------------
# Runtime convenience entrypoints
# ------------------------------------------------------------------------
RUN_ARGS ?=
LEDGER_PATH ?= artifacts/governance/policy_ledger.json
PHENOTYPE_DIR ?= artifacts/policies

.PHONY: run-paper
run-paper:
	python -m src.runtime.cli paper-run $(RUN_ARGS)

SIM_TIMEOUT ?= 15
SIM_TICK_INTERVAL ?= 0.5
SIM_MAX_TICKS ?= 120
SIM_SYMBOLS ?= EURUSD
SIM_SUMMARY ?= artifacts/sim/summary.json
SIM_DIARY ?= artifacts/diaries/sim.jsonl
SIM_DUCKDB ?= data/tier0.duckdb
SIM_EXTRA_ARGS ?=

.PHONY: run-sim
run-sim:
	python3 tools/runtime/run_simulation.py \
		--timeout $(SIM_TIMEOUT) \
		--tick-interval $(SIM_TICK_INTERVAL) \
		--max-ticks $(SIM_MAX_TICKS) \
		--symbols "$(SIM_SYMBOLS)" \
		--summary-path "$(SIM_SUMMARY)" \
		--diary-path "$(SIM_DIARY)" \
		--duckdb-path "$(SIM_DUCKDB)" $(SIM_EXTRA_ARGS)

.PHONY: rebuild-policy
rebuild-policy:
	@if [ -z "$(HASH)" ]; then \
		echo "HASH variable is required, e.g. make rebuild-policy HASH=alpha.policy"; \
		exit 1; \
	fi
	@echo "Rebuilding policy phenotype for $(HASH)"
	python3 -m tools.governance.rebuild_policy \
		--ledger "$(LEDGER_PATH)" \
		--policy "$(HASH)" \
		--phenotype-dir "$(PHENOTYPE_DIR)" \
		--output "$(PHENOTYPE_DIR)/summary.json" \
		--indent 2
	@echo "Phenotype bundles available under $(PHENOTYPE_DIR)"

.PHONY: nightly-replay
nightly-replay:
	python3 tools/operations/nightly_replay_job.py
