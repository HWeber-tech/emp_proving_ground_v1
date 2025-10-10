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
