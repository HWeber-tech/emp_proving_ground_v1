# EMP Proving Ground (FIX‑First)
[![Type Safety: 0 errors](https://img.shields.io/badge/Type%20Safety-0%20errors-brightgreen?style=flat&logo=python)](mypy_snapshots/mypy_summary_2025-08-27T09-43-50Z.txt:1)

A lean, FIX‑only algorithmic trading platform for research, simulation, and controlled execution. The repository reflects a consolidated, production‑lean layout with clear entry points, safe defaults, and an emphasis on risk‑aware design.

## What this project is

- A FIX‑first trading system focused on robust connectivity, clean separation of concerns, and reproducible research.
- A pragmatic codebase for building and validating signal pipelines, risk controls, and execution models offline before any live activity.
- An evolving foundation for professional workflows: generate features, size positions, enforce portfolio caps, and measure outcomes.

Policy note: OpenAPI/FastAPI code is removed by design; the platform operates exclusively with FIX.

## Core capabilities

- FIX connectivity with a compatibility manager in `src/operational/fix_connection_manager.py` (mockable for local runs)
- Strategy and risk consolidation in `src/core/*` with minimal surface area
- Offline backtesting utilities and feature generation
- Portfolio caps and USD beta constraints (`src/trading/risk/portfolio_caps.py`)
- Clean test suite under `tests/current/` (11/11 passing)

## Quick start

```bash
# Install
python -m pip install -r requirements.txt

# Set up environment variables (copy template and edit values)
cp env_templates/.env.example .env
# Edit .env and provide real credentials before running anything live

# (Optional) Configure venue specifics
# cp config/trading/icmarkets_config.py.example config/trading/icmarkets_config.py

# Run tests (current suite)
make test

# Backtest report (offline example)
python scripts/backtest_report.py --file data/mock/md.jsonl --macro-file data/macro/calendar.jsonl --yields-file data/macro/yields.jsonl --parquet || true

# Run main in mock FIX mode (safe)
EMP_USE_MOCK_FIX=1 python main.py
```

## Configuration

This project uses environment variables for configuration via a typed Pydantic `SystemConfig`.

1) Copy the example env file:

```bash
cp env_templates/.env.example .env
```

2) Edit `.env` and set your values. Key variables:

- `RUN_MODE` (default `paper`)
- `CONFIRM_LIVE` (must be `true` to enable live)
- `EMP_KILL_SWITCH` (path to kill-switch file)
- `EMP_ENVIRONMENT` (`demo` or `production`)
- `CONNECTION_PROTOCOL` (must be `fix`; OpenAPI disabled by policy)
- `EMP_TIER` (`tier_0`, `tier_1`, `tier_2`)
- `ICMARKETS_ACCOUNT`, `ICMARKETS_PASSWORD`

3) Tier‑0 ingest flags in `main.py`:

- `--symbols` Comma-separated symbols for ingest (default: `EURUSD,GBPUSD`)
- `--db` DuckDB path (default: `data/tier0.duckdb`)
- `--skip-ingest` Skip Tier-0 ingest at startup

## Repository layout (high‑level)

```text
config/                 # Canonical configuration (FIX, execution, vol, why, etc.)
docs/                   # Architecture, FIX guides, and cleanup reports
scripts/                # Utilities (cleanup report, backtest report, etc.)
src/
  core/                # Consolidated strategy/evolution/risk surfaces
  trading/             # Execution model, monitoring, portfolio risk utilities
  operational/         # FIX connection manager and metrics
  governance/          # SystemConfig and controls
  data_foundation/     # Config loaders, replay/persist utilities
  sensory/             # Minimal WHY utilities and shims
tests/current/          # Canonical test suite
```

## Documentation

- [FIX API Development Guide](docs/fix_api/FIX_API_MASTER_GUIDE.md)
- [Architecture Reality](docs/ARCHITECTURE_REALITY.md)
- [Roadmap (FIX‑first)](docs/reports/ROADMAP_FIX_FIRST.md)
- [Roadmap (Authoritative)](docs/ROADMAP.md)
- [Roadmap (Comprehensive, budget‑tiered)](docs/reports/ROADMAP_COMPREHENSIVE.md)
- [Cleanup Report](docs/reports/CLEANUP_REPORT.md)

### Type Safety

- The project maintains a zero-error mypy baseline; see the [Latest clean snapshot](mypy_snapshots/mypy_summary_2025-08-27T09-43-50Z.txt:1).
- PRs run changed-files strict-on-touch and a full-repo job in [typing.yml](.github/workflows/typing.yml:1).
- A nightly full-repo snapshot runs via [typing-nightly.yml](.github/workflows/typing-nightly.yml:1).
- Contributor guidance: see [Typing recipes](docs/development/typing_recipes.md:1).

### Transparency Metrics
Run quick status metrics:
```bash
python scripts/status_metrics.py
python scripts/cleanup/generate_cleanup_report.py
```

## Safety & production notes

- FIX‑only policy is enforced at config and runtime.
- Use mock FIX (`EMP_USE_MOCK_FIX=1`) unless you explicitly intend to connect.
- Portfolio and USD beta caps are applied in offline simulations.

## Deeper reference

- Root encyclopaedia: `EMP_ENCYCLOPEDIA_v2.3_CANONICAL.md` (kept for forthcoming consolidation)

## License

Proprietary — All rights reserved
