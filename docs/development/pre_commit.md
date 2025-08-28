# Pre-commit: local formatting, linting, and typing

This repository uses pre-commit to run fast, local checks on changed files before each commit. It keeps pull requests green by catching issues early and auto-fixing what can be fixed.

Config file:
- [.pre-commit-config.yaml](.pre-commit-config.yaml:1)

Related configs used by hooks:
- [mypy.ini](mypy.ini:1)

Related CI workflows for typing:
- [typing.yml](.github/workflows/typing.yml:1)
- [typing-nightly.yml](.github/workflows/typing-nightly.yml:1)

Python version alignment for typing
- The project’s typing standard is Python 3.11 with mypy==1.17.1 (pinned in [requirements/dev.txt](requirements/dev.txt:1)) to align with CI.
- If your host Python differs, prefer one of:
  - Run the Dockerized py311 runner [scripts/mypy_py311.sh](scripts/mypy_py311.sh:1) before committing to cross-check typing under the CI-aligned environment.
  - Rely on CI’s py311 typing jobs (PR workflow [typing.yml](.github/workflows/typing.yml:1) and nightly [typing-nightly.yml](.github/workflows/typing-nightly.yml:1)) for authoritative validation.
- Keep pre-commit hooks installed and run “pre-commit run -a” locally for fast feedback; note that authoritative typing results are under py311 + mypy==1.17.1.

Installation:
- pip install -r [requirements/dev.txt](requirements/dev.txt:1)

First-time setup:
- pre-commit install

Run on entire repository:
- pre-commit run --all-files

Update hooks:
- pre-commit autoupdate
- Review and commit changes to [.pre-commit-config.yaml](.pre-commit-config.yaml:1)

What runs and order:
1) Ruff (autofix) — ruff --fix --exit-non-zero-on-fix
2) Black
3) isort (profile=black)
4) mypy (changed files; uses [mypy.ini](mypy.ini:1))

Niceties:
- end-of-file-fixer
- trailing-whitespace

CI notes:
- Full-repo typing runs in [typing.yml](.github/workflows/typing.yml:1) and nightly via [typing-nightly.yml](.github/workflows/typing-nightly.yml:1).
- Pre-commit helps keep PRs green by catching formatting, linting, and typing issues locally.

Tips:
- If a hook modifies files (Ruff/Black/isort), re-stage and re-commit.
- Run a single hook: pre-commit run ruff --all-files