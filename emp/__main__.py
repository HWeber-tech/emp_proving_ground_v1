"""EMP command line entry points."""
from __future__ import annotations

import click


@click.group()
def cli() -> None:
    """EMP command line interface."""


@cli.command("mini-cycle")
def mini_cycle() -> None:
    """Run EMP mini-cycle Days 1–2 orchestrator."""
    from emp.experiments import run_day1_day2

    run_day1_day2()


if __name__ == "__main__":  # pragma: no cover - CLI manual execution
    cli()
