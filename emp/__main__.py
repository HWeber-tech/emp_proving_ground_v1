"""EMP command line entry points."""
from __future__ import annotations

import click

from emp.cli.final_dry_run import final_dry_run


@click.group()
def cli() -> None:
    """EMP command line interface."""


@cli.command("mini-cycle")
@click.option(
    "--days",
    type=click.Choice(["d1d2", "d3d4"]),
    default="d1d2",
    show_default=True,
    help="Select which mini-cycle orchestration to execute.",
)
def mini_cycle(days: str) -> None:
    """Run EMP mini-cycle orchestrations."""
    from emp.experiments import run_day1_day2, run_day3_day4

    if days == "d3d4":
        run_day3_day4()
    else:
        run_day1_day2()


cli.add_command(final_dry_run)


if __name__ == "__main__":  # pragma: no cover - CLI manual execution
    cli()
