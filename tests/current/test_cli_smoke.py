import importlib
import sys
import types
from pathlib import Path

import pytest


def _stub_heavy_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    # Provide minimal pandas with DataFrame attr for type annotations
    pd = types.ModuleType("pandas")

    class _DF:  # placeholder
        pass

    pd.DataFrame = _DF  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pandas", pd)

    # Provide minimal yfinance module
    yf = types.ModuleType("yfinance")
    monkeypatch.setitem(sys.modules, "yfinance", yf)


def _stub_ui_pkg(monkeypatch: pytest.MonkeyPatch) -> None:
    # Avoid executing src.ui.__init__ which tries to import non-existent 'app' from main_cli
    pkg = types.ModuleType("src.ui")
    pkg.__path__ = [str(Path("src/ui").resolve())]  # mark as package with filesystem path
    monkeypatch.setitem(sys.modules, "src.ui", pkg)


def test_help_shows(monkeypatch: pytest.MonkeyPatch) -> None:
    _ = pytest.importorskip("click")
    from click.testing import CliRunner  # type: ignore

    _stub_heavy_deps(monkeypatch)
    _stub_ui_pkg(monkeypatch)
    # Import CLI after stubbing
    mod = importlib.import_module("src.ui.cli.main_cli")
    cli = getattr(mod, "cli")
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "EMP Ultimate Architecture v1.1 CLI Interface" in result.output


def test_version_command(monkeypatch: pytest.MonkeyPatch) -> None:
    _ = pytest.importorskip("click")
    from click.testing import CliRunner  # type: ignore

    _stub_heavy_deps(monkeypatch)
    _stub_ui_pkg(monkeypatch)
    # Import CLI after stubbing; reload to ensure isolation if previously imported
    mod = importlib.import_module("src.ui.cli.main_cli")
    cli = getattr(mod, "cli")
    runner = CliRunner()
    result = runner.invoke(cli, ["version"])
    assert result.exit_code == 0
    out = result.output
    assert "EMP Ultimate Architecture v1.1" in out
