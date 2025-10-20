from src.governance.system_config import (
    ConnectionProtocol,
    RunMode,
    SystemConfig,
    render_config_diff,
)


def test_render_config_diff_defaults() -> None:
    diff = render_config_diff(SystemConfig())
    lines = diff.splitlines()

    assert lines[0] == "\x1b[1mLive config diff (vs defaults)\x1b[0m"
    assert "\x1b[90m= run_mode: paper\x1b[0m" in lines
    assert "\x1b[90m= confirm_live: false\x1b[0m" in lines
    assert any(line.startswith("\x1b[90m= kill_switch_path: ") for line in lines)
    assert lines[-1] == "\x1b[90m= extras: (none)\x1b[0m"


def test_render_config_diff_highlights_changes() -> None:
    config = SystemConfig().with_updated(
        run_mode=RunMode.live,
        confirm_live=True,
        connection_protocol=ConnectionProtocol.fix,
        extras={"LIVE_FLAG": "enabled"},
    )

    diff = render_config_diff(config)
    lines = diff.splitlines()

    assert "\x1b[31m- run_mode: paper\x1b[0m" in lines
    assert "\x1b[32m+ run_mode: live\x1b[0m" in lines
    assert "\x1b[31m- confirm_live: false\x1b[0m" in lines
    assert "\x1b[32m+ confirm_live: true\x1b[0m" in lines
    assert "\x1b[31m- connection_protocol: bootstrap\x1b[0m" in lines
    assert "\x1b[32m+ connection_protocol: fix\x1b[0m" in lines
    assert "\x1b[31m- extras.LIVE_FLAG: <unset>\x1b[0m" in lines
    assert "\x1b[32m+ extras.LIVE_FLAG: enabled\x1b[0m" in lines
