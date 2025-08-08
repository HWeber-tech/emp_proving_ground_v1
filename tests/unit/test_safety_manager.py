#!/usr/bin/env python3

import pytest
from src.governance.safety_manager import SafetyManager


def test_live_without_confirmation_raises():
    manager = SafetyManager(run_mode="live", confirm_live=False, kill_switch_path=None)
    with pytest.raises(RuntimeError):
        manager.enforce()


def test_paper_without_confirmation_ok():
    manager = SafetyManager(run_mode="paper", confirm_live=False, kill_switch_path=None)
    manager.enforce()  # should not raise


def test_kill_switch_raises(tmp_path):
    kill_file = tmp_path / "emp_pg.KILL"
    kill_file.write_text("halt")
    manager = SafetyManager(run_mode="paper", confirm_live=True, kill_switch_path=str(kill_file))
    with pytest.raises(RuntimeError):
        manager.enforce()


