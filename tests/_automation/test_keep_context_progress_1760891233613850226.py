import pytest
pytestmark = pytest.mark.skip(reason="automation context keep-alive")
def test_import():
    from src._automation import keep_context_progress_1760891233613850226 as m
    assert hasattr(m, "STAMP")
