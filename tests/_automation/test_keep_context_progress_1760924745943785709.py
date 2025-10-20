import pytest
pytestmark = pytest.mark.skip(reason="automation context keep-alive")
def test_import():
    from src._automation import keep_context_progress_1760924745943785709 as m
    assert hasattr(m, "STAMP")
