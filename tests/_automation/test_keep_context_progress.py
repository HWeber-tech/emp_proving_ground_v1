import pytest
pytestmark = pytest.mark.skip(reason="automation context keep-alive")
def test_import():
    import src._automation.keep_context_progress as m
    assert hasattr(m, "VERSION")
