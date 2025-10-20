from structlog import contextvars


def test_get_contextvars_returns_copy() -> None:
    contextvars.bind_contextvars(user="alice")
    try:
        snapshot = contextvars.get_contextvars()
        snapshot["user"] = "bob"
        assert contextvars.get_contextvars() == {"user": "alice"}
    finally:
        contextvars.unbind_contextvars("user")
