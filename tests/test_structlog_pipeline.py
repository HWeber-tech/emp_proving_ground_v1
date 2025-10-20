import logging

import structlog
from structlog.processors import JSONRenderer


def test_processor_chain_stops_after_renderer() -> None:
    structlog._reset_for_tests()
    calls: list[str] = []

    def first(logger: object, method_name: str, event_dict: dict[str, object]) -> dict[str, object]:
        calls.append("first")
        event_dict["first"] = True
        return event_dict

    def renderer(logger: object, method_name: str, event_dict: dict[str, object]) -> str:
        calls.append("renderer")
        return JSONRenderer()(logger, method_name, event_dict)

    def sentinel(logger: object, method_name: str, event_dict: dict[str, object]) -> dict[str, object]:
        calls.append("sentinel")
        return event_dict

    structlog.configure(processors=[first, renderer, sentinel])
    logger = structlog.get_logger("short-circuit")
    logger.logger.handlers = [logging.NullHandler()]
    try:
        logger.info("event")
        assert calls == ["first", "renderer"]
    finally:
        structlog._reset_for_tests()
