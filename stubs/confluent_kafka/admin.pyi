from typing import Any, Mapping, Sequence

class AdminClient:
    def __init__(self, config: Mapping[str, Any]) -> None: ...
    def list_topics(self, timeout: float | None = ...) -> Any: ...
    def create_topics(
        self, new_topics: Sequence[Any], request_timeout: float | None = ...
    ) -> Mapping[str, Any] | Sequence[Any]: ...

class NewTopic:
    def __init__(
        self,
        topic: str,
        num_partitions: int,
        replication_factor: int,
        config: Mapping[str, Any] | None = ...,
    ) -> None: ...
