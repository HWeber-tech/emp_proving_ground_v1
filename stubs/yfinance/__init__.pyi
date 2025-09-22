from typing import Sequence

class Ticker:
    def __init__(self, symbol: str) -> None: ...
    def history(
        self,
        period: str | None = ...,
        start: str | None = ...,
        end: str | None = ...,
        interval: str | None = ...,
        auto_adjust: bool | None = ...,
        actions: bool | None = ...,
        prepost: bool | None = ...,
        keepna: bool | None = ...,
        proxy: str | None = ...,
        rounding: bool | None = ...,
        timeout: float | None = ...,
    ) -> object: ...

def download(
    tickers: str | Sequence[str],
    start: str | None = ...,
    end: str | None = ...,
    interval: str = ...,
    progress: bool = ...,
) -> object: ...
