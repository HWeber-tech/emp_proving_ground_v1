from typing import Iterable

class Optimizer:
    def __init__(self, params: Iterable[object], lr: float = ...) -> None: ...
    def zero_grad(self) -> None: ...
    def step(self) -> None: ...

class Adam(Optimizer):
    def __init__(
        self,
        params: Iterable[object],
        lr: float = ...,
        betas: tuple[float, float] = ...,
        eps: float = ...,
        weight_decay: float = ...,
        amsgrad: bool = ...,
    ) -> None: ...
