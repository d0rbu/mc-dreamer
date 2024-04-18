import torch as th
from typing import Callable, TypeVar


Filter = Callable[
    [
        th.ByteTensor,  # input tensor blocks
        th.BoolTensor | None  # input mask
    ],
    th.BoolTensor
]
