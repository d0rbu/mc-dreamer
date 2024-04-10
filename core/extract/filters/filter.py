import torch as th
from typing import Callable


Filter = Callable[
    [
        th.Tensor[th.uint8],  # input tensor blocks
        th.Tensor[th.bool] | None  # input mask
    ],
    th.Tensor[th.bool]
]
