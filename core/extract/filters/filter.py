import torch as th
from typing import Callable, Any


Filter = Callable[[th.Tensor[th.uint8], th.Tensor[th.bool] | None], th.Tensor[th.bool]]
