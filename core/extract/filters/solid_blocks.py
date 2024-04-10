import torch as th
from core.extract.filters import Filter


def solid_block_filter(
    tensor_blocks: th.Tensor[th.uint8],  # (512, 256, 512)
    mask: th.Tensor[th.bool] | None, # unnecessary
) -> th.Tensor[th.bool]:
    # only get solid blocks
    return tensor_blocks > 0
