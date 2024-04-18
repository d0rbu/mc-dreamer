import torch as th


def solid_block_filter(
    tensor_blocks: th.ByteTensor,  # (512, 256, 512)
    mask: th.BoolTensor | None,  # unnecessary
) -> th.BoolTensor:
    # only get solid blocks
    return tensor_blocks > 0
