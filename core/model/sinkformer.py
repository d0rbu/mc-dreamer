import torch as th
from typing import Self


# TODO: subclass LlamaFlashAttention2 to use sink tokens and use passed absolute position


class SinkFormer(th.nn.Module):
    def __init__(self: Self) -> None:
        super().__init__()

    def forward(self: Self, x: th.Tensor, start_pos_indices: th.Tensor | None = None) -> th.Tensor:
        """
        Args:
            x: (B, T, d_model)
            start_pos_indices: (B, T) or None
        Returns:
            x: (B, T, d_model)
        """
        if start_pos_indices is None:
            start_pos_indices = th.zeros(x.shape[:2], dtype=th.int32, device=x.device)
