import torch as th
import torch.nn as nn
from core.model.util import generate_binary_mapping
from core.model.sinkformer import CausalSinkFormer, SinkFormerConfig
from typing import Self


class StructureTransformer(nn.Module):
    """
    Autoregressive transformer for structure prediction.
    """

    def __init__(
        self: Self,
        sample_size: tuple[int, int, int] = (16, 16, 16),
        tube_length: int = 8,
        config: SinkFormerConfig = SinkFormerConfig(),
    ) -> None:
        super().__init__()

        assert len(sample_size) == 3, f"sample_size must be of length 3"

        self.sample_size = sample_size
        self.tube_length = tube_length
        self.total_sample_size = sample_size[0] * sample_size[1] * sample_size[2]
        self.tubes_per_sample = self.total_sample_size // tube_length

        assert self.total_sample_size % tube_length == 0, f"sample_size must be divisible by tube_length"

        self.num_tube_types = 1 << tube_length
        self.num_token_types = self.num_tube_types + config.num_special_tokens

        # note: the below formulation only works for tubes of length up to 32
        tube_to_idx = th.Tensor([1 << i for i in range(tube_length)]).int()  # (tube_length,)
        idx_to_tube = generate_binary_mapping(self.num_tube_types, self.tube_length)  # (num_tube_types, tube_length)

        self.register_buffer("tube_to_idx", tube_to_idx)
        self.register_buffer("idx_to_tube", idx_to_tube)

        self.tube_in_embedding = nn.Embedding(self.num_token_types, config.hidden_size)
        self.tube_out_embedding = nn.Linear(config.hidden_size, self.num_token_types)

        self.model = CausalSinkFormer(config)

    def forward(
        self: Self,
        x: th.Tensor,
        y_indices: th.Tensor,
    ) -> th.Tensor:
        original_shape = x.shape

        # if we are converting from 3d structure to tube sequence
        if len(x.shape) == 4:
            x = x.view(-1, self.tubes_per_sample, self.tube_length)  # (B, Y, Z, X) -> (B, T, tube_length)

        x = x @ self.tube_to_idx  # (B, T, tube_length) -> (B, T)  [index of token type]

        special_tokens_mask = x < 0
        x[special_tokens_mask] = -x[special_tokens_mask]
        x[~special_tokens_mask] += self.model.config.num_special_tokens

        x = self.model(x, position_ids=y_indices)  # (B, T) -> (B, T, num_token_types)

        # turn distribution of tube types into distribution of solid/air blocks and special tokens
        x = th.cat([x[..., :self.num_tube_types] @ self.idx_to_tube, x[..., self.num_tube_types:]], dim = -1)  # (B, T, num_token_types) -> (B, T, tube_length + num_special_tokens)

        return x.view(original_shape)  # (B, T, tube_length) -> (B, Y, Z, X)
