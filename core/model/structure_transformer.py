import torch as th
import torch.nn as nn
from core.model.util import generate_binary_mapping
from typing import Self


class StructureTransformer(nn.Module):
    """
    Autoregressive transformer for structure prediction.
    """

    def __init__(
        self: Self,
        sample_size: tuple[int, int, int] = (16, 16, 16),
        tube_length: int = 8,
        num_blocks: int = 6,
        d_model: int = 512,
        n_head: int = 8,
        num_special_tokens: int = 3,
    ) -> None:
        super().__init__()

        assert len(sample_size) == 3, f"sample_size must be of length 3"

        self.sample_size = sample_size
        self.tube_length = tube_length
        self.total_sample_size = sample_size[0] * sample_size[1] * sample_size[2]
        self.tubes_per_sample = self.total_sample_size // tube_length
        self.num_blocks = num_blocks
        self.d_model = d_model

        assert self.total_sample_size % tube_length == 0, f"sample_size must be divisible by tube_length"

        self.num_tube_types = 1 << tube_length
        self.num_token_types = self.num_tube_types + num_special_tokens
        # note: the below formulation only works for tubes of length up to 32
        self.tube_to_idx = th.Tensor([1 << i for i in range(tube_length)]).int()  # (tube_length,)
        self.idx_to_tube = generate_binary_mapping(self.num_tube_types, self.tube_length)  # (num_tube_types, tube_length)

        self.tube_in_embedding = nn.Embedding(self.num_token_types, d_model)
        self.tube_out_embedding = nn.Linear(d_model, self.num_tube_types)

        # TODO: probably use a custom model based on llama or something that can accept an absolute position and uses sink tokens
        self.transformer = None

    def forward(
        self: Self,
        x: th.Tensor,
        y_indices: th.Tensor,
    ) -> th.Tensor:
        original_shape = x.shape

        x = x.view(-1, self.tubes_per_sample, self.tube_length)  # (B, Y, Z, X) -> (B, Y*Z*X/tube_length, tube_length)
        # let T = number of tubes (X*Y*Z/tube_length)

        x = x @ self.tube_to_idx  # (B, T, tube_length) -> (B, T)  [index of token type]
        x = x % self.num_token_types  # convert -1, -2, etc. to num_token_types-1, num_token_types-2, etc.
        x = self.tube_in_embedding(x)  # (B, T) -> (B, T, d_model)

        x = self.transformer(x, y_indices)  # (B, T, d_model) -> (B, T, d_model)

        x = self.tube_out_embedding(x)  # (B, T, d_model) -> (B, T, num_tube_types)
        # turn distribution of tube types into distribution of solid/air blocks
        x = x @ self.idx_to_tube  # (B, T, num_tube_types) -> (B, T, tube_length)

        return x.view(original_shape)  # (B, T, tube_length) -> (B, Y, Z, X)
