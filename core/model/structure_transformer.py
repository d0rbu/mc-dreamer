import torch as th
import torch.nn as nn
from typing import Self


class StructureTransformer(nn.Module):
    """
    Transformer for structure prediction.
    """

    def __init__(
        self: Self,
        sample_size: tuple[int, int, int] = (16, 16, 16),
        tube_length: int = 8,
        num_blocks: int = 6,
        d_model: int = 512,
        n_head: int = 8,
    ) -> None:
        super().__init__()

        assert len(sample_size) == 3, f"sample_size must be of length 3"

        self.sample_size = sample_size
        self.tube_length = tube_length
        self.total_sample_size = sample_size[0] * sample_size[1] * sample_size[2]
        self.tubes_per_sample = self.total_sample_size // tube_length
        self.num_blocks = num_blocks
        self.d_model = d_model

        assert sample_size[-1] % tube_length == 0, f"sample_size[-1] must be divisible by tube_length"

        self.num_tube_types = 2 ** tube_length
        self.tube_to_idx = th.Tensor([2 ** i for i in range(tube_length)]).long().unsqueeze(-1)  # (tube_length, 1)
        self.idx_to_tube = (th.arange(self.num_tube_types).unsqueeze(-1) >> th.arange(tube_length)) & 1  # (num_tube_types, tube_length)

        self.tube_in_embedding = nn.Embedding(self.num_tube_types, d_model)
        self.tube_out_embedding = nn.Linear(d_model, self.num_tube_types)

        self.transformer = None  # TODO: use a huggingface transformer or something (sequence modeling)
    
    def forward(
        self: Self,
        x: th.Tensor,
    ) -> th.Tensor:
        original_shape = x.shape

        x = x.view(-1, self.tubes_per_sample, self.tube_length)  # (B, X, Y, Z) -> (B, X*Y*Z/tube_length, tube_length)
        # let T = number of tubes (X*Y*Z/tube_length)

        x = x @ self.tube_to_idx  # (B, T, tube_length) -> (B, T, 1)  [index of tube type]
        x = self.tube_in_embedding(x.squeeze(-1))  # (B, T, 1) -> (B, T, d_model)

        x = self.transformer(x)  # (B, T, d_model) -> (B, T, d_model)

        x = self.tube_out_embedding(x)  # (B, T, d_model) -> (B, T, num_tube_types)
        # turn distribution of tube types into distribution of solid/air blocks
        x = x @ self.idx_to_tube  # (B, T, num_tube_types) -> (B, T, tube_length)

        return x.view(original_shape)  # (B, T, tube_length) -> (B, X, Y, Z)
