import math
import lightning as L
import torch as th
import torch.nn as nn
from modular_diffusion.diffusion.discrete import BitDiffusion
from typing import Callable, Optional, override, Self


# vaswani et al 2017
class PositionalEmbeddings(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = th.arange(max_len).unsqueeze(1)
        div_term = th.exp(th.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        positional_embeddings = th.zeros(max_len, 1, d_model)
        positional_embeddings[:, 0, 0::2] = th.sin(position * div_term)
        positional_embeddings[:, 0, 1::2] = th.cos(position * div_term)
        self.register_buffer('positional_embeddings', positional_embeddings)

    def forward(self, positions: th.Tensor):
        """
        Arguments:
            positions: LongTensor, shape ``[batch_size]``
            num_embeddings: int, number of embeddings to return
        """
        return self.positional_embeddings[positions]


class PositionEmbedder(nn.Module):
    def __init__(
        self: Self,
        embedding_dim: int,
        pos_embedding_dim: int | None = None,
    ) -> None:
        super().__init__()

        if pos_embedding_dim is None:
            pos_embedding_dim = embedding_dim

        self.pos_embedding = PositionalEmbeddings(pos_embedding_dim)
        self.projection = nn.Linear(pos_embedding_dim, embedding_dim, bias=False)
        self.activation = nn.GELU()

    def forward(
        self: Self,
        positions: th.LongTensor
    ) -> th.Tensor:
        return self.activation(self.projection(self.pos_embedding(positions)))


class StructureEmbedder(nn.Module):
    def __init__(
        self: Self,
        embedding_dim: int,
        sample_size: tuple[int, int, int] = (16, 16, 16),
        projection_ratio: float = 4,
    ) -> None:
        super().__init__()

        self.input_dim = sample_size[0] * sample_size[1] * sample_size[2]
        self.hidden_dim = int(self.embedding_dim * projection_ratio)

        self.pre_norm = nn.LayerNorm(self.input_dim)
        self.up_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.gate_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.activation = nn.GELU()
        self.down_proj = nn.Linear(self.hidden_dim, embedding_dim)
        self.post_norm = nn.LayerNorm(embedding_dim)

    def forward(
        self: Self,
        structure: th.Tensor,
    ) -> th.Tensor:
        structure = structure.view(structure.shape[0], -1)
        structure = self.pre_norm(structure)

        return self.norm(self.down_proj(self.activation(self.gate_proj(structure)) * self.up_proj(structure)))


class ControlEmbedder(nn.Module):
    def __init__(
        self: Self,
        embedding_dim: int,
        sample_size: tuple[int, int, int] = (16, 16, 16),
        pos_embedding_dim: int | None = None
    ) -> None:
        super().__init__()

        self.pos_embedder = PositionEmbedder(embedding_dim, pos_embedding_dim)
        self.structure_embedder = StructureEmbedder(embedding_dim, sample_size)

    def forward(
        self: Self,
        control_inputs: dict[str, th.Tensor],
    ) -> th.Tensor:
        structure, y_indices = control_inputs["structure"], control_inputs["y_index"]

        return self.pos_embedder(y_indices) + self.structure_embedder(structure)


class ColorModule(BitDiffusion):
    """
    Module for color prediction.
    """

    def __init__(
        self: Self,
        sample_size: int = (16, 16, 16),
        config = None,  # TODO: use hf config for whatever model we use
        **kwargs,
    ) -> None:
        model = None  # TODO: use a huggingface conditional unet or something compatible with modular-diffusion

        super().__init__(model, num_bits=8, img_size=sample_size, data_key="sample", ctrl_key="control", **kwargs)

        self.ctrl_emb = ControlEmbedder(config.hidden_size)

    @override
    def criterion(self) -> Callable:
        return th.nn.CrossEntropyLoss()  # CE for classification instead of regression

    # TODO: override loss function to apply criterion over structure instead of the whole sample
