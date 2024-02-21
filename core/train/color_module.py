import math
import lightning as L
import torch as th
import torch.nn as nn
from modular_diffusion.diffusion.elucidated import ElucidatedDiffusion
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


class ControlEmbedder(nn.Module):
    def __init__(self, embedding_dim: int, pos_embedding_dim: int | None = None):
        super(ControlEmbedder, self).__init__()

        if pos_embedding_dim is None:
            pos_embedding_dim = embedding_dim

        self.pos_embedding = PositionalEmbeddings(pos_embedding_dim)
        self.projection = nn.Linear(pos_embedding_dim, embedding_dim, bias=False)
        self.activation = nn.GELU()

    def forward(
        self: Self,
        control_labels: th.LongTensor,
    ) -> th.Tensor:
        return self.activation(self.projection(self.pos_embedding(control_labels)))


class ColorModule(ElucidatedDiffusion):
    """
    Module for color prediction.
    """

    def __init__(
        self: Self,
        img_size: int = (16,16,16), # Now is a tuple
        # Killed Loss type (now for classification)
        config = None,  # TODO: use hf config for whatever model we use
        **kwargs,
    ) -> None:
        model = None  # TODO: use a huggingface conditional unet or something

        super().__init__(model, img_size, data_key="sample", ctrl_key="y_index", **kwargs)

        self.ctrl_emb = ControlEmbedder(config.hidden_size)

    @override
    def criterion(self) -> Callable:
        return th.nn.CrossEntropyLoss()  # CE for classification instead of regression
    
    # TODO: override loss function to apply criterion over structure instead of the whole sample
