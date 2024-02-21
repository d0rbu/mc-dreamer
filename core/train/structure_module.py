import os
import lightning as L
import torch as th
from typing import Self
from core.model.sinkformer import SinkFormerConfig
from core.model.structure_transformer import StructureTransformer
from core.model.util import generate_binary_mapping


class StructureModule(L.LightningModule):
    """
    Module for autoregressive structure prediction.
    """

    def __init__(
        self: Self,
        sample_size: tuple[int, int, int] = (16, 16, 16),
        tube_length: int = 8,
        config: SinkFormerConfig = SinkFormerConfig(),
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        # use autoregressive model to do sequence modelling on the structure
        # break into tubes of voxels (does not necessarily span the entire sample). tube_length = 1 for voxel-level prediction

        self.sample_size = sample_size
        self.tube_length = tube_length
        self.total_sample_size = sample_size[0] * sample_size[1] * sample_size[2]
        self.tubes_per_sample = self.total_sample_size // tube_length
        self.special_tokens = ["PAD", "BOS", "EOS"]  # we can have up to 2 ^ tube_length - 1 special tokens due to the way we encode them as tubes
        self.special_token_tubes = self._generate_special_token_tubes(self.special_tokens)
        self.num_special_tokens = len(self.special_tokens)

        assert self.total_sample_size % self.tube_length == 0, f"sample_size must be divisible by tube_length"

        self.model = StructureTransformer(
            sample_size=self.sample_size,
            tube_length=self.tube_length,
            config=config,
        )
        self.lr = lr
        self.loss_fn = th.nn.CrossEntropyLoss()

    def _generate_special_token_tubes(
        self: Self,
        special_tokens: list[str],
    ) -> dict[str, th.Tensor]:
        special_token_tubes = -generate_binary_mapping(len(special_tokens) + 1, self.tube_length)[1:].int()
        return {token: tube for token, tube in zip(special_tokens, special_token_tubes)}

    def forward(
        self: Self,
        x: th.Tensor,
        y_indices: th.Tensor,
    ) -> th.Tensor:
        return self.model(x, y_indices)

    def training_step(self, batch, batch_idx):
        target, y_indices = batch["sample"], batch["y_index"]
        target = target > 0  # solid structure
        target = target.view(-1, self.tubes_per_sample, self.tube_length)  # (B, Y, Z, X) -> (B, T, tube_length)
        # Shift target back by 1 to get input: (A, B, C) -> (<|BOS|>, A, B)
        # BOS token (encoded as a tube) at the start of every sequence
        inputs = th.roll(target, shifts=1, dims=1)
        inputs[:, 0, :] = self.special_token_tubes["BOS"]

        output = self(inputs, y_indices)  # (B, T, tube_length) -> (B, T, tube_length)
        loss = self.loss_fn(output, target)
        return loss

    def configure_optimizers(self):
        return L.optim.Adam(self.model.parameters(), lr=1e-3)
