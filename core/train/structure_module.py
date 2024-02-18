import os
import lightning as L
import torch as th
from typing import Self
from core.model.structure_transformer import StructureTransformer


class StructureModule(L.LightningModule):
    """
    Module for structure prediction.
    """

    def __init__(
        self: Self,
        sample_size: tuple[int, int, int] = (16, 16, 16),
        tube_length: int = 8,
        num_blocks: int = 6,
        d_model: int = 512,
        n_head: int = 8,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        # Use autoregressive model to do sequeence modeling 
        # Break into 8 voxels at a time 
        
        self.sample_size = sample_size
        self.tube_length = tube_length
        self.total_sample_size = sample_size[0] * sample_size[1] * sample_size[2]
        self.tubes_per_sample = self.total_sample_size // tube_length

        assert sample_size[-1] % tube_length == 0, f"sample_size[-1] must be divisible by tube_length"

        self.model = StructureTransformer(
            sample_size=sample_size,
            tube_length=tube_length,
            num_blocks=num_blocks,
            d_model=d_model,
            n_head=n_head,
        )
        self.lr = lr
        self.loss_fn = th.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        target = batch > 0  # solid structure
        target = target.view(-1, self.tubes_per_sample, self.tube_length)  # (B, X, Y, Z) -> (B, X*Y*Z/tube_length, tube_length)
        # Shift target back by 1 to get input: (A, B, C) -> (<start>, A, B)
        # Start token at the start of every sequence (absolute positional embedding will give us sufficient information about y level)
        inputs = th.zeros_like(target)
        inputs[:, 1:] = target[:, :-1]
        inputs[:, 0] = -1  # start token

        output = self(inputs)
        loss = self.loss_fn(output, target)
        return loss

    def configure_optimizers(self):
        return L.optim.Adam(self.model.parameters(), lr=1e-3)
