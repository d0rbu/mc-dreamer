import lightning as L
import torch as th
from typing import Self

class ColorModule(L.LightningModule):
    """
    Module for color prediction.
    """

    def __init__(
        self: Self,
        sample_size: tuple[int, int, int] = (16, 16, 16),
        num_solid_block_types: int = 255,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()

        self.sample_size = sample_size
        self.total_sample_size = sample_size[0] * sample_size[1] * sample_size[2]
        self.num_solid_block_types = num_solid_block_types
        self.lr = lr

    def forward(self, x):
        return x
    
    def training_step(self, batch, batch_idx):
        samples = batch
        samples = samples.view(-1, self.total_sample_size)  # (B, X, Y, Z) -> (B, X*Y*Z)
        condition_structure = samples > 0  # solid structure

        output = thz(inputs, target)
        loss = th.nn.functional.nll_loss(output, target.view(-1))
        return loss

    def configure_optimizers(self):
        return th.optim.SGD(self.model.parameters(), lr=0.1)