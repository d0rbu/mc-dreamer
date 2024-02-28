import math
import yaml
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
    
    DEFAULTS = {
    }

    @classmethod
    def from_conf(
        cls: type[Self],
        path: str,
        **kwargs,
    ) -> Self:
        with open(path, "r") as f:
            conf = yaml.safe_load(f)

        # Store the configuration file
        cls.conf = conf

        net_par = cls.DEFAULTS.copy()
        net_par.update(conf["MODEL"])
        net_par = SinkFormerConfig(**net_par)
        lr_par = conf["LR"]

        # Grab the batch size for precise metric logging
        cls.batch_size = conf["DATASET"]["batch_size"]

        return cls(config = net_par, **lr_par, **kwargs)

    def __init__(
        self: Self,
        sample_size: tuple[int, int, int] = (16, 16, 16),
        tube_length: int = 8,
        config: SinkFormerConfig = SinkFormerConfig(),
        lr: float = 1e-3,
        wd: float = 1e-3,
        warmup_steps: int = 10000,
        restart_interval: int = 10000,
        lr_decay: float = 0.9,
        min_lr: float = 1e-5,
        plateau_patience: int = 10000,
    ) -> None:
        super().__init__()
        # use autoregressive model to do sequence modelling on the structure
        # break into tubes of voxels (does not necessarily span the entire sample length). tube_length = 1 for voxel-level prediction

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
        self.wd = wd
        self.warmup_steps = warmup_steps
        self.restart_interval = restart_interval
        self.lr_decay = lr_decay
        self.min_lr = min_lr
        self.plateau_patience = plateau_patience
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
        structure, y_indices, prev_tube, next_tube = batch
        batch_size = structure.shape[0]

        if next_tube is None:
            next_tube = self.special_token_tubes["EOS"]
        else:
            next_tube = next_tube.expand(batch_size, 1, self.tube_length)  # (tube_length,) -> (B, 1, tube_length)

        target = th.cat((structure.view(batch_size, self.tubes_per_sample, self.tube_length), next_tube), dim=1)  # (B, Y, Z, X) -> (B, T + 1, tube_length)

        del structure, batch, next_tube  # free up memory
        th.cuda.empty_cache()

        # Shift target back by 1 to get input: (B, C, D, E) -> (A, B, C, D)
        inputs = th.roll(target, shifts=1, dims=1)

        if prev_tube is None:
            inputs[:, 0] = self.special_token_tubes["BOS"]
        else:
            inputs[:, 0] = prev_tube.view(-1, self.tube_length)

        del prev_tube
        th.cuda.empty_cache()

        output = self(inputs, y_indices)  # (B, T + 1, tube_length) -> (B, T + 1, tube_length)
        loss = self.loss_fn(output, target)

        return loss

    def configure_optimizers(self):
        optimizer = th.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        scheduler = th.optim.lr_scheduler.SequentialLR(optimizer, [
            th.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.0, total_iters=self.warmup_steps),
            th.optim.lr_scheduler.ChainedScheduler([
                th.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self.restart_interval, T_mult=2, eta_min=self.min_lr),
                th.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: self.lr_decay ** (math.floor(math.log2(step/self.restart_interval + 1)))),
                th.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=self.plateau_patience, min_lr=self.min_lr),
            ])
        ], milestones=[self.warmup_steps])

        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
            "monitor": "val_loss",
        }

        return [optimizer], [scheduler_config]
