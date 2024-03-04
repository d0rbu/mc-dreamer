import math
import yaml
import lightning as L
import torch as th
from typing import Self
from transformers.modeling_outputs import CausalLMOutputWithPast
from core.model.sinkformer import SinkFormerConfig, CausalSinkFormer
from core.model.util import generate_binary_mapping


# stupid dumb pytorch!!! can't use a chainedscheduler in a sequentiallr!!! so i gotta fix this myself
class ChainedScheduler(th.optim.lr_scheduler.ChainedScheduler):
    @property
    def last_epoch(self):
        return self._schedulers[0].last_epoch
    
    @last_epoch.setter
    def last_epoch(self, value):
        for scheduler in self._schedulers:
            scheduler.last_epoch = value

    def step(self, metrics = None):
        for scheduler in self._schedulers:
            if isinstance(scheduler, th.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(metrics)
            else:
                scheduler.step()

        self._last_lr = [group['lr'] for group in self._schedulers[-1].optimizer.param_groups]


class StructureModule(L.LightningModule):
    """
    Module for autoregressive structure prediction.
    """
    
    DEFAULTS = {
    }
    NUM_SPECIAL_TOKENS = 3

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
        optim_par = conf["OPTIMIZER"]

        # Grab the batch size for precise metric logging
        cls.batch_size = conf["DATASET"]["batch_size"]

        kwargs.update(optim_par)

        return cls(config = net_par, **kwargs)
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
        plateau_metric: str = "val_loss",
        **kwargs: dict,
    ) -> None:
        super().__init__()
        # use autoregressive model to do sequence modelling on the structure
        # break into tubes of voxels (does not necessarily span the entire sample length). tube_length = 1 for voxel-level prediction

        self.sample_size = sample_size
        self.tube_length = tube_length
        self.num_tube_types = 1 << tube_length
        self.total_sample_size = sample_size[0] * sample_size[1] * sample_size[2]
        self.tubes_per_sample = self.total_sample_size // tube_length

        assert self.total_sample_size % self.tube_length == 0, f"sample_size must be divisible by tube_length"

        self.model = CausalSinkFormer(config)
        self.lr = lr
        self.wd = wd
        self.warmup_steps = warmup_steps
        self.restart_interval = restart_interval
        self.lr_decay = lr_decay
        self.min_lr = min_lr
        self.plateau_patience = plateau_patience
        self.plateau_metric = plateau_metric

        self._cpu = th.device("cpu")
        self._tube_to_idx = {
            self._cpu: th.tensor([1 << i for i in range(tube_length)], dtype=th.float32)  # (tube_length,)
        }
        self._idx_to_tube = {
            self._cpu: generate_binary_mapping(self.num_tube_types, self.tube_length)  # (num_tube_types, tube_length)
        }
    
    def tube_to_idx(
        self: Self,
        x: th.Tensor,
    ) -> th.Tensor:
        if (tube_to_idx := self._tube_to_idx.get(x.device, None)) is None:
            tube_to_idx = self._tube_to_idx[self._cpu].to(x.device)
            self._tube_to_idx[x.device] = tube_to_idx

        return (x.float() @ tube_to_idx).long() + self.NUM_SPECIAL_TOKENS
    
    def idx_to_tube(
        self: Self,
        x: th.Tensor,
    ) -> th.Tensor:
        return (x - self.NUM_SPECIAL_TOKENS) @ self._idx_to_tube

    def forward(
        self: Self,
        x: th.Tensor,
        y_indices: th.Tensor,
        **kwargs: dict[str, any],
    ) -> tuple | CausalLMOutputWithPast:
        return self.model(x, y_indices, **kwargs)
    
    def _tube_batch_to_sequence(self, structure: th.Tensor, prev_tube: th.Tensor, next_tube: th.Tensor) -> th.Tensor:
        batch_size = structure.shape[0]

        bos_tube_mask = prev_tube[..., 0] == -1
        num_bos_tubes = bos_tube_mask.sum()

        if num_bos_tubes < batch_size:
            print("found non starting sample!!")

        prev_token = self.tube_to_idx(prev_tube)  # (B, tube_length) -> (B, 1)  [index of token type]
        if num_bos_tubes > 0:
            prev_token[bos_tube_mask] = self.model.config.bos_token_id  # (B, tube_length,) -> (B, tube_length)


        eos_tube_mask = next_tube[..., 0] == -1
        num_eos_tubes = eos_tube_mask.sum()

        if num_eos_tubes < batch_size:
            print("found non ending sample!!")

        next_token = self.tube_to_idx(next_tube)  # (B, tube_length) -> (B, 1)  [index of token type]
        if num_eos_tubes > 0:
            next_token[eos_tube_mask] = self.model.config.eos_token_id  # (B, tube_length,) -> (B, tube_length)


        sequence = th.cat(
            (
                prev_token.unsqueeze(-1),
                self.tube_to_idx(structure.view(batch_size, self.tubes_per_sample, self.tube_length)).squeeze(),
                next_token.unsqueeze(-1),
            ),
            dim=1,
        )  # (B, Y, Z, X) -> (B, T + 2) [index of token type]

        del structure, next_tube, next_token, prev_tube, prev_token  # free up memory
        th.cuda.empty_cache()

        return sequence

    def training_step(self, batch, batch_idx):
        structure, y_indices, prev_tube, next_tube = batch
        batch_size = structure.shape[0]

        sequence = self._tube_batch_to_sequence(structure, prev_tube, next_tube)
        del structure, prev_tube, next_tube

        outputs = self(sequence, y_indices, labels=sequence)  # (B, T + 1) -> (B, T + 1, num_tokens)

        self.log("train_loss", outputs.loss.item(), on_step=True, batch_size=batch_size)

        return outputs.loss

    def validation_step(self, batch, batch_idx):
        structure, y_indices, prev_tube, next_tube = batch
        batch_size = structure.shape[0]

        sequence = self._tube_batch_to_sequence(structure, prev_tube, next_tube)

        outputs = self(sequence, y_indices, labels=sequence)  # (B, T + 1) -> (B, T + 1, num_tokens)
        predicted = th.argmax(outputs.logits, dim=-1)
        acc = (predicted[:, :-1] == sequence[:, 1:]).float().mean().item()

        self.log("val_loss", outputs.loss.item(), on_step=True, batch_size=batch_size)
        self.log("val_acc", acc, on_step=True, batch_size=batch_size)

    def configure_optimizers(self):
        optimizer = th.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)

        scheduler = th.optim.lr_scheduler.SequentialLR(optimizer, [
            th.optim.lr_scheduler.LinearLR(optimizer, start_factor=1/self.warmup_steps, total_iters=self.warmup_steps),
            ChainedScheduler([
                th.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self.restart_interval, T_mult=2, eta_min=self.min_lr),
                th.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: self.lr_decay ** (math.floor(math.log2(step/self.restart_interval + 1)))),
                th.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=self.plateau_patience, min_lr=self.min_lr),
            ])
        ], milestones=[self.warmup_steps])

        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
            "monitor": self.plateau_metric,
        }

        return [optimizer], [scheduler_config]
