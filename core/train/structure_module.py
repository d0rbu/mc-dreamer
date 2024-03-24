import math
import yaml
import lightning as L
import torch as th
import torch.nn.functional as F
from typing import Self, Any
from transformers.modeling_outputs import CausalLMOutputWithPast
from core.model.sinkformer import SinkFormerConfig, CausalSinkFormer
from core.model.util import generate_binary_mapping
from core.scheduler import SequentialLR, ChainedScheduler


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
        plateau_metric: str = "val_loss_epoch",
        val_check_interval: int = 1000,
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
        self.val_check_interval = val_check_interval

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
        if (idx_to_tube := self._idx_to_tube.get(x.device, None)) is None:
            idx_to_tube = self._idx_to_tube[self._cpu].to(x.device)
            self._idx_to_tube[x.device] = idx_to_tube

        tubes = th.empty((*x.shape, self.tube_length), dtype=th.long, device=x.device)
        special_tokens_mask = x < self.NUM_SPECIAL_TOKENS

        tubes[special_tokens_mask] = -1

        one_hot_indices = F.one_hot((x[~special_tokens_mask] - self.NUM_SPECIAL_TOKENS), self.model.config.vocab_size - self.model.config.num_special_tokens)
        tubes[~special_tokens_mask] = (one_hot_indices @ idx_to_tube)
        
        return tubes

    def forward(
        self: Self,
        x: th.Tensor,
        y_indices: th.Tensor,
        **kwargs: dict[str, Any],
    ) -> tuple | CausalLMOutputWithPast:
        return self.model(x, y_indices, **kwargs)

    def _tube_batch_to_sequence(self, structure: th.Tensor, prev_tube: th.Tensor | None = None, next_tube: th.Tensor | None = None) -> th.Tensor:
        batch_size = structure.shape[0]

        main_sequence = [self.tube_to_idx(structure.view(batch_size, -1, self.tube_length))]
        if len(main_sequence[0].shape) != 2:
            main_sequence[0] = main_sequence[0].squeeze(-1)

        if prev_tube is None:
            prev_token = None
        else:
            bos_tube_mask = prev_tube[..., 0] == -1
            num_bos_tubes = bos_tube_mask.sum()

            if num_bos_tubes < batch_size:
                print("found non starting sample!!")

            prev_token = self.tube_to_idx(prev_tube)  # (B, tube_length) -> (B, 1)  [index of token type]
            if num_bos_tubes > 0:
                prev_token[bos_tube_mask] = self.model.config.bos_token_id  # (B, tube_length,) -> (B, tube_length)

            main_sequence.insert(0, prev_token.unsqueeze(-1))

        if next_tube is None:
            next_token = None
        else:
            eos_tube_mask = next_tube[..., 0] == -1
            num_eos_tubes = eos_tube_mask.sum()

            if num_eos_tubes < batch_size:
                print("found non ending sample!!")

            next_token = self.tube_to_idx(next_tube)  # (B, tube_length) -> (B, 1)  [index of token type]
            if num_eos_tubes > 0:
                next_token[eos_tube_mask] = self.model.config.eos_token_id  # (B, tube_length,) -> (B, tube_length)

            main_sequence.append(next_token.unsqueeze(-1))

        sequence = th.cat(
            main_sequence,
            dim=1,
        )  # (B, Y, Z, X) -> (B, T (+2)) [index of token type]

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

    def lr_scheduler_step(self, scheduler: th.optim.lr_scheduler.LRScheduler, metric: Any | None) -> None:
        scheduler.step(epoch=self.global_step, metrics=metric)

    def configure_optimizers(self):
        optimizer = th.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)

        scheduler = SequentialLR(optimizer, [
            th.optim.lr_scheduler.LinearLR(optimizer, start_factor=1/self.warmup_steps, total_iters=self.warmup_steps),
            ChainedScheduler([
                th.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self.restart_interval, T_mult=2, eta_min=self.min_lr),
                th.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda step: self.lr_decay ** (math.floor(math.log2(step/self.restart_interval + 1)))),
                th.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=self.plateau_patience, min_lr=self.min_lr),
            ])
        ], milestones=[self.warmup_steps])

        scheduler = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
            "monitor": self.plateau_metric,
        }

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }
