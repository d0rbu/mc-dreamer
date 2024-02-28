import math
import yaml
import lightning as L
import torch as th
import torch.nn as nn
from modular_diffusion.diffusion.discrete import BitDiffusion
from modular_diffusion.module.utils.misc import default, exists, enlarge_as
from einops import reduce
from random import random
from core.model.unet import Unet3D
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
        self.register_buffer("positional_embeddings", positional_embeddings)

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
    
    NET_DEFAULTS = {
        "num_bits": 8,
        "img_size": (16, 16, 16),
        "data_key": "sample",
        "ctrl_key": "control",
    }
    OPT_DEFAULTS = {
        "lr": 1e-3,
        "weight_decay": 1e-3,
        "warmup_steps": 10000,
    }
    LR_DEFAULTS = {
        "warmup_steps": 10000,
        "restart_interval": 10000,
        "lr_decay": 0.9,
        "min_lr": 1e-5,
        "plateau_patience": 10000,
        "plateau_factor": 0.5,
        "plateau_metric": "val_loss",
        "plateau_mode": "min",
    }

    @classmethod
    def from_conf(
        cls: type[Self],
        path: str,
        **kwargs,
    ) -> Self:
        '''
            Initialize the Diffusion model from a YAML configuration file.
        '''

        with open(path, "r") as f:
            conf = yaml.safe_load(f)

        # Store the configuration file
        cls.conf = conf
        opt_conf = cls.OPT_DEFAULTS.copy()
        opt_conf.update(conf["OPTIMIZER"])
        lr_conf = cls.LR_DEFAULTS.copy()
        lr_conf.update(conf["LR"])

        cls.conf["OPTIMIZER"] = opt_conf
        cls.conf["LR"] = lr_conf

        net_par = cls.NET_DEFAULTS.copy()
        net_par.update(conf["MODEL"])
        dif_par = conf["DIFFUSION"]

        # Grab the batch size for precise metric logging
        cls.batch_size = conf["DATASET"]["batch_size"]

        # Initialize the network
        net = Unet3D(**net_par)

        return cls(net, ctrl_dim = net_par["ctrl_dim"], **kwargs, **dif_par)

    def __init__(
        self: Self,
        ctrl_dim: int,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.ctrl_emb = ControlEmbedder(ctrl_dim)

    @override
    def criterion(self) -> Callable:
        return th.nn.CrossEntropyLoss()  # CE for classification instead of regression

    def compute_loss(
        self: Self,
        x_0 : th.Tensor,
        ctrl : Optional[th.Tensor] = None,
        use_x_c : Optional[bool] = None,     
        norm_fn : Optional[Callable] = None,
    ) -> th.Tensor:
        use_x_c = default(use_x_c, self.self_cond)
        norm_fn = default(norm_fn, self.norm_forward)

        # Encode the condition using the sequence encoder
        ctrl = self.ctrl_emb(ctrl) if exists(ctrl) else ctrl

        # Extract the structure from the control
        structures = ctrl["structure"]

        # Normalize input images
        x_0 = norm_fn(x_0)

        bs, *_ = x_0.shape

        # Get the noise and scaling schedules
        sig = self.get_noise(bs)

        # NOTE: What to do with the scaling if present?
        # scales = self.get_scaling()

        eps = th.randn_like(x_0)
        x_t = x_0 + enlarge_as(sig, x_0) * eps # NOTE: Need to consider scaling here!

        x_c = None

        # Use self-conditioning with 50% dropout
        if use_x_c and random() < 0.5:
            with th.no_grad():
                x_c = self.predict(x_t, sig, ctrl = ctrl)
                x_c.detach_()

        x_p = self.predict(x_t, sig, x_c = x_c, ctrl = ctrl)

        # Compute the distribution loss ONLY for blocks that are solid in the structures
        solid_masks = [structure > 0 for structure in structures]

        loss = self.criterion(x_p, x_0, reduction = "none")
        device = loss.device

        # Compute the mean loss of each batch according to solid blocks
        loss = th.tensor([loss[i][mask].mean() for i, mask in enumerate(solid_masks)], device = device)

        # Add loss weight
        loss *= self.loss_weight(sig)
        return loss.mean()

    @override
    def configure_optimizers(self) -> tuple[list]:
        optim_conf = self.conf["OPTIMIZER"]

        optimizer = super().configure_optimizers()
        scheduler = th.optim.lr_scheduler.SequentialLR(optimizer, [
            th.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.0, total_iters=optim_conf["warmup_steps"]),
            th.optim.lr_scheduler.ChainedScheduler([
                th.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=optim_conf["restart_interval"], T_mult=2, eta_min=optim_conf["min_lr"]),
                th.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: optim_conf["lr_decay"] ** (math.floor(math.log2(step/self.restart_interval + 1)))),
                th.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=optim_conf["plateau_mode"], factor=optim_conf["plateau_factor"], patience=optim_conf["plateau_patience"], min_lr=optim_conf["min_lr"]),
            ])
        ], milestones=[optim_conf["warmup_steps"]])

        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
            "monitor": optim_conf["plateau_metric"],
        }

        return [optimizer], [scheduler_config]
