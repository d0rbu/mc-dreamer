import math
import yaml
import wandb
import torch as th
import torch.nn as nn
from functools import partial
from tqdm import tqdm
from einops import rearrange, reduce
from modular_diffusion.diffusion.discrete import BitDiffusion
from modular_diffusion.diffusion.module.utils.misc import default, exists, enlarge_as
from random import random
from core.model.unet import Unet3D
from typing import Callable, Optional, Self, Any, Generator
from core.scheduler import SequentialLR, ChainedScheduler, step_scheduler
from ema_pytorch import EMA

from modular_diffusion.diffusion.module.utils.misc import exists, default, enlarge_as, groupwise

INV_SQRT2 = 1. / math.sqrt(2)


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
        projection_ratio: float = 4.,
        num_tokens: int = 1,
    ) -> None:
        super().__init__()

        self.input_dim = sample_size[0] * sample_size[1] * sample_size[2]
        self.hidden_dim = int(embedding_dim * projection_ratio)
        self.num_tokens = num_tokens

        self.pre_norm = nn.LayerNorm(self.input_dim)
        self.up_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.up_gate_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.mid_proj_0 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.mid_norm_0 = nn.LayerNorm(self.hidden_dim)
        self.mid_gate_proj_0 = nn.Linear(self.input_dim, self.hidden_dim)
        self.mid_proj_1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.mid_norm_1 = nn.LayerNorm(self.hidden_dim)
        self.mid_gate_proj_1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.activation = nn.GELU()
        self.down_proj = nn.Linear(self.hidden_dim, num_tokens * embedding_dim)
        self.post_norm = nn.LayerNorm(embedding_dim)

    def forward(
        self: Self,
        structure: th.Tensor,
    ) -> th.Tensor:
        structure = structure.view(structure.shape[0], -1)
        structure = self.pre_norm(structure)

        hidden = self.activation(self.up_gate_proj(structure)) * self.up_proj(structure)
        hidden = self.mid_norm_0(self.activation(self.mid_gate_proj_0(structure)) * self.mid_proj_0(hidden))
        hidden = self.mid_norm_1(self.activation(self.mid_gate_proj_1(structure)) * self.mid_proj_1(hidden))
        hidden = self.down_proj(hidden)

        return self.post_norm(hidden.view(structure.shape[0], self.num_tokens, -1))


class ControlEmbedder(nn.Module):
    def __init__(
        self: Self,
        embedding_dim: int,
        sample_size: tuple[int, int, int] = (16, 16, 16),
        pos_embedding_dim: int | None = None,
        projection_ratio: float = 4.,
        num_tokens: int = 1,
    ) -> None:
        super().__init__()

        self.pos_embedder = PositionEmbedder(embedding_dim, pos_embedding_dim)
        self.structure_embedder = StructureEmbedder(embedding_dim, sample_size, projection_ratio, num_tokens)

    def forward(
        self: Self,
        control_inputs: dict[str, th.Tensor],
    ) -> th.Tensor:
        structure, y_indices = control_inputs["structure"], control_inputs["y_index"]
        structure = structure.float() * 2 - 1  # from {0, 1} to {-1, 1}

        return self.pos_embedder(y_indices) + self.structure_embedder(structure)


class ColorModule(BitDiffusion):
    """
    Module for color prediction.
    """
    
    DIFFUSION_DEFAULTS = {
        "num_bits": 8,
        "img_size": (16, 16, 16),
        "data_key": "sample",
        "ctrl_key": "control",
    }
    OPT_DEFAULTS = {
        "lr": 1e-3,
        "weight_decay": 1e-3,
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
        """
            Initialize the Diffusion model from a YAML configuration file.
        """

        with open(path, "r") as f:
            conf = yaml.safe_load(f)

        # Store the configuration file
        cls.conf = conf
        opt_conf = cls.OPT_DEFAULTS.copy()
        opt_conf.update(conf["OPTIMIZER"])

        cls.conf["OPTIMIZER"] = opt_conf

        net_par = conf["MODEL"]
        dif_par = cls.DIFFUSION_DEFAULTS.copy()
        dif_par.update(conf["DIFFUSION"])
        kwargs.update(dif_par)

        # Grab the batch size for precise metric logging
        cls.batch_size = conf["DATASET"]["batch_size"]

        # Initialize the network
        net = Unet3D(**net_par)

        return cls(model=net, **kwargs)

    def __init__(
        self: Self,
        model: nn.Module,
        ctrl_dim: int,
        num_bits: int = 1.,
        bit_scale: float = 1.,
        sample_size: tuple[int, int, int] = (16, 16, 16),
        pos_embedding_dim: int | None = None,
        projection_ratio: float = 4.,
        num_ctrl_tokens: int = 4,
        do_ema: bool = True,
        ema_beta: float = 0.9999,
        ema_update_after_step: int = 2000,  # lag time for the ema to not include the initial garbage initializations
        ema_update_every: int = 100,  # how often to actually update ema model
        average_loss_by_block: bool = False,  # if true, the loss will be averaged across all blocks. if false, it will be averaged across batch samples
        watch_model: bool = False,  # whether to track model params and gradients
        **kwargs,
    ) -> None:
        super().__init__(model=model, num_bits=num_bits, bit_scale=bit_scale, **kwargs)

        self.ctrl_emb = ControlEmbedder(ctrl_dim, sample_size, pos_embedding_dim, projection_ratio, num_ctrl_tokens)
        self.norm_backward = partial(self.bit2int, nbits=num_bits, scale=bit_scale)
        self.do_ema = do_ema
        self.average_loss_by_block = average_loss_by_block
        self.watch_model = watch_model

        if watch_model:
            wandb.init()
            wandb.watch(self.model, log="all", log_freq=500, log_graph=False)
            wandb.watch(self.ctrl_emb, log="all", log_freq=500, log_graph=False)

        if self.do_ema:
            self.ema = EMA(
                model,
                beta=ema_beta,
                update_after_step=ema_update_after_step,
                update_every=ema_update_every,
                include_online_model=False,
            )
            self.ctrl_emb_ema = EMA(
                self.ctrl_emb,
                beta=ema_beta,
                update_after_step=ema_update_after_step,
                update_every=ema_update_every,
                include_online_model=False,
            )

    # * Lightning Module functions
    def training_step(self, batch : dict[str, th.Tensor], batch_idx : int) -> th.Tensor:
        # Extract the starting images from data batch
        x_0  = batch[self.data_key]
        ctrl = batch[self.ctrl_key] if exists(self.ctrl_key) else None

        loss = self.compute_loss(x_0, ctrl = ctrl)

        self.log_dict({'train_loss' : loss}, logger = True, on_step = True, sync_dist = True)

        if self.watch_model:
            self.log_dict({"train_batch_total_block_count" : batch["block_count"].sum().float()}, logger = True, on_step = True, sync_dist = True)

        return loss

    def validation_step(self, batch : dict[str, th.Tensor], batch_idx : int) -> th.Tensor:
        # Extract the starting images from data batch
        x_0  = batch[self.data_key]
        ctrl = batch[self.ctrl_key] if exists(self.ctrl_key) else None

        loss = self.compute_loss(x_0, ctrl=ctrl, use_ema=self.do_ema)

        self.log_dict({"val_loss" : loss}, logger=True, on_step=True, sync_dist=True)

        if self.watch_model:
            self.log_dict({"val_batch_total_block_count" : batch["block_count"].sum().float()}, logger = True, on_step = True, sync_dist = True)

        self.val_outs = ((x_0, ctrl),)

        return self.val_outs

    @th.no_grad()
    def on_validation_epoch_end(self) -> None:
        """
            At the end of the validation cycle, we inspect how the denoising
            procedure is doing by sampling novel images from the learn distribution.
        """
        pass

        # val_outs: tuple[th.Tensor, ...] = self.val_outs

        # # Collect the input shapes
        # (x_0, ctrl), *_ = val_outs

        # # Produce 8 samples and log them
        # imgs = self(
        #         num_imgs = 8,
        #         ctrl = ctrl,
        #         verbose = False,
        #     )
        
        # assert not torch.isnan(imgs).any(), "NaNs detected in imgs!"

        # imgs = make_grid(imgs, nrow = 4)

        # # Log images using the default TensorBoard logger
        # self.logger.experiment.add_image(self.log_img_key, imgs, global_step = self.global_step)
    
    @th.no_grad()
    def forward(
        self,
        num_imgs : int = 4,
        num_steps : Optional[int] = None,
        ode_solver : Optional[str] = None,
        norm_undo : Optional[Callable] = None,
        ctrl : Optional[th.Tensor] = None,
        use_x_c : Optional[bool] = None,
        guide : float = 1.,
        use_ema : bool = False,
        **kwargs,
    ) -> Generator[th.Tensor, None, None]:
        '''
            Sample images using a given sampler (ODE Solver)
            from the trained model. 
        '''

        use_x_c = default(use_x_c, self.self_cond)
        num_steps = default(num_steps, self.sample_steps)
        norm_undo = default(norm_undo, self.norm_backward)
        self.ode_solver = default(ode_solver, self.ode_solver)

        timestep = self.get_timesteps(num_steps)
        schedule = self.get_schedule(timestep)
        scaling  = self.get_scaling (timestep)

        # schedule = repeat(schedule, '... -> b ...', b = num_imgs)
        # scaling  = repeat(scaling , '... -> b ...', b = num_imgs)

        # Encode the condition using the sequence encoder
        ctrl = (self.ctrl_emb_ema(ctrl) if use_ema else self.ctrl_emb(ctrl))[:num_imgs] if exists(ctrl) else ctrl

        shape = (num_imgs, self.model.channels, *self.img_size)

        for step_output in self.sampler(shape, schedule, scaling, ctrl=ctrl, use_x_c=use_x_c, guide=guide, **kwargs):
            yield norm_undo(step_output)

    @th.no_grad()
    def heun_sde_inpaint(
        self: Self,
        shape: tuple[int,...],
        schedule: th.Tensor,
        scaling: th.Tensor,
        ctrl: Optional[th.Tensor] = None,
        context: Optional[th.Tensor] = None,
        mask: Optional[th.BoolTensor] = None,  # mask of area to inpaint
        inpaint_strength: float = 1.,
        use_ema: bool = False,
        structure: Optional[th.LongTensor] = None,
        norm_fn: Optional[Callable] = None,
        use_x_c: Optional[bool] = None,
        clamp: bool = False,
        guide: float = 1.,
        s_tmin: float = 0.05,
        s_tmax: float = 50.,
        s_churn: float = 80,
        s_noise: float = 1.003,
        verbose: bool = False
    ) -> Generator[th.Tensor, None, None]:
        """
            based on diffusion.py heun sde implementation

            Stochastic Heun (2° order) solver from:
            https://arxiv.org/pdf/2206.00364 (Algorithm 2)
        """
        assert (context is None) == (mask is None), "Context and mask must both be provided or both be None"
        inpaint = context is not None

        use_x_c = default(use_x_c, self.self_cond)
        norm_fn = default(norm_fn, self.norm_forward)

        if inpaint:
            assert 0 <= inpaint_strength <= 1, "Inpaint strength must be in [0, 1]"
            mask = ~mask  # invert mask so it is a context mask instead
            structure = context > 0
            context = norm_fn(context)  # normalize context
            structure = structure.expand_as(context)
            mask = mask.expand_as(context)
            mask = mask & structure

        T = schedule.shape[0]

        # compute the gamma coefficients that increase the noise level
        gammas = th.where(
            (schedule < s_tmin) | (schedule > s_tmax),
            0., min(s_churn / T, INV_SQRT2)
        )

        x_c = None  # self-conditioning parameter
        if not inpaint:
            x_t = schedule[0] * th.randn(shape, device = schedule.device)
        if inpaint:
            inpaint_schedule = [context]

            for current_sigma, last_sigma in reversed(list(zip(schedule[:-1], schedule[1:]))):
                noise = th.randn(shape, device = schedule.device)
                sigma_delta = (current_sigma ** 2 - last_sigma ** 2) ** 0.5

                latest_images = inpaint_schedule[-1]

                noised_images = latest_images + sigma_delta * noise
                inpaint_schedule.append(noised_images)

            inpaint_schedule = inpaint_schedule[::-1]
            x_t = inpaint_schedule[0]

        pars = zip(groupwise(schedule, n = 2), gammas)
        for i, ((sig, sigp1), gamma) in tqdm(enumerate(pars), total = T, desc = "Stochastic Heun", disable = not verbose):
            # Patch in inpainting context if needed
            if inpaint and i > 0:
                x_t[mask] = inpaint_schedule[i][mask]
                x_t[~mask] = x_t[~mask] * inpaint_strength + inpaint_schedule[i][~mask] * (1 - inpaint_strength)

            # Sample additive noise
            eps = s_noise * th.randn_like(x_t)

            # Select temporarily increased noise level sig_hat
            sig_hat = sig * (1 + gamma)

            # Add new noise to move schedule to time "hat"
            x_hat = x_t + math.sqrt(sig_hat ** 2 - sig ** 2) * eps

            # Evaluate dx/dt at scheduled time "hat"
            p_hat = self.follow(x_hat, sig_hat, x_c = x_c if use_x_c else None, ctrl = ctrl, guide = guide, clamp = clamp, use_ema = use_ema)
            dx_dt = (x_hat - p_hat) / sig_hat

            # Take Euler step from schedule time "hat" to time + 1
            x_t = x_hat + (sigp1 - sig_hat) * dx_dt

            # Add second order correction only if schedule not finished yet
            if sigp1 != 0:
                # Now the conditioning can be last prediction
                p_hat = self.follow(x_t, sigp1, x_c = x_c if use_x_c else None, ctrl = ctrl, guide = guide, clamp = clamp, use_ema = use_ema)
                dxdtp = (x_t - p_hat) / sigp1

                x_t = x_hat + 0.5 * (sigp1 - sig_hat) * (dx_dt + dxdtp)

            x_c = p_hat

            yield x_t.clamp(-1., 1.) if clamp else x_t

    @property
    def sampler(self) -> Callable:
        solver = self.ode_solver
        if   solver == "heun" : return self.heun
        elif solver == "dpm++": return self.dpmpp
        elif solver == "ddim" : return self.ddim
        elif solver == "heun_sde": return self.heun_sde
        elif solver == "dpm++_sde": return self.dpmpp_sde
        elif solver == "heun_sde_inpaint": return self.heun_sde_inpaint
        else:
            raise ValueError(f"Unknown sampler {solver}")

    def compute_loss(
        self: Self,
        x_0 : th.Tensor,
        ctrl : Optional[th.Tensor] = None,
        use_x_c : Optional[bool] = None,     
        norm_fn : Optional[Callable] = None,
        use_ema : bool = False,
    ) -> th.Tensor:
        use_x_c = default(use_x_c, self.self_cond)
        norm_fn = default(norm_fn, self.norm_forward)

        # Extract the structure from the control
        structure = ctrl["structure"].unsqueeze(1)  # Add channel dimension

        # Encode the condition using the sequence encoder
        ctrl = ((self.ctrl_emb_ema(ctrl) if use_ema else self.ctrl_emb(ctrl)) if exists(ctrl) else ctrl)

        # Normalize input volumes
        x_0 = norm_fn(x_0)

        bs, bits, *_ = x_0.shape

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
                x_c = self.predict(x_t, sig, ctrl = ctrl, use_ema = use_ema)
                x_c.detach_()

        x_p = self.predict(x_t, sig, x_c = x_c, ctrl = ctrl, use_ema = use_ema)
        structure = structure.expand(bs, bits, *structure.shape[2:])  # (B, N, Y, Z, X)

        if self.average_loss_by_block:
            # Weighs samples in the batch according to how many solid blocks they have, discounting samples that have very few blocks

            # Compute the distribution loss ONLY for blocks that are solid in the structures
            loss = self.criterion(x_p[structure], x_0[structure], reduction = "none")  # (D,)

            batch_loss_weight = self.loss_weight(sig)  # (B,)
            bit_loss_weight = th.repeat_interleave(batch_loss_weight, bits)  # (B * N,)
            structure_block_sizes = structure.sum(dim=[-1, -2, -3]).flatten()  # (B * N,)

            block_loss_weight = th.repeat_interleave(bit_loss_weight, structure_block_sizes)  # (D,)

            # Add loss weight (fixed to be by block instead of by batch)
            loss *= block_loss_weight
        else:
            # Weighs each sample in the batch equally, discounting overly influential extremely full samples
            loss = self.criterion(x_p, x_0, reduction = "none")  # (B, N, Y, Z, X)

            # Compute average only over blocks in the structure
            bits_per_batch = structure.sum(dim=[-1, -2, -3])  # (B, N)
            loss = (loss * structure).sum(dim=[-1, -2, -3]) / bits_per_batch  # (B, N)
            loss = loss.mean(dim=-1)  # (B,)

            # Add loss weight
            batch_loss_weight = self.loss_weight(sig)  # (B,)
            loss *= batch_loss_weight

        return loss.mean()

    def predict(
        self,
        x_t : th.Tensor,
        sig : th.Tensor,
        x_c  : Optional[th.Tensor] = None,
        ctrl : Optional[th.Tensor] = None,
        clamp : bool = False,
        use_ema : bool = False,
    ) -> th.Tensor:
        '''
            Apply the backbone model to come up with a prediction, the
            nature of which depends on the diffusion objective (can either
            be noise|x_start|v prediction).
        '''

        bs, *_, device = x_t.shape, x_t.device

        if isinstance(sig, float): sig = torch.full((bs,), sig, device = device)

        # Inject appropriate noise value to images
        p_sig = enlarge_as(sig, x_t)
        x_sig = self.c_in(p_sig) * x_t
        t_sig = self.c_noise(sig)

        # Use the model to come up with a (hybrid) prediction the nature of
        # which depends on the implementation of the various c_<...> terms
        # so that the network can either predict the noise (eps) or the
        # input directly (better when noise is large!)
        model = self.ema if use_ema else self.model
        out : Tensor = model(x_sig, t_sig, x_c = x_c, ctrl = ctrl)
        out : Tensor = self.c_skip(p_sig) * x_t + self.c_out(p_sig) * out

        if clamp: out = out.clamp(-1., 1.)

        return out

    @classmethod
    def int2bit(cls, decs: th.Tensor, nbits: int = 8, scale: float = 1.) -> th.Tensor:
        """
            Convert input (int) tensor x (values in [0, 255])
            to analog bits in [-1, 1].
        """
        device = decs.device

        decs = decs.clamp(min = 0, max = 255).long()

        # Build the bitmask needed for decimal-to-bit conversion
        mask = 2 ** th.arange(nbits - 1, -1, -1, device = device, dtype = th.long)

        mask = rearrange(mask, "d -> d 1 1 1").contiguous()
        decs = rearrange(decs, "b c y z x -> b c 1 y z x").contiguous()

        # Get the analog bits
        bits = ((decs & mask) != 0).float()
        bits = rearrange(bits, "b c d y z x -> b (c d) y z x").contiguous()

        return (bits * 2 - 1) * scale

    @classmethod
    def bit2int(cls, bits: th.Tensor, nbits: int = 8, scale: float = 1.) -> th.Tensor:
        """
            Convert input (float) tensor x (values in [-1, 1])
            to discrete values in [0, 255].
        """
        device = bits.device

        # bits = (bits > 0).int()
        bits = bits / scale
        bits = bits.clamp(min=-scale, max=scale) / (scale * 2) + 0.5
        bits = bits.round().long()
        mask = 2 ** th.arange(nbits - 1, -1, -1, device = device, dtype = th.long)

        mask = rearrange(mask, "d -> d 1 1 1").contiguous()
        bits = rearrange(bits, "b (c d) y z x -> b c d y z x", d = nbits).contiguous()

        decs = reduce(bits * mask, "b c d y z x -> b c y z x", "sum").contiguous()

        return decs.clamp(0, 255)

    def lr_scheduler_step(self, scheduler: th.optim.lr_scheduler.LRScheduler, metric: Any | None) -> None:
        scheduler.step(epoch=self.global_step, metrics=metric)

    def configure_optimizers(self) -> tuple[list]:
        optim_conf = self.conf["OPTIMIZER"]

        optimizer = th.optim.AdamW(self.parameters(), lr=optim_conf["lr"], weight_decay=optim_conf["wd"])
        scheduler = SequentialLR(optimizer, [
            th.optim.lr_scheduler.LinearLR(optimizer, start_factor=1/optim_conf["warmup_steps"], total_iters=optim_conf["warmup_steps"]),
            ChainedScheduler([
                th.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=optim_conf["restart_interval"], T_mult=2, eta_min=optim_conf["min_lr"]),
                th.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda step: optim_conf["lr_decay"] ** (math.floor(math.log2(step/optim_conf["restart_interval"] + 1)))),
                th.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=optim_conf["plateau_mode"], factor=0.5, patience=optim_conf["plateau_patience"], min_lr=optim_conf["min_lr"]),
            ])
        ], milestones=[optim_conf["warmup_steps"]])

        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
            "monitor": optim_conf["plateau_metric"],
        }

        return [optimizer], [scheduler_config]
