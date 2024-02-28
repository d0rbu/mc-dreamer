import torch as th
import torch.nn as nn

from model.components.adapter import ConvAdapter
from model.components.adapter import ConvLinearAdapter
from model.components.convolution import Upsample
from model.components.convolution import Downsample
from model.components.convolution import ConvTimeRes3D

from modular_diffusiuon.diffusion.module.unet import UNet
from modular_diffustion.module.components.embedding import TimeEmbedding
from modular_diffustion.module.components.attention import EfficientAttention

from modular_diffustion.module.utils.misc import exists
from modular_diffustion.module.utils.misc import default
from typing import Callable, Optional, override, Self, List

class Unet3D(Unet):
    '''
        U-Net model as introduced in:
        "U-Net: Convolutional Networks for Biomedical Image Segmentation".
        It is a common choice as network backbone for diffusion models.         
    '''

    def __init__(
        self,
        net_dim : int = 4,
        out_dim : Optional[int] = None,
        attn_dim : int = 128,
        channels : int = 8,
        ctrl_dim : Optional[int] = None,
        chn_mult : List[int] = (1, 2, 4, 8),
        num_group : int = 8,
        num_heads : int = 4,
        qry_chunk : int = 512,
        key_chunk : int = 1024,
    ) -> None:
        super().__init__()

        out_dim = default(out_dim, channels)

        self.channels = channels

        # NOTE: We need channels * 2 to accomodate for the self-conditioning
        self.proj_inp = nn.Conv3d(self.channels * 2, net_dim, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # self.proj_inp = nn.Conv2d(self.channels, net_dim, 7, padding = 3)

        dims = [net_dim, *map(lambda m: net_dim * m, chn_mult)]
        mid_dim = dims[-1]

        dims = list(zip(dims, dims[1:]))

        # * Context embedding
        ctx_dim = net_dim * 4
        self.time_emb = nn.Sequential(
            TimeEmbedding(net_dim),
            nn.Linear(net_dim, ctx_dim),
            nn.GELU(),
            nn.Linear(ctx_dim, ctx_dim)
        )

        # * Building the model. It has three main components:
        # * 1) The downsampling module
        # * 2) The bottleneck module
        # * 3) The upsampling module
        self.downs = nn.ModuleList([])
        self.ups   = nn.ModuleList([])
        num_resolutions = len(dims)

        attn_kw = {
            'num_heads' : num_heads,
            'qry_chunk' : qry_chunk,
            'key_chunk' : key_chunk,
            'pre_norm'  : True,
        }

        qkv_adapt = ConvLinearAdapter if exists(ctrl_dim) else ConvAdapter

        # Build up the downsampling module
        for idx, (dim_in, dim_out) in enumerate(dims):
            is_last = idx >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ConvTimeRes3D(dim_in, dim_in, ctx_dim = ctx_dim, num_group = num_group),
                ConvTimeRes3D(dim_in, dim_in, ctx_dim = ctx_dim, num_group = num_group),
                EfficientAttention(dim_in, attn_dim, qkv_adapt = qkv_adapt, key_dim = ctrl_dim, **attn_kw),
                nn.Conv3d(dim_in, dim_out, kernel_size=(3, 3, 3), padding=(1, 1, 1)) if is_last else Downsample(dim_in, dim_out)  # Adjust Downsample for 3D
            ]))

        # Buildup the bottleneck module
        self.mid_block1 = ConvTimeRes3D(mid_dim, mid_dim, ctx_dim = ctx_dim, num_group = num_group)
        self.mid_attn   = EfficientAttention(mid_dim, attn_dim, qkv_adapt = qkv_adapt, key_dim = ctrl_dim, **attn_kw)
        self.mid_block2 = ConvTimeRes3D(mid_dim, mid_dim, ctx_dim = ctx_dim, num_group = num_group)

        # Build the upsampling module
        # NOTE: We need to make rooms for incoming residual connections from the downsampling layers
        for idx, (dim_in, dim_out) in enumerate(reversed(dims)):
            is_last = idx >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ConvTimeRes3D(dim_in + dim_out, dim_out, ctx_dim = ctx_dim, num_group = num_group),
                ConvTimeRes3D(dim_in + dim_out, dim_out, ctx_dim = ctx_dim, num_group = num_group),
                EfficientAttention(dim_out, attn_dim, qkv_adapt = qkv_adapt, key_dim = ctrl_dim, **attn_kw),
                #nn.Conv2d(dim_out, dim_in, 3, padding = 1) if is_last else Upsample(dim_out, dim_in)
                nn.ConvTranspose3d(dim_out, dim_in, kernel_size=(2, 2, 2), stride=(2, 2, 2)) # May be broken ----------------------------------------------------------------
            ]))

        self.final = ConvTimeRes3D(net_dim * 2, net_dim, ctx_dim = ctx_dim, num_group = num_group)
        self.proj_out = nn.Conv3d(net_dim, out_dim, kernel_size=1)
