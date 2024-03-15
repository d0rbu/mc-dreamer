import torch.nn as nn

from modular_diffusion.diffusion.module.components.convolution import ConvTimeRes
from modular_diffusion.diffusion.module.utils.misc import default

from typing import Optional

def Upsample3D(dim_in, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = "nearest"),
        nn.Conv3d(dim_in, default(dim_out, dim_in), 3, padding = 1)
    )

def Downsample3D(dim_in, dim_out = None):
    return nn.Conv3d(dim_in, default(dim_out, dim_in), 4, 2, 1)

class ConvTimeRes3D(ConvTimeRes):
    """
        Convolutional Residual Block with time embedding
        injection support, used by Diffusion Models. It is
        composed of two convolutional layers with normalization.
        The time embedding signal is injected between the two
        convolutions and is added to the input to the second one.
    """

    def __init__(
        self,
        inp_dim : int,
        out_dim : Optional[int] = None,
        hid_dim : Optional[int] = None,
        ctx_dim : Optional[int] = None,
        num_group : int = 8,
    ) -> None:
        # Skip ConvTimeRes init
        super(ConvTimeRes, self).__init__()

        out_dim = default(out_dim, inp_dim)
        hid_dim = default(hid_dim, out_dim)
        ctx_dim = default(ctx_dim, out_dim)

        self.time_emb = nn.Sequential(
            nn.SiLU(inplace = False),
            nn.Linear(ctx_dim, hid_dim),
        )

        self.conv1 = nn.Sequential(
            nn.Conv3d(inp_dim, hid_dim, kernel_size = 3, padding = 1),
            nn.GroupNorm(num_group, hid_dim),
            nn.SiLU(inplace = False),
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(hid_dim, out_dim, kernel_size = 3, padding = 1),
            nn.GroupNorm(num_group, out_dim),
            nn.SiLU(inplace = False),
        )

        self.skip = nn.Conv3d(inp_dim, out_dim, 1) if inp_dim != out_dim else nn.Identity()
