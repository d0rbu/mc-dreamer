import torch as th
import torch.nn as nn
from einops import rearrange
from modular_diffusion.diffusion.module.utils.misc import default
from modular_diffusion.diffusion.module.components.adapter import ConvAdapter, ConvLinearAdapter
from typing import Optional

class ConvLinearAdapter3D(ConvLinearAdapter):
    """
        Adapter needed by an Attention Layer to
        adjust its behaviour to image-like inputs
    """
    def __init__(
        self,
        qry_dim : int,
        emb_dim : int,    
        key_dim : Optional[int] = None,
        val_dim : Optional[int] = None,
    ) -> None:
        super(ConvLinearAdapter, self).__init__()

        key_dim = default(key_dim, qry_dim)
        val_dim = default(val_dim, key_dim)

        self.to_q = nn.Conv3d(qry_dim, emb_dim, 1, bias = False)
        self.to_k = nn.Linear(key_dim, emb_dim, bias = False) # Might need to change these lines ----------------------------------------------------------------
        self.to_v = nn.Linear(val_dim, emb_dim, bias = False) # Might need to change these lines ---

        self.from_q = nn.Conv3d(emb_dim, qry_dim, 1, bias = False)

    def _proj_in(
        self,
        qry : th.Tensor,
        key : th.Tensor,
        val : th.Tensor
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        # Store q shape for out projection
        *_, self.y, self.z, self.x = qry.shape

        q = self.to_q(qry)
        k = self.to_k(key)
        v = self.to_v(val)

        q = rearrange(q, "b c y z x -> b (y z x) c").contiguous()

        return q, k, v

    def _proj_out(self, qry : th.Tensor) -> th.Tensor:
        if not hasattr(self, "x"):
            raise ValueError("Cannot call adapt._out before the _in method")

        qry = rearrange(qry, "b (y z x) c -> b c y z x", y = self.y, z = self.z, x = self.x).contiguous()

        return self.from_q(qry)

class ConvAdapter3D(ConvAdapter):
    """
        Adapter needed by an Attention Layer to
        adjust its behaviour to image-like inputs
    """
    def __init__(
        self,
        qry_dim : int,
        emb_dim : int,    
        key_dim : Optional[int] = None,
        val_dim : Optional[int] = None,
    ) -> None:
        super(ConvAdapter, self).__init__()

        key_dim = default(key_dim, qry_dim)
        val_dim = default(val_dim, key_dim)

        self.to_q = nn.Conv3d(qry_dim, emb_dim, 1, bias = False)
        self.to_k = nn.Conv3d(key_dim, emb_dim, 1, bias = False)
        self.to_v = nn.Conv3d(val_dim, emb_dim, 1, bias = False)

        self.from_q = nn.Conv3d(emb_dim, qry_dim, 1, bias = False)

    def _proj_in(
        self,
        qry : th.Tensor,
        key : th.Tensor,
        val : th.Tensor
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        # Store q shape for out projection
        *_, self.y, self.z, self.x = qry.shape

        q = self.to_q(qry)
        k = self.to_k(key)
        v = self.to_v(val)

        q = rearrange(q, "b c y z x -> b (y z x) c").contiguous()
        k = rearrange(k, "b c y z x -> b (y z x) c").contiguous()
        v = rearrange(v, "b c y z x -> b (y z x) c").contiguous()

        return q, k, v
    
    def _proj_out(self, qry : th.Tensor) -> th.Tensor:
        if not hasattr(self, "y"):
            raise ValueError("Cannot call adapt._out before the _in method")
        
        qry = rearrange(qry, "b (y z x) c -> b c y z x", y = self.y, z = self.z, x = self.x).contiguous()

        return self.from_q(qry)
