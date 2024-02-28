import torch.nn as nn
from modular_diffusion.diffusion.module.utils.misc import default
from modular_diffusion.diffusion.module.components.adapter import ConvAdapter, ConvLinearAdapter
from typing import Optional

class ConvLinearAdapter3D(ConvLinearAdapter):
    '''
        Adapter needed by an Attention Layer to
        adjust its behaviour to image-like inputs
    '''
    def __init__(
        self,
        qry_dim : int,
        emb_dim : int,    
        key_dim : Optional[int] = None,
        val_dim : Optional[int] = None,
    ) -> None:
        super(nn.Module, self).__init__()

        key_dim = default(key_dim, qry_dim)
        val_dim = default(val_dim, key_dim)

        self.to_q = nn.Conv3d(qry_dim, emb_dim, 1, bias = False)
        self.to_k = nn.Linear(key_dim, emb_dim, bias = False) # Might need to change these lines ----------------------------------------------------------------
        self.to_v = nn.Linear(val_dim, emb_dim, bias = False) # Might need to change these lines ---

        self.from_q = nn.Conv3d(emb_dim, qry_dim, 1, bias = False)

class ConvAdapter3D(ConvAdapter):
    '''
        Adapter needed by an Attention Layer to
        adjust its behaviour to image-like inputs
    '''
    def __init__(
        self,
        qry_dim : int,
        emb_dim : int,    
        key_dim : Optional[int] = None,
        val_dim : Optional[int] = None,
    ) -> None:
        super(nn.Module, self).__init__()

        key_dim = default(key_dim, qry_dim)
        val_dim = default(val_dim, key_dim)

        self.to_q = nn.Conv3d(qry_dim, emb_dim, 1, bias = False)
        self.to_k = nn.Conv3d(key_dim, emb_dim, 1, bias = False)
        self.to_v = nn.Conv3d(val_dim, emb_dim, 1, bias = False)

        self.from_q = nn.Conv3d(emb_dim, qry_dim, 1, bias = False)
