# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import os
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp, RmsNorm
from functools import partial
from typing import Optional
import torch.nn.functional as F
from repa.repa_mapping import MAR_mapping
from einops import rearrange, repeat
from math import pi
from models.wavelet_layer import DWT_2D, IDWT_2D



def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
                nn.Linear(hidden_size, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, z_dim),
            )

def l2norm(t, dim = -1):
    return F.normalize(t, dim = dim, p = 2)

class L2Norm(nn.Module):
    def __init__(self, dim = -1):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        return l2norm(t, dim = self.dim)
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=False):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    
class GroupRMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, group=1):
        super().__init__()
        assert dim % group == 0, "dimension should be divisable the number of groups"
        self.eps = eps
        self.group = group
        self.dim = dim
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        size = x.size()
        x = x.reshape(*size[:-1], self.group, self.dim//self.group)
        norm_term = torch.rsqrt(x.pow(2).mean(-1, keepdim=True)+self.eps)
        x  = x * norm_term
        return x.reshape(*size)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
class GaussianFourierFeatureTransform(torch.nn.Module):
    def __init__(self, input_size, half_hidden_size, scale=10):
        super().__init__()
        self.half_hidden_size = half_hidden_size
        self._B = torch.randn((input_size, half_hidden_size), requires_grad=False) * scale

    def forward(self, x):
        x = x @ self._B.to(x.device)
        x = 2 * pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=1)
    
class GaussianFourierFeatureTransformv3(nn.Module):
    def __init__(self, input_size, output_dim, sigma=1.0):
        """
        input_dim: Dimension of the input data.
        output_dim: Number of Fourier features (embedding dimension).
        sigma: Bandwidth of the Gaussian kernel.
        """
        super().__init__()
        self.input_dim = input_size
        self.output_dim = output_dim
        self.sigma = sigma
        
        # Sample random Gaussian frequencies
        self.register_buffer('W', torch.randn(input_size, output_dim) / sigma)
        # Sample random phase shifts uniformly
        self.register_buffer('b', 2 * torch.pi * torch.rand(output_dim))

    def forward(self, X):
        """
        Transforms the input data X into the Fourier feature space.

        X: Input data, shape (n_samples, input_dim)
        Returns: Transformed data, shape (n_samples, output_dim)
        """
        projection = X @ self.W + self.b
        return torch.sqrt(torch.tensor(2.0 / self.output_dim)) * torch.cos(projection)
    
class GaussianFourierFeatureTransformv2(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.cat([torch.sin(x), torch.cos(x)], dim=1)

class GLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer = nn.SiLU,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.act_layer = act_layer()
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)
        # apply init
        # nn.init.kaiming_normal_(self.w12.weight)
        # nn.init.kaiming_normal_(self.w3.weight)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = self.act_layer(x1) * x2
        return self.w3(hidden)
    
class GLUFFN_Fourier(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        patch = None,
        act_layer = nn.SiLU,
        bias: bool = True,
        post = False,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.post = post
        self.act_layer = act_layer()
        self.p = patch
        self.fft = nn.Parameter(torch.cat([torch.ones((in_features, self.p, self.p // 2 + 1, 1)), \
            torch.zeros((in_features, self.p, self.p // 2 + 1, 1))], dim=-1))
        # self.fft = nn.Parameter(torch.ones((in_features, self.p, self.p // 2 + 1)))
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x): # [B, L, D]
        B, L, D = x.shape
        if not self.post:
            x = x.permute(0, 2, 1).reshape(B, D, self.p, self.p)
            x_fft = torch.fft.rfft2(x, norm="ortho")
            # x_fft = torch.fft.rfft2(x)
            x_fft = x_fft * torch.view_as_complex(self.fft)
            # x_fft = x_fft * self.fft
            x = torch.fft.irfft2(x_fft, s=(self.p, self.p), norm="ortho").reshape(B, D, L).permute(0,2,1)
            # x = torch.fft.irfft2(x_fft, s=(self.p, self.p)).reshape(B, D, L).permute(0,2,1)
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = self.act_layer(x1) * x2
        x = self.w3(hidden)
        if self.post:
            x = x.permute(0, 2, 1).reshape(B, D, self.p, self.p)
            x_fft = torch.fft.rfft2(x, norm="ortho")
            # x_fft = torch.fft.rfft2(x)
            x_fft = x_fft * torch.view_as_complex(self.fft)
            # x_fft = x_fft * self.fft
            x = torch.fft.irfft2(x_fft, s=(self.p, self.p), norm="ortho").reshape(B, D, L).permute(0,2,1)
            # x = torch.fft.irfft2(x_fft, s=(self.p, self.p)).reshape(B, D, L).permute(0,2,1)
        return x


class GLUFFN_Wavelet(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        patch = None,
        act_layer = nn.SiLU,
        bias: bool = True,
        post = False,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.post = post
        self.act_layer = act_layer()
        self.p = patch
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)
        self.dwt = DWT_2D(wave="haar")
        self.idwt = IDWT_2D(wave="haar")
        self.wavelet_mapper = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_features, out_channels=in_features*4, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x): # [B, L, D]
        B, L, D = x.shape
        if not self.post:
            x = x.permute(0, 2, 1).reshape(B, D, self.p, self.p)
            gate = self.wavelet_mapper(x)
            subbands = self.dwt(x)  # xll, xlh, xhl, xhh where each has shape of [b, 4c, h/2, w/2]
            subbands = subbands*gate
            out = self.idwt(subbands)
            x = out.reshape(B, D, L).permute(0, 2, 1)
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = self.act_layer(x1) * x2
        x = self.w3(hidden)
        if self.post:
            x = x.permute(0, 2, 1).reshape(B, D, self.p, self.p)
            gate = self.wavelet_mapper(x)
            subbands = self.dwt(x)  # xll, xlh, xhl, xhh where each has shape of [b, 4c, h/2, w/2]
            subbands = subbands*gate
            x = self.idwt(subbands)
            x = x.reshape(B, D, L).permute(0, 2, 1)
        return x

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.in_channels = num_classes + use_cfg_embedding
        self.embedding_table = nn.Embedding(self.in_channels, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings
    
    def get_in_channels(self):
        return self.in_channels
   
#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module): #NOTE: we are using both post and prev norm with wo_norm = False
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, norm, wo_norm, linear_act, freq_type="prev_mlp", patch=16, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.wo_norm = wo_norm
        self.freq_type = freq_type
        self.p = patch
        self.norm1 = norm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, norm_layer=norm, qk_norm=True, **block_kwargs) \
            if not wo_norm else Attention(hidden_size, num_heads=num_heads, qkv_bias=True, norm_layer=norm, **block_kwargs)
        self.norm2 = norm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio) ##### note for GluMLP they use half of mlp hidden dim, should consider double them
        if linear_act == "mish":
            self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.Mish, drop=0)
        elif linear_act == "relu":
            self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.ReLU, drop=0)   
        elif linear_act == "silu":
            self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.SiLU, drop=0)   
        elif linear_act == "gelu":
            self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=partial(nn.GELU, "tanh"), drop=0)
        elif linear_act == "gate_relu":
            self.mlp = GLUFFN(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.ReLU)
        else:
            raise Exception("Not implementated in this code, dumbass")
        
        if "prev_mlp" in self.freq_type:
            self.prev_mlp = nn.Parameter(torch.cat([torch.ones((hidden_size, patch, patch // 2 + 1, 1)), \
                torch.zeros((hidden_size, patch, patch // 2 + 1, 1))], dim=-1))
        if "post_mlp" in self.freq_type:
            self.post_mlp = nn.Parameter(torch.cat([torch.ones((hidden_size, patch, patch // 2 + 1, 1)), \
                torch.zeros((hidden_size, patch, patch // 2 + 1, 1))], dim=-1))
        if "prev_norm_mlp" in self.freq_type:
            self.prev_norm_mlp = nn.Parameter(torch.cat([torch.ones((hidden_size, patch, patch // 2 + 1, 1)), \
                torch.zeros((hidden_size, patch, patch // 2 + 1, 1))], dim=-1))
        if "prev_norm_attn" in self.freq_type:
            self.prev_norm_attn = nn.Parameter(torch.cat([torch.ones((hidden_size, patch, patch // 2 + 1, 1)), \
                torch.zeros((hidden_size, patch, patch // 2 + 1, 1))], dim=-1))
        if "prev_attn" in self.freq_type:
            self.prev_attn = nn.Parameter(torch.cat([torch.ones((hidden_size, patch, patch // 2 + 1, 1)), \
                torch.zeros((hidden_size, patch, patch // 2 + 1, 1))], dim=-1))
        if "post_attn" in self.freq_type:
            self.post_attn = nn.Parameter(torch.cat([torch.ones((hidden_size, patch, patch // 2 + 1, 1)), \
                torch.zeros((hidden_size, patch, patch // 2 + 1, 1))], dim=-1))
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
            # GroupRMSNorm(6 * hidden_size, group=6) if ("norm" in self.cond_type) else nn.Identity()
        )
        
    def forward(self, x, c, feat_rope=None):
        B, L, D = x.size()
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        # attention
        if "prev_norm_attn" in self.freq_type:
            x = x.permute(0, 2, 1).reshape(B, D, self.p, self.p)
            x_fft = torch.fft.rfft2(x, norm="ortho")
            x_fft = x_fft * torch.view_as_complex(self.prev_norm_attn)
            x = torch.fft.irfft2(x_fft, s=(self.p, self.p), norm="ortho").reshape(B, D, L).permute(0,2,1)
        temp = modulate(self.norm1(x), shift_msa, scale_msa)
        if "prev_attn" in self.freq_type:
            temp = temp.permute(0, 2, 1).reshape(B, D, self.p, self.p)
            temp_fft = torch.fft.rfft2(temp, norm="ortho")
            temp_fft = temp_fft * torch.view_as_complex(self.prev_attn)
            temp = torch.fft.irfft2(temp_fft, s=(self.p, self.p), norm="ortho").reshape(B, D, L).permute(0,2,1)
        temp = self.attn(temp, feat_rope=feat_rope)
        if "post_attn" in self.freq_type:
            temp = temp.permute(0, 2, 1).reshape(B, D, self.p, self.p)
            temp_fft = torch.fft.rfft2(temp, norm="ortho")
            temp_fft = temp_fft * torch.view_as_complex(self.post_attn)
            temp = torch.fft.irfft2(temp_fft, s=(self.p, self.p), norm="ortho").reshape(B, D, L).permute(0,2,1)
        x = x + gate_msa.unsqueeze(1) * temp
        # gate mlp or mlp
        if "prev_norm_mlp" in self.freq_type:
            x = x.permute(0, 2, 1).reshape(B, D, self.p, self.p)
            x_fft = torch.fft.rfft2(x, norm="ortho")
            x_fft = x_fft * torch.view_as_complex(self.prev_norm_mlp)
            x = torch.fft.irfft2(x_fft, s=(self.p, self.p), norm="ortho").reshape(B, D, L).permute(0,2,1)
        temp = modulate(self.norm2(x), shift_mlp, scale_mlp)
        if "prev_mlp" in self.freq_type:
            temp = temp.permute(0, 2, 1).reshape(B, D, self.p, self.p)
            temp_fft = torch.fft.rfft2(temp, norm="ortho")
            temp_fft = temp_fft * torch.view_as_complex(self.prev_mlp)
            temp = torch.fft.irfft2(temp_fft, s=(self.p, self.p), norm="ortho").reshape(B, D, L).permute(0,2,1)
        temp = self.mlp(temp)
        if "post_mlp" in self.freq_type:
            temp = temp.permute(0, 2, 1).reshape(B, D, self.p, self.p)
            temp_fft = torch.fft.rfft2(temp, norm="ortho")
            temp_fft = temp_fft * torch.view_as_complex(self.post_mlp)
            temp = torch.fft.irfft2(temp_fft, s=(self.p, self.p), norm="ortho").reshape(B, D, L).permute(0,2,1)
        x = x + gate_mlp.unsqueeze(1) * temp
        return x
    
    
#################################################################################
#                                 Core DiT Model                                #
#################################################################################
class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels, norm, wo_norm):
        super().__init__()
        self.wo_norm = wo_norm
        self.norm_final = norm(hidden_size, elementwise_affine=False, eps=1e-6) if not wo_norm else nn.Identity() 
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        # should we put layernorm here for robust
        x = self.linear(x)
        return x
    
# @persistence.persistent_class
class MPFourier(torch.nn.Module):
    def __init__(self, num_channels=128, bandwidth=1):
        super().__init__()
        self.register_buffer('freqs', 2 * np.pi * torch.randn(num_channels) * bandwidth)
        self.register_buffer('phases', 2 * np.pi * torch.rand(num_channels))

    def forward(self, x):
        y = x.to(torch.float32)
        y = y.ger(self.freqs.to(torch.float32))
        y = y + self.phases.to(torch.float32)
        y = y.cos() * np.sqrt(2)
        return y.to(x.dtype)

class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        norm_type = "layer",
        linear_act=None,
        wo_norm=False,
        num_register=0,
        use_repa=False,
        projector_dim=None,
        z_depth=None,
        z_dim=None,
        repa_mapper=None,
        mar_mapper_num_res_blocks=2,
        separate_cond = False,
        use_rope = False,
        cond_mapping = False,
        freq_type="prev_mlp",
        cond_mixing = False,
        uw = False,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.depth = depth
        self.num_register = num_register
        self.separate_cond = separate_cond
        self.use_rope = use_rope
        self.cond_mapping = cond_mapping
        self.freq_type = freq_type
        self.cond_mixing = cond_mixing
        self.uw = uw   
        
        if self.uw:
            self.uw_mlp = nn.Sequential(
                MPFourier(),
                nn.Linear(128, 1),
            )
        
        if self.cond_mixing:
            self.cond_mixing_mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                # nn.Linear(hidden_size, hidden_size),
                nn.Linear(hidden_size, 1),
                nn.Sigmoid()
            )
        
        # use freq condition
        if self.cond_mapping:
            # self.cond_mapper = GaussianFourierFeatureTransform(input_size=hidden_size, half_hidden_size=hidden_size//2)
            self.cond_mapper = nn.Sequential(GaussianFourierFeatureTransformv2(), 
                                             L2Norm(dim=-1))
        
        # use rotary position encoding, borrow from EVA
        if self.use_rope:
            half_head_dim = hidden_size // num_heads // 2
            hw_seq_len = input_size // patch_size
            self.feat_rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=hw_seq_len,
            )
        else:
            self.feat_rope = None

        if norm_type == "layer":
            self.norm = nn.LayerNorm
        else:
            self.norm = RMSNorm
            
        self.attn_blk = DiTBlock
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        if self.cond_mapping:
            self.y_embedder = LabelEmbedder(num_classes, hidden_size//2, class_dropout_prob)
        else:
            self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_register, hidden_size), requires_grad=False) if not self.separate_cond \
            else nn.Parameter(torch.zeros(1, num_patches + self.num_register + 1, hidden_size), requires_grad=False) # additional class dim
        if self.num_register > 0:
            self.register_tokens = nn.Parameter(torch.randn((1, self.num_register, hidden_size)))

        self.blocks = nn.ModuleList([
            self.attn_blk(hidden_size, 
                     num_heads, 
                     mlp_ratio=mlp_ratio, 
                     norm=self.norm, 
                     linear_act=linear_act, 
                     wo_norm=wo_norm, 
                     patch=int(input_size/patch_size),
                     freq_type=freq_type) \
                for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, norm=self.norm, wo_norm=wo_norm)
            
        ############## REPA ##############
        self.use_repa = use_repa
        self.repa_mapper = repa_mapper
        if self.use_repa:
            self.z_depth = z_depth
            print(f"\033[35mz depth mapping: {self.z_depth}\033[0m")
            print(f"\033[35mrepa mapper: {self.repa_mapper}\033[0m")
            print(f"\033[35mprojector dim: {projector_dim}\033[0m")
            
            if self.repa_mapper=="mar":
                self.projectors = MAR_mapping(
                        in_channels=hidden_size,  # DiT
                        model_channels=hidden_size,
                        out_channels=z_dim,  # SSL
                        num_res_blocks=mar_mapper_num_res_blocks,
                        grad_checkpointing=False,)
            elif self.repa_mapper=="repa":
                self.projectors = build_mlp(hidden_size, projector_dim, z_dim)
            
        ############## REPA ##############
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5), cls_token=(self.num_register>0), extra_tokens=self.num_register) if not self.separate_cond \
            else get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5), cls_token=(self.num_register>0), extra_tokens=self.num_register+1)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        # if "adain" in self.cond_type or "both" in self.cond_type:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y=None):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        
        # print(x.shape, t.shape, y)
        
        t_ = t.clone()
        if y is None:
            y = torch.ones(x.size(0), dtype=torch.long, device=x.device) * (self.y_embedder.get_in_channels() - 1)
        x = self.x_embedder(x)                   # (N, T, D)
        t = self.t_embedder(t)                   # (N, D)
        
        y = self.y_embedder(y, self.training)    # (N, D)
        if self.cond_mapping:
            y = self.cond_mapper(y)
        if self.num_register > 0:
            x = torch.cat([x, self.register_tokens.expand(x.shape[0], -1, -1)], dim=1) # (N, T+register, D)
        if self.separate_cond:
            x = torch.cat([x, y.unsqueeze(1)], dim=1) # (N, T+register+1, D)
        x = x + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2 + num_register
        N, T, D = x.shape
        if self.separate_cond:
            c = t
        else:
            if self.cond_mixing:
                alpha = self.cond_mixing_mlp(t) 
                c = t * alpha + y
            else:
                c = t + y                               # (N, D)
        # print(c.shape)
        projected_feat = None
        for idx, block in enumerate(self.blocks):
            x = block(x, c, feat_rope=self.feat_rope)
            if self.use_repa and self.z_depth==(idx + 1):
                # Project features using corresponding projector
                if self.repa_mapper=="repa":    
                    projected_feat = self.projectors(x[:, :self.x_embedder.num_patches, :].reshape(-1, D)).reshape(N, self.x_embedder.num_patches, -1)
                elif self.repa_mapper=="mar":
                    projected_feat = self.projectors(x[:, :self.x_embedder.num_patches, :].reshape(-1, D), t.repeat_interleave(T, dim=0)).reshape(N, self.x_embedder.num_patches, -1)
        # remove register when time comes.
        if self.num_register > 0 or self.separate_cond:
            x = x[:, :self.x_embedder.num_patches, :]
        
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        if self.uw:
            uw = self.uw_mlp(t_)
            return x, projected_feat, uw
        return x, projected_feat, None

    def forward_with_cfg(self, x, t, cfg_scale=1.0, y=None):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        eps, _, _ = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return eps, _, _


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([pos_embed, np.zeros([extra_tokens, embed_dim])], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def broadcat(tensors, dim = -1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), 'invalid dimensions for broadcastable concatentation'
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim = dim)

def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')

class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(
        self,
        dim,
        pt_seq_len=16,
        ft_seq_len=None,
        custom_freqs = None,
        freqs_for = 'lang',
        theta = 10000,
        max_freq = 10,
        num_freqs = 1,
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        if ft_seq_len is None: ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len
        self.pt_seq_len = pt_seq_len

        freqs = torch.einsum('..., f -> ... f', t, freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r = 2)
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim = -1)

        freqs_cos = freqs.cos().view(-1, freqs.shape[-1])
        freqs_sin = freqs.sin().view(-1, freqs.shape[-1])

        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

        # print('======== shape of rope freq', self.freqs_cos.shape, '========')

    def forward(self, t):
        # t_, rest = t[:, :, :self.pt_seq_len**2, :], t[:, :, self.pt_seq_len**2:, :]
        t = t * self.freqs_cos + rotate_half(t) * self.freqs_sin
        # out = torch.cat([t_, rest], dim=2)
        return t


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}