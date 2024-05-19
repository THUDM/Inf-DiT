import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat

from sat.model.mixins import BaseMixin
from sat.transformer_defaults import HOOKS_DEFAULT
from sat.ops.layernorm import LayerNorm

class TimeEmbedding(nn.Module):
    def __init__(self, hidden_size, num_channels=192, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint
        self.mlp = nn.Sequential(
            nn.Linear(num_channels, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True)            
        )
        
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.xavier_uniform_(self.mlp[2].weight)
        nn.init.constant_(self.mlp[0].bias, 0)
        nn.init.constant_(self.mlp[2].bias, 0)

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        x = self.mlp(x)
        return x
    
class DDPMTimeEmbedding(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.xavier_uniform_(self.mlp[2].weight)
        nn.init.constant_(self.mlp[0].bias, 0)
        nn.init.constant_(self.mlp[2].bias, 0)

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
        t_emb = self.mlp(t_freq.to(t.dtype))
        return t_emb


class ConditionEmbedding(nn.Module):
    def __init__(self, hidden_size, label_dim, augment_dim, vector_dim, label_dropout=0):
        super().__init__()
        self.label_dim = label_dim
        self.label_dropout = label_dropout
        
        self.map_augment = nn.Linear(in_features=augment_dim, out_features=hidden_size, bias=False) if augment_dim else None
        self.map_label = nn.Linear(in_features=label_dim, out_features=hidden_size, bias=False) if label_dim else None
        self.map_vector = nn.Sequential(
            nn.Linear(vector_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        ) if vector_dim != 0 else None

        if self.map_vector:
            #zero output init
            nn.init.constant_(self.map_vector[2].weight, 0)
            nn.init.constant_(self.map_vector[2].bias, 0)
    def forward(self, emb, **kwargs):
        x = kwargs["images"]
        
        if self.map_augment is not None and 'augment_labels' in kwargs:
            emb = emb + self.map_augment(kwargs['augment_labels'])
        if self.map_label is not None and 'class_labels' in kwargs:
            class_labels = kwargs['class_labels']
            tmp = torch.zeros((class_labels.shape[0], self.label_dim), dtype=x.dtype, device=class_labels.device)
            tmp[:, class_labels] = 1
            if self.training and self.label_dropout:
                tmp = tmp * (torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout).to(tmp.dtype)
            emb = emb + self.map_label(tmp)                     # (N, D)
        if self.map_vector is not None and 'vector' in kwargs:
            emb = emb + self.map_vector(kwargs['vector'])
        return emb
    
class ImagePatchEmbeddingMixin(BaseMixin):
    def __init__(self, in_channels, hidden_size, patch_size, bias=True, append_emb=False, add_emb=False, reg_token_num=0):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.append_emb = append_emb
        self.add_emb = add_emb

        self.reg_token_num  = reg_token_num
        if reg_token_num > 0:
            self.register_parameter('reg_token_emb', nn.Parameter(torch.zeros(reg_token_num, hidden_size)))
            nn.init.normal_(self.reg_token_emb, mean=0., std=0.02)

    
    def word_embedding_forward(self, input_ids, **kwargs):
        images = kwargs["images"]
        emb = self.proj(images)
        emb = emb.flatten(2).transpose(1, 2)
        if self.append_emb:
            emb = torch.cat((kwargs["emb"][:, None, :], emb), dim=1)
        if self.reg_token_num > 0:
            emb = torch.cat((self.reg_token_emb[None, ...].repeat(emb.shape[0], 1, 1), emb), dim=1)
        if self.add_emb:
            emb = emb + kwargs["emb"][:, None, :]
        return emb

    def reinit(self, parent_model=None):
        w = self.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.proj.bias, 0)
        del self.transformer.word_embeddings
    

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
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
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

class BasicPositionEmbeddingMixin(BaseMixin):
    def __init__(self, num_patches, hidden_size):
        super().__init__()
        self.num_patches = num_patches
        self.pos_embedding = nn.Parameter(torch.zeros(1, int(num_patches), int(hidden_size)), requires_grad=False)
        
    def position_embedding_forward(self, position_ids, **kwargs):
        return self.pos_embedding
    
    def reinit(self, parent_model=None):
        del self.transformer.position_embeddings
        pos_embed = get_2d_sincos_pos_embed(self.pos_embedding.shape[-1], int(self.num_patches ** 0.5))
        self.pos_embedding.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        
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
        
class RotaryPositionEmbeddingMixin(BaseMixin):
    def __init__(
        self,
        num_patches,
        hidden_size,
        hidden_size_head,
        ft_seq_len=None,
        custom_freqs = None,
        freqs_for = 'lang',
        theta = 10000,
        max_freq = 10,
        num_freqs = 1,
        rot_v=False,
        pix2struct=False,
        qk_ln=False,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.hidden_size = hidden_size
        self.rot_v = rot_v
        self.pix2struct = pix2struct
        self.qk_ln = qk_ln
        
        if qk_ln:
            self.query_layernorm = LayerNorm(hidden_size_head, eps=1e-6)
            self.key_layernorm = LayerNorm(hidden_size_head, eps=1e-6)
        
        if self.pix2struct:
            pt_seq_len = num_patches
        else:
            pt_seq_len = int(num_patches**0.5)
        dim = hidden_size_head // 2
        
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * math.pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        if ft_seq_len is None: ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len
        
        freqs = torch.einsum('..., f -> ... f', t, freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r = 2)
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim = -1)
        
        if self.pix2struct:
            freqs_cos = freqs.cos()
            freqs_sin = freqs.sin()
        else:
            freqs_cos = freqs.contiguous().cos().view(-1, freqs.shape[-1])
            freqs_sin = freqs.contiguous().sin().view(-1, freqs.shape[-1])

        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

        # print('======== shape of rope freq', self.freqs_cos.shape, '========')
        
    def rotary(self, t, **kwargs):
        if self.pix2struct:
            x_coords = kwargs['rope_position_ids'][:, :, 0]
            y_coords = kwargs['rope_position_ids'][:, :, 1]
            freqs_cos = self.freqs_cos[x_coords, y_coords].unsqueeze(1)
            freqs_sin = self.freqs_sin[x_coords, y_coords].unsqueeze(1)
        else:
            freqs_cos = self.freqs_cos
            freqs_sin = self.freqs_sin
   
        return  t * freqs_cos + rotate_half(t) * freqs_sin
    
    def position_embedding_forward(self, position_ids, **kwargs):
        return None
    
    def attention_fn(self, query_layer, key_layer, value_layer, attention_mask,
                     attention_dropout=None, log_attention_weights=None, scaling_attention_score=True, **kwargs):
        attention_fn_default = HOOKS_DEFAULT["attention_fn"]
        if self.qk_ln:
            query_layer = self.query_layernorm(query_layer)
            key_layer = self.key_layernorm(key_layer)
        if query_layer.shape[-2] == key_layer.shape[-2]: # only for self attention
            query_layer = torch.cat((query_layer[:, :, :-self.num_patches, :], self.rotary(query_layer[:, :, -self.num_patches:, :], **kwargs)), dim=2)
            key_layer = torch.cat((key_layer[:, :, :-self.num_patches, :], self.rotary(key_layer[:, :, -self.num_patches:, :], **kwargs)), dim=2)
            if self.rot_v:
                value_layer = torch.cat((value_layer[:, :, :-self.num_patches, :], self.rotary(value_layer[:, :, -self.num_patches:, :])), dim=2)
        
        return attention_fn_default(query_layer, key_layer, value_layer, attention_mask,
                                    attention_dropout=attention_dropout, 
                                    log_attention_weights=log_attention_weights, 
                                    scaling_attention_score=scaling_attention_score, 
                                    **kwargs)


class RotaryPositionEmbedding(nn.Module):
    def __init__(
            self,
            num_patches,
            hidden_size,
            hidden_size_head,
            ft_seq_len=None,
            custom_freqs=None,
            freqs_for='lang',
            theta=10000,
            max_freq=10,
            num_freqs=1,
            rot_v=False,
            pix2struct=False,
            qk_ln=False,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.hidden_size_head = hidden_size_head
        self.hidden_size = hidden_size
        self.rot_v = rot_v
        self.pix2struct = pix2struct
        if self.pix2struct:
            pt_seq_len = num_patches
        else:
            pt_seq_len = int(num_patches ** 0.5)
        dim = hidden_size_head // 2

        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * math.pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        if ft_seq_len is None: ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        freqs = torch.einsum('..., f -> ... f', t, freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r=2)
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim=-1)

        if self.pix2struct:
            freqs_cos = freqs.cos()
            freqs_sin = freqs.sin()
        else:
            freqs_cos = freqs.contiguous().cos().view(-1, freqs.shape[-1])
            freqs_sin = freqs.contiguous().sin().view(-1, freqs.shape[-1])

        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

        # print('======== shape of rope freq', self.freqs_cos.shape, '========')


    def reinit(self, num_patches):
        dtype = self.freqs_cos.dtype
        hidden_size_head = self.hidden_size_head

        if self.pix2struct:
            pt_seq_len = num_patches
        else:
            pt_seq_len = int(num_patches ** 0.5)
        dim = hidden_size_head // 2
        ft_seq_len = None
        custom_freqs = None
        freqs_for = 'lang'
        theta = 10000
        max_freq = 10
        num_freqs = 1

        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * math.pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        if ft_seq_len is None: ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        freqs = torch.einsum('..., f -> ... f', t, freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r=2)
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim=-1)


        if self.pix2struct:
            freqs_cos = freqs.cos()
            freqs_sin = freqs.sin()
        else:
            freqs_cos = freqs.contiguous().cos().view(-1, freqs.shape[-1])
            freqs_sin = freqs.contiguous().sin().view(-1, freqs.shape[-1])

        freqs_cos = freqs_cos.to(dtype).to(self.freqs_cos.device)
        freqs_sin = freqs_sin.to(dtype).to(self.freqs_sin.device)
        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

    def forward(self, t, **kwargs):
        if self.pix2struct:
            x_coords = kwargs['rope_position_ids'][:, :, 0]
            y_coords = kwargs['rope_position_ids'][:, :, 1]
            freqs_cos = self.freqs_cos[x_coords, y_coords].unsqueeze(2)
            freqs_sin = self.freqs_sin[x_coords, y_coords].unsqueeze(2)
        else:
            freqs_cos = self.freqs_cos
            freqs_sin = self.freqs_sin
        return t * freqs_cos + rotate_half(t) * freqs_sin

