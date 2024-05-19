import numpy as np
import scipy.stats as st
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from inspect import isfunction
import importlib
from dit import torch_dct

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]

def instantiate_from_config(config, **kwargs):
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", kwargs))


def get_obj_from_str(string, reload=False, invalidate_cache=True):
    module, cls = string.rsplit(".", 1)
    if invalidate_cache:
        importlib.invalidate_caches()
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def postprocess_pix2struct(images, rope_position_ids, make_grid=False):
    
    def crop_and_pad(image, target_height, target_width):
        channels, height, width = image.shape
        if height <= target_height:
            pad_h = target_height - height
            pad_top, pad_bottom = pad_h // 2, (pad_h + 1) // 2
            image = F.pad(image, (0, 0, pad_top, pad_bottom), mode='constant', value=-1)
        else:
            crop_h = height - target_height
            crop_top, crop_bottom = crop_h // 2, (crop_h + 1) // 2
            image = image[:, crop_top:-crop_bottom, :]
        if width <= target_width:
            pad_w = target_width - width
            pad_left, pad_right = pad_w // 2, (pad_w + 1) // 2
            image = F.pad(image, (pad_left, pad_right, 0, 0), mode='constant', value=-1)
        else:
            crop_w = width - target_width
            crop_left, crop_right = crop_w // 2, (crop_w + 1) // 2
            image = image[:, :, crop_left:-crop_right]
        return image

    patch_size = images.shape[-2]
    image_size = int((images.shape[-1] * images.shape[-2])**0.5)
    image_list = []
    for pos_id, image in zip(rope_position_ids, images):
        num_rows = pos_id[:, 0].max().item() + 1
        num_cols = pos_id[:, 1].max().item() + 1
        image = image[..., :num_rows * num_cols * patch_size]
        image = rearrange(image, "c p (h w q) -> c (h p) (w q)", h=num_rows, w=num_cols).to(torch.float64)

        if make_grid:
            scale = image_size / max(num_rows * patch_size, num_cols * patch_size)
            image = F.interpolate(image.unsqueeze(0), scale_factor=scale).squeeze(0)
            image = crop_and_pad(image, target_height=image_size, target_width=image_size)
        image_list.append(image.unsqueeze(0))

    if make_grid:
        image_list = torch.cat(image_list, dim=0)
    return image_list


def blocked_noise(ref_x, block_size=1, scale=1, device=None):
    g_noise = torch.randn_like(ref_x, device=device) * scale
    if block_size == 1:
        return g_noise
    
    blk_noise = torch.zeros_like(ref_x, device=device)
    for px in range(block_size):
        for py in range(block_size):
            blk_noise += torch.roll(g_noise, shifts=(px, py), dims=(-2, -1))
            
    blk_noise = blk_noise / block_size # to maintain the same std on each pixel

    return blk_noise

def pix2struct_blocked_noise(ref_x, block_size=1, scale=1, device=None, rope_position_ids=None):
    b, c, patch_size, total_len = ref_x.shape
    blk_noise_list = []
    for pos_id in rope_position_ids:
        rows = pos_id[:, 0].max().item() + 1
        cols = pos_id[:, 1].max().item() + 1
        real_x = torch.randn((3, rows*patch_size, cols*patch_size), dtype=ref_x.dtype, device=ref_x.device)
        blk_noise = blocked_noise(real_x, block_size, scale, ref_x.device)

        reshape_blk_noise = blk_noise.permute(1,2,0).reshape(rows, patch_size, cols, patch_size, 3).transpose(1,2).reshape(-1, patch_size, patch_size, 3).permute(3, 1, 0, 2).reshape(3, patch_size, -1)
        padding_blk_noise = torch.cat((reshape_blk_noise, reshape_blk_noise[:,:, :total_len-cols*rows*patch_size]), dim=2)
        blk_noise_list.append(padding_blk_noise.unsqueeze(0))

    return torch.cat(blk_noise_list, dim=0)

def DCTBlur(x, patch_size, blur_sigmas, device, min_scale=0.001):
    blur_sigmas = torch.as_tensor(blur_sigmas).to(device)
    freqs = torch.pi * torch.linspace(0, patch_size-1, patch_size).to(device) / patch_size
    frequencies_squared = freqs[:, None]**2 + freqs[None, :]**2

    t = blur_sigmas ** 2 / 2
    
    dct_coefs = torch_dct.dct_2d(x, patch_size, norm='ortho')
    h_scale = x.shape[-2] // patch_size
    w_scale = x.shape[-1] // patch_size
    dct_coefs = dct_coefs * (torch.exp(-frequencies_squared.repeat(h_scale,w_scale) * t) * (1 - min_scale) + min_scale)
    return torch_dct.idct_2d(dct_coefs, patch_size, norm='ortho')

def get_alpha_t(t, patch_size, img_shape, device, prob_length=0.93, blur_sigma_max=3, min_scale=0.001):
    """
    build blurring matrix on time t
    """
    P_mean, P_std = -1.2, 1.2
    origin_dtype = t.dtype
    t = st.norm.cdf((np.log(t.to(torch.float32).cpu()) - P_mean)/ P_std) / prob_length
    t = torch.from_numpy(t).to(origin_dtype).to(device)
    
    blur_sigmas = blur_sigma_max * torch.sin(t * torch.pi / 2)**2
    blur_sigmas = blur_sigmas.to(device)
    blur_ts = blur_sigmas**2 / 2
    
    freqs = torch.pi * torch.linspace(0, patch_size - 1, patch_size).to(device) / patch_size
    freqs_squared = freqs[:, None]**2 + freqs[None, :]**2
    h_scale = img_shape[-2] // patch_size
    w_scale = img_shape[-1] // patch_size
    
    return torch.exp(-freqs_squared.repeat(h_scale, w_scale) * blur_ts[:,None,None,None]) * (1 - min_scale) + min_scale