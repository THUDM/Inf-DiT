import os
from typing import Dict, Union

import torch
from omegaconf import ListConfig, OmegaConf
from tqdm import tqdm

from dit.utils import default, append_dims, instantiate_from_config, get_alpha_t, blocked_noise, pix2struct_blocked_noise
from dit import torch_dct
from PIL import Image
from torchvision.utils import make_grid
import numpy as np
from dit.sampling.utils import linear_multistep_coeff
import torchvision
from torchvision.transforms.functional import InterpolationMode
DEFAULT_GUIDER = {"target": "dit.sampling.guiders.IdentityGuider"}

def to_d(x, sigma, denoised):
    return (x - denoised) / append_dims(sigma, x.ndim)

class BaseDiffusionSampler:
    def __init__(
        self,
        discretization_config: Union[Dict, ListConfig, OmegaConf],
        num_steps: Union[int, None] = None,
        guider_config: Union[Dict, ListConfig, OmegaConf, None] = None,
        verbose: bool = False,
        device: str = "cuda",
    ):
        self.num_steps = num_steps
        self.discretization = instantiate_from_config(discretization_config)
        self.guider = instantiate_from_config(
            default(
                guider_config,
                DEFAULT_GUIDER,
            )
        )
        self.verbose = verbose
        self.device = device
        
    def prepare_sampling_loop(self, x, cond, uc=None, num_steps=None, init_noise=False):
        sigmas = self.discretization(
            self.num_steps if num_steps is None else num_steps, device=self.device
        ).to(x.dtype)
        uc = default(uc, cond)

        if init_noise:
            x = x + torch.randn_like(x) * sigmas[0]
        else:
            x = x * sigmas[0]
        num_sigmas = len(sigmas)
        
        s_in = x.new_ones([x.shape[0]])
        
        return x, s_in, sigmas, num_sigmas, cond, uc
    
    def denoise(self, x, denoiser, sigma, cond, uc, rope_position_ids, sample_step=None):
        images, sigmas, cond, rope_position_ids = self.guider.prepare_inputs(x, sigma, cond, uc, rope_position_ids)

        denoised = denoiser(images, sigmas, rope_position_ids, cond, sample_step)
        denoised = self.guider(denoised, sigma)
        return denoised
    
    def get_sigma_gen(self, num_sigmas):
        sigma_generator = range(num_sigmas - 1)
        if self.verbose:
            print("#" * 30, " Sampling setting ", "#" * 30)
            print(f"Sampler: {self.__class__.__name__}")
            print(f"Discretization: {self.discretization.__class__.__name__}")
            print(f"Guider: {self.guider.__class__.__name__}")
            sigma_generator = tqdm(
                sigma_generator,
                total=num_sigmas,
                desc=f"Sampling with {self.__class__.__name__} for {num_sigmas} steps",
            )
        return sigma_generator
        
        
class SingleStepDiffusionSampler(BaseDiffusionSampler):
    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc, *args, **kwargs):
        raise NotImplementedError

    def euler_step(self, x, d, dt):
        return x + dt * d

class EDMSampler(SingleStepDiffusionSampler):
    def __init__(
        self, s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.s_churn = s_churn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise
        
    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc=None, gamma=0, rope_position_ids=None, sample_step=None, return_attention_map=None):
        sigma_hat = sigma * (gamma + 1.0)
        if gamma > 0:
            eps = torch.torch.randn_like(x) * self.s_noise
            x = x + eps * append_dims(sigma_hat**2 - sigma**2, x.ndim) ** 0.5

        denoised = self.denoise(x, denoiser, sigma_hat, cond, uc, rope_position_ids, sample_step)
        d = to_d(x, sigma_hat, denoised)
        dt = append_dims(next_sigma - sigma_hat, x.ndim)

        euler_step = self.euler_step(x, d, dt)
        x = self.possible_correction_step(
            euler_step, x, d, dt, next_sigma, denoiser, cond, uc, rope_position_ids, sample_step
        )
        return x
    
    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, rope_position_ids=None):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        for i in self.get_sigma_gen(num_sigmas):
            gamma = (
                min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1)
                if self.s_tmin <= sigmas[i] <= self.s_tmax
                else 0.0
            )
            gamma = 0
            x = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc,
                gamma,
                rope_position_ids,
            )

        return x

    
class EulerEDMSampler(EDMSampler):
    def possible_correction_step(
        self, euler_step, x, d, dt, next_sigma, denoiser, cond, uc, rope_position_ids
    ):
        return euler_step


class HeunEDMSampler(EDMSampler):
    def possible_correction_step(
        self, euler_step, x, d, dt, next_sigma, denoiser, cond, uc, rope_position_ids
    ):
        if torch.sum(next_sigma) < 1e-14:
            # Save a network evaluation if all noise levels are 0
            return euler_step
        else:
            denoised = self.denoise(euler_step, denoiser, next_sigma, cond, uc, rope_position_ids)
            d_new = to_d(euler_step, next_sigma, denoised)
            d_prime = (d + d_new) / 2.0

            # apply correction if noise level is not 0
            x = torch.where(
                append_dims(next_sigma, x.ndim) > 0.0, x + d_prime * dt, euler_step
            )
            return x

# For super-resolution stage
    
class ConcatSRHeunEDMSampler(EDMSampler):

    def possible_correction_step(
        self, euler_step, x, d, dt, next_sigma, denoiser, cond, uc, rope_position_ids, sample_step
    ):
        if torch.sum(next_sigma) < 1e-14:
            # Save a network evaluation if all noise levels are 0
            return euler_step
        else:
            # concat_euler_step = torch.cat((euler_step, lr_images), dim=1)
            denoised = self.denoise(euler_step, denoiser, next_sigma, cond, uc, rope_position_ids, sample_step)
            d_new = to_d(euler_step, next_sigma, denoised)
            d_prime = (d + d_new) / 2.0

            # apply correction if noise level is not 0
            x = torch.where(
                append_dims(next_sigma, x.ndim) > 0.0, x + d_prime * dt, euler_step
            )

            return x
        

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, rope_position_ids=None, return_attention_map=False, init_noise=False):
        lr_images = cond["concat_lr_imgs"]

        if init_noise:
            if "image2" in cond.keys():
                images = cond["image2"]
            else:
                images = lr_images
        else:
            images = torch.randn_like(lr_images)

        images, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            images, cond, uc, num_steps, init_noise=init_noise
        )

        for i in tqdm(self.get_sigma_gen(num_sigmas)):
            gamma = (
                min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1)
                if self.s_tmin <= sigmas[i] <= self.s_tmax
                else 0.0
            )
            images = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                images,
                cond,
                uc,
                gamma,
                rope_position_ids,
                i,
                return_attention_map
            )

        return images

class ConcatDDIMSampler(SingleStepDiffusionSampler):
    def __init__(
        self, s_noise=0.1, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.s_noise = s_noise

    def possible_correction_step(
        self, euler_step, x, d, dt, next_sigma, denoiser, cond, uc, rope_position_ids, sample_step
    ):
        return euler_step

    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc=None, s_noise=0.0, rope_position_ids=None, sample_step=None, return_attention_map=None):

        denoised = self.denoise(x, denoiser, sigma, cond, uc, rope_position_ids, sample_step)

        d = to_d(x, sigma, denoised)
        dt = append_dims(next_sigma * (1 - s_noise**2)**0.5 - sigma, x.ndim)
        euler_step = x + dt * d + s_noise * append_dims(next_sigma, x.ndim) * torch.randn_like(x)
        x = self.possible_correction_step(
            euler_step, x, d, dt, next_sigma, denoiser, cond, uc, rope_position_ids, sample_step
        )
        return x

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, rope_position_ids=None, return_attention_map=False, init_noise=False):
        lr_images = cond["concat_lr_imgs"]
        if init_noise:
            images = lr_images
        else:
            images = torch.randn_like(lr_images)

        images, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            images, cond, uc, num_steps, init_noise=init_noise
        )


        for i in tqdm(self.get_sigma_gen(num_sigmas)):
            images = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                images,
                cond,
                uc,
                self.s_noise,
                rope_position_ids,
                i,
                return_attention_map
            )

        return images
