import torch
import scipy.stats as st
import numpy as np

from dit.utils import default, instantiate_from_config


class EDMSampling:
    def __init__(self, p_mean=-1.2, p_std=1.2, truncated_sigma_min=0.001):
        self.p_mean = p_mean
        self.p_std = p_std
        self.truncated_sigma_min = truncated_sigma_min

    def __call__(self, n_samples, rand=None):
        log_sigma = self.p_mean + self.p_std * default(rand, torch.randn((n_samples,)))
        sigma = torch.clip(log_sigma.exp(), self.truncated_sigma_min)
        return sigma


class RDMSampling:
    def __init__(self, p_mean=-1.2, p_std=1.2, truncated_sigma_min=7.5e-3, prob_length=0.9, blur_sigma_max=3):
        self.p_mean = p_mean
        self.p_std = p_std
        self.truncated_sigma_min = truncated_sigma_min

        self.prob_length = prob_length
        self.blur_sigma_max = blur_sigma_max

    def __call__(self, n_samples, rand=None):
        rnd_uniform = default(rand, torch.rand((n_samples,)))
        truncate_p = st.norm.cdf((np.log(self.truncated_sigma_min) - self.p_mean) / self.p_std) * self.prob_length
        rnd_uniform = torch.clamp(rnd_uniform, min=truncate_p)

        blur_sigmas = self.blur_sigma_max * torch.sin(rnd_uniform * torch.pi / 2) ** 2

        rnd_interval_uniform = rnd_uniform * self.prob_length
        rnd_interval_normal = st.norm.ppf(rnd_interval_uniform)
        rnd_interval_normal = torch.tensor(rnd_interval_normal)
        
        sigma = (rnd_interval_normal * self.p_std + self.p_mean).exp()

        return sigma, blur_sigmas


class DiscreteSampling:
    def __init__(self, discretization_config, num_idx, do_append_zero=False, flip=True):
        self.num_idx = num_idx
        self.sigmas = instantiate_from_config(discretization_config)(
            num_idx, do_append_zero=do_append_zero, flip=flip
        )

    def idx_to_sigma(self, idx):
        return self.sigmas[idx]

    def __call__(self, n_samples, rand=None):
        idx = default(
            rand,
            torch.randint(0, self.num_idx, (n_samples,)),
        )
        return self.idx_to_sigma(idx)
