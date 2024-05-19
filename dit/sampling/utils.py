import torch
from scipy import integrate

from dit.utils import append_dims


class NoDynamicThresholding:
    def __call__(self, uncond, cond, scale):
        return uncond + scale * (cond - uncond)

class StaticThresholding:
    def __call__(self, uncond, cond, scale):
        x = uncond + scale * (cond - uncond)
        x = x.clip(-1, 1)
        print(x.max(), x.min())
        return x

def dynamic_threshold(x, p=0.995):
    p = 0.95
    N, C, H, W = x.shape
    x = x.float()
    x = x.view(N, C, H * W)
    l, r = x.quantile(q=torch.tensor([1 - p, p], device=x.device), dim=-1, keepdim=True)
    s = torch.maximum(-l, r)
    threshold_mask = (s > 1).expand(-1, -1, H * W)
    if threshold_mask.any():
        x = torch.where(threshold_mask, (x / s).clamp(min=-1, max=1), x)
        # x[threshold_mask].clamp_(min=-s, max=s).mul_(1./s)
    return x.view(N, C, H, W)

class DynamicThresholding:
    def __call__(self, uncond, cond, scale):
        x = uncond + scale * (cond - uncond)
        x = dynamic_threshold(x)
        x = x.to(dtype=uncond.dtype)
        return x


def linear_multistep_coeff(order, t, i, j, epsrel=1e-4):
    if order - 1 > i:
        raise ValueError(f"Order {order} too high for step {i}")

    def fn(tau):
        prod = 1.0
        for k in range(order):
            if j == k:
                continue
            prod *= (tau - t[i - k]) / (t[i - j] - t[i - k])
        return prod

    return integrate.quad(fn, t[i], t[i + 1], epsrel=epsrel)[0]


def get_ancestral_step(sigma_from, sigma_to, eta=1.0):
    if not eta:
        return sigma_to, 0.0
    sigma_up = torch.minimum(
        sigma_to,
        eta
        * (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5,
    )
    sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
    return sigma_down, sigma_up


def to_d(x, sigma, denoised):
    return (x - denoised) / append_dims(sigma, x.ndim)


def to_neg_log_sigma(sigma):
    return sigma.log().neg()


def to_sigma(neg_log_sigma):
    return neg_log_sigma.neg().exp()
