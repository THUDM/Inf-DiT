import os
from functools import partial

import torch

from dit.utils import default, instantiate_from_config

from PIL import Image
class VanillaCFG:
    """
    implements parallelized CFG
    """

    def __init__(self, scale, dyn_thresh_config=None):
        scale_schedule = lambda scale, sigma: scale  # independent of step
        self.scale_schedule = partial(scale_schedule, scale)
        self.dyn_thresh = instantiate_from_config(
            default(
                dyn_thresh_config,
                {
                    "target": "dit.sampling.utils.NoDynamicThresholding"
                },
            )
        )


    def __call__(self, x, sigma):
        x_u, x_c = x.chunk(2)
        scale_value = self.scale_schedule(sigma)
        x_pred = self.dyn_thresh(x_u, x_c, scale_value)
        return x_pred

    def prepare_inputs(self, x, s, c, uc, rope_position_ids):
        c_out = dict()

        for k in c:
            c_out[k] = torch.cat((uc[k], c[k]), 0)
        
        if rope_position_ids is not None:
            rope_position_ids = torch.cat([rope_position_ids] * 2)
        
        return torch.cat([x] * 2), torch.cat([s] * 2), c_out, rope_position_ids


class IdentityGuider:
    def __call__(self, x, sigma):
        return x

    def prepare_inputs(self, x, s, c, uc, rope_position_ids):
        c_out = dict()

        for k in c:
            c_out[k] = c[k]

        return x, s, c_out, rope_position_ids
