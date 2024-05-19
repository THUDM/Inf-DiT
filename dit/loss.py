import torch
from torch import nn
import torch.nn.functional as F

from dit.utils import instantiate_from_config, append_dims

class StandardDiffusionLoss(nn.Module):
    def __init__(
        self,
        sigma_sampler_config,
        weighting_config,
        loss_type="l2",
    ):
        super().__init__()
        
        assert loss_type in ["l2", "l1", "lpips"]
        
        self.sigma_sampler = instantiate_from_config(sigma_sampler_config)
        self.weighting = instantiate_from_config(weighting_config) if weighting_config else None
        
        self.loss_type = loss_type
        
        # if loss_type == "lpips":
        #     self.lpips = LPIPS().eval()
            
    
    def __call__(self, model, images, text_inputs=None):
        sigmas = self.sigma_sampler(images.shape[0]).to(images.dtype).to(images.device)
        noise = torch.randn_like(images).to(images.dtype)
        noised_images = images + noise * append_dims(sigmas, images.ndim)
        model_output = model(images=noised_images, sigmas=sigmas, text_inputs=text_inputs)
        w = append_dims(self.weighting(sigmas), images.ndim)
        return self.get_loss(model_output, images, w)
        
    def get_loss(self, model_output, target, w):
        if self.loss_type == "l2":
            return torch.mean(
                (w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1
            )
        elif self.loss_type == "l1":
            return torch.mean(
                (w * (model_output - target).abs()).reshape(target.shape[0], -1), 1
            )
        # elif self.loss_type == "lpips":
        #     loss = self.lpips(model_output, target).reshape(-1)
        #     return loss


class ConcatSRLoss2(StandardDiffusionLoss):
    def __init__(
            self,
            sigma_sampler_config,
            weighting_config,
            up_scale=4,
            loss_type="l2",
            lr_image_every=False
    ):
        super().__init__(sigma_sampler_config, weighting_config, loss_type=loss_type)
        self.up_scale = up_scale
        self.lr_image_every = lr_image_every

    def __call__(self, model, images, text_inputs=None, rope_position_ids=None):
        concat_lr_imgs, lr_imgs, sr_imgs = images
        kwargs = {}
        if model.image_encoder:
            image_embedding = model.image_encoder(lr_imgs)
            kwargs["vector"] = image_embedding
        images = sr_imgs
        sigmas = self.sigma_sampler(images.shape[0]).to(images.dtype).to(images.device)
        noise = torch.randn_like(images).to(images.dtype)
        noised_images = images + noise * append_dims(sigmas, images.ndim)
        model_output = model(images=noised_images, sigmas=sigmas, text_inputs=text_inputs, concat_lr_imgs=concat_lr_imgs, lr_imgs=lr_imgs, **kwargs)
        w = append_dims(self.weighting(sigmas), images.ndim)
        return self.get_loss(model_output, images, w)
