import argparse
import torch
from torchvision.utils import make_grid

from sat import mpu, get_args
from sat.training.deepspeed_training import training_main

import os
from functools import partial
from PIL import Image
import numpy as np

from dit.model import DiffusionEngine
from dit.utils import get_obj_from_str
    
def save_texts(texts, save_dir, iterations):
    output_path = os.path.join(save_dir, f"{str(iterations).zfill(8)}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(text + '\n')

def save_images(data, save_dir, key, iterations):
    images_data = ((data.to(torch.float64) + 1) * 127.5).clip(0, 255)
    images_data = images_data.detach().cpu()
    grid = make_grid(images_data).permute(1, 2, 0)
    grid = grid.numpy().astype(np.uint8)
    Image.fromarray(grid).save(os.path.join(save_dir, f"{key}_{str(iterations).zfill(8)}.png"))

def forward_step_eval(data_iterator, model, args, timers, data_class=None):
    # Get the batch.
    timers('batch generator').start()
    concat_lr_imgs, lr_imgs, sr_imgs, texts = data_class.get_batch(
        data_iterator, model.text_encoder, args, timers, eval=True, dropout=0)
    timers('batch generator').stop()

    # sampling test
    if torch.distributed.get_rank() == 0:
        image_save_dir = os.path.join(args.save, "images")
        text_save_dir = os.path.join(args.save, "texts")
        os.makedirs(image_save_dir, exist_ok=True)
        os.makedirs(text_save_dir, exist_ok=True)

        save_texts(texts, text_save_dir, args.iteration)

        save_images(lr_imgs, image_save_dir, "low_lr_inputs", args.iteration)
        save_images(concat_lr_imgs, image_save_dir, "lr_inputs", args.iteration)
        samples = model.sample(shape=sr_imgs.shape,
                               lr_imgs=lr_imgs,
                               images=concat_lr_imgs, dtype=sr_imgs.dtype, device=sr_imgs.device)

        save_images(sr_imgs, image_save_dir, "inputs", args.iteration)
        save_images(samples, image_save_dir, "samples", args.iteration)
    torch.distributed.barrier()
    
    loss = model.loss_func(model, [concat_lr_imgs, lr_imgs, sr_imgs], texts)
    return loss.mean(), {}
    
def forward_step(data_iterator, model, args, timers, data_class=None):
    # Get the batch.
    timers('batch generator').start()
    concat_lr_imgs, lr_imgs, sr_imgs, txt = data_class.get_batch(
        data_iterator, model.text_encoder, args, timers, dropout=args.lr_dropout)

    timers('batch generator').stop()
    
    loss = model.loss_func(model, [concat_lr_imgs, lr_imgs, sr_imgs], txt)
    return loss.mean()
    
if __name__ == '__main__':
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser = DiffusionEngine.add_model_specific_args(py_parser)
    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    data_config = DiffusionEngine.get_data_config(args)
    data_class = get_obj_from_str(data_config.get("target", "dit.data.Text2ImageSRWebDataset"))
    create_data_params = data_config.get("params", {})
    create_data_params['patch_size'] = args.patch_size
    create_dataset_function = partial(data_class.create_dataset_function, **create_data_params)
    training_main(args, model_cls=DiffusionEngine,
        forward_step_function=partial(forward_step, data_class=data_class), 
        forward_step_eval=partial(forward_step_eval, data_class=data_class), 
        create_dataset_function=create_dataset_function)
    