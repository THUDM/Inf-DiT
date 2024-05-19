import io
import os
import sys
import numpy as np
from PIL import Image
from functools import partial
import math

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from torchvision.transforms.functional import InterpolationMode
from sat import mpu
from sat.data_utils.webds import MetaDistributedWebDataset
from dit.image_degradation import degradation_bsrgan_variant
from dit.realesr import RealESRGANTransform

def pil_resize(img, image_size):
    width, height = img.size
    scale = image_size / min(width, height)
    img1 = img.resize(
        (int(round(scale * width)), int(round(scale * height))),
        resample=Image.Resampling.BICUBIC, # previously BOX
    )
    return img1

def pil_to_np(img):
    return np.asarray(img).astype(np.float32)

def np_centercrop(arr):
    h, w, _ = arr.shape
    image_size = min(h, w)
    h_off = (h - image_size) // 2
    w_off = (w - image_size) // 2
    arr = arr[h_off : h_off + image_size, w_off : w_off + image_size]
    return arr

def resize_centercrop(img, image_size): # PIL Image
    return np_centercrop(pil_to_np(pil_resize(img, image_size))).transpose(2, 0, 1)

def process_fn_sr_text2image(src, lr_size, sr_size, patch_size, extra_texts, filters, norm_transformer, transformer=None, do_degradation=False, do_realesr=False, only_degradation=False, realesr_transform=None, sft=False, sdedit=False):
    pass_num, total_num = 0, 0

    for r in src:
        if ('png' not in r and 'jpg' not in r):
            continue

        # go through filters
        total_num += 1
        # filter_flag = 0
        # for key, filter in filters.items():
        #     default_score = -float('inf') if filter["greater"] else float('inf')
        #     score = r.get(key, default_score) or default_score
        #     judge = (lambda a: a > filter["val"]) if filter["greater"] else (lambda a: a < filter["val"])
        #     if not judge(score):
        #         filter_flag = 1
        #         break
        # if filter_flag:
        #     continue
        pass_num += 1

        img_bytes = r['png'] if 'png' in r else r['jpg']
        try:
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        except Exception as e:
            print(e)
            continue
        h, w = img.size
        if h < sr_size or w < sr_size:
            continue

        if sft:
            chosen_transformer = norm_transformer
            # if h < 1024 or w < 1024:
            #     continue
            # sft_transformer = [transforms.Resize(1024, interpolation=InterpolationMode.BICUBIC),
            #                     transforms.RandomCrop(512), ]
            # chosen_transformer = transforms.Compose(sft_transformer)
        elif random.random() < 0.7 and transformer is not None:
            chosen_transformer = transformer
        else:
            chosen_transformer = norm_transformer
        img = chosen_transformer(img)

        sr_img = np.array(img)

        if do_degradation and (only_degradation or random.random() < 0.8):
            if do_realesr:
                lr_img, sr_img = realesr_transform(sr_img)
            else:
                lr_img = degradation_bsrgan_variant(sr_img, sr_size//lr_size)['image']
        else:
            lr_img = transforms.functional.resize(img, lr_size, interpolation=InterpolationMode.BICUBIC)
            lr_img = np.array(lr_img)


        sr_img = (sr_img / 127.5 - 1.0).astype(np.float32)
        lr_img = (lr_img / 127.5 - 1.0).astype(np.float32)
        sr_img = sr_img.transpose(2, 0, 1)
        lr_img = lr_img.transpose(2, 0, 1)


        sr_img = torch.from_numpy(sr_img)
        lr_img = torch.from_numpy(lr_img)

        concat_lr_img = transforms.functional.resize(lr_img, [sr_size, sr_size], interpolation=InterpolationMode.BICUBIC)
        concat_lr_img = torch.clip(concat_lr_img, -1, 1)

        yield "txt", lr_img, concat_lr_img, sr_img

import cv2
import random
from torchvision import transforms
def add_blur(img):
    result = cv2.GaussianBlur(img, (3, 3), 0.4 + random.random()/5)
    return result

class Text2ImageSRWebDataset(MetaDistributedWebDataset):
    def __init__(self, path, lr_size, sr_size, patch_size, random_crop=False, do_degradation=False,
                 do_realesr=False, only_degradation=False, filters={}, extra_texts={}, nshards=sys.maxsize,
                 shuffle_buffer=1000, include_dirs=None, realesr_config="configs/degradation.yaml", sft=False, sdedit=False, **kwargs):
        # get a random seed
        seed = random.randint(0, 1000000)
        # seed = int(os.environ.get("PL_GLOBAL_SEED", '0'))
        meta_names = []
        for key in filters.keys():
            meta_names.append(key)
        for key in extra_texts.keys():
            meta_names.append(key)

        print("real esr config", realesr_config)
        if random_crop:
            transformer = [transforms.RandomHorizontalFlip(p=0.3),
                       transforms.RandomCrop(sr_size)]
            self.transformer = transforms.Compose(transformer)
        else:
            self.transformer = None

        norm_transformer = [transforms.Resize(sr_size, interpolation=InterpolationMode.BICUBIC),
                            transforms.RandomCrop(sr_size), ]
        self.norm_transformer = transforms.Compose(norm_transformer)

        realesr_transform = RealESRGANTransform(realesr_config)
        if do_degradation:
            print("do_degradation")
            if only_degradation:
                print("only_degradation")
            if do_realesr:
                print("do_realesr")

        super().__init__(
            path,
            partial(process_fn_sr_text2image, lr_size=lr_size, sr_size=sr_size, patch_size=patch_size,
                    do_degradation=do_degradation,
                    do_realesr=do_realesr,
                    only_degradation=only_degradation,
                    realesr_transform=realesr_transform,
                    transformer=self.transformer,
                    norm_transformer=self.norm_transformer,
                    filters=filters,
                    extra_texts=extra_texts,
                    sft=sft,
                    sdedit=sdedit),
            seed,
            meta_names=meta_names,
            shuffle_buffer=shuffle_buffer,
            nshards=nshards,
            include_dirs=include_dirs
        )

    @classmethod
    def create_dataset_function(cls, path, args, **kwargs):
        path, include_dirs = path.split(';', 1)
        if len(include_dirs) == 0:
            include_dirs = None
        return cls(path, include_dirs=include_dirs, **kwargs)

    @classmethod
    def get_batch(cls, data_iterator, text_encoder, args, timers, dropout=0., eval=False):
        # Broadcast data.
        timers('data loader').start()
        if data_iterator is not None:
            data = next(data_iterator)
        else:
            data = None
        timers('data loader').stop()

        texts, lr_imgs, concat_lr_imgs, sr_imgs = data

        batch_size = len(texts)
        if not eval:
            for i in range(batch_size):
                if np.random.random() < dropout:
                    lr_imgs[i] = torch.randn_like(lr_imgs[i]).clamp(-1, 1)
                    concat_lr_imgs[i] = torch.randn_like(concat_lr_imgs[i]).clamp(-1, 1)

        images_lr = mpu.broadcast_data(["images_lr"], {"images_lr": lr_imgs}, torch.float32)
        images_sr = mpu.broadcast_data(["images_sr"], {"images_sr": sr_imgs}, torch.float32)
        images_concat_lr = mpu.broadcast_data(["images_concat_lr"], {"images_concat_lr": concat_lr_imgs}, torch.float32)
        images_data_lr = images_lr["images_lr"]
        images_data_sr = images_sr["images_sr"]
        image_data_concat_lr = images_concat_lr["images_concat_lr"]


        texts = [text if np.random.random() > dropout else "" for text in texts]

        # Convert
        if args.fp16:
            images_data_lr = images_data_lr.half()
            images_data_sr = images_data_sr.half()
            image_data_concat_lr = image_data_concat_lr.half()
        elif args.bf16:
            images_data_lr = images_data_lr.to(torch.bfloat16)
            images_data_sr = images_data_sr.to(torch.bfloat16)
            image_data_concat_lr = image_data_concat_lr.to(torch.bfloat16)
        if eval:
            return image_data_concat_lr, images_data_lr, images_data_sr, texts
        return image_data_concat_lr, images_data_lr, images_data_sr, texts