import os
import re
import time
from tqdm import tqdm
from tqdm.contrib import tzip
import pickle
import json
import numpy as np
import torch
import PIL.Image
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import random
import scipy.stats as st
from math import e

from sat.data_utils import make_loaders
import argparse
from sat import get_args
from sat.model.base_model import get_model
from dit.model import DiffusionEngine
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
import webdataset as wds
import io
import cv2


def read_from_cli():
    try:
        while True:
            x = input('Please input low resolution image path (Ctrl-D quit): ').strip()
            image = Image.open(x).convert('RGB')
            # to tensor
            image = transforms.ToTensor()(image) * 2 - 1
            image = image.unsqueeze(0)
            yield image, x.split('/')[-1].split('.')[0]
    except EOFError as e:
        pass


def read_from_file(p):
    if p.endswith('.txt'):
        with open(p, 'r') as fin:
            lines = fin.readlines()
        for line in lines:
            if line.startswith("#"):
                continue
            line = line.strip().split(' ')
            if len(line) == 2:
                line, image_id = line
                image_id = int(image_id)
            else:
                line = line[0]
                image_id = 0
            image = Image.open(line).convert('RGB')
            if image_id == 1:
                image = image.crop((2, 2, 514, 514))
            elif image_id == 2:
                image = image.crop((516, 2, 1028, 514))
            yield image, line.split('/')[-1].split('.')[0]
    else:
        raise NotImplementedError

class ImageDirDataset(Dataset):
    def __init__(self, path, repeat=1):
        self.path = path
        self.images = []
        for root, dirs, files in os.walk(path):
            for file in files:
                l_file = file.lower()
                if l_file.endswith('.png') or l_file.endswith('.jpg') or l_file.endswith('.jpeg') or l_file.endswith('.bmp'):
                    self.images.append(os.path.join(root, file))
        self.repeat = repeat
        self.images.sort()
    def __len__(self):
        return len(self.images) * self.repeat
    def __getitem__(self, idx):
        count, idx = idx // len(self.images), idx % len(self.images)
        lr_path = os.path.join(self.path, self.images[idx])
        image = Image.open(lr_path).convert('RGB')
        image = transforms.ToTensor()(image) * 2 - 1
        image_name = self.images[idx].split('.')[0]
        if self.repeat > 1:
            image_name = f"{image_name}_{count}"
        return image, image_name#, hr_image

def preprocess(r):
    img_bytes = r['png'] if 'png' in r else r['jpg']
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    return img, r['__key__']

def read_from_path(p, _type):
    "datapath_webdataset"
    "datapath_dir"
    _, type = _type.split('_')
    if type == "webdataset":
        dataset = wds.WebDataset(p).map(preprocess)
    elif type == "dir":
        dataset = ImageDirDataset(p, 1)
    else:
        raise NotImplementedError
    dataloader = DataLoader(dataset, batch_size=args.inference_batch_size, shuffle=False, num_workers=4)
    return dataloader

def main(args, device=torch.device('cuda')):

    print(f'Loading network from "{args.network}"...')

    net = get_model(args, DiffusionEngine).to(device)

    if args.network is not None:
        data = torch.load(args.network, map_location='cpu')
        net.load_state_dict(data['module'], strict=False)
    print('Loading Fished!')

    if args.input_type == 'txt':
        data_iter = read_from_file(args.input_path)
    elif args.input_type == 'cli':
        data_iter = read_from_cli()
    elif args.input_type.startswith('datapath'):
        data_iter = read_from_path(args.input_path, args.input_type)

    rank = int(os.getenv('RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    print('rank:', rank, 'world_size:', world_size)
    inference_type = torch.bfloat16
    for index, [lr_image, image_name] in tqdm(enumerate(data_iter)):
        if index % world_size != rank:
            continue
        if args.lr_size != 0:
            lr_image = transforms.Resize(args.lr_size, interpolation=InterpolationMode.BICUBIC)(lr_image)
        if args.crop_size != 0:
            lr_image = lr_image[:, :, :args.crop_size, :args.crop_size]

        if args.round != 0:
            h, w = lr_image.shape[-2:]
            h = h//args.round*args.round
            w = w//args.round*args.round
            lr_image = transforms.CenterCrop((h, w))(lr_image)

        h, w = lr_image.shape[-2:]
        new_h = h*args.infer_sr_scale
        new_w = w*args.infer_sr_scale


        tmp_lr_image = transforms.functional.resize(lr_image, [new_h, new_w], interpolation=InterpolationMode.BICUBIC)


        concat_lr_image = torch.clip(tmp_lr_image, -1, 1).to(device).to(inference_type)
        if args.infer_sr_scale != 4:
            lr_image = transforms.Resize((h//2, w//2), interpolation=InterpolationMode.BICUBIC)(lr_image)

        lr_image = lr_image.to(device).to(inference_type)

        collect_attention = False
        ar = args.inference_type == 'ar'
        ar2 = args.inference_type == 'ar2'
        with torch.no_grad():
            if not collect_attention:
                samples = net.sample(shape=concat_lr_image.shape, images=concat_lr_image, lr_imgs=lr_image, dtype=concat_lr_image.dtype, device=device, init_noise=args.init_noise, do_concat=not args.no_concat)
            else:
                samples, attentions = net.sample(shape=concat_lr_image.shape, images=concat_lr_image, lr_imgs=lr_image,
                                     dtype=concat_lr_image.dtype, device=device, init_noise=args.init_noise,
                                     do_concat=not args.no_concat, return_attention_map=True, ar=ar, ar2=ar2, block_batch=args.block_batch)

        images_np = ((samples.to(torch.float64) + 1) * 127.5).clip(0, 255).detach().cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8)

        for i, image_np in enumerate(images_np):
            image_dir = args.out_dir
            text = image_name[i].split('/')[-1]
            os.makedirs(image_dir, exist_ok=True)
            image_path = os.path.join(image_dir, f'{text}_sr.png')
            print("save to", image_path)
            if image_np.shape[2] == 1:
                PIL.Image.fromarray(image_np[:, :, 0], 'L').save(image_path)
            else:
                PIL.Image.fromarray(image_np, 'RGB').save(image_path)


def add_sample_specific_args(parser):
    group = parser.add_argument_group('Sampling', 'Diffusion Sampling')
    group.add_argument('--network', type=str)
    group.add_argument('--input-path', type=str, default='input.txt')
    group.add_argument('--out-dir', type=str)
    group.add_argument('--input-type', type=str, help='Choose from ["cli", "txt"]')
    group.add_argument('--num-steps', type=int, default=18)
    group.add_argument('--lr_size', type=int, default=0)
    group.add_argument('--crop_size', type=int, default=0)
    group.add_argument('--inference-batch-size', type=int, default=1)
    group.add_argument('--round', type=int, default=0)
    group.add_argument('--init_noise', action='store_true')
    group.add_argument('--infer_sr_scale', type=int, default=4)
    group.add_argument('--no_concat', action='store_true')
    group.add_argument('--inference_type', type=str, default='full', choices=['full', 'ar', 'ar2'])
    group.add_argument('--block_batch', type=int, default=1)
    return parser


if __name__ == "__main__":
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser = DiffusionEngine.add_model_specific_args(py_parser)
    py_parser = add_sample_specific_args(py_parser)
    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    del args.deepspeed_config
    args = argparse.Namespace(**vars(args), **vars(known))
    main(args)


            