# Inf-DiT

[![arXiv](https://img.shields.io/badge/arXiv-2405.04312-b31b1b.svg)](https://arxiv.org/abs/2405.04312)[![Page Views Count](https://badges.toozhao.com/badges/01HXBVPE6J3YKGEWCFSBRAXFAK/blue.svg)](https://badges.toozhao.com/stats/01HXBVPE6J3YKGEWCFSBRAXFAK "Get your own page views count badge on badges.toozhao.com")

Official implementation of Inf-DiT: Upsampling Any-Resolution Image with Memory-Efficient Diffusion Transformer

![1715078130760](image/README/frontpage.png)

## 🆕 News

* **2024.07.01**: Inf-DiT has been accepted by ECCV2024!
* **2024.05.20**: This code and model weight is released.
* **2024.05.08**: This repo is released.

## ⏳ TODO

- [x] Code release
- [x] Model weight release
- [x] Complete the explanation for the inference code and hyperparameter
- [ ] Demo
- [ ] Comfyui

## 🔆 Abstract

Diffusion models have shown remarkable performance in image generation in recent years. However, due to a quadratic increase in memory during generating ultra-high-resolution images (e.g. 4096 × 4096), the resolution of generated images is often limited to 1024×1024. In this work, we propose a unidirectional block attention mechanism that can adaptively adjust the memory overhead during the inference process and handle global dependencies. Building on this module, we adopt the DiT structure for upsampling and develop an infinite super-resolution model capable of upsampling images of various shapes and resolutions. Comprehensive experiments show that our model achieves excellent performance in generating ultra-high-resolution images. Compared to commonly used UNet structures, our model can save more than 5× memory when generating 4096 × 4096 images.

## 📚 Model Inference
Model weights can be downloaded from [here](https://cloud.tsinghua.edu.cn/f/6e313f7e1236468e973b/?dl=1)

1. Download the model weights and put them in the 'ckpt'.
2. `bash generate_sr_big_cli.sh` and input the low resolution image path. 
3. You can change the "inference_type"(line 27 in generate_sr_big_cli.sh) to "ar"(parallel size=1), "ar2"(parallel size = block_batch(line 28)) or "full"(generate the entire image in one forward).

Hyperparameter explanation:
- `--input-type`: choose between cli and txt(each line is a low resolution image path).
- `--inference_type`: choose between "ar"(parallel size=1), "ar2"(parallel size = block_batch(line 28)) or "full"(generate the entire image in one forward).
- `--block_batch`: block parallel size, one forward will generate block_batch\*block_batch blocks. The current version requires that the image(after upsample) side length is divisible by block_batch * 128.
- `--image-size`: not used.
- `--out-dir`: output directory.
- `--infer_sr_scale`: the scale of the super-resolution, the current version only supports 2 and 4.

## 📚 Model Training 

As this is a large-scale pre-trained model that has undergone multiple restarts and data adjustments during training, we cannot guarantee that the training results can be reproduced, it is only for reference implementation. 

1. Prepare the dataset. We use [webdataset](https://github.com/webdataset/webdataset) to 
organize data. Only one key "jpg" is needed in webdataset. You can replace WDS_DIR in the train_text2image_sr_big_clip.sh with webdataset path.
2. train_text2image_sr_big_clip.sh and scripts/d[train_text2image_sr.py](train_text2image_sr.py)s_config_zero_clip.json contain the main hyperparameters, among which ds_config_zero_clip is a parameter related to DeepSpeed. 
3. Run train_text2image_sr_big_clip.sh with slurm or other distributed training tools.

## 🆚 Ultra-high-resolution generation Demo vs other methods

### （click to see the detail)

### vs DemoFusion

[<img src="image\README\vsdemofusion\woman.png" alt="woman" height="400px" />](https://imgsli.com/MjYxOTA1)                          [<img src="image\README\vsdemofusion\zoomin.png" alt="woman" height="400px"/>](https://imgsli.com/MjYyMDU2)

*Caption: A digital painting of a young goddess with flower and fruit adornments evoking symbolic metaphors.*

*Resolution:* $2048\times 2048$

### vs BSRGAN

[<img src="image\README\vsbsrgan\cat.png" alt="woman" height="400px" />](https://imgsli.com/MjYyMTE5)                          [<img src="image\README\vsbsrgan\zoomin.png" alt="cat" height="400px" />](https://imgsli.com/MjYyMTIx)

*Caption: The image depicts a concept art of Schrodinger's cat in a box with an abstract background of waves and particles in a dynamic composition.*

*Resolution:* $2048\times 2048$

### vs Patch-Super-Resolution(4096*4096)

[<img src="image\README\vspatch\man.png" alt="woman" height="400px" />](https://imgsli.com/MjYyMTI4)                          [<img src="image\README\vspatch\zoomin.png" alt="woman" height="400px" />](https://imgsli.com/MjYyMTMw)

*Caption: A portrait of a character in a scenic environment.*

*Resolution:* $4096\times 4096$

## 👀 Super-Resolution results

### （click to see the detail)

[<img src="image\README\sr1.png" alt="woman" style="zoom: 54%;" />](https://imgsli.com/MjYyMTU3)      

*Resolution:* $1920\times 1080$

[<img src="image\README\sr2.png" alt="woman" style="zoom: 50%;" />](https://imgsli.com/MjYyMTU5)    

*Resolution:* $1920\times 768$

## ⚙️ Setup

## 📖 Citation

Please cite us if our work is useful for your research.

```
@misc{yang2024infdit,
      title={Inf-DiT: Upsampling Any-Resolution Image with Memory-Efficient Diffusion Transformer}, 
      author={Zhuoyi Yang and Heyang Jiang and Wenyi Hong and Jiayan Teng and Wendi Zheng and Yuxiao Dong and Ming Ding and Jie Tang},
      year={2024},
      eprint={2405.04312},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## 📭 Contact

If you have any comments or questions, feel free to contact zhuoyiyang2000@gmail.com or jianghy0581@gmail.com.
