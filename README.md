# Inf-DiT

[![arXiv](https://img.shields.io/badge/arXiv-2405.04312-b31b1b.svg)](https://arxiv.org/abs/2405.04312)[![Page Views Count](https://badges.toozhao.com/badges/01HXBVPE6J3YKGEWCFSBRAXFAK/blue.svg)](https://badges.toozhao.com/stats/01HXBVPE6J3YKGEWCFSBRAXFAK "Get your own page views count badge on badges.toozhao.com")

Official implementation of Inf-DiT: Upsampling Any-Resolution Image with Memory-Efficient Diffusion Transformer

![1715078130760](image/README/frontpage.png)

## 🆕 News

* **2024.05.08**: This repo is released.

## ⏳ TODO

- [ ] Code(coming soon!!!)
- [ ] Demo

## 🔆 Abstract

Diffusion models have shown remarkable performance in image generation in recent years. However, due to a quadratic increase in memory during generating ultra-high-resolution images (e.g. 4096 × 4096), the resolution of generated images is often limited to 1024×1024. In this work, we propose a unidirectional block attention mechanism that can adaptively adjust the memory overhead during the inference process and handle global dependencies. Building on this module, we adopt the DiT structure for upsampling and develop an infinite super-resolution model capable of upsampling images of various shapes and resolutions. Comprehensive experiments show that our model achieves excellent performance in generating ultra-high-resolution images. Compared to commonly used UNet structures, our model can save more than 5× memory when generating 4096 × 4096 images.

## 👀 Super-Resolution results

### （click to see the detail)

[<img src="image\README\sr1.png" alt="woman" style="zoom: 54%;" />](https://imgsli.com/MjYyMTU3)      

[<img src="image\README\sr2.png" alt="woman" style="zoom: 50%;" />](https://imgsli.com/MjYyMTU5)      



## 🆚 Ultra-high-resolution generation Demo vs other methods

### （click to see the detail)

### vs DemoFusion(2048*2048)

[<img src="image\README\vsdemofusion\woman.png" alt="woman" style="zoom: 33%;" />](https://imgsli.com/MjYxOTA1)                                      [<img src="image\README\vsdemofusion\zoomin.png" alt="woman" style="zoom: 33%;" />](https://imgsli.com/MjYyMDU2)

### vs BSRGAN(2048*2048)

[<img src="image\README\vsbsrgan\cat.png" alt="woman" style="zoom: 33%;" />](https://imgsli.com/MjYyMTE5)                                      [<img src="image\README\vsbsrgan\zoomin.png" alt="cat" style="zoom: 33%;" />](https://imgsli.com/MjYyMTIx)

### vs Patch-Super-Resolution(4096*4096)

[<img src="image\README\vspatch\man.png" alt="woman" style="zoom: 33%;" />](https://imgsli.com/MjYyMTI4)                                      [<img src="image\README\vspatch\zoomin.png" alt="woman" style="zoom: 33%;" />](https://imgsli.com/MjYyMTMw)

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

If you have any comments or questions, feel free to contact [Zhuoyi Yang](zhuoyiyang2000@gmail.com) or [Heyang Jiang](jianghy0581@gmail.com).