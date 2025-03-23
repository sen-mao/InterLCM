<p align="center">
  <img src="assets/interlcm_logo.jpg" height=110>
</p>

## Official Implementations "InterLCM: Low-Quality Images as Intermediate States of Latent Consistency Models for Effective Blind Face Restoration" (ICLR'25)

[Project Page](https://sen-mao.github.io/InterLCM-Page/)

[//]: # ([![Hugging Face coming soon]&#40;https://img.shields.io/badge/Demo-%F0%9F%A4%97%20Hugging%20Face-blue&#41;]&#40;https://github.com/sen-mao/InterLCM&#41; )
[![arXiv](https://img.shields.io/badge/arXiv-InterLCM-<COLOR>.svg)](https://arxiv.org/abs/2502.02215) [![arXiv](https://img.shields.io/badge/paper-InterLCM-b31b1b.svg)](https://arxiv.org/pdf/2502.02215) ![Visitors](https://visitor-badge.laobi.icu/badge?page_id=sen-mao/InterLCM)  [![gradio](https://img.shields.io/badge/Demo-%F0%9F%A4%97%20Gradio-blue)](https://2lq8im394062.vicp.fun/)


[Senmao Li](https://sen-mao.github.io/)<sup>1,2*</sup>, [Kai Wang](https://wangkai930418.github.io/)<sup>2†</sup>, [Joost van de Weijer](https://scholar.google.com/citations?user=Gsw2iUEAAAAJ&hl=en)<sup>2</sup>, [Fahad Shahbaz Khan](https://sites.google.com/view/fahadkhans/home)<sup>3,4</sup>, [Chun-Le Guo](https://mmcheng.net/clguo/)<sup>1</sup>, [Shiqi Yang](https://www.shiqiyang.xyz/)<sup>5</sup>, [Yaxing Wang](https://scholar.google.es/citations?user=6CsB8k0AAAAJ&hl=en)<sup>1</sup>, [jian Yang](https://scholar.google.com.hk/citations?user=6CIDtZQAAAAJ)<sup>1</sup>, [Ming-Ming Cheng](https://mmcheng.net/cmm/)<sup>1</sup>  

1.VCIP,CS,Nankai University 2.Computer Vision Center, Universitat Autònoma de Barcelona 3.Mohamed bin Zayed University of AI 4.Linkoping University 5.SB Intuitions. SoftBank

*Work done during a research stay at Computer Vision Center, Universitat Autònoma de Barcelona

†The corresponding author.

<img src="assets/interlcm.jpg" width="800px"/>


:star: If InterLCM is helpful to your images or projects, please help star this repo. Thanks! :hugs: 


### Update
- **2025.03.21**: Release a Gradio demo! Try out the online demo!  [![gradio](https://img.shields.io/badge/Demo-%F0%9F%A4%97%20Gradio-blue)](https://2lq8im394062.vicp.fun/)
- **2025.01.15**: Release the [pre-trained models](https://github.com/sen-mao/InterLCM/releases/tag/v0.1.0) and inference code. 😀
- **2024.12.24**: This repo is created.

---

#### Gradio Demo [![gradio](https://img.shields.io/badge/Demo-%F0%9F%A4%97%20Gradio-blue)](https://2lq8im394062.vicp.fun/)

<img src="assets/interlcm_gradio.gif" width="800px"/>


#### Face Restoration (synthetic of CelebA-Test)

<img src="assets/face_restoration_result1.jpg" width="800px"/>

#### Face Restoration (real-world images from LFW-Test, WebPhoto-Test, and WIDER-Test)

<img src="assets/face_restoration_result2.jpg" width="800px"/>

#### Face Colorization

<img src="assets/colorization.jpg" width="800px"/>

[//]: # ()
[//]: # (#### Face Inpainting)

[//]: # ()
[//]: # (<img src="assets/inpainting_result1.png" width="400px"/> <img src="assets/inpainting_result2.png" width="400px"/>)



### Dependencies and Installation

```
# git clone this repository
git clone https://github.com/sen-mao/InterLCM.git
cd InterLCM


# create new anaconda env
conda create -n interlcm python=3.8 -y
conda activate interlcm


# install python dependencies
pip3 install -r requirements.txt
python basicsr/setup.py develop
conda install -c conda-forge dlib (only for face detection or cropping with dlib)
```

<!-- conda install -c conda-forge dlib -->

### Testing:

#### Download the InterLCM pretrained models (Visual Encoder and Spatial Encoder) from [[Releases](https://github.com/sen-mao/InterLCM/releases/tag/v0.1.0)|[Google Drive](https://drive.google.com/drive/folders/1iSYQ-cn6g7kcsTc2UkeGjEyrVmiEx_Ea?usp=sharing)] to the `weights/InterLCM` folder.

#### 1. 🧑🏻 Face Restoration

```
# For cropped and aligned faces (512x512) (3-step interlcm reconstruction)
python inference_InterLCM.py --has_aligned --num_inference_steps 4 \
                             --input_path inputs/cropped_faces \
                             --output_path results/cropped_faces
```

#### InterLCM enables 1-step⚡ BFR under non-extreme degradation conditions.
```
# For cropped and aligned faces (512x512) (1-step interlcm reconstruction)
python inference_InterLCM.py --has_aligned --num_inference_steps 2 \
                             --visual_encoder_path weights/InterLCM/visual_encoder_1step.pth \
                             --spatial_encoder_path weights/InterLCM/spatial_encoder_1step.pth \
                             --input_path inputs/cropped_faces \
                             --output_path results/cropped_faces
```

#### 2. 🖼️ Whole Image Enhancement

```
# For whole image
# Add '--bg_upsampler realesrgan' to enhance the background regions with Real-ESRGAN
# Add '--face_upsample' to further upsample restorated face with Real-ESRGAN
python inference_InterLCM.py --num_inference_steps 4 \
                             --input_path inputs/whole_imgs \
                             --output_path results/whole_imgs \
                             --bg_upsampler realesrgan
```

#### 3. 🎨 Face Colorization

```
# For cropped and aligned faces (512x512) (1-step interlcm colorization)
python inference_InterLCM.py --has_aligned --num_inference_steps 2 \
                             --visual_encoder_path weights/InterLCM/visual_encoder_colorization_1step.pth \
                             --spatial_encoder_path weights/InterLCM/spatial_encoder_colorization_1step.pth \
                             --input_path inputs/gray_faces \
                             --output_path results/gray_faces
```

---

### Training:

#### 1. 🧩 Preparing Dataset

- Download training dataset: [FFHQ](https://github.com/NVlabs/ffhq-dataset)
- Resize to 512 $\times$ 512 resolution

#### 2. 🚀 Training InterLCM

- Training Visual Encoder and Spatial Encoder:
  > python -m torch.distributed.launch --nproc_per_node=gpu_num --master_port=4323 basicsr/train.py -opt options/interlcm.yml --launcher pytorch
- Pre-trained Visual Encoder (`visual_encoder.pth`) and Spatial Encoder (`spatial_encoder.pth`) can be found in the folder of Releases v0.1.0: https://github.com/sen-mao/InterLCM/releases/tag/v0.1.0

---

### Citation
If our work is useful for your research, please consider citing:

    @inproceedings{li2025interlcm,
      title={InterLCM: Low-Quality Images as Intermediate States of Latent Consistency Models for Effective Blind Face Restoration},
      author={Li, Senmao and Wang, Kai and van de Weijer, Joost and Khan, Fahad Shahbaz and Guo, Chun-Le and Yang, Shiqi and Wang, Yaxing and Yang, Jian and Cheng, Ming-Ming},
      booktitle={ICLR},
      year={2025}
    }

### License

Licensed under a [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) for Non-commercial use only.
Any commercial use should get formal permission first.

### Acknowledgement

This project is based on [LCM](https://github.com/luosiallen/latent-consistency-model) and [CodeFormer](https://github.com/sczhou/CodeFormer). Some codes are brought from [StableSR](https://github.com/IceClear/StableSR). Thanks for their awesome works.

### Contact
If you have any questions, please feel free to reach out to me at  `senmaonk@gmail.com`. 
