# Text-to-Image Generation for Human Portrait Analysis by Fine-tuning Stable Diffusion Model

This is the repo for seminar.

We would like to thank Suraj Patil et. al. for the release of [Stable Diffusion version 1.5]( https://huggingface.co/blog/stable_diffusion?fbclid=IwAR3FZXW-YT8R-jvveFrJaAZInD3qsmpe2h5rlQaJB9rLlyUat4BHd3bdT10) and Rombach et. al. for the publication of paper [Latent Diffusion]( https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf).

## Usage
First, we need to run the following cmd for installing all neccessary libaries:
`pip install -r requirement`
## Dataset
Our method is used for fine-tuning a subset of [Celeb-A Dialog](https://github.com/ziqihuangg/CelebA-Dialog) aims at generating human faces that match with the detailed description. 
The subset of the dataset has been reconstructed for fine-tuning process in this repo and can be downloaded in here: [Train Dataset](https://drive.google.com/file/d/1q79oxT7aes_en6nr99n4zngfCs6HnXz4/view?usp=sharing) and [Test Dataset](https://drive.google.com/file/d/11YaO5hPMRfYPtmSjwefkzpTXLIqFGC2E/view?usp=share_link)
## Train
The fine-tuning process can be done via running: `sh train.sh`
## Inference
The inference process can be done via running: `sh test.sh`
## Pre-trained model.
The pretrained model for Stable Diffusion version 1.5 will be automatically downloaded when using training process.
# Evaluation protocol
The results generated from Stable Diffusion before and after fine-tuning process can be downloaded in:

[Before fine-tuning results](https://drive.google.com/file/d/1YPn_s9sWGaB2GMupTtvkFVyYh8Z4YTfB/view?usp=sharing)

[After fine-tuning results](https://drive.google.com/file/d/1fm2UWNlbuwZ4BVtpr1-viX10fp3Q7EVL/view?usp=sharing)

To evaluate the FID score, we need to download the results above, change the link in the code and run cmd: `python fid.py`
