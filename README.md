# HFC-and-HFF
[Are High-Frequency Components Beneficial for Training of Generative Adversarial Networks](https://arxiv.org/abs/2103.11093)
## Abstract 

Advancements in Generative Adversarial Networks (GANs) have the ability to generate realistic images that are visually indistinguishable from real images. However, recent studies of the image spectrum have demonstrated that generated and real images share significant differences at high frequency. Furthermore, the high-frequency components invisible to human eyes affect the decision of CNNs and are related to the robustness of it. Similarly, whether the discriminator will be sensitive to the high-frequency differences, thus reducing the fitting ability of the generator to the low-frequency components is an open problem. In this paper, we demonstrate that the discriminator in GANs is sensitive to such high-frequency differences that can not be distinguished by humans and the high-frequency components of images are not conducive to the training of GANs. Based on these, we propose two preprocessing methods eliminating high-frequency differences in GANs training: High-Frequency Confusion (HFC) and High-Frequency Filter (HFF). The proposed methods are general and can be easily applied to most existing GANs frameworks with a fraction of the cost. The advanced performance of the proposed method is verified on multiple loss functions, network architectures, and datasets. 

-----

## Installation
The library is forked by [mimicry](https://github.com/kwotsin/mimicry), a lightweight PyTorch library aimed towards the reproducibility of GAN research.
We need to install the following packages before running the code.
```
conda install pytorch torchvision cudatoolkit -c pytorch
```

```
conda install tensorflow
```
```
pip install torch-mimicry
```

## Training (example)

run SNGAN:

```
python sngan_example.py
```
## Tips
You need to replace the code in the package **torch_mimicry** in your environment with the code from this respository.

## HFF
```python
def generateDataWithDifferentFrequencies_3Channel(image,r):
    # HFF
    mask=mask_radial(64,32,32,r)
    mask=torch.from_numpy(mask).cuda()
    image=image.mul(0.5).add(0.5)
    img_fft = torch.fft.fft2(image)
    img_fft_center = torch.fft.fftshift(img_fft)
    fd_low=img_fft_center*mask
    img_low_center = torch.fft.ifftshift(fd_low)
    img_low = torch.fft.ifft2(img_low_center)
    img_low=torch.abs(img_low).float()
    img_low=torch.clip(img_low,0,1)
    img_low=img_low.add(-0.5).mul(2)
    return img_low
```
## HFC
```python
def generateDataWithDifferentFrequencies_exchange(real,fake,r):
    # HFC
    mask=mask_radial(64,32,32,r)
    mask=torch.from_numpy(mask).cuda()
    real=real.mul(0.5).add(0.5)
    real_fft = torch.fft.fft2(real)
    real_fft_center = torch.fft.fftshift(real_fft)
    real_fd_low=real_fft_center*mask
    real_fd_high=real_fft_center*(1-mask)

    fake=fake.mul(0.5).add(0.5)
    fake_fft = torch.fft.fft2(fake)
    fake_fft_center = torch.fft.fftshift(fake_fft)
    fake_fd_low=fake_fft_center*mask
    fake_fd_high=fake_fft_center*(1-mask)

    real_main=real_fd_low+real_fd_high
    fake_main=real_fd_high+fake_fd_low


    real_main = torch.fft.ifftshift(real_main)
    real_main = torch.fft.ifft2(real_main)
    real_main=torch.abs(real_main).float()
    real_main=torch.clip(real_main,0,1)
    real_main=real_main.add(-0.5).mul(2)

    fake_main = torch.fft.ifftshift(fake_main)
    fake_main = torch.fft.ifft2(fake_main)
    fake_main=torch.abs(fake_main).float()
    fake_main=torch.clip(fake_main,0,1)
    fake_main=fake_main.add(-0.5).mul(2)
    return real_main,fake_main
```

## Citation
If you have found this work useful, please consider citing [our work](https://arxiv.org/abs/2103.11093):
```
@misc{li2021highfrequency,
      title={Are High-Frequency Components Beneficial for Training of Generative Adversarial Networks}, 
      author={Ziqiang Li and Pengfei Xia and Xue Rui and Yanghui Hu and Bin Li},
      year={2021},
      eprint={2103.11093},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## References
[[1] Spectral Normalization for Generative Adversarial Networks](https://arxiv.org/abs/1802.05957)

[[2] cGANs with Projection Discriminator](https://arxiv.org/abs/1802.05637)

[[3] Self-Supervised GANs via Auxiliary Rotation Loss](https://arxiv.org/abs/1811.11212)

[[4] A Large-Scale Study on Regularization and Normalization in GANs](https://arxiv.org/abs/1807.04720)

[[5] InfoMax-GAN: Improved Adversarial Image Generation via Information Maximization and Contrastive Learning](https://arxiv.org/abs/2007.04589)

[[6] GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/abs/1706.08500)