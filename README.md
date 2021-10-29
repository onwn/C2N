# C2N: Practical Generative Noise Modeling for Real-World Denoising - Official pyTorch release

This is an official PyTorch release of the paper
[**"C2N: Practical Generative Noise Modeling for Real-World Denoising"**](https://openaccess.thecvf.com/content/ICCV2021/papers/Jang_C2N_Practical_Generative_Noise_Modeling_for_Real-World_Denoising_ICCV_2021_paper.pdf)
from **ICCV 2021**.

![architecture](./imgs/architecture.png)

If you find C2N useful in your research, please cite our work as follows:

```
@InProceedings{Jang_2021_ICCV,
    author    = {Jang, Geonwoon and Lee, Wooseok and Son, Sanghyun and Lee, Kyoung Mu},
    title     = {C2N: Practical Generative Noise Modeling for Real-World Denoising},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {2350-2359}
}
```

[[PDF](https://openaccess.thecvf.com/content/ICCV2021/papers/Jang_C2N_Practical_Generative_Noise_Modeling_for_Real-World_Denoising_ICCV_2021_paper.pdf)]
[[Supp](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Jang_C2N_Practical_Generative_ICCV_2021_supplemental.pdf)]
[arXiv]

---

## Dependencies

- Python 3.9.6
- numpy >= 1.16.4
- cudatoolkit >= 10.
- PyTorch 1.2.0
- scikit-image 0.15.0
- tqdm
- pillow
- pyyamml
- imutils

You can manually setup an environment or follow below steps:

#### With Pyenv

```bash
pyenv install 3.9.6 && pyenv virtualenv 3.7.9 C2N
pyenv activate C2N
pip install -r requirements.txt
```

#### With Conda

```bash
conda create -f requirements.yml
```

<!-- #### Docker

```

``` -->

---

## Demo

### Noise generation

| Generator | Noisy | Clean |  config   | Pre-trained |
| :-------: | :---: | :---: | :-------: | :---------: |
|    C2N    | SIDD  | SIDD  | C2N_DnCNN |  [model]()  |
|    C2N    |  DND  | SIDD  | C2N_DnCNN |  [model]()  |

### Denoising

| Denoiser | Generator | Noisy | Clean | Clean (denoiser) |  config   | Pre-trained |
| :------: | :-------: | :---: | :---: | :--------------: | :-------: | :---------: |
|  DnCNN   |    C2N    | SIDD  | SIDD  |       SIDD       | C2N_DnCNN |  [model]()  |
|   DIDN   |    C2N    | SIDD  | SIDD  |       SIDD       | C2N_DIDN  |  [model]()  |
|  DnCNN   |    C2N    |  DND  | SIDD  |       DND        | C2N_DnCNN |  [model]()  |
|   DIDN   |    C2N    |  DND  | SIDD  |       DND        | C2N_DIDN  |  [model]()  |
