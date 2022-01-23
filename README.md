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
[[arXiv]()]

---

## Setup

### Dependencies

- Python 3.9.6
- numpy >= 1.16.4
- cudatoolkit >= 10. (if using GPU)
- PyTorch 1.2.0
- opencv-python
- scikit-image >= 0.15.0
- tqdm
- pillow
- pyyamml
- imutils

<!-- You can manually setup an environment or follow below steps with Pyenv:

```bash
pyenv install 3.9.6 && pyenv virtualenv 3.7.9 C2N
pyenv activate C2N
pip install -r requirements.txt
``` -->

### Data

You can place any custom images in `./data` and image datasets in subdirectory `./data/[name_of_dataset]`

For the SIDD and DND benchmark images, you can find them at [SIDD Benchmark](https://www.eecs.yorku.ca/~kamel/sidd/benchmark.php) and [DND Benchmark]().
Convert them into .png images and place them in each subdirectory.

### Pre-trained Models

Download following pre-trained models:

| Generator | Clean | Noisy |  config   | Pre-trained |
| :-------: | :---: | :---: | :-------: | :---------: |
|    C2N    | SIDD  | SIDD  | C2N_DnCNN |  [model](https://drive.google.com/file/d/1Cn0KptLHd8p6v4_72PMvjssZbzTmgN4Z/view?usp=sharing)  |
|    C2N    | SIDD  |  DND  | C2N_DnCNN |  [model](https://drive.google.com/file/d/1Ce2Z9Gz7YssiIFIgGmj86xwDDqjgl2-S/view?usp=sharing)  |

| Denoiser | Generator | Clean | Noisy | Clean (denoiser train) |  config   | Pre-trained |
| :------: | :-------: | :---: | :---: | :--------------: | :-------: | :---------: |
|  DnCNN   |    C2N    | SIDD  | SIDD  |       SIDD       | C2N_DnCNN |  [model](https://drive.google.com/file/d/1wxuhXwhHYVLiAuUvwqIBiOX8NcFzdWmN/view?usp=sharing)  |
|   DIDN   |    C2N    | SIDD  | SIDD  |       SIDD       | C2N_DIDN  |  [model](https://drive.google.com/file/d/12Q5zZp3l_sH4pofXJraZbiEtAQ-kCtWD/view?usp=sharing)  |
|   DIDN   |    C2N    | SIDD  |  DND  |       SIDD       | C2N_DIDN  |  [model](https://drive.google.com/file/d/1gZQ3mfhLlnN0FZD3lxZGiApY8nVJBPT-/view?usp=sharing)  |

---

## Demo (Quick start)

### test_generate.py:

- `config`: Name of the configuration.
- `ckpt`: Name of the checkpoint to load. Choose between 'C2N-SIDD_to_SIDD' and 'C2N-DND_to_SIDD' depending on the noisy images it is trained on.
- `mode`: 'single' or 'dataset'.
- `data`: Filename of clean image if `mode` is 'single', dataset of clean images if `mode` is 'dataset'.
- `gpu`: GPU id. Currently this demo only supports single-GPU or CPU device.

Examples:

```bash
# Generate on single clean image
python test_generate.py --ckpt C2N-SIDD_to_SIDD.ckpt --mode single --data clean_ex1.png --gpu 0
python test_generate.py --ckpt C2N-DND_to_SIDD.ckpt --mode single --data clean_ex2.png --gpu 0

# Generate on clean images in a dataset
python test_generate.py --ckpt C2N-SIDD_to_SIDD.ckpt --mode dataset --data SIDD_clean_examples --gpu 0
python test_generate.py --ckpt C2N-DND_to_SIDD.ckpt --mode dataset --data SIDD_clean_examples --gpu 0
```

### test_denoise.py:

- `config`: Name of the configuration. Choose between 'C2N_DnCNN' and 'C2N_DIDN' depending on the denoiser to be used.
- `ckpt`: Name of the checkpoint to load.
  - Name format: '[denoiser model]-[C2N train noisy set]_to_[C2N train clean set]-on\_[denoiser train set]'
    - denoiser model: 'DnCNN' or 'DIDN'.
    - C2N train noisy set: Dataset that the noisy images for C2N training are sampled from.
    - C2N train clean set: Dataset that the clean images for C2N training are sampled from.
    - denoiser train set: Dataset that the clean->noisy images are generated from, to train the denoiser.
- `mode`: 'single' or 'dataset'.
- `data`: Filename of noisy/generated image if `mode` is 'single', dataset of noisy/generated images if `mode` is 'dataset'.
- `gpu`: GPU id. Currently this demo only supports single-GPU or CPU device.

Examples:

```bash
# Denoise single noisy image
python test_denoise.py --config C2N_DnCNN --ckpt DnCNN-SIDD_to_SIDD-on_SIDD --mode single --data noisy_ex1_SIDD.png --gpu 0
python test_denoise.py --config C2N_DIDN --ckpt DIDN-SIDD_to_SIDD-on_SIDD --mode single --data noisy_ex1_SIDD.png --gpu 0
python test_denoise.py --config C2N_DIDN --ckpt DIDN-SIDD_to_DND-on_SIDD --mode single --data noisy_ex2_DND.png --gpu 0

# Denoise noisy images in a dataset
python test_denoise.py --config C2N_DnCNN --ckpt DnCNN-SIDD_to_SIDD-on_SIDD --mode dataset --data SIDD_benchmark --gpu 0
python test_denoise.py --config C2N_DIDN --ckpt DIDN-SIDD_to_SIDD-on_SIDD --mode dataset --data SIDD_benchmark --gpu 0
python test_denoise.py --config C2N_DIDN --ckpt DIDN-SIDD_to_DND-on_SIDD --mode dataset --data DND_benchmark --gpu 0

# Denoise the generated images from C2N
# You may copy the generated images in `results/[input_clean_data_path*]` to `data/[input_clean_data_path*]_generated.png`, for example.
python test_denoise.py --config C2N_DIDN --ckpt DIDN-SIDD_to_SIDD-on_SIDD --mode single --data clean_ex1_generated.png --gpu 0
python test_denoise.py --config C2N_DIDN --ckpt DIDN-SIDD_to_DND-on_SIDD --mode single --data clean_ex2_generated.png --gpu 0
```
