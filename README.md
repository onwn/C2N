# C2N: Practical Generative Noise Modeling for Real-World Denoising - Official pyTorch implementation

Geonwoon Jang*, Wooseok Lee*, Sanghyun Son, Kyoung Mu Lee

**Abstract:**

Learning-based image denoising methods have been bounded to situations where well-aligned noisy and clean images are given, or samples are synthesized from predetermined noise models, e.g., Gaussian. While recent generative noise modeling methods aim to simulate the unknown distribution of real-world noise, several limitations still exist. In a practical scenario, a noise generator should learn to simulate the general and complex noise distribution without using paired noisy and clean images. However, since existing methods are constructed on the unrealistic assumption of real-world noise, they tend to generate implausible patterns and cannot express complicated noise maps. Therefore, we introduce a Clean-to-Noisy image generation framework, namely C2N, to imitate complex real-world noise without using any paired examples. We construct the noise generator in C2N accordingly with each component of real-world noise characteristics to express a wide range of noise accurately. Combined with our C2N, conventional denoising CNNs can be trained to outperform existing unsupervised methods on challenging real-world benchmarks by a large margin.

![architecture](./imgs/architecture.png)

## Demo for noise generation

| Generator | Noisy | Clean | config | Pre-trained |
|:--:|:--:|:--:|:--:|:--:|
| C2N | SIDD | SIDD | C2N_DnCNN | [model]() |
| C2N | DND | SIDD | C2N_DnCNN | [model]() |

## Demo for denoising

| Denoiser | Generator | Noisy | Clean | Clean (denoiser) | config | Pre-trained |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| DnCNN | C2N | SIDD | SIDD | SIDD | C2N_DnCNN | [model]() |
| DIDN | C2N | SIDD | SIDD | SIDD | C2N_DIDN | [model]() |
| DnCNN | C2N | DND | SIDD | DND | C2N_DnCNN | [model]() |
| DIDN | C2N | DND | SIDD | DND | C2N_DIDN | [model]() |
