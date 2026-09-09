# AI Image Generation using Diffusion Models

BEng final year project, Department of Electrical and Electronic Engineering, Imperial College London.

A Denoising Diffusion Probabilistic Model (DDPM) implemented from scratch in PyTorch, used to ask a
single architectural question: **does combining residual learning with the UNet backbone improve the
quality of generated images?**

---

## Research question

Within a DDPM the diffusion process is fixed. The only learned component is the network that
predicts the noise added at each step, so the choice of that network governs the quality of the
images produced. This project holds the diffusion framework constant and varies only the denoising
network.

Three networks are trained on each of two datasets, giving six runs.

| Network | Description | What it answers |
|---|---|---|
| **ResUNet** | Proposed. Every encoder and decoder stage built from residual blocks. | — |
| **Plain UNet** | Identical, but the identity shortcut is removed from each block. | Does the shortcut help? |
| **DDPM UNet** | Reference architecture of Ho et al.: four resolution stages, self-attention at 16×16, a skip from every residual block. | Is the proposal competitive with the standard design? |

A shortcut adds no parameters, so ResUNet and Plain UNet have **exactly** the same parameter count
and computational cost; that comparison isolates residual learning as the single variable. The
reference network is narrowed from its published width of 128 base channels (~35M parameters) until
its parameter count matches to within a few per cent, so that any difference reflects design rather
than capacity.

All six runs share the number of diffusion steps, the noise schedule, the objective, the optimiser
and learning rate, the number of epochs, the augmentation and the random seed. Sampling for every
model starts from the same initial noise.

## Method

- **Forward process** — closed form, `x_t = sqrt(ᾱ_t)·x_0 + sqrt(1−ᾱ_t)·ε`
- **Objective** — noise prediction, `MSE(ε, ε_θ(x_t, t))`, with `t` drawn uniformly per example
- **Schedule** — cosine, defining `ᾱ_t` as `cos²` after Nichol & Dhariwal
- **Sampling** — ancestral, using the posterior variance `β̃_t`, with the predicted `x_0` clipped to
  `[-1, 1]` before the posterior mean is reconstructed

The clipping step is not optional. Writing the posterior mean directly as
`(x_t − β_t/sqrt(1−ᾱ_t)·ε_θ)/sqrt(α_t)` is algebraically correct only while the implied estimate of
`x_0` stays inside the data range. Since `1/sqrt(α_t)` reaches 31.6 at `t = 999` under the clipped
cosine schedule, any error in `ε_θ` is amplified by that factor and accumulates over the thousand
reverse steps; without clipping the reverse process diverges (standard deviation grows from 1 to
about 16, and over 95% of pixels leave the valid range) in both single and mixed precision.

## Datasets

| Dataset | Resolution | Training images | Purpose |
|---|---|---|---|
| CIFAR-10 | 32×32 | 50,000 | Standard diffusion benchmark; comparable with published results |
| CelebA | 64×64 (centre crop, resized) | 50,000 subset | Architectural differences are hard to see at 32×32, where only two downsampling stages fit |

## Evaluation

Measurements fall into three groups that answer different questions.

- **Restoration** — SSIM and PSNR between held-out test images and their single-step
  reconstructions at six noise levels. These need a reference image, so they measure denoising, not
  generation.
- **Distribution** — FID and Inception Score, computed from samples drawn through the full reverse
  process. These need no reference and are the primary measure of generative quality. *(IS is read
  for CIFAR-10 only; CelebA is a single semantic category, so class diversity is undefined.)*
- **Validity** — nearest-neighbour distance from each generated sample to the training set, against
  two controls: real held-out images, and the same images blurred. The blur control separates
  "close because copied" from "close because smooth".

Negative Log-Likelihood is deliberately excluded: obtaining it requires the full variational bound,
which is dominated by the reverse-process variances, and this implementation fixes those variances
rather than learning them.

## Results

30 epochs per run on an RTX 4070, `T = 1000`.

| Dataset | Network | Parameters | Loss (epoch 1) | Loss (epoch 30) | SSIM | PSNR (dB) |
|---|---|---|---|---|---|---|
| CIFAR-10 | ResUNet | 2,473,475 | **0.13775** | 0.06108 | 0.7388 | 24.71 |
| CIFAR-10 | Plain UNet | 2,473,475 | 0.14633 | 0.06084 | 0.7373 | 24.70 |
| CIFAR-10 | DDPM UNet | 2,356,739 | 0.18508 | **0.05967** | **0.7449** | **24.83** |
| CelebA | ResUNet | 2,612,419 | **0.14205** | 0.03716 | 0.8016 | 27.10 |
| CelebA | Plain UNet | 2,612,419 | 0.16413 | 0.03679 | 0.8004 | 27.11 |
| CelebA | DDPM UNet | 2,356,739 | 0.15588 | **0.03476** | **0.8174** | **27.42** |

Three findings:

1. **Residual shortcuts accelerate early optimisation.** After one epoch the proposed network leads
   its shortcut-free twin by 5.9% on CIFAR-10 and 13.5% on CelebA. The two networks are otherwise
   identical, so the shortcut is the only possible cause.
2. **The advantage does not persist.** Plain UNet overtakes by the fifth epoch and finishes
   marginally ahead on loss, while ResUNet finishes marginally ahead on SSIM and level on PSNR. All
   three margins are below half a per cent and they do not agree, so no benefit from the shortcut
   can be demonstrated at convergence.
3. **Depth and attention matter more than the shortcut.** The reference network leads on every
   measurement, on both datasets, using fewer parameters — at roughly 2.3× the training cost per
   epoch.

The hypothesis that combining residual learning with a UNet backbone would improve generated image
quality is **not supported** by these measurements.

> FID and Inception Score figures are omitted here pending a run at the full sample size. Values
> computed from a small sample are unreliable: the Inception feature space has 2048 dimensions, so
> fewer samples than that leave the covariance term degenerate.

## Repository layout

```
imagegeneration.ipynb        # everything: models, training, sampling, evaluation
requirements.txt             # pinned environment (CUDA 13.0 wheels)
generated_samples/           # 64 samples per model, identical initial noise
denoising_process/           # reverse-trajectory animations
dissertation/main.pdf        # compiled thesis
model/                       # checkpoints (not tracked)
data/                        # datasets and tensor caches (not tracked)
```

## Running

```bash
python -m venv .diffusion
.diffusion\Scripts\activate          # PowerShell: .\.diffusion\Scripts\Activate.ps1
pip install -r requirements.txt
```

Open `imagegeneration.ipynb`, select the `.diffusion` interpreter as the kernel, and run all cells.

Notes:

- `requirements.txt` carries an `--extra-index-url` line for the CUDA wheels; `torch==2.14.0+cu130`
  is a local version tag that does not exist on PyPI.
- CelebA is fetched from Google Drive and needs `gdown`, which is pinned in the requirements.
- Datasets are decoded once into `uint8` tensor caches under `data/cache/`, so only the first run
  pays the preprocessing cost.
- The training cell skips any architecture whose checkpoint already records the full number of
  epochs, so a kernel restart does not trigger retraining.
- `self_check()` verifies the schedule, the closed-form forward process, timestep conditioning, the
  sampling path and the fairness of the parameter budgets. Run it before training.

## References

1. J. Ho, A. Jain and P. Abbeel, *Denoising Diffusion Probabilistic Models*, arXiv:2006.11239.
2. A. Nichol and P. Dhariwal, *Improved Denoising Diffusion Probabilistic Models*, arXiv:2102.09672.
3. Y. Song et al., *Score-Based Generative Modeling through Stochastic Differential Equations*,
   arXiv:2011.13456.
4. K. He, X. Zhang, S. Ren and J. Sun, *Deep Residual Learning for Image Recognition*,
   arXiv:1512.03385.
5. L. Yang et al., *Diffusion Models: A Comprehensive Survey of Methods and Applications*,
   arXiv:2209.00796.
