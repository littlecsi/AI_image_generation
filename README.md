# AI Image Generation using Diffusion Models

BEng final year project, Department of Electrical and Electronic Engineering, Imperial College London.

A Denoising Diffusion Probabilistic Model (DDPM) implemented from scratch in PyTorch, used to ask a
single architectural question: **does combining residual learning with the UNet backbone improve the
quality of generated images?**

The project has two parts:

| Part | Notebook | Thesis | Question |
|---|---|---|---|
| Architecture study | `imagegeneration.ipynb` | Chapter 3 | Which denoising network generates best under identical training? |
| [Extension](#extension-sharpening-the-samples) | `DissertationExtension.ipynb` | Chapter 4 | How much of the gap to the published result can training alone close? |

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

## Results (architecture study)

30 epochs per run on an RTX 4070, `T = 1000`.

Every configuration was trained under **three random seeds** (0, 1, 2). Figures are the mean over
the three runs; comparisons between architectures are paired within seed.

| Dataset | Network | Parameters | Loss (ep. 30) | SSIM | PSNR (dB) | FID ↓ |
|---|---|---|---|---|---|---|
| CIFAR-10 | ResUNet | 2,473,475 | 0.06138 | 0.7387 | 24.68 | 104.07 ± 21.13 |
| CIFAR-10 | Plain UNet | 2,473,475 | 0.06108 | 0.7356 | 24.68 | 107.52 ± 32.86 |
| CIFAR-10 | DDPM UNet | 2,356,739 | **0.06012** | **0.7445** | **24.84** | **72.88 ± 7.80** |
| CelebA | ResUNet | 2,612,419 | 0.03691 | 0.8011 | 27.10 | 83.08 ± 23.98 |
| CelebA | Plain UNet | 2,612,419 | 0.03661 | 0.7991 | 27.11 | 146.39 ± 22.71 |
| CelebA | DDPM UNet | 2,356,739 | **0.03449** | **0.8164** | **27.43** | **81.57 ± 10.76** |

FID from 2048 samples per model per seed. IS is omitted here: CelebA is a single semantic category,
so the class diversity the measure depends on is undefined.

**Paired ResUNet − Plain UNet, FID** (negative favours ResUNet):

| Dataset | Seed 0 | Seed 1 | Seed 2 | Mean ± SD | Sign |
|---|---|---|---|---|---|
| CIFAR-10 | +10.64 | −16.89 | −4.12 | −3.46 ± 13.78 | **flips** |
| CelebA | −59.12 | −21.68 | −109.13 | **−63.31 ± 43.88** | **consistent** |

Four findings:

1. **Residual shortcuts accelerate early optimisation.** The proposed network leads its
   shortcut-free twin at epoch 1 in all six paired runs, by 5.4% on average on CIFAR-10 and 11.1%
   on CelebA. The two are otherwise identical, so the shortcut is the only possible cause.
2. **Restoration measures cannot separate them.** Loss, SSIM and PSNR agree to within half a per
   cent, and the two that lean at all lean in opposite directions — loss favours Plain, SSIM
   favours ResUNet, PSNR changes sign between seeds.
3. **Generative quality separates them on the deeper configuration only.** On CelebA (three
   resolution stages) ResUNet leads in every seed, by a mean of 63.31 FID against a seed-to-seed SD
   of ~23. On CIFAR-10 (two stages) the sign flips and the difference is −3.46 ± 13.78, i.e. no
   effect. **Networks that denoise identically well do not generate equally well**, and no
   restoration measure predicted it.
4. **The reference architecture is strongest overall** and the most stable (FID SD 7.80 vs 21.13
   and 32.86). It leads CIFAR-10 FID by 31.19 in every seed, using fewer parameters — at ~2.3× the
   training cost. Only on CelebA FID is it matched, by ResUNet at less than half the compute.

The hypothesis receives **partial and conditional support**: at two resolution stages residual
learning makes no detectable difference; at three stages it leads in every seed by a margin larger
than the observed spread. Three seeds show the direction is consistent but are too few for a
significance test — the paired CelebA result gives *t*(2) = −2.50.

> A methodological note: **FID is far noisier than the restoration measures at this scale** (SD 7.8
> to 32.9, against thousandths for SSIM). Single-run FID comparisons of diffusion models trained
> this briefly are not reliable — the apparent CIFAR-10 reversal in the first single-seed run turned
> out to be noise.

A nearest-neighbour check against the training set, with real and blurred held-out images as
controls, found no evidence of memorisation on either dataset.

## Extension: sharpening the samples

The samples above are recognisable in kind but not in particular. The thesis traces this to two
causes: a **training budget** about three orders of magnitude below Ho et al. (14× fewer parameters,
68× fewer optimisation steps), and **regression to the mean**. Under a squared-error objective the
optimal prediction is `E[x_0 | x_t]`, and wherever several clean images fit a noisy one, their
average is blurry.

`DissertationExtension.ipynb` keeps the diffusion framework and the ResUNet family fixed and changes
only how the network is trained and sampled. The changes are applied as a **cumulative ladder**, so
the FID difference between adjacent rungs is the contribution of one change.

| Rung | Change added | Epochs | Why |
|---|---|---|---|
| E0 | none: reproduces Chapter 3, seed 0 | 30 | Validates the pipeline |
| E1 | EMA of the weights (0.9995, with warm-up) | 30 | Polyak averaging. The last SGD iterate is one draw from a fluctuation around the minimum |
| E2 | LR warm-up (2000 steps), grad clip (1.0), dropout (0.1) | 30 | Stabilisers from Ho et al.'s reference implementation |
| E3 | min-SNR loss weighting, γ = 5 | 30 | The ε loss weights the x₀ error by SNR(t), which is 24,221 at t = 0. Capping it down-weights the easiest 26% of timesteps |
| E4 | 5× training | 150 | Steps gap: 68× → 14× |
| E5 | Capacity: widths (128,256,256), 2 blocks per stage | 150 | 2.47M → 31.46M parameters (88% of Ho et al.) |
| E6 | Long training, EMA 0.9999, resumed from E5 | 2000 (CelebA 700) | 780,000 steps, 97.5% of Ho et al.'s 800,000 |

- The full ladder runs on CIFAR-10. CelebA runs E0, E5 and E6 only, to check that the effect carries
  over.
- E6 resumes from the E5 checkpoint, so E5 is a waypoint on E6's trajectory and costs nothing extra
  to train.
- The final rung is also sampled with DDIM (100 steps) and reported as a separate row, because
  changing the sampler changes FID.

**Status:** the full run (about 63 hours) is in progress, and the results will be added here and to
Chapter 4. A preliminary evaluation of E0 gives **FID 89.57** against Chapter 3's 94.93, within one
seed-to-seed SD (21.13). This validates the pipeline.

**Reading rule, fixed in advance:** each rung is a single seed. A ΔFID smaller than the Chapter 3
seed SD (21.13 on CIFAR-10, 23.98 on CelebA) is reported as inconclusive, not as an improvement.

### Keeping the GPU busy

The extension was fitted to a 72-hour budget on one RTX 4070. All numbers below were measured on an
otherwise idle card at batch 128.

| Measure | Effect |
|---|---|
| `torch.compile` (training) | 1.6–2.0× per step on the three smaller networks |
| No per-step `loss.item()` | +8% on the small network. GPU utilisation 91% → 97% |
| Compiled sampling (CUDA graphs) | 2.0–3.0× faster reverse process |
| 90% VRAM cap | Turns a silent slowdown into an immediate OOM (see below) |
| Sleep prevention | Keeps Windows awake while the kernel is alive |

> **VRAM overflow is a slowdown, not an error, on Windows.** Run eagerly, the 24.4M CelebA network
> peaks at 12.9 GB on a 12 GB card. The driver silently falls back to system memory, each step takes
> 3.5 s instead of 0.39 s, and power drops from ~190 W to 88 W. Compilation cuts the peak to 7.2 GB.
> Without it, CelebA E5–E6 would take 260+ hours instead of 30.

Compiled sampling reorders floating-point work, so samples are not bit-identical to eager sampling.
On the E0 checkpoint the effect was FID 89.57 → 90.97 (IS 4.649 → 4.650). That is under a fifth of
the smallest seed SD, and it applies equally to every rung. Set `COMPILE_SAMPLING = False` to match
Chapter 3 exactly.

Rejected after measurement: CUDA graphs for training (+2% to −6%), `max-autotune` (no gain, 3 min
compile), and keeping the dataset resident on the GPU (0%).

## Repository layout

```
imagegeneration.ipynb        # architecture study: models, training, sampling, evaluation
DissertationExtension.ipynb  # extension: the E0–E6 ladder, its evaluation and thesis figures
requirements.txt             # pinned environment (CUDA 13.0 wheels)
generated_samples/           # 64 samples per model, identical initial noise
denoising_process/           # reverse-trajectory animations
dissertation/main.pdf        # compiled thesis
Dissertation_v1.pdf          # earlier version of the thesis
model/seed_study.json        # three-seed results of the architecture study
model/CIFAR-10/, model/CelebA/   # architecture-study checkpoints (not tracked)
model/extension/             # extension checkpoints and results (not tracked)
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

### Running the extension

`DissertationExtension.ipynb` reads the tensor caches that `imagegeneration.ipynb` builds, so run
the architecture-study notebook's data section once first. The extension never writes to the
architecture-study checkpoints.

| `PRESET` | What runs | Output | Time on RTX 4070 |
|---|---|---|---|
| `"quick"` | CIFAR-10 only, short rungs, narrow E5/E6 | `model/extension/quick/` | ~20 min |
| `"full"` (default) | Full ladder on CIFAR-10, E0/E5/E6 on CelebA | `model/extension/` | ~63 h |

For the 63-hour run, running without an editor keeps a log on disk. From the repository root in
PowerShell:

```bash
cmd /c "set MPLBACKEND=Agg&& set PYTHONIOENCODING=utf-8&& set PYTHONUNBUFFERED=1&& .diffusion\Scripts\ipython.exe DissertationExtension.ipynb > model\extension\weekend.log 2>&1"
```

- Training resumes from the last saved epoch, so rerunning the same command after an interruption
  loses at most one epoch.
- A lock file prevents two runs from writing the same checkpoints.
- Pause Windows Update first. The notebook prevents sleep, but it cannot prevent a forced restart.
- If you run it from VS Code instead, restart the kernel before *Run All*, so no stale definitions
  from an older version of the notebook remain.
- Check that the GPU is saturated with:

  ```bash
  nvidia-smi --query-gpu=utilization.gpu,power.draw,memory.used --format=csv -l 5
  ```

  Expect 97–100% utilisation at 165–195 W. 100% utilisation below 100 W means VRAM has overflowed.

## References

1. J. Ho, A. Jain and P. Abbeel, *Denoising Diffusion Probabilistic Models*, arXiv:2006.11239.
2. A. Nichol and P. Dhariwal, *Improved Denoising Diffusion Probabilistic Models*, arXiv:2102.09672.
3. Y. Song et al., *Score-Based Generative Modeling through Stochastic Differential Equations*,
   arXiv:2011.13456.
4. K. He, X. Zhang, S. Ren and J. Sun, *Deep Residual Learning for Image Recognition*,
   arXiv:1512.03385.
5. L. Yang et al., *Diffusion Models: A Comprehensive Survey of Methods and Applications*,
   arXiv:2209.00796.

Extension:

6. B. T. Polyak and A. B. Juditsky, *Acceleration of Stochastic Approximation by Averaging*,
   SIAM J. Control Optim. 30(4), 1992.
7. T. Hang et al., *Efficient Diffusion Training via Min-SNR Weighting Strategy*, arXiv:2303.09556.
8. J. Song, C. Meng and S. Ermon, *Denoising Diffusion Implicit Models*, arXiv:2010.02502.
9. L. Liu et al., *On the Variance of the Adaptive Learning Rate and Beyond*, arXiv:1908.03265.
10. N. Srivastava et al., *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*,
    JMLR 15, 2014.
11. J. Ansel et al., *PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode
    Transformation and Graph Compilation*, ASPLOS 2024.
