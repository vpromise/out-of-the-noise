# Runs on this Mac: the digit model and the lighthouse

Everything here was trained or run on this Mac Studio (Mac16,9: Apple M4 Max, 16-core CPU, 40-core GPU, 64 GB unified
memory, macOS 27.0.1) on 2026-10-25/26, with **MLX 0.32.4** (Python 3.12.14, numpy 2.5.4, uv venv
`research/experiments/.venv-gpu`). Machine-readable numbers: `mac_runs.json` (a flat `headline` block of scalars, then
full tables; curves have at most 200 points). Images: `images/` (PNG). Arrays: `arrays/` (npz). Code: `gpu/`.
Logs: `logs/`. Weights: `weights/digit_*` (48 MB).

Total GPU time was about 3 h 05 min, run one job at a time: training 1 h 36 min, digit evaluation about 1 h, lighthouse
about 6 min, toys about 10 min.

---------------------------------------------------------------------------------------------------------------------

## 1. Framework: MLX (measured, not assumed)

The same U-Net was written in MLX and in PyTorch-MPS (torch 2.14.1). Parameter counts are identical (3,940,801), and
both were timed back to back (`gpu/bench_frameworks.py`, `logs/bench_frameworks_*.json`):

| U-Net config | params | MLX train img/s | torch-MPS train img/s | MLX infer img/s | torch-MPS infer img/s |
|---|---|---|---|---|---|
| ch 32, mults 1-2-4 (**chosen**) | 3.94 M | **1,455** | 942 | **5,603** | 5,259 |
| ch 48, mults 1-2-4 | 8.85 M | 803 | 628 | 3,004 | 2,792 |
| ch 64, mults 1-2-2 | 6.47 M | 742 | 659 | 2,234 | 2,780 |
| ch 64, mults 1-2-4 | 15.7 M | 571 | 506 | 2,308 | 1,990 |

(Training at batch 256, compiled MLX step vs eager PyTorch. Inference at batch 1,000.) MLX trained **1.55x faster** for
the chosen size. Inference was about equal. MLX is also the framework the lighthouse pipeline uses, so one stack
covers both. fp16/bf16 inference gave no speed-up for the digit model; GroupNorm costs about 25% of a forward pass.

## 2. Data

MNIST: 60,000 training and 10,000 test digits, 28x28 greyscale, from https://ossci-datasets.s3.amazonaws.com/mnist/
(the torchvision mirror of http://yann.lecun.com/exdb/mnist/). Checked against the canonical MD5s (listed in
`mac_runs.json` under data.md5). **License: CC BY-SA 3.0, Yann LeCun and Corinna Cortes.** Pixels are scaled to
[-1, 1], with no augmentation for the diffusion models. Files are in `data/mnist/`.

## 3. The digit models

### Architecture (Episode 8)
A class-conditional DDPM-style U-Net, **3,940,801 parameters**, NHWC layout (`gpu/models_mlx.py`):

- Resolutions 28x28 → 14x14 → 7x7 with **32 / 64 / 128 channels**.
- 2 ResBlocks per level going down, 3 going up (each up-block concatenates a skip connection). ResBlock: GroupNorm(8),
  SiLU, conv 3x3, plus a per-channel bias from the embedding, then GroupNorm, SiLU, conv 3x3 (zero-initialised). A 1x1
  conv on the skip path when the channel count changes.
- Downsample: conv 3x3 with stride 2. Upsample: nearest x2, then conv 3x3.
- **Self-attention (4 heads)** in every 7x7 block (2 down, 3 up) and in the middle (ResBlock, attention, ResBlock).
- **Time embedding:** sinusoidal (dim 32) → Linear 32→128 → SiLU → Linear 128→128. **Label embedding:** 11x128 table
  (index 10 = "no label"), added to the time embedding, then SiLU. Every ResBlock receives this vector.
- Parameters by part: 7x7 up-level 1.74 M, middle 0.69 M, 7x7 down-level 0.69 M, 14x14 up 0.44 M, 14x14 down 0.19 M,
  28x28 up 0.11 M, 28x28 down 0.05 M, embeddings 0.02 M. Most parameters sit at low resolution, where they are cheap;
  most compute is at high resolution.
- Activations of a noisy "7" (t = 500) through every layer: `arrays/digit_unet_activations.npz` (shape, first 8
  channels and mean |activation| per layer: 28x28x32 → 14x14x64 → 7x7x128 → back).

### Training (Episodes 5 and 11)
The three models below share the same recipe:

- 10,000 steps × batch 256 = 2.56 M images, **42.7 epochs**.
- Adam, peak lr 5e-4, 500-step linear warm-up, cosine decay to 5e-5, gradient-norm clip 1.0.
- EMA 0.999 (the EMA weights are evaluated). 10% label dropout (for CFG). Seed 0.
- Every random draw is a function of (seed, step), so a resumed run replays exactly.
- Command: `python gpu/train.py --name <ddpm|flow|dit> --objective <ddpm|flow> [--arch dit]`

| model | objective | params | wall time | s/step (steady) | peak memory | loss: first 50 steps → final |
|---|---|---|---|---|---|---|
| **digit U-Net (DDPM)** | ε-prediction MSE, linear β 1e-4→0.02, T = 1000 | 3,940,801 | **31.8 min** (1,907.6 s) | 0.184 | 6.32 GB | 0.810 → **0.0202** |
| **digit U-Net (flow)** | rectified flow, x_t = (1−t)x₀ + tε, target v = ε − x₀, t ~ U(0,1) | 3,940,801 | **29.2 min** (1,751.3 s) | 0.169 | 6.32 GB | 1.580 → 0.149 |
| **tiny DiT** | same as DDPM | 4,150,660 | 34.7 min (2,082.9 s) | 0.203 | 10.28 GB | 0.753 → 0.0214 |

The flow loss is a velocity MSE, on a different scale from the ε-loss: do not compare the two numbers. The DDPM run's
first 2,000 steps were slowed by my own CPU tests (0.196 s/step), so use 0.18 s/step as the typical step time. Loss
curves (200 points: step, loss, lr, wall_s) are in `mac_runs.json` under digit_models.*.loss_curve. Raw logs:
`logs/train_*.jsonl`.

The tiny DiT uses patch size 2 (196 tokens of 2x2 pixels), width 192, depth 6, 4 heads, adaLN-Zero conditioning on
time + label, fixed 2D sin-cos positions, and a linear unpatchify.

Test-set loss at fixed noise levels (`loss_vs_t` in the json):
- The ε-MSE is highest for nearly clean images (0.153 at t = 0, 0.033 at t = 200) and reaches 0 at t = 999, where
  predicting the noise is trivial.
- The flow's v-MSE is U-shaped: 0.133 at t = 0.05, a minimum of 0.095 at t ≈ 0.35, and 0.237 at t = 0.9.

Timestep convention: code uses t = 0 … 999 (0-based). The paper's t = 1 … 1000 is code t + 1, so ᾱ at code t = 999
equals the toy's ᾱ_1000 = 4.04e-5.

### The measuring instrument: our own classifier
A small CNN (conv32-conv32-pool-conv64-conv64-pool-fc128-fc10, 467,818 params, 6 epochs, 14 s) reaches **99.45% MNIST
test accuracy** (55 errors). Its 128-d penultimate features are used for:

- **Classifier FID (cFID):** the FID formula applied to *these* features for 10,000 generated vs the 10,000 test
  digits. **This is not Inception FID.** The numbers only compare with each other. Floor (10,000 real training digits
  vs test): **2.06** (2.10 ± 0.10 over 5 random subsets).
- **Conditional accuracy:** classifier label == requested label.
- **Precision / recall:** k-NN manifold estimate, k = 3 (Kynkäänniemi et al. 2019).
- **Diversity:** per class, the mean pairwise feature distance of generated digits ÷ the same for real test digits
  (1.0 = as varied as real data). Not meaningful at w = 0, where the requested label is ignored.

Error bar: bootstrap std of cFID ≈ **0.3** (0.27–0.36). Differences under about 0.7 are not meaningful.

## 4. Sampling the digit model

Protocol (`gpu/sampling.py`, `gpu/eval_digits.py`):
- 10,000 samples per configuration; sample i asks for label i mod 10.
- **Every configuration starts sample i from the same noise** (numpy seed 1234). DDPM's per-step noise is seeded too.
- Batch 1,000, timed on the GPU. "s per 1,000" = the mean of batches 2–10; batch 1 includes compilation.
- Samplers:
  - **DDPM ancestral:** 1,000 steps, posterior variance β̃.
  - **DDIM (η = 0):** S steps evenly spaced from t = 999 to 0, with the last step landing on the clean image.
  - **Flow:** Euler, S equal steps from t = 1 to 0.
  - No clipping until the final image.
- **Guidance convention** (same as Stable Diffusion): pred = uncond + w·(cond − uncond).
  - w = 0: unconditional.
  - **w = 1: plain conditional, one network pass** (used for all step comparisons).
  - w > 1: CFG, two passes batched together.
  - Ho & Salimans' w equals our w − 1.

### Quality vs steps (Episodes 7, 9, 11, 19)

| sampler | network calls per digit | s per 1,000 digits | cFID ↓ | cond. acc | recall |
|---|---|---|---|---|---|
| real data (floor) | – | – | 2.06 | 99.55%* | 0.89 |
| DDPM ancestral, 1,000 steps | 1000 | **172.1** | **2.64** | 98.15% | 0.89 |
| DDIM 100 | 100 | 17.7 | 3.50 | 98.03% | 0.90 |
| DDIM 50 | 50 | **8.87** | **3.35** | 98.10% | 0.89 |
| DDIM 20 | 20 | 3.54 | 3.30 | 98.27% | 0.87 |
| DDIM 10 | 10 | **1.78** | **4.57** | 98.51% | 0.83 |
| DDIM 5 | 5 | 0.89 | 21.6 | 99.32% | 0.56 |
| DDIM 2 | 2 | 0.35 | 939 | 46.8% | 0.00 |
| DDIM 1 | 1 | 0.18 | 943 | 45.9% | 0.00 |
| flow Euler 50 | 50 | 8.85 | 3.27 | 98.24% | 0.89 |
| flow Euler 20 | 20 | 3.59 | 3.57 | 98.37% | 0.88 |
| flow Euler 10 | 10 | 1.92 | 4.57 | 98.60% | 0.85 |
| flow Euler 5 | 5 | 0.96 | **9.28** | 98.94% | 0.76 |
| flow Euler 2 | 2 | 0.41 | **85.3** | 99.34% | 0.15 |
| flow Euler 1 | 1 | 0.19 | 886 | 99.99% | 0.00 |

*Classifier accuracy on the 10,000 real training digits.

Grids: `images/digit_grid_<config>.png` (column = label 0–9). Samples: `arrays/digit_samples_<config>.npz`.

What the numbers say:

- **DDPM 1,000 steps gets closest to real data, but is slow.** DDIM 50 is **19x faster** (8.9 vs 172 s per 1,000) and
  still good (3.35). DDIM 10 is **97x faster** (4.57).
- DDIM 20 / 50 / 100 and flow 50 are statistically tied (bootstrap std ≈ 0.3).
- **Below 10 steps, flow matching degrades far more gracefully:** 5 steps 9.3 vs 21.6, 2 steps 85 vs 939. At ≥ 10
  steps the two models tie (4.57 vs 4.57).
- **1–2-step DDIM produces speckled noise.** With ε-prediction, the clean-image guess at t = 999 divides by
  √ᾱ ≈ 0.0064, which amplifies the network's error about 157x.
- **1-step flow produces a smooth, blurry average digit** (the model's expected-value guess; diversity 0.08). It is
  still classified correctly (99.99%), which shows label accuracy alone is a poor quality metric.
- Precision reads ≈ 1.0 for the broken 1–2-step rows. That is a known blind spot of k-NN precision with off-manifold
  samples, so do not quote precision for those rows.
- **Same noise, same digit:** DDIM-50 and flow-50 turn the same starting noise into nearly the same "7", even though
  the two networks were trained separately (`images/digit_trajectory_7_*`).

Latency at small batches (`latency` in the json):

| one batch of | DDPM 1,000 | DDIM 50 | DDIM 10 | flow 10 | flow 1 |
|---|---|---|---|---|---|
| 1 digit | **4.34 s** | **0.205 s** | 0.035 s | 0.043 s | **0.003 s** |
| 10 digits | 4.85 s | 0.238 s | 0.044 s | 0.044 s | 0.004 s |
| 100 digits | 21.4 s | 1.18 s | 0.22 s | 0.22 s | 0.018 s |

At batch 1 the GPU is mostly idle: about 4 ms per step for 1 or 10 digits, about 22 ms for 100, about 176 ms for
1,000. Sampling peak memory was 5.1 GB at batch 1,000.

### Guidance (Episodes 14 and 19): DDIM 50, same noise for every w

| w | passes/step | s per 1,000 | cFID ↓ | cond. acc | precision | recall | diversity | ink fraction |
|---|---|---|---|---|---|---|---|---|
| 0 (unconditional) | 1 | 9.8 | 22.3 | 9.9% (chance) | 0.76 | 0.90 | n/a | 0.133 |
| **1 (no guidance)** | 1 | 8.9 | **3.35** | 98.10% | 0.83 | **0.89** | **1.01** | 0.133 |
| 2 | 2 | 19.5 | 21.4 | 99.99% | 0.88 | 0.79 | 0.87 | 0.146 |
| 3 | 2 | 19.4 | 42.0 | 100.00% | 0.89 | 0.70 | 0.83 | 0.155 |
| 5 | 2 | 19.3 | 66.9 | 99.99% | 0.88 | 0.54 | 0.78 | 0.169 |
| 8 | 2 | 19.3 | 85.4 | 99.99% | 0.87 | **0.38** | **0.74** | **0.182** |

Grids: `images/digit_grid_ddpm_ddim50_w{0,1,2,3,5,8}.png`.

Guidance buys label accuracy (98.1% → 99.99%) and precision (0.83 → 0.89), and makes strokes bolder (ink fraction
0.133 → 0.182, the digit version of over-saturation). It pays with diversity (1.01 → 0.74) and recall (0.89 → 0.38).
CFG also doubles the cost per step.

On MNIST the FID-like score is best with **no** guidance. That differs from ImageNet-scale models, where a little
guidance usually improves FID, so say "on our digits". The unconditional samples (w = 0) score 22.3: the
unconditional mode only saw 10% of training. Label histogram at w = 0: 8–11% per class.

The flow model shows the same trend (Euler 20): w = 0 28.9, w = 1 3.57, w = 2 27.6, w = 3 51.5, w = 5 82.5,
w = 8 107.2.

### U-Net vs tiny DiT (Episode 18): same data, steps, batch and optimiser; DDIM 50, w = 1

| | params | train time | peak train memory | s per 1,000 | cFID | cond. acc | recall |
|---|---|---|---|---|---|---|---|
| U-Net | 3.94 M | 31.8 min | 6.3 GB | 8.9 | **3.35** | **98.1%** | 0.89 |
| DiT, patch 2 | 4.15 M | 34.7 min | 10.3 GB | 10.4 | 15.9 | 94.2% | 0.87 |

At this tiny scale and short budget the U-Net wins clearly. The DiT's test loss is only slightly higher (0.035 vs
0.033 at t = 200), yet its samples are visibly worse. Transformers' advantage appears at scale (cite the DiT paper,
not this run). At w = 3 the DiT scores 101 vs the U-Net's 42.

### Memorisation check (Episode 20)
L2 distance in pixel space (pixels in [0, 1], 784 dims) from each of 10,000 generated digits to its nearest of the
60,000 training digits:

| | median | 1st percentile | min | share closer than real test digits' 1st percentile |
|---|---|---|---|---|
| real test digits → train | **4.34** | 1.29 | 0.83 | 1% (by definition) |
| generated (DDIM 50, w = 1) | **4.52** | 1.43 | 0.89 | 0.47% |
| generated (DDPM 1,000) | 4.40 | 1.40 | 1.04 | 0.43% |
| generated (flow 50) | 4.47 | 1.44 | 1.05 | 0.31% |
| two random digits | 10.1 | | | |

Generated digits are as far from the training set as genuinely new handwriting is, so there is no sign of copying.
The closest generated digits (distance 0.89–1.02) are all plain "1"s, the simplest digit, where real test 1s come
just as close.

Picture: `images/digit_nearest_train_neighbours_ddim50_w1.png`. Top row: 12 random generated digits plus the 4
closest. Bottom row: their nearest training images. Distances are in `arrays/digit_nearest_neighbours.npz`.

### Animation material
- **Forward process (Episode 4):** a training "7" at t = 0, 25, 50, 100 … 999, plus β, ᾱ and SNR for all 1,000
  steps. `arrays/digit_forward_process.npz`, `images/digit_forward_process_7.png`.
- **One "7" from noise (Episodes 1, 7, 9, 11):** `arrays/digit_trajectory_7.npz` holds every step of x_t (the input
  to each step) and x̂₀ (the model's clean-image guess) for DDPM 1,000, DDIM 50, DDIM 10, DDIM 50 at w = 3, flow 50
  and flow 10, all from the same starting noise. Also `*_frames40_index` (40 evenly spaced frames) and `many20_*`
  (20 digits emerging together, 40 frames of DDPM 1,000). Strips: `images/digit_trajectory_7_<config>.png` (top x_t,
  bottom x̂₀).
  - The DDPM-1,000 "7" has a European crossbar. Six more seeds with the same three samplers are in
    `arrays/digit_trajectory_7_more_seeds.npz`. Their finals are in `images/digit_trajectory_7_more_seeds_finals.png`
    (rows DDPM 1,000 / DDIM 50 / flow 50) for picking a clearer 7. Cleanest: DDPM 1,000 seeds 37 and 57; seed 47 with DDIM 50 and flow 50 (the same 7 from both models).
- **Same noise, ten labels (Episode 13):** 8 seeds × labels 0–9, with the same starting noise (and the same DDPM
  per-step noise) in each row. Samplers: DDIM 50 at w = 1 and 3, DDPM 1,000, flow 50 at w = 1 and 3.
  `arrays/digit_same_noise_ten_labels.npz`, `images/digit_same_noise_ten_labels_<config>.png`.
- **Moving digit (Episodes 21–22):** see section 6.

## 5. The lighthouse (Episodes 1, 10, 12, 14, 15, 16, 17, 23)

**Pipeline:** ml-explore/mlx-examples `stable_diffusion`, pinned at commit `796f5b53cab69a3d48a44233ce21aae889e94a08`
(2026-04-06), sparse-cloned to `gpu/third_party/mlx-examples/`. The UNet and text encoder run in float16 and the VAE
in float32 (the pipeline defaults). The SD 2.1 sampler is the library's deterministic Euler sampler; SD-Turbo uses
its Euler-ancestral sampler, as the library does for SDXL-Turbo.

My script `gpu/lighthouse.py` reproduces the library's `generate_latents` exactly: same seeding, CFG formula
`eps_neg + w·(eps_text − eps_neg)` with CFG off for w ≤ 1, and an empty negative prompt. It adds timers and
exposes ε for the x̂₀ previews. Check: identical latents to the library (max abs diff 0.0), and identical img2img
output (pixel diff 0).

Three small load-time adapters, no change to the maths:
- the mirror's fp32 VAE uses legacy attention key names (query/key/value/proj_attn) for the same linear layers;
- the fp16 text-encoder file carries a non-learned `position_ids` buffer, which is skipped;
- sd-turbo's config writes `transformer_layers_per_block` as an int.

**Models:**

| | repo (pinned revision) | license | UNet | text encoder | VAE |
|---|---|---|---|---|---|
| multi-step | `sd2-community/stable-diffusion-2-1-base` @ `4e63672c` | CreativeML Open RAIL++-M | 865.9 M (fp16, 1.73 GB) | 340.4 M OpenCLIP-H text (fp16, 0.68 GB) | 83.7 M (encoder 34.2 M, decoder 49.5 M; fp32, 0.33 GB) |
| distilled | `stabilityai/sd-turbo` @ `b261bac6` | Stability AI Community License (free for research, non-commercial and commercial use under US$1 M annual revenue) | 865.9 M, same architecture | same | same |

- Stability's own `stabilityai/stable-diffusion-2-1-base` repository has been removed from Hugging Face (404), so I
  used the `sd2-community` mirror, whose model card is Stability's. Its fp16 VAE file is byte-identical to sd-turbo's.
- SD-Turbo is SD 2.1 distilled by Adversarial Diffusion Distillation, with the same 865.9 M-parameter UNet, which
  makes it a like-for-like teacher-vs-student pair for Episode 12.
- Prompt: **"a lighthouse on a rocky coast at dusk, oil painting"** (no people, no artist names). Seeds 0, 1, 2, 3.
  512x512.

**Latent shape confirmed:** a 512x512x3 image is a **64x64x4 latent**, 48x fewer numbers. A VAE round trip of the
seed-0 lighthouse gives 38.1 dB PSNR. That flatters the VAE, because the source is itself a VAE output.

### Timing breakdown: one image, SD 2.1-base, 512x512, 50 steps, guidance 7.5 (median of 3 runs: 7.35 / 7.32 / 7.31 s)

| stage | time | share |
|---|---|---|
| text encoder (13 tokens, prompt + empty negative; see erratum) | **0.017 s** | 0.2% |
| UNet: 50 steps × 2 passes (CFG, batched) | **7.04 s** (0.141 s/step) | 96.2% |
| VAE decode 64x64x4 → 512x512x3 | **0.265 s** | 3.6% |
| **total** | **7.32 s** | |
| peak memory | **8.13 GB** | |

- Without CFG (guidance 1, one pass): 0.083 s/step, 4.43 s total. CFG costs 1.7x per step on this GPU, not 2x,
  because the two passes are batched.
- **SD-Turbo, 1 step, no CFG:** 0.364 s total (UNet 0.076 s, VAE 0.271 s, text 0.017 s). Peak 10.9 GB, with both
  models loaded.
- Loading takes about 0.5 s when the files are in the OS cache. The first, cold denoise (4 steps with CFG) took 1.19 s
  including kernel compilation.
- The VAE decode varied from 0.17 to 0.27 s across stages, so other jobs on the machine add jitter (see caveats).

### Steps (Episodes 10, 12), guidance 7.5, seeds 0–3

| steps | 1 | 2 | 4 | 8 | 20 | 50 |
|---|---|---|---|---|---|---|
| SD 2.1 total s (median of 4 seeds) | 0.33 | 0.48 | 0.75 | 1.34 | 2.93 | 6.90 |
| SD-Turbo total s, no CFG (median of 4 seeds) | **0.25** | 0.32 | 0.47 | – | – | – |

- The sweep's 50-step median (6.9 s) is a little faster than the timing run's 7.3 s; that gap is run-to-run jitter.
- **SD 2.1:** 1 step gives a coloured blur, 2 steps a blurry scene, a lighthouse is recognisable by 4, and 20–50 are
  sharp. `images/lighthouse_grid_sd21_steps_1-2-4-8-20-50_cfg7.5_seeds0-3.png`, singles in `images/lighthouse/`.
- **SD-Turbo** makes a complete picture in **one step**. `images/lighthouse_grid_sdturbo_steps_1-2-4_seeds0-3.png`.

### Guidance (Episode 14), 50 steps, seeds 0–3

| guidance | 1 | 3 | 5 | 7.5 | 10 | 15 |
|---|---|---|---|---|---|---|
| mean colour saturation (4 seeds) | 0.19 | – | – | 0.28 | – | 0.36 |
| seed 0: saturation / clipped pixels | 0.18 / 0% | 0.17 / 0% | 0.21 / 0% | 0.27 / 0.01% | 0.39 / 4.0% | 0.43 / **9.3%** |
| time (s) | 3.9 (1 pass) | 6.6 | 6.6 | 6.6 | 6.6 | 6.6 |

What the images show:
- guidance 1: washed out, with a weak lighthouse;
- 3–7.5: a coherent painting;
- 10–15: over-saturated, high contrast, clipped colours.

`images/lighthouse_grid_sd21_cfg_1-3-5-7.5-10-15_steps50_seeds0-3.png`.

### Picture out of static (Episodes 1, 17)
30 steps, guidance 7.5, seed 0. `images/lighthouse_frames30/`:
- `xt_00..30.png`: the noisy latent decoded at each step. It starts as coloured static.
- `x0hat_00..29.png`: the model's clean-picture guess at each step. It starts as a blur and sharpens.

Latents (float16, [31 or 30, 64, 64, 4]) with t and σ are in `arrays/lighthouse_latents_30steps.npz`. A 10-frame
strip is `images/lighthouse_frames30_strip_x0hat_top_xt_bottom.png`. The 4 latent channels of the finished image
are in `images/lighthouse_latent_4channels_64x64.png` and `arrays/lighthouse_vae_latent.npz`.

### Image-to-image / SDEdit (Episode 16)
Source: `lighthouse_sd21_cfg7.5_steps50_seed0.png`. Strength s means: encode the source, add noise to t = 1000·s,
then run int(50·s) Euler steps at guidance 7.5.

- **Prompt as specified**, "a lighthouse on a rocky coast at dusk in winter, snow, oil painting", strengths 0.3 /
  0.5 / 0.7 / 0.9 (15 / 25 / 35 / 45 steps):
  - honest result: the layout is kept at every strength and **no snow appears**; the texture gets crisper and more
    contrasted;
  - mean pixel change 6% / 9% / 10% / 9%;
  - files: `images/lighthouse_grid_sdedit_winter_source_0.3_0.5_0.7_0.9.png`, plus the noised starting points
    `images/lighthouse/lighthouse_sdedit_winter_strength*_noised_start.png`.
- **More explicit prompt**, "a lighthouse on a snowy rocky coast in winter, snow-covered rocks, falling snow, at dusk,
  oil painting", strengths 0.3 / 0.5 / 0.7 / 0.9 / 1.0:
  - 0.3–0.7 keep the picture;
  - **0.9 turns it into a snowy scene with the same layout**;
  - 1.0 is nearly a new picture, but the lighthouse still lands on the right, because SD's schedule keeps about 7%
    of the signal even at t = 999 (√ᾱ ≈ 0.07; the "non-zero terminal SNR" issue);
  - mean pixel change 6% / 9% / 10% / 13% / 15%;
  - file: `images/lighthouse_grid_sdedit_snow_source_0.3_0.5_0.7_0.9_1.0.png`. **Use this series on screen** and say
    the prompt was made explicit.

## 6. Moving digit (Episodes 21–22): independent vs consistent frames

**What was run (honest version):** the trained digit model generates 8 frames per clip (DDIM 50, w = 1, one label per
clip, 20 clips). Frame k starts from the noise

  ε_k = √ρ·ε_shared + √(1−ρ)·ε_k,own

which is the "mixed noise" idea from PYoCo (Ge et al. 2023). ρ = 0 means frames drawn independently; ρ = 1 means
one shared noise.

**Motion is imposed, not generated:** frame k is pasted 3 px further right on a 28x52 canvas. The model is never
asked to move anything. What is measured is whether the digit stays the same digit:

- flicker = mean |frame_{k+1} − frame_k shifted 3 px|, pixels in [0, 1];
- change = L2 distance between consecutive 28x28 digits.

| noise correlation ρ | 0 (independent) | 0.5 | 0.9 | 0.99 | 1 (shared) |
|---|---|---|---|---|---|
| flicker | **0.078** | 0.054 | 0.024 | 0.007 | **0** |
| frame-to-frame change (L2) | 9.2 | 7.2 | 3.9 | 1.5 | 0 |

With independent noise each frame is a different "0" or "1". At ρ = 0.9 there is an occasional jump; at 0.99 the
digit is steady. Files: `images/digit_video_rho{0,0.5,0.9,0.99,1}_clips0-3.png` and
`arrays/digit_video_noise_correlation.npz` (per ρ: [clip, frame, 28, 28] digits and [clip, frame, 28, 52] canvases).
Code: `gpu/video_toy_noise.py`.

On screen, say: "the cheapest consistency trick, correlated starting noise, removes the flicker; a real video model
learns consistency by denoising all frames together." Do not claim this toy is a video model.

**Failed attempt, do not use:** a jointly-denoising model (the U-Net with 8 stacked frames as channels) and a
per-frame model were trained for only 2,500 steps × 64 clips each (about 4 min each, `gpu/video_toy.py`). Both were
far too undertrained: their samples are noise-ridden and their flicker and motion numbers are meaningless. Outputs
were moved to `logs/video_toy_stacked_undertrained/`, the weights were deleted, and the raw numbers are kept in
`mac_runs.json` under video_toy_stacked_frames_attempt, flagged FAILED. Doing it properly needs about 10k steps per
model (about 25 min each).

## 7. What to say on screen (all measured on this Mac)

- **Episode 1:** "Generated on this Mac: Stable Diffusion 2.1 (base), 512x512, 30 steps; 50 steps take 7.3 seconds."
  Use the lighthouse_frames30 x̂₀ or x_t frames.
- **Episode 4:** the digit "7" dissolving under the course's linear schedule, β from 1e-4 to 0.02 over 1,000 steps.
- **Episode 5:** "A 3.9-million-parameter U-Net, trained on this Mac in 32 minutes: 10,000 steps of 256 digits,
  about 43 passes over MNIST. The loss fell from 0.81 to 0.020."
- **Episode 7:** "1,000 steps: 4.3 s for one digit, 172 s for a thousand."
- **Episode 8:** "32 → 64 → 128 channels at 28 → 14 → 7 pixels; attention only at 7x7; 3.94 M parameters, most of
  them in the smallest level."
- **Episode 9:** "Same noise, same digit. DDIM: 50 steps is 19x faster than 1,000, at a score of 3.3 vs 2.6 (real
  data: 2.1); 10 steps is 97x faster (4.6); at 1–2 steps the ε-model's guess is speckled noise."
- **Episode 11:** "Trained the same way for the same 29–32 minutes, the flow model matches DDPM at 10 or more steps
  and degrades far more gracefully below: 5 steps 9.3 vs 21.6; 2 steps 85 vs 939. One flow step gives the blurry
  average digit."
- **Episode 12:** "SD-Turbo, distilled from SD 2.1 with the same 866 M-parameter UNet, makes the lighthouse in one
  step: 0.25–0.36 s, against 7.3 s for 50 guided steps."
- **Episode 13:** the same noise grid becoming 0–9 (98.1% of generated digits are read as the requested label).
- **Episode 14:** "Turning w from 1 to 8: label accuracy 98.1% → 99.99%, but recall 0.89 → 0.38 and strokes get
  bolder. On our digits the FID-like score is best with no guidance." For the lighthouse: "guidance 10–15
  over-saturates; 9% of pixels clip at 15."
- **Episode 15:** text encoding takes 17 ms of a 7.3 s generation.
- **Episode 16:** the snow series. The layout survives up to strength 0.9; at 1.0 it is almost a new picture.
- **Episode 17:** "512x512x3 → 64x64x4: 48 times fewer numbers to denoise."
- **Episode 18:** "At 4 M parameters and 30 minutes, our U-Net beats a same-size DiT (score 3.3 vs 15.9). Transformers
  win at scale, not here."
- **Episode 19:** cFID vs steps and vs guidance from the tables. Always say "a classifier-based FID-like score, not
  Inception FID". Floor 2.06; error bar about ±0.3. Precision and label accuracy can look perfect while samples are
  broken (1-step rows).
- **Episode 20:** "Our digits are as far from their nearest training digit as real new handwriting: median 4.5 vs
  4.3; two random digits are 10.1 apart."
- **Episode 22:** the correlated-noise flicker table (section 6).
- **Episode 23:** the timing breakdown table. The UNet is 96% of the time; CFG makes each step 1.7x slower; peak
  memory 8.1 GB; 5.5 GB of weights downloaded for two models.

## 8. Caveats

- **The Mac was shared.** Other course jobs ran at the same time and were not controlled: a TTS narration process,
  VideoToolbox encoding and decoding, and macOS `mediaanalysisd` at up to 200% CPU. Timings carry about ±5–10%
  jitter. Examples: DDPM-1000 batches drifted from 162 s to 178 s (possibly thermal too); the VAE decode varied
  0.17–0.27 s. Comparisons within a table were run back to back.
- The DDPM training's first 2,000 steps overlapped my own CPU tests (+6%). Its 31.8 min vs flow's 29.2 min is not a
  property of the objectives: the step costs are the same.
- cFID depends on our classifier and on the sample count (10k vs 10k). Use it only for relative comparisons. Bootstrap
  std is about 0.3, and the floor varies 1.99–2.28 across subsets.
- k-NN precision is unreliable for off-manifold samples (1–2-step rows). The diversity ratio is meaningless at w = 0.
- One training seed per model. There are no repeats, so model-vs-model differences of a few tenths are not
  established.
- The lighthouse sampler is the mlx-examples Euler sampler (linear timestep spacing; it feeds t up to 1000 to a UNet
  trained on 0–999), not diffusers' default PNDM or DDIM, so images will not match other tools pixel for pixel.
- The SD 2.1-base weights come from a community mirror (`sd2-community`) because Stability's repository is gone.
  License and model card are Stability's (OpenRAIL++).
- The VAE round-trip PSNR (38.1 dB) is measured on a VAE-generated image, so it is optimistic for real photos.
- `weights/swirl_*.npz` belong to another agent (the swirl reference). I did not touch them.

## 9. Reproduce

```bash
cd research/experiments
uv venv --python 3.12 .venv-gpu && VIRTUAL_ENV=$PWD/.venv-gpu uv pip install mlx numpy pillow torch safetensors huggingface-hub regex tqdm
# MNIST -> data/mnist/ (4 files from https://ossci-datasets.s3.amazonaws.com/mnist/, MD5s in common.py)
P=.venv-gpu/bin/python
$P gpu/bench_frameworks.py mlx && $P gpu/bench_frameworks.py torch        # framework choice
gpu/queue_train.sh            # DDPM U-Net (10k steps), classifier, flow U-Net (resumable)
gpu/queue_eval.sh             # 10k-sample evaluations, extras, metrics, tiny DiT + its evaluation
$P gpu/eval_digits.py extras trajectories_more cfid_uncertainty
git clone --filter=blob:none --sparse https://github.com/ml-explore/mlx-examples gpu/third_party/mlx-examples \
  && git -C gpu/third_party/mlx-examples checkout 796f5b53cab69a3d48a44233ce21aae889e94a08 \
  && git -C gpu/third_party/mlx-examples sparse-checkout set stable_diffusion
$P gpu/download_lighthouse_models.py logs/lighthouse_downloads.json      # 5.50 GB, pinned revisions
HF_HUB_OFFLINE=1 $P gpu/lighthouse.py                                     # check timing steps guidance turbo frames sdedit vae
HF_HUB_OFFLINE=1 $P gpu/lighthouse.py sdedit2                             # explicit-snow SDEdit series
$P gpu/video_toy_noise.py
$P gpu/build_report.py                                                     # -> mac_runs.json
```

Versions: MLX 0.32.4, numpy 2.5.4, torch 2.14.1 (benchmark only), huggingface_hub 2.3.0, Python 3.12.14, macOS 27.0.1.

## 10. Downloads to delete after the course

| what | where | size |
|---|---|---|
| `sd2-community/stable-diffusion-2-1-base` @ 4e63672c03103b6c636b8fb4119ba982469b2955 (unet fp16, text_encoder fp16, vae fp32, configs, tokenizer) | default HF cache | 2.75 GB |
| `stabilityai/sd-turbo` @ b261bac6fd2cf515557d5d0707481eafa0485ec2 (same file set) | default HF cache | 2.75 GB |
| MNIST | `research/experiments/data/mnist/` | 11.6 MB |
| uv venv | `research/experiments/.venv-gpu/` | 0.9 GB |

This huggingface_hub version keeps file contents in a shared blob store (`~/.cache/huggingface/hub/blobs/`), so
deleting the `models--…` folders by hand does not free the space. Use the library instead (a dry run reported
5.5 GB freed):

```bash
research/experiments/.venv-gpu/bin/python -c "from huggingface_hub import scan_cache_dir as s; i=s(); i.delete_revisions(*[v.commit_hash for r in i.repos if r.repo_id in ('sd2-community/stable-diffusion-2-1-base','stabilityai/sd-turbo') for v in r.revisions]).execute()"
```

## Erratum (2026-10-27, coordinator, found by the Episode 15 agent)

The timing row once said "77 tokens". The MLX pipeline we ran (`mlx-examples/stable_diffusion/__init__.py`, `_tokenize`)
pads a batch only to its longest prompt (`N = max(lengths)`, pad id 0), so the lighthouse prompt was encoded as **13
tokens** (start + 11 + end) and the denoiser's cross-attention read 13 positions, not 77. The reference code and
diffusers pad to 77. Because the text model is causal, the first 13 vectors are the same either way; the 64 padding
vectors a 77-token pipeline adds are not, so images from the two pipelines can differ slightly.
