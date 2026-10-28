# Toy experiments: the swirl in 2D

Everything about the swirl that scenes show: the forward process, exact scores vs a trained MLP, every sampler,
flow matching, guidance, Langevin dynamics and FID's formula. All NumPy, all on this Mac's CPU. Results are in
`toy.json`; scenes read them with `measured("toy", …)` and recompute anything else live with
`diffusion_reference.py` (exact scores and the trained MLPs are fast enough to run inside a scene).

**Machine:** Apple M4 Max, 64 GB, macOS-27.0.1-arm64-arm-64bit-Mach-O; Python 3.13.15, NumPy 2.5.4; 2026-10-25.
**Wall times:** training 95 s (ε-model) + 94 s (flow model); experiments
99 s in total (samplers 72 s, Langevin 11 s, guidance
8 s, flow 5 s, the rest < 2 s each). `python diffusion_reference.py` (the self-checks) takes ≈ 3 s.

```
research/experiments/.venv-toy/bin/python research/experiments/scripts/toy_train.py        # ~3 min, writes weights/swirl_*.npz
research/experiments/.venv-toy/bin/python research/experiments/scripts/toy_experiments.py  # ~100 s, writes toy.json
research/experiments/.venv-toy/bin/python research/experiments/scripts/toy_report.py       # writes this file
```
(`.venv-toy` is a uv venv with numpy, Pillow and matplotlib; the scripts need only numpy.)

## Setup

- **The swirl** (`dr.swirl(n, seed)`): two spiral arms, each 1¼ turns from radius 0.15 out to 1.0, so it fits the unit
  disc the way pixels are scaled to [−1, 1]; std ≈ 0.5 per coordinate. It *is* a mixture of 256 round Gaussians
  (std 0.03) evenly spaced along the arms, so the noised density, score, perfect denoiser E[x₀ | x_t] and flow
  velocity are exact at every noise level (`dr.mixture_score`, `dr.exact_denoiser`, `dr.exact_eps`,
  `dr.exact_velocity`). The two arms are the labels for guidance.
- **Schedule:** DDPM's linear β (1e-4 → 0.02, T = 1000) unless "cosine" is named. t = 0 is clean data.
- **Trained models:** one tiny MLP, trained twice in NumPy (45,954 parameters; mlp_input(x, t) (98 numbers) → 3 × (Linear 128 → SiLU) → Linear 2).
  The input includes sinusoidal features of x itself (see caveats). ε-model: DDPM's simple loss, 40,000 Adam steps ×
  batch 512 of fresh swirl points (20.5 M points), EMA 0.999,
  95 s. Flow model: the same with the flow-matching loss, 94 s.
- **Shared noise:** every sampler comparison starts from the same z = dr.gaussian_noise(2000, seed=5).
- **Metrics** (`dr.swirl_distance`, `dr.off_swirl`, `dr.swirl_coverage`):
  - *dist* is the mean distance to the swirl's centre line. The data scores 0.02405, and that is the target, not 0.
  - *off* is the fraction of samples more than 0.1 from the curve (stragglers between the arms). The data scores 0.1 %.
  - *coverage* is the fraction of the 40 equal pieces of the curve that get at least a quarter of their fair share. The data scores 1.0.
  - *fd* is FID's formula on raw 2D coordinates against 20,000 data points.
  - *ode error* is the mean distance to the exact ODE solution from the same noise (Heun, 1,000 Karras steps).

## 1. The forward process

x_t = √ᾱ_t·x₀ + √(1 − ᾱ_t)·ε. *arm accuracy* = how often a perfect detective (the exact posterior p(arm | x_t))
names the arm a noisy point came from; 0.5 is a coin flip. (20,000 points.)

| t | β_t | ᾱ_t | √ᾱ (signal) | √(1−ᾱ) (noise) | SNR | σ | arm accuracy | cosine ᾱ_t | cosine SNR | cosine arm acc. |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 1.0 | 1.0 | 0.0 | inf | 0.0 | 1.0 | 1.0 | inf | 1.0 |
| 50 | 0.001076 | 0.971 | 0.9854 | 0.1702 | 33.5 | 0.1728 | 0.78 | 0.992 | 124.1 | 0.947 |
| 100 | 0.002072 | 0.897 | 0.9471 | 0.3209 | 8.71 | 0.3388 | 0.643 | 0.9721 | 34.83 | 0.786 |
| 250 | 0.00506 | 0.5241 | 0.7239 | 0.6899 | 1.101 | 0.9529 | 0.587 | 0.847 | 5.536 | 0.633 |
| 500 | 0.01004 | 0.07859 | 0.2803 | 0.9599 | 0.08529 | 3.424 | 0.528 | 0.4938 | 0.9757 | 0.582 |
| 750 | 0.01502 | 0.003351 | 0.05788 | 0.9983 | 0.003362 | 17.25 | 0.506 | 0.1443 | 0.1686 | 0.538 |
| 1000 | 0.02 | 4.036e-05 | 0.006353 | 1.0 | 4.036e-05 | 157.4 | 0.502 | 2.429e-09 | 2.429e-09 | 0.502 |

- Signal and noise are equal (SNR = 1, ᾱ = ½) at **t = 259** (linear) and **t = 496** (cosine).
  ᾱ drops below 1 % at t = 674 (linear) and t = 936 (cosine).
- Linear ᾱ_1000 = 4.036e-05 (pure noise to 4 parts in 100,000); cosine ᾱ_1000 = 2.429e-09
  (exactly 0 in the formula; Nichol & Dhariwal clip β at 0.999, which is why the last cosine β is 0.999).
- Snapshots at t = 0, 50, 100, 250, 500, 1000 for both schedules: `toy.json → forward/<schedule>/snapshots/<t>`
  (1,000 points, the same x₀ and ε at every t). Recipe to recompute them exactly: `x0 = dr.swirl(1000, seed=0); eps = dr.gaussian_noise(1000, seed=1); x_t = dr.q_sample(x0, t, eps, alpha_bar)  (alpha_bar = dr.ALPHA_BAR or dr.ALPHA_BAR_COSINE)`.
  Curves of ᾱ_t and log SNR every 10 steps: `forward/<schedule>/curve_alpha_bar`, `curve_log_snr`.

**On screen:** "Around step 260 of 1,000 the swirl is half signal, half noise. By step 500 only 8 % of the signal
variance is left, and even a perfect detective can tell which arm a point came from just 53 % of the
time, barely better than a coin. The cosine schedule keeps the picture alive twice as long: at step 500 it still has half its signal."

**Caveats:** "8 % of the signal" means ᾱ (signal *variance*); the signal *amplitude* √ᾱ is 28 %. Arm accuracy is
already down to 78 % at t = 50 because the arms interleave: a little noise is enough to confuse
neighbouring turns.

## 2. The exact score vs the trained MLP

The MLP predicts ε; its score is −ε̂/√(1 − ᾱ). Errors on 20,000 noisy points x_t drawn from p_t (where the model is
trained and used):

| t | σ | ε-loss floor (perfect model) | ε-loss of the MLP | excess | score error (relative RMS) | median arrow angle |
|---|---|---|---|---|---|---|
| 5 | 0.02645 | 0.7754 | 0.7798 | 0.004568 | 14.5 % | 3.86° |
| 10 | 0.04357 | 0.6633 | 0.6713 | 0.007754 | 15.4 % | 3.17° |
| 25 | 0.09227 | 0.759 | 0.7611 | 0.003173 | 11.7 % | 3.03° |
| 50 | 0.1728 | 0.7944 | 0.7952 | 0.0007224 | 6.1 % | 4.51° |
| 100 | 0.3388 | 0.6358 | 0.6361 | 0.000257 | 2.7 % | 1.07° |
| 250 | 0.9529 | 0.2168 | 0.2168 | 7.571e-05 | 1.0 % | 0.287° |
| 500 | 3.424 | 0.02123 | 0.02128 | 6.409e-05 | 0.8 % | 0.241° |
| 900 | 60.27 | 6.999e-05 | 0.0001399 | 7.065e-05 | 0.8 % | 0.226° |

- **The loss can't reach 0.** Averaged over t ~ U{1..1000} the best possible ε-loss is **0.1581** (computed with
  the exact denoiser on 200,000 points); the MLP finished at **0.1588**, within 0.4 % of it.
  The flow loss floor is 0.7402 (MLP: 0.7387). The floor is set by what the noisy point
  hides: which arm, and where across it, the point began.
- Loss curves (200 points each): `training/eps/loss_curve` with `loss_curve_steps`; the floor is the line to draw under them.
- On a 21 × 21 grid over [−1.5, 1.5]² (everywhere, including where no data goes) the arrows agree to:

| t | median angle (all grid) | high-density 75 % | lowest-density 25 % |
|---|---|---|---|
| 10 | 3.19° | 2.9° | 4.19° |
| 100 | 0.693° | 0.691° | 0.706° |
| 500 | 0.217° | 0.248° | 0.152° |

  Both fields at t = 10, 100 and 500 are stored as arrows: `score/fields/<t>/exact` and `…/mlp` (scores; the grid is
  `score/grid/axis`, all (x, y) pairs with x fastest). Normalise the arrows before drawing (on the grid the median arrow
  is ~100× longer at t = 10 than at t = 500).

**On screen:** "The arrows the network learned point the same way as the exact ones, within a few degrees. At low
noise the network's arrows are about 15 % off in length and direction, and at high noise under 1 %."

**Caveats:** without the sinusoidal features of x in its input, the same MLP learned the score well above t ≈ 30 but
blurred it below (excess error 25 % of the floor at t = 10 vs about 1 % now), and its DDPM samples filled the gaps between the
arms (*dist* 0.056, 21 % off-swirl). Training it 8× longer (5,000 → 40,000 steps) only partly helped (*dist* 0.087 → 0.056). This is the "spectral bias" of plain MLPs; the fix
(Fourier features, Tancik et al. 2020) is the same sinusoid trick as the time embedding. Low-density regions were *not*
noticeably worse here: on a 2D toy the network generalises smoothly into the corners.

## 3. Samplers: DDPM, DDIM, Euler and Heun

2,000 samples from the same noise; exact ε (left) and the trained MLP (right). Data: dist 0.02405, off 0.1 %,
coverage 1.0, fd 3.853e-05. NFE = model calls. Karras grid: sigma_min = σ(t=1) = 0.01, sigma_max = σ(t=1000) = 157.4, rho = 7.

| sampler | NFE | exact: dist | off | cov. | ode err | MLP: dist | off | cov. | ode err | MLP fd | MLP s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DDPM, σ²=β | 1000 | 0.02454 | 0.2 % | 1.0 | – | 0.02534 | 0.7 % | 1.0 | – | 0.001007 | 3.91 |
| DDPM, σ²=β̃ | 1000 | 0.02184 | 0.1 % | 1.0 | – | 0.02238 | 0.5 % | 1.0 | – | 0.001061 | 3.97 |
| DDIM η=0 (= Euler in σ) | 1000 | 0.02272 | 0.1 % | 1.0 | 0.00106 | 0.02379 | 1.0 % | 1.0 | 0.001289 | 0.0007493 | 3.9 |
| DDIM η=0 (= Euler in σ) | 100 | 0.01808 | 1.4 % | 1.0 | 0.01456 | 0.01988 | 2.6 % | 1.0 | 0.01493 | 0.0008604 | 0.382 |
| DDIM η=0 (= Euler in σ) | 50 | 0.0272 | 7.8 % | 1.0 | 0.02737 | 0.02877 | 7.5 % | 1.0 | 0.02802 | 0.0014 | 0.192 |
| DDIM η=0 (= Euler in σ) | 20 | 0.07288 | 32.0 % | 1.0 | 0.07852 | 0.0732 | 31.2 % | 1.0 | 0.07841 | 0.005751 | 0.0785 |
| DDIM η=0 (= Euler in σ) | 10 | 0.0901 | 46.9 % | 0.5 | 0.1521 | 0.08858 | 44.9 % | 0.5 | 0.1489 | 0.02337 | 0.0399 |
| DDIM η=0 (= Euler in σ) | 5 | 0.08132 | 38.4 % | 0.425 | 0.3187 | 0.08122 | 36.0 % | 0.4 | 0.3086 | 0.1008 | 0.02 |
| Euler, Karras σ grid | 100 | 0.0241 | 0.3 % | 1.0 | 0.01477 | 0.02474 | 1.1 % | 1.0 | 0.01517 | 0.0008879 | 0.381 |
| Euler, Karras σ grid | 50 | 0.02522 | 0.4 % | 1.0 | 0.03048 | 0.02638 | 1.8 % | 1.0 | 0.0311 | 0.001523 | 0.193 |
| Euler, Karras σ grid | 20 | 0.02962 | 1.5 % | 1.0 | 0.07858 | 0.03134 | 4.8 % | 1.0 | 0.07754 | 0.005683 | 0.0765 |
| Euler, Karras σ grid | 10 | 0.03887 | 10.9 % | 0.925 | 0.1617 | 0.04067 | 12.3 % | 0.95 | 0.1582 | 0.02323 | 0.0384 |
| Euler, Karras σ grid | 5 | 0.07618 | 34.1 % | 0.45 | 0.2753 | 0.07608 | 30.8 % | 0.45 | 0.276 | 0.07788 | 0.0194 |
| Euler in t (VP ODE) | 100 | 0.02694 | 2.1 % | 1.0 | 0.005338 | 0.02888 | 4.0 % | 1.0 | 0.006804 | 0.0007657 | 0.39 |
| Euler in t (VP ODE) | 50 | 0.03461 | 8.6 % | 1.0 | 0.01485 | 0.03648 | 9.2 % | 1.0 | 0.01683 | 0.0008225 | 0.193 |
| Euler in t (VP ODE) | 20 | 0.06688 | 28.9 % | 1.0 | 0.05018 | 0.06738 | 29.4 % | 1.0 | 0.05116 | 0.001394 | 0.0763 |
| Euler in t (VP ODE) | 10 | 0.07913 | 36.6 % | 1.0 | 0.06895 | 0.07938 | 36.9 % | 1.0 | 0.06958 | 0.002723 | 0.0391 |
| Euler in t (VP ODE) | 5 | 0.09912 | 45.2 % | 0.95 | 0.09901 | 0.09795 | 44.0 % | 0.925 | 0.09678 | 0.001059 | 0.02 |
| Heun, Karras σ grid | 99 | 0.02311 | 0.1 % | 1.0 | 0.002655 | 0.02388 | 1.0 % | 1.0 | 0.00425 | 0.0007684 | 0.387 |
| Heun, Karras σ grid | 49 | 0.02325 | 0.1 % | 1.0 | 0.01408 | 0.02362 | 0.8 % | 1.0 | 0.01597 | 0.001001 | 0.193 |
| Heun, Karras σ grid | 19 | 0.06424 | 20.4 % | 1.0 | 0.1092 | 0.06078 | 15.6 % | 1.0 | 0.1095 | 0.01221 | 0.0778 |
| Heun, Karras σ grid | 9 | 0.7908 | 93.3 % | 0.025 | 0.9404 | 0.688 | 89.8 % | 0.125 | 0.8465 | 0.9017 | 0.0354 |
| Heun, Karras σ grid | 5 | 4.76 | 99.8 % | 0.0 | 5.032 | 27.8 | 99.9 % | 0.0 | 28.67 | 1955.0 | 0.0194 |
| Heun, uniform t | 99 | 0.01661 | 2.8 % | 1.0 | 0.01343 | 0.0186 | 3.5 % | 1.0 | 0.01297 | 0.0007487 | 0.385 |
| Heun, uniform t | 49 | 0.05002 | 18.0 % | 1.0 | 0.04023 | 0.05224 | 19.1 % | 1.0 | 0.04145 | 0.001141 | 0.193 |
| Heun, uniform t | 19 | 0.09152 | 46.2 % | 0.925 | 0.1015 | 0.0916 | 46.0 % | 0.875 | 0.102 | 0.009594 | 0.0746 |
| Heun, uniform t | 9 | 0.08251 | 37.6 % | 0.45 | 0.2571 | 0.0882 | 41.2 % | 0.45 | 0.2916 | 0.0616 | 0.0376 |
| Heun, uniform t | 5 | 0.0858 | 41.9 % | 0.275 | 0.4215 | 0.3885 | 46.7 % | 0.475 | 1.003 | 1.842 | 0.02 |

**Same noise, same point?** (MLP; exact in brackets)
- DDIM is deterministic: running DDIM-100 twice gives identical points (max difference 0.0).
  DDIM-100 and DDIM-50 from the same noise land 0.01715 apart on average (0.01637), on the same arm
  97 % of the time; DDIM-10 lands 0.1379 away, same arm 63 %.
- DDPM forgets its starting noise: two runs from the same noise with different fresh noise land 0.9003 apart
  (0.9002), the same as two random swirl points (0.9053), and on the same arm
  50 % of the time, a coin flip.
- Trajectories of the first 8 noise points (≤ 200 states each): `samplers/<exact|mlp>/trajectories/<ddpm_1000|ddim_100|ddim_50|ddim_10|heun_karras_19>`
  (`xs` per particle, `ts` the DDPM step of each state); DDPM's clean guesses x̂₀ along the way: `samplers/exact/ddpm_x0_guesses`.
  End clouds (1,000 points): `samplers/<model>/clouds/<run>`.

**On screen:**
- "A thousand small noisy steps rebuild the swirl: the trained model's samples sit 0.02534 from the curve on
  average, the data itself 0.02405."
- "DDIM takes the same model and walks deterministically: 100 steps come close (2.6 % stranded vs
  0.7 % for DDPM's 1,000); at 50 steps 8 % of points
  get stranded between the arms, at 20 steps 31 %, at 10 steps 45 %."
- "Where you put the steps matters as much as how many: with Karras's spacing, 20 Euler steps leave only
  5 % stranded, and Heun with 49 model calls (0.8 %) is as clean as DDPM's 1,000."

**Caveats:**
- *dist below the data's 0.02405 is not better.* DDIM-100 scores 0.01988 because its last step returns the
  denoiser's average guess, which is tidier than real data (fine detail at the scale of the last step is averaged away;
  for images this is the slightly smooth look of deterministic samplers). DDPM with σ² = β̃ is also slightly too tidy
  (0.02238); σ² = β matches the data.
- *ode error vs quality:* Euler on the Karras grid has the same ODE error as DDIM at the same NFE, but far fewer stragglers.
  Its errors move points *along* the arms, which you cannot see; DDIM's big last jump (from σ = 0.17 at 20 steps) drops them
  *between* the arms.
- *Heun is not always better.* On the Karras grid it is the best solver from NFE ≈ 49 up (ODE error 0.01408 vs
  DDIM-50's 0.02737), but at NFE 9 and 5 it diverges (0.7908 and 4.76 from the curve): its
  second look at a very low σ, where the swirl's score is steep, overshoots. On the uniform t grid Heun is worse than
  DDIM at equal NFE, because half as many steps means a bigger final Euler jump.
- *Euler in t vs in σ:* on this toy, Euler on the VP ODE written in DDPM time is *not* worse than DDIM by ODE error
  (it is somewhat worse by stragglers at NFE ≥ 50). Don't claim on screen that the σ coordinate is what makes DDIM good;
  the robust lesson here is step placement.
- *fd on raw 2D coordinates is a poor judge:* Euler-in-t with 5 steps scores fd 0.001059 (as good as DDPM) while
  44 % of its points are off the swirl. See §7.

## 4. Flow matching vs the DDPM-trained model

Same noise, Euler steps evenly spaced in each model's own time (flow: t from 1 to 0; DDIM: DDPM steps from 1000 to 0).

| Euler steps | exact flow: dist | off | exact DDIM: dist | off | trained flow: dist | off | cov. | trained DDIM: dist | off | cov. |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.15 | 100.0 % | 0.1487 | 100.0 % | 0.1346 | 96.8 % | 0.05 | 2.305 | 95.8 % | 0.0 |
| 2 | 0.08457 | 40.4 % | 0.1078 | 62.4 % | 0.08431 | 40.4 % | 0.375 | 0.09863 | 42.2 % | 0.1 |
| 4 | 0.08938 | 45.6 % | 0.08281 | 38.5 % | 0.08977 | 46.2 % | 0.45 | 0.08222 | 38.2 % | 0.4 |
| 8 | 0.06415 | 25.8 % | 0.08881 | 45.6 % | 0.06416 | 25.1 % | 1.0 | 0.08813 | 46.5 % | 0.45 |
| 16 | 0.02553 | 6.0 % | 0.08149 | 36.8 % | 0.0265 | 6.8 % | 1.0 | 0.0807 | 35.5 % | 0.975 |
| 32 | 0.0205 | 1.1 % | 0.04609 | 17.2 % | 0.0218 | 2.4 % | 1.0 | 0.04693 | 15.6 % | 1.0 |

**Path straightness** (path length ÷ distance between its ends; 1 = a straight line):
- *Training paths* (what each loss sees; 5,000 (x₀, ε) pairs): flow matching's are exactly straight
  (1.0); DDPM's are arcs, x_t = √ᾱ·x₀ + √(1−ᾱ)·ε: median 1.056, mean 1.217,
  10 % above 1.494 (some loop when x₀ and ε point the same way). 8 of each: `flow/training_paths/paths_flow`, `paths_ddpm`.
- *Sampling paths* (200 Euler steps; trained models): flow median 1.561 (mean 2.161), DDPM ODE in its
  x_t picture median 1.137 (mean 1.537); exact fields: 1.56 vs
  1.075. Paths of 8 particles: `flow/<exact|trained>/paths_flow`, `paths_ddpm_ode`.
- With exact fields, flow Euler **is** DDIM on a different step grid (σ = t/(1 − t)): after 16 steps the two differ by at most
  0.0005337 (the self-checks test this). Both ODEs carry each noise point to the same place
  along the same route, drawn in different coordinates.

**On screen:** "Flow matching trains on straight lines from noise to data, while DDPM's training paths are arcs. Sampled with 8 or
more plain Euler steps, the flow model is much cleaner: with 8 steps 25 % of its points are stranded vs
46 % for the DDPM model, with 16 steps 7 % vs 36 %, with 32 steps 2 % vs 16 %.
One step sends every point to the middle, the average of the data (dist 0.15)."

**Caveats (important for Ep 11):**
- **The sampling paths are not straighter.** The flow model's actual sampling paths bend about as much as the DDPM ODE's
  (by length ÷ chord they bend *more* in the flow picture, median 1.561 vs 1.137), because every path first heads for the data's average and
  then turns toward its arm. Show "straight vs curved" with the *training* paths, or say "straighter on average, as trained".
  Truly straight sampling paths need reflow (rectified flow's second round) or distillation (Ep 12).
- With 2 and 4 steps both are poor (40–60 % stranded), and at 4 steps the DDPM model is slightly ahead; don't say "better at every budget".
- One DDIM step with the *trained* ε-model lands 2.305 from the swirl (the exact ε: 0.1487).
  The clean guess at t = 1000 is (x − √(1−ᾱ)·ε̂)/√ᾱ with √ᾱ = 0.0064, so a tiny error in ε̂ is magnified 157×. ε-prediction is
  ill-conditioned at pure noise; the flow model's velocity is not (one step: 0.1346). This is one reason for v-prediction and flow
  targets (Ep 5, 11). The same effect makes the trained model's Heun-5 runs blow up in §3.
- With perfect models the per-step advantage comes entirely from *where the steps fall*: uniform flow time puts more steps
  at low noise than uniform DDPM time under the linear β schedule. The trained models track the exact results closely,
  so on this toy the learned fields are not what differs.

## 5. Classifier-free guidance, exact

ε̃ = ε_u + w·(ε_c − ε_u), with both ε's exact (`dr.guided_eps(w, label)`). w is the *guidance scale* (Stable Diffusion,
diffusers): w = 0 ignores the label, w = 1 is the plain conditional model. Ho & Salimans' paper writes (1 + w′)·ε_c − w′·ε_u, so
**their w′ = this w − 1** (the course's w = 3 is their 2). DDIM, 100 steps, η = 0, same noise z for every w; DDPM-1000 (500 points) confirms the arm shares.

**The swirl, asking for arm 0:**

| w | on arm 0 | on arm 1 | off | coverage of arm 0 | arm-0 samples on its outer half | spread (√trace cov) | max radius | DDPM: on arm 0 |
|---|---|---|---|---|---|---|---|---|
| 0 | 49 % | 50 % | 1.4 % | 1.0 | 51 % | 0.7006 | 1.032 | 49 % |
| 1 | 100 % | 0 % | 0.0 % | 1.0 | 52 % | 0.6768 | 1.03 | 100 % |
| 3 | 100 % | 0 % | 0.0 % | 1.0 | 77 % | 0.6366 | 1.032 | 100 % |
| 7 | 100 % | 0 % | 0.0 % | 0.45 | 98 % | 0.4601 | 1.032 | 100 % |

Arm 0's own data: spread 0.6775, coverage 1.0. Where the samples sit along arm 0
(20 equal pieces, centre → tip, as shares of all samples; `guidance/profiles/<w>`): at w = 7, 98 % sit on the outer half
(data: 50 %), the part of arm 0 that looks least like arm 1, and the outermost piece (the tip) is nearly empty. Clouds: `guidance/clouds/<w>`.

**Three labelled blobs, asking for class 0 (the top blob):** (dr.blobs_mixture(): 3 round blobs, std 0.15, centres at radius 0.6 (90°, 210°, 330°); class 0 = top)

| w | class 0 | centre height (data 0.60) | std (data 0.1517) | beyond 2 std of the class | beyond 3 std |
|---|---|---|---|---|---|
| 0 | 34 % | 0.5945 | 0.1371 | 10 % | 0.7 % |
| 1 | 100 % | 0.6032 | 0.1361 | 9 % | 0.3 % |
| 3 | 100 % | 0.7877 | 0.09494 | 16 % | 0.8 % |
| 7 | 100 % | 0.9634 | 0.08244 | 86 % | 14.5 % |

Class 0's own data: 14 % beyond 2 std, 0.9 % beyond 3 std. Clouds: `guidance/blobs/clouds/<w>`,
data `guidance/blobs/data_cloud`.

**On screen:** "Guidance trades variety for certainty. At w = 1 we get arm 0, all of it. At w = 7 we still get arm 0,
but only its outer half (98 % of the samples): they crowd where the arm looks least like the other one, and their spread falls from
0.6768 to 0.4601. With blobs you can see the overshoot: at w = 7 the samples' centre moves from height 0.6032 to 0.9634, and
86 % of them lie outside where the class's own data lives (14 % for real data). They are pushed beyond the data,
away from the other classes. That is the over-saturated look of high guidance."

**Caveats:** on the swirl high guidance does *not* push points off the data (off stays 0, max radius 1.032): the thin arm
re-captures them at low noise, where the classifier is certain and its push vanishes. Use the blobs for "beyond the data".
The DDIM-100 samples are a little tidier than the data (blob std 0.1361 at w = 1 vs 0.1517), as in §3.

## 6. Langevin dynamics with the exact score

x ← x + (α/2)·∇log p_σ(x) + √α·z, starting from uniform noise in the square [−1.5, 1.5]², 2,000 particles, seed 11. Settings: annealed 10 noise levels σ = 1 → 0.01 (geometric), 100 steps each, ε = 2e-5 (α = ε·σ²/σ_L²);
plain σ = 0.01 only, 1,000 steps, α = 2e-5; blurry σ = 0.3 only, 1,000 steps, α = 2e-5·(0.3/0.01)² = 0.018.

| run | steps | dist | off | coverage | share on the outer half (data 50 %) | TV from even (data 0.0423) | arm 0 share |
|---|---|---|---|---|---|---|---|
| annealed, 10 levels σ = 1 → 0.01 | 1000 | 0.02592 | 0.1 % | 1.0 | 48 % | 0.0618 | 51 % |
| plain, σ = 0.01 | 1000 | 0.02554 | 0.2 % | 0.95 | 80 % | 0.296 | 48 % |
| plain, σ = 0.3 | 1000 | 0.1474 | 52.6 % | 0.975 | 40 % | 0.112 | 50 % |

Clouds and 8 particle paths (≤ 200 states): `langevin/clouds/<run>`, `langevin/paths/<run>`.

**On screen:** "Follow the exact arrows at one small noise level and every particle slides onto the nearest bit of
swirl, but the outer turns, which are closest to most starting points, collect 80 % of them instead of half. The proportions
are wrong. Start with big noise and turn it down slowly, and the particles
first spread over the right broad shape, then sharpen: 48 % on the outer half, the same as the data."

**Caveats:** with the *exact* score, plain Langevin at small σ still lands on the curve (dist 0.02554); its failure here is
the proportions. With a *learned* score it would also be steered wrongly in empty regions (Song & Ermon's other reason
for annealing); our MLP is accurate even there (§2), so this toy does not show that part. A single large σ samples the
blurred swirl (53 % off).

## 7. FID's formula on Gaussians

`dr.frechet_distance(μ1, C1, μ2, C2)` = ‖μ1 − μ2‖² + Tr(C1 + C2 − 2(C1·C2)^½). The matrix square root uses eigh on the
symmetric matrix √C1·C2·√C1, which has the same eigenvalues as C1·C2.

- Sanity: N((0,0), I) vs N((1, 0.5), diag(2, 0.5)): closed form ‖Δμ‖² + Σ(√a − √b)² = **1.5073593**, computed **1.5073593**. Rotating the second
  covariance by 0.6 rad (so the matrices no longer commute): 1.5073593 by our formula, 1.5073593 via eigenvalues of C1·C2.
- **Bias with few samples** (true distribution N(0, I); FD between the samples' mean/covariance and the truth, or between
  two sample sets of the same size; mean ± std over 200 (2D) / 40 (64-D) repeats). The true value is 0 in every row:

| dims | samples | FD(sample, truth) | FD(sample, sample) |
|---|---|---|---|
| 2 | 100 | 0.03698 ± 0.02422 | 0.07319 ± 0.04417 |
| 2 | 1,000 | 0.003579 ± 0.002483 | 0.007191 ± 0.004857 |
| 2 | 10,000 | 0.0003503 ± 0.0002418 | 0.0007324 ± 0.0004952 |
| 64 | 100 | 12.25 ± 0.3275 | 22.28 ± 0.5693 |
| 64 | 1,000 | 1.119 ± 0.02882 | 2.218 ± 0.06569 |
| 64 | 10,000 | 0.1109 ± 0.003412 | 0.2207 ± 0.006253 |

- **The blind spot:** a Gaussian with exactly the swirl's mean and covariance scores FD **0.0003315** against 10,000
  swirl points (another 10,000 real swirl points: 2.29e-05); both are inside the estimator's own noise at n = 10,000. Yet
  48 % of the Gaussian's points are off the swirl (dist 0.1212). Cloud: `fid/blind_spot/gaussian_cloud`.

**On screen:** "FID only compares averages and spreads. A shapeless blob with the swirl's average and spread scores as well as the
swirl itself. And it is biased: with 100 samples, two identical distributions already score 0.03698 in 2D. In 64 dimensions
the bias is 1.119 even with 1,000 samples, which is why FID is quoted on 50,000 images and only compared at equal sample counts."

**Caveats:** for N(0, I) the expected bias is ≈ (d + d(d + 1)/4)/n (0.0035 for 2-D and 1.1 for 64-D at n = 1,000, matching the
table): it falls as 1/n and grows like d², and Inception features have d = 2048. The two-sample estimate is about twice as
biased as comparing with exact statistics.

## Headline numbers (`measured("toy", "headline", key)`)

| key | value |
|---|---|
| `alpha_bar_500_linear` | 0.07859 |
| `alpha_bar_500_cosine` | 0.4938 |
| `alpha_bar_1000_linear` | 4.036e-05 |
| `t_snr_1_linear` | 259.1 |
| `t_snr_1_cosine` | 496.0 |
| `arm_accuracy_t500_linear` | 0.528 |
| `mlp_params` | 45954 |
| `mlp_train_seconds` | 95.1 |
| `loss_floor_eps` | 0.1581 |
| `final_loss_eps` | 0.1588 |
| `mlp_rel_score_error_t10` | 0.1541 |
| `mlp_rel_score_error_t500` | 0.00813 |
| `data_dist` | 0.02405 |
| `ddpm1000_exact_dist` | 0.02454 |
| `ddpm1000_mlp_dist` | 0.02534 |
| `ddim100_mlp_dist` | 0.01988 |
| `ddim50_mlp_off` | 0.075 |
| `ddim20_mlp_off` | 0.3125 |
| `ddim10_mlp_off` | 0.449 |
| `euler_karras20_mlp_off` | 0.0475 |
| `heun_karras49_mlp_off` | 0.0075 |
| `ddim100_vs_ddim50_same_arm_mlp` | 0.972 |
| `ddpm_seed1_vs_seed2_same_arm_mlp` | 0.499 |
| `training_path_ratio_ddpm_mean` | 1.217 |
| `training_path_ratio_flow_mean` | 1.0 |
| `flow8_trained_off` | 0.2505 |
| `ddim8_trained_off` | 0.465 |
| `flow16_trained_off` | 0.068 |
| `ddim16_trained_off` | 0.355 |
| `swirl_w7_coverage_arm0` | 0.45 |
| `swirl_w1_spread` | 0.6768 |
| `swirl_w7_spread` | 0.4601 |
| `blobs_w1_centre_y` | 0.6032 |
| `blobs_w7_centre_y` | 0.9634 |
| `blobs_w7_beyond_2sd` | 0.859 |
| `blobs_data_beyond_2sd` | 0.143 |
| `langevin_annealed_outer_share` | 0.481 |
| `langevin_plain_outer_share` | 0.796 |
| `fd_bias_2d_n100` | 0.03698 |
| `fd_bias_2d_n10000` | 0.0003503 |
| `fd_bias_64d_n1000` | 1.119 |
| `fd_swirl_vs_matched_gaussian` | 0.0003315 |


## General caveats

- Every number is from one fixed noise draw (z = dr.gaussian_noise(2000, seed=5)) or seed; sampler comparisons share the noise, so differences
  between rows are much more reliable than any single value. Re-running the scripts reproduces every number in toy.json
  except the wall times (checked; same NumPy build).
- The swirl is 2D and its data is an exact mixture, so "the exact denoiser" really exists here; for images it never does.
  Use the toy for mechanisms (which way arrows point, why steps strand points, what guidance does), not for image-model numbers.
- The DDPM step grids are "trailing" (S steps: t = round(linspace(1000, 0, S + 1))). Other spacings (diffusers' "leading")
  change the low-step numbers.

## Erratum (coordinator, 2026-10-26)
The forward-process snapshots stored in toy.json are the first 1,000 points of `swirl_labelled(20000, seed=0)` noised with `gaussian_noise(20000, seed=1)` — not `swirl(1000, seed=0)` as stated above (found by the Episode 4 agent, which asserted the real recipe against toy.json to within 5e-4).
