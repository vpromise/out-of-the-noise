# Out of the Noise: research notes (factual backbone)

**Fact date: October 2026** (compiled 2026-10-25). All web sources were read on 2026-10-25.
Fast-moving facts (model releases, product features, benchmark and leaderboard numbers, library defaults, licences) are tagged **[DATED]**. Anything not confirmed against a primary source is tagged **[UNVERIFIED]**. My own derivations are tagged **[DERIVED]**, with the arithmetic shown; every [DERIVED] number was re-computed with a short Python script (scratch folder /Users/vpromisex/.claude/jobs/61935fea/tmp/diff-research/scripts/: `schedules.py`, `derive.py`, `st_params.py`; not part of the course repo).

**Conventions**
- arXiv IDs are given as arXiv:YYMM.NNNNN; the URL is https://arxiv.org/abs/<id>. "v1" dates are the arXiv first-submission dates. Quotes were taken from the arXiv PDF version named in Appendix D (text extracted with pdftotext; spacing and ligatures normalised, wording unchanged).
- "Table N", "Fig. N", "eq. N", "Alg. N" refer to the paper's own numbering.
- FID is "lower is better"; IS, CLIP score, precision, recall are "higher is better". "FID-50K" = FID computed with 50,000 generated samples.
- **Steps vs NFE.** A *step* is one move along the sampling path; an *NFE* (number of function evaluations) is one call of the denoising network. Euler/DDIM: 1 NFE per step. Heun (EDM): 2 NFE per step except the last (2N − 1 total). Classifier-free guidance doubles the network work per step (a conditional and an unconditional prediction, usually run as one batch of 2). Always say which one a number counts.
- Parameter counts marked "(safetensors header)" were computed by summing tensor shapes in the official Hugging Face weight files (HTTP range request of the header only), 2026-10-25; "(HF metadata)" means the `safetensors.total` field of the Hugging Face Hub API for the official repo.
- The course's canonical algorithms and numbers are in `diffusion_reference.py`, and its measured runs are in `research/experiments/`. Where a worked example here and a measured run disagree, the measured run wins on screen and this file explains the textbook value.

**Notation** (one symbol per idea across the course; papers' own symbols are mapped in each section)

| Symbol | Meaning | Notes and conflicts |
|---|---|---|
| x₀ | a clean data sample (image, digit, swirl point) | DDPM's x₀. Rectified flow (Liu et al.) and Lipman et al. call the **data** X₁/x₁ and the **noise** X₀/x₀: the opposite. SD3 uses x₀ = data (like us). |
| x_t | the noisy sample at noise level t | VDM/Imagen/Salimans & Ho write **z_t** for this; we reserve z for the VAE latent. |
| ε | standard Gaussian noise, ε ~ N(0, I), same shape as x₀ | EDM writes n = σ·ε for the added noise. |
| t, T | timestep / noise level; T = number of discrete steps (DDPM: T = 1000) | **Direction conflict:** DDPM, score SDE, SD3: t = 0 is data, t = T (or 1) is noise. Lipman et al. (flow matching) and Liu et al. (rectified flow): t = 0 is noise, t = 1 is data. Continuous-time papers use t ∈ [0, 1]. We use t = 0 data → t = T (or 1) noise throughout. |
| β_t | per-step noise variance of the forward process (DDPM) | DDPM linear schedule 10⁻⁴ → 0.02. Score SDE: β(t) continuous, β(t) = 0.1 + 19.9t on [0, 1]. **Overloaded:** DPO's β (§20). |
| α_t | 1 − β_t (DDPM) | **Major conflict:** DDIM (Song, Meng & Ermon) writes α_t for DDPM's **ᾱ_t**. Kingma et al. (VDM), Salimans & Ho, Imagen, Lin et al. write α_t for the **signal coefficient √ᾱ_t**. Check every α before copying a formula. |
| ᾱ_t | ∏_{s≤t} α_s: the surviving signal *power*; x_t = √ᾱ_t·x₀ + √(1 − ᾱ_t)·ε | VP papers: α_t² = ᾱ_t, σ_t² = 1 − ᾱ_t. |
| σ_t | noise standard deviation at level t | **Three meanings:** (1) DDPM σ_t² is the *reverse-process* variance (β_t or β̃_t); (2) VP papers' σ_t = √(1 − ᾱ_t) is the forward noise std; (3) EDM/VE σ is the noise std **added to unscaled data**, σ_EDM = √((1 − ᾱ)/ᾱ) = 1/√SNR, ranging 0.002 → 80. Name which one. |
| SNR(t) | signal-to-noise ratio ᾱ_t / (1 − ᾱ_t) = α²/σ² (VP) = 1/σ_EDM² | λ = log SNR ("log-SNR"; Kingma et al., Salimans & Ho, DPM-Solver uses half of it). |
| ε_θ(x_t, t) | the network's noise prediction | DDPM, ADM. |
| s_θ(x, t) | the network's score estimate ≈ ∇ₓ log p_t(x) | **ε–score link (VP):** s_θ = −ε_θ / √(1 − ᾱ_t); EDM/VE: s = −ε/σ. **Overloaded "s":** cosine-schedule offset s = 0.008; ADM classifier-gradient scale s; GLIDE guidance scale s; SD3 logit-normal scale s. |
| x̂₀ / D_θ(x; σ) | the network's estimate of the clean image ("denoiser") | EDM calls it D_θ; Tweedie: x̂₀ = E[x₀ | x_t]. |
| v | **two different "v"s.** (1) Salimans & Ho's *v-prediction* target v = √ᾱ_t·ε − √(1 − ᾱ_t)·x₀ (their αε − σx). (2) The flow-matching / rectified-flow *velocity* dx_t/dt; for our interpolation x_t = (1 − t)x₀ + tε it is v = ε − x₀ | Lipman and Liu (t = 0 noise) get v = x₁ − x₀ = data − noise, the **opposite sign**. Say "v-prediction" or "velocity", never bare "v". |
| w | classifier-free guidance weight in the Ho & Salimans convention: ε̃ = (1 + w)ε_θ(x, c) − w·ε_θ(x) ; w = 0 = no guidance | **Guidance scale** s_cfg (GLIDE, Stable Diffusion, diffusers `guidance_scale`): ε̃ = ε_u + s(ε_c − ε_u); s = 1 + w; s = 1 = no guidance; SD default 7.5 ⇔ w = 6.5. COURSE.md uses w = 0…8 for digits (Ho–Salimans) and "guidance 1…15" for the lighthouse (scale). **Overloaded:** LoRA/weights W; CLIPScore's w = 2.5. |
| z | a latent (VAE-encoded image); SD 1.x: 512×512×3 → z of 64×64×4 | f = spatial downsampling factor (8); c or d = latent channels (4 for SD 1–XL, 16 for SD3/FLUX). SD3 also uses d for transformer **depth**. |
| c, y | the condition: class label y, or text c (its embedding τ_θ(c) in LDM; c_ctxt and c_vec in SD3) | ∅ = the "null" condition used for unconditional predictions. |
| q, p_θ | q = fixed forward (noising) process; p_θ = learned reverse process | |
| η | DDIM stochasticity: η = 0 deterministic DDIM, η = 1 DDPM-like | |
| N | number of sampling steps (EDM's N; DDIM's dim(τ) = S) | NFE ≠ N for Heun and for CFG. |
| ρ | EDM schedule exponent, ρ = 7 | |
| λ | log-SNR; also loss weights (iDDPM λ = 0.001; DreamBooth λ = 1) | |

**Running examples (from COURSE.md):** the swirl (2D spiral point cloud, numpy, `diffusion_reference.py`) · the digit model (small class-conditional diffusion model on MNIST, trained on this Mac) · the lighthouse (a real latent text-to-image model run on this Mac with the prompt "a lighthouse on a rocky coast at dusk, oil painting"). This file supplies the cited numbers around them; the measured numbers come from `research/experiments/`.

Seven numbers carry the course: **T = 1000, β 10⁻⁴ → 0.02** (DDPM's schedule; ᾱ_1000 = 4.0 × 10⁻⁵) · **FID 3.17** (DDPM, CIFAR-10, 2020) · **4.59 vs 6.95** (ADM-G vs BigGAN-deep, ImageNet 256, 2021: "diffusion beats GANs") · **1,000 → 50 → 10 steps: FID 4.04 → 4.67 → 13.36** (DDIM, CIFAR-10) · **7.5** (Stable Diffusion's default guidance scale = w 6.5) · **512×512×3 → 64×64×4** (48× fewer numbers; SD's latent) · **2.27** (DiT-XL/2, ImageNet 256, 2022).

---
## §0 Cheat sheet (the numbers and formulas worth memorising)

| # | Fact | Value | Source | § |
|---|---|---|---|---|
| 1 | Forward process in one line | x_t = √ᾱ_t·x₀ + √(1−ᾱ_t)·ε, ᾱ_t = ∏(1 − β_s) | Ho et al. 2020, eq. 4 | 4 |
| 2 | DDPM schedule | T = 1000, β linear 10⁻⁴ → 0.02; ᾱ_250 = 0.524, ᾱ_500 = 0.079, ᾱ_1000 = 4.0 × 10⁻⁵ (SNR = 1 at t ≈ 259) [DERIVED] | Ho et al. 2020 §4; Lin et al. 2023 Table 1 | 4 |
| 3 | Cosine schedule | ᾱ_t = f(t)/f(0), f = cos²(((t/T + s)/(1 + s))·π/2), s = 0.008, β ≤ 0.999; ᾱ_500 = 0.494 (SNR = 1 at t ≈ 496) | Nichol & Dhariwal 2021, eq. 17 | 4 |
| 4 | SD's schedule never reaches pure noise | SD scaled-linear 0.00085 → 0.012: SNR(T) = 0.004682 (linear 4.0 × 10⁻⁵) | Lin et al. 2023 Table 1 | 4, 14 |
| 5 | The training loss | L_simple = E‖ε − ε_θ(x_t, t)‖², t uniform | Ho et al. 2020, eq. 14, Alg. 1 | 5 |
| 6 | DDPM's result and cost | CIFAR-10 FID 3.17, IS 9.46; 35.7M-param U-Net; 800k steps, 10.6 h on a TPU v3-8 | Ho et al. 2020, abstract, App. B | 5, 8 |
| 7 | Score ↔ noise | ∇ log p_t(x_t) = −ε/√(1 − ᾱ_t); Tweedie: E[x₀|x_t] = (x_t + (1−ᾱ_t)∇log p_t)/√ᾱ_t | Dhariwal & Nichol eq. 11; Efron 2011 | 6 |
| 8 | DDPM sampling step | x_{t−1} = (1/√α_t)(x_t − (β_t/√(1−ᾱ_t))ε_θ) + σ_t z; σ_t² = β_t or β̃_t; 1,000 network calls | Ho et al. 2020, Alg. 2 | 7 |
| 9 | Probability-flow ODE | dx = [f − ½g²∇log p_t] dt: same marginals as the SDE, deterministic | Song et al. 2021, eq. 13 | 6, 9 |
| 10 | DDIM steps vs FID (CIFAR-10, η = 0) | 10 → 13.36; 20 → 6.84; 50 → 4.67; 100 → 4.16; 1000 → 4.04 | Song, Meng & Ermon 2021, Table 1 | 9 |
| 11 | EDM | σ_i = (σ_max^{1/ρ} + i/(N−1)(σ_min^{1/ρ} − σ_max^{1/ρ}))^ρ, ρ = 7, σ 0.002–80; Heun; CIFAR-10 FID 1.79 at 35 NFE (18 steps) | Karras et al. 2022 | 10 |
| 12 | DPM-Solver | CIFAR-10 FID 4.70 at 10 NFE, 2.87 at 20 NFE; DPM-Solver++ 15–20 steps with guidance | Lu et al. 2022a/b | 10 |
| 13 | Diffusion beats GANs | ImageNet 256: ADM-G FID 4.59 vs BigGAN-deep 6.95 (128: 2.97 vs 6.02); recall 0.52 vs 0.28 | Dhariwal & Nichol 2021, Table 5 | 3, 13 |
| 14 | Classifier-free guidance | ε̃ = ε_u + s(ε_c − ε_u) = (1 + w)ε_c − wε_u, s = 1 + w; SD default s = 7.5 (w = 6.5); condition dropped 10–20% | Ho & Salimans 2021/22; CompVis README | 14 |
| 15 | CFG trade-off (ImageNet 128) | w = 0: FID 7.27, IS 82; w = 0.3: FID 2.43 (best), IS 158; w = 4: FID 21.5, IS 421 | Ho & Salimans Table 2 | 14 |
| 16 | Flow matching / rectified flow | x_t = (1−t)x₀ + tε, velocity v = ε − x₀, loss ‖v_θ − (ε − x₀)‖² | Liu et al. 2022; Lipman et al. 2022; Esser et al. 2024 | 11 |
| 17 | SD3 | rectified flow, logit-normal(0, 1) t-sampling, MM-DiT up to 8B (d = 38), 16-channel latent, 3 text encoders, shift 3.0 at 1024² | Esser et al. 2024 | 11, 18 |
| 18 | Latent diffusion | 512×512×3 = 786,432 → 64×64×4 = 16,384 (48×); scale factor 0.18215; KL weight ~10⁻⁶ | Rombach et al. 2022; SD v1 config | 17 |
| 19 | Stable Diffusion sizes | SD 1.5: U-Net 859.5M + CLIP-L 123M + VAE 84M (≈ 2.06 GB fp16 for text+U-Net+decoder); SDXL U-Net 2.57B; FLUX.1 11.9B + T5-XXL 4.76B (33.7 GB bf16) | safetensors headers; CompVis README; SDXL paper | 8, 17, 23 |
| 20 | DiT | DiT-XL/2: 675M params, 118.6 Gflops, FID 9.62 unguided, **2.27** with CFG 1.5 (ImageNet 256) | Peebles & Xie 2022, Tables 2, 4 | 18 |
| 21 | Distillation | progressive: CIFAR-10 FID 3.00 in 4 steps (from 8192); consistency: 3.55 in 1 step; ADD: SDXL-quality in 1–4 steps, 512² | Salimans & Ho 2022; Song et al. 2023; Sauer et al. 2023 | 12 |
| 22 | Text encoders | CLIP: 400M pairs, contrastive, 77 tokens; SD 1.x: 77 × 768; Imagen: scaling T5 (4.6B) "much more" than scaling the U-Net; COCO FID 7.27 | Radford et al. 2021; Saharia et al. 2022 | 15 |
| 23 | FID | ‖Δm‖² + Tr(C₁ + C₂ − 2(C₁C₂)^{1/2}) on 2,048-d Inception pool features, 50k samples; biased by N | Heusel et al. 2017; Chong & Forsyth 2020 | 19 |
| 24 | Data and captions | LAION-5B: 5.85B pairs (2.32B English); DALL·E 3 trained on 95% synthetic descriptive captions | Schuhmann et al. 2022; Betker et al. 2023 | 20 |
| 25 | Memorization | SD: 109 near-copies from 175M generations of the 350K most-duplicated prompts; CIFAR-10 models: 1,280 | Carlini et al. 2023 | 20 |
| 26 | Video | Sora report 15 Feb 2024 ("a minute of high fidelity video", spacetime patches); Movie Gen 30B, 73K tokens = 16 s at 16 fps | OpenAI 2024; Meta 2024 | 21 |
| 27 | Mac speeds | SD 1.4, 50 steps, 512²: 9 s on M1 Ultra (Core ML, 2022); SDXL 20 steps 1024²: 20 s on M2 Ultra; FLUX schnell 2 steps 1024²: ~19 s on M4 Max (mflux, incl. loading) | Apple ml-stable-diffusion README; mflux README | 23 |

---

## §1 What generative image models do, and a dated timeline

### Mechanism (plain words)
- **Recognition maps a picture to a label; generation goes the other way.** A generative model learns the distribution of images (optionally given a label or a prompt) well enough to *sample* new ones: pictures that look like the training data without being copies of it.
- **The diffusion recipe in one breath.** Destroying an image with Gaussian noise is easy and needs no learning (§4). A network learns to undo a little noise at a time (§5). Starting from pure noise and applying it many times produces a new image (§7). Everything else in the course is a refinement: fewer steps (§9–§12), steering (§13–§16), scaling (§17–§18), video (§21–§22).
- **The 2020–2024 arc.** DDPM (2020) showed the recipe gives high-quality images; ADM (2021) beat GANs on ImageNet with better architectures and guidance; latent diffusion (Dec 2021) made it cheap enough to open-source as Stable Diffusion (Aug 2022); classifier-free guidance and big text encoders made text-to-image work; transformers replaced the U-Net (DiT 2022, SD3 2024, FLUX 2024); video followed (Sora report Feb 2024).
- **The three running examples** each isolate one layer: the swirl (the maths in 2D), the digit model (training and sampling a real model on this Mac), the lighthouse (a full latent text-to-image pipeline on this Mac).

### Key formulas
- Generation as sampling: draw x ~ p_θ(x) (or p_θ(x | c)) with p_θ ≈ p_data.
- The denoising loop: x_T ~ N(0, I); for t = T…1: x_{t−1} = step(x_t, ε_θ(x_t, t, c)); output x₀ (§7, §9).

### Timeline (arXiv v1 dates via the arXiv API; product dates from official posts; tags as noted)
| Date | Event | Headline fact | Source |
|---|---|---|---|
| 2013-12-20 | **VAE** (Kingma & Welling) | Variational lower bound + reparameterization trick | arXiv:1312.6114 |
| 2014-06-10 | **GAN** (Goodfellow et al.) | "a minimax two-player game" | arXiv:1406.2661 |
| 2015-03-12 | **Diffusion probabilistic models** (Sohl-Dickstein, Weiss, Maheswaranathan & Ganguli) | "systematically and slowly destroy structure … through an iterative forward diffusion process"; T = 1000; CIFAR-10 4.20 bits/dim | arXiv:1503.03585; PixelRNN Table 5 |
| 2015-05-18 | **U-Net** (Ronneberger, Fischer & Brox) | contracting + expanding path with skips; later the diffusion backbone | arXiv:1505.04597 |
| 2016-01-25 / 2016-06-16 | **PixelRNN / PixelCNN** (van den Oord et al.) | autoregressive pixels; CIFAR-10 3.00 / 3.03 bits/dim | arXiv:1601.06759, 1606.05328 |
| 2017-06-26 | **FID** (Heusel et al.) | the standard sample-quality metric | arXiv:1706.08500 |
| 2018-09-28 / 2018-12-12 | **BigGAN** / **StyleGAN** | ImageNet 128 FID 7.4, IS 166.5 / FFHQ faces | arXiv:1809.11096, 1812.04948 |
| 2019-07-12 | **NCSN** (Song & Ermon) | score matching at 10 noise levels + annealed Langevin; CIFAR-10 IS 8.87 | arXiv:1907.05600 |
| 2020-06-19 | **DDPM** (Ho, Jain & Abbeel) | ε-prediction + simple loss; CIFAR-10 FID 3.17 | arXiv:2006.11239 |
| 2020-10-06 | **DDIM** (Song, Meng & Ermon) | deterministic sampling, "10× to 50× faster" | arXiv:2010.02502 |
| 2020-11-26 | **Score SDE** (Song et al.) | VP/VE SDEs, probability-flow ODE; CIFAR-10 FID 2.20; 1024² samples | arXiv:2011.13456 (ICLR 2021) |
| 2021-02-18 | **Improved DDPM** (Nichol & Dhariwal) | cosine schedule, learned variances, 100-step sampling | arXiv:2102.09672 |
| 2021-02-26 | **CLIP** (Radford et al.) | 400M image–text pairs, contrastive | arXiv:2103.00020 |
| 2021-05-11 | **"Diffusion Models Beat GANs"** (Dhariwal & Nichol) | ADM-G ImageNet 256 FID 4.59 vs BigGAN-deep 6.95 | arXiv:2105.05233 |
| 2021-12 (workshop) / 2022-07-26 (arXiv) | **Classifier-free guidance** (Ho & Salimans) | guidance without a classifier | NeurIPS 2021 DGM workshop; arXiv:2207.12598 |
| 2021-12-20 | **GLIDE** (Nichol et al.) | 3.5B text-to-image with CFG; preferred to DALL·E 87% (photorealism) | arXiv:2112.10741 |
| 2021-12-20 | **Latent Diffusion** (Rombach et al.) | diffusion in an f = 8 autoencoder latent; cross-attention conditioning | arXiv:2112.10752 (CVPR 2022) |
| 2022-04-13 | **DALL·E 2** (unCLIP, Ramesh et al.) | CLIP-image-embedding prior + 3.5B diffusion decoder; COCO FID 10.39 | arXiv:2204.06125 |
| 2022-05-23 | **Imagen** (Saharia et al.) | frozen T5-XXL; COCO FID 7.27 | arXiv:2205.11487 |
| 2022-06-01 / 06-02 | **EDM** / **DPM-Solver** | 35-NFE FID 1.79 / 10–20-step sampling | arXiv:2206.00364, 2206.00927 |
| 2022-08-22 | **Stable Diffusion v1.4 public release** (CompVis with Stability AI compute and LAION data) | open weights, CreativeML OpenRAIL-M, 860M U-Net [DATED] | Stability AI release post (helper S24); CompVis README |
| 2022-09-07 / 10-06 | **Rectified flow** / **Flow matching** | straight-line noise↔data paths | arXiv:2209.03003, 2210.02747 |
| 2022-09-29 / 10-05 | **Make-A-Video** / **Imagen Video** | first large text-to-video diffusion systems | arXiv:2209.14792, 2210.02303 |
| 2022-12-19 | **DiT** (Peebles & Xie) | transformer denoiser; ImageNet 256 FID 2.27 | arXiv:2212.09748 |
| 2023-02-10 | **ControlNet** (Zhang, Rao & Agrawala) | spatial control via zero convolutions | arXiv:2302.05543 |
| 2023-03-02 | **Consistency models** (Song, Dhariwal, Chen & Sutskever) | one-step CIFAR-10 FID 3.55 | arXiv:2303.01469 |
| 2023-07-04 / 07-26 | **SDXL** paper / SDXL 1.0 release | 2.6B U-Net, 1024², refiner | arXiv:2307.01952; Stability post [DATED] |
| 2023-10-03 | **DALL·E 3** system card (paper "Improving Image Generation with Better Captions") | recaptioning: 95% synthetic captions | OpenAI (helper S29–S30); announcement page 20 Sep 2023 **[UNVERIFIED]** |
| 2023-11-21 / 11-25 | **Stable Video Diffusion** release / paper | 14/25 frames at 576×1024 | Stability post; arXiv:2311.15127 |
| 2023-11-28 | **SDXL Turbo** (ADD, Sauer et al.) | 1–4-step generation, 512² | arXiv:2311.17042 |
| 2024-02-15 | **Sora** technical report (OpenAI) | spacetime patches; "a minute of high fidelity video" | openai.com (via Wayback) |
| 2024-02-22 / 03-05 | **SD3** announcement / paper | rectified-flow MM-DiT up to 8B | Stability post; arXiv:2403.03206 |
| 2024-05-14 | **Veo** and **Imagen 3** announced (Google I/O) | 1080p video "beyond a minute" | Google blog [DATED] |
| 2024-06-12 | **SD3 Medium** open weights | 2B MM-DiT | Stability post [DATED] |
| 2024-08-01 | **FLUX.1** [pro]/[dev]/[schnell] (Black Forest Labs) | 12B flow-matching transformer; schnell Apache-2.0, 1–4 steps | bfl.ai blog [DATED] |
| 2024-10 | **Movie Gen** (Meta; paper dated 4 Oct, arXiv 17 Oct) · **SD 3.5** (22 Oct) | 30B video model, 16 s at 16 fps; SD 3.5 Large 8.1B | arXiv:2410.13720; Stability post [DATED] |
| 2024-12 | **HunyuanVideo** (3 Dec, 13B open) · **Sora Turbo** public (9 Dec) · **Veo 2** (16 Dec, up to 4K) | open and closed video models | GitHub; OpenAI; Google [DATED] |
| 2025-02-25 | **Wan 2.1** (Alibaba) | 1.3B/14B open video models, Apache 2.0 | GitHub [DATED] |
| 2025-03-25 | **GPT-4o image generation** (OpenAI) | "an autoregressive model natively embedded within ChatGPT" (not diffusion, per OpenAI) | GPT-4o system card addendum [DATED] |
| 2025-05-20 | **Veo 3** (native audio) and **Imagen 4** ("a latent diffusion model") | joint audio–video latent diffusion | Google blog; Veo 3 tech report; Imagen 4 model card [DATED] |
| 2025-08-05 | **Genie 3** (DeepMind) | real-time 720p, 24 fps interactive worlds | DeepMind blog [DATED] |
| 2025-09-30 | **Sora 2** | "our flagship video and audio generation model" | OpenAI (via Wayback) [DATED] |
| 2025-11-25 | **FLUX.2** (BFL) | [dev] 32B open-weight + 24B Mistral-3 text encoder | bfl.ai blog [DATED] |
| 2026-01-15 | **FLUX.2 [klein]** 4B/9B | step-distilled to 4 steps; 4B Apache-2.0 | bfl.ai blog [DATED] |
| 2026-05-19 | **Gemini Omni Flash** (Google) | replaces Veo models in the Gemini API; architecture not disclosed | Google blog; Gemini API deprecations [DATED] |
| 2026-07-23 | **FLUX 3** (BFL, early access) | multimodal (image, video, audio) "flow matching model" | bfl.ai blog, checked 2026-10-25 [DATED] |
| 2026-09-24 | Sora 2 / Videos API removed from the OpenAI API (notified 24 Mar 2026) | | OpenAI deprecations page, checked 2026-10-25 [DATED] |
| 2026-10-15 | **FLUX 3 Image Klein** (5.6B) · **Kling 4.0** (30-s clips) | latest entries at fact date | bfl.ai blog; Kuaishou release via reproduction [DATED] |

### Numbers with sources
- The canonical first-paper quotes are in §3 (Sohl-Dickstein), §5 (DDPM) and §3 (ADM).
- Course scope check (COURSE.md episode 1 history list: "GANs, VAEs, DDPM 2020, Stable Diffusion 2022, video 2024"): consistent with the table. Two refinements for the script: diffusion models date from **2015** (DDPM made them work in 2020); Stable Diffusion's *paper* is Dec 2021 (CVPR 2022), its public *release* 22 Aug 2022.

### Worked example [DERIVED]
**The lighthouse in 30 steps (episode 1's opening):** with a 30-step DDIM-style sampler on SD 1.5's 1,000-step schedule, the visited timesteps are every 33rd (diffusers "leading" spacing: 958, 925, …, 34, 1); each frame shown is the decoded x̂₀ estimate at that step. 30 steps × 2 CFG passes = 60 U-Net evaluations. Measured time and frames come from `research/experiments/`.

### Common misconceptions
- "Diffusion = Stable Diffusion." Stable Diffusion is one (latent) diffusion model; DALL·E 2/3, Imagen, Sora, Veo and FLUX are others (FLUX describes itself as flow matching; GPT-4o image generation is described by OpenAI as autoregressive).
- "Every 2025–2026 image model is a diffusion model." Several frontier systems do not state their architecture (Gemini image models, GPT Image), and OpenAI states GPT-4o image generation is autoregressive. Say "most open image and video models are diffusion or flow-matching transformers".
- "Diffusion was invented by Stability AI / OpenAI / Google." The 2015 paper lists Stanford University and UC Berkeley (Sohl-Dickstein et al.); DDPM lists UC Berkeley (Ho, Jain, Abbeel); NCSN lists Stanford University (Song & Ermon).

### Quotable
- "Creating noise from data is easy; creating data from noise is generative modeling." (Song et al. 2021, abstract)
- "We show that diffusion models actually are capable of generating high quality samples, sometimes better than the published results on other types of generative models (Section 4)." (Ho et al. 2020, §1)

---
## §2 Images as tensors: dimensionality, the manifold idea, and the datasets

### Mechanism (plain words)
- **A picture is a grid of numbers.** A greyscale 28×28 digit is 784 numbers; a colour image is height × width × 3 (red, green, blue). Stored images use integers 0–255. Diffusion models rescale them to [−1, 1] so that "data" and "noise" live on comparable scales.
- **An image is one point in a huge space.** A 512×512 colour picture is a single point in a space with 786,432 axes. A generator has to produce points in that space that look like real pictures.
- **Real pictures are rare.** Almost every point in that space is static. Pick each pixel at random and you will never see a digit, a face or a lighthouse. Real images crowd onto a thin, low-dimensional region (the **manifold hypothesis**). The swirl is the 2D version: a thin spiral inside a plane that is mostly empty.
- **Why that makes generation hard.** We must learn where that thin region is from examples alone, then land new samples on it. A model that only learns "the average image" lands in the empty space between digits (a blurry average; see §3).
- **Why noise helps (preview of §6).** Adding Gaussian noise "thickens" the thin manifold until it fills the whole space. At high noise every point is near *some* data. Song & Ermon (2019) built their method on exactly this observation.

### Key formulas
- Pixel scaling (DDPM §3.3): x = 2·(k/255) − 1 for integer k ∈ {0, …, 255}, so x ∈ [−1, 1]. One grey level is 2/255 ≈ 0.0078 in these units.
- Dimension of an H×W image with C channels: D = H·W·C.
- A data distribution p_data(x) over ℝ^D; a generator is anything that produces samples x ~ p_θ(x) ≈ p_data(x).

### Numbers with sources
**Dimensions** [DERIVED: H·W·C]
| Image | Numbers per image |
|---|---|
| MNIST digit 28×28×1 | 784 |
| CIFAR-10 32×32×3 | 3,072 |
| ImageNet 64×64×3 | 12,288 |
| 256×256×3 | 196,608 |
| 512×512×3 (SD 1.x output) | 786,432 |
| 1024×1024×3 (SDXL, SD3, FLUX output) | 3,145,728 |
| SD 1.x latent 64×64×4 | 16,384 |

**MNIST** (LeCun, Cortes & Burges; page http://yann.lecun.com/exdb/mnist/ read via the Wayback Machine 2024 snapshot, because the live URL now shows a bare directory listing **[DATED]**):
- "The MNIST database of handwritten digits, available from this page, has a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image."
- "The original black and white (bilevel) images from NIST were size normalized to fit in a 20x20 pixel box while preserving their aspect ratio. The resulting images contain grey levels as a result of the anti-aliasing technique used by the normalization algorithm. the images were centered in a 28x28 image by computing the center of mass of the pixels".
- Built from NIST Special Database 3 (Census Bureau employees) and Special Database 1 (high-school students). "The 60,000 pattern training set contained examples from approximately 250 writers." Training and test writers are disjoint.
- **Licence: sources disagree (Appendix A).** The LeCun page states no licence. Keras docs (https://keras.io/api/datasets/mnist/): "Yann LeCun and Corinna Cortes hold the copyright of MNIST dataset, which is a derivative work from original NIST datasets. MNIST dataset is made available under the terms of the Creative Commons Attribution-Share Alike 3.0 license." The Hugging Face card for `ylecun/mnist` says "MIT Licence". **[UNVERIFIED]** which is authoritative; on screen say "freely available for research; CC BY-SA 3.0 per Keras".
- [DERIVED] 60,000 × 784 bytes = 47,040,000 bytes (47 MB) as uint8; 188 MB as float32.

**CIFAR-10** (Krizhevsky 2009; https://www.cs.toronto.edu/~kriz/cifar.html): "The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images."

**ImageNet** (https://www.image-net.org/, read 2026-10-25): "14,197,122 images, 21841 synsets indexed". The ILSVRC-2012 subset used for class-conditional generation (Russakovsky et al. 2015, arXiv:1409.0575, Table 2): **1,281,167** training images (732–1,300 per class), **50,000** validation (50 per class), **100,000** test, 1,000 classes. Generative papers train at 64×64, 128×128, 256×256 or 512×512 by downsampling/cropping.

**The manifold hypothesis**
- Song & Ermon 2019 (NCSN, arXiv:1907.05600, §3.1): "The manifold hypothesis states that data in the real world tend to concentrate on low dimensional manifolds embedded in a high dimensional space (a.k.a., the ambient space)."
- Pope, Zhu, Abdelkader, Goldblum & Goldstein 2021, "The Intrinsic Dimension of Images and Its Impact on Learning" (ICLR 2021, arXiv:2104.08894): "we estimate that ImageNet, despite containing 224 × 224 × 3 = 150528 pixels per image, only has intrinsic dimension between 26 and 43". Table 1 (MLE estimates, k = 3 / 5 / 10 / 20 neighbours): **MNIST 7 / 11 / 12 / 13**; SVHN 9 / 14 / 18 / 19; CIFAR-100 11 / 18 / 22 / 23; CelebA 9 / 17 / 24 / 26; **CIFAR-10 13 / 21 / 25 / 26**; MS-COCO 22 / 33 / 37 / 36; **ImageNet 26 / 38 / 43 / 43**.

### Worked example [DERIVED]
1. **A digit as numbers.** A 28×28 MNIST "7" unrolls row by row into 784 integers. A white stroke pixel (255) becomes +1.0, the black background (0) becomes −1.0, a grey edge pixel (128) becomes 2·128/255 − 1 = +0.004.
2. **How many possible images.** 8-bit 28×28 greyscale images: 256^784 = 10^1888.1. Even binary 28×28 images: 2^784 = 10^236. MNIST has 7 × 10⁴ of them. Uniformly random points are static, never digits.
3. **The swirl as a stand-in.** The swirl is a 1D curve (intrinsic dimension 1) in a 2D plane (ambient dimension 2). MNIST is roughly "dimension 7–13 inside 784" (Pope et al.). The picture of a thin curve in an empty plane is the right intuition; only the numbers differ.

### Common misconceptions
- "A generator stores and retrieves training images." It learns a distribution; memorization is a measurable failure mode, rare but real (§20, Carlini et al. 2023).
- "Higher resolution means proportionally harder." Ambient dimension grows with pixel count, intrinsic dimension much less (ImageNet ≈ 26–43 at 224×224×3). This is also why latent diffusion (§17) works: most pixels are redundant.
- "MNIST is 70,000 digits by 250 people." The 250-writer figure is for the 60,000-image **training** set only.
- "Pixel values are 0–1 in diffusion models." DDPM-style models use [−1, 1]; NCSN used [0, 1]; latent models diffuse the VAE latent, scaled to unit variance (§17).

### Quotable
- "The manifold hypothesis states that data in the real world tend to concentrate on low dimensional manifolds embedded in a high dimensional space" (Song & Ermon 2019, §3.1).
- "natural image datasets whose images contain thousands of pixels can, in fact, be described by orders of magnitude fewer variables." (Pope et al. 2021, §1)
- "We assume that image data consists of integers in {0, 1, . . . , 255} scaled linearly to [−1, 1]." (Ho, Jain & Abbeel 2020, §3.3)

---

## §3 Before diffusion: autoencoders, VAEs, GANs, autoregressive pixels, and the "diffusion beats GANs" moment

### Mechanism (plain words)
- **Autoencoder.** An encoder squeezes an image into a short code; a decoder rebuilds the image. It compresses but does not generate: random codes decode to garbage because nothing makes the code space "full".
- **VAE** (Kingma & Welling, Dec 2013). The encoder outputs a *distribution* over codes, and a KL penalty pulls it toward N(0, I). Sampling z ~ N(0, I) then decoding gives new images. Training maximises the ELBO, a lower bound on log-likelihood. With a Gaussian pixel likelihood the decoder is penalised by squared error, so when it is unsure it predicts the *average* of the plausible images: **blur**.
- **GAN** (Goodfellow et al., Jun 2014). A generator G maps noise to images; a discriminator D tries to tell real from fake; they play a minimax game. Samples are sharp because D punishes anything that looks unreal. But training is unstable, and G can **mode-collapse**: cover a few kinds of output (one arm of the swirl) and ignore the rest, because D only judges realism, not coverage.
- **Autoregressive pixels** (PixelRNN/PixelCNN, 2016). Generate one pixel (or sub-pixel) at a time, each conditioned on all previous ones. Exact likelihoods and good coverage, but one network call per pixel: 3,072 sequential steps for a 32×32 colour image.
- **Diffusion** (Sohl-Dickstein et al., Mar 2015; made competitive by DDPM, Jun 2020). Train one network to undo a small amount of noise, at every noise level. Training is plain regression (stable, no adversary). Sampling covers the distribution (good recall). The price is many network calls per sample (§7, §9–§12).
- **Why diffusion won (as of 2021–2022).** It combined GAN-level sharpness with likelihood-model coverage and stable training, and it scaled. The cost (slow sampling) was then attacked by faster samplers (§9–§10) and distillation (§12).

### Key formulas
- **VAE ELBO** (Kingma & Welling 2013, eq. 3 in their notation): log p_θ(x) ≥ E_{q_φ(z|x)}[log p_θ(x|z)] − D_KL(q_φ(z|x) ‖ p(z)). The reparameterization trick: z = μ_φ(x) + σ_φ(x)·ε, ε ~ N(0, I).
- **GAN minimax** (Goodfellow et al. 2014, eq. 1): min_G max_D V(D, G) = E_{x~p_data}[log D(x)] + E_{z~p_z}[log(1 − D(G(z)))]. At the optimum "D equal to ½ everywhere".
- **Autoregressive factorisation:** p(x) = ∏_{i=1}^{D} p(x_i | x_1, …, x_{i−1}).
- **Diffusion's bound** (Sohl-Dickstein 2015; DDPM eq. 3): E[−log p_θ(x₀)] ≤ E_q[−log p(x_T) − Σ_{t≥1} log (p_θ(x_{t−1}|x_t)/q(x_t|x_{t−1}))]. A diffusion model is a VAE with a fixed, parameter-free encoder (the noising process) and a T-step decoder.

### Numbers with sources
**Dates (arXiv v1).** VAE: Kingma & Welling, "Auto-Encoding Variational Bayes", arXiv:1312.6114, Dec 2013. GAN: Goodfellow et al., "Generative Adversarial Nets", arXiv:1406.2661, Jun 2014 (NeurIPS 2014). Diffusion: Sohl-Dickstein, Weiss, Maheswaranathan & Ganguli, "Deep Unsupervised Learning using Nonequilibrium Thermodynamics", arXiv:1503.03585, Mar 2015 (ICML 2015). PixelRNN: van den Oord, Kalchbrenner & Kavukcuoglu, arXiv:1601.06759, Jan 2016. Gated PixelCNN: van den Oord et al., "Conditional Image Generation with PixelCNN Decoders", arXiv:1606.05328, Jun 2016.

**Likelihoods (CIFAR-10 test, bits/dim, lower is better).** PixelRNN Table 5: uniform 8.00; Sohl-Dickstein's 2015 "Deep Diffusion" **4.20**; original PixelCNN 3.14; Row LSTM 3.07; **Diagonal BiLSTM 3.00** (training 2.93). Gated PixelCNN **3.03** (PixelCNN-decoders paper, Table 1; training 2.90). DDPM L_simple ≤ 3.75, DDPM with the true bound ≤ 3.70 (DDPM Table 1); iDDPM 2.94 (Nichol & Dhariwal Table 3). Diffusion was *not* the likelihood winner; its case was sample quality.

**GAN quality milestones.**
- BigGAN (Brock, Donahue & Simonyan, arXiv:1809.11096, Sep 2018), abstract: "When trained on ImageNet at 128×128 resolution, our models (BigGANs) achieve an Inception Score (IS) of 166.5 and Fréchet Inception Distance (FID) of 7.4, improving over the previous best IS of 52.52 and FID of 18.65." Also 256×256: IS 232.5 / FID 8.1; 512×512: IS 241.5 / FID 11.5. The **truncation trick** trades variety for fidelity.
- StyleGAN (Karras, Laine & Aila, arXiv:1812.04948, Dec 2018): introduced FFHQ (70,000 faces at 1024×1024); FID 4.40 on FFHQ (config F, Table 1). Its generator "has a total of 26.2M trainable parameters" (Fig. 1 caption).
- StyleGAN2 (Karras et al., arXiv:1912.04958, Dec 2019): FFHQ FID **2.84** (config F, Table 1).
- StyleGAN-XL (Sauer, Schwarz & Geiger, arXiv:2202.00273, Feb 2022): ImageNet 256 FID **2.30**, 128 FID 1.81, 512 FID 2.41 (Table 2). A GAN briefly beat ADM-G-U (3.94) on ImageNet 256 in early 2022, so "diffusion beats GANs" was a 2021 result, not a permanent law.
- GAN mode collapse, in the original paper's words: G must not be trained too much without updating D, "in order to avoid “the Helvetica scenario” in which G collapses too many values of z to the same value of x to have enough diversity to model p_data" (Goodfellow et al. 2014, §6).

**The "diffusion beats GANs" table** (Dhariwal & Nichol, "Diffusion Models Beat GANs on Image Synthesis", arXiv:2105.05233, May 2021, Table 5; FID / sFID / precision / recall; ImageNet models sampled with 250 steps unless "25 steps"):

| Task | BigGAN-deep | ADM (no guidance) | ADM-G (classifier guidance) | ADM-G 25 DDIM steps |
|---|---|---|---|---|
| ImageNet 128×128 | 6.02 (P 0.86, R 0.35) | 5.91 | **2.97** (P 0.78, R 0.59) | 5.98 |
| ImageNet 256×256 | 6.95 (P 0.87, R 0.28) | 10.94 | **4.59** (P 0.82, R 0.52) | 5.44 |
| ImageNet 512×512 | 8.43 (P 0.88, R 0.29) | 23.24 | **7.72** (P 0.87, R 0.42) | 8.41 |
| ImageNet 64×64 | 4.06 (own BigGAN-deep) | 2.61 (2.07 with dropout) | n/a | n/a |
| LSUN Bedrooms 256 (uncond.) | StyleGAN 2.35 | 1.90 (dropout) | n/a | n/a |

- With an upsampling stack (ADM-G + ADM-U): ImageNet 256 FID **3.94**, ImageNet 512 FID **3.85** (abstract; Table 6).
- Abstract (exact): "We achieve an FID of 2.97 on ImageNet 128×128, 4.59 on ImageNet 256×256, and 7.72 on ImageNet 512×512, and we match BigGAN-deep even with as few as 25 forward passes per sample, all while maintaining better coverage of the distribution."
- **Compute** (ADM Table 10, V100-days): ImageNet 256 ADM-G (1,980K iterations) 916 generator + 46 classifier = **962 V100-days** for FID 4.59; BigGAN-deep "128-256" (converted from TPU estimates at 2 TPU-v3 days = 1 V100 day). An ADM-G trained for 750K iterations (393 V100-days) already reached 6.49 < 6.95.
- **Coverage:** Nichol & Dhariwal 2021 (iDDPM) Table 4, ImageNet 64: BigGAN-deep FID 4.06, precision 0.86, recall **0.59**; iDDPM (large, 270M) FID 2.92, precision 0.82, recall **0.71**. "This suggests that diffusion models are better at covering the modes of the distribution than comparable GANs."

### Worked example [DERIVED]
**Why a VAE blurs.** Suppose a pixel is white (+1) in half of the plausible images for a given code and black (−1) in the other half. The squared-error-optimal prediction is the mean, 0 (mid-grey), which appears in *no* real image. Averaging across many such pixels produces the soft, blurry "average digit".
**Mode collapse on the swirl.** A generator that maps every z onto one arm of the spiral produces only realistic points (high precision) but misses most of the data (low recall). FID penalises it; a discriminator looking at single points may not.

### Common misconceptions
- "Diffusion models were invented in 2020." Sohl-Dickstein et al. introduced them in 2015; DDPM (2020) made them competitive in sample quality.
- "GANs were abandoned because diffusion is better on every metric." In early 2022 StyleGAN-XL had the best ImageNet 256 FID (2.30). Diffusion won on stability, coverage, text conditioning and scaling. GAN losses came back inside diffusion distillation (ADD, §12).
- "VAEs are obsolete." Every latent diffusion model (§17) uses a VAE-style autoencoder (with KL weight ~10⁻⁶ plus perceptual and adversarial losses) as its first stage.
- "Diffusion models have the best likelihoods." DDPM's 3.70–3.75 bits/dim was worse than PixelCNN-family models; diffusion's 2020–2021 win was sample quality.

### Quotable
- "The essential idea, inspired by non-equilibrium statistical physics, is to systematically and slowly destroy structure in a data distribution through an iterative forward diffusion process. We then learn a reverse diffusion process that restores structure in data, yielding a highly flexible and tractable generative model of the data." (Sohl-Dickstein et al. 2015, abstract)
- "This framework corresponds to a minimax two-player game." (Goodfellow et al. 2014, abstract)
- "We show that diffusion models can achieve image sample quality superior to the current state-of-the-art generative models." (Dhariwal & Nichol 2021, abstract)
- "Diffusion models are straightforward to define and efficient to train, but to the best of our knowledge, there has been no demonstration that they are capable of generating high quality samples." (Ho et al. 2020, §1)

---

## §4 The forward process: noise schedules, the closed form, and SNR

### Mechanism (plain words)
- **Destroying an image takes no learning.** At each of T = 1000 steps, shrink the image slightly and add a little Gaussian noise. After enough steps nothing of the image is left: the result is indistinguishable from pure noise N(0, I).
- **We can jump straight to any step.** Because every step is Gaussian, the sum of many small noises is one Gaussian. x_t is "some of the image plus some noise" in one formula, so training never has to simulate the chain.
- **The schedule decides how fast information disappears.** β_t is the per-step variance; ᾱ_t is how much signal *power* survives to step t. The **signal-to-noise ratio** SNR = ᾱ/(1 − ᾱ) falls from about 10⁴ to about 10⁻⁵ (DDPM linear schedule).
- **Variance preserving.** The shrink factor √(1 − β_t) is chosen so that data with unit variance keeps unit variance: signal² + noise² = ᾱ + (1 − ᾱ) = 1. That is why it is called the VP process (§6).
- **Linear vs cosine.** DDPM's linear schedule destroys low-resolution images quickly: the last quarter of steps is almost pure noise. Nichol & Dhariwal's cosine schedule removes signal more evenly, so fewer steps are wasted.
- **The schedule is shared, not learned** (in DDPM). The same schedule is used in training and sampling (but samplers may visit only a subset of the steps, §9).

### Key formulas (DDPM notation = ours)
- One step (DDPM eq. 2): q(x_t | x_{t−1}) = N(x_t; √(1 − β_t)·x_{t−1}, β_t·I).
- Closed form (DDPM eq. 4): q(x_t | x₀) = N(x_t; √ᾱ_t·x₀, (1 − ᾱ_t)·I), with α_t = 1 − β_t and ᾱ_t = ∏_{s=1}^{t} α_s. Sampling form: **x_t = √ᾱ_t·x₀ + √(1 − ᾱ_t)·ε**, ε ~ N(0, I).
- SNR(t) = ᾱ_t / (1 − ᾱ_t) (Lin et al. 2023, eq. 5; Kingma et al. 2021). log-SNR λ_t = log SNR(t).
- **DDPM linear schedule** (DDPM §4): β_t increases linearly from β₁ = 10⁻⁴ to β_T = 0.02, T = 1000.
- **Cosine schedule** (Nichol & Dhariwal 2021, eq. 17): ᾱ_t = f(t)/f(0), f(t) = cos²( ((t/T + s)/(1 + s)) · π/2 ), s = 0.008; β_t = 1 − ᾱ_t/ᾱ_{t−1}, **clipped to ≤ 0.999**.
- **Stable Diffusion "scaled linear"** (Lin et al. 2023 Table 1; CompVis config `linear_start: 0.00085`, `linear_end: 0.0120`, `timesteps: 1000`): √β_t linear from √0.00085 to √0.012, i.e. β_t = (√0.00085·(1 − i) + √0.012·i)², i = (t − 1)/(T − 1).
- Continuous-time VP (Song et al. 2021, App. B): β(t) = β̄_min + t(β̄_max − β̄_min), β̄_min = 0.1, β̄_max = 20, t ∈ [0, 1]; with N = 1000 steps this matches DDPM's 10⁻⁴ → 0.02 (β_i ≈ β(t)/N).
- EDM noise level: σ = √((1 − ᾱ)/ᾱ) = 1/√SNR (unscaled-data convention).

### Numbers with sources
- DDPM §4 (exact): "We set T = 1000 for all experiments so that the number of neural network evaluations needed during sampling matches previous work [53, 55]. We set the forward process variances to constants increasing linearly from β1 = 10−4 to βT = 0.02. These constants were chosen to be small relative to data scaled to [−1, 1], ensuring that reverse and forward processes have approximately the same functional form while keeping the signal-to-noise ratio at xT as small as possible (LT = DKL(q(xT|x0) ‖ N(0, I)) ≈ 10−5 bits per dimension in our experiments)."
- DDPM App. B: the schedule was chosen "from a set of constant, linear, and quadratic schedules, all constrained so that L_T ≈ 0. We set T = 1000 without a sweep".
- Nichol & Dhariwal 2021 (iDDPM, arXiv:2102.09672, §3.2): the linear schedule "worked well for high resolution images, it was sub-optimal for images of resolution 64 × 64 and 32 × 32. In particular, the end of the forward noising process is too noisy, and so doesn't contribute very much to sample quality." Fig. 4: on ImageNet 64×64 a linear-schedule model "does not get much worse (as measured by FID) when we skip up to 20% of the reverse diffusion process." Fig. 3 caption: "The latents in the last quarter of the linear schedule are almost purely noise, whereas the cosine schedule adds noise more slowly". On s: "we selected s such that √β0 was slightly smaller than the pixel bin size 1/127.5, which gives s = 0.008." On cos²: "This choice was arbitrary".
- **Terminal SNR** (Lin, Liu, Li & Yang, "Common Diffusion Noise Schedules and Sample Steps are Flawed", arXiv:2305.08891, May 2023, Table 1, T = 1000): linear SNR(T) = 4.035993 × 10⁻⁵, √ᾱ_T = 0.006353; cosine SNR(T) = 2.428735 × 10⁻⁹, √ᾱ_T = 4.928220 × 10⁻⁵; Stable Diffusion SNR(T) = **0.004682**, √ᾱ_T = 0.068265. "None of the schedules has zero SNR on the last timestep t = T, causing inconsistency in train/inference behavior." Consequence: SD 1.x/2.x cannot make very dark or very bright images ("The generated images always have mean brightness around 0"). Fixes: rescale to zero terminal SNR, train with v-prediction, sample from the last timestep ("trailing"), rescale CFG (§14).
- SD3 (Esser et al. 2024, §5.3.2) shifts its schedule toward more noise at high resolution: "Intuitively, since higher resolutions have more pixels, we need more noise to destroy their signal." (worked example in §11).

### Worked example: ᾱ_t, SNR for both schedules [DERIVED; `schedules.py`, t indexed 1…1000]
| t | Linear ᾱ_t | √ᾱ_t | √(1−ᾱ_t) | SNR | Cosine ᾱ_t | √ᾱ_t | √(1−ᾱ_t) | SNR | SD scaled-linear ᾱ_t | SNR |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.9999 | 0.9999 | 0.0100 | 9,999 | 0.99996 | 1.0000 | 0.0064 | 24,220 | 0.99915 | 1,175 |
| 100 | 0.8970 | 0.947 | 0.321 | 8.71 | 0.9721 | 0.986 | 0.167 | 34.8 | 0.8955 | 8.57 |
| 250 | 0.5241 | 0.724 | 0.690 | 1.10 | 0.8470 | 0.920 | 0.391 | 5.54 | 0.6754 | 2.08 |
| 500 | 0.0786 | 0.280 | 0.960 | 0.0853 | 0.4938 | 0.703 | 0.711 | 0.976 | 0.2777 | 0.384 |
| 750 | 0.00335 | 0.0579 | 0.998 | 0.00336 | 0.1443 | 0.380 | 0.925 | 0.169 | 0.0566 | 0.0600 |
| 900 | 0.000275 | 0.0166 | 1.000 | 0.000275 | 0.0241 | 0.155 | 0.988 | 0.0247 | 0.0143 | 0.0145 |
| 1000 | **4.04 × 10⁻⁵** | 0.00635 | 1.000 | 4.04 × 10⁻⁵ | **2.43 × 10⁻⁹** (β clipped at 0.999) | 4.9 × 10⁻⁵ | 1.000 | 2.4 × 10⁻⁹ | **0.00466** | 0.00468 |

- SNR crosses 1 (half signal power, half noise power) at **t = 259** (linear), **t = 496** (cosine), t = 355 (SD scaled-linear).
- √ᾱ_t first drops below 0.1 at t = 674 (linear) vs t = 936 (cosine): the linear schedule leaves almost no visible image for the last third of its steps.
- Without the 0.999 clip, the cosine formula gives ᾱ_1000 = 3.7 × 10⁻³³ (f(T) = cos²(π/2) = 0 up to rounding). The clip is what produces 2.43 × 10⁻⁹, matching Lin et al.
- KL(q(x_T|x₀) ‖ N(0, I)) per dimension for a pixel at x₀ = 1: linear 2.9 × 10⁻⁵ bits, cosine 1.8 × 10⁻⁹ bits, SD 0.0034 bits. The linear value matches DDPM's "≈ 10⁻⁵ bits per dimension".
- **The swirl point** x₀ = (1.0, 0.5) with a fixed ε = (−0.4, 1.2): linear schedule x_100 = (0.819, 0.859), x_250 = (0.448, 1.190), x_500 = (−0.104, 1.292), x_1000 = (−0.394, 1.203) ≈ ε; cosine x_250 = (0.764, 0.930), x_500 = (0.418, 1.205), x_750 = (0.010, 1.300). Same noise, same end point; the cosine path lingers near the data longer.
- **A digit at t = 500 (linear):** each pixel is 0.280 × (its clean value) + 0.960 × (noise). A white stroke (+1) contributes 0.28 against noise of standard deviation 0.96. The digit is faint but visible when you average over the stroke; at t = 750 the stroke contributes 0.058 and is gone to the eye.

### Common misconceptions
- "Noise is added gradually, so training must simulate 1,000 steps." The closed form jumps to any t in one line; training samples a random t per example (§5).
- "At t = T the image is completely gone." For the DDPM linear schedule √ᾱ_T = 0.0064 (not 0); for SD 0.068. This leak is the root of SD's "medium brightness" problem (Lin et al. 2023).
- "β_t is how much noise is in x_t." β_t is the *per-step* variance; the total noise variance at t is 1 − ᾱ_t.
- "The image is scaled up as noise is added." In VP diffusion the image is shrunk (√ᾱ_t < 1) while noise grows, so the total variance stays near 1. In VE/EDM the image is not scaled and the noise variance explodes to σ² = 80² = 6,400.
- "The cosine schedule is a cosine of t." ᾱ_t is a squared cosine of a shifted, rescaled t, and β is clipped at 0.999.

### Quotable
- "These constants were chosen to be small relative to data scaled to [−1, 1], ensuring that reverse and forward processes have approximately the same functional form while keeping the signal-to-noise ratio at xT as small as possible" (Ho et al. 2020, §4)
- "Our cosine schedule is designed to have a linear drop-off of ᾱt in the middle of the process, while changing very little near the extremes of t = 0 and t = T to prevent abrupt changes in noise level." (Nichol & Dhariwal 2021, §3.2)
- "None of the schedules has zero SNR on the last timestep t = T, causing inconsistency in train/inference behavior." (Lin et al. 2023, Table 1 caption)

---

## §5 Training: ε-prediction, the simple loss, x₀- and v-prediction, loss weighting

### Mechanism (plain words)
- **One training step** (DDPM Alg. 1): take a clean image, pick a random t between 1 and 1000, draw noise ε, make x_t in one line, ask the network "what noise did I add?", and penalise the squared error. That is all. It is plain regression.
- **Why predict the noise?** Knowing the noise is the same as knowing the clean image: x₀ = (x_t − √(1 − ᾱ_t)·ε)/√ᾱ_t. Ho et al. found ε-prediction trained better than predicting the posterior mean, and "predicting x0 ... lead to worse sample quality early in our experiments".
- **The simple loss is a reweighted likelihood bound.** The true variational bound weights each t differently; dropping the weights ("L_simple") down-weights the easy, low-noise steps and focuses on harder denoising at larger t. It gave better samples (FID 3.17 vs 13.51).
- **Alternatives:** predict x₀ directly; or predict **v** (Salimans & Ho 2022), a mix of noise and image that stays well-behaved at both ends of the noise range. v-prediction was introduced to make progressive distillation work down to 1 step, and is used by SD 2.x (768 model), Imagen Video and Lin et al.'s zero-terminal-SNR fix.
- **Weighting matters.** Every target choice (ε, x₀, v, velocity) is a reweighting of the same underlying denoising problem across noise levels. EDM (§10) and SD3 (§11) choose where to spend training effort: EDM samples log σ from a normal (P_mean = −1.2, P_std = 1.2); SD3 samples t from a logit-normal concentrated mid-range.

### Key formulas (ours)
- **L_simple** (DDPM eq. 14): L_simple(θ) = E_{t, x₀, ε} ‖ ε − ε_θ(√ᾱ_t·x₀ + √(1 − ᾱ_t)·ε, t) ‖², t ~ Uniform{1, …, T}.
- **Algorithm 1 (Training), DDPM:** repeat { x₀ ~ q(x₀); t ~ Uniform({1, …, T}); ε ~ N(0, I); take a gradient descent step on ∇_θ ‖ε − ε_θ(√ᾱ_t x₀ + √(1−ᾱ_t) ε, t)‖² } until converged.
- The weighted bound term that L_simple replaces (DDPM eq. 12): β_t² / (2σ_t² α_t (1 − ᾱ_t)) · ‖ε − ε_θ‖².
- **x₀ from ε:** x̂₀ = (x_t − √(1 − ᾱ_t)·ε_θ)/√ᾱ_t.
- **v-prediction** (Salimans & Ho 2022, §4; their α_t = √ᾱ_t, σ_t = √(1−ᾱ_t), z_t = x_t): v ≡ α_t·ε − σ_t·x; x̂ = α_t·z_t − σ_t·v̂. In ours: v = √ᾱ_t·ε − √(1 − ᾱ_t)·x₀.
- Loss weightings as x-space errors (Salimans & Ho §4): ε-loss = SNR·‖x − x̂‖²; "truncated SNR" max(SNR, 1)·‖x − x̂‖²; v-loss ‖v − v̂‖² = (1 + SNR)·‖x − x̂‖² ("SNR+1").
- **iDDPM hybrid loss** (eq. 16): L_hybrid = L_simple + λ·L_vlb, λ = 0.001, with a learned interpolated variance (§7).
- **Latent version** (LDM eq. 2–3): replace x₀ by z₀ = E(x₀) and add a condition: L_LDM = E ‖ε − ε_θ(z_t, t, τ_θ(y))‖².

### Numbers with sources
**DDPM** (Ho, Jain & Abbeel, "Denoising Diffusion Probabilistic Models", arXiv:2006.11239, 19 Jun 2020; NeurIPS 2020):
- Abstract (exact): "On the unconditional CIFAR10 dataset, we obtain an Inception score of 9.46 and a state-of-the-art FID score of 3.17. On 256x256 LSUN, we obtain sample quality similar to ProgressiveGAN."
- FID is computed against the training set; "when we compute it with respect to the test set, the score is 5.24" (§4.1).
- **Table 2 (objective ablation, CIFAR-10, IS / FID):** μ̃-prediction with true bound L and learned diagonal Σ: 7.28 / 23.69; μ̃ with L, fixed isotropic Σ: 8.06 / 13.22; ‖μ̃ − μ̃_θ‖² unstable; ε-prediction with L, fixed Σ: 7.67 / 13.51; **ε-prediction with L_simple: 9.46 / 3.17**.
- NLL: ≤ 3.75 bits/dim (L_simple), ≤ 3.70 (L) (Table 1).
- **Training cost** (App. B, exact): "We used TPU v3-8 (similar to 8 V100 GPUs) for all experiments. Our CIFAR model trains at 21 steps per second at batch size 128 (10.6 hours to train to completion at 800k steps), and sampling a batch of 256 images takes 17 seconds." CelebA-HQ/LSUN 256² models train at 2.2 steps/s at batch 64; "sampling a batch of 128 images takes 300 seconds". Steps: CelebA-HQ 0.5M, LSUN Bedroom 2.4M, Cat 1.8M, Church 1.2M; the 256M-parameter Bedroom model 1.15M.
- Hyperparameters (App. B): Adam, learning rate 2 × 10⁻⁴ (2 × 10⁻⁵ at 256×256), batch 128 (CIFAR) / 64, EMA decay 0.9999, dropout 0.1 on CIFAR-10 (swept over {0.1, 0.2, 0.3, 0.4}; 0 elsewhere), random horizontal flips. Official code (`scripts/run_cifar.py`): warmup 5,000 steps, grad clip 1.0, `model_var_type='fixedlarge'` (σ_t² = β_t).
- Evaluation: "On CIFAR10, we calculated Inception and FID scores on 50000 samples". Reported numbers are "the minimum FID value over the course of training".
- LSUN 256 FIDs: Church 7.89, Bedroom 4.90 (Figs. 3–4 captions).
- Rate–distortion (§4.3): "our CIFAR10 model with the highest quality samples has a rate of 1.78 bits/dim and a distortion of 1.97 bits/dim, which amounts to a root mean squared error of 0.95 on a scale from 0 to 255. More than half of the lossless codelength describes imperceptible distortions."

**Salimans & Ho, "Progressive Distillation for Fast Sampling of Diffusion Models"** (arXiv:2202.00512, Feb 2022; ICLR 2022), §4: with ε-prediction, "the weighting function w(λt) gives zero weight to the reconstruction loss at this signal-to-noise ratio" (SNR = 0); they tried "Predicting x directly", "Predicting both x and ε", and "Predicting v ≡ αt ε − σt x" and "found all to work well with progressive distillation". Table 1 (undistilled CIFAR-10 models, FID/IS, stochastic sampler / DDIM sampler; mean of 3 seeds, "±0.1"): ε-prediction with SNR weighting (= DDPM's L_simple) 2.59/9.84 / 2.91/9.52; ε with truncated SNR "leads to divergence"; x-prediction, truncated SNR 2.53/9.92 / 2.51/9.58; **v-prediction, truncated SNR 2.45/9.80 / 2.75/9.52**; v with SNR+1 2.49/9.77 / 2.87/9.43. All parameterisations train about equally well; the choice matters for distillation.

**Lin et al. 2023**: "We recommend always using v prediction for the model and v loss for training" when the schedule has zero terminal SNR (with ᾱ_T = 0, ε-prediction is meaningless at t = T because x_T = ε).

### Worked example: one training step on the swirl [DERIVED; `derive.py`]
1. Data point x₀ = (1.0, 0.5). Draw t = 250 (linear schedule: √ᾱ = 0.724, √(1−ᾱ) = 0.690) and ε = (−0.4, 1.2).
2. x_250 = 0.724·(1.0, 0.5) + 0.690·(−0.4, 1.2) = **(0.448, 1.190)**.
3. The network sees (x_250, t = 250) and outputs, say, ε̂ = (−0.3, 1.0). Loss = (−0.4 + 0.3)² + (1.2 − 1.0)² = 0.01 + 0.04 = **0.05**.
4. Implied clean point: x̂₀ = (x_250 − 0.690·ε̂)/0.724 = ((0.448 + 0.207)/0.724, (1.190 − 0.690)/0.724) = **(0.905, 0.691)**, close to (1.0, 0.5).
5. Same step with v-prediction: target v = 0.724·ε − 0.690·x₀ = (−0.290 − 0.690, 0.869 − 0.345) = **(−0.980, 0.524)**.
- On MNIST the same step is 784-dimensional, and the network also gets the class label (§13).

### Common misconceptions
- "The network predicts the clean image." In DDPM it predicts the noise; the clean image follows from one line of algebra. Different models predict ε, x₀, v or velocity; they are interchangeable up to weighting.
- "L_simple is the likelihood." It is a reweighted bound that discards the per-t weights; it gives better samples but worse likelihoods than optimising the bound (iDDPM §3).
- "Training needs 1,000 forward passes per image." One forward pass per (image, random t) pair.
- "v in v-prediction is the flow-matching velocity." Different quantities (see the notation table); they coincide only up to scale and sign under particular schedules.

### Quotable
- "There is also the possibility of predicting x0, but we found this to lead to worse sample quality early in our experiments." (Ho et al. 2020, §3.2)
- "it is beneficial to down-weight them so that the network can focus on more difficult denoising tasks at larger t terms. We will see in our experiments that this reweighting leads to better sample quality." (Ho et al. 2020, §3.4)
- "Since our samples are nonetheless of high quality, we conclude that diffusion models have an inductive bias that makes them excellent lossy compressors." (Ho et al. 2020, §4.3)

---
## §6 Scores: the score function, Langevin dynamics, denoising score matching, Tweedie, and the score SDE

### Mechanism (plain words)
- **The score is an arrow field.** At every point x, the score ∇ₓ log p(x) points in the direction in which the data density increases fastest. On the swirl it points from empty space back toward the spiral. It does not need the normalising constant of p, which is why it is learnable when p itself is not.
- **Langevin dynamics turns arrows into samples.** Start anywhere, take a small step along the score, add a little fresh noise, repeat. With small enough steps and enough of them, the points end up distributed like the data.
- **The trouble with clean data.** Far from the manifold there is no data, so a score learned from clean examples is garbage there, exactly where sampling starts. Song & Ermon's fix: learn the score of the data *with noise added*, at many noise levels (NCSN). Big noise fills space, so the score is well-defined everywhere; then anneal to small noise.
- **Denoising = score.** Vincent (2011) showed that training a network to undo Gaussian noise is equivalent to matching the score of the noisy data. For Gaussian noise of std σ, the best denoising direction is the score times σ². So the DDPM noise predictor *is* a scaled score: **ε_θ = −σ·score** (with σ = √(1 − ᾱ_t) in VP).
- **Tweedie's formula** makes "denoising = score" exact: the best guess of the clean x given a noisy observation is the noisy point plus σ² times the score.
- **One framework (Song et al. 2021).** DDPM and NCSN are two discretisations of a stochastic differential equation (SDE) that turns data into noise. Running it backwards needs only the score. Every SDE also has a deterministic **probability-flow ODE** with the same marginal distributions; that ODE is what DDIM, EDM and DPM-Solver integrate (§9–§10).

### Key formulas
- **Langevin step** (Song & Ermon 2019, eq. 4; their ε is a step size): x̃_t = x̃_{t−1} + (ε/2)·∇ₓ log p(x̃_{t−1}) + √ε·z_t, z_t ~ N(0, I). "The distribution of x̃_T equals p(x) when ε → 0 and T → ∞".
- **Denoising score matching** (Vincent 2011, as stated in Song & Ermon eq. 2): minimise E_{q_σ(x̃|x) p_data(x)} ‖s_θ(x̃) − ∇_{x̃} log q_σ(x̃ | x)‖², and with q_σ(x̃|x) = N(x̃; x, σ²I), ∇_{x̃} log q_σ(x̃|x) = −(x̃ − x)/σ².
- **Annealed Langevin** (NCSN Alg. 1): for i = 1…L noise levels, step size α_i = ε·σ_i²/σ_L², run T Langevin steps with s_θ(x, σ_i).
- **Tweedie's formula** (Efron 2011, eq. 1.4, z ~ N(μ, σ²)): E{μ | z} = z + σ²·l′(z), l = log of the marginal density of z. For VP diffusion: E[x₀ | x_t] = (x_t + (1 − ᾱ_t)·∇ log p_t(x_t)) / √ᾱ_t.
- **ε ↔ score** (Dhariwal & Nichol 2021, eq. 11): ∇_{x_t} log p_θ(x_t) = −ε_θ(x_t)/√(1 − ᾱ_t).
- **Score SDE** (Song, Sohl-Dickstein, Kingma, Kumar, Ermon & Poole 2021):
  - Forward (eq. 5): dx = f(x, t) dt + g(t) dw.
  - Reverse-time SDE (eq. 6, from Anderson 1982): dx = [f(x, t) − g(t)²·∇ₓ log p_t(x)] dt + g(t) dw̄.
  - **VE SDE** (eq. 9; SMLD/NCSN): dx = √(d[σ²(t)]/dt) dw. σ_min = 0.01, σ_max chosen by Song & Ermon 2020's "Technique 1" (≈ largest pairwise distance between training points).
  - **VP SDE** (eq. 11; DDPM): dx = −½ β(t) x dt + √β(t) dw, β(t) = 0.1 + 19.9t.
  - **Probability-flow ODE** (eq. 13): dx = [f(x, t) − ½ g(t)²·∇ₓ log p_t(x)] dt. Same marginals p_t as the SDE, no noise.

### Numbers with sources
- **NCSN** (Song & Ermon, "Generative Modeling by Estimating Gradients of the Data Distribution", arXiv:1907.05600, Jul 2019; NeurIPS 2019), §5: "We choose L = 10 different standard deviations such that {σi}L i=1 is a geometric sequence with σ1 = 1 and σ10 = 0.01." Annealed Langevin: "we choose T = 100 and ε = 2 × 10−5, and use uniform noise as our initial samples". [DERIVED] 10 levels × 100 steps = **1,000 network calls** per sample (pixels in [0, 1]). Results: CIFAR-10 "Inception score of 8.87 for unconditional generative models, and achieves a competitive FID score of 25.32".
- NCSN §3: "the scarcity of training data in low data density regions, e.g., far from the manifold, hinders the accuracy of score estimation and slows down the mixing of Langevin dynamics sampling."
- **DDPM's link** (Ho et al. 2020, §3.2): the ε-parameterised sampler "resembles Langevin dynamics with εθ as a learned gradient of the data density", and eq. 12 "resembles denoising score matching over multiple noise scales indexed by t".
- **Score SDE** (arXiv:2011.13456, Nov 2020; ICLR 2021): abstract: CIFAR-10 "Inception score of 9.89 and FID of 2.20, a competitive likelihood of 2.99 bits/dim, and demonstrate high fidelity generation of 1024 × 1024 images for the first time from a score-based generative model." Table 3 (CIFAR-10 FID / IS): NCSN++ cont. (deep, VE) **2.20 / 9.89**; DDPM++ cont. (deep, VP) 2.41 / 9.68; DDPM 3.17 / 9.46; NCSN 25.32 / 8.87; NCSNv2 10.87 / 8.40. Table 2 (probability-flow ODE likelihoods): DDPM++ cont. (deep, sub-VP) 2.99 bits/dim, FID 2.92.
- PF-ODE efficiency (§4.3): with a black-box solver (RK45), "the number of function evaluations can be reduced by over 90% without affecting the visual quality of samples (Fig. 3)."
- **Tweedie** (Efron, "Tweedie's Formula and Selection Bias", JASA 106(496), 2011): "Tweedie's formula, first reported by Robbins in 1956"; "Herbert Robbins (1956) credits personal correspondence with Maurice Kenneth Tweedie".
- **Vincent 2011**: "A Connection Between Score Matching and Denoising Autoencoders", *Neural Computation* 23(7):1661–1674. The PDF could not be retrieved (university server returned HTML); the objective is taken from Song & Ermon's statement of it. **[UNVERIFIED wording]**. Score matching itself: Hyvärinen 2005, JMLR 6:695–709 (cited, not opened).

### Worked example [DERIVED; `derive.py` part C]
**Exact scores for Gaussian data.** Let the "data" be a tight Gaussian blob N(μ, s²I) with μ = (1.0, 0.5), s = 0.1 (a stand-in for one bit of the swirl). Under VP noising, p_t = N(√ᾱ_t·μ, (ᾱ_t s² + 1 − ᾱ_t)I), so score(x) = −(x − √ᾱ_t μ)/(ᾱ_t s² + 1 − ᾱ_t). At the point x = (0.5, 0.5), linear schedule:
| t | variance of p_t | score | ε* = −√(1−ᾱ)·score | Tweedie x̂₀ |
|---|---|---|---|---|
| 100 | 0.112 | (3.994, −0.236) | (−1.282, 0.076) | (0.962, 0.502) |
| 500 | 0.922 | (−0.238, −0.390) | (0.229, 0.375) | (1.001, 0.501) |
| 900 | 0.9997 | (−0.484, −0.492) | (0.483, 0.492) | (1.000, 0.500) |
- At t = 100 the blob sits at √ᾱμ = (0.947, 0.474), to the right of x, so the score points strongly right (+4.0). At t = 900 the blob is almost at the origin and has spread to unit variance; the score is ≈ −x.
- Tweedie recovers the clean mean at every t. With one tight blob there is nothing else to guess. With a two-armed swirl, Tweedie at large t returns a point *between* the arms (the average), which is why one-step denoising from pure noise looks blurry (see §9's toy).

### Common misconceptions
- "The score is the probability." It is the gradient of the *log* probability; it ignores the normalising constant and can be large where p is tiny.
- "ε_θ and the score point the same way." They point in opposite directions: ε_θ points from the clean image toward the noise; the score points back toward data. s = −ε/σ.
- "Langevin dynamics is gradient ascent." Without the injected √ε·z noise it collapses to the modes; the noise is what makes it sample.
- "The probability-flow ODE produces the same samples as the SDE." Same *marginal distributions*, different individual trajectories: one noise vector maps to one image under the ODE (DDIM), while the SDE (DDPM) gives a different image each run.
- "Score models and diffusion models are different families." Song et al. 2021 showed NCSN (VE SDE) and DDPM (VP SDE) are two instances of one framework.

### Quotable
- "Langevin dynamics can produce samples from a probability density p(x) using only the score function ∇x log p(x)." (Song & Ermon 2019, §2.2)
- "Creating noise from data is easy; creating data from noise is generative modeling." (Song et al. 2021, abstract, first sentence)
- "For all diffusion processes, there exists a corresponding deterministic process whose trajectories share the same marginal probability densities" (Song et al. 2021, §4.3)

---

## §7 DDPM sampling: ancestral sampling, the reverse variance, and the cost of 1,000 steps

### Mechanism (plain words)
- **Start from pure noise and walk back.** x_T ~ N(0, I). At each step the network predicts the noise in x_t; we subtract a scaled amount of it (moving toward the data), rescale, and **add a little fresh noise**. After T = 1000 steps we have a sample.
- **Why fresh noise?** Each reverse step is a *sample* from a Gaussian p_θ(x_{t−1}|x_t), not a deterministic map. The injected noise gives each step the variance the model was derived for and lets later steps correct earlier mistakes. (Simply deleting σ_t·z while keeping DDPM's mean formula is not a derived sampler; the principled deterministic version is DDIM with η = 0, §9.) It is the same role noise plays in Langevin dynamics. The last step (t = 1) adds no noise.
- **Two reasonable variances.** σ_t² = β_t (upper bound) or σ_t² = β̃_t (lower bound). DDPM found them about equal in quality; they differ only near t = 1.
- **The cost.** One network call per step and T = 1000 steps: 1,000 sequential calls per image. This is diffusion's original weakness. DDPM on a TPU v3-8: 256 CIFAR-10 images in 17 s; 128 images at 256×256 in 300 s.
- **Improved DDPM: learn the variance.** Nichol & Dhariwal let the network interpolate between β_t and β̃_t (log-space). With the hybrid loss, sampling with **100** evenly-spaced steps reaches near-optimal FID, and it keeps working when steps are skipped. ADM (2021) used 250 steps.

### Key formulas (DDPM notation = ours)
- **Algorithm 2 (Sampling), DDPM:**
  1: x_T ~ N(0, I)
  2: for t = T, …, 1 do
  3:   z ~ N(0, I) if t > 1, else z = 0
  4:   x_{t−1} = (1/√α_t)·( x_t − (β_t/√(1 − ᾱ_t))·ε_θ(x_t, t) ) + σ_t·z
  5: end for
  6: return x₀
- **Posterior of the forward process** (DDPM eqs. 6–7): q(x_{t−1}|x_t, x₀) = N(μ̃_t, β̃_t I), μ̃_t = (√ᾱ_{t−1} β_t/(1 − ᾱ_t))·x₀ + (√α_t (1 − ᾱ_{t−1})/(1 − ᾱ_t))·x_t, **β̃_t = ((1 − ᾱ_{t−1})/(1 − ᾱ_t))·β_t**.
- **Mean parameterisation** (eq. 11): μ_θ(x_t, t) = (1/√α_t)(x_t − (β_t/√(1 − ᾱ_t)) ε_θ(x_t, t)).
- **Learned variance** (iDDPM eq. 15): Σ_θ(x_t, t) = exp(v·log β_t + (1 − v)·log β̃_t), v output per dimension.
- **Respacing to K steps** (iDDPM §4, eq. 19): pick a subsequence S of timesteps ("K evenly spaced real numbers between 1 and T (inclusive), and then round each resulting number to the nearest integer"), then β_{S_t} = 1 − ᾱ_{S_t}/ᾱ_{S_{t−1}}, β̃_{S_t} = ((1 − ᾱ_{S_{t−1}})/(1 − ᾱ_{S_t}))·β_{S_t}.

### Numbers with sources
- DDPM §3.2 (exact): "Experimentally, both σt² = βt and σt² = β̃t = (1−ᾱt−1)/(1−ᾱt) βt had similar results. The first choice is optimal for x0 ∼ N(0, I), and the second is optimal for x0 deterministically set to one point. These are the two extreme choices corresponding to upper and lower bounds on reverse process entropy for data with coordinatewise unit variance".
- DDPM's official CIFAR-10 config uses `model_var_type='fixedlarge'` (σ_t² = β_t). DDIM (Song et al. 2021a, §5) notes the larger-variance σ̂ "is used by the implementation in Ho et al. (2020) only to obtain the CIFAR10 samples, but not samples of the other datasets."
- Sampling time (DDPM App. B): CIFAR-10 256 images in 17 s; 256×256 models 128 images in 300 s (TPU v3-8). [DERIVED] ≈ 0.066 s per CIFAR image and ≈ 2.3 s per 256² image, *for the whole batch-parallel 1,000-step chain*.
- Rombach et al. 2022 (LDM, §1), on pixel diffusion cost: "training the most powerful DMs often takes hundreds of GPU days (e.g. 150 - 1000 V100 days in [15])" and "producing 50k samples takes approximately 5 days [15] on a single A100 GPU" ([15] = ADM).
- **iDDPM** (Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models", arXiv:2102.09672, Feb 2021; ICML 2021): abstract: "learning variances of the reverse diffusion process allows sampling with an order of magnitude fewer forward passes with a negligible difference in sample quality". §4: "With this model, 100 sampling steps is sufficient to achieve near-optimal FIDs for our fully trained models." L_hybrid with λ = 0.001 "to prevent Lvlb from overwhelming Lsimple". On ImageNet 64×64 the DDPM setup (σ² = β, T = 1000) reached 3.99 bits/dim after 200K iterations; "increasing T from 1000 to 4000" improved it to 3.77 (§3).
- iDDPM on DDIM: "DDIM produces better samples with fewer than 50 sampling steps, but worse samples when using 50 or more steps."
- ADM (Dhariwal & Nichol 2021): ImageNet models "sampled using 250 steps, except when we use the DDIM sampler with 25 steps"; LSUN models 1,000 steps (Table 5 caption).

### Worked example: the reverse-step coefficients [DERIVED; `derive.py` part B, linear schedule]
| t | 1/√α_t | β_t/√(1−ᾱ_t) (multiplies ε_θ) | σ_t if σ² = β_t | σ_t if σ² = β̃_t |
|---|---|---|---|---|
| 1000 | 1.01015 | 0.02000 | 0.1414 | 0.1414 |
| 500 | 1.00506 | 0.01046 | 0.1002 | 0.1002 |
| 250 | 1.00254 | 0.00733 | 0.0711 | 0.0709 |
| 100 | 1.00104 | 0.00646 | 0.0455 | 0.0451 |
| 2 | 1.00006 | 0.00809 | 0.0110 | 0.0074 |
| 1 | 1.00005 | 0.01000 | 0.0100 | 0 (no noise added at t = 1 anyway) |
- Each step removes only ~1% of the predicted noise and adds back fresh noise of std 0.01–0.14. That is why 1,000 small steps are needed with this sampler.
- The two variance choices agree to 3 decimals until t ≈ 100 and differ only in the last handful of steps.
- **Fresh noise in action** (Gaussian toy, §6 data N((1, 0.5), 0.1²I), exact ε*, `ddim_toy.py`): 2,000 DDPM ancestral runs that all start from the *same* x_T = (−0.4, 1.2) end with mean (0.993, 0.500) and std (0.102, 0.102). They reproduce the whole data distribution (std 0.1) from one starting point. The deterministic ODE (§9) instead maps that x_T to one point, (0.959, 0.620).

### Common misconceptions
- "DDPM sampling is deterministic given the starting noise." It injects new noise at every step but the last; the same x_T gives different images.
- "The network predicts x_{t−1}." It predicts ε (or x₀, v); the step formula turns that into the mean of x_{t−1}.
- "σ_t in Algorithm 2 is the noise level of x_t." It is the std of the noise *added in this step* (≈ √β_t, at most 0.14). The noise level of x_t is √(1 − ᾱ_t).
- "You must use all 1,000 steps because the model was trained on 1,000." Respacing (iDDPM) and DDIM sample the same trained model with 25–250 steps.

### Quotable
- "The complete sampling procedure, Algorithm 2, resembles Langevin dynamics with εθ as a learned gradient of the data density." (Ho et al. 2020, §3.2)
- "learning variances of the reverse diffusion process allows sampling with an order of magnitude fewer forward passes with a negligible difference in sample quality" (Nichol & Dhariwal 2021, abstract)

---

## §8 The U-Net: from segmentation to denoising; time embeddings; ADM's changes; Stable Diffusion's U-Net

### Mechanism (plain words)
- **Same size in, same size out.** A denoiser takes a noisy image and returns a same-shaped prediction (the noise). The U-Net (Ronneberger et al. 2015, built for biomedical segmentation) does this well: a **contracting path** shrinks the image step by step to capture context ("what is this?"); an **expanding path** grows it back to full size; **skip connections** carry fine detail from each level of the way down straight across to the same level on the way up.
- **Residual blocks + attention.** Diffusion U-Nets (DDPM onward) use residual conv blocks with GroupNorm, and **self-attention** at low resolutions (DDPM: 16×16) so distant parts of the image can coordinate.
- **Telling the network the time.** The same weights serve every t, so t must be an input. DDPM turns t into a vector with the Transformer's sinusoidal position embedding, passes it through a small MLP, and adds it into every residual block. ADM instead uses it (plus the class) to scale and shift GroupNorm outputs (AdaGN).
- **Conditioning on text** (LDM/Stable Diffusion): insert **cross-attention** layers whose queries come from image features and keys/values from text embeddings (§15).
- **Scale.** DDPM's CIFAR-10 U-Net: 35.7M parameters. SD 1.x: 860M. SDXL: 2.6B. Then transformers took over at the largest scales (§18).

### Key formulas
- **Sinusoidal timestep embedding** (DDPM code `get_timestep_embedding`, "matches the implementation in tensor2tensor"): for embedding size d, half = d/2, freq_k = exp(−ln(10000)·k/(half − 1)) for k = 0…half − 1; emb(t) = [sin(t·freq_0), …, sin(t·freq_{half−1}), cos(t·freq_0), …, cos(t·freq_{half−1})].
- **AdaGN** (ADM §3.1): AdaGN(h, y) = y_s·GroupNorm(h) + y_b, where (y_s, y_b) is a linear projection of the timestep and class embedding.
- **Cross-attention** (LDM §3.3): Q = W_Q·φ(z_t), K = W_K·τ_θ(y), V = W_V·τ_θ(y); Attention = softmax(QKᵀ/√d)·V.

### Numbers with sources
- **U-Net** (Ronneberger, Fischer & Brox, arXiv:1505.04597, May 2015; MICCAI 2015), abstract: "The architecture consists of a contracting path to capture context and a symmetric expanding path that enables precise localization." "Segmentation of a 512x512 image takes less than a second on a recent GPU." The network "has 23 convolutional layers"; input tile 572×572.
- **DDPM's U-Net** (App. B): "Our neural network architecture follows the backbone of PixelCNN++ [52], which is a U-Net [48] based on a Wide ResNet [72]. We replaced weight normalization [49] with group normalization [66] to make the implementation simpler. Our 32 × 32 models use four feature map resolutions (32 × 32 to 4 × 4), and our 256 × 256 models use six. All models have two convolutional residual blocks per resolution level and self-attention blocks at the 16 × 16 resolution between the convolutional blocks [6]. Diffusion time t is specified by adding the Transformer sinusoidal position embedding [60] into each residual block. Our CIFAR10 model has 35.7 million parameters, and our LSUN and CelebA-HQ models have 114 million parameters. We also trained a larger variant of the LSUN Bedroom model with approximately 256 million parameters by increasing filter count."
- DDPM official CIFAR-10 config (`scripts/run_cifar.py`): `ch=128, ch_mult=(1, 2, 2, 2), num_res_blocks=2, attn_resolutions=(16,)`, dropout 0.1. [DERIVED] channel widths 128, 256, 256, 256 at 32, 16, 8, 4 pixels.
- **ADM changes** (Dhariwal & Nichol 2021, §3, Table 1, ImageNet 128, FID change vs a 160-channel, depth-2, 1-head, 16×16-attention baseline of 15.33 at 700K / 13.21 at 1200K iterations): depth 4 (128 channels) −0.21 / −0.48; 4 heads −0.54 / −0.82; attention at 32, 16, 8 −0.72 / −0.66; BigGAN up/downsampling −1.20 / −1.21; rescaling residuals by 1/√2 +0.16 / +0.25 (worse; dropped); all improvements combined **−3.14 / −3.00**. Table 3: AdaGN 13.06 vs "Addition + GroupNorm" (DDPM style) 15.08. Final architecture (exact): "variable width with 2 residual blocks per resolution, multiple heads with 64 channels per head, attention at 32, 16 and 8 resolutions, BigGAN residual blocks for up and downsampling, and adaptive group normalization for injecting timestep and class embeddings into residual blocks."
- iDDPM's ImageNet-64 models: 100M ("small", 1.7M steps) and 270M parameters ("large", 250K iterations) (§5).
- **Stable Diffusion 1.x U-Net** (CompVis README, https://github.com/CompVis/stable-diffusion): "With its 860M UNet and 123M text encoder, the model is relatively lightweight and runs on a GPU with at least 10GB VRAM." Safetensors header of `stable-diffusion-v1-5/stable-diffusion-v1-5`: U-Net **859,520,964** parameters; CLIP text encoder 123,060,480; VAE 83,653,863 (encoder 34,163,592 + decoder 49,490,179). Config `v1-inference.yaml`: `model_channels: 320`, `channel_mult: [1, 2, 4, 4]`, `num_res_blocks: 2`, `attention_resolutions: [4, 2, 1]`, `num_heads: 8`, `transformer_depth: 1`, `context_dim: 768`, `in_channels: 4`. [DERIVED] widths 320/640/1280/1280 at latent resolutions 64/32/16/8; cross-attention at 64, 32 and 16 (attention_resolutions are downsampling factors 1, 2, 4).
- **SDXL** (Podell et al., arXiv:2307.01952, Jul 2023), Table 1: SDXL U-Net **2.6B**, transformer blocks per level [0, 2, 10], channel mult [1, 2, 4], text encoders CLIP ViT-L & OpenCLIP ViT-bigG, context dim 2048, pooled embedding from OpenCLIP ViT-bigG; SD 1.4/1.5: 860M, [1, 1, 1, 1], [1, 2, 4, 4], CLIP ViT-L, 768; SD 2.0/2.1: 865M, [1, 1, 1, 1], [1, 2, 4, 4], OpenCLIP ViT-H, 1024. §2.1: "we omit the transformer block at the highest feature level, use 2 and 10 blocks at the lower levels, and remove the lowest level (8× downsampling) in the UNet altogether". Safetensors header: SDXL base U-Net **2,567,463,684**; refiner U-Net 2,259,526,660.

### Worked example [DERIVED]
**A 64×64×4 latent through SD 1.5's U-Net:** in 4 channels → conv to 320 channels at 64×64 → 640 at 32×32 → 1280 at 16×16 → 1280 at 8×8 (bottleneck) → back up 16, 32, 64 with skip connections → conv to 4 channels out. Every residual block receives the 1280-d time embedding (4 × 320); every transformer block at 64, 32, 16 cross-attends to the 77×768 text embedding.
**Time embedding at t = 500, d = 8** (half = 4, freqs = 1, 10000^(−1/3) = 0.0464, 0.00215, 0.0001): angles 500, 23.2, 1.077, 0.05 rad → [sin…, cos…] = [−0.468, −0.938, 0.881, 0.050, −0.884, −0.347, 0.474, 0.999]. Fast-varying coordinates distinguish nearby t; slow ones encode the coarse position in the schedule. [DERIVED; formula above]

### Common misconceptions
- "The U-Net was invented for diffusion." It was a 2015 biomedical segmentation network; PixelCNN++ adapted it to generation and DDPM inherited it.
- "Skip connections are the residual connections." U-Net skips concatenate encoder features into the decoder across the "U"; residual connections add within a block. Diffusion U-Nets have both.
- "One network per timestep." One network for all t; t enters as an embedding ("Parameters are shared across time", DDPM §4).
- "Stable Diffusion is a 1B U-Net." The U-Net is 860M; with the text encoder (123M) and VAE (84M) the pipeline is ≈ 1.07B parameters.

### Quotable
- "The architecture consists of a contracting path to capture context and a symmetric expanding path that enables precise localization." (Ronneberger et al. 2015, abstract)
- "Parameters are shared across time, which is specified to the network using the Transformer sinusoidal position embedding [60]. We use self-attention at the 16 × 16 feature map resolution" (Ho et al. 2020, §4)

---

## §9 DDIM: deterministic sampling in far fewer steps, and inversion

### Mechanism (plain words)
- **Same training, different sampler.** DDIM (Song, Meng & Ermon, Oct 2020) uses a DDPM-trained network unchanged. The DDPM loss only depends on the marginals q(x_t|x₀); many different forward processes, including **non-Markovian** ones, share those marginals. Each gives a different valid reverse sampler.
- **One step = predict x₀, then re-noise to the next level.** From x_t: estimate x̂₀ with the network; point back toward x_t with the predicted noise; place the result at the noise level of the next (possibly much earlier) timestep. With η = 0 no fresh noise is added, so sampling is **deterministic**.
- **Skipping steps.** Because each step jumps straight to the target noise level, you can visit 10, 20, 50 or 100 of the 1,000 timesteps. Quality degrades gracefully: CIFAR-10 FID 4.04 at 1,000 steps, 4.67 at 50, 13.36 at 10.
- **η dials stochasticity.** η = 0 is DDIM; η = 1 is DDPM-like ancestral sampling. With few steps, η = 0 is much better.
- **An ODE in disguise.** DDIM is Euler's method on the probability-flow ODE (§6), written in a convenient variable. Deterministic means: fixed x_T → fixed image ("consistency": the same noise gives the same high-level image at any step count). It also means you can run it **backwards**: image → noise (**DDIM inversion**), for reconstruction, interpolation and editing.

### Key formulas (DDIM writes α_t for DDPM's ᾱ_t; converted to ours)
- **DDIM update** (DDIM eq. 12): x_{t−1} = √ᾱ_{t−1}·x̂₀ + √(1 − ᾱ_{t−1} − σ_t²)·ε_θ(x_t, t) + σ_t·z, with x̂₀ = (x_t − √(1 − ᾱ_t)·ε_θ(x_t, t))/√ᾱ_t. The three terms are labelled in the paper "predicted x0", "direction pointing to xt", "random noise".
- **η** (eq. 16): σ_t(η) = η·√((1 − ᾱ_{t−1})/(1 − ᾱ_t))·√(1 − ᾱ_t/ᾱ_{t−1}). η = 1 gives DDPM's β̃ variance; η = 0 gives DDIM.
- **Skipping:** replace t − 1 by the previous element of a subsequence τ of [1, …, T] of length S.
- **ODE form** (eq. 14): with σ̄ = √((1 − ᾱ)/ᾱ) and x̄ = x/√ᾱ: dx̄(t) = ε_θ(x̄/√(σ̄² + 1), t)·dσ̄(t). DDIM is Euler on this ODE. Proposition 1: it is equivalent to the probability-flow ODE of the VE SDE; DDIM takes Euler steps in σ̄ while Song et al. take them in t.
- **Inversion:** run eq. 14 from t = 0 to T (x₀ → x_T), then back.

### Numbers with sources
- DDIM (arXiv:2010.02502, 6 Oct 2020; ICLR 2021), abstract: "We empirically demonstrate that DDIMs can produce high quality samples 10× to 50× faster in terms of wall-clock time compared to DDPMs, allow us to trade off computation for sample quality, perform semantically meaningful image interpolation directly in the latent space, and reconstruct observations with very low error."
- **Table 1 (FID; rows = η, columns = number of steps S):**

| | CIFAR-10 S = 10 | 20 | 50 | 100 | 1000 | CelebA 64 S = 10 | 20 | 50 | 100 | 1000 |
|---|---|---|---|---|---|---|---|---|---|---|
| η = 0.0 (DDIM) | **13.36** | **6.84** | **4.67** | **4.16** | 4.04 | 17.33 | 13.73 | 9.17 | 6.53 | 3.51 |
| η = 0.2 | 14.04 | 7.11 | 4.77 | 4.25 | 4.09 | 17.66 | 14.11 | 9.51 | 6.79 | 3.64 |
| η = 0.5 | 16.66 | 8.35 | 5.25 | 4.46 | 4.29 | 19.86 | 16.06 | 11.01 | 8.09 | 4.28 |
| η = 1.0 (DDPM, β̃) | 41.07 | 18.36 | 8.01 | 5.78 | 4.73 | 33.12 | 26.03 | 18.48 | 13.93 | 5.98 |
| σ̂ (DDPM, larger variance) | 367.43 | 133.37 | 32.72 | 9.99 | **3.17** | 299.71 | 183.83 | 71.71 | 45.20 | 3.26 |

- §5.1: "DDIM is able to produce samples with quality comparable to 1000 step models within 20 to 100 steps, which is a 10× to 50× speed up compared to the original DDPM." At S = 1000 with σ̂ (= the original DDPM sampler), DDPM's 3.17 is marginally better than DDIM's 4.04.
- **Table 2 (reconstruction error, CIFAR-10 test, per-dimension MSE on [0, 1] scale, S steps for both encoding and decoding):** S = 10: 0.014; 20: 0.0065; 50: 0.0023; 100: 0.0009; 200: 0.0004; 500: 0.0001; 1000: 0.0001.
- §5.2 consistency: "for the generated images with the same initial xT, most high-level features are similar, regardless of the generative trajectory."
- DDIM appendix on notation: "we have used the notation αt to represent the variable ᾱt in Ho et al. (2020)".

### Worked example: steps vs accuracy with an exact denoiser [DERIVED; `ddim_toy.py`]
Data N((1.0, 0.5), 0.1²I) with the exact ε* (§6); start x_T = (−0.4, 1.2); the exact ODE endpoint is (0.9594, 0.6197).
| DDIM steps (η = 0) | endpoint | error |
|---|---|---|
| 1 | (1.0000, 0.5001) | 0.126 |
| 2 | (0.9960, 0.5119) | 0.114 |
| 10 | (0.9860, 0.5411) | 0.083 |
| 50 | (0.9682, 0.5936) | 0.028 |
| 100 | (0.9641, 0.6057) | 0.015 |
| 1000 | (0.9600, 0.6177) | 0.0021 |
- With one step, DDIM returns Tweedie's x̂₀ from pure noise: the **mean** of the data (1.0, 0.5). Every starting noise maps to the same average point. That is the toy version of a blurry one-step sample, and why distillation (§12) is needed for 1–4-step models.
- With more steps the deterministic path bends toward the specific point this x_T should reach.

### Common misconceptions
- "DDIM needs a specially trained model." It uses any ε-prediction model trained with the DDPM objective.
- "DDIM is always better than DDPM." At 1,000 steps the original stochastic sampler had the better CIFAR-10 FID (3.17 vs 4.04); iDDPM found DDIM worse than its learned-variance sampler at ≥ 50 steps. DDIM wins at few steps.
- "Deterministic means no randomness at all." The randomness is all in x_T; the map x_T → x₀ is fixed.
- "Inversion recovers the exact noise used to make an image." DDIM inversion is approximate (Euler error), and the real image was not necessarily made from any x_T; reconstruction MSE is 0.014 at 10 steps, 0.0001 at 500.

### Quotable
- "DDIMs can produce high quality samples 10× to 50× faster in terms of wall-clock time compared to DDPMs" (Song, Meng & Ermon 2021, abstract)
- "DDIM is able to produce samples with quality comparable to 1000 step models within 20 to 100 steps" (DDIM §5.1)

---

## §10 Solvers and schedules: Euler, Heun, EDM, DPM-Solver, UniPC; steps vs NFE

### Mechanism (plain words)
- **Sampling is numerical integration.** The probability-flow ODE (§6) defines a smooth path from noise to image; a sampler follows it in N discrete steps. Big steps cut corners on curved paths. Better solvers (higher order) and better step placement buy quality per network call.
- **Euler** takes one slope evaluation per step (DDIM is Euler). **Heun** (improved Euler / trapezoid) takes a trial Euler step, evaluates the slope again at the landing point, and averages: 2 network calls per step, error O(h²) instead of O(h).
- **Karras et al. (EDM, 2022)** separated the design choices (schedule, scaling, solver, preconditioning, loss weighting) and fixed each: work directly in noise std σ (x = data + σ·ε), take **Heun** steps, and place steps by σ_i = (σ_max^{1/ρ} + i/(N−1)·(σ_min^{1/ρ} − σ_max^{1/ρ}))^ρ with **ρ = 7**, which crowds steps near low noise where detail is decided. Result: CIFAR-10 FID **1.79 in 35 network evaluations**.
- **DPM-Solver** (Lu et al. 2022) solves the linear part of the diffusion ODE exactly and approximates only the network term, in log-SNR time: "high-quality samples in only 10 to 20 function evaluations". **DPM-Solver++** switches to data (x₀) prediction and a multistep form so it stays stable with large guidance (15–20 steps for guided text-to-image). **UniPC** adds a corrector at no extra network cost and targets < 10 steps.
- **Steps vs NFE.** Heun with N steps costs 2N − 1 NFE; CFG doubles everything again (§14). Compare samplers by NFE, not by "steps".
- **Stochastic samplers** (EDM's S_churn, DDPM ancestral, "Euler a") add noise during sampling. EDM found stochasticity hurt on CIFAR-10 with their training but helped on ImageNet-64.

### Key formulas
- **EDM (Karras, Aittala, Aila & Laine 2022)** in our terms: noisy input x = x₀ + σ·ε (VE form); denoiser D_θ(x; σ) estimates x₀; score = (D_θ(x; σ) − x)/σ²; ODE (s(t) = 1, σ(t) = t): **dx/dσ = (x − D_θ(x; σ))/σ**.
- **Euler step:** x_{i+1} = x_i + (σ_{i+1} − σ_i)·d_i, d_i = (x_i − D_θ(x_i; σ_i))/σ_i.
- **Heun step** (EDM Alg. 1): x̃ = x_i + (σ_{i+1} − σ_i)·d_i; if σ_{i+1} ≠ 0: d′ = (x̃ − D_θ(x̃; σ_{i+1}))/σ_{i+1}, x_{i+1} = x_i + (σ_{i+1} − σ_i)·(d_i + d′)/2.
- **EDM schedule** (eq. 5): σ_{i<N} = (σ_max^{1/ρ} + (i/(N−1))·(σ_min^{1/ρ} − σ_max^{1/ρ}))^ρ, σ_N = 0; σ_min = 0.002, σ_max = 80, ρ = 7.
- **EDM preconditioning** (eq. 7): D_θ(x; σ) = c_skip(σ)·x + c_out(σ)·F_θ(c_in(σ)·x; c_noise(σ)), σ_data = 0.5. Training noise: ln σ ~ N(P_mean, P_std²), P_mean = −1.2, P_std = 1.2 (Table 1).
- **DPM-Solver** (Lu et al. 2022): with λ_t = log(α_t/σ_t) (half the log-SNR, their α_t = √ᾱ_t), the exact solution is x_t = (α_t/α_s)·x_s − α_t ∫_{λ_s}^{λ_t} e^{−λ} ε̂_θ(x̂_λ, λ) dλ: "an exponentially weighted integral of the neural network".

### Numbers with sources
- **EDM** ("Elucidating the Design Space of Diffusion-Based Generative Models", arXiv:2206.00364, Jun 2022; NeurIPS 2022), abstract: "Together, our improvements yield new state-of-the-art FID of 1.79 for CIFAR-10 in a class-conditional setting and 1.97 in an unconditional setting, with much faster sampling (35 network evaluations per image) than prior designs." It also improves "the FID of a previously trained ImageNet-64 model from 2.07 to near-SOTA 1.55, and after re-training with our proposed improvements to a new SOTA of 1.36."
- §3: "Heun’s method reaches the same FID as Euler’s method with considerably lower NFE." "We set ρ = 7 for the remainder of this paper."
- **Table 2 (EDM training ablation, FID with their deterministic sampler)**, CIFAR-10 conditional VP/VE · unconditional VP/VE · FFHQ 64 VP/VE · AFHQv2 64 VP/VE: (A) baseline 2.48/3.11 · 3.01/3.77 · 3.39/25.95 · 2.58/18.52; (B) hyperparameters 2.18/2.48 · 2.51/2.94 · …; (E) + their loss 1.88/1.86 · 2.05/1.99 · 2.60/2.81 · 2.29/2.28; **(F) + non-leaky augmentation 1.79/1.79 · 1.97/1.98 · 2.39/2.53 · 1.96/2.16**; NFE 35 at 32×32 and 79 at 64×64.
- Fig. 12 caption: "At 32×32 resolution, reasonable image quality is reached around NFE = 13, but FID keeps improving until NFE = 35. At 64×64 resolution, reasonable image quality is reached around NFE = 19, but FID keeps improving until NFE = 79."
- ImageNet-64 1.36 uses the stochastic sampler with 256 steps (NFE = 511) (Fig. 5c and text). Previous record 1.48.
- [DERIVED] 35 NFE = 18 Heun steps (2·18 − 1; the last step to σ = 0 is Euler, Alg. 1 line "if σ_{i+1} ≠ 0"); 79 NFE = 40 steps.
- **DPM-Solver** (Lu, Zhou, Bao, Chen, Li & Zhu, arXiv:2206.00927, Jun 2022; NeurIPS 2022), abstract: "DPM-Solver can generate high-quality samples in only 10 to 20 function evaluations on various datasets. We achieve 4.70 FID in 10 function evaluations and 2.87 FID in 20 function evaluations on the CIFAR10 dataset, and a 4 ∼ 16× speedup compared with previous state-of-the-art training-free samplers on various datasets."
- **DPM-Solver++** (Lu et al., arXiv:2211.01095, Nov 2022), abstract: "The commonly-used fast sampler for guided sampling is DDIM, a first-order diffusion ODE solver that generally needs 100 to 250 steps for high-quality samples." "previous high-order fast samplers suffer from instability issues, and they even become slower than DDIM when the guidance scale grows large." "DPM-Solver++ can generate high-quality samples within only 15 to 20 steps for guided sampling by pixel-space and latent-space DPMs."
- **UniPC** (Zhao, Bai, Rao, Zhou & Lu, arXiv:2302.04867, Feb 2023; NeurIPS 2023), abstract: "Our UniPC can achieve 3.87 FID on CIFAR10 (unconditional) and 7.51 FID on ImageNet 256×256 (conditional) with only 10 function evaluations."
- **PLMS** (Liu, Ren, Lin & Zhao, "Pseudo Numerical Methods for Diffusion Models on Manifolds", arXiv:2202.09778, Feb 2022; ICLR 2022) was Stable Diffusion v1's default sampler: CompVis README, "By default, this uses a guidance scale of `--scale 7.5`, Katherine Crowson's implementation of the PLMS sampler, and renders images of size 512x512 (which it was trained on) in 50 steps."
- Consistency-models paper Table 1 (CIFAR-10, unconditional; collected baselines, NFE → FID): DDIM 10 → 8.23, 20 → 6.84, 50 → 4.67; DPM-solver-2 10 → 5.94; DPM-solver-fast 10 → 4.70; EDM 35 → 2.04. (The DDIM 10-step value 8.23 differs from DDIM's own 13.36, which used DDIM's original model and step spacing; see Appendix A.)
- **Timestep spacing matters** (Lin et al. 2023, Table 2): "leading", "linspace", "trailing" choices of which 50 of the 1,000 timesteps to visit; "trailing" starts at t = 999 and is "noticeably better than linspace" at S = 5.

### Worked examples [DERIVED; `derive.py` parts D–E]
- **EDM σ schedule, N = 18, ρ = 7:** 80, 57.6, 40.8, 28.4, 19.4, 12.9, 8.4, 5.32, 3.26, 1.92, 1.09, 0.585, 0.296, 0.140, 0.0599, 0.0229, 0.00753, 0.002, then 0. Half the steps are spent below σ ≈ 2 (≈ SNR 0.25 in VP terms). Heun cost: 2·18 − 1 = **35 NFE**.
- **Which timesteps does a "50-step" DDIM visit?** diffusers "leading" with offset 1 (SD default): 981, 961, …, 21, 1. "trailing" (Lin et al.): 999, 979, …, 39, 19. "linspace": 999, 979, 958, …, 20, 0. Leading spacing never visits t = 999, so it never starts from the noisiest level the model knows (one of Lin et al.'s "flaws").
- **Swirl solver study:** the course measures error vs steps for Euler and Heun on the swirl itself (`diffusion_reference.py`; measured numbers in `research/experiments/`). Expected shape: Euler error halves when steps double (first order); Heun error quarters (second order), once steps are small enough.

### Common misconceptions
- "More steps is always better." Beyond a point FID plateaus (EDM: no gain past 35 NFE at 32×32); with stochastic samplers, extra steps can add noise artefacts.
- "DPM-Solver/UniPC need retraining." All of these are training-free; they plug into any ε/x₀-prediction model.
- "20 steps of Heun = 20 steps of Euler in cost." 20 Heun steps ≈ 39 NFE.
- "The noise schedule used for sampling must match training exactly." The *noise levels* must be ones the model understands, but which levels you visit, and in what spacing, is a free sampling choice (EDM's σ_i, DDIM's τ, Lin et al.'s trailing).

### Quotable
- "We argue that the theory and practice of diffusion-based generative models are currently unnecessarily convoluted and seek to remedy the situation by presenting a design space that clearly separates the concrete design choices." (Karras et al. 2022, abstract)
- "Here ρ controls how much the steps near σmin are shortened at the expense of longer steps near σmax." (Karras et al. 2022, §3)
- "DPM-Solver can generate high-quality samples in only 10 to 20 function evaluations on various datasets." (Lu et al. 2022, abstract)

---
## §11 Flow matching and rectified flow: straight paths from noise to data; SD3 and FLUX

### Mechanism (plain words)
- **Draw a straight line.** Pair each data point x₀ with a noise sample ε and define the in-between points on the straight segment: x_t = (1 − t)·x₀ + t·ε. At t = 0 it is data, at t = 1 it is noise.
- **Learn the velocity.** Along that line the velocity is constant: ε − x₀. Train a network v_θ(x_t, t) to predict it with a squared error. That is **flow matching / rectified flow**. It is still "add noise, regress", like DDPM, with a different path and target.
- **Sample by following the velocity.** Start at t = 1 with fresh noise and step backwards, x ← x − Δt·v_θ(x, t): an ODE (like DDIM, §9). Because many lines cross, the *learned* (averaged) velocity field is curved, but much straighter than diffusion's, so coarse steps work better.
- **Reflow** (Liu et al.): generate (noise, image) pairs with the trained flow and retrain on those pairs. The new paths cross less and become nearly straight; one Euler step then gives a decent image.
- **One family.** Lipman et al. showed diffusion paths are special cases of Gaussian probability paths and introduced an "optimal transport" (straight-line) path; Albergo & Vanden-Eijnden's **stochastic interpolants** cover the same ground. Flow matching with a particular schedule and v-type target is a reweighted diffusion model.
- **At scale:** Stable Diffusion 3 (Mar 2024) trained rectified-flow transformers up to 8B parameters, sampling training times from a **logit-normal** distribution that emphasises the middle of the path. FLUX.1 (Aug 2024, 12B) "build[s] on flow matching". Movie Gen (2024) chose flow matching for video (§21).

### Key formulas
- **Ours (= SD3's convention; t = 0 data, t = 1 noise):** x_t = (1 − t)·x₀ + t·ε; velocity v = dx_t/dt = **ε − x₀**; loss L = E_{t, x₀, ε} ‖v_θ(x_t, t) − (ε − x₀)‖². Conversions: x̂₀ = x_t − t·v̂; ε̂ = x_t + (1 − t)·v̂. SNR(t) = (1 − t)²/t².
- **Euler sampling:** t_N = 1 > t_{N−1} > … > t_0 = 0; x_{t_{i−1}} = x_{t_i} + (t_{i−1} − t_i)·v_θ(x_{t_i}, t_i).
- **Rectified flow** (Liu, Gong & Liu 2022, eq. 1; X₀ ~ π₀ noise, X₁ ~ π₁ data): min_v ∫₀¹ E‖(X₁ − X₀) − v(X_t, t)‖² dt, X_t = tX₁ + (1 − t)X₀. **Reflow:** (Z₀, Z₁) = (X₀, ODE_v(X₀)); retrain on these pairs ("2-rectified flow").
- **Flow matching, OT path** (Lipman et al. 2022, eqs. 20–23; x₀ noise, x₁ data): μ_t(x₁) = t·x₁, σ_t = 1 − (1 − σ_min)t; ψ_t(x₀) = (1 − (1 − σ_min)t)·x₀ + t·x₁; conditional flow matching loss L_CFM = E ‖v_t(ψ_t(x₀)) − (x₁ − (1 − σ_min)x₀)‖². With σ_min = 0 this is rectified flow.
- **SD3 logit-normal t-sampling** (Esser et al. 2024, §3.1): π_ln(t; m, s) = (1/(s√(2π)))·(1/(t(1 − t)))·exp(−(logit(t) − m)²/(2s²)), logit(t) = log(t/(1 − t)). Best: m = 0, s = 1 ("rf/lognorm(0.00, 1.00)").
- **SD3 resolution shift** (eq. 23, as read from the PDF): t_m = α·t_n / (1 + (α − 1)·t_n), α = √(m/n) (ratio of pixel counts); equivalently λ_{t_m} = λ_{t_n} − 2 log α. SD3 uses **α = 3.0** at 1024×1024 for training and sampling.

### Numbers with sources
- **Rectified flow** (Liu, Gong & Liu, "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow", arXiv:2209.03003, 7 Sep 2022; ICLR 2023), abstract: "The idea of rectified flow is to learn the ODE to follow the straight paths connecting the points drawn from π0 and π1 as much as possible." "on image generation and translation, our method yields nearly straight flows that give high quality results even with a single Euler discretization step." §1: "we claim a state-of-the-art result of FID (4.85) and recall (0.51) on CIFAR10 for one-step fast diffusion/flow models". Table 1 (CIFAR-10, 1 step): 1-rectified flow (no distillation) FID **378**; 1-rectified + distill 6.18; **2-rectified + distill 4.85**; 3-rectified + distill 5.21. Recommendation (discussion of non-linear interpolations): "The canonical linear interpolation Xt = tX1 + (1 − t)X0 should be recommended as a default choice."
- **Flow matching** (Lipman, Chen, Ben-Hamu, Nickel & Le, "Flow Matching for Generative Modeling", arXiv:2210.02747, 6 Oct 2022; ICLR 2023), abstract: "employing FM with diffusion paths results in a more robust and stable alternative for training diffusion models", and OT paths "are more efficient than diffusion paths, provide faster training and sampling, and result in better generalization." **Table 1** (same architecture, adaptive solver; NLL / FID / NFE): CIFAR-10 DDPM 3.12 / 7.48 / 274 vs **FM w/ OT 2.99 / 6.35 / 142**; ImageNet 32 DDPM 3.54 / 6.99 / 262 vs FM-OT 3.53 / 5.02 / 122; ImageNet 64 DDPM 3.32 / 17.36 / 264 vs FM-OT 3.31 / 14.45 / 138; FM w/ diffusion path in between. ImageNet 128 (unconditional): FM-OT NLL 2.90, FID 20.9. Low-NFE: OT path needs "roughly only 60% of the NFEs to reach the same error threshold as diffusion models".
- **Stochastic interpolants:** Albergo & Vanden-Eijnden, "Building Normalizing Flows with Stochastic Interpolants", arXiv:2209.15571, Sep 2022 (ICLR 2023): "The velocity field of this flow is inferred from the probability current of a time-dependent density that interpolates between the base and the target in finite time." Unifying follow-up: Albergo, Boffi & Vanden-Eijnden, "Stochastic Interpolants: A Unifying Framework for Flows and Diffusions", arXiv:2303.08797 (JMLR 26, 2025).
- **SD3** (Esser, Kulal, Blattmann, Entezari, Müller, Saini, Levi, Lorenz, Sauer, Boesel, Podell, Dockhorn, English, Lacey, Goodwin, Marek & Rombach (Stability AI), "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis", arXiv:2403.03206, 5 Mar 2024; ICML 2024):
  - Abstract: "Rectified flow is a recent generative model formulation that connects data and noise in a straight line." "we improve existing noise sampling techniques for training rectified flow models by biasing them towards perceptually relevant scales."
  - Study: "We train each of 61 different formulations on the two datasets" (ImageNet with "a photo of a ⟨class name⟩" captions, and CC12M). "We observe that rf/lognorm(0.00, 1.00) consistently achieves a good rank. It outperforms a rectified flow formulation with uniform timestep sampling (rf) and … (eps/linear) used previously."
  - Scale: "we scale our models up to 8B parameters"; width set by depth d: hidden size 64·d, MLP 4·64·d, d heads. Largest: **d = 38** [DERIVED hidden size 2,432]. 2×2 latent patches.
  - Autoencoder: f = 8 and **d = 16 latent channels** ("For the remainder of this paper, we thus choose d = 16", Table 3).
  - Text: CLIP-L/14 + OpenCLIP bigG/14 pooled outputs concatenated to c_vec ∈ ℝ^2048; their penultimate token outputs (77×768 and 77×1280) concatenated channel-wise to 77×2048, zero-padded to 4096 and concatenated along the sequence with T5-v1.1-XXL's 77×4096 → **c_ctxt ∈ ℝ^{154×4096}** (App. B.2).
  - CFG training: "we set the outputs of each of the three text encoders independently to zero with a probability of 46.4%, such that we roughly train an unconditional model in 10% of all steps" (App. B.3; main text says 46.3%). [DERIVED] 0.464³ = 0.0999.
  - Shift: "we thus use a shift value of α = 3.0 both during training and sampling at resolution 1024 × 1024". Then aligned with DPO (App. C).
  - Evaluation samplers (App. B.3): "we always use a Euler discretization … 50 steps with classifier-free-guidance scales 1.0, 2.5, 5.0, and 5, 10, 25 steps with classifier-free-guidance scale 5.0." Table 6: bigger models "require fewer steps to reach their peak performance".
  - Results: d = 38 beats DALL·E 3 on GenEval overall (Table 5) and "outperforms current proprietary … and open … SOTA generative image models in human preference evaluation".
  - Speed (Stability research-paper post, via helper file S24): "our largest SD3 model with 8B parameters fits into the 24GB VRAM of a RTX 4090 and takes 34 seconds to generate an image of resolution 1024x1024 when using 50 sampling steps." "By removing the memory-intensive 4.7B parameter T5 text encoder for inference, SD3’s memory requirements can be significantly decreased with only small performance loss." [DATED]
- **FLUX.1** (Black Forest Labs, blog 1 Aug 2024, https://bfl.ai/blog/24-08-01-bfl) [DATED]: "All public FLUX.1 models are based on a hybrid architecture of multimodal and parallel diffusion transformer blocks and scaled to 12B parameters. We improve over previous state-of-the-art diffusion models by building on flow matching". Three variants: **[pro]** (API only), **[dev]** ("an open-weight, guidance-distilled model for non-commercial applications. Directly distilled from FLUX.1 [pro]"; FLUX.1 [dev] Non-Commercial License), **[schnell]** (Apache-2.0; card: "a 12 billion parameter rectified flow transformer"; "Trained using latent adversarial diffusion distillation, FLUX.1 [schnell] can generate high-quality images in only 1 to 4 steps"). Resolutions "between 0.1 and 2.0 megapixels". No FLUX.1 technical paper on arXiv; FLUX.1 Kontext has one (arXiv:2506.15742, Jun 2025).
  - Parameters (HF metadata): FLUX.1-schnell transformer **11,891,178,560**; FLUX.1-dev **11,901,408,320** (the 10.2M difference is consistent with dev's guidance embedding [helper inference]).
  - Architecture (diffusers `FluxTransformer2DModel` defaults): **19** double-stream (MM-DiT) blocks + **38** single-stream blocks; 24 heads × 128 = 3,072 hidden; `in_channels` 64 (16 latent channels × 2×2 patches packed); text tokens from T5-XXL (4,096-d), pooled CLIP-L (768-d); rotary position embeddings over 3 axes (16, 56, 56).
  - diffusers `FluxPipeline` defaults [DATED, v0.42]: `num_inference_steps=28`, `guidance_scale=3.5` (the *distilled* guidance input for [dev], not CFG; `true_cfg_scale=1.0` = no CFG), `max_sequence_length=512`. Schnell is run with 4 steps and guidance 0.

### Worked examples [DERIVED; `derive.py` parts G–H]
- **The swirl point on a straight path:** x₀ = (1.0, 0.5), ε = (−0.4, 1.2). x_0.25 = 0.75·x₀ + 0.25·ε = (0.65, 0.675); x_0.5 = (0.30, 0.85); x_0.75 = (−0.05, 1.025). Velocity everywhere = ε − x₀ = (−1.4, 0.7). One Euler step from t = 1 with the *exact* velocity lands exactly on x₀; with a learned (averaged) velocity it lands at the average of plausible x₀s, as in DDIM's 1-step toy (§9).
- **Signal and noise on the straight path:** at t = 0.5, signal coefficient 0.5 and noise coefficient 0.5: SNR = 1 but total variance 0.5 for unit-variance data (the path is not variance-preserving).
- **Logit-normal(0, 1) vs uniform:** P(0.25 < t < 0.75) = **0.728** (uniform: 0.5); P(t < 0.1) = P(t > 0.9) = **0.014** each (uniform: 0.1). Training effort moves from the nearly-clean and nearly-pure-noise ends to the middle.
- **SD3 shift α = 3:** t = 0.1 → 0.25; 0.25 → 0.5; 0.5 → 0.75; 0.75 → 0.9; 0.9 → 0.964. At 1024², half of the sampling path is spent above t = 0.75 (noisier), because a megapixel image keeps more signal at the same t.
- **Token counts:** SD3/FLUX at 1024×1024: latent 128×128×16 → 2×2 patches → **4,096 image tokens** of dimension 64; plus 154 (SD3) or up to 512 (FLUX T5) text tokens.

### Common misconceptions
- "Flow matching is not diffusion." It is the same family: Gaussian paths between data and noise, trained by regression; rectified flow is a particular schedule and parameterisation. BFL calls FLUX blocks "diffusion transformer blocks" while "building on flow matching".
- "Straight training paths mean straight sampling paths." The *conditional* paths are straight; the learned marginal velocity field is curved where paths cross. Reflow or distillation straightens it.
- "Rectified flow samples in one step." 1-rectified flow at 1 step gave FID 378 on CIFAR-10; one-step quality needed reflow **plus distillation** (4.85).
- "t = 0 is noise." Depends on the paper: Lipman and Liu use t = 0 noise, SD3 (and this course) t = 0 data.
- "The flow-matching velocity is SD 2's v-prediction." Different targets (notation table).

### Quotable
- "Rectified flow is a recent generative model formulation that connects data and noise in a straight line." (Esser et al. 2024, abstract)
- "The straight paths are special and preferred because they are the shortest paths between two points, and can be simulated exactly without time discretization" (Liu et al. 2022, abstract)
- "Flow Matching is compatible with a general family of Gaussian probability paths for transforming between noise and data samples—which subsumes existing diffusion paths as specific instances." (Lipman et al. 2022, abstract)

---

## §12 Distillation: from many steps to a few (progressive distillation, consistency models, LCM, ADD/SDXL Turbo, SDXL-Lightning, DMD)

### Mechanism (plain words)
- **The goal:** the same picture quality with 1–8 network calls instead of 25–1,000.
- **Progressive distillation** (Salimans & Ho 2022): a student learns to do in **one** step what the teacher does in **two** DDIM steps. Then the student becomes the teacher and the number of steps halves again: 8192 → … → 4 → 2 → 1. Needs a parameterisation that stays sensible at zero SNR (x₀- or v-prediction).
- **Consistency models** (Song, Dhariwal, Chen & Sutskever 2023): learn a function f(x_t, t) that maps *any* point on a probability-flow trajectory straight to that trajectory's start (the clean image). Train by forcing f to agree on adjacent points (with a teacher ODE solver: consistency distillation; or without: consistency training). One call gives an image; extra calls refine it.
- **Latent consistency models** (LCM, 2023) apply this to Stable Diffusion's latent space with guidance folded in: 2–4 steps, distilled in ~32 A100-hours; LCM-LoRA packages it as a plug-in adapter.
- **Adversarial distillation** (ADD → SDXL Turbo, Nov 2023): a student generates in 1–4 steps and is trained with two signals: a GAN discriminator (trainable heads on a frozen pretrained ViT feature network; Table 1a: "Small, modern DINO networks perform best") for realism, and a distillation loss to a frozen teacher's denoised output. **LADD** (latent version) produced SD3-Turbo and FLUX.1 [schnell].
- **SDXL-Lightning** (ByteDance, Feb 2024) combines progressive and adversarial distillation; **DMD** (2023) matches the teacher's output *distribution*: the gradient of an approximate KL "can be expressed as the difference between 2 score functions" (real vs generated), plus a regression loss on teacher noise–image pairs; **DMD2** (2024) drops the regression loss and adds a GAN loss.
- **Guidance distillation** (Meng et al. 2022) bakes CFG into one network call (§14); FLUX.1 [dev] is "guidance-distilled".
- **Trade-off:** distilled models usually lose diversity and/or the ability to use CFG and negative prompts freely; ADD does not use CFG at inference.

### Key formulas
- **Progressive distillation target** (Salimans & Ho Alg. 2): from z_t, run 2 teacher DDIM steps to z_{t−1/N}; solve for the x̃ that would make one student DDIM step land there; train the student's x̂ toward x̃ (weighted by truncated SNR). Then N ← N/2.
- **Consistency function** (Song et al. 2023): f_θ(x_t, t) = c_skip(t)·x + c_out(t)·F_θ(x, t), with c_skip(ε) = 1, c_out(ε) = 0 so f_θ(x, ε) = x (boundary condition). Self-consistency: f(x_t, t) = f(x_{t′}, t′) for all t, t′ on one PF-ODE trajectory. Consistency distillation loss: d(f_θ(x_{t_{n+1}}, t_{n+1}), f_{θ⁻}(x̂^φ_{t_n}, t_n)), x̂^φ_{t_n} = one ODE-solver step of the teacher; θ⁻ = EMA "target network".
- **ADD objective** (Sauer et al. 2023, eq. 1): L = L_adv^G(x̂_θ(x_s, s), φ) + λ·L_distill(x̂_θ(x_s, s), ψ); student timesteps s chosen from N = 4 values.

### Numbers with sources
- **Progressive distillation** (Salimans & Ho, arXiv:2202.00512, Feb 2022; ICLR 2022), abstract: "we start out with state-of-the-art samplers taking as many as 8192 steps, and are able to distill down to models taking as few as 4 steps without losing much perceptual quality; achieving, for example, a FID of 3.0 on CIFAR-10 in 4 steps." Table 2 (CIFAR-10 FID by model evaluations): **1 → 9.12, 2 → 4.51, 4 → 3.00, 8 → 2.57**; DDIM 10 → 13.36, 20 → 6.84, 50 → 4.67, 100 → 4.16; Luhman & Luhman knowledge distillation 1 → 9.36; LSGM 138 → 2.10. Cost: 50K updates per halving (100K for the 2- and 1-step stages); "the computational cost of progressive distillation to 4 sampling steps is comparable or less than for training the original model."
- **Consistency models** (Song, Dhariwal, Chen & Sutskever, arXiv:2303.01469, 2 Mar 2023; ICML 2023), abstract: "achieving the new state-of-the-art FID of 3.55 on CIFAR-10 and 6.20 on ImageNet 64 × 64 for one-step generation." Tables 1–2: consistency distillation (CD, LPIPS) CIFAR-10 **1 step 3.55, 2 steps 2.93**; ImageNet 64 1 step 6.20, 2 steps 4.70; consistency training (CT, no teacher) CIFAR-10 1 step 8.70, 2 steps 5.83; ImageNet 64 CT 13.0 / 11.1. Progressive distillation in the same table: CIFAR-10 1 step 8.34, ImageNet 64 1 step 15.39.
- **Later consistency work [DATED]:** iCT (Song & Dhariwal, arXiv:2310.14189, Oct 2023): "FID scores of 2.51 and 3.25 on CIFAR-10 and ImageNet 64×64 respectively in a single sampling step" (no distillation). sCM (Lu & Song, arXiv:2410.11081, Oct 2024): two-step FID 2.06 CIFAR-10, 1.48 ImageNet 64, **1.88 ImageNet 512**. MeanFlow (Geng et al., arXiv:2505.13447, May 2025): "FID of 3.43 with a single function evaluation (1-NFE) on ImageNet 256x256 trained from scratch".
- **LCM** (Luo, Tan, Huang, Li & Zhao, arXiv:2310.04378, Oct 2023), abstract: "a high-quality 768×768 2∼4-step LCM takes only 32 A100 GPU hours for training." **LCM-LoRA** (Luo et al., arXiv:2311.05556, Nov 2023): LoRA distillation for SD-V1.5, SSD-1B and SDXL as "a universal Stable-Diffusion acceleration module".
- **ADD / SDXL Turbo** (Sauer, Lorenz, Blattmann & Rombach, arXiv:2311.17042, 28 Nov 2023), abstract: "a novel training approach that efficiently samples large-scale foundational image diffusion models in just 1–4 steps while maintaining high image quality." "ADD is the first method to unlock single-step, real-time image synthesis with foundation models." §1: "Using four sampling steps, ADD-XL outperforms its teacher model SDXL-Base at a resolution of 512² px." "During inference, our approach does not use classifier-free guidance". **Table 2** (all SD 1.5-based, A100, 512×512, mixed precision; steps / time s / FID / CLIP): DPM Solver 25 / 0.88 / 20.1 / 0.318; DPM Solver 8 / 0.34 / 31.7 / 0.320; Progressive Distillation 1 / 0.09 / 37.2, 2 / 0.13 / 26.0, 4 / 0.21 / 26.4; CFG-Aware Distillation 8 / 0.34 / 24.2; InstaFlow-0.9B 1 / 0.09 / 23.4; InstaFlow-1.7B 1 / 0.12 / 22.4; UFOGen 1 / 0.09 / 22.5; **ADD-M 1 / 0.09 / 19.7 / 0.326**.
- SDXL Turbo release (Stability AI, 28 Nov 2023, via helper file S24) [DATED]: "On an A100, SDXL Turbo generates a 512x512 image in 207ms (prompt encoding + a single denoising step + decoding, fp16), where 67ms are accounted for by a single UNet forward evaluation." Licence at release: non-commercial research (now Stability community licence terms).
- **LADD** (Sauer et al., "Fast High-Resolution Image Synthesis with Latent Adversarial Diffusion Distillation", arXiv:2403.12015, Mar 2024): the method behind SD3-Turbo and FLUX.1 [schnell] (schnell card).
- **SDXL-Lightning** (Lin, Wang & Yang (ByteDance), arXiv:2402.13929, Feb 2024): "Our method combines progressive and adversarial distillation to achieve a balance between quality and mode coverage." Released 1-, 2-, 4- and 8-step models at 1024 px, as full U-Nets and LoRAs.
- **DMD** (Yin, Gharbi, Zhang, Shechtman, Durand, Freeman & Park, arXiv:2311.18828, Nov 2023; CVPR 2024): "reaching 2.62 FID on ImageNet 64×64 and 11.49 FID on zero-shot COCO-30k, comparable to Stable Diffusion but orders of magnitude faster. Utilizing FP16 inference, our model can generate images at 20 FPS on modern hardware." **DMD2** (arXiv:2405.14867, May 2024; NeurIPS 2024): "FID scores of 1.28 on ImageNet-64×64 and 8.35 on zero-shot COCO 2014, surpassing the original teacher despite a 500× reduction in inference cost."
- **Rectified-flow + distillation:** 2-rectified flow + distill, CIFAR-10 one-step FID 4.85 (§11).

### Worked example [DERIVED]
**Halving schedule:** 8192 → 4096 → 2048 → 1024 → 512 → 256 → 128 → 64 → 32 → 16 → 8 → 4 → 2 → 1 is 13 halvings; at 50K updates each (100K for the last two) that is 11×50K + 2×100K = **750K** student updates from 8192 steps to 1 (CIFAR-10 setting).
**Wall-clock (ADD Table 2):** 25 DPM-Solver steps 0.88 s vs 1 ADD step 0.09 s: ≈ **10×** faster for a better FID (20.1 → 19.7) on the same SD 1.5 base. The 1-step time is not 1/25 of 0.88 s because text encoding and VAE decoding are a fixed cost (SDXL Turbo: 140 of 207 ms outside the U-Net).
**Lighthouse on this Mac:** the course measures a one-step turbo model vs 50 steps (`research/experiments/`); expect fixed costs (text encoder, VAE) to dominate the 1-step timing.

### Common misconceptions
- "Distillation makes the network smaller." Usually the student has the teacher's architecture; it is the *number of calls* that shrinks.
- "One-step models are free lunches." They typically lose diversity or controllability (no CFG/negative prompt in ADD; distilled guidance in FLUX dev), and progressive distillation's 1-step FID (9.12) was far from 4-step (3.00).
- "Consistency models require a teacher." Consistency *training* (CT) learns from data alone (CIFAR-10 1-step 8.70; improved CT 2.51).
- "SDXL Turbo is 1024 px." It is trained for 512×512 ("Preferably, the model generates images of size 512x512", card).

### Quotable
- "we start out with state-of-the-art samplers taking as many as 8192 steps, and are able to distill down to models taking as few as 4 steps without losing much perceptual quality" (Salimans & Ho 2022, abstract)
- "They support fast one-step generation by design, while still allowing multistep sampling to trade compute for sample quality." (Song et al. 2023, abstract, on consistency models)
- "ADD is the first method to unlock single-step, real-time image synthesis with foundation models." (Sauer et al. 2023, abstract)

---

## §13 Conditioning and classifier guidance

### Mechanism (plain words)
- **A condition is an extra input.** A class label (0–9 for digits, 1,000 ImageNet classes) or a text embedding is fed to the denoiser alongside x_t and t, so it learns p(x | c) instead of p(x). Class labels enter as a learned embedding added to the time embedding (ADM: through AdaGN); text enters through cross-attention (§15).
- **Same noise, different label.** With a deterministic sampler, one starting noise turned into 0, 1, …, 9 by changing only the label shows that the label steers the denoising path (the digit model's ten-label demo).
- **Classifier guidance** (Dhariwal & Nichol 2021): train a separate classifier p_φ(y | x_t) on *noisy* images; during sampling, nudge each step along the classifier's gradient ∇ log p_φ(y | x_t), scaled by s. Bayes: ∇ log p(x | y) = ∇ log p(x) + ∇ log p(y | x), so this samples from the conditional; with s > 1 it samples from a *sharpened* distribution ∝ p(x)·p(y|x)^s: higher fidelity and Inception Score, lower diversity (recall).
- **The catch:** you need a classifier trained on noisy images at every noise level, and the gradients can act like adversarial perturbations. Classifier-free guidance (§14) removed the classifier.

### Key formulas
- **Classifier-guided DDPM step** (ADM Alg. 1): μ, Σ ← μ_θ(x_t), Σ_θ(x_t); x_{t−1} ~ N(μ + s·Σ·∇_{x_t} log p_φ(y | x_t), Σ).
- **Classifier-guided DDIM** (ADM Alg. 2, eq. 14): ε̂(x_t) = ε_θ(x_t) − √(1 − ᾱ_t)·s·∇_{x_t} log p_φ(y | x_t), then a DDIM step with ε̂.
- **Why s > 1 sharpens** (ADM §4.2): s·∇ₓ log p(y|x) = ∇ₓ log (1/Z)·p(y|x)^s.
- Score form: ∇ log p_t(x | y) = ∇ log p_t(x) + ∇ log p_t(y | x) (ADM eqs. 12–13).

### Numbers with sources
- **ADM Table 4** (ImageNet 256×256, 2M iterations, batch 256; FID / sFID / IS / precision / recall):
| Model | Guidance scale | FID | sFID | IS | Prec. | Recall |
|---|---|---|---|---|---|---|
| Unconditional | none | 26.21 | 6.35 | 39.70 | 0.61 | 0.63 |
| Unconditional | 1.0 | 33.03 | 6.99 | 32.92 | 0.56 | 0.65 |
| Unconditional | 10.0 | 12.00 | 10.40 | 95.41 | 0.76 | 0.44 |
| Conditional | none | 10.94 | 6.02 | 100.98 | 0.69 | 0.63 |
| Conditional | 1.0 | **4.59** | 5.25 | 186.70 | 0.82 | 0.52 |
| Conditional | 10.0 | 9.11 | 10.93 | 283.92 | 0.88 | 0.32 |
- ADM §4.2 text: "classifier guidance improves precision at the cost of recall, thus introducing a trade-off in sample fidelity versus diversity." Fig. 3 (unconditional model guided to "Pembroke Welsh corgi"): "Using classifier scale 1.0 (left; FID: 33.0) does not produce convincing samples in this class, whereas classifier scale 10.0 (right; FID: 12.0) produces much more class-consistent images." "When using a scale of 1, we observed that the classifier assigned reasonable probabilities (around 50%) to the desired classes for the final samples, but these samples did not match the intended classes upon visual inspection."
- Fig. 5 (ImageNet 128): trading off with the gradient scale is "strictly better than BigGAN-deep when trading off FID for Inception Score" (via truncation for BigGAN).
- Classifier cost (Table 10): 9–46 V100-days on top of 916 for the generator (ImageNet 256).

### Worked example [DERIVED]
**Why scale 1 is not enough for an unconditional model:** at scale s the effective target is ∝ p(x)·p(y|x)^s. If a noisy "4" has p(y = 9 | x) = 0.3, then at s = 1 the weight is 0.3, at s = 10 it is 0.3¹⁰ ≈ 6 × 10⁻⁶ relative to a confident 9 with p = 1: large s pushes samples to where the classifier is sure.
**The digit model:** the course's model is class-conditional (classifier-free style, §14), so the "one seed, ten labels" grid uses conditioning, not a classifier.

### Common misconceptions
- "Guidance adds information." It reweights the distribution toward samples the classifier is confident about; it reduces diversity (recall 0.63 → 0.52 → 0.32 as scale rises).
- "Scale 1 means conditional sampling works." For an unconditional model, scale 1 is the "correct" Bayes posterior but gave FID 33 and wrong-looking classes; the useful setting was ~10.
- "Classifier guidance uses an ImageNet classifier off the shelf." It needs a classifier trained on noisy x_t (GLIDE notes that "the public CLIP models, which have not been trained on noised images, can still be used to guide diffusion models", citing Crowson 2021).

### Quotable
- "a simple, compute-efficient method for trading off diversity for fidelity using gradients from a classifier" (Dhariwal & Nichol 2021, abstract)
- "Table 4 also shows that classifier guidance improves precision at the cost of recall, thus introducing a trade-off in sample fidelity versus diversity." (Dhariwal & Nichol 2021, §4.2)

---

## §14 Classifier-free guidance (CFG)

### Mechanism (plain words)
- **One network, two questions.** Train a single conditional model, but replace the condition with a "null" token ∅ some fraction of the time (10–20% is typical). The same network then predicts both ε(x_t, c) ("what noise, given the prompt?") and ε(x_t, ∅) ("what noise, for any image?").
- **Extrapolate.** At sampling time compute both and move *past* the conditional prediction, away from the unconditional one: ε̃ = ε_u + s·(ε_c − ε_u). The difference ε_c − ε_u is (up to scale) the gradient of an *implicit classifier* log p(c | x). With s > 1 images match the prompt more strongly, at the cost of diversity, and at high s colours oversaturate.
- **Cost:** every step needs two predictions (usually one batch of 2): CFG doubles sampling compute. Guidance distillation (Meng et al. 2022) trains a student to produce the guided output in one pass (FLUX.1 [dev]).
- **Two conventions.** Ho & Salimans: ε̃ = (1 + w)ε_c − w·ε_u, w = 0 = no guidance. GLIDE / Stable Diffusion / diffusers: guidance scale s with s = 1 = no guidance. **s = 1 + w**: SD's default 7.5 is w = 6.5.
- **Fixes for high scales:** dynamic thresholding (Imagen, pixel space), CFG rescale (Lin et al. 2023, latent space: match the guided output's std to the conditional one's, φ = 0.7), guidance only in a middle interval of noise levels (Kynkäänniemi et al. 2024), autoguidance (Karras et al. 2024: guide with a worse version of the model).

### Key formulas
- **CFG (Ho & Salimans eq. 6):** ε̃_θ(z_λ, c) = (1 + w)·ε_θ(z_λ, c) − w·ε_θ(z_λ).
- **Scale form (GLIDE §2.3; CompVis `--scale`; diffusers):** ε̂ = ε_θ(x_t | ∅) + s·(ε_θ(x_t | c) − ε_θ(x_t | ∅)); "s ≥ 1 is the guidance scale". CompVis help text: "unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))".
- **Implicit classifier** (Ho & Salimans §3.2): p^i(c | z) ∝ p(z | c)/p(z), so ∇ log p^i(c|z) = −(1/σ)(ε*(z, c) − ε*(z)).
- **Training (Ho & Salimans Alg. 1):** "c ← ∅ with probability p_uncond. Randomly discard conditioning to train unconditionally".
- **CFG rescale** (Lin et al. 2023, Alg. 2, eq. 13–16): x_cfg = x_neg + w(x_pos − x_neg); factor = std(x_pos)/std(x_cfg); factor ← φ·factor + (1 − φ); output x_cfg·factor; default φ = 0.7 with w = 7.5. diffusers: `guidance_rescale` (default 0.0).

### Numbers with sources
- **Ho & Salimans**, "Classifier-Free Diffusion Guidance": NeurIPS 2021 Workshop on Deep Generative Models; arXiv:2207.12598 (26 Jul 2022). Abstract: "we jointly train a conditional and an unconditional diffusion model, and we combine the resulting conditional and unconditional score estimates to attain a trade-off between sample quality and diversity similar to that obtained using classifier guidance."
- **Table 1 (ImageNet 64×64; p_uncond = 0.1 / 0.2 / 0.5; FID, IS; 50,000 samples; 400K training steps):**
| w | FID (0.1 / 0.2 / 0.5) | IS (0.1 / 0.2 / 0.5) |
|---|---|---|
| 0.0 (no guidance) | 1.8 / 1.8 / 2.21 | 53.71 / 52.9 / 47.61 |
| 0.1 | **1.55** / 1.62 / 1.91 | 66.11 / 64.58 / 56.1 |
| 0.3 | 3.03 / 2.93 / 2.65 | 92.8 / 88.64 / 74.92 |
| 0.5 | 5.74 / 5.19 / 4.34 | 119.3 / 112.15 / 92.95 |
| 1.0 | 12.6 / 11.21 / 9.13 | 170.1 / 158.29 / 131.1 |
| 2.0 | 21.03 / 18.79 / 16.16 | 225.5 / 212.98 / 183 |
| 4.0 | 26.22 / 23.84 / 21.48 | **260.2** / 248.97 / 225.1 |
- **Table 2 (ImageNet 128×128, T = 128 / 256 / 1024 sampling steps; 2.7M training steps):** w = 0: FID 8.11 / 7.27 / 7.22, IS ≈ 82; **w = 0.3: FID 3.04 / 2.43 / 2.43**, IS ≈ 157; w = 1.0: 7.88 / 7.86 / 7.8, IS ≈ 296; w = 4.0: 21.55 / 21.53 / 21.45, IS ≈ 420. Comparison rows: BigGAN-deep 5.7 / 124.5 (max-IS truncation 25 / 253); ADM-G 2.97.
- Text: "We obtain the best FID results with a small amount of guidance (w = 0.1 or w = 0.3, depending on the dataset) and the best IS result with strong guidance (w ≥ 4)." "At w = 0.3, our model's FID score on 128 × 128 ImageNet outperforms the classifier-guided ADM-G, and at w = 4.0, our model outperforms BigGAN-deep at both FID and IS when BigGAN-deep is evaluated its best-IS truncation level." Cost: "each sampling step for our method requires evaluating the denoising model twice".
- **Note:** the paper's sentence "FID monotonically decreasing and IS monotonically increasing with w" contradicts its own tables beyond w ≈ 0.1–0.3 (FID rises). Quote the tables, not that sentence (Appendix A).
- **Dropout rates in practice:** Ho & Salimans 0.1/0.2/0.5 (best FID with 0.1–0.2); GLIDE: "20% of text token sequences are replaced with the empty sequence"; Imagen: "zeroing out the text embeddings with 10% probability"; SD v1.3–v1.5: "10% dropping of the text-conditioning"; DALL·E 2 decoder: CLIP embedding zeroed 10%, caption dropped 50%; SD3: each of 3 encoders zeroed with 46.4% → ≈ 10% fully unconditional; ControlNet: "randomly replace 50% text prompts ct with empty strings".
- **Typical scales [DATED; library defaults]:** CompVis SD v1 README: "By default, this uses a guidance scale of `--scale 7.5`"; diffusers v0.42: SD 7.5, SDXL 5.0, SD3 7.0 (28 steps), FLUX [dev] distilled guidance 3.5; GLIDE used "classifier-free guidance with scale 3.0" for MS-COCO; LDM text-to-image s = 1.5 for its COCO FID (12.63); DiT s = 1.5 (§18); SD3 experiments 1.0–5.0.
- **Guidance distillation:** Meng, Rombach, Gao, Kingma, Ermon, Ho & Salimans, "On Distillation of Guided Diffusion Models", arXiv:2210.03142 (Oct 2022): distils the two-pass guided model into one network that takes w as input, then progressively distils steps.
- **Guidance interval** (Kynkäänniemi et al., arXiv:2404.07724, Apr 2024): "guidance is clearly harmful toward the beginning of the chain (high noise levels), largely unnecessary toward the end (low noise levels), and only beneficial in the middle." Record ImageNet-512 FID "from 1.81 to 1.40". **Autoguidance** (Karras et al., arXiv:2406.02507, Jun 2024): "guiding generation using a smaller, less-trained version of the model itself"; "record FIDs of 1.01 for 64x64 and 1.25 for 512x512".

### Worked example [DERIVED; `derive.py` part F]
At one latent value, suppose ε_u = 0.20 and ε_c = 0.50:
| guidance scale s (w = s − 1) | ε̃ = ε_u + s(ε_c − ε_u) |
|---|---|
| 1 (w = 0) | 0.50 (plain conditional) |
| 3 (w = 2) | 1.10 |
| **7.5 (w = 6.5)** | **2.45** |
| 15 (w = 14) | 4.70 |
- Check in the Ho–Salimans form at w = 6.5: 7.5·0.50 − 6.5·0.20 = 3.75 − 1.30 = 2.45. Same number.
- At s = 7.5 the guided noise prediction is ~5× the size of either raw prediction: the sampler removes "too much" in the prompt direction each step, which is where saturation and burned colours come from at large s.
- **Cost:** 50 steps × 2 predictions = **100 U-Net evaluations** per SD 1.5 image (Apple's benchmark: "classifier-free guidance (batch size of 2 for unet)").

### Common misconceptions
- "Guidance scale 1 is weak guidance." In the scale convention 1 is *no* guidance; in Ho–Salimans w = 1 is already strong (ImageNet 64 FID 1.8 → 12.6).
- "Higher CFG = better images." Best FID is at small guidance (w 0.1–0.3); high scales buy prompt adherence and IS with less diversity and oversaturation.
- "The negative prompt is a separate mechanism." It replaces ε_u with ε(x_t, c_neg) in the same formula (Lin et al.'s x_neg).
- "CFG needs two networks." One network, trained with condition dropout, queried twice.

### Quotable
- "We show that guidance can be indeed performed by a pure generative model without such a classifier" (Ho & Salimans, abstract)
- "we choose joint training because it is extremely simple to implement, does not complicate the training pipeline, and does not increase the total number of parameters." (Ho & Salimans §3.2)
- "We find that as the terminal SNR approaches zero, classifier-free guidance becomes very sensitive and can cause images to be overexposed." (Lin et al. 2023, §3.4)

---

## §15 Text conditioning: CLIP, T5 and cross-attention

### Mechanism (plain words)
- **Tokens, then vectors.** A prompt is split into tokens (CLIP: byte-pair encoding, padded/truncated to 77 positions) and a frozen text encoder turns them into one vector per token (SD 1.x: 77 × 768).
- **CLIP** (Radford et al. 2021) learned text and image encoders together on 400M web image–caption pairs with a contrastive loss: matching pairs should have high cosine similarity, all other pairs in the batch low. Its text vectors are therefore "about what images look like".
- **T5** is a text-only language model. Imagen found that a frozen T5-XXL helped text-to-image more than making the image model bigger. SD3 and FLUX combine CLIP and T5.
- **Cross-attention** connects words to places. Inside the denoiser, image features ask questions (queries Q) and the text tokens answer (keys K, values V). For each image location, softmax(QKᵀ/√d) is a distribution over the prompt's tokens; averaging these maps over heads and layers gives a heat map per word ("lighthouse" lights up the tower).
- **Prompt-to-Prompt** showed these maps control layout: injecting the cross-attention maps from one prompt into the generation of an edited prompt keeps the composition while changing a word.
- **Prompt sensitivity:** tokens beyond 77 are cut (CLIP); rare words map to fragments; a small text change can change the whole image unless attention or noise is kept fixed.

### Key formulas
- **CLIP contrastive loss** (Radford et al. 2021, Fig. 3 pseudocode): logits = (I_e · T_eᵀ)·exp(τ) for a batch of N pairs; loss = (CE(logits, labels, axis=0) + CE(logits, labels, axis=1))/2 with labels = 0…N−1 (symmetric InfoNCE).
- **Cross-attention** (LDM §3.3): Attention(Q, K, V) = softmax(QKᵀ/√d)·V, Q = W_Q^{(i)}·φ_i(z_t), K = W_K^{(i)}·τ_θ(y), V = W_V^{(i)}·τ_θ(y). With SD 1.x at the 64×64 level: Q has 4,096 rows (one per latent position), K and V have 77 rows (one per token); each attention map is 4,096 × 77.
- **CLIPScore** (Hessel et al. 2021): CLIP-S(c, v) = w·max(cos(c, v), 0), w = 2.5.

### Numbers with sources
- **CLIP** (Radford, Kim, Hallacy, Ramesh, Goh, Agarwal, Sastry, Askell, Mishkin, Clark, Krueger & Sutskever, "Learning Transferable Visual Models From Natural Language Supervision", arXiv:2103.00020, Feb 2021; ICML 2021), abstract: "predicting which caption goes with which image is an efficient and scalable way to learn SOTA image representations from scratch on a dataset of 400 million (image, text) pairs collected from the internet." "we match the accuracy of the original ResNet-50 on ImageNet zero-shot without needing to use any of the 1.28 million training examples it was trained on." Dataset WIT built from "one of a set of 500,000 queries", "up to 20,000 (image, text) pairs per query". Text encoder base: "a 63M-parameter 12-layer 512-wide model with 8 attention heads … lower-cased byte pair encoding (BPE) representation of the text with a 49,152 vocab size … the max sequence length was capped at 76" (§2.4). Batch: "a very large minibatch size of 32,768". Compute: "The largest ResNet model, RN50x64, took 18 days to train on 592 V100 GPUs while the largest Vision Transformer took 12 days on 256 V100 GPUs. For the ViT-L/14 we also pre-train at a higher 336 pixel resolution for one additional epoch".
- **CLIP ViT-L/14 as used by SD 1.x** (`openai/clip-vit-large-patch14` config): text transformer 12 layers, width 768, 77 positions, vocab 49,408, projection 768; whole CLIP model 427,616,512 parameters (HF metadata); text encoder alone 123,060,480 (safetensors header, SD 1.5). CompVis README: "Stable Diffusion is a latent diffusion model conditioned on the (non-pooled) text embeddings of a CLIP ViT-L/14 text encoder." [DERIVED] SD 1.x uses the final-layer token outputs, 77 × 768.
- **SD 2.x:** OpenCLIP ViT-H (context dim 1024) (SDXL Table 1). **SDXL:** "we use OpenCLIP ViT-bigG [19] in combination with CLIP ViT-L [34], where we concatenate the penultimate text encoder outputs along the channel-axis", plus the pooled OpenCLIP embedding; "The text encoders have a total size of 817M parameters." Context 77 × 2048 (768 + 1280).
- **Imagen** (Saharia et al., arXiv:2205.11487, May 2022): "Our key discovery is that generic large language models (e.g. T5), pretrained on text-only corpora, are surprisingly effective at encoding text for image synthesis: increasing the size of the language model in Imagen boosts both sample fidelity and image-text alignment much more than increasing the size of the image diffusion model." §4.4: "Scaling text encoder size is extremely effective. … Imagen trained with our largest text encoder, T5-XXL (4.6B parameters), yields the best results (Fig. 4a)." Architecture: frozen T5-XXL → 2B 64×64 model → 600M 64→256 and 400M 256→1024 super-resolution models; batch 2048, 2.5M steps; ≈ 460M internal + ≈ 400M LAION image–text pairs. Zero-shot COCO FID-30K **7.27** (vs GLIDE 12.4, DALL·E 2 10.4). Introduced DrawBench and dynamic thresholding.
- **T5-XXL encoder size:** [DERIVED from the `google/t5-v1_1-xxl` config: d_model 4096, d_ff 10240, 24 layers, 64 heads × 64, vocab 32,128, gated-GELU] **4,762,310,656** parameters including the 131.6M token embedding (4.63B without it; Imagen says 4.6B, SD3 says "4.7B"). bf16 size 9.52 GB.
- **SD3:** three encoders, 154 × 4096 context (§11). **FLUX.1:** CLIP-L (pooled) + T5-XXL (sequence, up to 512 tokens in diffusers).
- **DALL·E 2 (unCLIP)** (Ramesh, Dhariwal, Nichol, Chu & Chen, arXiv:2204.06125, Apr 2022): "a prior that generates a CLIP image embedding given a text caption, and a decoder that generates an image conditioned on the image embedding". Decoder "the 3.5 billion parameter GLIDE model"; prior 1B; upsamplers 700M (64→256) and 300M (256→1024); zero-shot COCO FID 10.39.
- **GLIDE** (Nichol, Dhariwal, Ramesh, Shyam, Mishkin, McGrew, Sutskever & Chen, arXiv:2112.10741, Dec 2021): 3.5B text-conditional 64×64 model + 1.5B upsampler to 256×256; trained "on the same dataset as DALL-E". "our samples are preferred to those from DALL-E (Ramesh et al., 2021) 87% of the time when evaluated for photorealism, and 69% of the time when evaluated for caption similarity." Zero-shot MS-COCO FID 12.24 (Table 2).
- **Prompt-to-Prompt** (Hertz, Mokady, Tenenbaum, Aberman, Pritch & Cohen-Or, arXiv:2208.01626, Aug 2022; ICLR 2023), abstract: "we analyze a text-conditioned model in depth and observe that the cross-attention layers are the key to controlling the relation between the spatial layout of the image to each word in the prompt."

### Worked example [DERIVED]
**"a lighthouse on a rocky coast at dusk, oil painting" through SD 1.5:** the CLIP tokenizer gives [start] + about a dozen word/punctuation tokens + [end], padded to 77 (the exact count is measured in the course's run). The text encoder outputs 77 × 768 numbers = **59,136** values, computed **once** per image (twice with CFG: prompt and empty prompt). In each of the U-Net's cross-attention layers at the 64×64 level, 4,096 latent positions each attend over the 77 tokens: a 4,096 × 77 attention matrix per head (8 heads). The column for "lighthouse" reshaped to 64×64 is the heat map shown on screen.

### Common misconceptions
- "The model reads the prompt like a chatbot." SD 1.x sees 77 CLIP token vectors from a frozen encoder trained only to match captions to images; word order and counting are weak points (DrawBench, GenEval probe them).
- "Bigger image models fix prompt following." Imagen found scaling the text encoder mattered more; DALL·E 3 found better *captions* mattered (§20).
- "Cross-attention maps are explanations." They are useful heat maps averaged over heads/layers/steps, not a faithful causal account.
- "CLIP's context is 77 tokens of text." 77 positions including start and end tokens (the CLIP paper says the sequence was "capped at 76" plus bracketing tokens; the released config has 77 positions).

### Quotable
- "predicting which caption goes with which image is an efficient and scalable way to learn SOTA image representations from scratch" (Radford et al. 2021, abstract)
- "increasing the size of the language model in Imagen boosts both sample fidelity and image-text alignment much more than increasing the size of the image diffusion model." (Saharia et al. 2022, abstract)
- "the cross-attention layers are the key to controlling the relation between the spatial layout of the image to each word in the prompt." (Hertz et al. 2022, abstract)

---

## §16 Editing and control: SDEdit, inpainting, ControlNet, T2I-Adapter, LoRA, DreamBooth, textual inversion, IP-Adapter

### Mechanism (plain words)
- **SDEdit / image-to-image** (Meng et al. 2021): take an existing picture (or a rough sketch), add noise up to an intermediate level t₀, and run the normal reverse process from there. Small t₀ keeps the picture and changes texture; large t₀ keeps only the rough layout; t₀ = 1 is a fresh sample. The "strength" slider in Stable Diffusion img2img is this t₀ (fraction of the schedule).
- **Inpainting:** regenerate only a masked region. Either sample normally while overwriting the unmasked area with the (appropriately noised) original at every step, or use a model fine-tuned to take the mask and masked image as extra input channels (GLIDE; SD inpainting models).
- **ControlNet** (Zhang, Rao & Agrawala 2023): freeze the big model; clone its encoder into a trainable copy that also sees a control image (edges, depth, pose, segmentation…); connect the copy back through **zero-initialised 1×1 convolutions**, so at the start of training the control branch adds exactly zero and cannot damage the base model.
- **T2I-Adapter:** a small (~77M) side network that turns a control image into features added to the frozen U-Net.
- **LoRA** (Hu et al. 2021, from LLMs): instead of updating a weight matrix W, learn a low-rank update ΔW = BA (rank r ≪ d). Files are megabytes instead of gigabytes; applied to attention layers of SD, it became the standard way to share styles and subjects.
- **DreamBooth** (Ruiz et al. 2022/2023): fine-tune the whole model on 3–5 photos of one subject bound to a rare token ("a [V] dog"), with a **prior-preservation** loss on generated images of the class so the model doesn't forget what dogs in general look like.
- **Textual inversion** (Gal et al. 2022): learn only a *new word embedding* S* (one vector) from 3–5 images; the model is untouched.
- **IP-Adapter** (one line): adds separate cross-attention layers for image-prompt features (decoupled from text), 22M trainable parameters, so a reference image can steer generation alongside text.

### Key formulas
- **SDEdit:** x_{t₀} = √ᾱ_{t₀}·x_guide + √(1 − ᾱ_{t₀})·ε, then reverse from t₀ to 0. SD img2img "strength" s ⇒ start at t₀ ≈ s·T and run ≈ s·N of the N scheduled steps [DERIVED from the diffusers img2img convention].
- **ControlNet** (Zhang et al. eqs. 1–2): y_c = F(x; Θ) + Z(F(x + Z(c; Θ_z1); Θ_c); Θ_z2), Z = 1×1 convolution with weights and bias initialised to 0; at initialisation y_c = y.
- **LoRA** (Hu et al. §4.1): h = W₀x + ΔWx = W₀x + BAx, B ∈ ℝ^{d×r}, A ∈ ℝ^{r×k}, r ≪ min(d, k); "We use a random Gaussian initialization for A and zero for B, so ∆W = BA is zero at the beginning of training"; ΔWx scaled by α/r. Trainable parameters per adapted matrix: r(d + k).
- **DreamBooth loss** (Ruiz et al. eq. 2): E[w_t‖x̂_θ(α_t x + σ_t ε, c) − x‖² + λ·w_{t′}‖x̂_θ(α_{t′}x_pr + σ_{t′}ε′, c_pr) − x_pr‖²], x_pr = images generated by the frozen model for the class prompt; λ = 1.

### Numbers with sources
- **SDEdit** (Meng, He, Song, Song, Wu, Zhu & Ermon, "SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations", arXiv:2108.01073, Aug 2021; ICLR 2022): "t0 = 0 indicates the guide itself, whereas t0 = 1 indicates a random sample." Realism–faithfulness trade-off: good results for "t0 ∈ [0.3, 0.6] in this example" (Fig. 3). Human study: "outperforms state-of-the-art GAN-based approaches by up to 98.09% on realism score and 91.72% on overall satisfaction score".
- **SDXL's refiner** uses the same idea: "we utilize a specialized high-resolution refinement model and apply SDEdit [28] on the latents generated in the first step, using the same prompt" (SDXL Fig. 1 caption).
- **ControlNet** (Zhang, Rao & Agrawala, "Adding Conditional Control to Text-to-Image Diffusion Models", arXiv:2302.05543, Feb 2023; ICCV 2023): abstract: "ControlNet locks the production-ready large diffusion models, and reuses their deep and robust encoding layers pretrained with billions of images as a strong backbone to learn a diverse set of conditional controls. The neural architecture is connected with “zero convolutions” (zero-initialized convolution layers) that progressively grow the parameters from zero and ensure that no harmful noise could affect the finetuning." Conditions tested: "Canny edges, Hough lines, user scribbles, human key points, segmentation maps, shape normals, depths, and cartoon line drawings". "the training of ControlNets is robust with small (<50k) and large (>1m) datasets." Cost: "optimizing Stable Diffusion with ControlNet requires only about 23% more GPU memory and 34% more time in each training iteration" (A100 40GB). "we randomly replace 50% text prompts ct with empty strings". **Sudden convergence:** the model "abruptly succeeds in following the input conditioning image; usually in less than 10K optimization steps." Depth example: a ControlNet for SD 2 trained on "200k training samples, one single NVIDIA RTX 3090Ti, and 5 days of training" vs SD's own depth model (A100 cluster, "more than 12M training images").
- **T2I-Adapter** (Mou et al., arXiv:2302.08453, Feb 2023): "lightweight with ∼ 77 M parameters and ∼ 300 M storage space"; trained on 4 V100 GPUs.
- **LoRA** (Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang & Chen, arXiv:2106.09685, Jun 2021; ICLR 2022), abstract: "Compared to GPT-3 175B fine-tuned with Adam, LoRA can reduce the number of trainable parameters by 10,000 times and the GPU memory requirement by 3 times." §: "a very low rank (i.e., r in Figure 1 can be one or two) suffices".
- **DreamBooth** (Ruiz, Li, Jampani, Pritch, Rubinstein & Aberman, arXiv:2208.12242, Aug 2022; CVPR 2023): "With just a few images (typically 3-5) of a subject"; "∼ 1000 iterations with λ = 1 and learning rate 10−5 for Imagen [61] and 5 × 10−6 for Stable Diffusion [59], and with a subject dataset size of 3-5 images is enough to achieve good results. During this process, ∼ 1000 “a [class noun]” samples are generated … The training process takes about 5 minutes on one TPUv4 for Imagen, and 5 minutes on a NVIDIA A100 for Stable Diffusion."
- **Textual inversion** (Gal, Alaluf, Atzmon, Patashnik, Bermano, Chechik & Cohen-Or, "An Image is Worth One Word", arXiv:2208.01618, Aug 2022; ICLR 2023): "Using only 3-5 images of a user-provided concept"; "a single word embedding is sufficient for capturing unique and varied concepts". Training: "2×V100 GPUs with a batch size of 4", base learning rate 0.005 (effective 0.04), "5,000 optimization steps" (on LDM).
- **IP-Adapter** (Ye, Zhang, Liu, Han & Yang, arXiv:2308.06721, Aug 2023): "an IP-Adapter with only 22M parameters can achieve comparable or even better performance to a fully fine-tuned image prompt model".

### Worked examples [DERIVED]
- **img2img strength on the lighthouse (SD, linear-ish schedule, 50 steps):** strength 0.3 → run the last ≈ 15 of 50 steps, starting near t₀ ≈ 300, where SD's ᾱ_300 = 0.592 (√ᾱ = 0.77: 77% of the signal amplitude survives): colours and brushwork change, layout stays. Strength 0.9 → ≈ 45 of 50 steps from t₀ ≈ 900 (ᾱ = 0.0143, √ᾱ = 0.12): only the coarse composition survives. (SD scaled-linear ᾱ at t = 100/300/500/700/900: 0.896 / 0.592 / 0.278 / 0.083 / 0.014.) The course measures the actual frames (`research/experiments/`).
- **LoRA size for one SD 1.5 attention projection** (768 → 320 at the 64×64 level, k = 768, d = 320): full matrix 245,760 parameters; rank-4 LoRA r(d + k) = 4·1,088 = 4,352 (1.8%). Rank 4 over all ~100+ projection matrices of the U-Net is typically a few MB in fp16 vs 1.7 GB for the U-Net. [order-of-magnitude; exact file sizes depend on which layers are adapted **[UNVERIFIED general claim]**]
- **ControlNet clone size:** the trainable copy duplicates SD's encoder half plus middle block. Safetensors header of `lllyasviel/sd-controlnet-canny` (SD 1.5): **361,279,120** parameters, 42% of the 859.5M U-Net it controls (fp32 file ≈ 1.45 GB).

### Common misconceptions
- "img2img strength is the amount of blending with the original." It is how far up the noise schedule the original is pushed before regenerating.
- "ControlNet retrains Stable Diffusion." The base model is locked; only the copy and zero convolutions train.
- "Zero-initialised layers can't learn." The gradient with respect to a zero-initialised conv's weights depends on its input, which is non-zero, so they move away from zero after the first step. The paper: "Zero convolutions protect this backbone by eliminating random noise as gradients in the initial training steps", with the gradient calculation in its supplementary material.
- "DreamBooth = LoRA." DreamBooth is a fine-tuning recipe (rare token + prior preservation); it is often *implemented* with LoRA to save memory.
- "Textual inversion teaches the model new visuals." It only finds a new input embedding that the frozen model already maps to the concept.

### Quotable
- "The neural architecture is connected with “zero convolutions” (zero-initialized convolution layers) that progressively grow the parameters from zero and ensure that no harmful noise could affect the finetuning." (Zhang et al. 2023, abstract)
- "t0 = 0 indicates the guide itself, whereas t0 = 1 indicates a random sample." (Meng et al. 2021, Fig. 3)
- "a single word embedding is sufficient for capturing unique and varied concepts." (Gal et al. 2022, abstract)

---
## §17 Latent diffusion and Stable Diffusion: diffuse in a compressed space

### Mechanism (plain words)
- **Most pixels are detail you can't see.** DDPM found that "more than half of the lossless codelength describes imperceptible distortions" (§5). Rombach et al. split the job: an **autoencoder** handles *perceptual* compression (pixels → a small latent grid that still decodes to a sharp image), and the diffusion model handles *semantic* composition in that latent space.
- **The autoencoder** (a VAE trained with reconstruction + perceptual (LPIPS) + patch-GAN losses and a tiny KL penalty, weight ~10⁻⁶) downsamples by f = 8 in each direction: a 512×512×3 image becomes a **64×64×4** latent, 48× fewer numbers. It is trained once and frozen.
- **Diffusion runs on z.** Training: encode, add noise to z, predict the noise (same loss as §5). Sampling: denoise a 64×64×4 noise grid, then **decode once** with the VAE decoder to 512×512×3.
- **Why it is cheaper:** the U-Net works on 64×64 instead of 512×512 (64× fewer positions per channel), so both training and sampling cost drop by large factors. LDM-4 on ImageNet beat ADM-G's FID with ~3.5× less training compute and ~6× higher sampling throughput.
- **Latent scaling.** The KL latents have a std far from 1, so they are multiplied by a constant before diffusion (SD 1.x: **0.18215**, "scale_factor" in the config) and divided by it before decoding.
- **Stable Diffusion** (Aug 2022) is a latent diffusion model: CLIP ViT-L/14 text encoder + 860M U-Net + f = 8, 4-channel VAE, trained on LAION-5B subsets with compute from Stability AI. Its open release (CreativeML OpenRAIL-M) is why the ecosystem in §16 exists. Later: SD 2.x (OpenCLIP, 768 px, v-prediction), SDXL (2.6B U-Net, 1024 px, refiner), SD3/3.5 (MM-DiT rectified flow, 16-channel VAE).

### Key formulas
- Encoder/decoder: z = E(x), x̃ = D(z) = D(E(x)); x ∈ ℝ^{H×W×3} → z ∈ ℝ^{h×w×c}, h = H/f, w = W/f (LDM §3.1).
- **L_LDM** (LDM eq. 2): E_{E(x), ε~N(0,1), t} ‖ε − ε_θ(z_t, t)‖²; conditional (eq. 3): ε_θ(z_t, t, τ_θ(y)).
- Autoencoder objective (LDM eq. 25): L_Autoencoder = min_{E,D} max_ψ [L_rec(x, D(E(x))) − L_adv(D(E(x))) + log D_ψ(x) + L_reg(x; E, D)].
- Latent rescale (LDM App. G): σ̂² = (1/bchw)Σ(z − μ̂)² estimated "from the first batch in the data"; "The output of E is scaled such that the rescaled latent has unit standard deviation, i.e. z ← z/σ̂ = E(x)/σ̂". SD v1 config `scale_factor: 0.18215` [DERIVED: implies σ̂ ≈ 5.49].
- Compression ratio: (H·W·3)/((H/f)·(W/f)·c) = 3f²/c: f = 8, c = 4 → **48**; f = 8, c = 16 → 12.

### Numbers with sources
- **LDM** (Rombach, Blattmann, Lorenz, Esser & Ommer, "High-Resolution Image Synthesis with Latent Diffusion Models", arXiv:2112.10752, 20 Dec 2021; CVPR 2022). §1: "training the most powerful DMs often takes hundreds of GPU days (e.g. 150 - 1000 V100 days in [15]) and repeated evaluations on a noisy version of the input space render also inference expensive, so that producing 50k samples takes approximately 5 days [15] on a single A100 GPU."
- Perceptual vs semantic compression (Fig. 2 caption): "Most bits of a digital image correspond to imperceptible details."
- KL weight (App. G): "we only use a very small regularization for both scenarios, i.e. we either weight the KL term by a factor ∼ 10−6 or choose a high codebook dimensionality |Z|."
- **First-stage autoencoders (Table 8, OpenImages-trained; f / c / R-FID / PSNR):** KL f = 4, c = 3: 0.27 / 27.53; **KL f = 8, c = 4: 0.90 / 24.19** (the SD-style config); KL f = 16, c = 16: 0.87 / 24.08; KL f = 32, c = 64: 2.04 / 22.27; VQ f = 8 |Z| = 16384 c = 4: 1.14 / 23.07; DALL·E's dVAE f = 8: 32.01 / 22.8.
- Downsampling choice (§4.1): LDM-1 (pixel space) is slow; "LDM-4 and -8 offer the best conditions for achieving high-quality synthesis results"; too much compression (f = 32) limits quality.
- **Class-conditional ImageNet 256 (Table 3 / Table 18; FID / IS / P / R / params / compute V100-days / A100 throughput samples·s⁻¹):** ADM-G 4.59 / 186.7 / 0.82 / 0.52 / 608M / **962** / 0.07; **LDM-4-G (cfg 1.5) 3.60 / 247.67 / 0.87 / 0.48 / 400M / 271 / 0.4**; LDM-4 no guidance 10.56. [DERIVED] 962/271 = 3.5× less training compute, 0.4/0.07 = 5.7× higher throughput.
- **Text-to-image** (§4.3.1, Table 2): "we train a 1.45B parameter KL-regularized LDM conditioned on language prompts on LAION-400M"; MS-COCO 256 FID **12.63** with CFG s = 1.5 (250 DDIM steps), 23.31 without.
- **Stable Diffusion v1** (CompVis README): "we were able to train a Latent Diffusion Model on 512x512 images from a subset of the LAION-5B database"; "this model uses a frozen CLIP ViT-L/14 text encoder"; "With its 860M UNet and 123M text encoder, the model is relatively lightweight and runs on a GPU with at least 10GB VRAM." "pretrained on 256x256 images and then finetuned on 512x512 images." Checkpoints:
  - `sd-v1-1`: "237k steps at resolution `256x256` on laion2B-en. 194k steps at resolution `512x512` on laion-high-resolution (170M examples from LAION-5B with resolution `>= 1024x1024`)."
  - `sd-v1-2`: "515k steps at resolution `512x512` on laion-aesthetics v2 5+ (a subset of laion2B-en with estimated aesthetics score `> 5.0`, and additionally filtered to images with an original size `>= 512x512`, and an estimated watermark probability `< 0.5`."
  - `sd-v1-3`: 195k more steps on laion-aesthetics v2 5+ "and 10% dropping of the text-conditioning to improve classifier-free guidance sampling"; `sd-v1-4`: 225k such steps from v1-2.
  - v1-5 (RunwayML; card): "fine-tuned on 595k steps at resolution 512x512 on "laion-aesthetics v2 5+" and 10% dropping of the text-conditioning".
  - Compute (v1-4/v1-5 cards, via helper file S22) [DATED]: "Hardware: 32 x 8 x A100 GPUs"; "Batch: 32 x 8 x 2 x 4 = 2048"; "Hours used: 150000" (A100 PCIe 40GB, AWS US-east); "11250 kg CO2 eq."
- **Releases** (helper file, primary sources there; all [DATED]):

| Model | Date | Size | Native res. | Licence |
|---|---|---|---|---|
| SD v1.4 (CompVis/Stability) | 22 Aug 2022 | U-Net 860M + CLIP-L 123M + VAE 84M | 512² | CreativeML OpenRAIL-M |
| SD v1.5 (RunwayML) | Oct 2022 (earliest archive 20 Oct 2022) | same | 512² | CreativeML OpenRAIL-M |
| SD 2.0 | 24 Nov 2022 | U-Net 865M + OpenCLIP ViT-H | 512² and 768² (v-prediction) | CreativeML Open RAIL++-M |
| SD 2.1 | 7 Dec 2022 | same | 768² / 512² | Open RAIL++-M |
| SDXL 0.9 | 22 Jun 2023 | "3.5B parameter base model and a 6.6B parameter model ensemble pipeline" | 1024² | SDXL 0.9 Research License |
| SDXL 1.0 base + refiner | 26 Jul 2023 | U-Net 2.6B; text encoders 817M | 1024² | CreativeML Open RAIL++-M |
| SDXL Turbo | 28 Nov 2023 | SDXL U-Net (ADD) | 512² | non-commercial research → Stability licence |
| Stable Cascade | 12 Feb 2024 | Würstchen stages (Stage C 1B/3.6B) | 1024² ("compression factor of 42") | non-commercial |
| SD3 announcement | 22 Feb 2024 | "800M to 8B parameters" | — | waitlist |
| SD3 Medium | 12 Jun 2024 | 2B | ~1 MP | non-commercial research → Stability AI Community License (5 Jul 2024) |
| SD 3.5 Large / Large Turbo | 22 Oct 2024 | 8.1B (HF 8,146,280,768) | 1 MP | Stability AI Community License |
| SD 3.5 Medium | 29 Oct 2024 | 2.5B (HF 2,469,663,936), MMDiT-X | 0.25–2 MP | Community License |
- Stability AI Community License (5 Jul 2024): "free for research, non-commercial, and commercial use. You only need a paid Enterprise license if your yearly revenues exceed USD$1M". No new Stability base text-to-image model appears in its Hugging Face orgs through 25 Oct 2026 (helper). As of 25 Oct 2026 the official SD 2.x Hugging Face repos return 401 and the SD 1.5 weights live at the `stable-diffusion-v1-5/stable-diffusion-v1-5` mirror ("a mirror of the now deprecated `ruwnayml/stable-diffusion-v1-5`") [DATED].
- **SDXL** (Podell, English, Lacey, Blattmann, Dockhorn, Müller, Penna & Rombach, arXiv:2307.01952, 4 Jul 2023): "SDXL leverages a three times larger UNet backbone"; "These changes result in a model size of 2.6B parameters in the UNet, see Tab. 1. The text encoders have a total size of 817M parameters." **Micro-conditioning** on original image size and crop coordinates: without size conditioning, discarding images below 256 px "would lead to a significant 39% of discarded data" (§2.2). Multi-aspect-ratio training. **Refiner:** "We generate initial latents of size 128 × 128 using SDXL. Afterwards, we utilize a specialized high-resolution refinement model and apply SDEdit [28] on the latents generated in the first step, using the same prompt. SDXL and the refinement model use the same autoencoder." User study win rates (§2.5): "SDXL w/ refinement: 48.44%, SDXL base: 36.93%, Stable Diffusion 1.5: 7.91%, Stable Diffusion 2.1: 6.71%". Safetensors: base U-Net 2,567,463,684; refiner U-Net 2,259,526,660; pipeline total (U-Net + 2 text encoders + VAE) 3,468,837,867 = the "3.5B parameter base model".
- **SD3's autoencoder:** f = 8, **16 channels** (§11); FLUX.1 also 16 channels (`in_channels` 64 = 16 × 2 × 2).

### Worked example: 512×512 → 64×64×4 and back [DERIVED]
- Pixels: 512 × 512 × 3 = **786,432** values. Latent: 64 × 64 × 4 = **16,384** values. Ratio **48×**.
- Positions the U-Net touches per channel: 4,096 (64²) instead of 262,144 (512²): **64×** fewer. Self-attention cost scales with positions², so one attention map at full latent resolution is 4,096² ≈ 1.7 × 10⁷ entries vs 262,144² ≈ 6.9 × 10¹⁰ in pixel space (4,096× more).
- SDXL 1024×1024: 3,145,728 pixels → 128×128×4 = 65,536 latent values (48×). SD3/FLUX 1024²: 128×128×16 = 262,144 values (12×) → 4,096 patch tokens (2×2).
- One generation decodes **once**: the VAE decoder (49.5M parameters) runs on the final latent only; the 860M U-Net runs at every step.
- Latent scaling: a latent with std ≈ 5.49 times 0.18215 ≈ 1.0, matching the unit-variance noise the schedule assumes.

### Common misconceptions
- "Stable Diffusion diffuses pixels." It diffuses a 64×64×4 latent; what you see is the decoder's rendering of it. Decoding intermediate latents shows the "noise turning into a picture" frames.
- "The 4 latent channels are RGB+alpha." They are learned features; a fixed linear map gives a rough RGB preview, but they are not colour channels.
- "Latent diffusion loses detail." Reconstruction quality bounds final quality (f = 8, c = 4: R-FID 0.90, PSNR 24.2 on OpenImages); SD3 found that "increasing the number of latent channels d significantly boosts reconstruction performance" and moved to 16 channels (FLUX too).
- "SD was trained on all of LAION-5B." It was trained on subsets: laion2B-en, laion-high-resolution (170M), laion-aesthetics v2 5+.
- "The VAE is a VAE in the classic blurry sense." Its KL weight is ~10⁻⁶ and it is trained with perceptual and adversarial losses; it behaves like a near-deterministic compressor.

### Quotable
- "Most bits of a digital image correspond to imperceptible details." (Rombach et al. 2022, Fig. 2)
- "With its 860M UNet and 123M text encoder, the model is relatively lightweight and runs on a GPU with at least 10GB VRAM." (CompVis Stable Diffusion README)
- "By introducing cross-attention layers into the model architecture, we turn diffusion models into powerful and flexible generators for general conditioning inputs such as text or bounding boxes" (Rombach et al. 2022, abstract)

---

## §18 Diffusion transformers: DiT, U-ViT, PixArt-α, MM-DiT, FLUX

### Mechanism (plain words)
- **Patches as tokens.** Cut the noisy latent into p×p patches (DiT: p = 2, 4 or 8), flatten each into a vector, add a position embedding: a sequence of tokens, like words. A standard transformer (self-attention + MLP blocks) processes the sequence; a final linear layer turns each token back into a patch of noise prediction (and variance).
- **adaLN-Zero.** The timestep and class embedding set the scale and shift of each block's layer norms (adaptive LayerNorm), plus a gating scale on each residual branch initialised to **zero**, so every block starts as the identity. This was the best of DiT's four conditioning designs (vs in-context tokens, cross-attention, plain adaLN).
- **Compute predicts quality.** DiT measured model complexity in Gflops per forward pass and found FID falls steadily as Gflops rise, whether by a bigger model or smaller patches (more tokens). DiT-XL/2 reached FID **2.27** on ImageNet 256 with guidance.
- **U-ViT** keeps U-Net-style *long skip connections* between shallow and deep transformer blocks and treats time and condition as tokens. **PixArt-α** showed a DiT text-to-image model could be trained for about a tenth of SD 1.5's compute.
- **MM-DiT** (SD3): text tokens and image tokens get *separate* weights (two streams) but attend **jointly** in one attention operation, so information flows both ways. **FLUX.1**: 19 such double-stream blocks then 38 single-stream blocks (text and image concatenated, shared weights), 12B parameters, rotary position embeddings.
- **Why transformers won at scale:** predictable scaling, mature tooling, easy mixing of modalities (text, image, video, audio tokens), and patch-based handling of any resolution or aspect ratio. Video models (Sora, Movie Gen, Wan, HunyuanVideo) are all DiT-style (§21).

### Key formulas
- **Token count** (DiT §3.2): an I×I×C latent with patch size p gives T = (I/p)² tokens; "halving p will quadruple T, and thus at least quadruple total transformer Gflops."
- **adaLN-Zero block** (DiT §3.2, Fig. 3): (γ₁, β₁, α₁, γ₂, β₂, α₂) = MLP(t_emb + c_emb); h ← h + α₁·Attn(γ₁·LN(h) + β₁); h ← h + α₂·MLP(γ₂·LN(h) + β₂); the MLP producing α is initialised so α = 0 ("this initializes the full DiT block as the identity function").
- **MM-DiT joint attention** (SD3 §4): Q = [Q_txt; Q_img], K = [K_txt; K_img], V = [V_txt; V_img] computed with separate weights per modality; one softmax attention over the concatenated sequence; separate MLPs and modulation per modality.

### Numbers with sources
- **DiT** (Peebles & Xie, "Scalable Diffusion Models with Transformers", arXiv:2212.09748, 19 Dec 2022; ICCV 2023), abstract: "We find that DiTs with higher Gflops—through increased transformer depth/width or increased number of input tokens—consistently have lower FID. In addition to possessing good scalability properties, our largest DiT-XL/2 models outperform all prior diffusion models on the class-conditional ImageNet 512×512 and 256×256 benchmarks, achieving a state-of-the-art FID of 2.27 on the latter."
- **Configs (Table 1; layers / hidden / heads / Gflops at I = 32, p = 4):** DiT-S 12 / 384 / 6 / 1.4; DiT-B 12 / 768 / 12 / 5.6; DiT-L 24 / 1024 / 16 / 19.7; DiT-XL 28 / 1152 / 16 / 29.1.
- **Table 4 (ImageNet 256, 400K steps, batch 256, no guidance; Gflops / params / FID-50K):** S/8 0.36 / 33M / 153.60; S/2 6.06 / 33M / 68.40; B/2 23.01 / 130M / 43.47; L/2 80.71 / 458M / 23.33; XL/8 7.39 / 676M / 106.41; XL/4 29.05 / 675M / 43.01; **XL/2 118.64 / 675M / 19.47**. Block designs for XL/2 at 400K: in-context 35.24, cross-attention 26.14 (137.62 Gflops), adaLN 25.21, **adaLN-Zero 19.47**. Longer training: XL/2 2,352K steps 10.67; 7,000K steps **9.62**. 512×512 XL/2 (524.60 Gflops): 13.78 (1,301K), 11.93 (3,000K).
- **Table 2 (ImageNet 256, with CFG):** "DiT-XL/2-G (cfg=1.50) 2.27" FID, sFID 4.60, IS 278.24, precision 0.83, recall 0.57; vs LDM-4-G 3.60, ADM-G-U 3.94. ImageNet 512: DiT-XL/2-G 3.04 (ADM previous best 3.85).
- **Guidance on 3 of 4 channels** (App.): "we applied guidance only to the first three channels of the latents instead of all four channels"; "three-channel guidance with a scale of 1.5 gives an FID-50K of 2.27, and four-channel guidance with a scale of 1.375 gives an FID-50K of 2.20".
- Training (§4): "We use a constant learning rate of 1 × 10−4, no weight decay and a batch size of 256"; EMA 0.9999; SD's off-the-shelf VAE; "DiT-XL/2, our most compute-intensive model, trains at roughly 5.7 iterations/second on a TPU v3-256 pod with a global batch size of 256." FID with "250 DDPM sampling steps" and ADM's evaluation suite.
- **U-ViT** (Bao, Nie, Xue, Cao, Li, Su & Zhu, "All are Worth Words: A ViT Backbone for Diffusion Models", arXiv:2209.12152, Sep 2022; CVPR 2023), abstract: "treating all inputs including the time, condition and noisy image patches as tokens and employing long skip connections between shallow and deep layers"; "record-breaking FID scores of 2.29 in class-conditional image generation on ImageNet 256×256, and 5.48 in text-to-image generation on MS-COCO"; "the long skip connection is crucial while the downsampling and up-sampling operators in CNN-based U-Net are not always necessary."
- **PixArt-α** (Chen et al., arXiv:2310.00426, Sep 2023; ICLR 2024): 0.6B parameters. arXiv **v3** abstract: "PixArt-α only takes 12% of Stable Diffusion v1.5's training time (∼753 vs. ∼6,250 A100 GPU days), saving nearly $300,000 ($28,400 vs. $320,000) and reducing 90% CO2 emissions." arXiv **v1** said "10.8% … (675 vs. 6,250 A100 GPU days) … ($26,000 vs. …)". Use v3 (Appendix A).
- **MM-DiT / SD3:** depth d → hidden 64·d; largest d = 38 (8B); 2×2 patches of a 16-channel latent; separate weights for text and image; QK-normalisation (RMSNorm on Q and K) to stabilise large runs (§5.3; Fig. 2 caption: "The RMS-Norm for Q and K can be added to stabilize training runs").
- **FLUX.1:** "hybrid architecture of multimodal and parallel diffusion transformer blocks and scaled to 12B parameters" (BFL); 19 double-stream + 38 single-stream blocks, hidden 3,072, 24 heads, RoPE (diffusers config). Parameters: 11.89B (schnell), 11.90B (dev).
- **Later DiT-era records [DATED]:** REPA (Yu et al., arXiv:2410.06940): "FID=1.42 using classifier-free guidance with the guidance interval" (ImageNet 256), by aligning DiT hidden states with a pretrained visual encoder. EDM2 (Karras et al., arXiv:2312.02696, a U-Net): ImageNet-512 "previous record FID of 2.41 … to 1.81". VAR (Tian et al., arXiv:2404.02905, an autoregressive "next-scale" model): ImageNet 256 FID 1.73: "for the first time, makes GPT-like AR models surpass diffusion transformers in image generation".

### Worked example [DERIVED]
- **DiT tokens at 256×256:** SD VAE → 32×32×4 latent. p = 8 → (32/8)² = **16** tokens of 8·8·4 = 256 values; p = 4 → **64** tokens of 64 values; p = 2 → **256** tokens of 16 values. DiT-XL/8 → XL/2: 7.39 → 118.64 Gflops (16.1×, tracking the 16× increase in tokens), FID 106.41 → 19.47 with the same 675M parameters.
- **DiT-XL/2 training time:** 7,000K steps ÷ 5.7 steps/s = 1.23 × 10⁶ s ≈ **14.2 days** on a TPU v3-256 pod.
- **FLUX at 1024×1024:** 4,096 image tokens + 512 T5 tokens = 4,608 tokens in the joint attention; one attention map per head is 4,608² ≈ 2.1 × 10⁷ entries, ×24 heads × 57 blocks per step.
- **The digit model:** the course trains a tiny DiT on 28×28 digits for comparison (e.g. 28×28 → p = 4 → 49 tokens); its measured numbers live in `research/experiments/`.

### Common misconceptions
- "Transformers replaced the VAE." DiT, SD3 and FLUX all still diffuse VAE latents; only the denoiser changed.
- "Patch size doesn't change parameters." Correct for parameters ("changing p has no meaningful impact on downstream parameter counts"), but it changes compute and quality a lot.
- "MM-DiT is cross-attention." Cross-attention lets image queries read text; MM-DiT lets text and image tokens attend to each other in one joint attention (text tokens are updated too).
- "DiT's 2.27 is unguided." It uses CFG 1.5 (and on 3 of 4 latent channels); unguided DiT-XL/2 is 9.62.

### Quotable
- "We find that DiTs with higher Gflops—through increased transformer depth/width or increased number of input tokens—consistently have lower FID." (Peebles & Xie 2022, abstract)
- "the long skip connection is crucial while the downsampling and up-sampling operators in CNN-based U-Net are not always necessary." (Bao et al. 2022, abstract)

---

## §19 Evaluation: FID, IS, precision/recall, CLIP score, CMMD, human preference

### Mechanism (plain words)
- **There is no single "accuracy" for pictures.** We want samples that look real (fidelity), cover the variety of real data (diversity/recall), and follow the prompt (alignment). Each metric sees only part of that.
- **FID** (Heusel et al. 2017): run real and generated images through Inception-v3, take the 2,048-d activations of the last pooling layer, fit a Gaussian (mean and covariance) to each set, and compute the Fréchet (Wasserstein-2) distance between the two Gaussians. Lower is better. Sensitive to both quality and diversity, but tied to ImageNet-trained features, biased by sample size, and sensitive to resizing/compression details.
- **IS** (Salimans et al. 2016): high when each image is confidently classified (sharp) and the class mix is varied; ignores the real data entirely.
- **Precision/recall** (Kynkäänniemi et al. 2019): put a k-NN ball around every real (or generated) feature vector; precision = fraction of generated samples inside the real manifold estimate, recall = fraction of real samples inside the generated one. Separates fidelity from coverage (ADM-G: precision up, recall down).
- **CLIP score:** cosine similarity between CLIP embeddings of prompt and image, a proxy for prompt alignment. **CMMD** (2024): replace Inception features with CLIP and the Gaussian assumption with an MMD kernel distance.
- **Human preference:** pairwise human votes (PartiPrompts, DrawBench) and learned preference models trained on such votes (PickScore, ImageReward, HPS v2). Increasingly the headline evaluation for text-to-image (SDXL, SD3, ADD all lead with user studies).

### Key formulas
- **FID** (Heusel et al. eq. 6): d²((m, C), (m_w, C_w)) = ‖m − m_w‖²₂ + Tr(C + C_w − 2(C·C_w)^{1/2}).
- **IS** (Salimans et al. 2016): IS = exp(E_x KL(p(y|x) ‖ p(y))).
- **Precision** = (1/|Φ_g|) Σ_{φ_g} f(φ_g, Φ_r), f = 1 if φ_g lies within the k-th-nearest-neighbour radius of some real φ_r; recall swaps roles. k = 3.
- **CLIPScore** (Hessel et al. 2021): CLIP-S(c, v) = w·max(cos(c, v), 0), w = 2.5.
- **CMMD** (Jayasumana et al. 2024): MMD² between CLIP ViT-L/14@336px embeddings with a Gaussian RBF kernel, σ = 10, reported ×1000.

### Numbers with sources
- **FID** (Heusel, Ramsauer, Unterthiner, Nessler & Hochreiter, "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium", arXiv:1706.08500, Jun 2017; NeurIPS 2017): "we use the last pooling layer as coding layer"; "To approximate these moments for the model distribution, we generate 50,000 images, propagate them through the Inception-v3 model, and then compute the mean m and the covariance matrix C." The pool3 layer of Inception-v3 is 2,048-d [standard; the paper does not print "2048" in the passage read].
- **Bias** (Chong & Forsyth, "Effectively Unbiased FID and Inception Score and where to find them", arXiv:1911.07023, CVPR 2020), abstract: FID and IS "are biased – the expected value of the score computed for a finite sample set is not the true value of the score. Worse, the paper shows that the bias term depends on the particular model being evaluated, so model A may get a better score than model B simply because model A's bias term is smaller." "Evaluation procedures with FID vary: some authors use FID50k … while others use FID10k".
- **Implementation sensitivity:** Parmar, Zhang & Zhu, "On Aliased Resizing and Surprising Subtleties in GAN Evaluation" (clean-fid, arXiv:2104.11222, CVPR 2022): resizing and JPEG choices change FID. DDPM computed FID "with respect to the training set" (3.17) vs 5.24 against the test set. DiT: "FID is known to be sensitive to small implementation details"; they used ADM's TensorFlow suite.
- **ImageNet leakage:** Kynkäänniemi, Karras, Aittala, Aila & Lehtinen, "The Role of ImageNet Classes in Fréchet Inception Distance" (arXiv:2203.06026, ICLR 2023): FID can be improved by matching ImageNet class statistics without improving quality.
- **Precision/recall** (Kynkäänniemi, Karras, Laine, Lehtinen & Aila, arXiv:1904.06991, NeurIPS 2019): features from "a pre-trained VGG-16"; "k = 3 is a robust choice"; "standard practice to quote FIDs with 50k samples, we adopt the same design point". (ADM and later papers compute P/R with Inception features.)
- **IS** (Salimans, Goodfellow, Zaremba, Cheung, Radford & Chen, "Improved Techniques for Training GANs", arXiv:1606.03498, Jun 2016). CIFAR-10 real-data IS ≈ 11.24 **[UNVERIFIED: commonly cited; not checked in this PDF]**.
- **CLIPScore** (Hessel, Holtzman, Forbes, Le Bras & Choi, arXiv:2104.08718, EMNLP 2021): "with textual CLIP embedding c, we set w = 2.5"; ViT-B/32.
- **CMMD** (Jayasumana, Ramalingam, Veit, Glasner, Chakrabarti & Kumar, "Rethinking FID: Towards a Better Evaluation Metric for Image Generation", arXiv:2401.09603, Jan 2024; CVPR 2024), abstract: "We highlight important drawbacks of FID: Inception's poor representation of the rich and varied content generated by modern text-to-image models, incorrect normality assumptions, and poor sample complexity." "We empirically demonstrate that FID contradicts human raters, it does not reflect gradual improvement of iterative text-to-image models, it does not capture distortion levels, and that it produces inconsistent results when varying the sample size." CMMD "is an unbiased estimator that does not make any assumptions on the probability distribution of the embeddings and is sample efficient."
- **Preference models:**
  - **PickScore / Pick-a-Pic** (Kirstain, Polyak, Singer, Matiana, Penna & Levy, arXiv:2305.01569, May 2023; NeurIPS 2023): "the Pick-a-Pic dataset contains over 500,000 examples and 35,000 distinct prompts" (training set version: 583,747 examples, 37,523 prompts, 4,375 users); PickScore "achieves superhuman performance in the task of predicting user preferences (a 70.5% accuracy rate, compared to humans’ 68.0%), while zero-shot CLIP-H (60.8%) and the popular aesthetics predictor [14] (56.8%) perform closer to chance". A later Pick-a-Pic release has "more than one million examples".
  - **ImageReward** (Xu et al., arXiv:2304.05977, Apr 2023; NeurIPS 2023): "collects 137k expert comparisons to date".
  - **HPS v2** (Wu et al., arXiv:2306.09341, Jun 2023): "HPD v2 comprises 798,090 human preference choices on 433,760 pairs of images, making it the largest dataset of its kind."
- **Text-to-image FID convention:** zero-shot MS-COCO FID-30K (Imagen 7.27, DALL·E 2 10.39, GLIDE 12.24, LDM 12.63). SDXL §2.5: "when using classical performance metrics such as FID and CLIP scores the improvements of SDXL over previous methods are not reflected as shown in Fig. 12"; they lead with human preference instead.
- **Benchmarks for prompt following:** DrawBench (Imagen §4.2: "DrawBench comprises 200 prompts in total" across "11 categories"), PartiPrompts (P2, used by SDXL, Diffusion-DPO), GenEval (Ghosh et al. 2023; used by SD3).

### Worked example [DERIVED; `derive.py` part O]
**FID between two 2D Gaussians** (the "two clouds" animation): real N((0, 0), I) vs generated N((1, 0), diag(2, 0.5)): ‖Δm‖² = 1; Tr(C₁ + C₂) = 2 + 2.5 = 4.5; (C₁C₂)^{1/2} = diag(√2, √0.5), trace 1.414 + 0.707 = 2.121; FID = 1 + 4.5 − 2·2.121 = **1.257**. Shifting the mean costs; mismatched spread costs even with matching means.
**Sample-size effect:** with N samples the covariance estimate is noisy and FID is biased upward roughly ∝ 1/N (Chong & Forsyth); compare models only at the same N, ideally 50K, and with the same reference set and preprocessing. The digit model's "FID-like score" in the course uses a small classifier's features, so its absolute values are not comparable with published Inception FIDs.

### Common misconceptions
- "FID measures realism." It measures distance between feature statistics; it rewards diversity, can be gamed via ImageNet-class matching, and disagrees with humans on modern text-to-image models (CMMD paper).
- "FID numbers are comparable across papers." Only with the same N, reference set (train vs test: DDPM 3.17 vs 5.24), resizing and implementation.
- "Higher CLIP score = better images." It measures alignment with the caption via CLIP, which has its own blind spots (counting, spatial relations).
- "IS is a quality metric for any dataset." It uses an ImageNet classifier and never looks at real data; it is meaningful mainly for ImageNet-like data.
- "Human preference is ground truth." It depends on raters, prompts and the question asked (SDXL vs Midjourney, Diffusion-DPO's three questions).

### Quotable
- "This means all comparisons using FID or IS as currently computed are unreliable." (Chong & Forsyth 2020, abstract)
- "We call for a reevaluation of FID’s use as the primary quality metric for generated images." (Jayasumana et al. 2024, abstract)

---

## §20 Data and alignment: web-scale data, recaptioning, aesthetic filtering, memorization, preference tuning

### Mechanism (plain words)
- **The data is the model's world.** Text-to-image models learn from billions of image–alt-text pairs scraped from the web. LAION-5B (2022) made such data openly available: Common Crawl image links whose alt text matched the image according to CLIP.
- **Filtering shapes style.** Subsets chosen by a learned *aesthetic score* (e.g. "laion-aesthetics v2 5+"), resolution and watermark probability push models toward polished images; SD v1.2–1.5 were trained on such subsets.
- **Captions matter as much as images.** Alt text is short, noisy and often unrelated. DALL·E 3 trained a captioner to write long descriptive captions and trained on 95% synthetic captions: prompt following improved substantially. Sora reused the idea for video.
- **Memorization.** Models can reproduce training images, mostly ones duplicated many times in the data. Carlini et al. extracted about a hundred near-copies from Stable Diffusion out of 175 million generations, and over a thousand from small CIFAR-10 models. Deduplication reduces it.
- **Preference tuning.** After pretraining, models can be pushed toward what people prefer: **DDPO** treats the denoising chain as a multi-step decision process and applies policy gradients with a reward (aesthetics, compressibility, VLM-judged alignment); **Diffusion-DPO** adapts Direct Preference Optimization to diffusion's likelihood bound and trains on pairwise human preferences (Pick-a-Pic). SD3 was DPO-aligned after pretraining.
- **Dataset governance [DATED]:** LAION-5B was taken offline in December 2023 after a Stanford Internet Observatory report found links to illegal material, and re-released in August 2024 as Re-LAION-5B with those links removed. (Scope note: this course treats the event as a dataset-maintenance fact; misuse topics are out of scope.)

### Key formulas
- **LAION filtering** (Schuhmann et al. 2022, §3.1): keep a pair if cos(CLIP-ViT-B/32(image), CLIP(text)) ≥ 0.28 (English) or ≥ 0.26 (other languages).
- **Diffusion-DPO** (Wallace et al. 2023): DPO's loss −log σ(β[log π_θ(y_w)/π_ref(y_w) − log π_θ(y_l)/π_ref(y_l)]) with the intractable log-likelihoods replaced by the diffusion ELBO; per noise level this becomes a comparison of denoising errors: L = −E log σ(−βT·ω(λ_t)·(‖ε_w − ε_θ(x_t^w, t)‖² − ‖ε_w − ε_ref(x_t^w, t)‖² − (‖ε_l − ε_θ(x_t^l, t)‖² − ‖ε_l − ε_ref(x_t^l, t)‖²))) [form as in the paper's eq. 14; β = 2000 (SD 1.5) or 5000 (SDXL)].
- **DDPO** (Black et al. 2023): maximise J(θ) = E_{c, x₀~p_θ(·|c)}[r(x₀, c)] treating each denoising step p_θ(x_{t−1}|x_t, c) as a policy action; gradient E[Σ_t ∇_θ log p_θ(x_{t−1}|x_t, c)·r(x₀, c)] (REINFORCE) or with PPO-style importance ratios and clipping.

### Numbers with sources
- **LAION-5B** (Schuhmann, Beaumont, Vencu, Gordon, Wightman, Cherti, Coombes, Katta, Mullis, Wortsman, Schramowski, Kundurthy, Crowson, Schmidt, Kaczmarczyk & Jitsev, "LAION-5B: An open large-scale dataset for training next generation image-text models", arXiv:2210.08402, Oct 2022; NeurIPS 2022 Datasets & Benchmarks), abstract: "we present LAION-5B - a dataset consisting of 5.85 billion CLIP-filtered image-text pairs, of which 2.32B contain English language." Composition: 2.32B English (LAION-2B-en), 2.26B multilingual (100+ languages), 1.27B "not specific to a particular language". Filtering: "We removed all English image-text pairs with cosine similarity below 0.28, and all other pairs with similarity below 0.26. This step removed around 90% of the original 50 billion images, leaving just short of 6 billion examples." Subsets named in the paper: "a 170M subset for superresolution models, and LAION-Aesthetic, a 120M subset of aesthetic images".
- **Re-LAION-5B** (LAION blog, 30 Aug 2024, https://laion.ai/blog/relaion-5b/) [DATED]: LAION-5B was taken down on **19 December 2023**: "The Stanford report prompted an immediate takedown of LAION-5B and we initiated a comprehensive safety revision." Re-LAION-5B: **5,526,641,167** pairs; 2,236 links removed that matched partner hash lists (including the 1,008 from the Stanford report); released as "Re-LAION-5B research" and "Re-LAION-5B research-safe" (the latter also removing NSFW-tagged samples, 3.044% of the original).
- **DALL·E 3 recaptioning** (Betker, Goh, Jing, Brooks, Wang, Li, Ouyang, Zhuang, Lee, Guo, Manassra, Dhariwal, Chu, Jiao & Ramesh (OpenAI), "Improving Image Generation with Better Captions", 2023, https://cdn.openai.com/papers/dall-e-3.pdf), abstract: "We show that prompt following abilities of text-to-image models can be substantially improved by training on highly descriptive generated image captions." "We hypothesize that this issue stems from noisy and inaccurate image captions in the training dataset." Two caption styles: "short synthetic captions" (SSC) and "descriptive synthetic captions" (DSC). Blending ablation: "We chose blends of 65%, 80%, 90% and 95% synthetic captions"; the 65% blend "was far behind". Final model: "we use a mixture of 95% synthetic captions and 5% ground truth captions." Evaluation models: T5-XXL-conditioned latent diffusion, "trained to 500,000 training steps". Scope caveat: "It does not cover training or implementation details of the DALL-E 3 model." DALL·E 3's own decoder: "we trained our own diffusion decoder on top of the latent space learned by the VAE … we used the consistency distillation process … to bring it down to two denoising steps" (App. B, via helper). At inference, GPT-4 "upsamples" short user prompts into descriptive ones (§ on caption upsampling).
- **Memorization** (Carlini, Hayes, Nasr, Jagielski, Sehwag, Tramèr, Balle, Ippolito & Wallace, "Extracting Training Data from Diffusion Models", arXiv:2301.13188, 30 Jan 2023; USENIX Security 2023), abstract: "we show that diffusion models memorize individual images from their training data and emit them at generation time. With a generate-and-filter pipeline, we extract over a thousand training examples from state-of-the-art models". Stable Diffusion: "select the 350,000 most-duplicated examples from the training dataset and generate 500 candidate images for each of these prompts (totaling 175 million generated images)"; "We find 94 images are (ℓ2, 0.15)-extracted"; manual review adds 13 "(for a total of 109 images)". "out of 175 million generated images, we can identify 50 memorized images with 0 false positives". Duplication: extraction mostly for images "duplicated at least k = 100 times". Imagen: of the 500 most out-of-distribution examples, it "memorized and regurgitated 3 of these images (which were unique in the training dataset)"; none found for SD among 10,000 outliers. CIFAR-10 diffusion models: "1,280 unique extracted images" (after 1M generations). Conclusion: "diffusion models are much less private than prior generative models such as GANs".
- **Diffusion-DPO** (Wallace, Dang, Rafailov, Zhou, Lou, Purushwalkam, Ermon, Xiong, Joty & Naik, "Diffusion Model Alignment Using Direct Preference Optimization", arXiv:2311.12908, Nov 2023; CVPR 2024): "Using the Pick-a-Pic dataset of 851K crowdsourced pairwise preferences, we fine-tune the base model of the state-of-the-art Stable Diffusion XL (SDXL)-1.0 model". "Human evaluators prefer DPO-tuned SDXL images over the SDXL(base + refinement) model 69% of the time on the PartiPrompts dataset". Against SDXL-base on PartiPrompts: "DPO-SDXL is preferred 70.0% of the time for General Preference"; HPS benchmark win rate 64.7%. β = 2000 (SD 1.5), 5000 (SDXL).
- **DDPO** (Black, Janner, Du, Kostrikov & Levine, "Training Diffusion Models with Reinforcement Learning", arXiv:2305.13301, May 2023; ICLR 2024), abstract: adapts diffusion models "to objectives that are difficult to express via prompting, such as image compressibility, and those derived from human feedback, such as aesthetic quality. Finally, we show that DDPO can improve prompt-image alignment using feedback from a vision-language model without the need for additional data collection or human annotation."
- **SD3 used DPO** after pretraining (§5.3.2: "we align the model using Direct Preference Optimization (DPO)"). SD3 Medium card [DATED]: "The model was pre-trained on 1 billion images. The fine-tuning data includes 30M high-quality aesthetic images … as well as 3M preference data images."
- **SD 1.x data** (CompVis README): laion2B-en, laion-high-resolution (170M ≥ 1024²), laion-aesthetics v2 5+ (aesthetics > 5.0, size ≥ 512², watermark probability < 0.5).

### Worked example [DERIVED]
- **LAION scale:** 5.85B pairs vs MNIST's 70,000 images: **8.4 × 10⁴×** more examples. If SD v1's 2048-image batches ran ~1.2M steps in total (237k + 194k + 515k + 225k ≈ 1.17M for v1-4) that is ≈ 2.4 × 10⁹ image presentations, fewer than one pass over LAION-5B [arithmetic on the README step counts; batch 2048 per model card].
- **Caption blend:** 95% synthetic / 5% original means a training image is shown with its original alt text only 1 time in 20.
- **Memorization base rate (SD):** 109 near-copies / 175,000,000 generations ≈ 6 × 10⁻⁷ per generation, *for prompts chosen to be the most-duplicated captions*; the rate for typical prompts is lower (no outlier memorization found for SD).
- **The digit model:** the course shows each generated digit next to its nearest training neighbours (pixel L2), the same check Carlini et al. use (ℓ2 threshold 0.15 on a [0, 1] scale).

### Common misconceptions
- "LAION is a curated dataset of images." It is a list of URLs and alt texts filtered by CLIP similarity; images are fetched by the user, and some links rot.
- "Memorization means the model stores all its training images." Extraction succeeded mostly for heavily duplicated images; an 860M-parameter model cannot store billions of images, but it can store some.
- "RLHF for images needs a reward model." Diffusion-DPO trains directly on preference pairs, without one; DDPO needs a reward function (which can be a learned model or a simple program like JPEG size).
- "Synthetic captions are a data-augmentation trick." DALL·E 3 changed what the model is *trained to read*; prompts at inference are also expanded by GPT-4 to match.

### Quotable
- "We show that prompt following abilities of text-to-image models can be substantially improved by training on highly descriptive generated image captions." (Betker et al. 2023, abstract)
- "we present LAION-5B - a dataset consisting of 5.85 billion CLIP-filtered image-text pairs, of which 2.32B contain English language." (Schuhmann et al. 2022, abstract)
- "we show that diffusion models memorize individual images from their training data and emit them at generation time." (Carlini et al. 2023, abstract)

---
## §21 Video diffusion: from 16 frames at 64×64 to minutes of 1080p

### Mechanism (plain words)
- **A video is a stack of frames, diffused together.** Instead of one H×W×3 image the model denoises a T×H×W×3 block (or its latent). Denoising all frames *jointly* is what makes them agree with each other (§22).
- **First generation (2022): factorised 3D U-Nets in pixel space.** Video Diffusion Models (Ho et al.) turned 3×3 convolutions into 1×3×3 (space-only) and added temporal attention across frames; 16 frames at 64×64. Imagen Video and Make-A-Video built **cascades**: a small base video, then temporal super-resolution (more frames) and spatial super-resolution (more pixels).
- **Second generation (2023): latent video models from image models.** Video LDM ("Align your Latents") and Stable Video Diffusion inserted temporal layers into a pretrained Stable Diffusion U-Net and trained only those (or fine-tuned everything) on curated video data.
- **Third generation (2024–): latent video transformers.** A **video VAE** compresses time as well as space (e.g. 4× in time, 8×8 in space, 16 channels), the latent is cut into **spacetime patches**, and a DiT with **full 3D attention** denoises the whole token sequence. Sora (Feb 2024) described this; Movie Gen, HunyuanVideo, Wan, CogVideoX, Mochi, LTX-Video and Open-Sora published it with numbers. Movie Gen (and Open-Sora from v1.2) state flow-matching / rectified-flow training.
- **What grew:** 16 frames → 129–256 frames per pass; 64×64 → 720p–1080p (4K via upsampling); a few seconds → 5–30 s per pass, longer by extension; and since 2025 synchronised **audio** in the same model (Veo 3, Movie Gen Audio, Sora 2, LTX-2).
- **Cost:** tokens grow with frames × height × width; full attention grows with tokens². A 5 s 720p clip is ~75,600–118,800 tokens (vs 4,096 for a 1024² image), which is why video models need aggressive VAEs, many GPUs, and minutes per clip.

### Key formulas
- **Token count:** tokens = T_lat × (H/(s·p)) × (W/(s·p)), with T_lat = 1 + (T − 1)/c_t for causal VAEs (first frame kept separately), s = spatial VAE stride, p = patch size, c_t = temporal stride.
- **Factorised (VDM) attention:** spatial attention within each frame (frames as batch), then temporal attention across frames at each spatial location (positions as batch): cost ∝ T·(HW)² + HW·T² instead of (T·HW)² for full attention.
- **Reconstruction guidance** (VDM eq. 7) for extension: x̃_θ^b(z_t) = x̂_θ^b(z_t) − (w_r·α_t/2)·∇_{z_t^b}‖x^a − x̂_θ^a(z_t)‖² (frames b generated consistent with given frames a).

### Numbers with sources (helper file `helper_video.md` has full quotes and page references)
- **VDM** (Ho, Salimans, Gritsenko, Chan, Norouzi & Fleet, "Video Diffusion Models", arXiv:2204.03458, Apr 2022): "we change each 3x3 convolution into a 1x3x3 convolution"; "after each spatial attention block, we insert a temporal attention block that performs attention over the first axis and treats the spatial axes as batch axes." Base 16×64×64. Joint image–video training "is important for sample quality" (Table 4 FVD 202.28 → 68.11 → 57.84 with 0/4/8 extra images). Text-to-video data: "a dataset of 10 million captioned videos". Reconstruction guidance for 64-frame extension: FVD 136.22 vs 451.45 for naive replacement (Table 6).
- **Imagen Video** (Ho et al., arXiv:2210.02303, 5 Oct 2022): "generates high definition 1280×768 (width × height) videos at 24 frames per second, for 128 frames (≈ 5.3 seconds)—approximately 126 million pixels." "In total, we have 1 frozen text encoder, 1 base video diffusion model, 3 SSR (spatial super-resolution), and 3 TSR (temporal super-resolution) models – for a total of 7 video diffusion models, with a total of 11.6B diffusion model parameters." Base 5.6B at 16×40×24 @ 3 fps. v-prediction "for all our models". Progressive distillation to 8 steps per stage: 618 s → 35 s per batch ("about 18× faster"). Data: "14 million video-text pairs and 60 million image-text pairs" + LAION-400M.
- **Make-A-Video** (Singer et al. (Meta), arXiv:2209.14792, 29 Sep 2022): "learn what the world looks like and how it is described from paired text-image data, and learn how the world moves from unsupervised video footage"; "it does not require paired text-video data". 16 frames at 64×64 → frame interpolation to 76 frames → 256² → 768². Pseudo-3D convolutions and attention initialised as identity.
- **Video LDM** (Blattmann, Rombach, Ling, Dockhorn, Kim, Fidler & Kreis, arXiv:2304.08818, Apr 2023): freeze SD, train temporal layers (656M); "113 frames … clips of 4.7 seconds length at 24 fps"; up to 1280×2048 with an upscaler.
- **Stable Video Diffusion** (Blattmann et al. (Stability AI), arXiv:2311.15127, Nov 2023; release 21 Nov 2023) [DATED]: data curation from LVD 577M clips (212 years) to LVD-F **152M**; U-Net "bumping its total size … to 1521M parameters" (656M temporal); 14 frames (SVD) and 25 frames (SVD-XT) at **576×1024**; conditioned on fps and motion score ("motion bucket"); "approximately 200,000 A100 80GB hours"; "SVD takes ~100s for generation, and SVD-XT takes ~180s on an A100 80GB card". UCF-101 zero-shot FVD 242.02.
- **Sora** (OpenAI technical report "Video generation models as world simulators", 15 Feb 2024; read via Wayback capture because openai.com returns 403): "we train text-conditional diffusion models jointly on videos and images of variable durations, resolutions and aspect ratios. We leverage a transformer architecture that operates on spacetime patches of video and image latent codes. Our largest model, Sora, is capable of generating a minute of high fidelity video." "Sora is a diffusion model; given input noisy patches (and conditioning information like text prompts), it's trained to predict the original "clean" patches. Importantly, Sora is a diffusion transformer." "Model and implementation details are not included in this report." Sizes: "widescreen 1920x1080p videos, vertical 1080x1920 videos and everything inbetween". Recaptioning: "We apply the re-captioning technique introduced in DALL·E 3 to videos." **The world-simulator claim, exactly:** "Our results suggest that scaling video generation models is a promising path towards building general purpose simulators of the physical world." **Its own limitation paragraph:** "Sora currently exhibits numerous limitations as a simulator. For example, it does not accurately model the physics of many basic interactions, like glass shattering. Other interactions, like eating food, do not always yield correct changes in object state."
- **Sora products [DATED]:** public "Sora Turbo" 9 Dec 2024 ("up to 1080p resolution, up to 20 sec long"); Sora 2 30 Sep 2025 ("our flagship video and audio generation model"; "synchronized dialogue and sound effects"); OpenAI deprecations page (checked 2026-10-25): "On March 24th, 2026, we notified developers using the Videos API and Sora 2 video generation model aliases and snapshots … of their deprecation and removal from the API on September 24, 2026."
- **Movie Gen** (Meta, arXiv:2410.13720, Oct 2024): "Our largest video generation model is a 30B parameter transformer trained with a maximum context length of 73K video tokens, corresponding to a generated video of 16 seconds at 16 frames-per-second." TAE compresses "8× across each of the spatio-temporal dimensions", C = 16; 2×2 spatial patches; "768 × 768 px video with 256 frames" → 73,728 tokens; 48 layers, width 6,144, LLaMa-3-style; trained with **flow matching** ("more robust to the exact choice of noise schedules and it outperforms diffusion losses"); 7B spatial upsampler to 1080p; Movie Gen Audio 13B, 48 kHz; "up to 6,144 H100 GPUs"; "O(100)M videos and O(1)B images"; 50-step linear-quadratic sampling schedule "emulating N = 250 linear steps".
- **Google Veo family [DATED]:** Veo (14 May 2024: "1080p", "beyond a minute"); Veo 2 (16 Dec 2024: "resolutions up to 4K, and extended to minutes in length"); Veo 3 (20 May 2025: "for the first time, can also generate videos with audio"; tech report: "Veo 3 uses latent diffusion, in which the diffusion process is applied jointly to the temporal audio latents, and the spatio-temporal video latents"); Veo 3.1 (preview 15 Oct 2025; GA 17 Nov 2025: 4/6/8 s clips, 720p/1080p/4K, 24 fps). Gemini API deprecation table lists all Veo models with replacement `gemini-omni-1.1-flash` (Gemini Omni Flash announced 19 May 2026; architecture not disclosed). No "Veo 4" on official pages as of 2026-10-25.
- **Open-weight models [DATED]** (details in helper file A8):

| Model | Release | Params | Native output | Latent compression (t×h×w), patch | Licence | Stated speed/memory |
|---|---|---|---|---|---|---|
| CogVideoX 2B/5B | Aug 2024 | 2B / 5B | 720×480, 6 s @ 8 fps (v1.5: 1360×768, 16 fps) | 4×8×8 | Apache 2.0 (2B) / CogVideoX License | 5B ~90 s on H100 (50 steps) |
| Mochi 1 preview (Genmo) | 22 Oct 2024 | 10B | 848×480, ≤ 5.4 s @ 30 fps | 6×8×8, 12 ch | Apache 2.0 | ~60 GB VRAM single GPU |
| LTX-Video (Lightricks) | 21 Nov 2024 | 1.9B | 768×512, 121 f @ 24 fps | 8×32×32, 128 ch | OpenRAIL-M (v0.9.5) | "5 seconds of 24 fps video at 768×512 resolution in just 2 seconds on an Nvidia H100" |
| HunyuanVideo (Tencent) | 3 Dec 2024 | "over 13 billion" | 1280×720, 129 f @ 24 fps | 4×8×8, C 16, 1×2×2 | Tencent Hunyuan Community | 60 GB peak; 1,904 s on 1 GPU (50 steps) |
| Wan 2.1 (Alibaba) | 25 Feb 2025 | 1.3B / 14B | 480p / 720p, 81 f @ 16 fps | 4×8×8, C 16, 1×2×2 | Apache 2.0 | 1.3B: 8.19 GB VRAM; 5 s 480p ≈ 4 min on RTX 4090 |
| Open-Sora 2.0 | 12 Mar 2025 | 11B | 768×768, < 129 f | — | Apache 2.0 | "trained for only $200k" |
| Wan 2.2 | 28 Jul 2025 | A14B MoE (27B total / 14B active); TI2V-5B | 720p @ 24 fps | 4×16×16 (5B) | Apache 2.0 | 5B: 5 s 720p "under 9 minutes on a single consumer-grade GPU" |
| HunyuanVideo 1.5 | 24 Nov 2025 | 8.3B | 480–720p, 5–10 s → SR 1080p | 4×16×16 | Tencent Hunyuan Community | "consumer-grade GPUs" |
| LTX-2 | announced 23 Oct 2025; paper 6 Jan 2026 | 14B video + 5B audio streams | "native 4K and up to 50 fps"; audio up to 10 s | — | weights public | — |
- **Later closed models [DATED; helper A9]:** Seedance 1.0 (Jun 2025: 5 s 1080p in 41.4 s on an L20); Runway Gen-4.5 (1 Dec 2025); Kling 3.0 (5 Feb 2026, up to 15 s with native audio); Seedance 2.0 (Feb 2026, 4–15 s, 480p/720p, audio); Wan 3.0 (API, GA 24 Aug 2026, up to 30 s); Kling 4.0 (15 Oct 2026, "up to 30 seconds"). Trends: joint audio+video, 15–30 s single passes, causal/streaming generation.

### Worked examples: tokens per clip [DERIVED; helper A10]
| Clip | Arithmetic | Tokens |
|---|---|---|
| Wan 2.1, 5 s 720p, 16 fps (81 frames) | (1 + 80/4) × (720/16) × (1280/16) = 21 × 45 × 80 | **75,600** |
| Same at 480p (832×480) | 21 × 30 × 52 | 32,760 (attention pairs 5.3× fewer) |
| HunyuanVideo 1280×720 × 129 f | 33 × 45 × 80 | **118,800** |
| Movie Gen 768² × 256 f (16 s @ 16 fps) | 32 × 48 × 48 | **73,728** (= paper's "(73728, 6144)") |
| Mochi 848×480 × 163 f | 28 × 30 × 53 | 44,520 (= README "Visual Tokens 44520") |
| LTX-Video 768×512 × 121 f | 16 × 16 × 24 | 6,144 |
| SD3/FLUX image 1024² (for comparison) | 64 × 64 | 4,096 |
- Pixels vs latent for the Wan clip: 81 × 720 × 1280 × 3 = 223,948,800 values → 21 × 90 × 160 × 16 = 4,838,400 latent values (46×).
- One full-attention map for HunyuanVideo's clip: 118,800² ≈ 1.4 × 10¹⁰ entries per head per layer.
- **The moving digit (course demo):** a 16-frame 28×28 clip is 16 × 784 = 12,544 numbers; treated as one tensor, it is denoised jointly (§22).

### Common misconceptions
- "Sora is a world simulator." OpenAI's report frames simulators as "a promising path", and lists physics failures (glass shattering, eating food) in the same report.
- "Video models generate frame by frame." Most current models denoise a whole clip's latent jointly with full attention; frame-by-frame (causal) generation is a separate, newer line (§22).
- "A minute of video in one pass." Sora's report says "a minute"; public Sora Turbo was "up to 20 sec"; open models do 5–10 s per pass and extend by conditioning on previous frames.
- "Video models are separate from image models." Almost all are initialised from or co-trained with images (VDM's joint training, Video LDM/SVD from SD, Movie Gen's O(1)B images).

### Quotable
- "Our results suggest that scaling video generation models is a promising path towards building general purpose simulators of the physical world." (OpenAI, Sora technical report, 15 Feb 2024)
- "Sora currently exhibits numerous limitations as a simulator. For example, it does not accurately model the physics of many basic interactions, like glass shattering." (same report)
- "Our largest video generation model is a 30B parameter transformer trained with a maximum context length of 73K video tokens, corresponding to a generated video of 16 seconds at 16 frames-per-second." (Movie Gen, abstract)

---

## §22 Keeping frames together: temporal consistency, long videos, world models

### Mechanism (plain words)
- **Independent frames flicker.** Denoise each frame from its own noise with an image model and every frame is a different, unrelated sample: the object changes shape, colour, position. Even with the same noise per frame, small differences in the denoising path accumulate.
- **Joint denoising holds frames together.** If every denoising step sees all frames at once (through temporal attention, 3D convolutions, or full spacetime attention), the network can make them agree. Temporal layers are often initialised as identity so a pretrained image model is unchanged at the start (Make-A-Video, Video LDM).
- **Local vs global consistency.** Temporal *convolutions* keep neighbouring frames smooth cheaply; temporal or full *attention* can link distant frames (an object leaving and returning). Imagen Video used attention in the base model and convolutions in the upsamplers.
- **More frames, fewer models.** Earlier cascades added frames with **frame interpolation** / temporal super-resolution (3 → 6 → 12 → 24 fps). Video VAEs that compress time (Movie Gen's 8×) "forego frame-interpolation models".
- **Long videos** exceed one forward pass. Options: extend autoregressively, conditioning on the last frames (VDM's reconstruction guidance; Veo 3.1's extension from "the final second of the previous clip"); generate keyframes then fill in (NUWA-XL); use a sliding window whose later frames are noisier (Rolling Diffusion; Diffusion Forcing); or make the model causal and distil it for streaming (CausVid, Self Forcing). Errors accumulate ("drift"); noise augmentation of the context and training on the model's own rollouts fight it.
- **World models** take the next step: generate the next frames conditioned on *actions*, interactively (GameNGen plays DOOM; Genie 2/3 make explorable worlds). Not all are diffusion models (Genie 1 uses MaskGIT).

### Key formulas
- **Rolling diffusion noise ramp** (Ruhe et al. 2024; schematic, see the paper for the exact schedule): within a window of W frames, frame k gets a local time t_k ≈ clip((k + t_global)/W) so later frames are noisier; when the first frame reaches t = 0 it is emitted and a fresh noisy frame enters.
- **Diffusion Forcing** (Chen et al. 2024): train with an independent noise level per token/frame, so at sampling time the past can be clean and the future noisy in any pattern.
- **Cost of full vs factorised attention:** (T·HW)² vs T·(HW)² + HW·T² [DERIVED]: for T = 21 latent frames and HW = 3,600 tokens per frame, full attention has 5.7 × 10⁹ pairs, factorised 2.7 × 10⁸ + 1.6 × 10⁶ (≈ 21× fewer).

### Numbers with sources (helper file B1–B4)
- **Factorised designs:** VDM (1×3×3 convs + temporal attention); Make-A-Video ("we apply an additional 3x1x1 convolution projection (after each 1x3x3)"; temporal layers identity-initialised; full spatiotemporal attention "outright infeasible in terms of memory"); Video LDM (temporal attention + 3D-conv residual blocks blended with the frozen spatial output via a learned α).
- **Arguments for full 3D attention:** CogVideoX: separated attention "requires extensive implicit transmission of visual information, significantly increasing the learning complexity and making it challenging to maintain the consistency of large-movement objects." HunyuanVideo: Full Attention "has demonstrated superior performance compared to divided spatiotemporal attention". LTX-Video: its compressed latent "enables the transformer to efficiently perform full spatiotemporal self-attention, which is essential for generating high-resolution videos with temporal consistency."
- **Interpolation:** Imagen Video TSR models "increase temporal resolution by filling in intermediate frames between input frames"; Make-A-Video's ↑F turns 16 frames into 76 ("(16-1)×5+1"); Movie Gen: the 8× temporal TAE "allows us to forego frame-interpolation models commonly used in prior work".
- **Long video:**
  - NUWA-XL (Yin et al., arXiv:2303.12346, Mar 2023): "A global diffusion model is applied to generate the keyframes across the entire time range, and then local diffusion models recursively fill in the content between nearby frames"; trained directly on 3,376-frame videos; average inference time for 1,024 frames from 7.55 min to 26 s.
  - Rolling Diffusion (Ruhe, Heek, Salimans & Hoogeboom, arXiv:2402.09470, Feb 2024): Kinetics-600 rollout FVD 39.8 vs 58.1 for standard diffusion (stride 8); standard wins at stride 1 (123.7 vs 211.2). Helps when "data is highly dynamic".
  - Diffusion Forcing (Chen, Martí Monsó, Du, Simchowitz, Tedrake & Sitzmann, arXiv:2407.01392, Jul 2024; NeurIPS 2024): "a diffusion model is trained to denoise a set of tokens with independent per-token noise levels"; rolls out "2x-5x times longer than it's trained on without sliding window".
  - CausVid (Yin et al., arXiv:2412.07772, Dec 2024): block-causal student distilled with DMD from "50-step diffusion model into a 4-step generator"; the bidirectional teacher takes "219 seconds to generate a 128-frame video"; CausVid "initial latency of only 1.3 seconds", then ≈ 9.4 FPS.
  - Self Forcing (Huang, Li, He, Zhou & Shechtman, arXiv:2506.08009, Jun 2025): trains on the model's own autoregressive rollouts; "real-time video generation at 17 FPS with sub-second latency on a single H100 GPU" (Wan2.1-1.3B base, 832×480, 16 fps; base model 0.78 FPS, 103 s latency).
  - Product extension [DATED]: Veo 3.1 Extend conditions on "the final second of the previous clip"; Gemini Omni 1.1 on "up to 10 seconds of prior context", "up to a total cumulative length of 40 seconds".
- **World models [DATED]:**
  - Genie (Bruce et al. (Google DeepMind), arXiv:2402.15391, Feb 2024): "At 11B parameters, Genie can be considered a foundation world model." **Not diffusion**: video tokenizer + MaskGIT dynamics + latent action model with 8 latent actions; trained on 30,000 hours (6.8M 16-s clips at 10 fps, 160×90) filtered from "over 200,000 hours"; "operates around 1FPS".
  - GameNGen (Valevski, Leviathan, Arar & Fruchter, arXiv:2408.14837, Aug 2024; ICLR 2025): "runs at 20 frames per second on a single TPU"; built on SD v1.4 with v-prediction; conditions on the last 64 frames and actions; 4 DDIM steps; without noise augmentation quality "degrades quickly after 10-20 frames".
  - Genie 2 (DeepMind blog, 4 Dec 2024): "an autoregressive latent diffusion model"; "consistent worlds for up to a minute".
  - Oasis (Decart + Etched, 31 Oct 2024): DiT backbone "trained using Diffusion Forcing", "real-time output in 20 frames per second"; 500M open model.
  - Genie 3 (DeepMind blog, 5 Aug 2025): "navigate in real time at 24 frames per second, retaining consistency for a few minutes at a resolution of 720p"; "visual memory extending as far back as one minute ago". Project Genie prototype (29 Jan 2026) limits generations to 60 s. No "Genie 4" as of 2026-10-25.

### Worked example: independent vs joint denoising on the moving digit [DERIVED, conceptual]
- 16 frames of a "7" sliding right, each 28×28. **Independent:** 16 separate draws from the digit model → 16 different 7s (stroke thickness, slant) that jump between frames. **Shared noise, independent denoising:** frames start alike but drift apart as positions differ. **Joint (3D) denoising:** one network sees all 16 frames, so the same stroke style persists and only the position changes. The course measures a frame-to-frame difference metric for the three cases (`research/experiments/`).
- **Real-time budget:** 24 fps leaves 41.7 ms per frame. GameNGen: 4 denoising steps × 10 ms + 10 ms decoder = 50 ms → 20 fps; 1 step → 20 ms → 50 fps (paper §3.3.2).

### Common misconceptions
- "Temporal consistency is solved by fixing the seed." Same noise per frame does not make frames agree once their content must move; joint modelling does.
- "World models are just video generators." They condition on actions and must respond interactively; Genie 3 runs at 24 fps but "a few minutes of continuous interaction, rather than extended hours".
- "Autoregressive video = no diffusion." CausVid, Self Forcing, MAGI-1 and Genie 2 are autoregressive *across* chunks and diffusion *within* each chunk.

### Quotable
- "this separated attention approach requires extensive implicit transmission of visual information, significantly increasing the learning complexity and making it challenging to maintain the consistency of large-movement objects." (CogVideoX, §2.2)
- "GameNGen runs at 20 frames per second on a single TPU and remains stable over extended multi-minute play sessions." (Valevski et al. 2024)

---

## §23 Running diffusion on a Mac: Core ML, MLX, DiffusionKit, mflux; where the time goes

### Mechanism (plain words)
- **Unified memory.** On Apple silicon the CPU, GPU and Neural Engine share one RAM pool. A model fits if its weights + activations + the OS fit in RAM; there is no separate VRAM limit, but swapping kills speed.
- **Three costs per image:** (1) text encoder, run once (twice with CFG); (2) the denoiser, run once per step (twice with CFG): this dominates; (3) VAE decoder, run once at the end. Plus one-off model loading/compilation.
- **Ways to run:** Apple's **Core ML** conversion (`ml-stable-diffusion`, Dec 2022; Neural Engine or GPU; weight palettization to 6 bits or mixed 2–8 bits); **MLX** (Apple's array framework; `mlx-examples/stable_diffusion` for SD 2.1 and SDXL Turbo; `flux`); **DiffusionKit** (Argmax; SD3/FLUX via MLX + Core ML; archived); **mflux** (MLX port of FLUX and newer models, with 4/8-bit quantisation); PyTorch with the `mps` backend (diffusers).
- **Levers:** fewer steps (distilled models), lower precision (fp16/bf16, 8-/4-bit weights), smaller resolution, dropping the T5 encoder (SD3), batching (MLX throughput rises with batch size), and avoiding swap.
- **This course's Mac** (Mac Studio, M4 Max, 64 GB): an fp16 SD 1.5 pipeline (≈ 2.1 GB of weights) or SDXL (≈ 6.9 GB) fits easily; bf16 FLUX.1 (≈ 33.7 GB) fits without quantisation; FLUX.2 [dev] (32B + 24B text encoder) does not without heavy quantisation [DERIVED from the size table below and the helper file].

### Key formulas [DERIVED]
- Weight memory = parameters × bytes per parameter (fp32 4, fp16/bf16 2, 8-bit 1, 4-bit 0.5) + small overheads (quantisation scales; unquantised layers).
- Time per image ≈ t_text + N_steps × (CFG ? 2 : 1) × t_denoiser + t_vae + t_load.
- Steps/s ("it/s") reported by tools usually counts denoising steps only (with CFG folded in as a batch of 2).

### Numbers with sources [DATED throughout; full tables in `helper_mac_models.md`]
- **Apple Core ML, Nov 2022** (README at first commit; SD v1.4, 50 steps, 512×512, CFG as "batch size of 2 for unet", 77 tokens): Mac Studio M1 Ultra 64-core GPU **9 s**; M1 Max 32-core 18 s; MacBook Pro M2 (Neural Engine) 23 s; MacBook Pro M1 35 s; iPad Pro M1 38 s. "The Swift CLI program consumes a peak memory of approximately 2.6GB (without the safety checker), 2.1GB of which is model weights in float16 precision." Apple ML blog (6 Dec 2022): "4 different neural networks totaling approximately 1.275 billion parameters" (does not match a recount; Appendix A).
- **Core ML, June–Aug 2023** (SD 2.1 base, 20 steps, 512×512, 6-bit weights): Mac Studio M2 Ultra **3 s** end-to-end, **7.6 it/s**; M1 Ultra 4 s, 6.3 it/s; iPhone 14 Pro Max 7.9 s, 2.69 it/s; iPad Pro M2 7.0 s, 3.07 it/s.
- **Core ML SDXL 1.0 base** (July 2023, 1024×1024, 20 steps, fp16): MacBook Pro M1 Max 46 s (0.46 it/s); M2 Max 37 s (0.57); Mac Studio M1 Ultra 25 s (0.89); **M2 Ultra 20 s (1.11 it/s)**. Mixed-bit palettization: "`recipe_4.50_bit_mixedpalette` … compressed from ~5.2GB to ~1.46GB for SDXL"; "We recommend a minimum of 6 bits for palettizing Stable Diffusion."
- **MLX Stable Diffusion** (mlx-examples README, Nov 2023–Mar 2024 version; SD 2.1, M2 Ultra, U-Net images/s at batch 1/4/16): PyTorch MPS 6.25 / 7.69 / 6.32; MLX 4.17 / 7.14 / **8.79**; "it takes about 90 seconds to fully generate 16 images with MLX and 50 diffusion steps with classifier free guidance and about 120 for PyTorch." Current README: SDXL's "UNet alone has 2.6B parameters"; with quantisation (text encoders 4-bit, U-Net 8-bit) SDXL Turbo runs "on an 8GB Mac Mini with no-swapping".
- **mflux FLUX.1 [schnell]** (community-reported table, README Nov 2024–Jan 2026; non-quantised, 2 steps, 1024×1024, *including model loading*): M2 Ultra < 15 s; **M4 Max (128 GB) ~19 s**; M3 Max ~20 s; M2 Max (96 GB) ~25 s; M4 Pro (64 GB) ~34 s; M1 Max (64 GB) ~55 s; M1 Pro (16 GB) ~175 s "Might freeze your mac"; M1 (8 GB) ~335 s at 512×512. Quantised sizes (both schnell and dev): 3-bit 7.52 GB, 4-bit 9.61 GB, 6-bit 13.81 GB, 8-bit 18.01 GB, "Original (16 bit)" **33.73 GB**. 8-bit gave "a ~2x speedup" on a 32 GB M1 Pro (less swapping).
- **DiffusionKit:** no primary latency table; archived (GitHub API); default steps "4 for FLUX.1-schnell, 50 for SD3 and FLUX.1-dev". Third-party (Replicate blog, 18 Jun 2024): SD3 Medium 512×512, 28 steps on an M3 Max: ~16.7 s (DiffusionKit MLX) vs ~14.1 s (PyTorch MPS) **[secondary source]**.
- **Draw Things** (engineering blog, 7 Jan 2025): FLUX.1 "up to 25% faster than mflux implementation on M2 Ultra for each iteration" (different precisions; relative numbers only).
- **Weights in bf16/fp16** (§8, §11, helper 1f; [DERIVED] = params × 2 bytes):

| Component | Parameters | fp16/bf16 |
|---|---|---|
| SD 1.5 U-Net | 859,520,964 | 1.72 GB |
| SD 1.5 CLIP-L text encoder | 123,060,480 | 0.25 GB |
| SD VAE: encoder + decoder (decoder alone) | 83,653,863 (49,490,179) | 0.17 GB (0.10) |
| **SD 1.5 text + U-Net + decoder** | 1,032,071,643 | **2.06 GB** (= Apple's "2.1GB") |
| SDXL U-Net | 2,567,463,684 | 5.13 GB |
| SDXL text encoders (L + bigG) | 817,720,320 | 1.64 GB |
| SDXL base pipeline | 3,468,837,867 | 6.94 GB |
| FLUX.1 transformer | 11.89–11.90B | 23.8 GB |
| T5-XXL encoder | 4,762,310,656 | 9.52 GB |
| **FLUX.1 full pipeline** | ≈ 16.87B | **33.73 GB** (= mflux "33.73GB") |
| SD 3.5 Large | 8,146,280,768 | 16.29 GB (+ T5 9.52 + CLIPs 1.64) |
| SD 3.5 Medium | 2,469,663,936 | 4.94 GB |
- **Where the time goes:**
  - SDXL Turbo on an A100 (Stability, 28 Nov 2023): "generates a 512x512 image in 207ms (prompt encoding + a single denoising step + decoding, fp16), where 67ms are accounted for by a single UNet forward evaluation." [DERIVED] With one step, everything except the U-Net is 140 ms = 68%.
  - Core ML SDXL on an M2 Ultra [DERIVED, assuming Apple's it/s counts only the denoising loop]: 20 steps ÷ 1.11 it/s = 18.0 s of the 20 s total → denoiser ≈ **90%**; text encoders + VAE + overhead ≈ 2 s. M1 Max: 43.5 of 46 s.
  - VAE decode on a Mac (mflux Z-Image README, 1024×1024, 36 GB Mac): full-image decode 0.65 s at 7.3 GB peak; tiled ("strips") 1.29 s at 1.6 GB peak. FLUX.2 klein VAE: 0.87 s, 10.3 GB peak; tiled 1.27 s, 1.6 GB.
  - diffusers MPS guide: "M1/M2 performance is very sensitive to memory pressure. When this occurs, the system automatically swaps if it needs to which significantly degrades performance"; recommends attention slicing "if your computer has less than 64GB of system RAM".
  - SD3 without T5 (Stability): "By removing the memory-intensive 4.7B parameter T5 text encoder for inference, SD3’s memory requirements can be significantly decreased with only small performance loss." diffusers SD3 blog (A100): default 18.8 GB; "Offloading + no T5" 4.3 GB.

### Worked example: a lighthouse budget on this Mac [DERIVED; the measured numbers in `research/experiments/` replace these on screen]
- SD 1.5, 512², 50 steps, CFG 7.5: 1 text-encoder call (batch 2) + 50 U-Net calls (batch 2) + 1 VAE decode. If the denoiser runs at r U-Net batch-2 calls per second, time ≈ 50/r + ~1 s fixed. At Apple's 2022 M1 Ultra figure (9 s for the whole run) that is ≈ 0.16 s per step; an M4 Max should be faster (measure it).
- SDXL Turbo, 512², 1 step, no CFG: one U-Net call; fixed costs dominate (≈ 2/3 of time on an A100).
- FLUX.1 [schnell], 1024², 4 steps: 4 transformer calls on 4,096 + 512 tokens, no CFG; bf16 weights 33.7 GB, fits in 64 GB. mflux's M4 Max (128 GB) report: ~19 s for 2 steps including loading.

### Common misconceptions
- "Macs can't run these models." SD 1.x ran in 9–35 s on 2022 M1-family Macs; FLUX runs in tens of seconds on Max/Ultra chips.
- "it/s tells you the total time." It excludes loading, text encoding and decoding; with 1–4-step models these fixed costs can dominate.
- "4-bit halves memory compared with 8-bit exactly." Some layers stay unquantised (mflux 4-bit 9.61 GB vs ideal 8.4 GB).
- "The Neural Engine is always fastest." Apple's own tables use the GPU (CPU_AND_GPU) for the fastest Mac results and the Neural Engine on iPhone/iPad and smaller Macs.

### Quotable
- "This benchmark was conducted by Apple using public beta versions of iOS 16.2, iPadOS 16.2 and macOS 13.1 in November 2022." (apple/ml-stable-diffusion README, 2022)
- "these numbers do not represent **peak** HW capability." (apple/ml-stable-diffusion README)

---

## §24 The whole pipeline revisited: one generation, every stage, real sizes

### Mechanism (plain words)
The lighthouse prompt through Stable Diffusion 1.5 (the reference latent model), stage by stage. FLUX/SD3 differences are noted at the end. Every number is either a config/parameter fact (sources in §8, §15, §17) or arithmetic on them [DERIVED].

### Worked example: "a lighthouse on a rocky coast at dusk, oil painting", SD 1.5, 512×512, 50 steps, guidance scale 7.5 [DERIVED]
1. **Tokenise.** CLIP BPE (vocab 49,408) → start token + the prompt's tokens + end token, padded to **77** IDs. A second, empty prompt "" is also tokenised for CFG.
2. **Encode text.** Frozen CLIP ViT-L/14 text transformer (12 layers, width 768, **123.06M** parameters) → **77 × 768** embeddings per prompt; batch of 2 (prompt, empty) → 2 × 77 × 768 = 118,272 numbers. Run **once**.
3. **Start from noise.** Draw z_T ~ N(0, I) of shape **1 × 4 × 64 × 64 = 16,384** numbers (the seed fixes it). This is the "static" frame.
4. **Denoise, 50 times.** Visit timesteps 981, 961, …, 1 (diffusers "leading" spacing). At each step:
   - the U-Net (**859.5M** parameters; 320/640/1280/1280 channels at 64/32/16/8; cross-attention to the text at 64, 32, 16) runs on a batch of 2: (z_t, prompt) and (z_t, empty);
   - CFG: ε̃ = ε_u + 7.5·(ε_c − ε_u);
   - the sampler (DDIM/PNDM/Euler…) turns ε̃ into z_{t−1}.
   - Total: 50 calls at batch 2 = **100** U-Net evaluations. The first visited step (diffusers index t = 981) has ᾱ = 0.0058: signal coefficient √ᾱ = 0.076, noise coefficient 0.997 (EDM-style σ = 13.1; σ_max at t = 999 is 14.6, the Euler samplers' starting scale). The last step (t = 1) has noise coefficient 0.041.
5. **Decode once.** Divide by the scale factor (z₀ / 0.18215), run the VAE decoder (**49.5M** parameters): 64×64×4 → **512×512×3** values in [−1, 1] → map to 0–255 integers → the picture. Total pixels 786,432 numbers from 16,384 latent numbers (48×).
6. **Weights touched:** 123M + 860M + 49.5M ≈ **1.03B** parameters ≈ 2.06 GB in fp16 (Apple: peak ≈ 2.6 GB for the whole program).
7. **Work done:** 1 text-encoder pass (batch 2), 50 U-Net passes (batch 2), 1 decoder pass. The denoiser is ≈ 90% of the time (SDXL on M2 Ultra) when steps ≥ 20; with 1–4-step models, steps shrink and the fixed costs show.

**Where intermediate frames come from:** at any step, x̂₀ = (z_t − √(1−ᾱ_t)·ε̃)/√ᾱ_t decoded by the VAE gives the "picture emerging out of noise" frames (Tweedie's estimate, §6).

**Same idea, other models:**
| | SD 1.5 | SDXL | SD3 Medium/Large | FLUX.1 [schnell] / [dev] |
|---|---|---|---|---|
| Output | 512² | 1024² | ~1 MP | 0.1–2 MP |
| Text encoders | CLIP-L (77×768) | CLIP-L + OpenCLIP bigG (77×2048 + pooled 1280) | CLIP-L + CLIP-G + T5-XXL (154×4096 + pooled 2048) | CLIP-L pooled + T5-XXL (≤ 512 tokens ×4096) |
| Latent | 64×64×4 | 128×128×4 | 128×128×16 | 128×128×16 → 4,096 tokens ×64 |
| Denoiser | U-Net 0.86B | U-Net 2.57B | MM-DiT 2B / 8.1B | MM-DiT+single-stream 11.9B |
| Objective | ε-prediction | ε-prediction | rectified flow | rectified flow |
| Steps × passes (defaults) | 50 × 2 (CFG 7.5) | 50 × 2 (CFG 5.0) | 28 × 2 (CFG 7.0) | 4 × 1 / 28 × 1 (distilled guidance 3.5) |
| Network evaluations | 100 | 100 | 56 | 4 / 28 |

### Common misconceptions
- "The model paints the image pixel by pixel." It refines the whole 64×64×4 latent at every step; the VAE turns it into pixels once at the end.
- "More steps always cost linearly." Steps scale the denoiser cost; text encoding and decoding are paid once per image.
- "CFG is free." It doubles denoiser work unless the model is guidance-distilled (FLUX dev) or step-distilled without CFG (ADD/SDXL Turbo, FLUX schnell).

### Quotable
- "By default, this uses a guidance scale of `--scale 7.5`, Katherine Crowson's implementation of the PLMS sampler, and renders images of size 512x512 (which it was trained on) in 50 steps." (CompVis Stable Diffusion README)

---
## Appendix A: Where sources disagree (read before quoting a number)

Each item says which version to use on screen. Details are in the section named.

| § | Topic | Disagreement | Use on screen |
|---|---|---|---|
| 2 | MNIST licence | LeCun's page states none; Keras docs: CC BY-SA 3.0 ("Yann LeCun and Corinna Cortes hold the copyright"); Hugging Face `ylecun/mnist` card: MIT | "freely available; CC BY-SA 3.0 per Keras" or avoid |
| 4, all | The symbol α | DDPM α_t = 1 − β_t; DDIM's α_t = DDPM's ᾱ_t; VDM/Salimans & Ho/Imagen/Lin α_t = √ᾱ_t | always write formulas in the course notation |
| 4, 11 | Direction of t | DDPM/score SDE/SD3: t = 0 data; Lipman (FM) and Liu (RF): t = 0 noise, so their velocity has the opposite sign | t = 0 data, v = ε − x₀ |
| 4 | Cosine ᾱ_T | formula without clipping gives ≈ 4 × 10⁻³³; with β ≤ 0.999 clipping 2.43 × 10⁻⁹ (Lin et al. Table 1) | 2.4 × 10⁻⁹ (clipped) |
| 5, 11 | "v" | v-prediction (Salimans & Ho) vs flow-matching velocity | say "v-prediction" or "velocity" |
| 9 | DDIM 10-step CIFAR-10 FID | DDIM Table 1: 13.36; consistency-models Table 1 (other model/spacing): 8.23 | 13.36 (DDIM paper) |
| 9 | Best CIFAR-10 sampler at 1,000 steps | DDPM with σ̂: 3.17; DDIM η = 0: 4.04 | DDPM wins at 1,000, DDIM wins at ≤ 100 |
| 10 | EDM "35 NFE" | the paper states NFE, not steps; 35 = 18 Heun steps [DERIVED] | "35 network evaluations (18 Heun steps)" |
| 3, 13 | ADM-G 25-step ImageNet 256 FID | ADM Table 5: 5.44; LDM Table 18 (quoting [15]): 5.58 | 5.44 (ADM's own table) |
| 3 | BigGAN-deep ImageNet 128 FID | ADM Table 5: 6.02; CFG paper Table 2: 5.7 (BigGAN-deep row) ; BigGAN abstract 7.4 is BigGAN (not -deep) | name the paper; 6.02 next to ADM |
| 3 | "Diffusion beats GANs" | true in ADM (May 2021); StyleGAN-XL (Feb 2022) had ImageNet 256 FID 2.30 vs ADM-G-U 3.94 | "in 2021"; mention StyleGAN-XL if needed |
| 11 | SD3 text-encoder dropout | main text 46.3%; App. B.3 46.4% (0.464³ = 0.0999 ≈ 10%) | "about 46% each, so ≈ 10% fully unconditional" |
| 11, 18 | SD3 shift formula | extracted PDF text was garbled around eq. 23; form used here t_m = αt/(1 + (α−1)t) is consistent with eq. 24 (λ shift −2 log α) | quote α = 3.0; show the curve, not the formula, unless re-checked |
| 11, 18, 23 | FLUX.1 size | BFL "12B"; diffusers memory docs "12.5B"; HF metadata 11.89B (schnell) / 11.90B (dev) | "12 billion parameters" (11.9B counted) |
| 12 | 1-step progressive distillation, CIFAR-10 | Salimans & Ho Table 2: 9.12; consistency-models Table 1 lists PD 1 step 8.34 | 9.12 (own paper) |
| 14 | CFG paper's FID sentence | text: "FID monotonically decreasing … with w"; Tables 1–2: FID rises for w > 0.1–0.3 | use the tables; best FID at small w |
| 14 | CFG paper date | NeurIPS 2021 Workshop on Deep Generative Models; arXiv 26 Jul 2022 | "Ho & Salimans (2021; arXiv 2022)" |
| 15 | CLIP context length | paper: "max sequence length was capped at 76" plus [SOS]/[EOS] bracketing; released config: 77 positions | "77 tokens" |
| 15 | T5-XXL encoder size | Imagen: 4.6B; SD3: 4.7B; derived from config 4.76B incl. 131.6M embedding (4.63B without) | "about 4.7B" |
| 17 | SD v1.4 training steps | release post: "v1.4 470k … v1.3 440k"; model card: "Resumed from stable-diffusion-v1-2. 225,000 steps" (cumulative vs incremental) | the model card |
| 17, 23 | SD parameter total | Apple blog "approximately 1.275 billion parameters" in 4 networks; recount: text + U-Net + VAE = 1.066B; with safety checker and decoder only 1.336B | component counts (860M + 123M + 84M) |
| 17, 23 | SDXL U-Net fp16 size | Apple "~5.2GB"; 2,567,463,684 × 2 B = 5.13 GB | "about 5 GB" |
| 18 | PixArt-α training cost | arXiv v1: 10.8%, 675 vs 6,250 A100-days, $26,000; v3: 12%, ~753 vs ~6,250, $28,400 | v3 |
| 18 | DiT's 2.27 | guidance on 3 of 4 latent channels at scale 1.5; 4-channel at 1.375 gives 2.20 | "2.27 with guidance 1.5" |
| 19 | DDPM FID reference set | vs training set 3.17; vs test set 5.24 | 3.17 (standard), say "vs training set" if comparing |
| 21 | SVD dataset size | Table 1: 577M clips; text: "580M"; intro: "roughly 600 million" | 577M → 152M curated |
| 21 | VDM UCF-101 | arXiv v1: FID 330, IS 57.8; v2: FID 295, IS 57 | v2, or avoid |
| 21 | Imagen Video component count | "7 sub-models" (diffusion) vs 8 components including T5-XXL | "7 diffusion models + a text encoder" |
| 21 | HunyuanVideo size | own paper "over 13 billion"; Open-Sora README "11B HunyuanVideo" | 13B |
| 21 | Sora duration | report: "a minute of high fidelity video"; public Sora Turbo (Dec 2024): "up to 20 sec long" | name which |
| 21 | Sora "simulator" wording | "highly-capable simulators" (capabilities section) vs "capable simulators" (closing) vs abstract "general purpose simulators of the physical world" | quote one passage exactly |
| 21–22 | Video speed claims | LTX-Video own: 2 s for 121 frames (H100); Self Forcing's re-measurement: 13.5 s latency; CausVid own 9.4 FPS vs 17 FPS on another base | quote each paper's own number with its setting |
| 22 | Genie size and data | 11B (abstract) vs 10.6B components; 200,000 h (intro) vs ~244k h raw vs 30,000 h used | "11B; trained on 30,000 hours of filtered gameplay" |
| 23 | mflux quantised FLUX sizes | 2024-11 README: 4-bit 9.85 GB, 8-bit 18.16 GB; 2026-01 README: 9.61 / 18.01 GB | latest (9.61 / 18.01), with date |
| 17 | SD3 Medium licence | same page says "Stability Community License" and "Non-Commercial Research Community License" (page edited after the 5 Jul 2024 change) | "released non-commercial; moved to the Community License in July 2024" |
| 1 | DALL·E 3 date | system card 3 Oct 2023; API 6 Nov 2023; announcement page (403) reportedly 20 Sep 2023 | "autumn 2023" or "October 2023 (system card)" |

---

## Appendix B: Not verified against a primary source

Everything here is also tagged [UNVERIFIED] where it appears, or is stated with its secondary source.
- **§2** MNIST licence (Keras docs, not LeCun); the LeCun page facts were read from a 2024 Wayback snapshot (live page is a directory listing).
- **§6** Vincent 2011 (PDF unreachable; objective as stated by Song & Ermon 2019); Hyvärinen 2005 (cited, not opened); Song & Ermon 2020's "Technique 1" value of σ_max (not read).
- **§11** SD3 eq. 23 exact typography (garbled extraction; formula reconstructed and cross-checked against eq. 24); FLUX dev − schnell 10.2M difference = guidance embedding (inference, not stated by BFL).
- **§15** The exact token count of the lighthouse prompt (to be measured in the course run).
- **§16** LoRA file sizes for SD (order of magnitude only); diffusers img2img "strength → start step" convention (derived from library behaviour, not from a paper).
- **§19** CIFAR-10 real-data IS ≈ 11.24 (commonly cited); "2,048-d" for FID's pool3 features is standard Inception-v3 knowledge, not printed in the FID-paper passage read.
- **§20** DALL·E 3 announcement date (openai.com 403).
- **§21–§22** (from the video helper): Sora app/web end date (26 Apr 2026, secondary only); Runway Gen-4 date (Mar 2025); Wan 2.5–2.7 specs; Veo 2 launch limits; Make-A-Video final fps; Kling 3.0/4.0 read through press-release reproductions (company sites returned 403); Oasis resolution; Genie 3 and Gemini Omni architectures (not disclosed); exact LTX-2.5 release date (only HF repo creation).
- **§23** DiffusionKit timings (Replicate blog, third party); Argmax/Apple-staff posts on X (not fetched); "Where time goes" splits derived from end-to-end vs it/s, assuming Apple's it/s covers only the denoising loop; Draw Things absolute timings (charts only).
- **Venues not printed in the arXiv PDFs read** (given from general knowledge; the arXiv IDs and dates are verified): DDPM NeurIPS 2020; NCSN NeurIPS 2019; ADM NeurIPS 2021; iDDPM ICML 2021; CLIP ICML 2021; GLIDE ICML 2022; Imagen NeurIPS 2022; LDM CVPR 2022; GAN NeurIPS 2014; U-Net MICCAI 2015; PLMS ICLR 2022; DiT ICCV 2023; U-ViT CVPR 2023; PixArt-α ICLR 2024; SD3 ICML 2024; flow matching ICLR 2023; rectified flow ICLR 2023; LoRA ICLR 2022; DreamBooth CVPR 2023; textual inversion ICLR 2023; ControlNet ICCV 2023; prompt-to-prompt ICLR 2023; CLIPScore EMNLP 2021; clean-fid CVPR 2022; Chong & Forsyth CVPR 2020; CMMD CVPR 2024; LAION-5B NeurIPS 2022 Datasets & Benchmarks; Diffusion-DPO CVPR 2024; DDPO ICLR 2024; DMD CVPR 2024; DMD2 NeurIPS 2024; IS NeurIPS 2016; FID NeurIPS 2017. Venues confirmed in the PDFs: Sohl-Dickstein et al. ICML 2015; Pope et al. ICLR 2021; DDIM ICLR 2021; score SDE ICLR 2021; progressive distillation ICLR 2022; EDM NeurIPS 2022; DPM-Solver NeurIPS 2022; UniPC NeurIPS 2023; consistency models ICML 2023; stochastic interpolants ICLR 2023; precision/recall NeurIPS 2019; FID-and-ImageNet-classes ICLR 2023; PickScore NeurIPS 2023; ImageReward NeurIPS 2023. Not confirmed in the PDF header: SDEdit ICLR 2022 and Carlini et al. USENIX Security 2023 (general knowledge).
- **Blogs read via the Wayback Machine** (openai.com returns 403 to scripts): Sora report (capture 5 Jan 2025), Sora landing page, "Sora is here", "Sora 2 is here". Quotes are exact from those snapshots.

---

## Appendix C: Episode → section → the number to get right

| Ep | Section | The number or fact that must be exactly right |
|---|---|---|
| 1 | §1 | First diffusion paper Mar 2015; DDPM 19 Jun 2020; SD public release 22 Aug 2022; Sora report 15 Feb 2024 |
| 2 | §2 | 28×28 = 784 numbers; MNIST 60,000 train / 10,000 test; 512×512×3 = 786,432 |
| 3 | §3 | ImageNet 256: ADM-G FID 4.59 vs BigGAN-deep 6.95; recall 0.52 vs 0.28 (May 2021) |
| 4 | §4 | T = 1000, β 10⁻⁴ → 0.02; x_t = √ᾱ·x₀ + √(1−ᾱ)·ε; ᾱ_1000 = 4.0 × 10⁻⁵; SNR = 1 near t = 259 |
| 5 | §5 | L_simple = ‖ε − ε_θ‖²; CIFAR-10 FID 3.17 / IS 9.46; 800k steps in 10.6 h on a TPU v3-8 |
| 6 | §6 | score = −ε/√(1−ᾱ) (sign!); Tweedie x̂₀ = (x_t + (1−ᾱ)·score)/√ᾱ; NCSN 10 noise levels, σ 1 → 0.01 |
| 7 | §7 | x_{t−1} = (1/√α_t)(x_t − (β_t/√(1−ᾱ_t))ε_θ) + σ_t z, no noise at t = 1; 1,000 network calls |
| 8 | §8 | DDPM CIFAR U-Net 35.7M params, attention at 16×16, sinusoidal t; SD 1.5 U-Net 860M (859,520,964) |
| 9 | §9 | DDIM CIFAR-10 FID 13.36 / 6.84 / 4.67 / 4.16 / 4.04 at 10 / 20 / 50 / 100 / 1000 steps |
| 10 | §10 | EDM ρ = 7, σ 0.002 → 80, Heun, FID 1.79 at 35 NFE (= 18 steps) |
| 11 | §11 | x_t = (1−t)x₀ + tε, velocity ε − x₀ (t = 0 data); SD3 up to 8B; FLUX.1 12B (11.9B), 1 Aug 2024 |
| 12 | §12 | 8192 → 4 steps at CIFAR-10 FID 3.0; consistency 1-step FID 3.55; SDXL Turbo 1–4 steps at 512² |
| 13 | §13 | ADM Table 4: conditional 10.94 → 4.59 at classifier scale 1.0; scale 10 → FID 9.11, recall 0.32 |
| 14 | §14 | s = 1 + w; SD default 7.5 ⇔ w = 6.5; 2 network passes per step; best FID at w ≈ 0.1–0.3 |
| 15 | §15 | CLIP 400M pairs; SD 1.x text = 77 × 768; Imagen: T5-XXL 4.6B, COCO FID 7.27 |
| 16 | §16 | ControlNet: zero-initialised 1×1 convs, +23% memory / +34% time; DreamBooth 3–5 images, ~1,000 steps, ~5 min on an A100 |
| 17 | §17 | 512×512×3 → 64×64×4 (48×); scale 0.18215; LDM-4-G FID 3.60 at 271 vs ADM-G 4.59 at 962 V100-days |
| 18 | §18 | DiT-XL/2: 675M params, 118.6 Gflops, FID 9.62 unguided, 2.27 with CFG 1.5 |
| 19 | §19 | FID = ‖Δm‖² + Tr(C₁ + C₂ − 2(C₁C₂)^{1/2}); Inception pool features, 50k samples; biased at small N |
| 20 | §20 | LAION-5B 5.85B pairs (2.32B English); DALL·E 3: 95% synthetic captions; Carlini: 109 near-copies from 175M generations |
| 21 | §21 | Sora quote exactly ("a promising path towards building general purpose simulators of the physical world"); Movie Gen 30B, 73,728 tokens = 16 s at 16 fps |
| 22 | §22 | GameNGen 20 fps on one TPU; Genie 3 720p at 24 fps "for a few minutes"; Genie 1 is not diffusion |
| 23 | §23 | Apple 2022: SD 1.4, 50 steps, 512² in 9 s on M1 Ultra; FLUX.1 bf16 33.73 GB; denoiser ≈ 90% of time at ≥ 20 steps; our measured numbers override |
| 24 | §24 | 77 tokens → 77×768; noise 1×4×64×64; 50 steps × 2 passes = 100 U-Net evaluations; decode once to 512×512×3 |

---

## Appendix D: Sources (read Oct 2026), grouped by section

Each list is the set of sources actually used. "via helper" = read by a research helper and recorded with quotes in the scratch files `helper_video.md` / `helper_mac_models.md` (/Users/vpromisex/.claude/jobs/61935fea/tmp/diff-research/); "not opened" marks a source cited second-hand.

### §1–§3 (history, data, earlier generators)
- Kingma, D. P. & Welling, M. (2013). Auto-Encoding Variational Bayes. arXiv:1312.6114. https://arxiv.org/abs/1312.6114
- Goodfellow, I. et al. (2014). Generative Adversarial Nets. arXiv:1406.2661. https://arxiv.org/abs/1406.2661
- Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N. & Ganguli, S. (2015). Deep Unsupervised Learning using Nonequilibrium Thermodynamics. arXiv:1503.03585. https://arxiv.org/abs/1503.03585
- van den Oord, A., Kalchbrenner, N. & Kavukcuoglu, K. (2016). Pixel Recurrent Neural Networks. arXiv:1601.06759.
- van den Oord, A. et al. (2016). Conditional Image Generation with PixelCNN Decoders. arXiv:1606.05328.
- Brock, A., Donahue, J. & Simonyan, K. (2018). Large Scale GAN Training for High Fidelity Natural Image Synthesis. arXiv:1809.11096.
- Karras, T., Laine, S. & Aila, T. (2018). A Style-Based Generator Architecture for GANs. arXiv:1812.04948.
- Karras, T. et al. (2019). Analyzing and Improving the Image Quality of StyleGAN. arXiv:1912.04958.
- Sauer, A., Schwarz, K. & Geiger, A. (2022). StyleGAN-XL: Scaling StyleGAN to Large Diverse Datasets. arXiv:2202.00273.
- Dhariwal, P. & Nichol, A. (2021). Diffusion Models Beat GANs on Image Synthesis. arXiv:2105.05233. https://arxiv.org/abs/2105.05233
- LeCun, Y., Cortes, C. & Burges, C. J. C. The MNIST database. http://yann.lecun.com/exdb/mnist/ (Wayback 2024 snapshot); Keras MNIST docs https://keras.io/api/datasets/mnist/ ; HF card https://huggingface.co/datasets/ylecun/mnist
- Krizhevsky, A. (2009). CIFAR-10. https://www.cs.toronto.edu/~kriz/cifar.html
- ImageNet. https://www.image-net.org/ ; Russakovsky, O. et al. (2015). ImageNet Large Scale Visual Recognition Challenge. arXiv:1409.0575 (Table 2).
- Pope, P. et al. (2021). The Intrinsic Dimension of Images and Its Impact on Learning. arXiv:2104.08894.
- Timeline product dates: helper files (Stability AI, BFL, Google, OpenAI, Meta posts with URLs); BFL FLUX 3 post https://bfl.ai/blog/flux-3 and OpenAI deprecations https://developers.openai.com/api/docs/deprecations re-checked 2026-10-25.

### §4–§7 (forward process, training, scores, DDPM sampling)
- Ho, J., Jain, A. & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. arXiv:2006.11239. https://arxiv.org/abs/2006.11239 ; code https://github.com/hojonathanho/diffusion (`scripts/run_cifar.py`, `diffusion_tf/nn.py`)
- Nichol, A. & Dhariwal, P. (2021). Improved Denoising Diffusion Probabilistic Models. arXiv:2102.09672.
- Lin, S., Liu, B., Li, J. & Yang, X. (2023). Common Diffusion Noise Schedules and Sample Steps are Flawed. arXiv:2305.08891.
- Salimans, T. & Ho, J. (2022). Progressive Distillation for Fast Sampling of Diffusion Models. arXiv:2202.00512.
- Song, Y. & Ermon, S. (2019). Generative Modeling by Estimating Gradients of the Data Distribution. arXiv:1907.05600.
- Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S. & Poole, B. (2021). Score-Based Generative Modeling through Stochastic Differential Equations. arXiv:2011.13456.
- Efron, B. (2011). Tweedie's Formula and Selection Bias. JASA 106(496). PDF https://efron.ckirby.su.domains/papers/2011TweediesFormula.pdf
- Vincent, P. (2011). A Connection Between Score Matching and Denoising Autoencoders. Neural Computation 23(7):1661–1674 (not opened).
- Hyvärinen, A. (2005). Estimation of Non-Normalized Statistical Models by Score Matching. JMLR 6:695–709 (not opened).

### §8 (U-Net)
- Ronneberger, O., Fischer, P. & Brox, T. (2015). U-Net. arXiv:1505.04597.
- Dhariwal & Nichol 2021 (above), §3, Tables 1–3.
- CompVis Stable Diffusion README and `configs/stable-diffusion/v1-inference.yaml`: https://github.com/CompVis/stable-diffusion
- Podell, D. et al. (2023). SDXL. arXiv:2307.01952 (Table 1).
- Hugging Face safetensors headers (2026-10-25): `stable-diffusion-v1-5/stable-diffusion-v1-5` (unet, text_encoder, vae), `stabilityai/stable-diffusion-xl-base-1.0`, `stabilityai/stable-diffusion-xl-refiner-1.0`, `lllyasviel/sd-controlnet-canny`; HF API metadata for FLUX.1, SD 3.5, SD-Turbo, SDXL-Turbo, CLIP ViT-L/14.

### §9–§10 (DDIM, solvers)
- Song, J., Meng, C. & Ermon, S. (2020). Denoising Diffusion Implicit Models. arXiv:2010.02502.
- Karras, T., Aittala, M., Aila, T. & Laine, S. (2022). Elucidating the Design Space of Diffusion-Based Generative Models. arXiv:2206.00364.
- Lu, C. et al. (2022). DPM-Solver. arXiv:2206.00927. — (2022). DPM-Solver++. arXiv:2211.01095.
- Zhao, W. et al. (2023). UniPC. arXiv:2302.04867.
- Liu, L., Ren, Y., Lin, Z. & Zhao, Z. (2022). Pseudo Numerical Methods for Diffusion Models on Manifolds (PLMS). arXiv:2202.09778.
- Song, Y., Dhariwal, P., Chen, M. & Sutskever, I. (2023). Consistency Models. arXiv:2303.01469 (Table 1 baselines).

### §11–§12 (flow matching, distillation)
- Liu, X., Gong, C. & Liu, Q. (2022). Flow Straight and Fast. arXiv:2209.03003.
- Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M. & Le, M. (2022). Flow Matching for Generative Modeling. arXiv:2210.02747.
- Albergo, M. S. & Vanden-Eijnden, E. (2022). Building Normalizing Flows with Stochastic Interpolants. arXiv:2209.15571. Albergo, Boffi & Vanden-Eijnden (2023). Stochastic Interpolants: A Unifying Framework. arXiv:2303.08797.
- Esser, P. et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis. arXiv:2403.03206.
- Black Forest Labs (2024). Announcing Black Forest Labs / FLUX.1. https://bfl.ai/blog/24-08-01-bfl ; model cards https://huggingface.co/black-forest-labs ; diffusers `transformer_flux.py`, `pipeline_flux.py` (v0.42).
- Salimans & Ho 2022 (above); Song et al. 2023 (above); Song & Dhariwal (2023). Improved Techniques for Training Consistency Models. arXiv:2310.14189; Lu & Song (2024). sCM. arXiv:2410.11081; Geng et al. (2025). Mean Flows. arXiv:2505.13447.
- Luo, S. et al. (2023). Latent Consistency Models. arXiv:2310.04378; LCM-LoRA arXiv:2311.05556.
- Sauer, A., Lorenz, D., Blattmann, A. & Rombach, R. (2023). Adversarial Diffusion Distillation. arXiv:2311.17042. Sauer et al. (2024). LADD. arXiv:2403.12015. Stability AI SDXL Turbo post (via helper).
- Lin, S., Wang, A. & Yang, X. (2024). SDXL-Lightning. arXiv:2402.13929.
- Yin, T. et al. (2023). DMD. arXiv:2311.18828; (2024). DMD2. arXiv:2405.14867.
- Meng, C. et al. (2022). On Distillation of Guided Diffusion Models. arXiv:2210.03142.

### §13–§16 (conditioning, CFG, text, editing)
- Dhariwal & Nichol 2021 (Table 4, Algs. 1–2).
- Ho, J. & Salimans, T. (2021/2022). Classifier-Free Diffusion Guidance. arXiv:2207.12598.
- Nichol, A. et al. (2021). GLIDE. arXiv:2112.10741.
- Kynkäänniemi, T. et al. (2024). Applying Guidance in a Limited Interval. arXiv:2404.07724. Karras, T. et al. (2024). Guiding a Diffusion Model with a Bad Version of Itself. arXiv:2406.02507.
- diffusers pipelines (v0.42): `pipeline_stable_diffusion.py`, `pipeline_stable_diffusion_xl.py`, `pipeline_stable_diffusion_3.py`, `pipeline_flux.py` (defaults; `rescale_noise_cfg`).
- Radford, A. et al. (2021). CLIP. arXiv:2103.00020; `openai/clip-vit-large-patch14` config.
- Saharia, C. et al. (2022). Imagen. arXiv:2205.11487. `google/t5-v1_1-xxl` config (T5 count derivation).
- Ramesh, A. et al. (2022). DALL·E 2 (unCLIP). arXiv:2204.06125.
- Hertz, A. et al. (2022). Prompt-to-Prompt. arXiv:2208.01626.
- Meng, C. et al. (2021). SDEdit. arXiv:2108.01073.
- Zhang, L., Rao, A. & Agrawala, M. (2023). ControlNet. arXiv:2302.05543.
- Mou, C. et al. (2023). T2I-Adapter. arXiv:2302.08453.
- Hu, E. J. et al. (2021). LoRA. arXiv:2106.09685.
- Ruiz, N. et al. (2022). DreamBooth. arXiv:2208.12242.
- Gal, R. et al. (2022). Textual Inversion. arXiv:2208.01618.
- Ye, H. et al. (2023). IP-Adapter. arXiv:2308.06721.

### §17–§18 (latent diffusion, transformers)
- Rombach, R., Blattmann, A., Lorenz, D., Esser, P. & Ommer, B. (2021). High-Resolution Image Synthesis with Latent Diffusion Models. arXiv:2112.10752.
- CompVis README and config (above); Stability AI release posts and model cards for SD 1.4–3.5, SDXL, SDXL Turbo, Stable Cascade, licences (via helper, URLs in `helper_mac_models.md` S22–S25).
- Podell et al. 2023 (SDXL).
- Peebles, W. & Xie, S. (2022). Scalable Diffusion Models with Transformers. arXiv:2212.09748.
- Bao, F. et al. (2022). All are Worth Words (U-ViT). arXiv:2209.12152.
- Chen, J. et al. (2023). PixArt-α. arXiv:2310.00426 (v1 and v3 abstracts).
- Yu, S. et al. (2024). REPA. arXiv:2410.06940. Karras, T. et al. (2023). EDM2. arXiv:2312.02696. Tian, K. et al. (2024). VAR. arXiv:2404.02905.

### §19 (evaluation)
- Heusel, M. et al. (2017). GANs Trained by a Two Time-Scale Update Rule… (FID). arXiv:1706.08500.
- Salimans, T. et al. (2016). Improved Techniques for Training GANs (IS). arXiv:1606.03498.
- Kynkäänniemi, T. et al. (2019). Improved Precision and Recall Metric. arXiv:1904.06991.
- Chong, M. J. & Forsyth, D. (2019). Effectively Unbiased FID and Inception Score. arXiv:1911.07023.
- Parmar, G., Zhang, R. & Zhu, J.-Y. (2021). On Aliased Resizing and Surprising Subtleties in GAN Evaluation. arXiv:2104.11222.
- Kynkäänniemi, T. et al. (2022). The Role of ImageNet Classes in FID. arXiv:2203.06026.
- Jayasumana, S. et al. (2024). Rethinking FID (CMMD). arXiv:2401.09603.
- Hessel, J. et al. (2021). CLIPScore. arXiv:2104.08718.
- Kirstain, Y. et al. (2023). Pick-a-Pic / PickScore. arXiv:2305.01569. Xu, J. et al. (2023). ImageReward. arXiv:2304.05977. Wu, X. et al. (2023). HPS v2. arXiv:2306.09341.

### §20 (data and alignment)
- Schuhmann, C. et al. (2022). LAION-5B. arXiv:2210.08402. LAION (2024). Releasing Re-LAION-5B. https://laion.ai/blog/relaion-5b/
- Betker, J. et al. (2023). Improving Image Generation with Better Captions (DALL·E 3). https://cdn.openai.com/papers/dall-e-3.pdf ; DALL·E 3 System Card (via helper).
- Carlini, N. et al. (2023). Extracting Training Data from Diffusion Models. arXiv:2301.13188.
- Wallace, B. et al. (2023). Diffusion Model Alignment Using Direct Preference Optimization. arXiv:2311.12908.
- Black, K. et al. (2023). Training Diffusion Models with Reinforcement Learning (DDPO). arXiv:2305.13301.

### §21–§22 (video, temporal consistency, world models) — via helper `helper_video.md` unless noted
- Ho, J. et al. (2022). Video Diffusion Models. arXiv:2204.03458. — (2022). Imagen Video. arXiv:2210.02303.
- Singer, U. et al. (2022). Make-A-Video. arXiv:2209.14792.
- Blattmann, A. et al. (2023). Align your Latents. arXiv:2304.08818. — (2023). Stable Video Diffusion. arXiv:2311.15127; Stability post 21 Nov 2023; SVD/SVD-XT model cards.
- OpenAI (2024). Video generation models as world simulators. https://openai.com/index/video-generation-models-as-world-simulators/ (Wayback capture 20250105124831); "Sora is here" (9 Dec 2024); "Sora 2 is here" (30 Sep 2025); API deprecations page (re-checked directly 2026-10-25).
- Meta (2024). Movie Gen: A Cast of Media Foundation Models. arXiv:2410.13720.
- Google: Veo (14 May 2024), Veo 2 (16 Dec 2024), Veo 3 (20 May 2025) blog posts and Veo 3 Tech Report; Veo 3.1 developer blog and Cloud model page; Gemini API deprecations; Gemini Omni posts (URLs in helper file).
- Kong, W. et al. (2024). HunyuanVideo. arXiv:2412.03603; HunyuanVideo 1.5 arXiv:2511.18870.
- Wan Team (2025). Wan. arXiv:2503.20314; GitHub Wan-Video/Wan2.1, Wan2.2.
- HaCohen, Y. et al. (2024). LTX-Video. arXiv:2501.00103; LTX-2 arXiv:2601.03233.
- Yang, Z. et al. (2024). CogVideoX. arXiv:2408.06072. Genmo Mochi 1 blog and README. Open-Sora 2.0 arXiv:2503.09642.
- Seedance 1.0 arXiv:2506.09113; MAGI-1 arXiv:2505.13211; SkyReels-V2 arXiv:2504.13074; Seedance 2.0 arXiv:2604.14148; Wan-Streamer arXiv:2606.25041; Runway Gen-4.5 page; Kling 3.0/4.0 releases; Alibaba Cloud Wan 3.0 blog.
- Yin, S. et al. (2023). NUWA-XL. arXiv:2303.12346. Ruhe, D. et al. (2024). Rolling Diffusion Models. arXiv:2402.09470. Chen, B. et al. (2024). Diffusion Forcing. arXiv:2407.01392. Yin, T. et al. (2024). CausVid. arXiv:2412.07772. Huang, X. et al. (2025). Self Forcing. arXiv:2506.08009.
- Bruce, J. et al. (2024). Genie. arXiv:2402.15391. DeepMind Genie 2 (4 Dec 2024) and Genie 3 (5 Aug 2025) blogs; Project Genie (29 Jan 2026). Valevski, D. et al. (2024). Diffusion Models Are Real-Time Game Engines (GameNGen). arXiv:2408.14837. Oasis project page and open-oasis README.

### §23–§24 (Mac, pipeline) — via helper `helper_mac_models.md` unless noted
- Apple. ml-stable-diffusion README (initial commit 29 Nov 2022; WWDC23 update; current). https://github.com/apple/ml-stable-diffusion ; Apple ML Research (6 Dec 2022). Stable Diffusion with Core ML on Apple Silicon. https://machinelearning.apple.com/research/stable-diffusion-coreml-apple-silicon
- Hugging Face blogs: Faster Stable Diffusion with Core ML (15 Jun 2023); SDXL on Mac with Advanced Core ML Quantization (27 Jul 2023); Diffusers welcomes SD3 (12 Jun 2024); Exploring Quantization Backends in Diffusers (21 May 2025); diffusers MPS and memory docs.
- ml-explore/mlx-examples `stable_diffusion` (README versions 2024-02-01 and current) and `flux`.
- argmaxinc/DiffusionKit README (archived). filipstrand/mflux README (versions 2024-11-09, 2026-01-04, current) and model READMEs.
- Draw Things engineering blog (7 Jan 2025). Replicate blog (18 Jun 2024; secondary).
- Stability AI SDXL Turbo post (28 Nov 2023), SD3 research-paper post (5 Mar 2024).
- Safetensors header counts and `derive.py` (this file's §8, §15, §17 sources).
