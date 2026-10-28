"""Per-episode notes for "Out of the Noise" briefs: number → (slug, ClassName, [reference episodes],
research-notes sections, notes). Reference keys: aiNN / inNN / rlNN (sister courses), dfNN (this course)."""

T = "research/experiments/toy.md (the swirl; load with measured('toy', …); headline keys in toy.json)"
G = ("research/experiments/mac_runs.md (the digit model and the lighthouse; load with measured('mac_runs', …); "
     "images in research/experiments/images/, arrays in research/experiments/arrays/)")

SPEC = {
    # ------------------------------------------------------------------ Module 1 · Pictures from Noise
    1: ("out-of-the-noise", "OutOfTheNoise", ["rl01", "ai35", "in02"], "§0, §1 (all), §24",
        "The series OPENER. Hook: the lighthouse appearing out of pure static, frame by frame, from our real run on this Mac (" + G +
        ": the decoded x̂₀ previews / step images) — say 'generated on this Mac' with the model and steps. The idea in one "
        "sentence: destroying a picture with noise is easy; a network learns to undo it a little at a time; repeat. The "
        "denoise_loop (kit) turning. Generation vs recognition (a classifier maps a picture to a label; a generator goes the "
        "other way). A dated timeline strip from notes §1 (VAEs 2013, GANs 2014, diffusion 2015 (Sohl-Dickstein), DDPM 2020, "
        "Stable Diffusion paper Dec 2021 / public release Aug 2022, video diffusion from 2022; cite years). Then the three running examples as cards (the swirl, the digit model, the lighthouse) and the course map: six "
        "module cards in the module colours. End on Episode 2's question: what is a picture to a computer? Link 'AI, Explained' "
        "Episode 35 on screen."),
    2: ("images-are-numbers", "ImagesAreNumbers", ["rl02", "in02", "df01"], "§2 (all)",
        "A real MNIST digit from the digit model's training data (" + G + ") zoomed until its 28×28 pixels are numbers, "
        "unrolled into a column of 784 values; a colour picture as three channels (the lighthouse: 512×512×3 = 786,432 numbers "
        "— compute). Random points in that space are static (noise_image) — the chance of hitting a digit by luck is "
        "astronomically small (say why, don't invent a number). The data distribution: real images are a thin, structured "
        "set (the manifold idea) — the swirl as the 2D stand-in: points on a spiral vs uniform noise in the square. MNIST facts "
        "from notes §2 (60k/10k, 28×28, license). End: we need a way to walk from noise to the thin set (Episodes 3–4)."),
    3: ("before-diffusion", "BeforeDiffusion", ["rl03", "ai35", "df02"], "§3 (all), §1",
        "Earlier generators and their trade-offs. Autoencoders → VAEs (blurry averages: show why averaging two digits gives a "
        "blur, computed on real MNIST arrays); GANs: generator vs critic, sharp but unstable; mode collapse shown on the swirl "
        "(a 'collapsed generator' covering one arm — illustrative, label it); autoregressive pixel models (slow). The moment "
        "diffusion won: Dhariwal & Nichol 2021 'Diffusion Models Beat GANs' — FID table numbers from notes §3 (exact table and "
        "row). Pronunciation: say 'a gan' as the lexicon allows (see the pronunciation notes); never start a line with GAN."),
    4: ("adding-noise", "AddingNoise", ["rl05", "in13", "df02"], "§4 (all)",
        "The forward process. The swirl melting into a Gaussian blob as t grows (q_sample with the linear schedule; snapshots "
        "from " + T + "); a real digit dissolving at t = 0, 50, 100, 250, 500, 1000 (use a digit from the run's arrays and "
        "diffusion_reference.q_sample). The closed form x_t = √ᾱ x₀ + √(1−ᾱ) ε — why we can jump to any t in one step "
        "(adding Gaussians gives a Gaussian). The schedule: DDPM's linear β 1e-4 → 0.02, T = 1000; ᾱ_t at t = 100/250/500/750/1000 "
        "for linear vs cosine (Nichol & Dhariwal 2021) computed and plotted (Plot); the signal-to-noise ratio on a log axis. "
        "Close Module 1: now learn to go backwards (Module 2)."),

    # ------------------------------------------------------------------ Module 2 · Learning to Denoise
    5: ("learning-to-undo-noise", "LearningToUndoNoise", ["rl16", "in13", "df04"], "§5 (all)",
        "Training is a regression. One training step animated: pick a real digit, pick t, draw ε, make x_t, the network "
        "predicts ε̂, loss = ‖ε − ε̂‖² (Ho et al. 2020, eq. 14, Algorithm 1 — show the ≤ 8-line loop). Why predict the noise "
        "(the target has the same scale at every t); x₀- and v-prediction in one line each (Salimans & Ho 2022). The digit "
        "model's training on this Mac (" + G + "): loss curve, steps, batch, wall time, parameter count; x̂₀ = (x_t − √(1−ᾱ) ε̂)/√ᾱ "
        "for a real noisy digit at a few t (the model's guess of the clean picture — blurrier at high t). DDPM's CIFAR-10 "
        "numbers from notes §5 (FID 3.17, IS 9.46)."),
    6: ("the-score", "TheScore", ["rl09", "df05", "df04"], "§6 (all)",
        "The score ∇log p(x): an arrow field pointing toward more-likely points — on the swirl, computed EXACTLY with "
        "diffusion_reference.mixture_score at several noise levels (Plot.field in LIME): sharp near the spiral at low noise, "
        "smooth at high noise. Langevin dynamics: particles following the score plus a little noise (langevin_sample) slide onto "
        "the spiral; why one noise level fails and annealing works (" + T + "). The link to Episode 5: ε̂ = −σ·score "
        "(denoising score matching, Vincent 2011; Tweedie's formula) — predicting the noise IS estimating the score. Song & "
        "Ermon 2019 and Song et al. 2021 (the SDE view) in one card each. Pronunciation: never start a line with Tweedie or "
        "Langevin."),
    7: ("sampling-step-by-step", "SamplingStepByStep", ["rl10", "df05", "df06"], "§7 (all)",
        "DDPM ancestral sampling (Ho et al. 2020, Algorithm 2 — the ≤ 6-line loop): start from pure noise, predict ε, take one "
        "step back, add a little fresh noise, repeat 1,000 times. The swirl un-melting step by step (ddpm_sample trajectory "
        "with the exact denoiser and with the trained MLP — " + T + "). A real digit emerging over 1,000 steps from the digit "
        "model (" + G + ": the saved trajectory; strip of x_t and of x̂₀ predictions). Why fresh noise each step (stochastic "
        "paths: same start, different digits — if the run recorded two seeds/paths, show it). The cost: 1,000 network calls "
        "per picture — measured time per sample batch on this Mac (" + G + ") — the problem Module 3 attacks."),
    8: ("the-unet", "TheUNet", ["in13", "df05", "df07"], "§8 (all)",
        "The denoiser's architecture: a picture shrinking through the encoder (28 → 14 → 7 for our digits), a bottleneck, "
        "growing back through the decoder, skip connections carrying detail across (kit UNet with .flow()). The digit model's "
        "real architecture and parameter count from " + G + " (channels per level, attention where, time embedding "
        "dimension) — label every block. The time embedding: sinusoidal timestep_embedding (diffusion_reference) — show a few "
        "embedding vectors as heat strips; it enters every block. Attention in one line (lets distant pixels talk). Sizes "
        "for scale from notes §8: DDPM's CIFAR-10 U-Net (~35.7M), Stable Diffusion 1.x's U-Net (~860M). Close Module 2."),

    # ------------------------------------------------------------------ Module 3 · Faster Sampling
    9: ("fewer-steps", "FewerSteps", ["in13", "df07", "df06"], "§9 (all), §6 (probability-flow ODE)",
        "DDIM (Song, Meng & Ermon 2020): the same trained model sampled deterministically (η = 0) with far fewer steps. "
        "On the swirl: the same noise point following a smooth path to the same spiral point (ddim_sample trajectories), and "
        "sample quality vs steps (" + T + "). On the digit model: the SAME seed sampled with DDPM 1,000 vs DDIM 100/50/20/10/5 "
        "steps — sample grids, quality numbers (the run's FID-like score and conditional accuracy — say it is a classifier-"
        "based FID, not Inception FID) and wall time per batch, measured (" + G + "). DDIM's own CIFAR-10 FID-vs-steps table "
        "from notes §9. Deterministic = invertible: inversion in one line (Episode 16 uses it)."),
    10: ("solvers-and-schedules", "SolversAndSchedules", ["in13", "rl09", "df09"], "§10 (all)",
         "Sampling is numerical integration of an ODE. Euler vs Heun on the swirl's probability-flow ODE (euler_ode / heun_ode "
         "with the exact denoiser): paths with big steps cutting corners, error vs number of network evaluations (NFE) "
         "measured (" + T + "); 'steps' vs 'NFE' (Heun costs two per step). Step placement: Karras/EDM's σ schedule (ρ = 7) "
         "puts steps where they matter (karras_sigmas plotted); DPM-Solver and DPM-Solver++ (Lu et al. 2022: good images in "
         "10–20 steps) in one card. The lighthouse at different step counts from our run (" + G + ") as the real-world "
         "version. EDM's CIFAR-10 FID 1.79 at 35 NFE (notes §10)."),
    11: ("flow-matching", "FlowMatching", ["rl18", "df09", "df10"], "§11 (all)",
         "Flow matching / rectified flow: draw a straight line from each data point to a noise point, learn the velocity "
         "(ε − x₀) — sampling follows a velocity field (LIME). On the swirl: curved diffusion paths vs nearly straight flow "
         "paths, path straightness and quality vs Euler steps 1/2/4/8/16/32 (" + T + ": the flow-trained MLP vs the "
         "DDPM-trained one). On the digit model: our flow-matching model vs the DDPM model at few steps, measured (" + G + "). "
         "Who uses it: Stable Diffusion 3 (Esser et al. 2024: rectified-flow transformers, 8B) and FLUX.1 (Black Forest Labs "
         "2024) — facts from notes §11, dated. Lipman et al. 2022 and Liu et al. 2022 cited."),
    12: ("one-step", "OneStepAtATime", ["in13", "df11", "df09"], "§12 (all)",
         "Distillation: a student learns to do in one step what the teacher does in many. Progressive distillation "
         "(Salimans & Ho 2022: halving steps repeatedly — a folding animation 64 → 32 → … → 1), consistency models "
         "(Song et al. 2023: any point on a path maps to its end), latent consistency models (Luo et al. 2023), adversarial "
         "diffusion distillation / SDXL Turbo (Sauer et al. 2023: 1–4 steps). If our run has a turbo/distilled model (" + G +
         "): its 1/2/4-step lighthouses vs the multi-step model's 50 steps, with timings — the honest trade-off (diversity, "
         "detail). Otherwise show the swirl's 1-step flow sample as the limit case and cite the papers' numbers. Close Module 3."),

    # ------------------------------------------------------------------ Module 4 · Steering the Picture
    13: ("asking-for-a-digit", "AskingForADigit", ["df07", "df08", "rl04"], "§13 (all)",
         "Conditioning: the denoiser gets an extra input — the label. The SAME initial noise becoming 0, 1, … 9 depending on "
         "the label, from our digit model (" + G + ": one seed, ten labels — a 2×5 grid emerging together). How the label "
         "enters (a learned embedding added to the time embedding — use the run's actual architecture). Conditional accuracy "
         "of generated digits measured with the run's classifier. Classifier guidance (Dhariwal & Nichol 2021): push along "
         "the gradient of a classifier's log-probability — its guidance-scale trade-off from notes §13; on the swirl's "
         "labelled toy with exact scores (" + T + ") — the arrow field bending toward the requested arm."),
    14: ("classifier-free-guidance", "ClassifierFreeGuidance", ["df13", "in13", "rl14"], "§14 (all)",
         "Classifier-free guidance (Ho & Salimans 2022): train with the label dropped 10% of the time, then extrapolate "
         "ε_u + w(ε_c − ε_u) (cfg in diffusion_reference — show both conventions and say which our run uses). On the swirl's "
         "labelled toy with exact scores: samples at w = 0, 1, 3, 7 — diversity shrinking, samples pushed beyond the data at "
         "high w (" + T + "). On the digit model: grids at w = 0 … 8 with the measured quality/diversity numbers (" + G + "). "
         "The lighthouse at guidance 1, 3, 5, 7.5, 10, 15 from our run — oversaturation at high scale. Cost: two network "
         "passes per step. Typical defaults (SD 7.5) from notes §14."),
    15: ("words-to-pictures", "WordsToPictures", ["df14", "ai24", "rl20"], "§15 (all)",
         "Text conditioning: the prompt is tokenised and encoded (CLIP, Radford et al. 2021: 400M image-text pairs, contrastive "
         "— matching pairs pulled together, others pushed apart, animated); the denoiser attends to the text through "
         "cross-attention (queries from image positions, keys/values from words — the mechanism drawn on a small grid). "
         "Real sizes from notes §15 and our run's pipeline (" + G + "): SD's CLIP ViT-L/14 77 tokens × 768 (or whatever our "
         "model uses — check), SDXL's two encoders, Imagen's finding that scaling the text encoder matters, SD3's three "
         "encoders. The lighthouse prompt word by word (token_row in PINK); prompt sensitivity shown with our run's images if "
         "it has a second prompt, otherwise cited."),
    16: ("editing-and-control", "EditingAndControl", ["df15", "df09", "rl21"], "§16 (all)",
         "The same machinery edits. SDEdit / image-to-image: add noise part-way (strength), then denoise with a new prompt — "
         "the lighthouse repainted at strengths 0.3/0.5/0.7/0.9 from our run (" + G + ") and why low strength keeps the "
         "layout; on the digit model: a '3' noised to t and denoised as an '8' (if the run has it; else the swirl). "
         "Inpainting (only the masked region is regenerated), ControlNet (Zhang et al. 2023: a trainable copy with zero "
         "convolutions, conditions like edges/pose), LoRA (low-rank adapters, sizes) and DreamBooth (Ruiz et al. 2023) in one "
         "card each — facts from notes §16. Close Module 4."),

    # ------------------------------------------------------------------ Module 5 · Scaling Up
    17: ("latent-diffusion", "LatentDiffusion", ["df08", "in23", "df15"], "§17 (all)",
         "Latent diffusion (Rombach et al. 2022): an autoencoder squeezes 512×512×3 into 64×64×4 (factor 8 per side, 48× "
         "fewer numbers — compute), diffusion happens there, the decoder paints the pixels. Our lighthouse run's latents "
         "decoded at each step (" + G + ": x̂₀ previews) and the latent shape confirmed; show a latent's 4 channels as tinted "
         "pictures. Why it's cheaper (compute per step scales with positions). Stable Diffusion's releases and sizes from "
         "notes §17 (dated), training data LAION subsets in one line (detail in Episode 20)."),
    18: ("diffusion-transformers", "DiffusionTransformers", ["df17", "df08", "in13"], "§18 (all)",
         "Diffusion transformers: cut the latent into patches, treat them as tokens, a transformer denoises; adaLN-Zero "
         "conditioning on t and the label. DiT (Peebles & Xie 2023): FID vs Gflops scaling, DiT-XL/2 FID 2.27 on ImageNet "
         "256 (notes §18). MM-DiT (SD3) and FLUX in one card each. If our run trained a tiny DiT on digits (" + G + "), "
         "compare it honestly with the U-Net at the same budget; otherwise say we didn't and keep to the cited results. Patch "
         "size vs number of tokens computed (64×64 latent, patch 2 → 1,024 tokens)."),
    19: ("measuring-pictures", "MeasuringPictures", ["in02", "rl24", "df18"], "§19 (all)",
         "How do you score a generator? FID (Heusel et al. 2017): compare the mean and covariance of features of real vs "
         "generated images — frechet_distance from diffusion_reference on two 2D Gaussians first (exact), then our digit "
         "model's classifier-feature 'FID-like' score vs steps and vs guidance (" + G + ") — say clearly it is not Inception "
         "FID and not comparable to published numbers. FID's small-sample bias (" + T + "). Precision/recall, CLIP score, "
         "human preference (notes §19) — what each misses (e.g., FID rewards matching statistics, not following the prompt)."),
    20: ("data-and-alignment", "DataAndAlignment", ["rl21", "rl25", "df19"], "§20 (all)",
         "A model learns what its data shows. Web-scale image–text data (LAION-5B: 5.85B pairs; the 2023 takedown and "
         "re-release, dated, neutral), aesthetic filtering, recaptioning (DALL·E 3, Betker et al. 2023: dense synthetic "
         "captions improve prompt following — their numbers). Memorization: Carlini et al. 2023's extraction numbers; our "
         "own check — generated digits vs their nearest training neighbours, measured (" + G + "). Preference tuning for "
         "images (Diffusion-DPO, Wallace et al. 2023; DDPO, Black et al. 2023) — link 'Trial and Reward' Episodes 21–22. "
         "Keep it technical and neutral. Close Module 5."),

    # ------------------------------------------------------------------ Module 6 · Video and Your Mac
    21: ("pictures-to-video", "FromPicturesToVideo", ["df17", "df18", "in13"], "§21 (all)",
         "A video is a stack of frames: diffuse the whole cube (frames × height × width × channels) together. 3D U-Nets / "
         "factorised space-time attention (Ho et al. 2022), latent video, spacetime patches (Sora technical report, Feb 2024 "
         "— quote carefully, no capability claims beyond the report), Movie Gen and later open models — sizes, frames, "
         "resolution, compute from notes §21 (dated). Our own tiny moving-digit clip if the run made one (" + G + "); "
         "otherwise a computed illustration labelled as such. Cost: tokens grow with frames (compute patches for a few clip "
         "sizes)."),
    22: ("keeping-frames-together", "KeepingFramesTogether", ["df21", "df14", "rl11"], "§22 (all)",
         "The hard part of video is consistency: frames denoised independently flicker (different details every frame), "
         "frames denoised jointly with temporal attention hold together — shown on our moving digit (" + G + ": independent "
         "vs joint) or on the swirl (a toy where each frame's samples are independent vs shared noise — computed and "
         "labelled). Temporal attention, frame interpolation, long-video tricks (rolling / autoregressive diffusion), world "
         "models in one card each (notes §22, dated)."),
    23: ("diffusion-on-your-mac", "DiffusionOnYourMac", ["in23", "in02", "df17"], "§23 (all), §17",
         "What it takes to run a text-to-image model locally, all measured on this Mac (" + G + "): model sizes (parameters, "
         "GB on disk, dtype), peak memory, the timing breakdown of one generation (text encoder once, denoiser × steps × 2 "
         "with guidance, VAE decode once) as a Timeline-style bar, steps × time-per-step = total, the turbo model's 1-step "
         "time if measured. Published Apple-silicon numbers (Core ML / MLX / DiffusionKit / mflux) from notes §23 (dated, "
         "cited). The digit model's training time and sampling speed as the small-scale counterpart. Link 'Tokens per Second' "
         "Episode 23."),
    24: ("revisited", "OutOfTheNoiseRevisited", ["rl26", "in24", "df01"], "§24 (all)",
         "The FINALE: the whole pipeline in one picture, with the three running examples emerging side by side — the swirl "
         "(exact sampler), a digit from our digit model (" + G + "), and the lighthouse from our run — each from pure noise. "
         "One generation, every stage, with real sizes (prompt → tokens → embeddings → 64×64×4 latent noise → N denoiser calls "
         "× 2 for guidance → VAE decode → 512×512×3; notes §24) computed and asserted. Then the course revisited: one card per "
         "module with its key idea and episode numbers (six module colours). A three-word recap. Override series_outro() with "
         "a course-closing card: 'That's Out of the Noise', the six module names in their colours, Pixel, 'Thanks for "
         "watching!' (see /Users/vpromisex/videos/rl-series/ep26-grpo-on-this-mac/scene.py for the sister course's finale)."),
}

# ---------------------------------------------------------------------------- what the swirl runs actually showed
# (toy.md, 2026-10-25; keys from measured('toy', 'headline', …)). Appended to each episode's notes.
TOY_FINDINGS = {
    4: "SWIRL FACTS (toy.json headline): ᾱ_500 = 0.0786 (linear) vs 0.494 (cosine); SNR = 1 at t ≈ 259 (linear) vs 496 "
       "(cosine); ᾱ_1000 = 4.0e-5; at t = 500 even a perfect classifier names the right arm only 52.8% of the time.",
    5: "SWIRL FACTS: the tiny MLP (45,954 parameters, Fourier features of x — a plain MLP blurred the score) reaches loss "
       "0.1588 against an exact floor of 0.1581; score error 15% at t = 10, 0.8% at t = 500. One DDIM step from pure "
       "noise with the ε-model lands 2.3 away from the swirl because dividing by √ᾱ_1000 ≈ 0.0064 magnifies errors ~157× "
       "— a good reason for x₀- and v-prediction targets (and Episode 11's flow model, whose one step is fine).",
    6: "SWIRL FACTS: plain Langevin at one noise level puts 80% of particles on the outer half of the arms; annealed "
       "Langevin puts 48% (the data: 50%).",
    7: "SWIRL FACTS: DDPM with 1,000 steps reaches swirl distance 0.0245 (exact denoiser) / 0.0253 (MLP) vs the data's "
       "0.024. DDPM forgets its starting noise: two DDPM runs from the same noise land on the same arm only 49.9% of the "
       "time, 0.90 apart — like two random points (say this; it sets up Episode 9's determinism).",
    9: "SWIRL FACTS: DDIM strands samples between the arms as steps drop: 2.6% at 100 steps, 7.5% at 50, 31% at 20, 45% at "
       "10 (MLP). DDIM-100 and DDIM-50 land on the same arm 97% of the time (deterministic), vs 49.9% for two DDPM runs. "
       "DDIM-100 is 'too tidy' (distance 0.020, below the data's 0.024) — always pair mean distance with the off-swirl "
       "fraction. Identity (exact, 1e-9): DDIM with η = 1 is DDPM with β̃; DDIM with η = 0 is Euler in σ.",
    10: "SWIRL FACTS (be careful): Euler on a Karras grid at 20 steps strands 4.8% (vs DDIM-20's 31%); Heun on a Karras "
        "grid is best from about 49 model calls (0.8% stranded) but DIVERGES at 9 or fewer calls (lands 0.79 and 4.8 from "
        "the swirl) — show both. Euler in DDPM time is NOT worse than DDIM on this toy, so do not claim the σ coordinate "
        "is what makes DDIM good; the step PLACEMENT is what matters here.",
    11: "SWIRL FACTS — CHANGES THE STORY: flow matching's TRAINING paths are straight (length ÷ chord 1.0 vs 1.22 for the "
        "DDPM noising arcs), but the SAMPLING paths of the learned (averaged) velocity field are curved too (1.56 for flow "
        "vs 1.08–1.14 for the DDPM ODE, each in its own coordinates). With perfect models, flow Euler is exactly DDIM with "
        "steps placed differently — its per-step advantage here comes from step placement: stranded 25% / 7% / 2% (flow) vs "
        "47% / 36% / 16% (DDIM) at 8 / 16 / 32 steps. Show 'straight' with the training pairs, show the learned paths "
        "bending honestly, and cite rectified flow's reflow (Liu et al. 2022) as the step that actually straightens "
        "sampling paths (we did not run reflow).",
    12: "SWIRL FACTS: one Euler step of an UNDISTILLED (exact) flow model lands every sample on the data's mean — the 'blurry "
        "average' that distillation exists to avoid; one DDIM step with the ε-model lands 2.3 away. Use these as the "
        "honest motivation for distillation.",
    13: "SWIRL FACTS: at t = 500 (linear) a perfect classifier names the right arm only 52.8% of the time — guidance has "
        "little to grip at high noise.",
    14: "SWIRL FACTS — CHANGES THE STORY: on the swirl's arms guidance only CONCENTRATES samples (arm coverage drops to 0.45 "
        "at w = 7, 98% on the arm's outer half; spread 0.68 → 0.46) because the thin arm catches them again at low noise. "
        "To show samples pushed BEYOND the data use the three-blob toy (blobs_mixture / guided_eps, after Ho & Salimans' "
        "Fig. 2): at w = 7 the class centre moves from height 0.60 to 0.96 and 86% of samples lie beyond 2 std of the class "
        "(real data: 14%). Our cfg convention: ε_u + w(ε_c − ε_u), w = 1 = plain conditional (Ho & Salimans' w′ = w − 1).",
    19: "SWIRL FACTS: FD's small-sample bias in 2D is 0.037 at n = 100 and 0.00035 at n = 10,000; in 64-D it is 1.12 at "
        "n = 1,000. FD's blind spot: a Gaussian with the swirl's mean and covariance scores FD 0.0003 although 48% of its "
        "points are off the swirl — a perfect on-screen demonstration that matching statistics ≠ matching the data.",
}
for _n, _extra in TOY_FINDINGS.items():
    _s = SPEC[_n]
    SPEC[_n] = _s[:4] + (_s[4] + " " + _extra,)

# ---------------------------------------------------------------------------- corrections from research/notes.md (2026-10-26)
NOTES_FIXES = {
    1: "NOTES: VAEs (2013) predate GANs (2014); diffusion models date from Sohl-Dickstein et al. 2015; DDPM made them work "
       "in 2020; the Stable Diffusion paper is Dec 2021 and the public release 22 Aug 2022; video diffusion began in 2022.",
    11: "NOTES: Salimans & Ho's v-prediction is NOT the flow-matching velocity — don't conflate them; papers disagree on "
        "which end of t is noise (our convention: t = 0 data, t = 1 noise — say so on screen).",
    14: "NOTES — ONE CONVENTION ON SCREEN: use the guidance SCALE (1 = no guidance, as Stable Diffusion's 7.5 and our cfg's "
        "w), and convert Ho & Salimans' w by adding 1 (SD's 7.5 is their w = 6.5). Label every number with the convention. "
        "Best FID comes at small guidance (ImageNet 128: 2.43 at Ho & Salimans' w = 0.3); quality/diversity trade-off from §14.",
    21: "NOTES: Sora's report gives no implementation details and calls simulators only 'a promising path' — quote it exactly "
        "from §21 with its own limitations; video diffusion started in 2022 (Video Diffusion Models, Imagen Video, "
        "Make-A-Video). Date every product fact (October 2026).",
    17: "NOTES: SD 1.5 sizes counted from the weight files: U-Net 859,520,964 parameters, CLIP text encoder 123M, VAE 84M, "
        "≈ 2.06 GB fp16; latent scale factor 0.18215; 48× fewer numbers.",
}
for _n, _extra in NOTES_FIXES.items():
    _s = SPEC[_n]
    SPEC[_n] = _s[:4] + (_s[4] + " " + _extra,)

# ---------------------------------------------------------------------------- what the GPU runs actually showed
# (mac_runs.md, 2026-10-26; keys from measured('mac_runs', 'headline', …)). "cFID" = a FID computed on the run's
# MNIST classifier features — NOT Inception FID; real-data floor 2.06, ±0.3. Guidance convention: uncond + w(cond −
# uncond), w = 1 means no guidance (the Stable Diffusion scale).
GPU_FINDINGS = {
    1: "GPU RUN: 30 decoded x̂₀ / x_t frames of the lighthouse in research/experiments/images/lighthouse_frames30/ (SD 2.1-base "
       "via MLX, 50 steps, guidance 7.5, 7.32 s on this Mac); the digit model's 7 emerging (arrays/).",
    5: "GPU RUN: the digit model — a 3,940,801-parameter U-Net trained 10,000 steps × batch 256 (42.7 epochs) with Adam, "
       "EMA 0.999, 10% label dropout, in 31.8 min on this Mac (MLX, peak 6.32 GB); loss 0.81 → 0.0202 (curve in "
       "mac_runs.json). x̂₀ trajectories of a 7 are in arrays/.",
    7: "GPU RUN: DDPM with 1,000 steps: cFID 2.64 (floor 2.06); 4.3 s per digit, 172 s per 1,000 digits on this Mac — the "
       "cost Module 3 attacks. Full x_t / x̂₀ trajectories of a 7 plus six more seeds in arrays/ (each seed has its OWN starting noise; within a seed, DDPM-1000 / DDIM-50 / flow-50 share it).",
    8: "GPU RUN: the real architecture — channels 32/64/128 at 28/14/7 px, 2 ResBlocks down and 3 up, 4-head attention at 7×7 "
       "and in the middle, sinusoidal time embedding → MLP to 128-d plus a label embedding; 3,940,801 parameters; U-Net "
       "activations saved in arrays/ (show real feature maps).",
    9: "GPU RUN (cFID, same model): DDPM-1000 2.64; DDIM 100 / 50 / 20 / 10 / 5 / 2 / 1 steps: 3.50 / 3.35 / 3.30 / 4.57 / "
       "21.6 / 939 / 943 (1 step = speckled noise); time per 1,000 digits 172 s (DDPM-1000) vs 8.9 s (DDIM-50) vs 1.8 s "
       "(DDIM-10). Same-noise DDPM / DDIM / flow trajectories of a 7 in arrays/.",
    11: "GPU RUN: the flow-matching U-Net (same architecture, 29.2 min) vs the DDPM one at few steps (cFID): 10 steps 4.57 vs "
        "4.57; 5 steps 9.3 vs 21.6; 2 steps 85 vs 939; 1 step 886 (a blurry average digit) vs 943 (speckles). Flow wins at "
        "very few steps; at 10+ they tie — say both.",
    12: "GPU RUN: SD-Turbo (distilled from SD 2.1, same UNet) makes a full lighthouse in 1 step in 0.25–0.36 s vs 7.32 s for "
        "SD 2.1 at 50 steps with guidance; 2 / 4 turbo steps 0.32 / 0.47 s. SD 2.1 itself at 1 step is a blur, at 4 a "
        "lighthouse. The digit models at 1 step: speckles (DDPM) / a blurry average (flow) — why distillation exists.",
    13: "GPU RUN: same-noise grids for labels 0–9 in arrays/digit_same_noise_ten_labels.npz; label accuracy (the run's "
        "classifier, 99.45% on test) 98.1% at w = 1, 9.9% (chance) unconditional.",
    14: "GPU RUN (honest!): on the digits guidance raises label accuracy 98.1% → 99.99% and precision 0.83 → 0.89, but recall "
        "falls 0.89 → 0.38, diversity 1.01 → 0.74 and cFID gets WORSE: 3.35 / 21.4 / 42 / 67 / 85 at w = 1 / 2 / 3 / 5 / 8 — on "
        "these digits the best cFID is with no guidance. The lighthouse at 1 … 15: mean saturation 0.19 (w = 1), 0.28 (7.5), "
        "0.36 (15), 9.3% of pixels clip at 15. Guidance costs 1.7× per step (0.141 vs 0.083 s).",
    15: "GPU RUN: our lighthouse model is SD 2.1-base, whose text encoder is OpenCLIP (340.4M parameters) — not SD 1.x's "
        "CLIP ViT-L/14; check the token count and width in mac_runs.md and say which model.",
    16: "GPU RUN (honest): with the prompt '… in winter, snow' NO snow appears at any SDEdit strength — genuine pipeline "
        "behaviour (matches the library), documented in mac_runs.md; a more explicit snow prompt gives the classic "
        "progression (snowy at 0.9, nearly a new picture at 1.0) — show that series and say why prompts matter.",
    17: "GPU RUN: SD 2.1-base sizes — UNet 865.9M, text encoder 340.4M, VAE 83.7M; 512×512×3 → 64×64×4 confirmed (48×); "
        "VAE round-trip PSNR 38.1 dB.",
    18: "GPU RUN (honest): a tiny DiT (patch 2, 4,150,660 parameters, 34.7 min) vs our U-Net at the same budget: cFID 15.9 vs "
        "3.35, label accuracy 94.2% vs 98.1% — the U-Net wins at this small size; DiT's advantage is scaling (cite DiT's "
        "Gflops-vs-FID results).",
    19: "GPU RUN: cFID real-data floor 2.06 (±0.10 across subsets), bootstrap ±0.36; precision/recall at w = 1 / 3 / 8: "
        "0.83/0.89, 0.89/0.70, 0.87/0.38 — a metric that rises while another falls.",
    20: "GPU RUN: memorisation check — median distance from a generated digit to its nearest training digit 4.52, vs 4.34 for "
        "real test digits and 10.1 for two random digits: no sign of copying (at this scale).",
    21: "GPU RUN: our jointly-denoised video toys FAILED (undertrained) and are quarantined — don't show them. Use the "
        "correlated-noise demo only as a pointer to Episode 22.",
    22: "GPU RUN: the trained digit model on 'frames' with correlated starting noise (the PYoCo idea): flicker 0.078 "
        "(independent noise) → 0.054 (ρ 0.5) → 0.024 (ρ 0.9) → 0.007 (ρ 0.99) → 0 (shared). The motion is pasted on, not "
        "generated — say so.",
    23: "GPU RUN: one SD 2.1-base image at 50 steps with guidance 7.5 = 7.32 s on this Mac: text encoder 0.017 s, UNet 7.04 s "
        "(0.141 s per step, 96%), VAE decode 0.265 s; peak 8.1 GB; SD-Turbo 1 step 0.25–0.36 s; median totals 0.75 s (4 "
        "steps), 2.9 s (20), 6.9 s (50). Download 5.5 GB fp16 (both models). Digit model: 31.8 min training, 0.2 s per "
        "digit with DDIM-50.",
    24: "GPU RUN: use the same three sources — the swirl (exact sampler), the digit model's 7 trajectory (arrays/) and the "
        "lighthouse frames (images/lighthouse_frames30/) — with their real timings (7.32 s; 0.2 s per digit at DDIM-50).",
}
for _n, _extra in GPU_FINDINGS.items():
    _s = SPEC[_n]
    SPEC[_n] = _s[:4] + (_s[4] + " " + _extra,)
