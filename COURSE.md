# Out of the Noise — how diffusion models make pictures and video

A video course in 24 short episodes (3.5–4.5 minutes each), narrated and animated. It starts from a single
idea — destroy an image with noise, then learn to undo it one small step at a time — and follows it all the
way to the text-to-image and text-to-video models people use today, with three running examples computed
or trained on this Mac.

- **One idea, many tricks:** adding noise is easy and needs no learning; removing it is a network's job.
  Every episode is either about learning that network (what it predicts, how it is built, how it is trained),
  about sampling with it (how many steps, which solver, how to steer it), or about scaling it (latents,
  transformers, video).
- **Three running examples:**
  - **The swirl** — a cloud of 2D points shaped like a spiral. Noise, scores, flows and samplers are all
    visible on a flat plane; every number is computed in the scene (`diffusion_reference.py`, numpy).
  - **The digit model** — a small diffusion model trained on this Mac on handwritten digits (MNIST),
    class-conditional, so we can watch noise turn into a "7" and turn guidance up and down
    (`research/experiments/`).
  - **The lighthouse** — a real latent text-to-image model run on this Mac with the prompt "a lighthouse on
    a rocky coast at dusk, oil painting": steps, guidance, time and memory measured.
- **Mascot:** Pixel, a small square face that is half noise and half picture (`kit.pixel()`).
- **Measured, not just quoted:** training curves, sample grids, step counts, timings and memory are real runs
  on this Mac Studio (Apple M4 Max, 64 GB). Every other number is cited (`research/notes.md`).
- **Who it's for:** developers who know what a neural network and a gradient are. "AI, Explained" covered
  diffusion in one episode; this course is the full story, from the forward process to video.
- **Visual language:** a dark "darkroom" stage; colours fixed across the course —
  noise VIOLET · data and clean images TEAL · the model / denoiser AMBER · time and the noise schedule BLUE ·
  text and conditioning PINK · scores, velocities and flows LIME · good GREEN · bad RED.
- **Format:** 1080p60 MP4 with captions (.srt) and HTML slides; series kit in `kit.py`.

Status: ✅ produced · 🎬 in production · ○ planned

## Module 1 · Pictures from Noise — violet

| # | Episode | The viewer will understand | Key concepts | Mechanism to animate | Running example | Status |
|---|---|---|---|---|---|---|
| 1 | **Out of the Noise** | A diffusion model turns noise into a picture by removing a little noise many times | generation vs recognition; the denoising loop; a short history (VAEs 2013, GANs 2014, diffusion 2015, DDPM 2020, Stable Diffusion 2022, video from 2022); the course map | a lighthouse appearing out of static in 30 steps; the three running examples as cards | the lighthouse, generated on this Mac, frame by frame | ✅ `ep01-out-of-the-noise` |
| 2 | **Images Are Numbers** | A picture is a point in a space with millions of dimensions, and real pictures are rare | pixels, channels, tensors, the data distribution, why sampling it is hard, the manifold idea | a 28×28 digit unrolling into 784 numbers; random points that are all static; real images as a thin swirl | the digit model's training data; the swirl as a 2D stand-in | ✅ `ep02-images-are-numbers` |
| 3 | **Before Diffusion** | Earlier generators traded quality, diversity and stability | autoencoders, VAEs (blurry), GANs (sharp but unstable, mode collapse), autoregressive pixels; why diffusion won | a VAE's blurry average digit; a GAN's generator–critic tug of war; mode collapse on the swirl | the swirl: a collapsed generator covering one arm | ✅ `ep03-before-diffusion` |
| 4 | **Adding Noise** | The forward process destroys a picture in a way we can jump to in one step | Gaussian noise, the noise schedule (β_t, ᾱ_t), x_t = √ᾱ·x₀ + √(1−ᾱ)·ε, signal-to-noise ratio | the swirl melting into a Gaussian blob; a digit dissolving at t = 0 … 1000; the schedule curve | the swirl and the digit model's schedule | ✅ `ep04-adding-noise` |

## Module 2 · Learning to Denoise — amber

| # | Episode | The viewer will understand | Key concepts | Mechanism to animate | Running example | Status |
|---|---|---|---|---|---|---|
| 5 | **Learning to Undo Noise** | A network learns to predict the noise that was added — a simple regression | ε-prediction, the simple MSE loss, the training loop, x₀- and v-prediction | one training step: pick an image, pick t, add noise, predict it, compare | the digit model's training loss on this Mac | ✅ `ep05-learning-to-undo-noise` |
| 6 | **The Score** | Denoising is the same as learning which way is "more likely" | score function ∇log p, Langevin dynamics, denoising score matching, Tweedie, the score at many noise levels | an arrow field on the swirl pointing back to the spiral; Langevin particles sliding in | the swirl's exact score, computed | ✅ `ep06-the-score` |
| 7 | **Sampling Step by Step** | Start from pure noise and apply the denoiser a thousand times | DDPM ancestral sampling, the reverse step, fresh noise each step, stochasticity | the swirl un-melting step by step; a digit emerging over 1,000 steps | the digit model: one sample, 1,000 steps | ✅ `ep07-sampling-step-by-step` |
| 8 | **The U-Net** | The workhorse denoiser sees the whole picture and the fine detail | encoder–decoder, skip connections, time embedding, attention blocks, parameter counts | a picture shrinking and growing through the U; the timestep entering every block | the digit model's U-Net, layer by layer | ✅ `ep08-the-unet` |

## Module 3 · Faster Sampling — blue

| # | Episode | The viewer will understand | Key concepts | Mechanism to animate | Running example | Status |
|---|---|---|---|---|---|---|
| 9 | **Fewer Steps** | The same model can sample deterministically in far fewer steps | DDIM, the probability-flow ODE, deterministic vs stochastic sampling, inversion | the same noise taking a smooth path to the same digit; 1,000 → 50 → 10 steps side by side | the digit model sampled with 1,000 / 50 / 10 steps, measured | ✅ `ep09-fewer-steps` |
| 10 | **Solvers and Schedules** | Better ODE solvers and step placement buy quality per step | Euler vs Heun, DPM-Solver, Karras/EDM, noise schedules (linear, cosine), sampling as numerical integration | paths on the swirl with big and small steps; error vs steps | the swirl: error vs steps per solver; the lighthouse at 4–50 steps | ✅ `ep10-solvers-and-schedules` |
| 11 | **Flow Matching** | Train on straight lines from data to noise and learn the velocity along them | flow matching, rectified flow, reflow, optimal-transport paths, the velocity target (not the same as v-prediction); Stable Diffusion 3, FLUX | straight training pairs vs the curved paths the learned field follows; Euler steps on both | the swirl and the digit model trained with flow matching vs DDPM, measured | ✅ `ep11-flow-matching` |
| 12 | **One Step at a Time?** | Distillation compresses many steps into a few | progressive distillation, consistency models, latent consistency, adversarial distillation (SDXL Turbo) | a teacher's 64 steps folding into 8, then 1 | a one-step turbo model on this Mac vs 50 steps | ✅ `ep12-one-step` |

## Module 4 · Steering the Picture — pink

| # | Episode | The viewer will understand | Key concepts | Mechanism to animate | Running example | Status |
|---|---|---|---|---|---|---|
| 13 | **Asking for a Digit** | A condition tells the denoiser what to make | class conditioning, embeddings, classifier guidance | the same noise becoming 0 … 9 depending on the label | the digit model, one seed, ten labels | ✅ `ep13-asking-for-a-digit` |
| 14 | **Classifier-Free Guidance** | Push away from "anything" toward "what I asked for" | CFG, the guidance scale, label dropout, quality vs diversity, oversaturation | two predictions and the extrapolation between them; a grid of guidance scales | the digit model and the lighthouse at guidance scales from 1 (no guidance) upward — one convention on screen | ✅ `ep14-classifier-free-guidance` |
| 15 | **Words to Pictures** | Text encoders turn a prompt into vectors the U-Net attends to | tokenizers, CLIP and T5 text encoders, cross-attention, contrastive training, prompt sensitivity | words lighting up regions through cross-attention maps | the lighthouse prompt, word by word | ✅ `ep15-words-to-pictures` |
| 16 | **Editing and Control** | The same machinery repaints, extends and follows sketches | SDEdit / image-to-image, inpainting, ControlNet, LoRA and DreamBooth fine-tunes | noise added part way then removed; a masked region repainted | the lighthouse repainted at strengths 0.3 … 0.9 | ✅ `ep16-editing-and-control` |

## Module 5 · Scaling Up — teal

| # | Episode | The viewer will understand | Key concepts | Mechanism to animate | Running example | Status |
|---|---|---|---|---|---|---|
| 17 | **Latent Diffusion** | Diffuse in a small compressed space, then decode | the VAE autoencoder, 8× downsampling, latent channels, why it's cheaper; Stable Diffusion | a 512×512 picture squeezed to 64×64×4 and back | the lighthouse's latents, decoded at each step | ✅ `ep17-latent-diffusion` |
| 18 | **Diffusion Transformers** | Transformers on patches replace the U-Net at scale | DiT, patches as tokens, adaLN conditioning, scaling laws, MM-DiT (SD3) | a latent cut into patches flowing through a transformer | DiT's scaling results (cited); our U-Net vs a tiny DiT on digits | ✅ `ep18-diffusion-transformers` |
| 19 | **Measuring Pictures** | Judging generators is hard; every metric has a blind spot | FID, Inception features, precision/recall, CLIP score, human preference studies | two clouds of features compared; FID dropping with steps | the digit model's FID-like score vs steps and guidance, measured | ✅ `ep19-measuring-pictures` |
| 20 | **Data and Alignment** | What a model learns depends on its data and what it's tuned to prefer | web-scale image–text data, recaptioning (DALL·E 3), aesthetic filtering, preference tuning (Diffusion-DPO), memorization | captions improving; a model drifting toward preferred samples | recaptioned vs original captions (cited); our digits: nearest training neighbours | ✅ `ep20-data-and-alignment` |

## Module 6 · Video and Your Mac — lime

| # | Episode | The viewer will understand | Key concepts | Mechanism to animate | Running example | Status |
|---|---|---|---|---|---|---|
| 21 | **From Pictures to Video** | A video is a stack of frames diffused together | the time axis, 3D U-Nets, spacetime patches, temporal attention, latent video (Sora, 2024) | a cube of noise becoming a short clip; patches spanning space and time | a tiny moving-digit clip (computed); published video models (cited) | ✅ `ep21-from-pictures-to-video` |
| 22 | **Keeping Frames Together** | Consistency across frames is the hard part of video | flicker, temporal attention, frame interpolation, long videos, world models | frames drawn independently flickering vs jointly holding steady | independent vs joint denoising on the moving digit | ✅ `ep22-keeping-frames-together` |
| 23 | **Diffusion on Your Mac** | What it takes to run these models locally, and where the time goes | model sizes, memory, steps × time per step, text encoder vs U-Net vs VAE cost, MLX | a timeline of one generation; memory bars | the lighthouse: time and memory measured on this Mac | ✅ `ep23-diffusion-on-your-mac` |
| 24 | **Out of the Noise, Revisited** | The whole pipeline in one picture, trained and run on this Mac | the digit model end to end; the course revisited | the swirl, the digit and the lighthouse emerging side by side | all three running examples, one last time | ✅ `ep24-out-of-the-noise-revisited` |

## Paths through the course

- **Just the core:** 1 → 4 → 5 → 7 → 9 → 14 → 17.
- **The maths:** 4 → 6 → 9 → 10 → 11.
- **Using the models:** 1 → 14 → 15 → 16 → 23.

## Production notes

- Series kit: `kit.py` (palette, diffusion components, Pixel, cards); component tour: `kit_demo.py`; canonical
  algorithms and numbers: `diffusion_reference.py`; research: `research/notes.md`; runs on this Mac:
  `research/experiments/` (load with `measured(...)`).
- Episodes are made by subagents from `briefs/epNN.md` (generated by `briefs/make_briefs.py` from
  `briefs/spec.py`), one shared narrator; each episode is reviewed by the coordinator (numbers, storyboards,
  `briefs/check.sh N`).
