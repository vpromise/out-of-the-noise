# Out of the Noise — how diffusion models make pictures and video

A video course in 24 short episodes (3.5–4.5 minutes each), narrated and animated. It starts from a single idea — destroy an image with noise, then learn to undo it one small step at a time — and follows it all the way to the text-to-image and text-to-video models people use today, with three running examples computed or trained on this Mac.

Part of a family of narrated, animated video courses, rendered with [ManimGX](https://pypi.org/project/manimgx/) and voiced by a local narrator ([tutorial-video](https://github.com/vpromise/tutorial-video)):

- **AI, Explained** — 46 episodes on modern AI, from first principles to frontier research · [ai-explained](https://github.com/vpromise/ai-explained)
- **Agents, Animated** — how AI agents work, from one loop to a team · [agents-animated](https://github.com/vpromise/agents-animated)
- **Tokens per Second** — how LLM inference really works · [tokens-per-second](https://github.com/vpromise/tokens-per-second)
- **Trial and Reward** — reinforcement learning, from slot machines to reasoning models · [trial-and-reward](https://github.com/vpromise/trial-and-reward)
- **Out of the Noise** — how diffusion models make pictures and video *(this repository)*

This repository holds the **source code** of the course: one scene per episode, the series kit, reference code, research notes and the production tools. The rendered videos are not included.

## Episodes


### Module 1 · Pictures from Noise

| # | Episode | What you'll understand | Length | Source |
|---|---|---|---|---|
| 1 | Out of the Noise | A diffusion model turns noise into a picture by removing a little noise many times | 3:46 | [`ep01-out-of-the-noise/scene.py`](../ep01-out-of-the-noise/scene.py) |
| 2 | Images Are Numbers | A picture is a point in a space with millions of dimensions, and real pictures are rare | 3:45 | [`ep02-images-are-numbers/scene.py`](../ep02-images-are-numbers/scene.py) |
| 3 | Before Diffusion | Earlier generators traded quality, diversity and stability | 4:02 | [`ep03-before-diffusion/scene.py`](../ep03-before-diffusion/scene.py) |
| 4 | Adding Noise | The forward process destroys a picture in a way we can jump to in one step | 3:58 | [`ep04-adding-noise/scene.py`](../ep04-adding-noise/scene.py) |

### Module 2 · Learning to Denoise

| # | Episode | What you'll understand | Length | Source |
|---|---|---|---|---|
| 5 | Learning to Undo Noise | A network learns to predict the noise that was added — a simple regression | 4:05 | [`ep05-learning-to-undo-noise/scene.py`](../ep05-learning-to-undo-noise/scene.py) |
| 6 | The Score | Denoising is the same as learning which way is "more likely" | 4:09 | [`ep06-the-score/scene.py`](../ep06-the-score/scene.py) |
| 7 | Sampling Step by Step | Start from pure noise and apply the denoiser a thousand times | 3:36 | [`ep07-sampling-step-by-step/scene.py`](../ep07-sampling-step-by-step/scene.py) |
| 8 | The U-Net | The workhorse denoiser sees the whole picture and the fine detail | 3:41 | [`ep08-the-unet/scene.py`](../ep08-the-unet/scene.py) |

### Module 3 · Faster Sampling

| # | Episode | What you'll understand | Length | Source |
|---|---|---|---|---|
| 9 | Fewer Steps | The same model can sample deterministically in far fewer steps | 4:26 | [`ep09-fewer-steps/scene.py`](../ep09-fewer-steps/scene.py) |
| 10 | Solvers and Schedules | Better ODE solvers and step placement buy quality per step | 3:43 | [`ep10-solvers-and-schedules/scene.py`](../ep10-solvers-and-schedules/scene.py) |
| 11 | Flow Matching | Train on straight lines from data to noise and learn the velocity along them | 4:13 | [`ep11-flow-matching/scene.py`](../ep11-flow-matching/scene.py) |
| 12 | One Step at a Time? | Distillation compresses many steps into a few | 4:15 | [`ep12-one-step/scene.py`](../ep12-one-step/scene.py) |

### Module 4 · Steering the Picture

| # | Episode | What you'll understand | Length | Source |
|---|---|---|---|---|
| 13 | Asking for a Digit | A condition tells the denoiser what to make | 3:57 | [`ep13-asking-for-a-digit/scene.py`](../ep13-asking-for-a-digit/scene.py) |
| 14 | Classifier-Free Guidance | Push away from "anything" toward "what I asked for" | 4:21 | [`ep14-classifier-free-guidance/scene.py`](../ep14-classifier-free-guidance/scene.py) |
| 15 | Words to Pictures | Text encoders turn a prompt into vectors the U-Net attends to | 3:42 | [`ep15-words-to-pictures/scene.py`](../ep15-words-to-pictures/scene.py) |
| 16 | Editing and Control | The same machinery repaints, extends and follows sketches | 4:05 | [`ep16-editing-and-control/scene.py`](../ep16-editing-and-control/scene.py) |

### Module 5 · Scaling Up

| # | Episode | What you'll understand | Length | Source |
|---|---|---|---|---|
| 17 | Latent Diffusion | Diffuse in a small compressed space, then decode | 3:55 | [`ep17-latent-diffusion/scene.py`](../ep17-latent-diffusion/scene.py) |
| 18 | Diffusion Transformers | Transformers on patches replace the U-Net at scale | 4:28 | [`ep18-diffusion-transformers/scene.py`](../ep18-diffusion-transformers/scene.py) |
| 19 | Measuring Pictures | Judging generators is hard; every metric has a blind spot | 4:20 | [`ep19-measuring-pictures/scene.py`](../ep19-measuring-pictures/scene.py) |
| 20 | Data and Alignment | What a model learns depends on its data and what it's tuned to prefer | 4:31 | [`ep20-data-and-alignment/scene.py`](../ep20-data-and-alignment/scene.py) |

### Module 6 · Video and Your Mac

| # | Episode | What you'll understand | Length | Source |
|---|---|---|---|---|
| 21 | From Pictures to Video | A video is a stack of frames diffused together | 4:28 | [`ep21-from-pictures-to-video/scene.py`](../ep21-from-pictures-to-video/scene.py) |
| 22 | Keeping Frames Together | Consistency across frames is the hard part of video | 3:53 | [`ep22-keeping-frames-together/scene.py`](../ep22-keeping-frames-together/scene.py) |
| 23 | Diffusion on Your Mac | What it takes to run these models locally, and where the time goes | 4:26 | [`ep23-diffusion-on-your-mac/scene.py`](../ep23-diffusion-on-your-mac/scene.py) |
| 24 | Out of the Noise, Revisited | The whole pipeline in one picture, trained and run on this Mac | 3:39 | [`ep24-out-of-the-noise-revisited/scene.py`](../ep24-out-of-the-noise-revisited/scene.py) |

## Repository layout

| Path | What |
|---|---|
| `COURSE.md` | the full syllabus: key concepts, what each episode animates, the running example |
| `kit.py` | the series kit: palette, fonts, components, intro and outro cards |
| `diffusion_reference.py` | canonical algorithms and numbers (numpy, seeded, self-checked) |
| `epNN-*/scene.py` | one narrated scene per episode (`self.say("…[cue]…", animation)`: the narration drives the timing) |
| `assets/` | icons and the backdrop |
| `lexicon.json` | pronunciation overrides for the narrator |
| `research/` | research notes (sources, dated facts) and the measurement or experiment scripts |
| `briefs/` | the production briefs each episode was made from, and the QA tools (`check.sh`, `make_index.py`) |
| `LISTEN.md` | narration lines flagged for a human listen |

## Rendering an episode

Requirements: macOS on Apple silicon (the narrator runs Chatterbox on MPS and Whisper on MLX), ManimGX (`uv tool install --python 3.13 manimgx`) and the [tutorial-video](https://github.com/vpromise/tutorial-video) toolkit (set it up with its `scripts/setup_tts.sh`).

```sh
cd epNN-*/
VOICE=draft manimgx inspect scene.py      # silent draft: layout check + storyboards
# in the course folder, start the narrator (it reads lexicon.json):
#   <tutorial-video>/scripts/narrator.sh
manimgx render scene.py                    # MP4 + SRT captions (speaks new lines once, then caches them)
manimgx present scene.py --no-open         # HTML slides
```

## Not in this repository

Rendered MP4s, captions, HTML slides, the narration cache (`voice/`), storyboards and logs.

Experiment summaries are included (`research/experiments/toy.md|json` for the swirl, `research/experiments/mac_runs.md|json` for the digit model and the lighthouse). The digit models' weights, sample arrays, images (including the lighthouse frames), MNIST and logs are not — regenerate them with `research/experiments/gpu/` (MLX on Apple silicon; the lighthouse uses mlx-examples' `stable_diffusion` and downloads Stable Diffusion 2.1-base). Almost every scene reads them.
