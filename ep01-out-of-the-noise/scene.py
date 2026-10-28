"""Out of the Noise · Episode 1 · Out of the Noise (Module 1 · Pictures from Noise — the series opener)

A diffusion model turns noise into a picture by removing a little noise many times. Running example: the
lighthouse, generated on this Mac (Stable Diffusion 2.1 base via MLX, prompt "a lighthouse on a rocky coast at dusk,
oil painting", 512 × 512, 30 steps, guidance 7.5, seed 0): its 31 decoded x_t frames and 30 decoded x̂₀ previews
(research/experiments/images/lighthouse_frames30/, mac_runs.md §5 "Picture out of static"). The digit model (seed 47,
DDIM 50, research/experiments/arrays/) and the swirl (diffusion_reference: DDIM 50 with the exact denoiser) appear as
the other two running examples.

Sections: hook (static → lighthouse, frame by frame) · the idea in one sentence (destroy with noise: easy; undo a
little at a time: learned; repeat) · the denoising loop (kit denoise_loop turning through the real frames; the x̂₀
guesses; one big step vs fifty small ones on the digit model) · recognition vs generation · a dated history strip ·
the three running examples · the course map (six modules in the module colours) · recap · Episode 2's question.

Sources (research/notes.md §1, Appendix A, Appendix C row 1; facts dated October 2026):
- VAE: Kingma & Welling, arXiv:1312.6114 (20 Dec 2013). GAN: Goodfellow et al., arXiv:1406.2661 (10 Jun 2014).
- Diffusion: Sohl-Dickstein, Weiss, Maheswaranathan & Ganguli, arXiv:1503.03585 (12 Mar 2015): "systematically and
  slowly destroy structure … through an iterative forward diffusion process. We then learn a reverse diffusion
  process that restores structure in data".
- DDPM: Ho, Jain & Abbeel, arXiv:2006.11239 (19 Jun 2020), abstract: "a state-of-the-art FID score of 3.17"
  (unconditional CIFAR-10); T = 1000.
- Latent diffusion / Stable Diffusion: Rombach et al., arXiv:2112.10752 (20 Dec 2021, CVPR 2022); Stable Diffusion
  v1.4 public release 22 Aug 2022 (Stability AI release post).
- Video: Video Diffusion Models, Ho et al., arXiv:2204.03458 (Apr 2022); Sora technical report, OpenAI, 15 Feb 2024
  ("a minute of high fidelity video").
- "Most open image and video models are diffusion or flow-matching transformers": notes §1, common misconceptions
  (the recommended wording), October 2026.
- 'AI, Explained' Episode 35 · Diffusion Models (the sister course's one-episode overview).

Measured on this Mac (research/experiments/mac_runs.json via measured(), and the run's arrays and images; asserted):
the 30-step run (steps, guidance, seed, 31 x_t + 30 x̂₀ frames, 512 × 512, its own total of 4.02 s: text 0.008 s +
denoise 3.84 s + decode 0.17 s, the 30 preview decodes not counted); the default 50-step generation 7.32 s; the
digit model (3,940,801 parameters, trained in 31.8 min; DDIM 1 step vs 50 steps from the same noise, the
classifier-based FID-like score 943 vs 3.35 on 10,000 digits); seed-47 and other-seed sevens (DDIM 50, w = 1).
Computed here: the lighthouse noised in pixel space with q_sample (DDPM's linear schedule, noise seed 1, at 256 × 256);
the first 7 in MNIST's training set (load_mnist, index 15) dissolving; the swirl sampled by DDIM 50 with exact_eps.

Illustrative: the lighthouse is noised in pixels here to show the forward process; the model itself denoises a
64 × 64 × 4 latent (Episode 17). The swirl card's view zooms in as the cloud tightens.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403
from diffusion_reference import *  # noqa: E402,F403

PREFIX = "ep01"
PX = 135                                   # pixels per scene unit at 1080p
IMG, ARR = EXPERIMENTS / "images", EXPERIMENTS / "arrays"
HL = lambda *keys: measured("mac_runs", "headline", *keys)  # noqa: E731

# ---------------------------------------------------------------------------- the lighthouse (30-step run)

PROMPT_TEXT = "a lighthouse on a rocky coast at dusk, oil painting"
RUN = measured("mac_runs", "lighthouse", "frames")
STEPS, CFG, SEED = RUN["steps"], RUN["cfg"], RUN["seed"]
assert (STEPS, CFG, SEED) == (30, 7.5, 0)                                            # "thirty steps"
assert RUN["n_xt_frames"] == STEPS + 1 and RUN["n_x0hat_frames"] == STEPS
assert RUN["image_shape"] == [512, 512, 3] and RUN["latent_shape"] == [64, 64, 4]
LAT = np.load(ARR / "lighthouse_latents_30steps.npz")
assert LAT["x_t"].shape == (STEPS + 1, 64, 64, 4) and LAT["x0_hat"].shape == (STEPS, 64, 64, 4)
assert PROMPT_TEXT in str(LAT["note"]) and "30 steps, cfg 7.5, seed 0" in str(LAT["note"])
RUN_S = RUN["timing"]["total_s"]
assert f"{RUN_S:.1f}" == "4.0"                                                        # "about four seconds"
T50 = HL("lighthouse_total_s_50steps_cfg7.5")
assert T50 == 7.32
MODEL_REPO = next(iter(measured("mac_runs", "lighthouse", "models")))
assert MODEL_REPO.endswith("stable-diffusion-2-1-base")
MLX_VERSION = measured("mac_runs", "lighthouse", "mlx")


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))


FRAMES = IMG / "lighthouse_frames30"
XT = [load_rgb(FRAMES / f"xt_{k:02d}.png") for k in range(STEPS + 1)]
X0H = [load_rgb(FRAMES / f"x0hat_{k:02d}.png") for k in range(STEPS)]
assert all(f.shape == (512, 512, 3) for f in XT + X0H)
assert np.array_equal(XT[STEPS], X0H[STEPS - 1])          # the last guess is the finished picture
LIGHTHOUSE = XT[STEPS]
assert np.abs(XT[0].astype(int) - XT[STEPS]).mean() > 30                         # static is not the picture

# the forward process on the finished picture, in pixels (illustrative: the model works on a latent)
LH_SMALL = np.asarray(Image.fromarray(LIGHTHOUSE).resize((256, 256), Image.LANCZOS)) / 127.5 - 1.0
LH_EPS = np.random.default_rng(1).standard_normal(LH_SMALL.shape)


def lh_noised(t):
    return to_rgb8(np.clip(q_sample(LH_SMALL, t, LH_EPS), -1, 1), -1, 1)


assert np.abs(lh_noised(0).astype(float) / 127.5 - 1 - LH_SMALL).max() < 0.01
assert np.corrcoef(lh_noised(1000).ravel(), LH_SMALL.ravel())[0, 1] < 0.02          # pure noise at t = 1000

# the lamp: the reddest pixel in the top of the picture, and its 3 × 3 neighbourhood (Episode 2's teaser)
_top = LIGHTHOUSE[:300].astype(int)
LAMP_R, LAMP_C = np.unravel_index((_top[..., 0] - (_top[..., 1] + _top[..., 2]) / 2).argmax(), _top.shape[:2])
LAMP = LIGHTHOUSE[LAMP_R - 1:LAMP_R + 2, LAMP_C - 1:LAMP_C + 2]
assert (LAMP_R, LAMP_C) == (74, 335) and LAMP.shape == (3, 3, 3)
N_NUMBERS = int(np.prod(RUN["image_shape"]))
assert N_NUMBERS == 786_432

# ---------------------------------------------------------------------------- the digit model (trained on this Mac)

PARAMS = HL("digit_unet_params")
TRAIN_MIN = HL("digit_ddpm_train_minutes")
assert PARAMS == 3_940_801 and f"{PARAMS / 1e6:.1f}" == "3.9" and round(TRAIN_MIN) == 32   # "thirty-two minutes"
CFID_1, CFID_50 = HL("cfid_ddim1"), HL("cfid_ddim50")
assert f"{CFID_1:.0f}" == "943" and f"{CFID_50:.2f}" == "3.35"
S1 = np.load(ARR / "digit_samples_ddpm_ddim1_w1.npz")
S50 = np.load(ARR / "digit_samples_ddpm_ddim50_w1.npz")
assert list(S1["labels"][:10]) == list(range(10)) == list(S50["labels"][:10])           # same noise, labels 0 … 9
assert "seed 1234" in str(S1["note"]) and "seed 1234" in str(S50["note"])
ROW_1, ROW_50 = S1["samples"][:10], S50["samples"][:10]

TRAJ = np.load(ARR / "digit_trajectory_7_more_seeds.npz")
assert int(TRAJ["label"]) == 7 and 47 in list(TRAJ["seeds"])
SEVEN_SEED = 47
SEVEN_XT = TRAJ[f"seed{SEVEN_SEED}_ddpm_ddim50_w1_x_t"].astype(float)      # 50 inputs, x_T first
SEVEN = TRAJ[f"seed{SEVEN_SEED}_ddpm_ddim50_w1_final"].astype(float)
assert SEVEN_XT.shape == (50, 28, 28) and abs(SEVEN_XT[0].std() - 1) < 0.1          # x_T is standard noise
SEVEN_FRAMES = list(SEVEN_XT) + [SEVEN]
FAN_SEEDS = [17, 37, 47, 67]
FAN = [(TRAJ[f"seed{s}_ddpm_ddim50_w1_x_t"][0].astype(float), TRAJ[f"seed{s}_ddpm_ddim50_w1_final"].astype(float))
       for s in FAN_SEEDS]

TEN = np.load(ARR / "digit_same_noise_ten_labels.npz")
assert list(TEN["labels"]) == list(range(10))
TEN_ROW = TEN["ddpm_ddim50_w1"][0]                                               # seed 100, labels 0 … 9
VIDEO = np.load(ARR / "digit_video_noise_correlation.npz")
CLIP = VIDEO["rho1_canvas"][7]
assert int(VIDEO["labels"][7]) == 7 and CLIP.shape == (8, 28, 52)

# the first 7 in MNIST's training set, for recognition and the map (as in Episode 4)
IMAGES, LABELS = load_mnist("train")
DIGIT_INDEX = int(np.flatnonzero(LABELS == 7)[0])
assert DIGIT_INDEX == 15
X0_DIGIT = IMAGES[DIGIT_INDEX] / 127.5 - 1.0
EPS_DIGIT = np.random.default_rng(4).standard_normal((28, 28))

# ---------------------------------------------------------------------------- the swirl (computed)

SW_N = 500
SW_PATH = ddim_sample(exact_eps, gaussian_noise(SW_N, seed=5), steps=50).xs
assert SW_PATH.shape == (51, SW_N, 2) and swirl_distance(SW_PATH[-1]) < 0.03                 # it lands on the spiral
SW_EXT = np.maximum.accumulate(np.array([np.abs(p).max() for p in SW_PATH])[::-1])[::-1]       # the view only zooms in

# ---------------------------------------------------------------------------- the course

assert len(EPISODES) == 24 and len(MODULES) == 6                                    # "twenty-four … six modules"
MODULE_EPS = {k: [n for n, (_, mod) in EPISODES.items() if mod == k] for k in MODULES}
assert [len(v) for v in MODULE_EPS.values()] == [4] * 6

# ---------------------------------------------------------------------------- look

FLAT = set("abcdefhiklmnorstuvwxzABCDEFGHIJKLMNOPRSTUVWXYZ0123456789")


def cite(s, size=20):
    return text(s, size, MUTED)


def baseline(words):
    """The baseline of a Text: the median bottom of its flat-bottomed letters."""
    chars = [c for c in words.text if not c.isspace()]
    if len(words) == len(chars):
        bottoms = sorted(g.get_bottom()[1] for g, c in zip(words, chars) if c in FLAT)
        if bottoms:
            return bottoms[len(bottoms) // 2]
    return float(np.median([g.get_bottom()[1] for g in words]))


def sit(words, y):
    """Put the baseline of Text `words` at height y."""
    return words.shift((y - baseline(words)) * m.UP)


_TEX = {}


def texture_of(rgb, px, nearest=False):
    """An RGB array → a read-only RGBA texture px pixels wide: small arrays blown up with crisp square pixels when
    there is room for at least 3 screen pixels per pixel, otherwise resized smoothly (the engine samples, not
    averages, when it draws a texture smaller than it is)."""
    key = (id(rgb), px, nearest)
    if key in _TEX:
        return _TEX[key][1]
    h, w = rgb.shape[:2]
    if nearest and px >= 3 * w:
        k = int(np.ceil(px / w))
        arr = np.repeat(np.repeat(rgb, k, axis=0), k, axis=1)
    elif px != w:
        arr = np.asarray(Image.fromarray(rgb).resize((px, max(1, round(px * h / w))),
                                                     Image.LANCZOS if px < w else Image.BICUBIC))
    else:
        arr = rgb
    out = np.concatenate([arr, np.full(arr.shape[:2] + (1,), 255, np.uint8)], axis=-1)
    out.flags.writeable = False
    _TEX[key] = (rgb, out)
    return out


def px_wide(rgb, height):
    h, w = rgb.shape[:2]
    return int(np.ceil(height * PX * w / h))


def photo(rgb, height, nearest=None):
    """A picture `height` units tall from an RGB array (crisp pixels for small arrays)."""
    nearest = max(rgb.shape[:2]) <= 64 if nearest is None else nearest
    image = m.ImageMobject(texture_of(rgb, px_wide(rgb, height), nearest))
    image.height = height
    return image


class Reel:
    """A picture that steps through a list of frames (real frames of a run): `.image`, `.show(k)`."""

    def __init__(self, frames, height, k=0, nearest=None):
        nearest = max(frames[0].shape[:2]) <= 64 if nearest is None else nearest
        px = px_wide(frames[0], height)
        self.tex = [texture_of(f, px, nearest) for f in frames]
        self.image = m.ImageMobject(self.tex[k])
        self.image.height = height
        self.k = k

    def show(self, k):
        k = int(np.clip(round(k), 0, len(self.tex) - 1))
        if k != self.k:
            self.image.paint = self.image.paint.but(texture=self.tex[k])
            self.k = k


class Counter(m.VGroup):
    """A mono number in BLUE (time, steps), left-aligned at a fixed point: `.set(k)`."""

    def __init__(self, left, y, k=0, size=44, fmt="{}", color=TIME):
        super().__init__()
        self.anchor = np.array([left, y, 0.0])
        self.size, self.fmt, self.color = size, fmt, color
        self.words = self._make(k)
        self.add(self.words)
        self.k = k

    def _make(self, k):
        return code(self.fmt.format(int(round(k))), self.size, self.color, weight=m.BOLD).move_to(
            self.anchor, aligned_edge=m.LEFT)

    def set(self, k):
        if int(round(k)) != self.k:
            self.words.become(self._make(k))
            self.k = int(round(k))


class Progress(m.VGroup):
    """A thin track with a BLUE bar growing from the left: `.set(fraction)`."""

    def __init__(self, left, y, span=5.6, bar_h=0.12, color=TIME):
        super().__init__()
        self.left, self.y, self.span, self.bar_h, self.color = left, y, span, bar_h, color
        self.track = m.Rectangle(width=span, height=bar_h).set_stroke(EDGE, 1.5).set_fill(CARD, 1)
        self.track.move_to([left + span / 2, y, 0])
        self.bar = self._bar(0.0)
        self.add(self.track, self.bar)

    def _bar(self, f):
        w = max(self.span * float(np.clip(f, 0, 1)), 0.004)
        return m.Rectangle(width=w, height=self.bar_h).set_stroke(width=0).set_fill(self.color, 1).move_to(
            [self.left + w / 2, self.y, 0])

    def set(self, f):
        self.bar.become(self._bar(f))


def drive(mob, a0, a1, hooks, run_time=1.4, rate=m.linear):
    """Animate a value from a0 to a1, calling every hook(value) each frame."""

    def step(_, a):
        v = a0 + (a1 - a0) * a
        for hook in hooks:
            hook(v)

    return m.UpdateFromAlphaFunc(mob, step, run_time=run_time, rate_func=rate)


def flash(path, color=INK, width=6, time_width=0.6):
    """A light running along a copy of a path (an arc or a line)."""
    return m.ShowPassingFlash(path.copy().set_stroke(color, width), time_width=time_width)


def side(t):
    """Teal while the picture outweighs the noise (SNR > 1), violet once the noise wins (as in Episode 4)."""
    return TEAL if float(snr(alpha_bar_at(t))) > 1 else VIOLET


def digit_rgb(x, color=DATA, lo=-1, hi=1):
    return tint(x, color, lo, hi)


def seven_rgb(i):
    """Frame i of the seed-47 seven: violet while it is noise, teal by the end."""
    return tint(SEVEN_FRAMES[i], mix(NOISE, DATA, min(1.0, 1.25 * i / (len(SEVEN_FRAMES) - 1))), -1, 1)


def fit(mob, w, h):
    s = min(w / mob.width, h / mob.height, 1.0)
    return mob.scale(s) if s < 1 else mob


# ---------------------------------------------------------------------------- the episode


class Ep01OutOfTheNoise(Episode):
    number = 1

    def lesson(self):
        self.hook()
        self.the_idea()
        self.the_loop()
        self.two_directions()
        self.history()
        self.examples()
        self.course_map()
        self.recap()
        self.next_question()

    # ------------------------------------------------------------------ 0 · hook

    def hook(self):
        self.next_section("Hook", notes=(
            f"Generated on this Mac (Apple M4 Max, MLX {MLX_VERSION}): {MODEL_REPO} (the community mirror of Stability's "
            f"SD 2.1 base; its own repository is gone), prompt '{PROMPT_TEXT}', 512 × 512, {STEPS} steps of the "
            f"mlx-examples Euler sampler, guidance {CFG}, seed {SEED}. The frames are the decoded latents x_t at every "
            f"step (xt_00 = pure noise … xt_30 = the picture), mac_runs.md §5. This run's own time: {RUN_S:.2f} s "
            f"(text encoder, {STEPS} denoising steps, one decode; the 30 preview decodes for this film not counted); "
            f"the default 50 steps take {T50} s (median of 3 runs). Fact date October 2026."))
        big = Reel(XT, 4.7)
        pic = framed(big.image).move_to([-3.05, -0.3, 0])
        left = 0.45
        step_lab = text("denoising step", 24, MUTED).move_to([0, 1.88, 0]).align_to([left, 0, 0], m.LEFT)
        counter = Counter(left, 1.2, 0, size=52, fmt="{} / " + str(STEPS))
        bar = Progress(left, 0.55, span=5.9)
        readout = m.VGroup(step_lab, counter, bar)

        p_box = card(5.9, 1.6, PROMPT, PINK_FILL, radius=0.16, stroke=2).move_to([left + 2.95, -0.72, 0])
        p_lab = code("prompt", 20, PROMPT, weight=m.BOLD)
        p_lab.move_to(p_box.get_corner(m.UL) + [0.3, -0.3, 0], aligned_edge=m.LEFT)
        p_txt = code('"a lighthouse on a rocky coast\n at dusk, oil painting"', 22, INK)
        p_txt.next_to(p_lab, m.DOWN, buff=0.16, aligned_edge=m.LEFT)
        prompt = m.VGroup(p_box, p_lab)
        made = pill("generated on this Mac", INK, CARD, 22).move_to([0, -2.05, 0]).align_to([left, 0, 0], m.LEFT)
        details = m.VGroup(
            cite(f"Stable Diffusion 2.1 (base) · MLX {MLX_VERSION} · 512 × 512"),
            cite(f"{STEPS} steps · guidance {CFG} · seed {SEED} · {RUN_S:.1f} s on an M4 Max"),
            cite(f"(the default 50 steps: {T50:.2f} s)"),
        ).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.1)
        details.next_to(made, m.DOWN, buff=0.2, aligned_edge=m.LEFT)

        title = title_text("diffusion models", 52, INK).move_to([3.3, 0.75, 0])
        sub1 = text("pictures, and now video,", 28, MUTED)
        sub2 = text("out of pure noise", 28, MUTED)
        subs = m.VGroup(sub1, sub2).arrange(m.DOWN, buff=0.14).next_to(title, m.DOWN, buff=0.35)
        mascot = pixel(0.55).next_to(subs, m.DOWN, buff=0.5)

        def at(k):
            big.show(k)
            counter.set(k)
            bar.set(k / STEPS)

        self.say(
            "[Here is a square of pure static]: [random numbers, with no picture in them at all.]",
            m.AnimationGroup(self.new_header("00 · hook", "A lighthouse out of static"), m.FadeIn(pic, shift=0.2 * m.UP)),
            m.FadeIn(readout, shift=0.15 * m.LEFT),
        )
        self.say(
            "[Now watch it closely]: [step by step, a picture comes out of the noise], [first shapes], [then colours], "
            "[then the fine details.]",
            drive(big.image, 0, 4, [at], run_time=0.8),
            drive(big.image, 4, 13, [at], run_time=1.0),
            drive(big.image, 13, 18, [at], run_time=0.5),
            drive(big.image, 18, 23, [at], run_time=0.5),
            drive(big.image, 23, STEPS, [at], run_time=0.8),
        )
        self.say(
            "[Nobody painted this.] [A model made it from one sentence]: "
            "[a lighthouse on a rocky coast at dusk, oil painting.]",
            m.Circumscribe(pic, color=INK, buff=0.08),
            m.FadeIn(prompt, shift=0.15 * m.UP),
            m.AddTextLetterByLetter(p_txt, time_per_char=0.03),
        )
        self.say(
            "[It was generated right here, on this Mac], [with Stable Diffusion, in thirty steps and about four seconds.]",
            m.FadeIn(made, shift=0.1 * m.UP),
            m.FadeIn(details, shift=0.1 * m.UP),
        )
        self.say(
            "[This course is about how that works]: [how diffusion models make pictures, and now video, out of pure noise.]",
            m.FadeOut(m.VGroup(readout, prompt, p_txt, made, details), shift=0.15 * m.RIGHT),
            m.AnimationGroup(m.FadeIn(title, shift=0.15 * m.UP), m.FadeIn(subs, shift=0.1 * m.UP),
                             m.FadeIn(mascot, shift=0.15 * m.UP), lag_ratio=0.25),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 1 · the idea

    def the_idea(self):
        self.next_section("1 · The idea", notes=(
            "The idea in one sentence (notes §1, mechanism): destroying an image with Gaussian noise is easy and needs no "
            "learning; a network learns to undo a little noise at a time; starting from noise and repeating makes a new "
            "image. Top row: the finished lighthouse noised in PIXEL space with q_sample (DDPM's linear schedule, T = 1000, "
            "noise seed 1, at 256 × 256) at t = 0, 100, 250, 500, 1000: illustrative of the forward process (the model "
            "itself works on a 64 × 64 × 4 latent, Episode 17). Bottom row, read right to left: the model's own steps from "
            "our run (x_t at steps 0, 8, 15, 22, 30 of 30). Sohl-Dickstein et al. 2015: 'systematically and slowly destroy "
            "structure … then learn a reverse diffusion process that restores structure'."))
        fh = 1.42
        top_t = [0, 100, 250, 500, 1000]
        bot_k = [STEPS, 22, 15, 8, 0]
        top = m.Group(*[framed(photo(lh_noised(t), fh)) for t in top_t]).arrange(m.RIGHT, buff=0.42).move_to([0, 1.55, 0])
        bot = m.Group(*[framed(photo(XT[k], fh)) for k in bot_k]).arrange(m.RIGHT, buff=0.42).move_to([0, -1.4, 0])
        x0, x1 = top.get_left()[0], top.get_right()[0]
        a1 = arrow([x0, 0.5, 0], [x1, 0.5, 0], NOISE, buff=0)
        l1 = text("adding noise: easy, no learning", 24, NOISE)
        l1.move_to([0, 0.08, 0]).align_to([x0, 0, 0], m.LEFT)
        c1 = cite("computed · DDPM's schedule").move_to([0, 0, 0]).align_to([x1, 0, 0], m.RIGHT)
        sit(c1, baseline(l1))
        a2 = arrow([x1, -2.42, 0], [x0, -2.42, 0], MODEL, buff=0)
        l2 = text("removing it, a little at a time: learned", 24, MODEL)
        l2.move_to([0, -2.85, 0]).align_to([x0, 0, 0], m.LEFT)
        c2 = cite("the model's own steps, our run").move_to([0, 0, 0]).align_to([x1, 0, 0], m.RIGHT)
        sit(c2, baseline(l2))
        badge = pill(f"× {STEPS}", TIME, BLUE_FILL, 26, mono=True).move_to([x1 + 0.95, -1.4, 0])
        rep = text("steps", 22, TIME).next_to(badge, m.DOWN, buff=0.14)

        self.say(
            "[Here is the whole idea, in one sentence.] [Destroying a picture with noise is easy], "
            "[and needs no learning at all.]",
            m.AnimationGroup(self.new_header("01 · the idea", "Destroy, then undo"), m.FadeIn(top[0], shift=0.15 * m.UP)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(f, shift=0.15 * m.RIGHT) for f in top[1:]], lag_ratio=0.35),
                             m.GrowArrow(a1)),
            m.AnimationGroup(m.FadeIn(l1, shift=0.1 * m.UP), m.FadeIn(c1)),
        )
        self.say(
            "[A network learns to undo it], [removing just a little of the noise at a time.]",
            m.AnimationGroup(m.FadeIn(bot[4], shift=0.15 * m.UP), m.GrowArrow(a2)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(f, shift=0.15 * m.LEFT) for f in bot[3::-1]], lag_ratio=0.35),
                             m.FadeIn(l2, shift=0.1 * m.UP), m.FadeIn(c2)),
        )
        self.say(
            "[Then we repeat, step after step], [until a brand-new picture comes out of the noise.]",
            m.AnimationGroup(m.FadeIn(badge, scale=0.7), m.FadeIn(rep),
                             m.LaggedStart(*[m.Circumscribe(f[0], color=TIME, buff=0.05) for f in bot[::-1]], lag_ratio=0.3)),
            m.Circumscribe(bot[0][0], color=DATA, buff=0.08),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 2 · the loop

    def the_loop(self):
        self.next_section("2 · The loop", notes=(
            "kit.denoise_loop with the real frames of our 30-step run: turn n takes x_t (step n − 1) in and gives x_t "
            "(step n) out. At each step the denoiser also implies a guess of the clean picture, x̂₀ = (z_t − σ·ε̂), decoded "
            "(Tweedie's estimate, Episode 6): x0hat_k in images/lighthouse_frames30/; columns are steps 1, 3, 5, 8, 13, 30. "
            f"The digit model, DDIM with 1 step vs 50 steps from the same starting noise (seed 1234, samples 0–9, labels "
            f"0–9; mac_runs.md §4): the classifier-based FID-like score over 10,000 digits is {CFID_1:.1f} vs {CFID_50:.2f} "
            "(lower is better; real data 2.06; not Inception FID, Episode 19). One-step speckle: the ε-model's clean guess "
            "at t = 999 divides by √ᾱ ≈ 0.0064, amplifying its error ~157× (mac_runs.md). DDPM's T = 1000: Ho et al. 2020."))
        loop = denoise_loop(center=(0, 0.05, 0), width=10.6, size=1.4,
                            labels=("noisy picture", "denoiser", "a little less noise"))
        h_img = loop.left[1].height
        lr, rr = Reel(XT, h_img, 0), Reel(XT, h_img, 1)
        for panel, reel in ((loop.left, lr), (loop.right, rr)):
            old = panel[1]
            reel.image.move_to(old)
            panel.remove(old)
            panel.add(reel.image)
        back = m.ArcBetweenPoints(loop.back.get_start(), loop.back.get_end(), angle=-0.35 * PI_).set_stroke(MUTED, 3)
        back.add_tip(tip_length=0.24)
        back.set_color(MUTED)
        loop.remove(loop.back)
        loop.back = back
        loop.add(back)
        counter = Counter(-1.05, 1.92, 1, size=30, fmt="step {} / " + str(STEPS))
        rep = text("and round again", 22, TIME).move_to([0, -2.6, 0])
        ddpm = cite("the DDPM paper: 1,000 steps · Ho et al., 2020").move_to([0, -3.2, 0])

        def turn(n):
            lr.show(n - 1)
            rr.show(n)
            counter.set(n)

        # the guesses: x_t (top) and x̂₀ (bottom) at six steps
        ks = [0, 2, 4, 7, 12, STEPS - 1]
        sh = 1.55
        row_t = m.Group(*[framed(photo(XT[k], sh)) for k in ks]).arrange(m.RIGHT, buff=0.3).move_to([0, 1.08, 0])
        row_g = m.Group(*[framed(photo(X0H[k], sh)) for k in ks]).arrange(m.RIGHT, buff=0.3).move_to([0, -1.42, 0])
        lab_t = text("the noisy picture going in", 22, NOISE).next_to(row_t, m.UP, buff=0.18).align_to(row_t, m.LEFT)
        lab_g = text("the denoiser's guess of the finished picture", 22, DATA)
        lab_g.next_to(row_g, m.UP, buff=0.18).align_to(row_g, m.LEFT)
        steps = m.VGroup(*[code(f"step {k + 1}", 20, MUTED).next_to(f, m.DOWN, buff=0.14) for k, f in zip(ks, row_g)])
        s_cite = cite(f"decoded from our {STEPS}-step run · generated on this Mac").move_to([0, -3.3, 0])
        strip_all = m.Group(row_t, row_g, lab_t, lab_g, steps, s_cite)

        # one step vs fifty
        dh = 0.86
        g1 = m.Group(*[framed(photo(digit_rgb(d, DATA, 0, 255), dh)) for d in ROW_1]).arrange(m.RIGHT, buff=0.14)
        g50 = m.Group(*[framed(photo(digit_rgb(d, DATA, 0, 255), dh)) for d in ROW_50]).arrange(m.RIGHT, buff=0.14)
        g1.move_to([0, 0.95, 0])
        g50.move_to([0, -1.15, 0])
        h1 = code(f"1 step   · FID-like score {CFID_1:.0f}", 22, RED, weight=m.BOLD)
        h1.next_to(g1, m.UP, buff=0.22).align_to(g1, m.LEFT)
        h50 = code(f"50 steps · FID-like score {CFID_50:.2f}", 22, GREEN, weight=m.BOLD)
        h50.next_to(g50, m.UP, buff=0.22).align_to(g50, m.LEFT)
        n1 = cite("the digit model, trained on this Mac · the same starting noise in both rows")
        n2 = cite("score over 10,000 digits, lower is better (our classifier's, not Inception FID)")
        notes_ = m.VGroup(n1, n2).arrange(m.DOWN, buff=0.1).move_to([0, -2.75, 0])

        self.say(
            "[This is the denoising loop]: [every model in this course is built around it.]",
            m.AnimationGroup(self.new_header("02 · the loop", "The denoising loop"),
                             m.FadeIn(m.Group(loop.left, loop.model, loop.right, loop.go, loop.out), shift=0.15 * m.UP)),
            m.AnimationGroup(m.Create(loop.back), m.FadeIn(counter)),
        )
        self.say(
            "[A noisy picture goes into the denoiser], [a slightly cleaner one comes out], "
            "[and that goes back in, for another turn.]",
            m.AnimationGroup(pulse(loop.go, NOISE), loop.model.animate(rate_func=m.there_and_back, run_time=1.0).scale(1.05)),
            pulse(loop.out, DATA),
            m.AnimationGroup(flash(loop.back, INK, 6), drive(loop.back, 1, 2, [turn], run_time=0.8)),
        )
        self.say(
            "[Round and round, thirty turns], [each one removing only a little noise.]",
            m.AnimationGroup(drive(loop.back, 2, STEPS, [turn], run_time=4.6), m.FadeIn(rep)),
            m.AnimationGroup(m.Circumscribe(loop.model, color=MODEL, buff=0.08), m.FadeIn(ddpm)),
        )
        self.say(
            "[At every turn, the denoiser also guesses the finished picture]: [a blur at first], "
            "[then sharper with every step.]",
            m.AnimationGroup(m.FadeOut(m.Group(loop, counter, rep, ddpm)),
                             m.LaggedStart(*[m.FadeIn(f, shift=0.1 * m.UP) for f in row_t], lag_ratio=0.15),
                             m.FadeIn(lab_t), m.FadeIn(lab_g)),
            m.AnimationGroup(m.FadeIn(row_g[0], shift=0.1 * m.UP), m.FadeIn(row_g[1], shift=0.1 * m.UP),
                             m.FadeIn(steps[:2])),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(f, shift=0.1 * m.UP) for f in row_g[2:]], lag_ratio=0.25),
                             m.FadeIn(steps[2:]), m.FadeIn(s_cite)),
        )
        self.say(
            "[Why not remove all the noise in one go?] [On our digit model, one giant step gives speckled smudges], "
            "[while fifty small steps give clean digits.]",
            m.FadeOut(strip_all),
            m.AnimationGroup(m.FadeIn(h1, shift=0.1 * m.UP),
                             m.LaggedStart(*[m.FadeIn(f, shift=0.1 * m.UP) for f in g1], lag_ratio=0.12)),
            m.AnimationGroup(m.FadeIn(h50, shift=0.1 * m.UP),
                             m.LaggedStart(*[m.FadeIn(f, shift=0.1 * m.UP) for f in g50], lag_ratio=0.12),
                             m.FadeIn(notes_)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 3 · two directions

    def two_directions(self):
        self.next_section("3 · Two directions", notes=(
            "Recognition maps a picture to a label; generation goes the other way (notes §1). The picture is the first 7 in "
            f"MNIST's training set (load_mnist('train')[{DIGIT_INDEX}]); the new picture is the digit model's seven from "
            f"seed {SEVEN_SEED} (DDIM 50, w = 1, arrays/digit_trajectory_7_more_seeds.npz). The four pairs: starting noise "
            f"x_T (tinted violet, clipped to −1…1) and the finished seven for seeds {FAN_SEEDS}, same model and sampler. "
            "Labels and prompts are conditions: PINK."))
        ph = 1.3
        ry, gy = 1.72, -0.25
        src = framed(photo(digit_rgb(X0_DIGIT), ph)).move_to([-4.6, ry, 0])
        clf = m.VGroup(card(2.9, 1.0, MODEL, AMBER_FILL), text("classifier", 26, INK, weight=m.BOLD)).move_to([0, ry, 0])
        clf[1].move_to(clf[0])
        lab1 = chip("7", PROMPT, PINK_FILL, width=1.0, height=1.0, size=40).move_to([4.6, ry, 0])
        r_a1 = arrow(src.get_right(), clf[0].get_left(), MUTED, buff=0.2)
        r_a2 = arrow(clf[0].get_right(), lab1[0].get_left(), MUTED, buff=0.2)
        r_cap = text("recognition: picture → label", 22, MUTED).next_to(clf, m.DOWN, buff=0.22)

        lab2 = chip("7", PROMPT, PINK_FILL, width=1.0, height=1.0, size=40).move_to([4.6, gy, 0])
        gen = m.VGroup(card(2.9, 1.0, MODEL, AMBER_FILL), text("generator", 26, INK, weight=m.BOLD)).move_to([0, gy, 0])
        gen[1].move_to(gen[0])
        new = framed(photo(digit_rgb(SEVEN), ph), DATA).move_to([-4.6, gy, 0])
        g_a1 = arrow(lab2[0].get_left(), gen[0].get_right(), PROMPT, buff=0.2)
        g_a2 = arrow(gen[0].get_left(), new.get_right(), DATA, buff=0.2)
        g_cap = text("generation: label → a new picture", 22, MUTED).next_to(gen, m.DOWN, buff=0.22)
        new_cap = cite("digit model · trained on this Mac").next_to(new, m.DOWN, buff=0.14)

        pairs, outs, noises, links = m.Group(), m.Group(), m.Group(), m.VGroup()
        for xT, fin in FAN:
            n_ = framed(photo(digit_rgb(xT, NOISE), 0.9))
            o_ = framed(photo(digit_rgb(fin, DATA), 0.9), DATA)
            pair = m.Group(n_, o_).arrange(m.RIGHT, buff=0.6)
            pairs.add(pair)
        pairs.arrange(m.RIGHT, buff=0.55).move_to([0, -2.75, 0])
        for pair in pairs:
            noises.add(pair[0])
            outs.add(pair[1])
            links.add(arrow(pair[0].get_right(), pair[1].get_left(), MUTED, buff=0.08))
        fan_cap = cite("four seeds: four kinds of static, four different sevens").move_to([0, -1.85, 0])

        self.say(
            "[Most networks you have met do recognition]: [a classifier maps a picture to a label.]",
            m.AnimationGroup(self.new_header("03 · two directions", "Recognise, or generate"),
                             m.FadeIn(src, shift=0.15 * m.UP)),
            m.Succession(m.AnimationGroup(m.GrowArrow(r_a1), m.FadeIn(clf), m.FadeIn(r_cap), run_time=0.7),
                         m.AnimationGroup(m.GrowArrow(r_a2), m.FadeIn(lab1, scale=0.7), run_time=0.6)),
        )
        self.say(
            "[A generator goes the other way]: [from a label, or a sentence], [to a brand-new picture.]",
            m.AnimationGroup(m.FadeIn(gen), m.FadeIn(g_cap)),
            m.AnimationGroup(m.FadeIn(lab2, target_position=lab1), m.GrowArrow(g_a1)),
            m.AnimationGroup(m.GrowArrow(g_a2), m.FadeIn(new, shift=0.15 * m.LEFT), m.FadeIn(new_cap)),
        )
        self.say(
            "[That is the harder direction], [because one label fits countless different pictures.]",
            m.Circumscribe(gen, color=MODEL, buff=0.08),
            m.AnimationGroup(m.FadeIn(fan_cap),
                             m.LaggedStart(*[m.FadeIn(o, shift=0.15 * m.UP) for o in outs], lag_ratio=0.25)),
        )
        self.say(
            "[Diffusion gets that variety from noise]: [every seed starts as different static], "
            "[and turns into a different seven.]",
            m.LaggedStart(*[m.FadeIn(n_, shift=0.15 * m.RIGHT) for n_ in noises], lag_ratio=0.25),
            m.LaggedStart(*[m.Circumscribe(n_[0], color=NOISE, buff=0.05) for n_ in noises], lag_ratio=0.2),
            m.AnimationGroup(m.LaggedStart(*[m.GrowArrow(a) for a in links], lag_ratio=0.25),
                             m.LaggedStart(*[m.Circumscribe(o[0], color=DATA, buff=0.05) for o in outs], lag_ratio=0.25)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 4 · history

    def history(self):
        self.next_section("4 · History", notes=(
            "Dated strip (notes §1 timeline, arXiv v1 dates; Appendix C row 1): VAE, Kingma & Welling, arXiv:1312.6114, "
            "20 Dec 2013; GAN, Goodfellow et al., arXiv:1406.2661, 10 Jun 2014; diffusion probabilistic models, "
            "Sohl-Dickstein, Weiss, Maheswaranathan & Ganguli, arXiv:1503.03585, 12 Mar 2015 (ICML 2015); DDPM, Ho, Jain & "
            "Abbeel, arXiv:2006.11239, 19 Jun 2020 (abstract: 'a state-of-the-art FID score of 3.17' on CIFAR-10); latent "
            "diffusion (the Stable Diffusion paper), Rombach et al., arXiv:2112.10752, 20 Dec 2021 (CVPR 2022), public "
            "release of Stable Diffusion v1.4 on 22 Aug 2022; Video Diffusion Models, Ho et al., arXiv:2204.03458, Apr 2022; "
            "Sora technical report, 15 Feb 2024: 'a minute of high fidelity video'. 'Most open image and video models are "
            "diffusion or flow-matching transformers': notes §1, as of October 2026. Episode pills point to where the "
            "course covers each."))
        axis_y = -0.35
        axis = m.Line([-6.45, axis_y, 0], [6.45, axis_y, 0]).set_stroke(EDGE, 4)
        xs = [-5.4, -3.24, -1.08, 1.08, 3.24, 5.4]
        icons_ = [
            icon(recolored("picture", MUTED, CARD, PREFIX), 0.42), icon(recolored("bolt", MUTED, CARD, PREFIX), 0.42),
            icon("noise", 0.42), icon(recolored("llm", AMBER, AMBER_FILL, PREFIX), 0.42),
            icon(recolored("brush", TEAL, TEAL_FILL, PREFIX), 0.42), icon(recolored("film", LIME, LIME_FILL, PREFIX), 0.4),
        ]
        eras = [
            ("2013", "VAEs", ["pictures, but blurry"], "Kingma & Welling, 2013", 3, MUTED),
            ("2014", "GANs", ["sharp, but hard to train"], "Goodfellow et al., 2014", 3, MUTED),
            ("2015", "Diffusion", ["destroy slowly with noise,", "then learn to reverse it"], "Sohl-Dickstein et al., 2015", 4, VIOLET),
            ("2020", "DDPM", ["a simple recipe;", "state-of-the-art FID 3.17"], "Ho, Jain & Abbeel, 2020", 5, AMBER),
            ("2021–22", "Stable Diffusion", ["paper: December 2021", "open release: 22 Aug 2022"], "Rombach et al., 2022", 17, TEAL),
            ("2022 →", "Video", ["short clips in 2022;", "a minute by February 2024"], "Ho et al., 2022 · OpenAI, 2024", 21, LIME),
        ]
        items = []
        for i, ((year, name, fact, src, ep, color), x, art) in enumerate(zip(eras, xs, icons_)):
            above = i % 2 == 0
            node = m.Circle(radius=0.4).set_stroke(color, 3).set_fill(FILLS.get(color, CARD), 1).move_to([x, axis_y, 0])
            art.move_to(node)
            _, acc, fill = MODULES[EPISODES[ep][1]]
            y_words = code(year, 30, color, weight=m.BOLD)
            tag = pill(f"Ep {ep:02d}", acc, fill, 22, mono=True)
            head = m.VGroup(y_words, tag).arrange(m.RIGHT, buff=0.22)
            tag.set_y(y_words.get_y())
            n_words = text(name, 26, INK, weight=m.BOLD)
            facts = m.VGroup(*[text(f, 20, INK) for f in fact]).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.07)
            block = m.VGroup(head, n_words, facts, cite(src)).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.12)
            block.move_to([x, 0, 0])
            if block.get_left()[0] < -6.55:
                block.shift((-6.55 - block.get_left()[0]) * m.RIGHT)
            if block.get_right()[0] > 6.55:
                block.shift((6.55 - block.get_right()[0]) * m.RIGHT)
            if above:
                block.align_to([0, axis_y + 0.68, 0], m.DOWN)
                stem = m.Line([x, axis_y + 0.4, 0], [x, block.get_bottom()[1] - 0.1, 0])
            else:
                block.align_to([0, axis_y - 0.68, 0], m.UP)
                stem = m.Line([x, axis_y - 0.4, 0], [x, block.get_top()[1] + 0.1, 0])
            stem.set_stroke(color, 2)
            items.append(dict(node=m.VGroup(node, art), stem=stem, block=block, facts=facts, head=head, name=n_words,
                              src=block[3], above=above))

        def show(k, with_facts=True):
            it = items[k]
            parts = [it["head"], it["name"], it["src"]] + ([it["facts"]] if with_facts else [])
            d = m.UP if it["above"] else m.DOWN
            return m.AnimationGroup(m.GrowFromCenter(it["node"]), m.Create(it["stem"]),
                                    m.AnimationGroup(*[m.FadeIn(p, shift=0.12 * d) for p in parts]), lag_ratio=0.25)

        today = cite("as of October 2026: most open image and video models are diffusion or flow-matching transformers")
        today.move_to([0, -3.35, 0])

        self.say(
            "[Making pictures with networks is not new.] [Variational autoencoders came in 2013], [and GANs in 2014]: "
            "[blurry, or sharp but hard to train.]",
            m.AnimationGroup(self.new_header("04 · history", "How we got here"), m.Create(axis)),
            show(0),
            show(1),
            m.AnimationGroup(m.Circumscribe(items[0]["facts"], color=MUTED, buff=0.06),
                             m.Circumscribe(items[1]["facts"], color=MUTED, buff=0.06)),
        )
        self.say(
            "[In 2015, the first diffusion paper had a slower idea]: [destroy the data with noise], "
            "[then learn to reverse it.]",
            show(2, with_facts=False),
            m.FadeIn(items[2]["facts"][0], shift=0.1 * m.UP),
            m.FadeIn(items[2]["facts"][1], shift=0.1 * m.UP),
        )
        self.say(
            "[In 2020, the DDPM paper made it work], [with a simple recipe and state-of-the-art results.]",
            show(3, with_facts=False),
            m.FadeIn(items[3]["facts"], shift=0.1 * m.DOWN),
        )
        self.say(
            "[Then latent diffusion made it cheap]: [the Stable Diffusion paper came in December 2021], "
            "[and its open release in August 2022.]",
            show(4, with_facts=False),
            m.FadeIn(items[4]["facts"][0], shift=0.1 * m.UP),
            m.FadeIn(items[4]["facts"][1], shift=0.1 * m.UP),
        )
        self.say(
            "[And from 2022, the same idea began making video]: [short clips at first], [and a minute long by 2024.]",
            show(5, with_facts=False),
            m.FadeIn(items[5]["facts"][0], shift=0.1 * m.DOWN),
            m.FadeIn(items[5]["facts"][1], shift=0.1 * m.DOWN),
        )
        self.say(
            "[Today, most open image and video models are diffusion models or close cousins], "
            "[the same idea, scaled up.]",
            m.FadeIn(today, shift=0.1 * m.UP),
            m.LaggedStart(*[m.Circumscribe(it["node"], color=c[-1], buff=0.06) for it, c in zip(items[2:], eras[2:])],
                          lag_ratio=0.25),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 5 · the three running examples

    def examples(self):
        self.next_section("5 · Three running examples", notes=(
            f"The swirl: {SW_N} points of gaussian_noise(seed=5) walked onto the swirl by ddim_sample(exact_eps, steps=50) "
            "(diffusion_reference.py; the swirl's exact denoiser, no training); the view zooms in as the cloud tightens. "
            f"The digit model: a {PARAMS:,}-parameter class-conditional U-Net trained on MNIST on this Mac in {TRAIN_MIN} "
            f"minutes (10,000 steps × 256; mac_runs.md §3); the seven is seed {SEVEN_SEED}, DDIM 50 steps, w = 1, every x_t "
            "shown (tinted from violet to teal as it clears). The lighthouse: the 30-step run of the hook (31 decoded x_t "
            "frames)."))
        W, H = 4.2, 4.95
        xs = [-4.45, 0.0, 4.45]
        cy = -0.42
        top = cy + H / 2
        boxes = [card(W, H, EDGE, CARD, radius=0.2, stroke=2).move_to([x, cy, 0]) for x in xs]
        ah = 2.25
        ay = top - 0.22 - ah / 2

        # the swirl, zooming
        sc = np.array([xs[0], ay, 0.0])
        half = ah / 2 - 0.08
        zoom = lambda i: half / float(np.interp(i, np.arange(len(SW_EXT)), SW_EXT))  # noqa: E731
        dots = m.VGroup(*[m.Dot(radius=0.024, color=NOISE) for _ in range(SW_N)])

        def put(i):
            j = int(np.floor(i))
            k = min(j + 1, len(SW_PATH) - 1)
            xy = SW_PATH[j] + (SW_PATH[k] - SW_PATH[j]) * (i - j)
            s = zoom(i)
            col = mix(NOISE, DATA, min(1.0, 1.25 * i / (len(SW_PATH) - 1)))
            for d, (x, y) in zip(dots, xy):
                d.move_to(sc + s * np.array([x, y, 0.0]))
                d.set_fill(col, 0.95)

        put(0)
        sw_box = m.Square(ah).set_stroke(EDGE, 1.5).set_fill(BG, 1).move_to(sc)
        sw_art = m.VGroup(sw_box, dots)

        seven = Reel([seven_rgb(i) for i in range(len(SEVEN_FRAMES))], ah, 0)
        seven_art = framed(seven.image).move_to([xs[1], ay, 0])
        lh = Reel(XT, ah, 0)
        lh_art = framed(lh.image).move_to([xs[2], ay, 0])

        specs = [
            ("The swirl", ["points on a spiral, in 2D:", "every number exact"],
             "DDIM · 50 steps", ("computed in the scene", MUTED)),
            ("The digit model", [f"{PARAMS / 1e6:.1f} M parameters, MNIST,", f"trained in {round(TRAIN_MIN)} minutes"],
             f"50 steps · seed {SEVEN_SEED}", ("trained on this Mac", INK)),
            ("The lighthouse", ["a real text-to-image model:", "Stable Diffusion 2.1 base"],
             f"{STEPS} steps · guidance {CFG}", ("generated on this Mac", INK)),
        ]
        cards_ = []
        for x, box, (name, subs, run, (tag, tcol)) in zip(xs, boxes, specs):
            left = box.get_left()[0] + 0.28
            t = title_text(name, 30).move_to([0, top - 2.78, 0]).align_to([left, 0, 0], m.LEFT)
            s = m.VGroup(*[text(line, 20, MUTED) for line in subs]).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.06)
            s.next_to(t, m.DOWN, buff=0.14).align_to([left, 0, 0], m.LEFT)
            r = code(run, 20, TIME).next_to(s, m.DOWN, buff=0.16).align_to([left, 0, 0], m.LEFT)
            p = pill(tag, tcol, CARD, 20).move_to([x, box.get_bottom()[1] + 0.42, 0])
            cards_.append(dict(title=t, subs=s, run=r, pill=p))

        def walk(_, a):
            put(a * (len(SW_PATH) - 1))

        self.say(
            "[This course follows three running examples], [each one computed, trained or run for real.]",
            m.AnimationGroup(self.new_header("05 · examples", "Three running examples"),
                             m.LaggedStart(*[m.FadeIn(b, scale=0.95) for b in boxes], lag_ratio=0.2)),
            m.LaggedStart(*[m.FadeIn(c["pill"], shift=0.1 * m.UP) for c in cards_], lag_ratio=0.25),
        )
        self.say(
            "[First, the swirl]: [points on a spiral, in two dimensions], [where we can compute every number exactly.]",
            m.AnimationGroup(m.FadeIn(cards_[0]["title"], shift=0.1 * m.UP), m.FadeIn(sw_art)),
            m.AnimationGroup(m.UpdateFromAlphaFunc(dots, walk, run_time=3.0, rate_func=m.smooth),
                             m.FadeIn(cards_[0]["subs"])),
            m.FadeIn(cards_[0]["run"], shift=0.1 * m.UP),
        )
        self.say(
            "[Second, the digit model]: [a small network we trained on this Mac, in thirty-two minutes], "
            "[turning noise into a seven.]",
            m.AnimationGroup(m.FadeIn(cards_[1]["title"], shift=0.1 * m.UP), m.FadeIn(seven_art)),
            m.AnimationGroup(m.FadeIn(cards_[1]["subs"]), m.FadeIn(cards_[1]["run"])),
            drive(seven.image, 0, len(SEVEN_FRAMES) - 1, [seven.show], run_time=2.8),
        )
        self.say(
            "[And third, the lighthouse]: [a real text-to-image model, run on this Mac], [painting its picture out of static.]",
            m.AnimationGroup(m.FadeIn(cards_[2]["title"], shift=0.1 * m.UP), m.FadeIn(lh_art)),
            m.AnimationGroup(m.FadeIn(cards_[2]["subs"]), m.FadeIn(cards_[2]["run"])),
            drive(lh.image, 0, STEPS, [lh.show], run_time=2.8),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 6 · the course map

    def course_map(self):
        self.next_section("6 · The course", notes=(
            "Six modules, 24 episodes, in the module colours (COURSE.md; kit MODULES/EPISODES). Card art, all real: "
            "Module 1, MNIST training 7 #15 noised with q_sample at t = 0, 250, 1000 (Episode 4's look); Module 2, the "
            "digit model's U-Net shape (28 → 14 → 7, 32/64/128 channels); Module 3, the lighthouse at 1, 4 and 50 steps "
            "(seed 0, guidance 7.5, mac_runs steps sweep); Module 4, one starting noise (seed 100) turned into 0 … 9 by "
            "changing only the label (DDIM 50); Module 5, the lighthouse and its 64 × 64 × 4 latent's four channels; "
            "Module 6, three frames of the moving-digit clip (rho = 1). COURSE.md: every episode is about learning the "
            "network, sampling with it, or scaling it."))
        W, H = 4.2, 2.85
        centres = [(-4.45, 1.07), (0.0, 1.07), (4.45, 1.07), (-4.45, -1.98), (0.0, -1.98), (4.45, -1.98)]
        names = ["Pictures from Noise", "Learning to Denoise", "Faster Sampling", "Steering the Picture", "Scaling Up",
                 "Video and Your Mac"]
        subs = ["what a picture is; noise", "how the network learns", "fewer steps, same picture",
                "labels, guidance, words", "latents, transformers, data", "frames together; your Mac"]
        for k, n in enumerate(names, start=1):
            assert n.upper() == MODULES[k][0]

        # art
        a1 = m.Group(*[framed(photo(tint(q_sample(X0_DIGIT, t, EPS_DIGIT), side(t), -1, 1), 0.76)) for t in (0, 250, 1000)])
        a1.arrange(m.RIGHT, buff=0.18)
        a2 = UNet(sizes=(28, 14, 7), channels=(32, 64, 128), scale=0.32, labels=False)
        a3 = m.Group()
        for s in (1, 4, 50):
            im = framed(photo(load_rgb(IMG / "lighthouse" / f"lighthouse_sd21_steps{s:02d}_cfg7.5_seed0.png"), 0.5))
            lab = code(f"{s} step" + ("s" if s > 1 else ""), 20, TIME).next_to(im, m.DOWN, buff=0.08)
            a3.add(m.Group(im, lab))
        a3.arrange(m.RIGHT, buff=0.25)
        a4 = m.Group()
        for d, lab in zip(TEN_ROW, range(10)):
            im = framed(photo(digit_rgb(d, DATA, 0, 255), 0.29), buff=0.02, stroke=1.5)
            a4.add(m.Group(im, code(str(lab), 20, PROMPT).next_to(im, m.DOWN, buff=0.08)))
        a4.arrange(m.RIGHT, buff=0.04)
        lat_img = np.asarray(Image.open(IMG / "lighthouse_latent_4channels_64x64.png").convert("L"))
        a5_l = framed(photo(LIGHTHOUSE, 0.78))
        a5_z = framed(photo(tint(lat_img, DATA, 0, 255), 0.42, nearest=False))
        a5_arrow = arrow([0, 0, 0], [0.45, 0, 0], MUTED, buff=0)
        a5_lab = code("64 × 64 × 4", 20, DATA)
        a5 = m.Group(a5_l, a5_arrow, m.Group(a5_z, a5_lab)).arrange(m.RIGHT, buff=0.15)
        a5_lab.next_to(a5_z, m.DOWN, buff=0.08)
        a6 = m.Group(*[framed(photo(digit_rgb(CLIP[f], DATA, 0, 255), 0.62)) for f in (0, 3, 7)]).arrange(m.RIGHT, buff=0.08)
        arts = [a1, a2, a3, a4, a5, a6]

        cards_ = []
        for k, ((cx, cy), name, sub, art) in enumerate(zip(centres, names, subs, arts), start=1):
            _, acc, fill = MODULES[k]
            box = card(W, H, acc, CARD, radius=0.2, stroke=2.5).move_to([cx, cy, 0])
            top, left, right = cy + H / 2, cx - W / 2, cx + W / 2
            tag = pill(f"MODULE {k}", acc, fill, 22, mono=True)
            tag.move_to([left + 0.25 + tag.width / 2, top - 0.38, 0])
            eps = MODULE_EPS[k]
            rng = code(f"Ep {eps[0]}–{eps[-1]}", 20, MUTED).move_to([0, top - 0.38, 0]).align_to([right - 0.25, 0, 0], m.RIGHT)
            fit(art, W - 0.5, 0.9).move_to([cx, top - 1.16, 0])
            t = title_text(name, 28).move_to([0, top - 2.02, 0]).align_to([left + 0.28, 0, 0], m.LEFT)
            s = text(sub, 20, MUTED).next_to(t, m.DOWN, buff=0.1).align_to([left + 0.28, 0, 0], m.LEFT)
            cards_.append(dict(box=box, tag=tag, rng=rng, art=art, title=t, sub=s, accent=acc))

        mascot = pixel(0.42).move_to([6.15, 3.18, 0])

        def content(c, art=True):
            parts = [c["title"], c["sub"]] + ([c["art"]] if art else [])
            return m.AnimationGroup(*[m.FadeIn(p, shift=0.1 * m.UP) for p in parts], lag_ratio=0.15)

        self.say(
            "[Here is the map of the course]: [twenty-four episodes, in six modules.]",
            m.AnimationGroup(self.new_header("06 · the course", "Six modules, one loop"), m.FadeIn(mascot, shift=0.1 * m.DOWN)),
            m.LaggedStart(*[m.AnimationGroup(m.FadeIn(c["box"], scale=0.94), m.FadeIn(c["tag"]), m.FadeIn(c["rng"]))
                            for c in cards_], lag_ratio=0.15),
        )
        self.say(
            "[Module one, pictures from noise]: [what a picture is to a computer], [and how noise destroys it.]",
            content(cards_[0], art=False),
            m.FadeIn(a1[0], shift=0.1 * m.UP),
            m.LaggedStart(*[m.FadeIn(f, shift=0.1 * m.RIGHT) for f in a1[1:]], lag_ratio=0.4),
        )
        self.say(
            "[Module two, learning to denoise]: [what the network predicts, and how it is built.]",
            content(cards_[1], art=False),
            m.Succession(m.FadeIn(a2, run_time=0.6), a2.flow(MODEL, run_time=1.4)),
        )
        self.say(
            "[Then faster sampling, with fewer and fewer steps], [and steering the picture with labels and words.]",
            content(cards_[2]),
            content(cards_[3]),
        )
        self.say(
            "[Then scaling up, with latents and transformers], [and finally video, and running it all on your Mac.]",
            content(cards_[4]),
            content(cards_[5]),
        )
        self.say(
            "[Every episode is about one of three things]: [learning the denoiser], [sampling with it], [or scaling it up.]",
            m.Wait(0.1),
            m.Circumscribe(cards_[1]["box"], color=MODEL, buff=0.05),
            m.AnimationGroup(m.Circumscribe(cards_[2]["box"], color=BLUE, buff=0.05),
                             m.Circumscribe(cards_[3]["box"], color=PINK, buff=0.05)),
            m.AnimationGroup(m.Circumscribe(cards_[4]["box"], color=TEAL, buff=0.05),
                             m.Circumscribe(cards_[5]["box"], color=LIME, buff=0.05)),
        )
        self.wait(0.6)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes=(
            "Noise (destroying a picture is easy and needs no learning), denoise (a network learns to remove a little), "
            f"repeat (our lighthouse took {STEPS} steps; the DDPM paper used 1,000)."))
        halo = glow(VIOLET, radius=4.0)
        words = m.VGroup(*[title_text(w, 80, c) for w, c in (("Noise", NOISE), ("Denoise", MODEL), ("Repeat", TIME))])
        subs = m.VGroup(*[code(s, 20, MUTED) for s in ("easy to add", "learned, a little at a time", f"{STEPS} steps, or 1,000")])
        slots = [max(w.width, s.width) for w, s in zip(words, subs)]
        x, gap = 0.0, 0.9
        for w, s, slot in zip(words, subs, slots):
            w.move_to([x + slot / 2, 0, 0])
            s.move_to([x + slot / 2, 0, 0])
            x += slot + gap
        on_baseline(words)
        line_y = min(w.get_bottom()[1] for w in words) - 0.45
        for s in subs:
            s.set_y(line_y)
        mid = baseline(words[0]) + 0.32
        dots = m.VGroup(*[m.Dot([(a.get_right()[0] + b.get_left()[0]) / 2, mid, 0], radius=0.07, color=MUTED)
                          for a, b in zip(words, words[1:])])
        row = m.VGroup(words, dots, subs).move_to(0.6 * m.UP)
        tagline = text("Out of the noise, one small step at a time.", 30, MUTED).next_to(row, m.DOWN, buff=0.7)
        mascot = pixel(0.45).next_to(tagline, m.DOWN, buff=0.45)

        self.say(
            "[To sum up]: [noise], [denoise], [repeat.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.3 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(dots[0]), m.FadeIn(words[1], shift=0.3 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(dots[1]), m.FadeIn(words[2], shift=0.3 * m.UP), m.FadeIn(subs[2])),
        )
        self.say(
            "[Destroying a picture is easy]; [undoing it, one small step at a time, is what a diffusion model learns.]",
            m.Circumscribe(words[0], color=NOISE, buff=0.1),
            m.AnimationGroup(m.Write(tagline), m.FadeIn(mascot, shift=0.2 * m.UP)),
        )
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, tagline, mascot)], run_time=0.9)

    # ------------------------------------------------------------------ 7 · next

    def next_question(self):
        self.next_section("7 · Next", notes=(
            "'AI, Explained' Episode 35 · Diffusion Models is the sister course's one-episode overview. The lamp: the "
            f"reddest pixel in the top of our lighthouse (row {LAMP_R}, column {LAMP_C} of 512) and its eight neighbours, "
            f"their real red, green and blue values (0–255). The whole picture is 512 × 512 × 3 = {N_NUMBERS:,} numbers: "
            "Episode 2."))
        box = card(8.4, 2.9, EDGE, CARD, radius=0.22, stroke=2).move_to([0, 0.15, 0])
        kicker = code("the short version, in the sister course", 20, MUTED)
        brand = title_text("AI, Explained", 52, INK)
        ep = code("Episode 35 · Diffusion Models", 24, INK, t2c={r"35": AMBER})
        line = text("diffusion in a single episode", 22, MUTED)
        inner = m.VGroup(kicker, brand, ep, line).arrange(m.DOWN, buff=0.18).move_to(box)
        short = m.VGroup(box, inner)

        pic = framed(photo(LIGHTHOUSE, 3.9)).move_to([-3.5, 0.15, 0])
        img = pic[1]
        px_x = img.get_left()[0] + (LAMP_C + 0.5) / 512 * img.width
        px_y = img.get_top()[1] - (LAMP_R + 0.5) / 512 * img.height
        win = m.Square(0.2).set_stroke(INK, 2.5).move_to([px_x, px_y, 0])
        cell = 0.95
        gc = np.array([2.6, 0.55, 0.0])
        grid = m.VGroup()
        for r in range(3):
            for c in range(3):
                rgb = LAMP[r, c]
                pos = gc + np.array([(c - 1) * (cell + 0.95), (1 - r) * (cell + 0.12), 0])
                sw = m.Square(cell * 0.62).set_stroke(EDGE, 1.5).set_fill("#%02X%02X%02X" % tuple(int(v) for v in rgb), 1)
                sw.move_to(pos + [0, 0.17, 0])
                nums = code(",".join(str(int(v)) for v in rgb), 20, INK).next_to(sw, m.DOWN, buff=0.08)
                grid.add(m.VGroup(sw, nums))
        frame_ = m.SurroundingRectangle(grid, buff=0.18).set_stroke(INK, 2.5)
        zoom = m.VGroup(m.Line(win.get_corner(m.UR), frame_.get_corner(m.UL)).set_stroke(MUTED, 1.5),
                        m.Line(win.get_corner(m.DR), frame_.get_corner(m.DL)).set_stroke(MUTED, 1.5))
        ask = title_text("What is a picture, to a computer?", 48, INK).move_to([0, -2.85, 0])

        self.say(
            "[Want the short version first?] [The sister course, AI, Explained, covers diffusion in a single episode], "
            "[number thirty-five.]",
            m.AnimationGroup(self.new_header("07 · next", "A picture, to a computer"), m.FadeIn(box, scale=0.95),
                             m.FadeIn(kicker)),
            m.FadeIn(brand, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(ep, shift=0.1 * m.UP), m.FadeIn(line)),
        )
        self.say(
            "[But before a computer can clean up a picture], [it has to hold one.] [So what is a picture, to a computer?]",
            m.AnimationGroup(m.FadeOut(short, shift=0.15 * m.DOWN), m.FadeIn(pic, shift=0.15 * m.UP)),
            m.AnimationGroup(m.Create(win), m.Create(zoom), m.Create(frame_),
                             m.LaggedStart(*[m.FadeIn(g, scale=0.8) for g in grid], lag_ratio=0.08)),
            m.FadeIn(ask, shift=0.15 * m.UP),
        )
        self.wait(0.6)
        self.clear()
