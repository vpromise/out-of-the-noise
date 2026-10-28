"""Out of the Noise · Episode 24 · Out of the Noise, Revisited (Module 6 · Video and Your Mac — the course finale)

The whole pipeline in one picture, trained and run on this Mac. Running examples: all three, one last time — the
swirl (computed here: ddim_sample with the swirl's exact denoiser, 2,000 points of gaussian_noise(seed=5), 50 steps,
asserted against toy.json), the digit model (trained on this Mac: the seed-47 "7", DDIM 50, every x_t and x̂₀ from
arrays/digit_trajectory_7_more_seeds.npz, the same seven as Episode 1) and the lighthouse (generated on this Mac:
SD 2.1 base via MLX, the 30-step frames run for the pictures, the 50-step timing run for the stage-by-stage numbers).

Sections: hook (three pictures out of static, side by side) · three runs, three costs (and each denoiser's guess) ·
every stage of one generation with real sizes (13 tokens → 2 × 13 × 1024 → 64 × 64 × 4 noise → 50 steps × 2 passes →
decode once → 512 × 512 × 3) · the digit model end to end (data → training → sampling → measuring) · the course
revisited (Episode 1's map, one card per module with its key idea) · honest findings (Episodes 11, 14, 16, 18) ·
recap (Episode 1's three words) · the course-closing card (series_outro overridden).

Sources (research/notes.md §24, §15, §17, Appendix C row 24; research/experiments/mac_runs.md incl. its 2026-10-27
erratum; toy.md; facts dated October 2026):
- notes §24 describes SD 1.5 (77 × 768, 123M + 860M + 49.5M); our run is SD 2.1 base, whose numbers are the run's own
  (mac_runs.json lighthouse.models / timing): OpenCLIP-H text encoder 340,387,840 parameters, width 1024; denoiser
  865,910,724; VAE 83,653,863 (decoder 49,490,199). 50 steps × 2 passes (CFG 7.5) = 100 denoiser passes (notes §24).
- Tokens: Episode 15's ids, computed by the course coordinator with the SD 2.1-base tokenizer (local cache,
  2026-10-27; hard-coded here: the downloaded models are deleted after the course and are never read by this scene).
  The MLX pipeline pads a batch to its longest prompt (13), not to 77 (mac_runs.md erratum).

Measured on this Mac (research/experiments via measured() and the run's arrays; asserted below). Computed here: the
swirl's 50 DDIM steps (0.19 s in toy.json, plain NumPy on the CPU). Illustrative: the embedding grid's cells are a
picture of the shape (13 columns × 1024 numbers), not the real values.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403
from diffusion_reference import *  # noqa: E402,F403

EP_DIR = Path(__file__).resolve().parent          # set AFTER the import: diffusion_reference exports HERE (the series)
PX = 135                                          # pixels per scene unit at 1080p
IMG, ARR = EXPERIMENTS / "images", EXPERIMENTS / "arrays"
HL = lambda *keys: measured("mac_runs", "headline", *keys)  # noqa: E731
LH = measured("mac_runs", "lighthouse")
MAC_MD = (EXPERIMENTS / "mac_runs.md").read_text()
NOTES = (HERE / "research" / "notes.md").read_text()
SD_REPO = "sd2-community/stable-diffusion-2-1-base"
M = lambda p: f"{p / 1e6:.1f}"  # noqa: E731


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))


# ---------------------------------------------------------------------------- the swirl (computed here)

SW_N, SW_SHOW, SW_STEPS = 2000, 700, 50
SW_RUN = ddim_sample(exact_eps, gaussian_noise(SW_N, seed=5), steps=SW_STEPS)
SW_TOY = measured("toy", "samplers", "exact", "runs", f"ddim_{SW_STEPS}")
assert measured("toy", "samplers", "noise") == "z = dr.gaussian_noise(2000, seed=5)"
assert SW_RUN.xs.shape == (SW_STEPS + 1, SW_N, 2) and SW_RUN.x0s.shape == (SW_STEPS, SW_N, 2)
assert abs(swirl_distance(SW_RUN.xs[-1]) - SW_TOY["dist"]) < 5e-4                 # the same run as toy.json
assert abs(off_swirl(SW_RUN.xs[-1]) - SW_TOY["off"]) < 1e-3 and SW_TOY["nfe"] == SW_STEPS
SW_S = SW_TOY["seconds"]
assert f"{SW_S:.3f}" == "0.185" and 0.15 < SW_S < 0.25                               # "a fifth of a second"
SW_XT = SW_RUN.xs[:, :SW_SHOW]
SW_G = SW_RUN.x0s[:, :SW_SHOW]
SW_EXT = np.maximum.accumulate(np.array([np.abs(p).max() for p in SW_XT])[::-1])[::-1]   # the view only zooms in
SW_GEXT = max(float(np.abs(SW_G).max()), float(SW_EXT[-1]))

# ---------------------------------------------------------------------------- the digit model (trained on this Mac)

PARAMS = HL("digit_unet_params")
TRAIN_MIN = HL("digit_ddpm_train_minutes")
assert PARAMS == 3_940_801 and f"{PARAMS / 1e6:.2f}" == "3.94" and round(TRAIN_MIN) == 32   # "almost four million"
DM = measured("mac_runs", "digit_models", "ddpm")
assert (DM["steps"], DM["batch"]) == (10_000, 256) == (HL("digit_train_steps"), HL("digit_train_batch"))
LOSS = DM["loss_curve"]
LOSS_STEP, LOSS_V = np.array(LOSS["step"], float), np.array(LOSS["loss"], float)
LOSS_0, LOSS_1 = float(LOSS_V[0]), HL("digit_ddpm_final_loss")
assert f"{LOSS_0:.2f}" == "0.81" and f"{LOSS_1:.3f}" == "0.020"                     # "zero point eight one … zero point zero two"
CFID, FLOOR = HL("cfid_ddim50"), HL("cfid_real_data_floor")
assert (f"{CFID:.2f}", f"{FLOOR:.2f}") == ("3.35", "2.06")
LAT_ROWS = {(r["cfg"], r["batch"]): r["s"] for r in measured("mac_runs", "latency", "rows")}
ONE_DIGIT = LAT_ROWS[("ddpm_ddim50_w1", 1)]
PER_1000 = HL("seconds_per_1000_digits_ddim50")
assert f"{ONE_DIGIT:.2f}" == "0.20" and f"{PER_1000:.1f}" == "8.9"                  # "a fifth of a second … about nine"
IMAGES, LABELS = load_mnist("train")
assert len(IMAGES) == 60_000                                                        # "sixty thousand"

TRAJ = np.load(ARR / "digit_trajectory_7_more_seeds.npz")
SEVEN_SEED = 47                                   # the cleanest DDIM-50 seven (mac_runs.md §4); Episode 1's seven
assert int(TRAJ["label"]) == 7 and SEVEN_SEED in list(TRAJ["seeds"])
_key = f"seed{SEVEN_SEED}_ddpm_ddim50_w1"
SEVEN_XT = TRAJ[f"{_key}_x_t"].astype(float)      # x_t[i] is the input to step i (x_T first)
SEVEN_G = TRAJ[f"{_key}_x0_hat"].astype(float)    # the model's clean-image guess at step i
SEVEN = TRAJ[f"{_key}_final"].astype(float)
assert SEVEN_XT.shape == SEVEN_G.shape == (50, 28, 28) and abs(SEVEN_XT[0].std() - 1) < 0.1
SEVEN_FRAMES = list(SEVEN_XT) + [SEVEN]

# ---------------------------------------------------------------------------- the lighthouse (generated on this Mac)

PROMPT_TEXT = "a lighthouse on a rocky coast at dusk, oil painting"
assert f'Prompt: **"{PROMPT_TEXT}"**' in MAC_MD
RUN = LH["frames"]
STEPS, CFG, SEED = RUN["steps"], RUN["cfg"], RUN["seed"]
assert (STEPS, CFG, SEED) == (30, 7.5, 0) and RUN["n_xt_frames"] == STEPS + 1     # "a shorter run of thirty steps"
RUN_S = RUN["timing"]["total_s"]
assert f"{RUN_S:.2f}" == "4.02"
FRAMES = IMG / "lighthouse_frames30"
XT = [load_rgb(FRAMES / f"xt_{k:02d}.png") for k in range(STEPS + 1)]
X0H = [load_rgb(FRAMES / f"x0hat_{k:02d}.png") for k in range(STEPS)]
assert all(f.shape == (512, 512, 3) for f in XT + X0H) and np.array_equal(XT[STEPS], X0H[STEPS - 1])
LAT = np.load(ARR / "lighthouse_latents_30steps.npz")
assert "30 steps, cfg 7.5, seed 0" in str(LAT["note"])
X_T = LAT["x_t"][0].astype(float)                 # x_T for seed 0: the same draw in the 50-step run (gpu/lighthouse.py:
assert X_T.shape == (64, 64, 4) and abs(X_T.std() - 1) < 0.05   # mx.random.seed(seed), then sample_prior, steps unused)
VAE_NPZ = np.load(ARR / "lighthouse_vae_latent.npz")
assert "lighthouse_sd21_cfg7.5_steps50_seed0.png" in str(VAE_NPZ["note"])
Z0 = VAE_NPZ["latent"].astype(float)              # the finished picture's latent (its VAE encoding)
PIC50 = load_rgb(IMG / "lighthouse" / "lighthouse_sd21_cfg7.5_steps50_seed0.png")   # seed 0, 50 steps, guidance 7.5:
assert PIC50.shape == (512, 512, 3)               # the timing run's settings (deterministic Euler sampler)

# one generation, every stage: the 50-step timing run (median of 3)
TM = LH["timing"]["cfg7.5_50steps"]
TXT_S, DEN_S, VAE_S, TOTAL_S = TM["text_encoder_s"], TM["denoise_total_s"], TM["vae_decode_s"], TM["total_s"]
N_STEPS, PASSES = TM["denoise_steps"], TM["unet_passes_per_step"]
assert (N_STEPS, PASSES) == (50, 2) and N_STEPS * PASSES == 100                     # "a hundred passes"
assert round(TOTAL_S, 2) == HL("lighthouse_total_s_50steps_cfg7.5") == 7.32         # "seven point three seconds"
assert abs(TXT_S + DEN_S + VAE_S - TOTAL_S) < 0.01
assert (f"{TXT_S:.3f}", f"{DEN_S:.2f}", f"{VAE_S:.3f}") == ("0.017", "7.04", "0.265")  # "seventeen milliseconds" …
SHARES = [TXT_S / TOTAL_S, DEN_S / TOTAL_S, VAE_S / TOTAL_S]
assert [f"{100 * s:.1f}" for s in SHARES] == ["0.2", "96.2", "3.6"]                 # "ninety-six percent"
assert 0.24 < VAE_S < 0.29                                                          # "about a quarter of a second"
assert TM["latent_shape"] == [1, 64, 64, 4] and TM["image_shape"] == [512, 512, 3]
N_LAT = int(np.prod(TM["latent_shape"]))
N_PIX = int(np.prod(TM["image_shape"]))
assert (N_LAT, N_PIX, N_PIX // N_LAT) == (16_384, 786_432, 48)                     # "sixteen thousand three hundred …"
PARTS = LH["models"][SD_REPO]["parts"]
TE_P, UNET_P = PARTS["text_encoder"]["params"], PARTS["unet"]["params"]
VAE_P, DEC_P = PARTS["vae"]["params"], PARTS["vae"]["decoder_params"]
assert (M(TE_P), M(UNET_P), M(VAE_P), M(DEC_P)) == ("340.4", "865.9", "83.7", "49.5")
assert round(TE_P / 1e6) == 340 and round(UNET_P / 1e6) == 866                    # "three hundred and forty … eight hundred and sixty-six million"
RATIO = UNET_P / PARAMS
assert round(RATIO) == 220                                                          # "two hundred and twenty times fewer"

# the prompt, token by token (Episode 15; SD 2.1-base tokenizer, local cache, read by the coordinator 2026-10-27)
TOKENS = ["<|startoftext|>", "a", "lighthouse", "on", "a", "rocky", "coast", "at", "dusk", ",", "oil", "painting",
          "<|endoftext|>"]
IDS = [49406, 320, 13717, 525, 320, 9648, 3720, 536, 19708, 267, 2870, 3086, 49407]
VOCAB, WIDTH, CONTEXT = 49_408, 1024, 77          # text_encoder/config.json: vocab_size, hidden_size, max positions
N_TOK = len(TOKENS)
WORDS = [t for t in TOKENS[1:-1] if t.isalpha()]
assert N_TOK == len(IDS) == 13 and len(WORDS) == 10 and TOKENS.count(",") == 1     # "ten words and a comma"
assert TOKENS[1:-1] == re.findall(r"[a-z]+|,", PROMPT_TEXT) and IDS[0] == VOCAB - 2 and IDS[-1] == VOCAB - 1
assert "OpenCLIP ViT-H (context dim 1024)" in NOTES                                 # "one thousand and twenty-four"
assert "encoded as **13\ntokens**" in MAC_MD and "pad to 77" in MAC_MD             # the erratum: 13 here, 77 elsewhere
EMB = N_TOK * WIDTH
assert (EMB, 2 * EMB) == (13_312, 26_624)

# ---------------------------------------------------------------------------- the course (Episode 1's map)

assert len(EPISODES) == 24 and len(MODULES) == 6                                    # "twenty-four episodes in six modules"
MODULE_EPS = {k: [n for n, (_, mod) in EPISODES.items() if mod == k] for k in MODULES}
assert [len(v) for v in MODULE_EPS.values()] == [4] * 6
DIGIT_INDEX = int(np.flatnonzero(LABELS == 7)[0])
assert DIGIT_INDEX == 15                                                            # Episode 1's first training 7
X0_DIGIT = IMAGES[DIGIT_INDEX] / 127.5 - 1.0
EPS_DIGIT = np.random.default_rng(4).standard_normal((28, 28))
TEN = np.load(ARR / "digit_same_noise_ten_labels.npz")
assert list(TEN["labels"]) == list(range(10))
TEN_ROW = TEN["ddpm_ddim50_w1"][0]
VIDEO = np.load(ARR / "digit_video_noise_correlation.npz")
CLIP = VIDEO["rho1_canvas"][7]
assert int(VIDEO["labels"][7]) == 7 and CLIP.shape == (8, 28, 52)

# ---------------------------------------------------------------------------- honest findings (Episodes 11, 14, 16, 18)

FLOW_TOY = measured("toy", "flow")
BEND = FLOW_TOY["exact"]["straightness_flow"]["median"]
assert f"{BEND:.2f}" == "1.56" and FLOW_TOY["training_paths"]["straightness_flow"]["median"] == 1.0
FLOW_PATHS = [np.array(p, float) for p in FLOW_TOY["exact"]["paths_flow"]]
assert len(FLOW_PATHS) == 8
CFID_W8, REC_W1, REC_W8 = HL("guidance_w8_cfid"), HL("guidance_w1_recall"), HL("guidance_w8_recall")
assert (f"{CFID_W8:.1f}", f"{REC_W1:.2f}", f"{REC_W8:.2f}") == ("85.4", "0.89", "0.38")
W1 = np.load(ARR / "digit_samples_ddpm_ddim50_w1.npz")
W8 = np.load(ARR / "digit_samples_ddpm_ddim50_w8.npz")
DIT = np.load(ARR / "digit_samples_dit_ddim50_w1.npz")
PICK = list(range(10, 15))                        # samples 10–14 (labels 0–4), the same starting noise in every file
assert all("seed 1234" in str(f["note"]) and list(f["labels"][PICK]) == [0, 1, 2, 3, 4] for f in (W1, W8, DIT))
SDE = LH["sdedit"]
assert SDE["prompt"] == "a lighthouse on a rocky coast at dusk in winter, snow, oil painting"
assert "**no snow appears**" in MAC_MD
WINTER = load_rgb(IMG / "lighthouse" / "lighthouse_sdedit_winter_strength0.9.png")
DIT_CFID, DIT_P = HL("cfid_dit_ddim50"), HL("dit_params")
assert f"{DIT_CFID:.1f}" == "15.9" and f"{DIT_P / 1e6:.2f}" == "4.15"

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


def left_at(mob, x, y):
    return mob.move_to([x + mob.width / 2, y, 0])


_TEX = {}


def texture_of(rgb, px, nearest=False):
    """An RGB array → a read-only RGBA texture px pixels wide (crisp square pixels for small arrays)."""
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

    def __init__(self, left, y, k=0, size=22, fmt="{}", color=TIME):
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


def drive(mob, a0, a1, hooks, run_time=1.4, rate=m.linear):
    """Animate a value from a0 to a1, calling every hook(value) each frame."""

    def step(_, a):
        v = a0 + (a1 - a0) * a
        for hook in hooks:
            hook(v)

    return m.UpdateFromAlphaFunc(mob, step, run_time=run_time, rate_func=rate)


def flash(path, color=INK, width=6, time_width=0.6):
    return m.ShowPassingFlash(path.copy().set_stroke(color, width), time_width=time_width)


def digit_rgb(x, color=DATA, lo=-1, hi=1):
    return tint(x, color, lo, hi)


def seven_rgb(i):
    """Frame i of the seed-47 seven: violet while it is noise, teal by the end (as in Episode 1)."""
    return tint(SEVEN_FRAMES[i], mix(NOISE, DATA, min(1.0, 1.25 * i / (len(SEVEN_FRAMES) - 1))), -1, 1)


def chan_rgb(lat, c, color, lo, hi):
    """One latent channel as a picture in one colour, stretched between lo and hi (as in Episode 17)."""
    return tint(np.clip((lat[..., c] - lo) / (hi - lo), 0, 1), color, 0, 1)


def latent_tile(lat, color, side, lohi):
    """A 64 × 64 × 4 latent as its four channels in a 2 × 2 block `side` units wide."""
    cell = (side - 0.06) / 2
    tiles = m.Group(*[framed(photo(chan_rgb(lat, c, color, *lohi[c]), cell), EDGE, buff=0.0, stroke=1.5) for c in range(4)])
    return tiles.arrange_in_grid(rows=2, cols=2, buff=0.06)


def fit(mob, w, h):
    s = min(w / mob.width, h / mob.height, 1.0)
    return mob.scale(s) if s < 1 else mob


def side_colour(t):
    """Teal while the picture outweighs the noise (SNR > 1), violet once the noise wins (as in Episode 4)."""
    return TEAL if float(snr(alpha_bar_at(t))) > 1 else VIOLET


class SwirlView:
    """The swirl's points in a dark square: `.put(path, i, ext, colour)` draws state i (fractional) of a trajectory,
    zoomed so the points at that state fill the square."""

    def __init__(self, center, side, n, radius=0.022):
        self.center, self.half = np.array(center, float), side / 2 - 0.08
        self.box = m.Square(side).set_stroke(EDGE, 2).set_fill(BG, 1).move_to(center)
        self.dots = m.VGroup(*[m.Dot(radius=radius, color=NOISE) for _ in range(n)])

    def put(self, path, i, ext, colour):
        j = int(np.floor(i))
        k = min(j + 1, len(path) - 1)
        xy = path[j] + (path[k] - path[j]) * (i - j)
        s = self.half / ext
        for d, (x, y) in zip(self.dots, xy):
            d.move_to(self.center + s * np.array([x, y, 0.0]))
            d.set_fill(colour, 0.95)


# ---------------------------------------------------------------------------- the episode


class Ep24OutOfTheNoiseRevisited(Episode):
    number = 24

    def lesson(self):
        self.hook()
        self.three_runs()
        self.every_stage()
        self.end_to_end()
        self.the_course()
        self.honestly()
        self.recap()

    # ------------------------------------------------------------------ 0 · hook

    def hook(self):
        self.next_section("Hook", notes=(
            f"All three running examples from pure noise, side by side, all real. The swirl: {SW_N:,} points of "
            "gaussian_noise(seed=5) walked by ddim_sample(exact_eps, steps=50) — the swirl's exact denoiser, no training "
            f"(diffusion_reference.py); {SW_SHOW} of them drawn; toy.json samplers/exact/ddim_50: dist {SW_TOY['dist']}, "
            f"{100 * SW_TOY['off']:.1f}% stranded between the arms (Episode 10's lesson), asserted here. The digit model: "
            f"seed {SEVEN_SEED}, DDIM 50, w = 1, every x_t (arrays/digit_trajectory_7_more_seeds.npz; mac_runs.md §4 names "
            "it the cleanest DDIM-50 seven; Episode 1's seven). The lighthouse: the 30-step frames run (SD 2.1 base, MLX, "
            f"guidance {CFG}, seed {SEED}; xt_00 … xt_30 decoded; mac_runs.md §5). A shared fraction of the way drives all "
            "three: 50, 50 and 30 steps."))
        W, H = 4.2, 6.0
        self.xs = xs = [-4.45, 0.0, 4.45]
        top = 2.5
        self.cards = [card(W, H, EDGE, CARD, radius=0.2, stroke=2).move_to([x, top - H / 2, 0]).set_z_index(-1)
                      for x in xs]
        ah = 2.3
        ay = 0.62
        self.ay, self.ah = ay, ah
        titles = [sit(title_text(s, 30).move_to([x, 2.12, 0]), 2.0) for s, x in
                  zip(("The swirl", "The digit model", "The lighthouse"), xs)]
        self.titles = titles

        # the swirl
        sw = SwirlView([xs[0], ay, 0], ah, SW_SHOW)
        self.sw = sw
        sw.put(SW_XT, 0, SW_EXT[0], NOISE)
        # the seven: 51 x_t frames, then 50 guesses
        sev_frames = [seven_rgb(i) for i in range(len(SEVEN_FRAMES))] + [digit_rgb(g) for g in SEVEN_G]
        self.seven = Reel(sev_frames, ah, 0)
        seven_art = framed(self.seven.image).move_to([xs[1], ay, 0])
        # the lighthouse: 31 x_t frames, then 30 guesses
        self.lh = Reel(XT + X0H, ah, 0)
        lh_art = framed(self.lh.image).move_to([xs[2], ay, 0])
        self.arts = [m.Group(sw.box, sw.dots), seven_art, lh_art]

        cy = ay - ah / 2 - 0.33
        lefts = [x - W / 2 + 0.3 for x in xs]
        self.counters = [Counter(lefts[0], cy, 0, size=20, fmt="step {} / 50"),
                         Counter(lefts[1], cy, 0, size=20, fmt="step {} / 50"),
                         Counter(lefts[2], cy, 0, size=20, fmt=f"step {{}} / {STEPS}")]
        self.mode_x = [code("x_t", 20, NOISE) for _ in xs]
        self.mode_g = [code("guess", 20, DATA) for _ in xs]
        for x, a, b in zip(xs, self.mode_x, self.mode_g):
            for t in (a, b):
                t.move_to([0, cy, 0]).align_to([x + W / 2 - 0.3, 0, 0], m.RIGHT)
                sit(t, baseline(self.counters[0].words))
        pills = [pill(s, c, CARD, 20).move_to([x, top - H + 0.4, 0]) for s, c, x in
                 zip(("computed in the scene", "trained on this Mac", "generated on this Mac"), (MUTED, INK, INK), xs)]
        self.pills = pills

        def at(f):
            sw.put(SW_XT, f * SW_STEPS, float(np.interp(f * SW_STEPS, np.arange(len(SW_EXT)), SW_EXT)),
                   mix(NOISE, DATA, min(1.0, 1.25 * f)))
            self.seven.show(f * 50)
            self.lh.show(f * STEPS)
            self.counters[0].set(f * 50)
            self.counters[1].set(f * 50)
            self.counters[2].set(f * STEPS)

        self.say(
            "[Twenty-three episodes ago, a lighthouse came out of static.] "
            "[Today, all three running examples do it together, side by side.]",
            m.AnimationGroup(self.new_header("00 · hook", "Three pictures out of static"),
                             m.FadeIn(self.cards[2], scale=0.96), m.FadeIn(lh_art, shift=0.15 * m.UP),
                             m.FadeIn(titles[2])),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(c, scale=0.96) for c in self.cards[:2]], lag_ratio=0.25),
                             m.FadeIn(self.arts[0], shift=0.15 * m.UP), m.FadeIn(seven_art, shift=0.15 * m.UP),
                             m.FadeIn(titles[0]), m.FadeIn(titles[1])),
        )
        self.say(
            "[Each one starts as pure noise]: [random points], [random pixels], [and a random latent.]",
            m.AnimationGroup(*[m.FadeIn(c) for c in self.counters], *[m.FadeIn(t) for t in self.mode_x]),
            m.Circumscribe(sw.box, color=NOISE, buff=0.06),
            m.Circumscribe(seven_art[0], color=NOISE, buff=0.06),
            m.Circumscribe(lh_art[0], color=NOISE, buff=0.06),
        )
        self.say(
            "[Then the same loop runs in all three]: [remove a little noise], [and repeat], "
            "[until only the picture is left.]",
            drive(sw.dots, 0.0, 0.1, [at], run_time=1.0),
            drive(sw.dots, 0.1, 0.35, [at], run_time=1.2),
            drive(sw.dots, 0.35, 0.7, [at], run_time=1.0),
            drive(sw.dots, 0.7, 1.0, [at], run_time=1.6),
        )
        self.say(
            "[A swirl, a seven, and a lighthouse]: [computed, trained and generated, all on this Mac.]",
            m.LaggedStart(m.Circumscribe(sw.box, color=DATA, buff=0.06), m.Circumscribe(seven_art[0], color=DATA, buff=0.06),
                          m.Circumscribe(lh_art[0], color=DATA, buff=0.06), lag_ratio=0.3),
            m.LaggedStart(*[m.FadeIn(p, shift=0.1 * m.UP) for p in pills], lag_ratio=0.3),
        )
        self.wait(0.3)

    # ------------------------------------------------------------------ 1 · three runs

    def three_runs(self):
        self.next_section("1 · Three runs", notes=(
            f"Costs, each from its own run on this Mac. The swirl: 50 DDIM steps with the exact denoiser for {SW_N:,} "
            f"points took {SW_S} s (toy.json samplers/exact/ddim_50, plain NumPy on the CPU; no training). The digit model: "
            f"{PARAMS:,} parameters, trained in {TRAIN_MIN} min; DDIM 50 for one digit {ONE_DIGIT:.3f} s at batch 1, "
            f"{PER_1000} s per 1,000 at batch 1,000 (mac_runs.json latency, digit_sampling). The lighthouse: SD 2.1 base, "
            f"50 guided steps {TOTAL_S:.2f} s (median of 3 runs: 7.35 / 7.32 / 7.31 s); the frames shown are the 30-step "
            f"run, {RUN_S:.2f} s of its own (the 30 preview decodes not counted). The guesses: the swirl's x̂₀ from the "
            "exact denoiser (Trajectory.x0s), the digit model's x̂₀ per step, the lighthouse's decoded x̂₀ previews "
            "(Tweedie's estimate, Episode 6)."))
        xs, ah, ay = self.xs, self.ah, self.ay
        y1, y2, y3, y4 = -1.27, -1.6, -2.12, -2.6
        specs = [
            ("exact denoiser · no training", "DDIM 50 · 2,000 points", f"{SW_S:.3f} s", "for 2,000 points, on the CPU"),
            (f"{PARAMS / 1e6:.2f} M parameters", f"DDIM 50 · seed {SEVEN_SEED}", f"{ONE_DIGIT:.2f} s",
             f"for one digit · 1,000 in {PER_1000:.1f} s"),
            ("Stable Diffusion 2.1 base", f"{STEPS} steps · guidance {CFG}", f"{TOTAL_S:.2f} s",
             f"for 50 steps · these {STEPS}: {RUN_S:.2f} s"),
        ]
        rows = []
        for x, (d1, d2, big, cap) in zip(xs, specs):
            a = text(d1, 20, MUTED).move_to([x, y1, 0])
            b = text(d2, 20, TIME).move_to([x, y2, 0])
            n = title_text(big, 40, INK).move_to([x, y3, 0])
            c = text(cap, 20, MUTED).move_to([x, y4, 0])
            rows.append(dict(detail=m.VGroup(a, b), cost=m.VGroup(n, c)))
            for t in (a, b, n, c):
                assert t.width < 4.2 - 0.4, (t.text, t.width)

        def at_g(f):
            self.sw.put(SW_G, f * (SW_STEPS - 1), SW_GEXT, DATA)
            self.seven.show(len(SEVEN_FRAMES) + f * 49)
            self.lh.show(STEPS + 1 + f * (STEPS - 1))
            self.counters[0].set(1 + f * 49)
            self.counters[1].set(1 + f * 49)
            self.counters[2].set(1 + f * (STEPS - 1))

        self.say(
            "[The swirl needs no training at all]: [its denoiser is exact], "
            "[and fifty steps for two thousand points take a fifth of a second.]",
            m.AnimationGroup(self.new_header("01 · three runs", "Three runs, three costs"),
                             m.FadeIn(rows[0]["detail"][0], shift=0.1 * m.UP)),
            m.FadeIn(rows[0]["detail"][1], shift=0.1 * m.UP),
            m.FadeIn(rows[0]["cost"], shift=0.1 * m.UP),
        )
        self.say(
            "[Our digit model has almost four million parameters]: [one seven takes a fifth of a second], "
            "[and a thousand digits, batched together, about nine seconds.]",
            m.FadeIn(rows[1]["detail"], shift=0.1 * m.UP),
            m.FadeIn(rows[1]["cost"][0], shift=0.1 * m.UP),
            m.FadeIn(rows[1]["cost"][1], shift=0.1 * m.UP),
        )
        self.say(
            "[The lighthouse model is far bigger]: [fifty guided steps take seven point three seconds.] "
            "[The frames you saw came from a shorter run of thirty steps.]",
            m.FadeIn(rows[2]["detail"][0], shift=0.1 * m.UP),
            m.FadeIn(rows[2]["cost"][0], shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(rows[2]["detail"][1], shift=0.1 * m.UP), m.FadeIn(rows[2]["cost"][1])),
        )
        self.say(
            "[And at every step, each denoiser also guesses the finished result]: [blurry at first], "
            "[then sharper and sharper.]",
            m.AnimationGroup(*[m.FadeOut(t) for t in self.mode_x], *[m.FadeIn(t) for t in self.mode_g],
                             drive(self.sw.dots, 0.0, 0.0, [at_g], run_time=0.5)),
            drive(self.sw.dots, 0.0, 0.3, [at_g], run_time=1.4),
            drive(self.sw.dots, 0.3, 1.0, [at_g], run_time=2.2),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 2 · every stage

    def every_stage(self):
        self.next_section("2 · Every stage", notes=(
            "One generation of the lighthouse, stage by stage, with the 50-step timing run's numbers (mac_runs.json "
            f"lighthouse.timing.cfg7.5_50steps: text {TXT_S:.4f} s, denoiser {DEN_S:.3f} s, decode {VAE_S:.4f} s, total "
            f"{TOTAL_S:.3f} s, the median of 3 runs) and the run's own model sizes (lighthouse.models: text encoder "
            f"{TE_P:,}, denoiser {UNET_P:,}, VAE {VAE_P:,} of which the decoder {DEC_P:,}). Tokens: Episode 15's ids "
            f"({', '.join(map(str, IDS))}), SD 2.1-base tokenizer, vocab {VOCAB:,}; the MLX pipeline pads the batch "
            "(prompt + empty negative prompt) to its longest prompt, 13 tokens, while diffusers and the reference code "
            f"pad to {CONTEXT} (mac_runs.md erratum, 2026-10-27). Embeddings 2 × 13 × {WIDTH} = {2 * EMB:,} numbers "
            "(width: notes §15, OpenCLIP ViT-H); the grid is a picture of the shape, not the values. Latents: x_T of seed "
            "0 (the same draw in the 30- and 50-step runs: gpu/lighthouse.py seeds, then samples the prior) and the "
            "finished picture's VAE encoding (arrays/lighthouse_vae_latent.npz), each channel stretched between its "
            "1st and 99th percentile. 50 steps × 2 passes = 100 denoiser passes (batched in pairs: 50 calls of batch 2; notes §24). The picture: "
            "lighthouse_sd21_cfg7.5_steps50_seed0.png, the timing run's settings (deterministic Euler sampler)."))
        # ---- the prompt and its tokens (top)
        TOK_Y = 2.28
        p_txt = code(f'"{PROMPT_TEXT}"', 24, INK)
        p_box = card(p_txt.width + 0.7, 0.7, PROMPT, PINK_FILL, radius=0.14, stroke=2)
        prompt = m.VGroup(p_box, p_txt.move_to(p_box)).move_to([0, TOK_Y, 0])
        shown = ["<start>"] + [t if t != "," else '","' for t in TOKENS[1:-1]] + ["<end>"]
        ref = code("Mx", 20)
        cols = []
        for word, i in zip(shown, IDS):
            w = code(word, 20, INK)
            num = code(str(i), 20, MUTED)
            box = card(max(w.width, num.width) + 0.16, ref.height + 0.3, PROMPT, PINK_FILL, radius=0.08, stroke=1.8)
            w.move_to(box)
            num.next_to(box, m.DOWN, buff=0.13)
            cols.append(m.VGroup(box, w, num))
        toks = m.VGroup(*cols).arrange(m.RIGHT, buff=0.05, aligned_edge=m.UP)
        toks.move_to([0, 0, 0]).shift((TOK_Y - toks[0][0].get_y()) * m.UP)
        assert toks.width < 12.9, toks.width
        chips = m.VGroup(*[c[:2] for c in cols])
        nums = m.VGroup(*[c[2] for c in cols])
        pad_note = cite(f"the reference code and diffusers pad to {CONTEXT}; our MLX run kept {N_TOK} · Episode 15")
        pad_note.move_to([0, 1.36, 0]).align_to([6.5, 0, 0], m.RIGHT)

        # ---- the text encoder and the embeddings (middle)
        MID = 0.62
        te = m.VGroup(card(2.45, 1.0, PROMPT, PINK_FILL), text("text encoder", 24, INK, weight=m.BOLD),
                      code(f"{M(TE_P)} M", 20, MUTED))
        m.VGroup(te[1], te[2]).arrange(m.DOWN, buff=0.1).move_to(te[0])
        te.move_to([-5.55, MID, 0])
        te_in = arrow([-5.55, nums.get_bottom()[1] - 0.02, 0], [-5.55, te[0].get_top()[1], 0], PROMPT, buff=0.04)
        te_time = code(f"{TXT_S:.3f} s, once", 20, PROMPT).next_to(te, m.DOWN, buff=0.16)
        cell, gap = 0.12, 0.03
        grid = m.VGroup(*[m.VGroup(*[m.Square(cell).set_stroke(PROMPT, 1.2).set_fill(PINK_FILL, 1) for _ in range(5)])
                          .arrange(m.DOWN, buff=gap) for _ in range(N_TOK)]).arrange(m.RIGHT, buff=gap)
        grid.move_to([-2.55, MID + 0.05, 0])
        back = grid.copy().set_stroke(opacity=0.45).shift(0.12 * m.UP + 0.12 * m.RIGHT).set_z_index(-1)
        emb_dots = code("⋮", 20, PROMPT).next_to(grid, m.DOWN, buff=0.04)
        emb_in = arrow(te[0].get_right(), grid.get_left() + [-0.02, 0, 0], PROMPT, buff=0.08)
        e1 = code(f"{N_TOK} × {WIDTH}", 22, PROMPT, weight=m.BOLD)
        e2 = text("numbers per prompt", 20, MUTED)
        e3 = text("× 2, with an empty prompt", 20, MUTED)
        elab = m.VGroup(e1, e2, e3).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.08)
        elab.move_to([0, MID, 0]).align_to([grid.get_right()[0] + 0.45, 0, 0], m.LEFT)

        # ---- noise → denoiser → latent → decoder → picture (bottom)
        BOT = -1.45
        lo_n = [(-2.5, 2.5)] * 4
        lo_z = [tuple(np.percentile(Z0[..., c], [1, 99])) for c in range(4)]
        noise = latent_tile(X_T, NOISE, 1.5, lo_n).move_to([-5.55, BOT, 0])
        n_l1 = code("64×64×4", 20, NOISE).next_to(noise, m.DOWN, buff=0.16)
        n_l2 = cite(f"{N_LAT:,} numbers").next_to(n_l1, m.DOWN, buff=0.1)
        den = m.VGroup(card(2.6, 1.5, MODEL, AMBER_FILL), text("denoiser", 26, INK, weight=m.BOLD),
                       code(f"{M(UNET_P)} M", 20, MUTED), code(f"guidance {CFG}", 20, PROMPT))
        m.VGroup(*den[1:]).arrange(m.DOWN, buff=0.08).move_to(den[0])
        den.move_to([-2.55, BOT, 0])
        go = arrow(noise.get_right(), den[0].get_left(), NOISE, buff=0.12)
        cond = arrow(emb_dots.get_bottom(), den[0].get_top(), PROMPT, buff=0.06)
        loop = m.ArcBetweenPoints(den[0].get_corner(m.DR) + [-0.25, -0.08, 0], den[0].get_corner(m.DL) + [0.25, -0.08, 0],
                                  angle=-0.45 * PI_).set_stroke(TIME, 3)
        loop.add_tip(tip_length=0.2)
        loop.set_color(TIME)
        _fmt = "step {} / 50 · " + f"{PASSES} passes"
        _w = code(_fmt.format(50), 20, TIME, weight=m.BOLD).width
        step_ctr = Counter(-2.55 - _w / 2, loop.get_bottom()[1] - 0.24, 0, size=20, fmt=_fmt)
        calls = code(f"{N_STEPS * PASSES} passes", 20, MODEL)
        den_time = code(f"{DEN_S:.2f} s · {100 * SHARES[1]:.1f}%", 20, MODEL)
        m.VGroup(calls, den_time).arrange(m.RIGHT, buff=0.4).move_to([-2.55, step_ctr.get_y() - 0.34, 0])
        assert den_time.get_bottom()[1] > -3.5 and step_ctr.get_left()[0] > n_l2.get_right()[0] + 0.1
        z = latent_tile(Z0, DATA, 1.5, lo_z).move_to([0.55, BOT, 0])
        out = arrow(den[0].get_right(), z.get_left(), DATA, buff=0.12)
        z_l1 = code("64×64×4", 20, DATA).next_to(z, m.DOWN, buff=0.16)
        dec = m.VGroup(card(2.05, 1.2, DATA, TEAL_FILL), text("decoder", 24, INK, weight=m.BOLD), code(f"{M(DEC_P)} M", 20, MUTED))
        m.VGroup(dec[1], dec[2]).arrange(m.DOWN, buff=0.08).move_to(dec[0])
        dec.move_to([2.95, BOT, 0])
        to_dec = arrow(z.get_right(), dec[0].get_left(), DATA, buff=0.1)
        pic = framed(photo(PIC50, 1.6), DATA).move_to([5.55, BOT, 0])
        to_pic = arrow(dec[0].get_right(), pic.get_left(), DATA, buff=0.1)
        p_l1 = code("512×512×3", 20, DATA).next_to(pic, m.DOWN, buff=0.16)
        p_l2 = cite(f"{N_PIX:,} numbers").next_to(p_l1, m.DOWN, buff=0.1)
        dec_time = code(f"{VAE_S:.3f} s", 20, DATA).next_to(dec, m.DOWN, buff=0.16)
        dec_note = cite(f"of the VAE's {M(VAE_P)} M").next_to(dec_time, m.DOWN, buff=0.1)
        assert dec_note.get_left()[0] > z_l1.get_right()[0] + 0.2 and dec_note.get_right()[0] < p_l2.get_left()[0] - 0.2

        # ---- the total (middle right)
        total = title_text(f"{TOTAL_S:.2f} s", 48, INK)
        t_cap = text("in all", 24, MUTED).next_to(total, m.RIGHT, buff=0.2).align_to(total, m.DOWN)
        tot = m.VGroup(total, t_cap)
        left_at(tot, 2.35, 0.82)
        bar_w, bar_y, bar_x0 = 4.0, 0.28, 2.35
        segs = m.VGroup()
        x = bar_x0
        for s_, col in zip(SHARES, (PROMPT, MODEL, DATA)):
            w = max(bar_w * s_, 0.03)
            segs.add(m.Rectangle(width=w, height=0.16).set_stroke(width=0).set_fill(col, 1).move_to([x + w / 2, bar_y, 0]))
            x += w
        ep23 = pill("Ep 23: where the time goes", LIME, LIME_FILL, 20)
        left_at(ep23, 2.35, -0.2)
        assert ep23.get_right()[0] < 6.6 and tot.get_top()[1] < pad_note.get_bottom()[1] - 0.05
        made = cite(f"generated on this Mac · SD 2.1 base · seed {SEED}").move_to([0, -3.32, 0])
        made.align_to([6.5, 0, 0], m.RIGHT)
        assert made.get_left()[0] > den_time.get_right()[0] + 0.3, (made.get_left()[0], den_time.get_right()[0], den_time.get_y(), made.get_y())

        def tick(k):
            step_ctr.set(k)

        self.say(
            "[Now let's follow one lighthouse through every stage], [with the real sizes from our run.]",
            m.AnimationGroup(self.new_header("02 · every stage", "One prompt, every stage"), m.FadeIn(p_box, scale=0.95)),
            m.AddTextLetterByLetter(p_txt, time_per_char=0.025),
        )
        self.say(
            "[The prompt becomes thirteen tokens]: [ten words and a comma], [plus a start and an end.]",
            m.AnimationGroup(m.FadeOut(prompt, shift=0.1 * m.UP),
                             m.LaggedStart(*[m.FadeIn(c, shift=0.1 * m.DOWN) for c in chips], lag_ratio=0.08)),
            m.LaggedStart(*[m.FadeIn(n_, shift=0.05 * m.DOWN) for n_ in nums], lag_ratio=0.06),
            m.AnimationGroup(m.Circumscribe(chips[0], color=PROMPT, buff=0.05), m.Circumscribe(chips[-1], color=PROMPT, buff=0.05)),
        )
        self.say(
            "[The reference code pads a prompt to seventy-seven tokens]; [our run kept just thirteen.]",
            m.FadeIn(pad_note, shift=0.1 * m.UP),
            m.Circumscribe(toks, color=PROMPT, buff=0.08),
        )
        self.say(
            "[A text encoder with three hundred and forty million parameters] "
            "[turns each token into one thousand and twenty-four numbers.]",
            m.AnimationGroup(m.GrowArrow(te_in), m.FadeIn(te, shift=0.1 * m.DOWN)),
            m.AnimationGroup(m.GrowArrow(emb_in), m.LaggedStart(*[m.FadeIn(c) for c in grid], lag_ratio=0.08),
                             m.FadeIn(emb_dots), m.FadeIn(e1), m.FadeIn(e2)),
        )
        self.say(
            "[It runs once, for the prompt and for an empty prompt], [in seventeen milliseconds.]",
            m.AnimationGroup(m.FadeIn(back, shift=0.1 * m.UP + 0.1 * m.RIGHT), m.FadeIn(e3)),
            m.FadeIn(te_time, shift=0.1 * m.UP),
        )
        self.say(
            "[Then the static]: [a latent of sixty-four by sixty-four by four], "
            "[sixteen thousand three hundred and eighty-four random numbers.]",
            m.LaggedStart(*[m.FadeIn(t, scale=0.9) for t in noise], lag_ratio=0.15),
            m.FadeIn(n_l1, shift=0.1 * m.UP),
            m.FadeIn(n_l2, shift=0.1 * m.UP),
        )
        self.say(
            "[The denoiser, with eight hundred and sixty-six million parameters], [reads the latent and the words together.]",
            m.AnimationGroup(m.FadeIn(den, shift=0.1 * m.UP), m.GrowArrow(go)),
            m.AnimationGroup(m.GrowArrow(cond), m.Succession(m.Wait(0.4), m.AnimationGroup(pulse(go, NOISE), pulse(cond, PROMPT))),
                             den.animate(rate_func=m.there_and_back, run_time=1.2).scale(1.04)),
        )
        self.say(
            "[Fifty steps, with two passes each for guidance]: [a hundred passes through the denoiser], "
            "[and ninety-six percent of the time.]",
            m.Succession(m.AnimationGroup(m.Create(loop), m.FadeIn(step_ctr), run_time=0.6),
                         m.AnimationGroup(drive(loop, 0, N_STEPS, [tick], run_time=2.2), flash(loop, TIME, 6), run_time=2.2),
                         m.AnimationGroup(m.GrowArrow(out), m.FadeIn(z, shift=0.1 * m.RIGHT), m.FadeIn(z_l1), run_time=0.7)),
            m.FadeIn(calls, shift=0.1 * m.UP),
            m.FadeIn(den_time, shift=0.1 * m.UP),
        )
        self.say(
            "[Last, the decoder paints the pixels, just once]: [five hundred and twelve by five hundred and twelve by three], "
            "[in about a quarter of a second.]",
            m.AnimationGroup(m.GrowArrow(to_dec), m.FadeIn(dec, shift=0.1 * m.UP)),
            m.AnimationGroup(m.GrowArrow(to_pic), m.FadeIn(pic, shift=0.1 * m.LEFT), m.FadeIn(p_l1), m.FadeIn(p_l2)),
            m.AnimationGroup(m.FadeIn(dec_time, shift=0.1 * m.UP), m.FadeIn(dec_note)),
        )
        self.say(
            "[Seven point three seconds in all], [and episode twenty-three takes them apart, stage by stage.]",
            m.AnimationGroup(m.FadeIn(tot, shift=0.1 * m.UP), m.LaggedStart(*[m.GrowFromEdge(s, m.LEFT) for s in segs], lag_ratio=0.3),
                             m.FadeIn(made)),
            m.FadeIn(ep23, shift=0.1 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 3 · the digit model end to end

    def end_to_end(self):
        self.next_section("3 · End to end", notes=(
            f"The digit model's whole life on this Mac (mac_runs.md §2–4): MNIST, 60,000 training digits of 28 × 28 "
            f"(load_mnist; the nine shown are training digits 0–8); a {PARAMS:,}-parameter class-conditional U-Net, "
            f"ε-prediction, {DM['steps']:,} steps × {DM['batch']} (42.7 epochs), {TRAIN_MIN} min, MLX; loss curve = mean "
            f"of each 50 steps, {LOSS_0:.3f} → {LOSS_1} (log scale). Sampling: seed {SEVEN_SEED}, label 7, DDIM 50, "
            f"{ONE_DIGIT:.3f} s at batch 1. Measuring: the classifier-based FID-like score over 10,000 digits (FID's "
            f"formula on our own MNIST classifier's 128-d features; not Inception FID; Episode 19): {CFID} vs the real-data "
            f"floor {FLOOR}; bootstrap std ≈ 0.3. Size: the lighthouse's denoiser has {UNET_P:,} parameters, "
            f"{RATIO:.1f}× ours."))
        W, H = 2.95, 4.75
        xs = [-5.055, -1.685, 1.685, 5.055]
        top = 2.3
        cy = top - H / 2
        names = [("Data", DATA), ("Training", MODEL), ("Sampling", TIME), ("Measuring", GREEN)]
        boxes = [card(W, H, col, CARD, radius=0.2, stroke=2.5).move_to([x, cy, 0]) for x, (_, col) in zip(xs, names)]
        heads = [sit(title_text(n, 30, col).move_to([x, top - 0.42, 0]), top - 0.55) for x, (n, col) in zip(xs, names)]
        art_y = top - 1.85
        arrows_ = [arrow([xs[i] + W / 2, art_y, 0], [xs[i + 1] - W / 2, art_y, 0], MUTED, buff=0.04) for i in range(3)]

        def lines(x, rows, y0=top - 3.38, size=20):
            g = m.VGroup()
            for k, (s, col, mono) in enumerate(rows):
                t = code(s, size, col) if mono else text(s, size, col)
                g.add(t.move_to([x, y0 - 0.36 * k, 0]))
            return g

        # data
        nine = m.Group(*[framed(photo(digit_rgb(IMAGES[i], DATA, 0, 255), 0.66), EDGE, buff=0.02, stroke=1.5) for i in range(9)])
        nine.arrange_in_grid(rows=3, cols=3, buff=0.07).move_to([xs[0], art_y, 0])
        t_data = lines(xs[0], [("60,000 handwritten", INK, False), ("digits, 28 × 28", INK, False)])
        # training
        pl = Plot((0, 10_000), (0.015, 1.0), width=1.85, height=1.45, center=(xs[1] + 0.2, art_y - 0.12, 0), log_y=True,
                  x_ticks=[], y_ticks=[0.02, 1], y_fmt="{:g}")
        curve = pl.line(LOSS_STEP, LOSS_V, MODEL, 3)
        t_train = lines(xs[1], [(f"{DM['steps']:,} steps × {DM['batch']}", INK, False), (f"{TRAIN_MIN} minutes", MUTED, False),
                                (f"loss {LOSS_0:.2f} → {LOSS_1:.3f}", MODEL, False)])
        # sampling
        reel = Reel([seven_rgb(i) for i in range(len(SEVEN_FRAMES))], 1.8, 0)
        s_art = framed(reel.image)
        lab7 = chip("7", PROMPT, PINK_FILL, width=0.55, height=0.55, size=28)
        m.Group(lab7, s_art).arrange(m.RIGHT, buff=0.15).move_to([xs[2], art_y, 0])
        t_samp = lines(xs[2], [(f"DDIM · 50 steps", TIME, True), (f"seed {SEVEN_SEED}, label 7", MUTED, False),
                               (f"{ONE_DIGIT:.2f} s per digit", INK, False)])
        # measuring
        score = title_text(f"{CFID:.2f}", 64, GREEN).move_to([xs[3], art_y + 0.2, 0])
        real = text(f"real digits: {FLOOR:.2f}", 22, TEAL).next_to(score, m.DOWN, buff=0.22)
        t_meas = lines(xs[3], [("FID-like score,", INK, False), ("our own classifier,", MUTED, False),
                               ("lower is better", MUTED, False)])
        for x, g in zip(xs, (t_data, t_train, t_samp, t_meas)):
            assert g.width < W - 0.3, (x, [(t.text, round(t.width, 2)) for t in g])
        foot = cite(f"trained and run on this Mac · {RATIO:.0f}× fewer parameters than the lighthouse's denoiser: "
                    f"{PARAMS / 1e6:.2f} M vs {M(UNET_P)} M")
        foot.move_to([0, -3.05, 0])
        assert foot.width < 13.0

        def play_seven(k):
            reel.show(k)

        self.say(
            "[Our digit model lived its whole life on this Mac]: [data], [training], [sampling], [and measuring.]",
            self.new_header("03 · end to end", "The digit model, start to finish"),
            m.AnimationGroup(m.FadeIn(boxes[0], scale=0.96), m.FadeIn(heads[0])),
            m.AnimationGroup(m.GrowArrow(arrows_[0]), m.FadeIn(boxes[1], scale=0.96), m.FadeIn(heads[1])),
            m.AnimationGroup(m.GrowArrow(arrows_[1]), m.FadeIn(boxes[2], scale=0.96), m.FadeIn(heads[2])),
            m.AnimationGroup(m.GrowArrow(arrows_[2]), m.FadeIn(boxes[3], scale=0.96), m.FadeIn(heads[3])),
        )
        self.say(
            "[Sixty thousand handwritten digits], [and a network learning to predict the noise added to them]: "
            "[ten thousand steps in thirty-two minutes.]",
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(d, scale=0.9) for d in nine], lag_ratio=0.08), m.FadeIn(t_data)),
            m.AnimationGroup(m.FadeIn(pl), m.Create(curve, run_time=2.0)),
            m.FadeIn(t_train[:2], shift=0.1 * m.UP),
        )
        self.say(
            "[The loss fell from zero point eight one to zero point zero two.] "
            "[Then fifty steps turn noise into the seven we asked for.]",
            m.AnimationGroup(m.FadeIn(t_train[2], shift=0.1 * m.UP), m.Circumscribe(curve, color=MODEL, buff=0.06)),
            m.AnimationGroup(m.FadeIn(s_art), m.FadeIn(lab7, scale=0.8), m.FadeIn(t_samp),
                             drive(reel.image, 0, len(SEVEN_FRAMES) - 1, [play_seven], run_time=2.4)),
        )
        self.say(
            "[Finally, our own classifier scores the digits at three point three five], "
            "[close to real data's two point zero six.]",
            m.AnimationGroup(m.FadeIn(score, shift=0.15 * m.UP), m.FadeIn(t_meas)),
            m.FadeIn(real, shift=0.1 * m.UP),
        )
        self.say(
            "[It is the same idea as the lighthouse's denoiser], [with two hundred and twenty times fewer parameters.]",
            m.LaggedStart(*[m.Circumscribe(b, color=c, buff=0.05) for b, (_, c) in zip(boxes, names)], lag_ratio=0.2),
            m.FadeIn(foot, shift=0.1 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 4 · the course revisited

    def the_course(self):
        self.next_section("4 · The course", notes=(
            "Episode 1's map, revisited: six modules, 24 episodes, in the module colours (COURSE.md; kit MODULES/EPISODES), "
            "each card with its key idea. Card art as in Episode 1, all real: Module 1, MNIST training 7 #15 noised with "
            "q_sample at t = 0, 250, 1000; Module 2, the digit model's U-Net shape (28 → 14 → 7, 32/64/128 channels); "
            "Module 3, the lighthouse at 1, 4 and 50 steps (seed 0, guidance 7.5); Module 4, one starting noise (seed 100) "
            "turned into 0 … 9 by the label (DDIM 50); Module 5, the lighthouse and its 64 × 64 × 4 latent; Module 6, three "
            "frames of the moving-digit clip (ρ = 1, Episode 22). Key ideas: the forward process jumps to any t in one step "
            "(Ep 4); predicting ε is learning the score, score = −ε/√(1−ᾱ) (Ep 6); DDIM, solvers, flow matching and "
            "distillation cut 1,000 steps to 50, 10, 1 (Ep 9–12); labels, classifier-free guidance and text "
            "cross-attention steer (Ep 13–16); latents, transformers and data scale it (Ep 17–20); frames denoised "
            "together, and run on this Mac (Ep 21–24)."))
        W, H = 4.2, 2.85
        centres = [(-4.45, 1.07), (0.0, 1.07), (4.45, 1.07), (-4.45, -1.98), (0.0, -1.98), (4.45, -1.98)]
        names = ["Pictures from Noise", "Learning to Denoise", "Faster Sampling", "Steering the Picture", "Scaling Up",
                 "Video and Your Mac"]
        ideas = ["noise jumps in, in one step", "predict noise, learn the score", "1,000 steps → 50 → 10 → 1",
                 "labels, guidance and words", "latents, transformers and data", "frames together, on this Mac"]
        for k, n in enumerate(names, start=1):
            assert n.upper() == MODULES[k][0]

        a1 = m.Group(*[framed(photo(tint(q_sample(X0_DIGIT, t, EPS_DIGIT), side_colour(t), -1, 1), 0.76)) for t in (0, 250, 1000)])
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
        a5_l = framed(photo(PIC50, 0.78))
        a5_z = framed(photo(tint(lat_img, DATA, 0, 255), 0.42, nearest=False))
        a5_arrow = arrow([0, 0, 0], [0.45, 0, 0], MUTED, buff=0)
        a5_lab = code("64 × 64 × 4", 20, DATA)
        a5 = m.Group(a5_l, a5_arrow, m.Group(a5_z, a5_lab)).arrange(m.RIGHT, buff=0.15)
        a5_lab.next_to(a5_z, m.DOWN, buff=0.08)
        a6 = m.Group(*[framed(photo(digit_rgb(CLIP[f], DATA, 0, 255), 0.62)) for f in (0, 3, 7)]).arrange(m.RIGHT, buff=0.08)
        arts = [a1, a2, a3, a4, a5, a6]

        cards_ = []
        for k, ((cx, cy), name, idea, art) in enumerate(zip(centres, names, ideas, arts), start=1):
            _, acc, fill = MODULES[k]
            box = card(W, H, acc, CARD, radius=0.2, stroke=2.5).move_to([cx, cy, 0])
            top, left, right = cy + H / 2, cx - W / 2, cx + W / 2
            tag = pill(f"MODULE {k}", acc, fill, 22, mono=True)
            tag.move_to([left + 0.25 + tag.width / 2, top - 0.38, 0])
            eps = MODULE_EPS[k]
            rng = code(f"Ep {eps[0]}–{eps[-1]}", 20, MUTED).move_to([0, top - 0.38, 0]).align_to([right - 0.25, 0, 0], m.RIGHT)
            fit(art, W - 0.5, 0.9).move_to([cx, top - 1.16, 0])
            t = title_text(name, 28).move_to([0, top - 2.02, 0]).align_to([left + 0.28, 0, 0], m.LEFT)
            s = text(idea, 20, INK).next_to(t, m.DOWN, buff=0.1).align_to([left + 0.28, 0, 0], m.LEFT)
            assert s.get_right()[0] < right - 0.2, (idea, s.width)
            cards_.append(dict(box=box, tag=tag, rng=rng, art=art, title=t, sub=s, accent=acc))

        mascot = pixel(0.42).move_to([6.15, 3.18, 0])

        def content(c):
            return m.AnimationGroup(*[m.FadeIn(p, shift=0.1 * m.UP) for p in (c["title"], c["sub"], c["art"])], lag_ratio=0.15)

        self.say(
            "[Episode one promised this map]: [twenty-four episodes in six modules], [all built around one loop.]",
            m.AnimationGroup(self.new_header("04 · the course", "Six modules, one loop"), m.FadeIn(mascot, shift=0.1 * m.DOWN)),
            m.LaggedStart(*[m.AnimationGroup(m.FadeIn(c["box"], scale=0.94), m.FadeIn(c["tag"]), m.FadeIn(c["rng"]))
                            for c in cards_], lag_ratio=0.15),
            m.LaggedStart(*[m.Circumscribe(c["box"], color=c["accent"], buff=0.04) for c in cards_], lag_ratio=0.12),
        )
        self.say(
            "[Pictures from noise]: [a picture is a point in a huge space], [and noise can be added in a single jump.]",
            m.AnimationGroup(m.FadeIn(cards_[0]["title"], shift=0.1 * m.UP), m.FadeIn(a1[0], shift=0.1 * m.UP)),
            m.FadeIn(cards_[0]["sub"], shift=0.1 * m.UP),
            m.LaggedStart(*[m.FadeIn(f, shift=0.1 * m.RIGHT) for f in a1[1:]], lag_ratio=0.4),
        )
        self.say(
            "[Learning to denoise]: [predict the noise, and you have learned the score.]",
            m.AnimationGroup(m.FadeIn(cards_[1]["title"], shift=0.1 * m.UP), m.FadeIn(a2)),
            m.AnimationGroup(m.FadeIn(cards_[1]["sub"], shift=0.1 * m.UP), a2.flow(MODEL, run_time=1.4)),
        )
        self.say(
            "[Faster sampling]: [from a thousand steps to fifty, ten, or even one.]",
            m.FadeIn(cards_[2]["title"], shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(cards_[2]["sub"], shift=0.1 * m.UP),
                             m.LaggedStart(*[m.FadeIn(f, shift=0.1 * m.UP) for f in a3[::-1]], lag_ratio=0.35)),
        )
        self.say(
            "[Steering, with labels, guidance and words], [and scaling up, with latents, transformers and data.]",
            content(cards_[3]),
            content(cards_[4]),
        )
        self.say(
            "[And finally video], [and running it all on this Mac.]",
            m.AnimationGroup(m.FadeIn(cards_[5]["title"], shift=0.1 * m.UP), m.FadeIn(a6, shift=0.1 * m.UP)),
            m.FadeIn(cards_[5]["sub"], shift=0.1 * m.UP),
        )
        self.say(
            "[Every episode was about one of three things]: [learning the denoiser], [sampling with it], [or scaling it up.]",
            m.Wait(0.1),
            m.Circumscribe(cards_[1]["box"], color=MODEL, buff=0.05),
            m.AnimationGroup(m.Circumscribe(cards_[2]["box"], color=BLUE, buff=0.05),
                             m.Circumscribe(cards_[3]["box"], color=PINK, buff=0.05)),
            m.AnimationGroup(m.Circumscribe(cards_[4]["box"], color=TEAL, buff=0.05),
                             m.Circumscribe(cards_[5]["box"], color=LIME, buff=0.05)),
        )
        self.wait(0.6)
        self.clear()

    # ------------------------------------------------------------------ 5 · honestly

    def honestly(self):
        self.next_section("5 · Honestly", notes=(
            f"Four findings the course kept in. Ep 11 (toy.md §4): flow matching trains on straight lines (length ÷ chord "
            f"1.00), yet the exact flow field's sampling paths bend, median {BEND} (the eight drawn: toy.json "
            "flow/exact/paths_flow, with their straight chords dashed). Ep 14 (mac_runs.md §4, DDIM 50, same noise): from "
            f"w = 1 to w = 8 the FID-like score goes {CFID:.2f} → {CFID_W8:.1f}, recall {REC_W1:.3f} → {REC_W8}; samples "
            "10–14 (labels 0–4). Ep 16 (mac_runs.md §5): SDEdit with '… at dusk in winter, snow, oil painting' at "
            "strengths 0.3–0.9: no snow at any strength (strength 0.9 shown next to its source); a more explicit prompt "
            f"did make snow. Ep 18 (mac_runs.md §4): a {DIT_P:,}-parameter tiny DiT, same data, steps and optimiser: "
            f"{DIT_CFID:.2f} vs the U-Net's {CFID:.2f}; samples 10–14 from the same noise. One training seed per model; "
            "scores are our classifier's FID-like score (lower is better), not Inception FID."))
        W, H = 6.3, 2.45
        centres = [(-3.375, 1.0), (3.375, 1.0), (-3.375, -1.7), (3.375, -1.7)]
        AW = 2.1

        # art: Ep 11 — exact flow paths on the swirl
        def flow_art():
            sz = 1.95
            box = m.Square(sz).set_stroke(EDGE, 1.5).set_fill(BG, 1)
            ext = max(float(np.abs(np.concatenate(FLOW_PATHS)).max()), 1.05)
            s = (sz / 2 - 0.08) / ext
            curve = swirl_curve(200)
            arms = m.VGroup(*[m.VMobject().set_points_smoothly([np.array([x * s, y * s, 0]) for x, y in arm])
                              .set_stroke(DATA, 2, opacity=0.35) for arm in curve])
            paths, chords = m.VGroup(), m.VGroup()
            for p in FLOW_PATHS:
                pts = [np.array([x * s, y * s, 0]) for x, y in p]
                paths.add(m.VMobject().set_points_as_corners(pts).set_stroke(FLOW, 3))
                chords.add(m.DashedLine(pts[0], pts[-1], dash_length=0.07).set_stroke(INK, 2, opacity=0.7))
            return m.VGroup(box, arms, chords, paths)

        def digit_rows(a, b, la, lb, ca=DATA, cb=DATA):
            out = m.Group()
            for arr, lab, col in ((a, la, ca), (b, lb, cb)):
                row = m.Group(*[framed(photo(digit_rgb(arr[i], col, 0, 255), 0.36), EDGE, buff=0.015, stroke=1.2) for i in PICK])
                row.arrange(m.RIGHT, buff=0.04)
                tag = code(lab, 20, MUTED)
                out.add(m.Group(tag, row).arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT))
            return out.arrange(m.DOWN, buff=0.16, aligned_edge=m.LEFT)

        def lh_art():
            src = framed(photo(PIC50, 1.0))
            win = framed(photo(WINTER, 1.0))
            return m.Group(m.Group(src, code("before", 20, MUTED).next_to(src, m.DOWN, buff=0.08)),
                           m.Group(win, code("'in winter'", 20, PROMPT).next_to(win, m.DOWN, buff=0.08))).arrange(m.RIGHT, buff=0.14)

        arts = [flow_art(), digit_rows(W1["samples"], W8["samples"], "w = 1", "w = 8"), lh_art(),
                digit_rows(W1["samples"], DIT["samples"], "denoiser", "transformer")]
        specs = [
            (11, "Flow paths still bend", ["trained on straight lines,", f"paths {BEND:.2f}× longer (median)"]),
            (14, "Bolder, but worse", ["bolder strokes, worse score:", f"{CFID:.2f} → {CFID_W8:.1f} at w = 8"]),
            (16, "No snow in winter", ["'… in winter, snow' made", "no snow at any strength"]),
            (18, "A transformer lost", ["same size, same budget:", f"{DIT_CFID:.1f} against {CFID:.2f}"]),
        ]
        tiles = []
        for (cx, cy), art, (ep, title, facts) in zip(centres, arts, specs):
            _, acc, fill = MODULES[EPISODES[ep][1]]
            box = card(W, H, EDGE, CARD, radius=0.2, stroke=2).move_to([cx, cy, 0])
            left, top = cx - W / 2, cy + H / 2
            fit(art, AW, H - 0.4).move_to([left + 0.25 + AW / 2, cy, 0])
            tx = left + AW + 0.42
            tag = pill(f"Ep {ep}", acc, fill, 20, mono=True)
            left_at(tag, tx, top - 0.42)
            t = left_at(title_text(title, 24), tx, top - 0.98)
            f = m.VGroup(*[text(s, 20, MUTED) for s in facts]).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.08)
            left_at(f, tx, top - 1.62)
            assert f.get_right()[0] < cx + W / 2 - 0.15 and t.get_right()[0] < cx + W / 2 - 0.15, (title, round(f.width, 2), round(t.width, 2), round(f.get_right()[0] - cx, 2), round(t.get_right()[0] - cx, 2))
            tiles.append(dict(box=box, art=art, head=m.VGroup(tag, t), facts=f, accent=acc))
        foot = cite("scores: our classifier's FID-like score, lower is better · one training seed per model")
        foot.move_to([0, -3.3, 0])

        def show(tl):
            return m.AnimationGroup(m.FadeIn(tl["art"], shift=0.1 * m.UP), m.FadeIn(tl["head"], shift=0.1 * m.UP),
                                    m.FadeIn(tl["facts"]), lag_ratio=0.2)

        self.say(
            "[We also kept the surprises], [because small, honest runs teach what papers leave out.]",
            m.AnimationGroup(self.new_header("05 · honestly", "What surprised us"),
                             m.LaggedStart(*[m.FadeIn(tl["box"], scale=0.96) for tl in tiles], lag_ratio=0.15)),
            m.FadeIn(foot, shift=0.1 * m.UP),
        )
        self.say(
            "[Flow matching trains on straight lines], [yet on the swirl, its paths still bend.]",
            m.AnimationGroup(m.FadeIn(tiles[0]["art"][:3]), m.FadeIn(tiles[0]["head"], shift=0.1 * m.UP)),
            m.AnimationGroup(m.LaggedStart(*[m.Create(p) for p in tiles[0]["art"][3]], lag_ratio=0.1, run_time=1.6),
                             m.FadeIn(tiles[0]["facts"])),
        )
        self.say(
            "[Guidance made our digits bolder], [but their score got far worse.]",
            show(tiles[1]),
            m.Circumscribe(tiles[1]["facts"][1], color=RED, buff=0.06),
        )
        self.say(
            "[Asking for winter painted no snow.] [And a same-size transformer lost to our convolutional denoiser.]",
            show(tiles[2]),
            show(tiles[3]),
        )
        self.wait(0.6)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes=(
            "Episode 1's three words for the whole course: noise (destroying a picture is easy and needs no learning; "
            "the forward process jumps to any step at once), denoise (a network learns to remove a little: ε, the score, "
            "or a velocity), repeat (1,000 DDPM steps, 50 DDIM steps, or 1 distilled step). The three finished pictures: "
            f"the swirl after 50 exact DDIM steps ({SW_SHOW} of {SW_N:,} points), the seed-{SEVEN_SEED} seven, and the "
            f"lighthouse ({STEPS}-step run)."))
        halo = glow(VIOLET, radius=4.0)
        words = m.VGroup(*[title_text(w, 80, c) for w, c in (("Noise", NOISE), ("Denoise", MODEL), ("Repeat", TIME))])
        subs = m.VGroup(*[code(s, 20, MUTED) for s in ("easy, in one jump", "learned by a network",
                                                        "1,000, 50 or 1 steps")])
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
        row = m.VGroup(words, dots, subs).move_to(1.25 * m.UP)
        assert row.width < 12.8
        tagline = text("Out of the noise, one small step at a time.", 30, MUTED).next_to(row, m.DOWN, buff=0.6)

        fh = 1.35
        sw = SwirlView([0, 0, 0], fh, SW_SHOW, radius=0.016)
        sw.put(SW_XT, SW_STEPS, SW_EXT[-1], DATA)
        sw_g = m.Group(sw.box, sw.dots)
        sev = framed(photo(digit_rgb(SEVEN), fh), DATA)
        lh = framed(photo(XT[STEPS], fh), DATA)
        mascot = pixel(0.5)
        finals = m.Group(sw_g, sev, lh, mascot).arrange(m.RIGHT, buff=0.45).next_to(tagline, m.DOWN, buff=0.5)
        block = m.Group(row, tagline, finals)
        block.shift((0.15 - block.get_center()[1]) * m.UP)
        assert finals.get_bottom()[1] > -3.5 and row.get_top()[1] < 3.4

        self.say(
            "[To sum up the whole course, in episode one's three words]: [noise], [denoise], [repeat.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.3 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(dots[0]), m.FadeIn(words[1], shift=0.3 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(dots[1]), m.FadeIn(words[2], shift=0.3 * m.UP), m.FadeIn(subs[2])),
        )
        self.say(
            "[A swirl, a seven, or a lighthouse]: [out of the noise, one small step at a time.]",
            m.LaggedStart(*[m.FadeIn(f, shift=0.2 * m.UP) for f in finals], lag_ratio=0.25),
            m.Write(tagline),
        )
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, tagline, finals)], run_time=0.9)

    # ------------------------------------------------------------------ the course-closing card

    def series_outro(self):
        self.next_section("The end", notes="The course's closing card: Out of the Noise, from static to pictures and "
                          "video; the six modules in their colours (kit.MODULES); Pixel.")
        if self.head:
            self.play(self.drop_header(), run_time=0.6)
        halo = glow(TEAL, radius=4.3).move_to(0.2 * m.UP)
        brand = title_text("That's Out of the Noise", 64, INK, gradient=(VIOLET, TEAL))
        sub = text("from static to pictures and video", 30, MUTED)
        top = m.VGroup(brand, sub).arrange(m.DOWN, buff=0.3)
        mods = m.VGroup(*[pill(f"{k} · {name}", color, fill, 22, mono=True) for k, (name, color, fill) in MODULES.items()])
        rows = m.VGroup(m.VGroup(mods[0], mods[1]).arrange(m.RIGHT, buff=0.25),
                        m.VGroup(mods[2], mods[3]).arrange(m.RIGHT, buff=0.25),
                        m.VGroup(mods[4], mods[5]).arrange(m.RIGHT, buff=0.25))
        grid = rows.arrange(m.DOWN, buff=0.18)
        thanks = title_text("Thanks for watching!", 42, INK)
        mascot = pixel(0.6)
        foot = m.VGroup(mascot, thanks).arrange(m.RIGHT, buff=0.4)
        group = m.VGroup(top, grid, foot).arrange(m.DOWN, buff=0.45).move_to(0.05 * m.UP)
        assert grid.width < 12.8 and group.height < 7.0 and brand.width < TITLE_WIDTH

        self.say(
            "[That's Out of the Noise]: [from static to pictures and video.] [Thanks for watching!]",
            m.AnimationGroup(m.FadeIn(halo, scale=0.6), m.FadeIn(brand, shift=0.2 * m.DOWN)),
            m.AnimationGroup(m.FadeIn(sub, shift=0.1 * m.UP), m.LaggedStart(*[m.FadeIn(p, shift=0.1 * m.UP) for p in mods], lag_ratio=0.15)),
            m.AnimationGroup(m.FadeIn(mascot, shift=0.2 * m.UP), m.Write(thanks)),
        )
        self.wait(1.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, group)], run_time=1.0)
        self.wait(0.5)
