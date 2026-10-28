"""Out of the Noise · Episode 20 · Data and Alignment (Module 5 · Scaling Up — the module's last episode)

What a model learns depends on its data and on what it is tuned to prefer. Running example: the digit model (trained on
this Mac on the 60,000 MNIST training digits, research/experiments/mac_runs.md §3–4) — what it was shown, and whether
it copies it (the "Memorisation check": every generated digit against its nearest training digit). The lighthouse
(generated on this Mac) stands in for a web picture whose caption is rewritten. The swirl carries a made-up preference
to show tuning drift. Episodes 15 (how a prompt is read), 17 (Stable Diffusion's LAION subsets), 18 (transformers) and
19 (preference models as judges) are linked by number, not re-taught. Sister course "Trial and Reward": Episodes 21
(Learning from Human Preferences) and 22 (Direct Preference Optimization).

Every number on screen is computed below from mac_runs.json (measured(...)), the run's arrays and MNIST (load_mnist),
or cited with its source and asserted. Nothing is run or downloaded.

Sources (research/notes.md §20, Appendix A, Appendix C row 20; primary sources re-read via arXiv, October 2026):
- Schuhmann et al., "LAION-5B: An open large-scale dataset for training next generation image-text models",
  arXiv:2210.08402 (Oct 2022; NeurIPS 2022 Datasets & Benchmarks): 5.85 B CLIP-filtered image–text pairs (2.32 B
  English, 2.26 B multilingual, 1.27 B no particular language); pairs kept if the CLIP ViT-B/32 cosine similarity of
  image and text is ≥ 0.28 (English) or ≥ 0.26 (others), which "removed around 90% of the original 50 billion images";
  "LAION-Aesthetic, a 120M subset of aesthetic images".
- LAION blog "Releasing Re-LAION-5B", 30 Aug 2024 [DATED]: LAION-5B taken down 19 Dec 2023 after a safety audit (a
  Stanford Internet Observatory report); re-released in cleaned form as Re-LAION-5B. Shown in one neutral, dated line.
- CompVis Stable Diffusion README (notes §20): SD 1.x data laion2B-en → laion-high-resolution → "laion-aesthetics v2 5+"
  (aesthetics score > 5.0, original size ≥ 512 × 512, estimated watermark probability < 0.5).
- The LAION aesthetics predictor "is trained on 176,000 human image ratings" (Black et al. 2023, §5.2, arXiv HTML).
- Betker et al. (OpenAI), "Improving Image Generation with Better Captions" (DALL·E 3, 2023): a captioner writes
  "descriptive synthetic captions"; blends of 65, 80, 90 and 95% synthetic captions, the 65% blend "was far behind";
  final "mixture of 95% synthetic captions and 5% ground truth captions"; abstract: "prompt following abilities of
  text-to-image models can be substantially improved by training on highly descriptive generated image captions".
- Carlini et al., "Extracting Training Data from Diffusion Models", arXiv:2301.13188 (30 Jan 2023; USENIX Security
  2023), §4.2: the 350,000 most-duplicated training examples × 500 generations = 175 million generated images; 109
  extracted near-copies in total (94 automatic at (ℓ2, 0.15) + manual review, the paper's total); extraction mostly for
  images duplicated ≥ 100 times.
- Black et al., "Training Diffusion Models with Reinforcement Learning" (DDPO), arXiv:2305.13301 (May 2023; ICLR 2024):
  "posing denoising as a multi-step decision-making problem"; policy gradients on a reward (aesthetic quality, JPEG
  compressibility, prompt alignment judged by a vision-language model); Stable Diffusion v1.4, 50 denoising steps
  (App. D.5). Fig. 7 / App. A (overoptimisation): "the diffusion model exploits the VLM with a typographic attack,
  writing text that is interpreted as the specified number n instead of generating the correct number of animals."
- Wallace et al., "Diffusion Model Alignment Using Direct Preference Optimization", arXiv:2311.12908 (Nov 2023; CVPR
  2024): DPO's loss with the diffusion ELBO — per noise level, the model's denoising error on the chosen picture vs a
  frozen reference, minus the same on the rejected one; SDXL-1.0 base fine-tuned on Pick-a-Pic's 851K crowdsourced
  pairwise preferences; "DPO-SDXL is preferred 70.0% of the time for General Preference" vs SDXL base on PartiPrompts
  (69% vs base + refiner). SD3 was DPO-aligned after pretraining (Esser et al., 2024, §5.3.2).

Measured on this Mac (mac_runs §4 "Memorisation check"; arrays/digit_nearest_neighbours.npz): pixel-space L2 distance
(pixels in [0, 1], 784 numbers) from each of 10,000 generated digits to the nearest of the 60,000 training digits.
Medians: DDIM 50 (w = 1) 4.52, DDPM 1,000 4.40, flow 50 4.47; real test digits → train 4.34; two random digits 10.1.
Shares closer than the real test digits' 1st percentile (1.29): 0.47% / 0.43% / 0.31% (1% by definition). Closest
generated digits (0.89–1.02) are all plain "1"s; the closest real test digits are "1"s too (0.83, 0.95; recomputed here
from MNIST and asserted against the run's array).

Illustrative (labelled on screen "for example" and noted per section): the web page drawing and its alt text, the two
CLIP pairs, the aesthetic scores of our five lighthouse pictures, the short alt texts and the dense caption (written for
this episode), the swirl preference (a made-up reward), the chosen/rejected lighthouse pair. Funnel widths: log scale.

    VOICE=draft manimgx inspect scene.py      # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403
from diffusion_reference import *  # noqa: E402,F403

EP_DIR = Path(__file__).resolve().parent          # set AFTER the import: diffusion_reference exports HERE (the series)
PREFIX = "ep20"
ARR = EXPERIMENTS / "arrays"
IMG = EXPERIMENTS / "images"
PX = 135                                          # pixels per scene unit at 1080p
HL = lambda *keys: measured("mac_runs", "headline", *keys)  # noqa: E731
NNJ = measured("mac_runs", "nearest_neighbours")
REAL, GEN = DATA, MODEL                           # real digits TEAL, generated AMBER (as in Episode 19)
CAPTION = PROMPT                                  # captions and text PINK

# ---------------------------------------------------------------------------- the digit model and its data

TRAIN_X, TRAIN_Y = load_mnist("train")
TEST_X, TEST_Y = load_mnist("test")
assert TRAIN_X.shape == (60_000, 28, 28) and TEST_X.shape == (10_000, 28, 28)
assert measured("mac_runs", "data", "train") == 60_000 and measured("mac_runs", "data", "test") == 10_000
PARAMS = HL("digit_unet_params")
assert PARAMS == 3_940_801 and f"{PARAMS / 1e6:.1f}" == "3.9"                     # "three point nine million weights"

SAMP = np.load(ARR / "digit_samples_ddpm_ddim50_w1.npz")
GEN_X, GEN_Y = SAMP["samples"], SAMP["labels"].astype(int)
assert GEN_X.shape == (10_000, 28, 28) and (GEN_Y == np.arange(10_000) % 10).all()
assert "ddpm_ddim50_w1" in str(SAMP["note"])
SEVENS = GEN_X[GEN_Y == 7][:9]

# ---------------------------------------------------------------------------- the memorisation check (mac_runs §4)

NN = np.load(ARR / "digit_nearest_neighbours.npz")
D = {"ddim": NN["ddpm_ddim50_w1_dist"].astype(np.float64), "ddpm": NN["ddpm_ddpm1000_w1_dist"].astype(np.float64),
     "flow": NN["flow_euler50_w1_dist"].astype(np.float64)}
D_TEST = NN["test_to_train_dist"].astype(np.float64)
KEYS = {"ddim": "ddpm_ddim50_w1", "ddpm": "ddpm_ddpm1000_w1", "flow": "flow_euler50_w1"}
assert all(len(v) == 10_000 for v in D.values()) and len(D_TEST) == 10_000
MED = {k: NNJ[KEYS[k]]["median"] for k in KEYS}
MED_TEST, RAND = NNJ["test_to_train"]["median"], NNJ["random_pair_median"]
for k in KEYS:
    assert abs(np.median(D[k]) - MED[k]) < 1e-3
assert abs(np.median(D_TEST) - MED_TEST) < 1e-3
assert (f"{MED['ddim']:.2f}", f"{MED['ddpm']:.2f}", f"{MED['flow']:.2f}", f"{MED_TEST:.2f}") == ("4.52", "4.40", "4.47", "4.34")
assert f"{RAND:.1f}" == "10.1" and f"{HL('nn_dist_random_pair_median'):.1f}" == "10.1"     # "about ten point one"
assert f"{HL('nn_dist_generated_median'):.2f}" == "4.52" and f"{HL('nn_dist_test_median'):.2f}" == "4.34"
P1 = NNJ["test_to_train"]["p1"]
assert abs(np.percentile(D_TEST, 1) - P1) < 1e-3 and f"{P1:.2f}" == "1.29"
BELOW = {k: float((D[k] < P1).mean()) for k in KEYS}
for k in KEYS:
    assert abs(BELOW[k] - NNJ[KEYS[k]]["frac_below_test_p1"]) < 1e-4
PCT = {k: f"{100 * BELOW[k]:.2f}%" for k in KEYS}
assert (PCT["ddim"], PCT["ddpm"], PCT["flow"]) == ("0.47%", "0.43%", "0.31%")        # "zero point four seven percent"

EX_I, EX_G, EX_T = NN["examples_index"], NN["examples_generated"], NN["examples_nearest_train"]
EX_D = NN["examples_dist"].astype(np.float64)
NN_IDX = NN["ddpm_ddim50_w1_nn_index"]
assert len(EX_I) == 16 and (GEN_X[EX_I] == EX_G).all() and np.allclose(EX_D, D["ddim"][EX_I])
assert all((TRAIN_X[NN_IDX[i]] == t).all() for i, t in zip(EX_I, EX_T))
CLOSEST = np.argsort(D["ddim"])[:4]
assert list(CLOSEST) == list(EX_I[12:]) and (GEN_Y[CLOSEST] == 1).all() and (TRAIN_Y[NN_IDX[CLOSEST]] == 1).all()
assert (f"{D['ddim'][CLOSEST[0]]:.2f}", f"{D['ddim'][CLOSEST[3]]:.2f}") == ("0.89", "1.02")   # "plain ones"
assert f"{D['ddim'].min():.2f}" == f"{NNJ['ddpm_ddim50_w1']['min']:.2f}"

TRAIN_F = TRAIN_X.reshape(60_000, 784).astype(np.float32) / 255
TRAIN_SQ = (TRAIN_F ** 2).sum(1)


def distances(img):
    """Pixel-space L2 distance (pixels in [0, 1], 784 numbers) from one 28×28 uint8 digit to every training digit."""
    q = img.reshape(784).astype(np.float32) / 255
    return np.sqrt(np.maximum(TRAIN_SQ - 2 * TRAIN_F @ q + (q * q).sum(), 0)).astype(np.float64)


TEST_ONES = [int(i) for i in np.argsort(D_TEST)[:2]]
TEST_NN = []
for i in TEST_ONES:
    d_i = distances(TEST_X[i])
    TEST_NN.append(int(d_i.argmin()))
    assert abs(d_i.min() - D_TEST[i]) < 2e-4 and TEST_Y[i] == 1 and TRAIN_Y[TEST_NN[-1]] == 1
assert (f"{D_TEST[TEST_ONES[0]]:.2f}", f"{D_TEST[TEST_ONES[1]]:.2f}") == ("0.83", "0.95")   # "just as close"

# the search animation: example 0 (a generated 7) against all 60,000 training digits
SEARCH_G = EX_G[0]
D_SEARCH = distances(SEARCH_G)
assert int(D_SEARCH.argmin()) == int(NN_IDX[EX_I[0]]) and abs(D_SEARCH.min() - EX_D[0]) < 2e-4 and GEN_Y[EX_I[0]] == 7
_rng = np.random.default_rng(20)
ORDER = _rng.permutation(60_000)
_best = np.minimum.accumulate(D_SEARCH[ORDER])
assert abs(_best[-1] - D_SEARCH.min()) < 1e-12
CHECKS = sorted(set(np.unique(np.geomspace(1, 60_000, 70).astype(int))) | {60_000})   # how many checked, per frame
SEARCH_FRAMES = [(n, int(ORDER[n - 1]), int(ORDER[:n][np.argmin(D_SEARCH[ORDER[:n]])]), float(_best[n - 1]))
                 for n in CHECKS]                                                   # (checked, now, best, best d)
assert SEARCH_FRAMES[-1][0] == 60_000 and SEARCH_FRAMES[-1][2] == int(NN_IDX[EX_I[0]])

EDGES = np.arange(0, 9.01, 0.25)
assert max(D["ddim"].max(), D_TEST.max()) < EDGES[-1]
H_GEN = np.histogram(D["ddim"], EDGES)[0] / 10_000
H_TEST = np.histogram(D_TEST, EDGES)[0] / 10_000
H_TOP = float(max(H_GEN.max(), H_TEST.max()))

# ---------------------------------------------------------------------------- web-scale data (cited, notes §20)

N_WEB, N_LAION, N_EN, N_AES = 50e9, 5.85e9, 2.32e9, 120e6      # Schuhmann et al. 2022 ("around 90% of … 50 billion")
assert abs(2.32 + 2.26 + 1.27 - 5.85) < 0.005                                      # English + multilingual + none
assert 0.85 < 1 - N_LAION / N_WEB < 0.92                                           # "about ninety percent removed"
CLIP_EN, CLIP_OTHER = 0.28, 0.26
RATINGS = 176_000                                  # the LAION aesthetics predictor's training ratings (Black et al. §5.2)
TAKEDOWN, RERELEASE = "19 Dec 2023", "30 Aug 2024"  # LAION blog, 30 Aug 2024 [DATED]

# ---------------------------------------------------------------------------- recaptioning (Betker et al. 2023)

BLENDS, FINAL = (65, 80, 90, 95), 95
assert FINAL == max(BLENDS) and 100 // (100 - FINAL) == 20                          # "one time in twenty"
ALT_TEXTS = ("lighthouse.jpg", "IMG_2041", "free shipping!")                        # for example
DENSE = ("An oil painting of a tall white lighthouse with a red lantern room,",
         "on brown cliffs at the right, a small red-roofed house at its foot.",
         "White waves break over dark rocks below a grey-blue evening sky.")   # for example: written for this episode

# ---------------------------------------------------------------------------- memorisation in big models (Carlini et al.)

CARLINI_PROMPTS, CARLINI_EACH, CARLINI_FOUND, CARLINI_DUP = 350_000, 500, 109, 100
CARLINI_GEN = CARLINI_PROMPTS * CARLINI_EACH
assert CARLINI_GEN == 175_000_000 and f"{CARLINI_GEN / CARLINI_FOUND / 1e6:.1f}" == "1.6"   # "one in one point six million"

# ---------------------------------------------------------------------------- preference tuning

PICK_PAIRS, DPO_WIN, DPO_WIN_REFINER, DDPO_STEPS = 851_000, 70.0, 69, 50           # Wallace et al.; Black et al. App. D.5
assert f"{PICK_PAIRS / 1e3:.0f}" == "851" and round(DPO_WIN) == 70

# the swirl with a made-up preference: reward r(x) = −|x − c|², c just off the swirl (for example)
PREF_C = np.array([0.95, 0.72])
MEANS = np.asarray(SWIRL.means)
PER_ARM = len(MEANS) // 2
assert PER_ARM == 128 and np.linalg.norm(PREF_C) > 1.1                              # c lies outside the swirl
N_PTS = 300
_r2 = np.random.default_rng(5)
ARM_OF = np.repeat([0, 1], N_PTS // 2)
U_OF = np.concatenate([(np.arange(N_PTS // 2) + _r2.uniform(0.1, 0.9, N_PTS // 2)) / (N_PTS // 2) for _ in range(2)])
OFFSET = SWIRL_NOISE * _r2.standard_normal((N_PTS, 2))
REWARD = -np.sum((MEANS - PREF_C) ** 2, axis=1)
LAM = 3.0                                          # how hard the (made-up) preference pulls, with the leash on


def tilted(lam):
    """Points of the swirl under a tilted mixture p(x)·exp(lam·r(x)), each point sliding along its own arm
    (inverse CDF of the arm's component weights at the point's fixed quantile)."""
    out = np.empty((N_PTS, 2))
    for a in (0, 1):
        sel = ARM_OF == a
        r = REWARD[a * PER_ARM:(a + 1) * PER_ARM]
        w = np.exp(lam * (r - r.max()))
        cdf = np.concatenate([[0.0], np.cumsum(w / w.sum())])
        s = np.clip(np.interp(U_OF[sel], cdf, np.arange(PER_ARM + 1) - 0.5), 0, PER_ARM - 1)
        arm = MEANS[a * PER_ARM:(a + 1) * PER_ARM]
        idx = np.arange(PER_ARM)
        out[sel] = np.stack([np.interp(s, idx, arm[:, 0]), np.interp(s, idx, arm[:, 1])], axis=1)
    return out + OFFSET


BASE_XY = tilted(0.0)
TUNED_XY = tilted(LAM)
NEAR = lambda xy: float((np.linalg.norm(xy - PREF_C, axis=1) < 0.75).mean())  # noqa: E731
assert NEAR(TUNED_XY) > 2 * NEAR(BASE_XY)                                          # tuning moved samples toward c
assert nearest_on_swirl(TUNED_XY)[0].max() < 0.15                                       # … and they stayed on the swirl
HACK_XY = PREF_C + 0.035 * _r2.standard_normal((N_PTS, 2))
assert nearest_on_swirl(HACK_XY)[0].min() > 0.15                                        # the hacked samples left the data
FAR = np.array([np.clip(nearest_on_swirl(TUNED_XY + (HACK_XY - TUNED_XY) * a)[0] / 0.15, 0, 1)
                for a in np.linspace(0, 1, 31)])                                   # RED by distance off the swirl

# ---------------------------------------------------------------------------- look


def cite(s, size=20, color=MUTED):
    return text(s, size, color)


FLAT = set("abcdefhiklmnorstuvwxzABCDEFGHIJKLMNOPRSTUVWXYZ0123456789")


def baseline(words):
    chars = [c for c in words.text if not c.isspace()]
    if len(words) == len(chars):
        bottoms = sorted(g.get_bottom()[1] for g, c in zip(words, chars) if c in FLAT)
        if bottoms:
            return bottoms[len(bottoms) // 2]
    return float(np.median([g.get_bottom()[1] for g in words]))


def sit(words, y):
    return words.shift((y - baseline(words)) * m.UP)


_TEX = {}


def texture_of(rgb, px, nearest=False):
    """An RGB array → an RGBA texture about px pixels wide (small arrays blown up with crisp square pixels, large
    ones resized smoothly: the engine samples, not averages, when it draws a texture smaller than it is)."""
    key = (id(rgb), px, nearest)
    if key in _TEX:
        return _TEX[key][1]
    h, w = rgb.shape[:2]
    if nearest:
        k = max(1, int(np.ceil(2 * px / w)))
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


def photo(rgb, height, nearest=None):
    nearest = max(rgb.shape[:2]) <= 64 if nearest is None else nearest
    image = m.ImageMobject(texture_of(rgb, int(np.ceil(height * PX)), nearest))
    image.height = height
    return image


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))


_TINT = {}


def tinted(img, color):
    key = (id(img), color)
    if key not in _TINT:
        _TINT[key] = (img, tint(img, color, 0, 255))
    return _TINT[key][1]


def digit(img, color, height):
    """One 28×28 digit (uint8) in a colour, with crisp pixels."""
    return photo(tinted(img, color), height, nearest=True)


def digit_sheet(images, rows, cols, color, height, gap_px=4):
    """Digits in a rows × cols grid as ONE picture `height` units tall, with dark gaps."""
    canvas = np.zeros((rows * 28 + (rows - 1) * gap_px, cols * 28 + (cols - 1) * gap_px))
    for i, img in enumerate(images[:rows * cols]):
        r, c = divmod(i, cols)
        canvas[r * (28 + gap_px):r * (28 + gap_px) + 28, c * (28 + gap_px):c * (28 + gap_px) + 28] = img / 255
    return photo(tint(canvas, color, 0, 1), height, nearest=True)


class Reel:
    """A picture whose texture can be swapped: `.image`, `.show(rgb)`."""

    def __init__(self, rgb, height, nearest=True):
        self.px, self.nearest = int(np.ceil(height * PX)), nearest
        self.image = m.ImageMobject(texture_of(rgb, self.px, nearest))
        self.image.height = height
        self.cur = id(rgb)

    def show(self, rgb):
        if id(rgb) != self.cur:
            self.image.paint = self.image.paint.but(texture=texture_of(rgb, self.px, self.nearest))
            self.cur = id(rgb)


class Readout(m.VGroup):
    """A mono text left-aligned at a fixed point; .show(s) replaces it."""

    def __init__(self, left, y, s, size=22, color=INK, bold=False):
        super().__init__()
        self.anchor, self.size, self.color, self.bold, self.s = np.array([left, y, 0.0]), size, color, bold, None
        self.words = self._make(s)
        self.add(self.words)
        self.s = s

    def _make(self, s):
        w = code(s, self.size, self.color, weight=m.BOLD if self.bold else m.NORMAL)
        return w.move_to(self.anchor, aligned_edge=m.LEFT)

    def show(self, s):
        if s != self.s:
            self.words.become(self._make(s))
            self.s = s


def drive(mob, hooks, run_time=1.4, rate=m.linear):
    """Animate a value 0 → 1, calling every hook(value) each frame."""

    def step(_, a):
        for hook in hooks:
            hook(a)

    return m.UpdateFromAlphaFunc(mob, step, run_time=run_time, rate_func=rate)


def framed_photo(rgb, height, color=EDGE, stroke=2.0, nearest=None):
    image = photo(rgb, height, nearest)
    frame = m.Rectangle(width=height * rgb.shape[1] / rgb.shape[0] + 0.08, height=height + 0.08)
    frame.set_stroke(color, stroke).set_fill(opacity=0).move_to(image)
    return m.Group(frame, image)


def mark(ok, size=24):
    return text("✓" if ok else "✗", size, GOOD if ok else BAD, weight=m.BOLD)


def boxed(group, color=EDGE, buff=0.3, radius=0.16, fill=CARD, stroke=2):
    box = card(group.width + 2 * buff, group.height + 2 * buff, color, fill, radius=radius, stroke=stroke).move_to(group)
    return m.VGroup(box, group)


def slim_pill(label, color, fill, size=20):
    """kit's pill with less padding (for a row of four)."""
    words = text(label, size, color, weight=m.BOLD)
    h = words.height + 0.3
    box = card(words.width + 0.36, h, color, fill, radius=h / 2, stroke=2)
    return m.VGroup(box, words.move_to(box))


def swirl_band(plot, color=REAL, width=10, opacity=0.13):
    band = m.VGroup()
    for arm in swirl_curve(200):
        band.add(m.VMobject().set_points_smoothly([plot.coords(*p) for p in arm[::3]] + [plot.coords(*arm[-1])])
                 .set_stroke(color, width, opacity=opacity))
    return band.set_z_index(-1)


LH_RGB = load_rgb(IMG / "lighthouse" / "lighthouse_sd21_cfg7.5_steps50_seed0.png")
LH_B = load_rgb(IMG / "lighthouse" / "lighthouse_sd21_cfg7.5_steps50_seed1.png")
assert LH_RGB.shape == (512, 512, 3) and LH_B.shape == (512, 512, 3)
TILES = [("steps01_cfg7.5_seed0", 0.22), ("cfg1_steps50_seed2", 0.36), ("steps02_cfg7.5_seed1", 0.44),
         ("cfg7.5_steps50_seed2", 0.78), ("cfg7.5_steps50_seed0", 0.71)]           # (picture, score: for example)
TILE_RGB = [load_rgb(IMG / "lighthouse" / f"lighthouse_sd21_{name}.png") for name, _ in TILES]
KEEP_AT = 0.55                                     # the drawn threshold ("score > 5"), on the drawn bars' scale
LH_GEN = "generated on this Mac · Stable Diffusion 2.1 base · 50 steps · guidance 7.5"

# the funnel: four levels, widths on a log scale of the counts
FUNNEL_X = 1.35
LEVELS = [(N_WEB, "≈ 50 B", "Common Crawl images", MUTED),
          (N_LAION, "5.85 B", "pairs CLIP says match", DATA),
          (N_EN, "2.32 B", "of them in English", DATA),
          (N_AES, "120 M", "LAION-Aesthetics subset", GOOD)]


def funnel_width(n):
    return 0.75 + 2.25 * (np.log10(n) - 8) / (np.log10(N_WEB) - 8)


LEVEL_Y = [1.5, 0.5, -0.5, -1.5]
LEVEL_H = 0.8
assert f"{funnel_width(N_WEB):.2f}" == "3.00"


def funnel_level(k):
    n, big, l1, col = LEVELS[k]
    top = funnel_width(n)
    bot = funnel_width(LEVELS[k + 1][0]) if k + 1 < len(LEVELS) else 0.78 * top
    y, h, x = LEVEL_Y[k], LEVEL_H, FUNNEL_X
    shape = m.Polygon([x - top / 2, y + h / 2, 0], [x + top / 2, y + h / 2, 0], [x + bot / 2, y - h / 2, 0],
                      [x - bot / 2, y - h / 2, 0], stroke_color=col, stroke_width=2.5,
                      fill_color=FILLS.get(col, CARD), fill_opacity=1)
    num = title_text(big, 30, INK if col == MUTED else col)
    d1 = text(l1, 20, INK)
    block = m.VGroup(num, d1).arrange(m.DOWN, buff=0.07, aligned_edge=m.LEFT)
    block.move_to([0, y, 0]).align_to([3.15, 0, 0], m.LEFT)
    return m.VGroup(shape, num, d1)


def half_width_at(y):
    """The funnel's half-width at height y (levels joined, gaps bridged)."""
    ys, ws = [], []
    for k, (n, *_rest) in enumerate(LEVELS):
        top = funnel_width(n)
        bot = funnel_width(LEVELS[k + 1][0]) if k + 1 < len(LEVELS) else 0.78 * top
        ys += [LEVEL_Y[k] + LEVEL_H / 2, LEVEL_Y[k] - LEVEL_H / 2]
        ws += [top / 2, bot / 2]
    return float(np.interp(-y, -np.array(ys), ws))


class Rain(m.VGroup):
    """Dots falling through the funnel: most leave at the CLIP filter (turn RED, slide out, fade), a few stop at
    the English level. Shares as in LAION-5B: about 1 in 9 kept (5.85 of 50), 40% of those English."""

    def __init__(self, n=72, seed=4):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.n = n
        self.xi = rng.uniform(-0.8, 0.8, n)
        self.start = np.sort(rng.uniform(0, 0.55, n))
        keep = np.zeros(n, bool)
        keep[rng.choice(n, int(round(n * N_LAION / N_WEB)), replace=False)] = True
        english = keep & (rng.uniform(0, 1, n) < N_EN / N_LAION)
        self.fate = np.where(~keep, 0, np.where(english, 2, 1))   # 0: out at CLIP, 1: out at English, 2: stays
        self.y_top = LEVEL_Y[0] + LEVEL_H / 2 - 0.12
        self.y_cut = [LEVEL_Y[0] - LEVEL_H / 2 - 0.05, LEVEL_Y[1] - LEVEL_H / 2 - 0.05]
        self.y_end = LEVEL_Y[2] - 0.15 + rng.uniform(-0.2, 0.2, n)
        self.side = np.where(self.xi < 0, -1, 1)
        for _ in range(n):
            self.add(m.Dot(radius=0.045, color=INK).set_opacity(0))
        self.kept = [d for d, f in zip(self, self.fate) if f == 2]
        self.gone = [d for d, f in zip(self, self.fate) if f != 2]

    def frame(self, a):
        for i, d in enumerate(self):
            t = float(np.clip((a - self.start[i]) / 0.42, 0, 1))
            if t <= 0:
                d.set_opacity(0)
                continue
            f = self.fate[i]
            y_stop = self.y_cut[0] if f == 0 else self.y_cut[1] if f == 1 else self.y_end[i]
            fall = 0.7 if f != 2 else 1.0
            if t < fall or f == 2:
                y = self.y_top + (y_stop - self.y_top) * (t / fall)
                x = FUNNEL_X + self.xi[i] * half_width_at(y)
                d.move_to([x, y, 0]).set_color(INK).set_opacity(min(1.0, 4 * t))
            else:
                u = (t - fall) / (1 - fall)
                x0 = FUNNEL_X + self.xi[i] * half_width_at(y_stop)
                x1 = FUNNEL_X + self.side[i] * (half_width_at(y_stop) + 0.3)
                d.move_to([x0 + (x1 - x0) * u, y_stop - 0.1 * u, 0]).set_color(BAD).set_opacity(1 - u)

    def anim(self, run_time=3.2):
        return drive(self, [self.frame], run_time=run_time)


GRID_PICS = TRAIN_X[:30]

# ---------------------------------------------------------------------------- the episode


class Ep20DataAndAlignment(Episode):
    number = 20

    def lesson(self):
        self.hook()
        self.web_data()
        self.filtering()
        self.captions()
        self.copying()
        self.preference()
        self.recap()

    # ------------------------------------------------------------------ 0 · hook

    def hook(self):
        self.next_section("Hook", notes=(
            "The digit model (mac_runs §3): a 3,940,801-parameter class-conditional U-Net trained on this Mac on the "
            "60,000 MNIST training digits (the first 30 shown, TEAL). Right: the first nine generated 7s of the DDIM 50, "
            "w = 1 run (arrays/digit_samples_ddpm_ddim50_w1.npz, sample i asks for label i mod 10; AMBER = generated). "
            "'Never seen' chips: MNIST holds only digits. Text-to-image data: billions of captioned web pictures "
            "(section 1)."))
        y0 = 0.75
        grid = digit_sheet(GRID_PICS, 5, 6, REAL, 2.6)
        grid_f = m.Group(m.Rectangle(width=grid.width + 0.12, height=grid.height + 0.12).set_stroke(REAL, 2)
                         .set_fill(opacity=0), grid).move_to([-4.35, y0, 0])
        g_t = text("60,000 training digits", 22, INK, weight=m.BOLD).next_to(grid_f, m.UP, buff=0.2)
        g_c = cite("MNIST: the only pictures it was shown").next_to(grid_f, m.DOWN, buff=0.18)
        model = m.VGroup(card(2.5, 1.55, MODEL, AMBER_FILL),
                         m.VGroup(text("our digit model", 24, INK, weight=m.BOLD), code("3.9 M weights", 20, MUTED),
                                  cite("trained on this Mac")).arrange(m.DOWN, buff=0.08))
        model[1].move_to(model[0])
        model.move_to([0, y0, 0])
        sev = digit_sheet(SEVENS, 3, 3, GEN, 2.1)
        sev_f = m.Group(m.Rectangle(width=sev.width + 0.12, height=sev.height + 0.12).set_stroke(GEN, 2)
                        .set_fill(opacity=0), sev).move_to([4.35, y0, 0])
        s_t = text("asked for a 7", 22, INK, weight=m.BOLD).next_to(sev_f, m.UP, buff=0.2)
        sit(s_t, baseline(g_t))
        s_c = m.VGroup(cite("generated on this Mac"), cite("DDIM, 50 steps")).arrange(m.DOWN, buff=0.06)
        s_c.next_to(sev_f, m.DOWN, buff=0.18)
        a1 = arrow(grid_f.get_right(), model[0].get_left(), REAL, buff=0.15)
        a2 = arrow(model[0].get_right(), sev_f.get_left(), GEN, buff=0.15)
        never_t = text("never shown:", 20, MUTED)
        never = m.VGroup(*[m.VGroup(chip(w, EDGE, CARD, width=1.9, height=0.58, size=20), mark(False, 22))
                           for w in ("letters", "faces", "lighthouses")])
        for c_ in never:
            c_[1].next_to(c_[0], m.RIGHT, buff=0.1)
        never.arrange(m.RIGHT, buff=0.35)
        never_row = m.VGroup(never_t, never).arrange(m.RIGHT, buff=0.3).move_to([0, -1.75, 0])
        t2i = pill("a text-to-image model: billions of web pictures, each with a caption", CAPTION, PINK_FILL, 22)
        t2i.move_to([0, -1.75, 0])
        two = m.VGroup(pill("1 · the data it is shown", DATA, TEAL_FILL, 22),
                       pill("2 · what it is tuned to prefer", GOOD, GOOD_FILL, 22)).arrange(m.RIGHT, buff=0.5)
        two.move_to([0, -2.85, 0])
        assert g_t.get_top()[1] < 2.62 and s_c.get_bottom()[1] > never_row.get_top()[1] + 0.15
        assert never_row.width < 12.8 and t2i.width < 12.8 and two.get_bottom()[1] > -3.5

        self.say(
            "[Everything our digit model knows], [it learned from sixty thousand handwritten digits].",
            m.AnimationGroup(self.new_header("00 · hook", "A model learns what it is shown"),
                             m.FadeIn(grid_f, shift=0.15 * m.UP), m.FadeIn(g_t), m.FadeIn(g_c)),
            m.AnimationGroup(m.GrowArrow(a1), m.FadeIn(model, shift=0.1 * m.RIGHT)),
        )
        self.say(
            "[Ask it for a seven, and it draws a seven like theirs], "
            "[but it has never seen a letter, a face, or a lighthouse].",
            m.AnimationGroup(m.GrowArrow(a2), m.FadeIn(sev_f, shift=0.1 * m.RIGHT), m.FadeIn(s_t), m.FadeIn(s_c)),
            m.AnimationGroup(m.FadeIn(never_t),
                             m.LaggedStart(*[m.FadeIn(c_, shift=0.1 * m.UP) for c_ in never], lag_ratio=0.3)),
        )
        self.say(
            "[A text-to-image model works the same way], "
            "[only its training set is billions of pictures from the web, each with a caption].",
            m.Circumscribe(model[0], color=MODEL, buff=0.08),
            m.AnimationGroup(m.FadeOut(never_row), m.FadeIn(t2i, shift=0.1 * m.UP)),
        )
        self.say(
            "[So what it makes depends on that data], [and on what it is tuned to prefer afterwards].",
            m.FadeIn(two[0], shift=0.1 * m.UP),
            m.FadeIn(two[1], shift=0.1 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 1 · web-scale data

    def web_data(self):
        self.next_section("1 · Web-scale data", notes=(
            "Schuhmann et al., 'LAION-5B', arXiv:2210.08402 (Oct 2022; NeurIPS 2022 Datasets & Benchmarks): image links "
            "and alt text from Common Crawl; a pair is kept if cos(CLIP ViT-B/32 image, text) ≥ 0.28 (English) or ≥ 0.26 "
            "(other languages), which removed around 90% of the original 50 billion images; 5.85 B pairs remain (2.32 B "
            "English, 2.26 B multilingual, 1.27 B no particular language). Funnel widths on a log scale. The web page, its "
            "alt text and the two pairs are drawn for this episode (for example); the picture is our lighthouse, "
            "generated on this Mac. Falling dots: about 1 in 9 kept, 40% of those English (to scale, 72 dots). "
            "[DATED, LAION blog 30 Aug 2024] Taken offline 19 Dec 2023 after a safety audit; re-released in cleaned "
            "form as Re-LAION-5B on 30 Aug 2024 — one neutral line, no further detail (scope note, notes §20). "
            "Pronunciation: the narrator spelled 'LAION' letter by letter (Whisper heard 'L.A.I.O.N.', 2026-10-27), so "
            "the name stays on screen and the voice says 'a public dataset of web images'."))
        # the web page (left column, top)
        web = card(5.95, 2.0, EDGE, CARD, radius=0.16, stroke=2).move_to([-3.5, 1.5, 0])
        bar_y = web.get_top()[1] - 0.36
        w_bar = m.Line([web.get_left()[0], bar_y, 0], [web.get_right()[0], bar_y, 0]).set_stroke(EDGE, 2)
        w_dots = m.VGroup(*[m.Dot(radius=0.05, color=c) for c in (RED, AMBER, GREEN)]).arrange(m.RIGHT, buff=0.1)
        w_dots.move_to([web.get_left()[0] + 0.38, bar_y + 0.18, 0])
        w_url = code("a web page", 20, FAINT).move_to([web.get_x(), bar_y + 0.18, 0])
        w_pic = framed_photo(LH_RGB, 1.1).move_to([web.get_left()[0] + 0.74, bar_y - 0.8, 0])
        w_src = '<img src="lh.jpg"\nalt="lighthouse painting">'
        w_code = code(w_src, 20, MUTED, t2c={r'"lighthouse painting"': CAPTION, r"\balt\b": CAPTION, r"<img|>": FAINT})
        w_code.next_to(w_pic, m.RIGHT, buff=0.22)
        alt_g = glyphs(w_code, '"lighthouse painting"')
        w_note = cite("drawn for this episode · picture generated on this Mac").next_to(web, m.DOWN, buff=0.12)
        assert w_code.get_right()[0] < web.get_right()[0] - 0.12 and web.get_left()[0] > -6.6, (w_code.get_right(), web.get_right())
        page = m.Group(web, w_bar, w_dots, w_url, w_pic, w_code)

        # the CLIP check: two pairs (for example) and the rule
        pairs = m.Group()
        for k, (cap, ok) in enumerate((("lighthouse painting", True), ("Free shipping on all orders", False))):
            th = framed_photo(LH_RGB, 0.55).move_to([-6.05, -0.42 - 0.72 * k, 0])
            c_ = chip(cap, CAPTION, PINK_FILL, width=text(cap, 20, weight=m.BOLD).width + 0.45, height=0.5, size=20)
            c_.next_to(th, m.RIGHT, buff=0.22)
            v = mark(ok, 28).next_to(c_, m.RIGHT, buff=0.25)
            pairs.add(m.Group(th, c_, v))
        rule = code(f"keep if cos(image, text) ≥ {CLIP_EN}", 22, INK, t2c={r"0\.28": DATA})
        rule.move_to([0, -1.82, 0]).align_to([-6.3, 0, 0], m.LEFT)
        r_c1 = cite(f"CLIP features · {CLIP_OTHER} for other languages").next_to(rule, m.DOWN, buff=0.12)
        r_c1.align_to(rule, m.LEFT)
        r_c2 = cite("the two pairs: for example").next_to(r_c1, m.DOWN, buff=0.08).align_to(rule, m.LEFT)
        assert max(p.get_right()[0] for p in pairs) < -0.6 and r_c2.get_bottom()[1] > -3.5
        assert pairs[0].get_top()[1] < w_note.get_bottom()[1] - 0.12

        # the funnel (right column)
        levels = [funnel_level(k) for k in range(4)]
        self.funnel = levels
        f_title = title_text("LAION-5B", 34, DATA).move_to([FUNNEL_X, 2.3, 0])
        f_cite = cite("Schuhmann et al., 2022").next_to(f_title, m.RIGHT, buff=0.3)
        sit(f_cite, baseline(f_title))
        f_note = cite("funnel widths: log scale").move_to([0, -2.17, 0]).align_to([3.15, 0, 0], m.LEFT)
        assert f_title.get_top()[1] < 2.62 and f_cite.get_right()[0] < 6.5
        assert all(lv.get_right()[0] < 6.55 for lv in levels), [lv.get_right()[0] for lv in levels]
        assert levels[0][0].get_left()[0] > -0.45
        rain = Rain().set_z_index(3)

        # the dated line (left column, replaces the pairs)
        d_rows = m.VGroup()
        for date, what in ((TAKEDOWN, "taken offline after a safety audit"),
                           (RERELEASE, "re-released in cleaned form: Re-LAION-5B")):
            row = m.VGroup(code(date, 22, TIME, weight=m.BOLD), text(what, 22, INK)).arrange(m.DOWN, buff=0.08,
                                                                                              aligned_edge=m.LEFT)
            d_rows.add(row)
        d_rows.arrange(m.DOWN, buff=0.25, aligned_edge=m.LEFT)
        dated = m.VGroup(d_rows, cite("LAION, 2024")).arrange(m.DOWN, buff=0.2, aligned_edge=m.LEFT)
        dated.move_to([0, -1.15, 0]).align_to([-6.3, 0, 0], m.LEFT)
        assert dated.get_top()[1] < w_note.get_bottom()[1] - 0.15, (dated.get_top(), w_note.get_bottom())
        assert dated.get_right()[0] < -0.5, dated.get_right()

        self.say(
            "[Start with the data]: [pictures from web pages], "
            "[each with its alt text, the short caption a page gives an image].",
            self.new_header("01 · web data", "Billions of captioned pictures"),
            m.AnimationGroup(m.FadeIn(page, shift=0.1 * m.UP), m.FadeIn(w_note)),
            m.Circumscribe(alt_g, color=CAPTION, buff=0.06),
        )
        self.say(
            "[The best-known open one is a public dataset of web images], [published in twenty twenty-two].",
            m.FadeIn(f_title, shift=0.1 * m.DOWN),
            m.FadeIn(f_cite),
        )
        self.say(
            "[It started from about fifty billion images found in Common Crawl], [a public archive of the web].",
            m.FadeIn(levels[0][0], shift=0.1 * m.DOWN),
            m.AnimationGroup(m.FadeIn(levels[0][1:]), m.FadeIn(f_note)),
        )
        self.say(
            "[A pair was kept only if CLIP said that picture and text matched]: [here, yes], [and here, no].",
            m.AnimationGroup(m.FadeIn(rule, shift=0.1 * m.UP), m.FadeIn(r_c1), m.FadeIn(r_c2)),
            m.FadeIn(pairs[0], shift=0.1 * m.RIGHT),
            m.FadeIn(pairs[1], shift=0.1 * m.RIGHT),
        )
        self.add(rain)
        self.say(
            "[That removed about ninety percent]: [it left five point eight five billion pairs], "
            "[two point three two billion of them in English].",
            m.AnimationGroup(m.FadeIn(levels[1][0]), m.FadeIn(levels[2][0]), rain.anim(4.2)),
            m.FadeIn(levels[1][1:], shift=0.1 * m.UP),
            m.FadeIn(levels[2][1:], shift=0.1 * m.UP),
        )
        self.remove(*rain.gone)
        self.say(
            "[In December twenty twenty-three, the dataset was taken offline after a safety audit], "
            "[and in twenty twenty-four it was re-released in cleaned form].",
            m.AnimationGroup(m.FadeOut(pairs), m.FadeOut(rule), m.FadeOut(r_c1), m.FadeOut(r_c2),
                             m.FadeIn(d_rows[0], shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(d_rows[1], shift=0.1 * m.UP), m.FadeIn(dated[1])),
        )
        self.wait(0.3)
        self.web_left = m.Group(page, w_note, dated)
        self.rain_kept = m.VGroup(*rain.kept)
        self.f_extra = m.VGroup(f_title, f_cite, f_note)

    # ------------------------------------------------------------------ 2 · aesthetic filtering

    def filtering(self):
        self.next_section("2 · Aesthetic filtering", notes=(
            f"The LAION aesthetics predictor 'is trained on {RATINGS:,} human image ratings' (Black et al. 2023, §5.2). "
            "LAION-5B paper: 'LAION-Aesthetic, a 120M subset of aesthetic images'. CompVis README (notes §20): SD v1.2–1.5 "
            "finished on 'laion-aesthetics v2 5+' (aesthetics score > 5.0, original size ≥ 512 × 512, estimated "
            "watermark probability < 0.5). ILLUSTRATIVE: the five pictures are ours (generated on this Mac: SD 2.1 base, "
            "1, 50, 2 and 50 steps, guidance 7.5 or 1, seeds as named in TILES); their score bars are drawn by hand, not "
            "predicted. Funnel: level 4 is the 120 M aesthetic subset (log width)."))
        x_l = -6.35
        t1 = text("an aesthetic predictor scores each picture", 22, INK, weight=m.BOLD)
        t1.move_to([0, 2.38, 0]).align_to([x_l, 0, 0], m.LEFT)
        t2 = cite(f"{RATINGS:,} human ratings · Black et al., 2023")
        t2.next_to(t1, m.DOWN, buff=0.1).align_to(t1, m.LEFT)
        th, gap = 0.92, 0.2
        xs = [x_l + th / 2 + k * (th + gap) for k in range(5)]
        tiles = m.Group(*[framed_photo(rgb, th).move_to([x, 1.25, 0]) for rgb, x in zip(TILE_RGB, xs)])
        base_y, bar_h = -0.45, 1.0
        bars, marks = m.VGroup(), m.VGroup()
        for (name, s), x in zip(TILES, xs):
            ok = s > KEEP_AT
            col = GOOD if ok else BAD
            r_ = m.Rectangle(width=0.32, height=bar_h * s).set_stroke(col, 2).set_fill(FILLS[col], 1)
            bars.add(r_.move_to([x, base_y + bar_h * s / 2, 0]))
            marks.add(mark(ok, 26).move_to([x, base_y - 0.3, 0]))
        floor = m.Line([x_l, base_y, 0], [xs[-1] + th / 2, base_y, 0]).set_stroke(FAINT, 2)
        y_keep = base_y + bar_h * KEEP_AT
        keep_line = m.DashedLine([x_l, y_keep, 0], [xs[-1] + th / 2, y_keep, 0], dash_length=0.1).set_stroke(INK, 2.5)
        keep_lab = code("> 5", 20, INK, weight=m.BOLD).next_to(keep_line, m.RIGHT, buff=0.12)
        ill = cite("for example: our pictures, scores drawn by hand").move_to([0, base_y - 0.72, 0])
        ill.align_to([x_l, 0, 0], m.LEFT)
        assert tiles.get_top()[1] < t2.get_bottom()[1] - 0.1 and max(b.get_top()[1] for b in bars) < tiles.get_bottom()[1] - 0.15
        assert keep_lab.get_right()[0] < self.funnel[1][0].get_left()[0] - 0.1, keep_lab.get_right()
        dims = [m.Rectangle(width=th + 0.09, height=th + 0.09).set_stroke(width=0).set_fill(BG, 0.62).move_to(t)
                for t, (_, s) in zip(tiles, TILES) if s <= KEEP_AT]

        # Stable Diffusion's subset (a wide card under the funnel)
        sd_t = text("Stable Diffusion 1.x finished training on", 22, INK, weight=m.BOLD)
        sd_n = code("laion-aesthetics v2 5+", 22, GOOD, weight=m.BOLD)
        row1 = m.VGroup(sd_t, sd_n).arrange(m.RIGHT, buff=0.25)
        sit(sd_n, baseline(sd_t))
        sd_r = text("aesthetic score > 5.0 · at least 512 × 512 · watermark probability < 0.5", 20, INK)
        sd_c = cite("CompVis README")
        row2 = m.VGroup(sd_r, sd_c).arrange(m.RIGHT, buff=0.35)
        sit(sd_c, baseline(sd_r))
        sd_in = m.VGroup(row1, row2).arrange(m.DOWN, buff=0.14, aligned_edge=m.LEFT)
        sd = boxed(sd_in, GOOD, 0.18, fill=CARD).move_to([0, -2.95, 0]).align_to([x_l, 0, 0], m.LEFT)
        lean = pill("polished pictures in, polished pictures out", GOOD, GOOD_FILL, 20).move_to([0, -1.78, 0])
        lean.align_to([x_l, 0, 0], m.LEFT)
        f_low = min(self.funnel[3][0].get_bottom()[1], self.funnel[3][2].get_bottom()[1], self.f_extra[2].get_bottom()[1])
        assert sd.get_top()[1] < f_low - 0.08 and sd.get_right()[0] < 6.55 and sd.get_bottom()[1] > -3.55, (sd.get_top(), f_low, sd.get_right(), sd.get_bottom())
        assert lean.get_top()[1] < ill.get_bottom()[1] - 0.08 and lean.get_bottom()[1] > sd.get_top()[1] + 0.08, (lean.get_top(), ill.get_bottom(), sd.get_top())
        assert lean.get_right()[0] < self.funnel[3][0].get_left()[0] - 0.2
        lv = self.funnel[3]

        self.say(
            "[Next, taste]: [a small model, trained on a hundred seventy-six thousand human ratings, "
            "gives each picture an aesthetic score].",
            m.AnimationGroup(self.new_header("02 · filtering", "Then, keep the pretty ones"), m.FadeOut(self.web_left)),
            m.AnimationGroup(m.FadeIn(t1), m.FadeIn(t2), m.LaggedStart(*[m.FadeIn(t, shift=0.1 * m.UP) for t in tiles],
                                                                        lag_ratio=0.15),
                             m.Create(floor), m.LaggedStart(*[m.GrowFromEdge(b, m.DOWN) for b in bars], lag_ratio=0.15),
                             m.FadeIn(ill)),
        )
        self.say(
            "[Keep only the high scorers], [and the funnel narrows again]: "
            "[the paper's aesthetic subset holds a hundred twenty million pictures].",
            m.AnimationGroup(m.Create(keep_line), m.FadeIn(keep_lab), m.LaggedStart(*[m.FadeIn(k_) for k_ in marks],
                                                                                    lag_ratio=0.15),
                             *[m.FadeIn(d_) for d_ in dims]),
            m.AnimationGroup(m.FadeIn(lv[0], shift=0.1 * m.DOWN), m.FadeOut(self.rain_kept)),
            m.FadeIn(lv[1:], shift=0.1 * m.UP),
        )
        self.say(
            "[The first versions of Stable Diffusion finished on a subset like this]: "
            "[an aesthetic score above five, and at least five hundred twelve pixels a side].",
            m.AnimationGroup(m.FadeIn(sd[0]), m.FadeIn(row1)),
            m.AnimationGroup(m.FadeIn(sd_r, shift=0.1 * m.UP), m.FadeIn(sd_c)),
        )
        self.say(
            "[So the model leans toward polished pictures], [because polished pictures are what it was shown].",
            m.Circumscribe(sd[0], color=GOOD, buff=0.06),
            m.FadeIn(lean, shift=0.1 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 3 · recaptioning

    def captions(self):
        self.next_section("3 · Better captions", notes=(
            "Betker et al. (OpenAI), 'Improving Image Generation with Better Captions' (the DALL·E 3 paper, 2023, "
            "cdn.openai.com/papers/dall-e-3.pdf; notes §20): web alt text is noisy; a captioner trained to write "
            "'descriptive synthetic captions'; blends of 65, 80, 90 and 95% synthetic captions tested on T5-XXL latent "
            "diffusion models (500,000 steps); 65% 'was far behind'; final mix 95% synthetic, 5% ground truth; abstract: "
            "prompt following 'substantially improved'. 95/5 ⇒ the original alt text 1 time in 20 (derived). "
            "ILLUSTRATIVE: the three alt texts and the dense caption were written for this episode (not their captioner's "
            "output); the caption describes our lighthouse, generated on this Mac (SD 2.1 base, 50 steps, guidance 7.5, "
            "seed 0). Episode 15: how a prompt is read (text encoders, cross-attention)."))
        pic = framed_photo(LH_RGB, 2.45, DATA).move_to([-4.6, 1.05, 0])
        p_c = m.VGroup(cite("generated on this Mac"), cite("SD 2.1 base · 50 steps · seed 0")).arrange(m.DOWN, buff=0.05)
        p_c.next_to(pic, m.DOWN, buff=0.14)
        x_r = -2.55
        a_t = text("alt text on the web, for example", 20, MUTED).move_to([0, 2.45, 0]).align_to([x_r, 0, 0], m.LEFT)
        alts = m.VGroup(*[chip(s, CAPTION, CARD, width=code(s, 20).width + 0.4, height=0.5, size=24, mono=True)
                          for s in ALT_TEXTS]).arrange(m.RIGHT, buff=0.22)
        alts.next_to(a_t, m.DOWN, buff=0.16).align_to(a_t, m.LEFT)
        bad = text("short, noisy, often unrelated", 20, BAD, weight=m.BOLD).next_to(a_t, m.RIGHT, buff=0.45)
        sit(bad, baseline(a_t))
        assert alts.get_right()[0] < 6.5 and bad.get_right()[0] < 6.5, (alts.get_right(), bad.get_right())
        capr = m.VGroup(card(2.2, 0.62, INK, CARD_HI, radius=0.14, stroke=2),
                        text("a captioner", 22, INK, weight=m.BOLD))
        capr[1].move_to(capr[0])
        capr.move_to([0, alts.get_bottom()[1] - 0.52, 0]).align_to([x_r, 0, 0], m.LEFT)
        capr_c = cite("Betker et al., 2023 (DALL·E 3)").next_to(capr, m.RIGHT, buff=0.3)
        dense = m.VGroup(*[text(s, 20, INK) for s in DENSE]).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        dense_box = boxed(dense, CAPTION, 0.22, fill=PINK_FILL)
        dense_box.next_to(capr, m.DOWN, buff=0.36).align_to(capr, m.LEFT)
        d_c = cite("for example: written for this episode").next_to(dense_box, m.DOWN, buff=0.1).align_to(dense_box, m.LEFT)
        a_cap = arrow(capr[0].get_bottom(), [capr.get_x(), dense_box.get_top()[1], 0], CAPTION, buff=0.04)
        assert dense_box.get_right()[0] < 6.5 and capr.get_top()[1] < alts.get_bottom()[1] - 0.12, (dense_box.get_right(), capr.get_top())
        assert pic.get_right()[0] < x_r - 0.25 and capr_c.get_right()[0] < 6.5

        # the blend bar
        bx0, bx1, by = -6.2, 2.4, -2.12
        span = bx1 - bx0
        b_t = text("share of synthetic captions in training", 20, INK, weight=m.BOLD).move_to([0, by + 0.45, 0])
        b_t.align_to([bx0, 0, 0], m.LEFT)
        b_c = cite("Betker et al., 2023").next_to(b_t, m.RIGHT, buff=0.3)
        sit(b_c, baseline(b_t))
        syn = m.Rectangle(width=span * FINAL / 100, height=0.36).set_stroke(CAPTION, 2).set_fill(PINK_FILL, 1)
        syn.move_to([bx0 + span * FINAL / 200, by, 0])
        orig = m.Rectangle(width=span * (100 - FINAL) / 100, height=0.36).set_stroke(MUTED, 2).set_fill(CARD_HI, 1)
        orig.move_to([bx1 - span * (100 - FINAL) / 200, by, 0])
        ticks, t_labs = m.VGroup(), m.VGroup()
        for b in BLENDS:
            x = bx0 + span * b / 100
            ticks.add(m.Line([x, by - 0.24, 0], [x, by + 0.24, 0]).set_stroke(INK if b in (65, FINAL) else FAINT, 2.5))
            lab_col = BAD if b == 65 else GOOD if b == FINAL else MUTED
            t_labs.add(code(f"{b}%", 20, lab_col, weight=m.BOLD if b in (65, FINAL) else m.NORMAL)
                       .move_to([x, by + (0.42 if b == FINAL else -0.45), 0]))
        far = text("far behind", 20, BAD).next_to(t_labs[0], m.LEFT, buff=0.15)
        used = text("used", 20, GOOD).next_to(t_labs[-1], m.RIGHT, buff=0.12)
        sit(far, baseline(t_labs[0]))
        sit(used, baseline(t_labs[-1]))
        one20 = m.VGroup(title_text("1 in 20", 44, INK), text("keep their own alt text", 20, MUTED))
        one20.arrange(m.DOWN, buff=0.08).move_to([4.85, by + 0.05, 0])
        assert b_t.get_top()[1] < p_c.get_bottom()[1] - 0.05 or b_t.get_left()[0] > p_c.get_right()[0], (b_t.get_top(), p_c.get_bottom())
        assert b_t.get_top()[1] < d_c.get_bottom()[1] - 0.05, (b_t.get_top(), d_c.get_bottom())
        assert one20.get_right()[0] < 6.55 and one20.get_left()[0] > max(bx1, used.get_right()[0]) + 0.2, (one20.get_left(), used.get_right())
        res = pill("prompt following: “substantially improved”", GOOD, GOOD_FILL, 22).move_to([0, -3.22, 0])
        res.align_to([bx0, 0, 0], m.LEFT)
        ep15 = pill("Episode 15 · how a prompt is read", CAPTION, PINK_FILL, 20).next_to(res, m.RIGHT, buff=0.35)
        assert ep15.get_right()[0] < 6.55 and res.get_top()[1] < used.get_bottom()[1] - 0.05, (ep15.get_right(), used.get_bottom())

        self.say(
            "[Captions matter as much as pictures], [and alt text is short, noisy, and often unrelated to the image].",
            m.AnimationGroup(self.new_header("03 · captions", "Better captions, better prompt following"),
                             m.FadeIn(pic, shift=0.1 * m.UP), m.FadeIn(p_c)),
            m.AnimationGroup(m.FadeIn(a_t), m.LaggedStart(*[m.FadeIn(a_, shift=0.1 * m.UP) for a_ in alts], lag_ratio=0.3),
                             m.FadeIn(bad)),
        )
        self.say(
            "[For DALL·E three, OpenAI trained a captioner to describe each training image in detail].",
            m.AnimationGroup(m.FadeIn(capr, shift=0.1 * m.UP), m.FadeIn(capr_c)),
        )
        self.say(
            "[A dense caption names what is really there]: "
            "[the white tower, the red lantern room, the cliffs, the waves, the evening sky].",
            m.AnimationGroup(m.GrowArrow(a_cap), m.FadeIn(dense_box[0]), m.FadeIn(d_c)),
            m.AnimationGroup(*[m.AddTextLetterByLetter(line_, time_per_char=0.018) for line_ in dense]),
        )
        self.say(
            "[They trained on blends of synthetic and original captions]: "
            "[sixty-five percent synthetic was far behind], [and the final mix was ninety-five percent].",
            m.AnimationGroup(m.FadeIn(b_t), m.FadeIn(b_c), m.GrowFromEdge(syn, m.LEFT), m.FadeIn(orig), m.Create(ticks),
                             m.FadeIn(t_labs[1:3])),
            m.AnimationGroup(m.FadeIn(t_labs[0]), m.FadeIn(far)),
            m.AnimationGroup(m.FadeIn(t_labs[3]), m.FadeIn(used)),
        )
        self.say(
            "[So an image appears with its original alt text only one time in twenty].",
            m.AnimationGroup(m.FadeIn(one20, shift=0.1 * m.UP), m.Circumscribe(orig, color=MUTED, buff=0.05)),
        )
        self.say(
            "[Prompt following improved substantially]. [Episode fifteen showed how a prompt is read]; "
            "[what changed here is what the model was trained to read].",
            m.FadeIn(res, shift=0.1 * m.UP),
            m.FadeIn(ep15, shift=0.1 * m.UP),
            m.Circumscribe(dense_box[0], color=CAPTION, buff=0.06),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 4 · memorisation

    def copying(self):
        self.next_section("4 · Does it copy?", notes=(
            "Carlini et al., 'Extracting Training Data from Diffusion Models', arXiv:2301.13188 (30 Jan 2023; USENIX "
            "Security 2023), §4.2: the 350,000 most-duplicated training examples' captions, 500 generations each = 175 M "
            "images; 109 extracted near-copies in total (the paper's total); extraction mostly for images duplicated ≥ 100 "
            "times. Rate 109 / 175 M ≈ 1 in 1.6 million, for prompts chosen to be the most duplicated (derived). Their "
            "distance is ℓ2 normalised by the dimension with a 0.15 threshold, tuned for photos: on 28×28 digits that "
            "threshold (≈ 4.2 in our units) sits below the median of REAL test digits, so we compare with real test "
            "digits instead of borrowing it."))
        y0 = 1.3
        blocks = [("350,000", "most-duplicated", "training captions", INK), ("× 500", "pictures", "for each", INK),
                  ("= 175 M", "generated", "pictures", INK), ("109", "near-copies of", "training images", BAD)]
        xs = [-5.0, -2.05, 0.95, 4.65]
        stats = []
        for (big, l1, l2, col), x in zip(blocks, xs):
            b = title_text(big, 48, col)
            sub = m.VGroup(text(l1, 20, MUTED), text(l2, 20, MUTED)).arrange(m.DOWN, buff=0.05)
            g = m.VGroup(b, sub).arrange(m.DOWN, buff=0.14).move_to([x, y0, 0])
            stats.append(g)
        for g in stats[1:]:
            sit(g[0], baseline(stats[0][0]))
        a_f = arrow([stats[2].get_right()[0] + 0.12, baseline(stats[0][0]) + 0.3, 0],
                    [stats[3].get_left()[0] - 0.12, baseline(stats[0][0]) + 0.3, 0], BAD, buff=0.05)
        a_lab = cite("copies?").next_to(a_f, m.DOWN, buff=0.1)
        rate = code(f"≈ 1 in {CARLINI_GEN / CARLINI_FOUND / 1e6:.1f} million generations", 24, INK).move_to([0, -0.35, 0])
        dup = text(f"mostly pictures repeated {CARLINI_DUP} times or more in the training data", 22, INK)
        dup.move_to([0, -1.0, 0])
        c_c = cite("Carlini et al., 2023 · Stable Diffusion · prompts chosen to be the most duplicated").move_to([0, -1.6, 0])
        assert stats[0].get_left()[0] > -6.55 and stats[3].get_right()[0] < 6.55 and stats[0].get_top()[1] < 2.62
        assert a_f.get_length() > 0.5
        screen_a = m.VGroup(*stats, a_f, a_lab, rate, dup, c_c)

        # the search on our digits
        sy = 0.85
        g_img = digit(SEARCH_G, GEN, 2.0)
        g_f = m.Group(m.Rectangle(width=2.1, height=2.1).set_stroke(GEN, 2.5), g_img).move_to([-5.1, sy, 0])
        g_l = m.VGroup(text("one generated digit", 20, INK, weight=m.BOLD), cite("DDIM 50 steps · w = 1")).arrange(m.DOWN, buff=0.06)
        g_l.next_to(g_f, m.DOWN, buff=0.15)
        frames = SEARCH_FRAMES
        now_r = Reel(tinted(TRAIN_X[frames[0][1]], REAL), 1.45)
        best_r = Reel(tinted(TRAIN_X[frames[0][2]], REAL), 2.0)
        now_f = m.Group(m.Rectangle(width=1.55, height=1.55).set_stroke(EDGE, 2), now_r.image).move_to([-2.35, sy, 0])
        best_f = m.Group(m.Rectangle(width=2.1, height=2.1).set_stroke(REAL, 2.5), best_r.image).move_to([0.45, sy, 0])
        now_l = text("checking", 20, MUTED).next_to(now_f, m.UP, buff=0.12)
        best_l = text("closest so far", 20, INK, weight=m.BOLD).next_to(best_f, m.UP, buff=0.12)
        cnt = Readout(1.95, sy + 0.4, f"checked {frames[0][0]:>6,} / 60,000", 20, INK)
        dist = Readout(1.95, sy - 0.2, f"distance {frames[0][3]:.2f}", 24, REAL, bold=True)
        formula = code("distance = ‖generated − training‖ · 784 pixels in [0, 1]", 20, MUTED).move_to([0, -1.5, 0])
        mac = cite("generated on this Mac · 10,000 digits · 60,000 training digits").move_to([0, -1.95, 0])
        assert cnt.get_right()[0] < 6.5 and now_f.get_left()[0] > g_f.get_right()[0] + 0.3
        assert best_f.get_left()[0] > now_f.get_right()[0] + 0.4 and cnt.get_left()[0] > best_f.get_right()[0] + 0.25
        best_txt = [tinted(TRAIN_X[b], REAL) for _, _, b, _ in frames]
        now_txt = [tinted(TRAIN_X[c], REAL) for _, c, _, _ in frames]

        def search(a):
            k = min(int(a * len(frames)), len(frames) - 1)
            n, _, _, d = frames[k]
            now_r.show(now_txt[k])
            best_r.show(best_txt[k])
            cnt.show(f"checked {n:>6,} / 60,000")
            dist.show(f"distance {d:.2f}")

        screen_b = m.Group(g_f, g_l, now_f, best_f, now_l, best_l, cnt, dist, formula, mac)

        # twelve pairs
        n_ex = 12
        pitch, dh = 0.92, 0.8
        x0 = -4.5
        top_y, bot_y = 1.15, 0.2
        tops = m.Group(*[digit(EX_G[i], GEN, dh).move_to([x0 + i * pitch, top_y, 0]) for i in range(n_ex)])
        bots = m.Group(*[digit(EX_T[i], REAL, dh).move_to([x0 + i * pitch, bot_y, 0]) for i in range(n_ex)])
        nums = m.VGroup(*[code(f"{EX_D[i]:.2f}", 20, INK).move_to([x0 + i * pitch, bot_y - 0.66, 0]) for i in range(n_ex)])
        l_top = text("generated", 20, GEN, weight=m.BOLD)
        l_bot = m.VGroup(text("nearest of", 20, REAL, weight=m.BOLD), text("60,000", 20, REAL, weight=m.BOLD)).arrange(m.DOWN, buff=0.04, aligned_edge=m.LEFT)
        l_num = cite("distance")
        for lab, y in ((l_top, top_y), (l_bot, bot_y), (l_num, bot_y - 0.66)):
            lab.move_to([0, y, 0]).align_to([-6.45, 0, 0], m.LEFT)
        sit(l_num, baseline(nums[0]))
        assert l_top.get_right()[0] < tops[0].get_left()[0] - 0.12 and tops[-1].get_right()[0] < 6.55
        assert l_bot.get_right()[0] < bots[0].get_left()[0] - 0.12
        same = pill("the same digits, in different handwriting", REAL, TEAL_FILL, 22).move_to([0, -1.3, 0])
        ex_mac = cite("12 of the 10,000, chosen at random · generated on this Mac · DDIM 50 steps, w = 1").move_to([0, -1.95, 0])
        screen_c = m.Group(tops, bots, nums, l_top, l_bot, l_num, same, ex_mac)

        # the histogram
        plot = Plot((0, 11), (0, H_TOP * 1.15), width=8.2, height=3.6, center=(-2.1, -0.15, 0),
                    x_ticks=[0, 2, 4, 6, 8, 10], x_label="distance to the nearest training digit", size=20)
        xs_h = np.repeat(EDGES, 2)[1:-1]

        def outline(h):
            return np.repeat(h, 2)

        c_gen = plot.line(xs_h, outline(H_GEN), GEN, 4)
        c_test = plot.line(xs_h, outline(H_TEST), REAL, 4)
        a_gen = plot.area(xs_h, outline(H_GEN), GEN, 0.16)
        a_test = plot.area(xs_h, outline(H_TEST), REAL, 0.14)
        y_lab = plot.y1 * 0.97
        v_gen = plot.vline(MED["ddim"], GEN, 2.5)
        v_test = plot.vline(MED_TEST, REAL, 2.5)
        l_gen = m.VGroup(code(f"{MED['ddim']:.2f}", 22, GEN, weight=m.BOLD), text("generated, median", 20, GEN))
        l_gen.arrange(m.RIGHT, buff=0.12).next_to(plot.coords(MED["ddim"], y_lab), m.RIGHT, buff=0.12)
        l_test = m.VGroup(code(f"{MED_TEST:.2f}", 22, REAL, weight=m.BOLD), text("real test digits", 20, REAL))
        l_test.arrange(m.RIGHT, buff=0.12).next_to(plot.coords(MED_TEST, y_lab), m.LEFT, buff=0.12)
        for g_ in (l_gen, l_test):
            sit(g_[1], baseline(g_[0]))
        l_test.shift((baseline(l_gen[0]) - baseline(l_test[0])) * m.UP)
        l_test.shift((plot.coords(MED_TEST, 0)[0] - 0.12 - l_test.get_right()[0]) * m.RIGHT)
        v_rand = plot.vline(RAND, MUTED, 2.5)
        l_rand = m.VGroup(code(f"{RAND:.1f}", 22, INK, weight=m.BOLD), text("two random", 20, MUTED),
                          text("digits", 20, MUTED)).arrange(m.DOWN, buff=0.04)
        l_rand.next_to(plot.coords(RAND, plot.y1 * 0.6), m.LEFT, buff=0.15)
        v_p1 = m.DashedLine(plot.coords(P1, 0), plot.coords(P1, plot.y1 * 0.8), dash_length=0.1).set_stroke(BAD, 3)
        l_p1 = m.VGroup(text("closest 1%", 20, BAD, weight=m.BOLD), text("of real test", 20, BAD),
                        code(f"< {P1:.2f}", 20, BAD)).arrange(m.DOWN, buff=0.04, aligned_edge=m.LEFT)
        l_p1.next_to(plot.coords(P1, plot.y1 * 0.62), m.RIGHT, buff=0.1)
        assert l_test.get_left()[0] > l_p1.get_right()[0] + 0.1 or l_p1.get_top()[1] < l_test.get_bottom()[1] - 0.1, (l_test.get_left(), l_p1.get_right())
        assert l_rand.get_left()[0] > l_gen.get_right()[0] + 0.1 or l_rand.get_top()[1] < l_gen.get_bottom()[1] - 0.05
        tab_rows = [("real test digits", "1%", REAL), ("DDIM, 50 steps", PCT["ddim"], GEN),
                    ("DDPM, 1,000", PCT["ddpm"], GEN), ("flow, 50 steps", PCT["flow"], GEN)]
        rows = m.VGroup()
        for name, v, col in tab_rows:
            r_ = m.VGroup(text(name, 20, INK), code(v, 20, col, weight=m.BOLD))
            rows.add(r_)
        t_head = m.VGroup(text("closer than the closest", 20, INK, weight=m.BOLD),
                          text("1% of real test digits", 20, INK, weight=m.BOLD))
        t_head.arrange(m.DOWN, buff=0.04, aligned_edge=m.LEFT)
        tab_in = m.VGroup(t_head, rows).arrange(m.DOWN, buff=0.2, aligned_edge=m.LEFT)
        rows.arrange(m.DOWN, buff=0.14, aligned_edge=m.LEFT)
        for r_ in rows:
            r_[1].move_to([0, r_[0].get_y(), 0]).align_to(rows.get_left() + [2.2, 0, 0], m.LEFT)
            sit(r_[1], baseline(r_[0]))
        t_foot = cite("real test: 1% by definition")
        tab_in = m.VGroup(t_head, rows, t_foot).arrange(m.DOWN, buff=0.2, aligned_edge=m.LEFT)
        tab = boxed(tab_in, BAD, 0.25).move_to([4.6, 0.2, 0])
        h_mac = cite("10,000 generated vs 10,000 real test digits · measured on this Mac").move_to([-2.1, -3.3, 0])
        assert tab.get_right()[0] < 6.55 and tab.get_left()[0] > plot.coords(11, 0)[0] + 0.15, (tab.get_left(), plot.coords(11, 0))
        assert plot.get_bottom()[1] > h_mac.get_top()[1] + 0.05 and plot.get_top()[1] < 2.62, (plot.get_bottom(), plot.get_top())
        assert l_gen.get_right()[0] < plot.coords(RAND, 0)[0] - 0.2

        # the closest ones, and real test ones
        ch, cpitch = 1.0, 1.25
        cl_x0 = -5.6
        cl = m.Group()
        for j, i in enumerate(CLOSEST):
            x = cl_x0 + j * cpitch
            cl.add(m.Group(digit(GEN_X[i], GEN, ch).move_to([x, 0.95, 0]),
                           digit(TRAIN_X[NN_IDX[i]], REAL, ch).move_to([x, -0.2, 0]),
                           code(f"{D['ddim'][i]:.2f}", 20, INK).move_to([x, -0.95, 0])))
        cl_t = m.VGroup(text("the closest generated digits", 20, GEN, weight=m.BOLD),
                        cite("of 10,000 · above: generated"), cite("below: nearest training digit"))
        cl_t.arrange(m.DOWN, buff=0.05).move_to([cl_x0 + 1.5 * cpitch, 2.05, 0])
        tl = m.Group()
        tl_x0 = 1.9
        for j, (i, k) in enumerate(zip(TEST_ONES, TEST_NN)):
            x = tl_x0 + j * cpitch
            tl.add(m.Group(digit(TEST_X[i], REAL, ch).move_to([x, 0.95, 0]),
                           digit(TRAIN_X[k], REAL, ch).move_to([x, -0.2, 0]),
                           code(f"{D_TEST[i]:.2f}", 20, INK).move_to([x, -0.95, 0])))
        tl_t = m.VGroup(text("real test digits, never trained on", 20, REAL, weight=m.BOLD),
                        cite("above: a test digit"), cite("below: nearest training digit"))
        tl_t.arrange(m.DOWN, buff=0.05).move_to([tl_x0 + 0.5 * cpitch, 2.05, 0])
        if tl_t.get_right()[0] > 6.5:
            tl_t.shift((6.5 - tl_t.get_right()[0]) * m.RIGHT)
        assert cl_t.get_left()[0] > -6.55 and cl_t.get_right()[0] < tl_t.get_left()[0] - 0.3, (cl_t.get_right(), tl_t.get_left())
        verdict = pill("no sign of copying, at this scale", GOOD, GOOD_FILL, 24).move_to([0, -1.75, 0])
        scale = m.VGroup(code("3.9 M weights", 22, MODEL), m.Dot(radius=0.05, color=MUTED), code("60,000 digits", 22, REAL),
                         m.Dot(radius=0.05, color=MUTED), text("not billions of web pictures", 22, INK)).arrange(m.RIGHT, buff=0.3)
        sit(scale[2], baseline(scale[0]))
        sit(scale[4], baseline(scale[0]))
        for d_ in (scale[1], scale[3]):
            d_.set_y(baseline(scale[0]) + 0.1)
        scale.move_to([0, -2.55, 0])
        sc_c = cite("big models on duplicated web data can copy · Carlini et al., 2023").move_to([0, -3.2, 0])
        assert scale.width < 12.8

        self.say(
            "[Can a model copy its training pictures]? [In twenty twenty-three, Carlini and colleagues went looking].",
            self.new_header("04 · copying", "Does it copy its training data?"),
            m.FadeIn(c_c),
        )
        self.say(
            "[They took the three hundred fifty thousand most-duplicated captions in Stable Diffusion's training data], "
            "[made five hundred pictures for each], [a hundred seventy-five million in all].",
            m.FadeIn(stats[0], shift=0.1 * m.UP),
            m.FadeIn(stats[1], shift=0.1 * m.UP),
            m.FadeIn(stats[2], shift=0.1 * m.UP),
        )
        self.say(
            "[A hundred and nine were near-copies of training images], [about one in one point six million], "
            "[mostly pictures repeated a hundred times or more].",
            m.AnimationGroup(m.GrowArrow(a_f), m.FadeIn(a_lab), m.FadeIn(stats[3], shift=0.1 * m.LEFT)),
            m.FadeIn(rate, shift=0.1 * m.UP),
            m.FadeIn(dup, shift=0.1 * m.UP),
        )
        self.say(
            "[So we checked our digits]: [for each of ten thousand generated digits, "
            "we searched all sixty thousand training digits for the closest one].",
            m.AnimationGroup(m.FadeOut(screen_a), m.FadeIn(g_f, shift=0.1 * m.UP), m.FadeIn(g_l), m.FadeIn(now_f),
                             m.FadeIn(now_l), m.FadeIn(best_f), m.FadeIn(best_l), m.FadeIn(cnt), m.FadeIn(dist),
                             m.FadeIn(formula), m.FadeIn(mac)),
            drive(now_r.image, [search], run_time=4.4, rate=m.smooth),
        )
        self.say(
            "[Here are twelve]: [generated above], [the nearest training digit below], [and the distance between them]: "
            "[the same digits, in different handwriting].",
            m.AnimationGroup(m.FadeOut(screen_b), m.FadeIn(ex_mac)),
            m.AnimationGroup(m.FadeIn(l_top), m.LaggedStart(*[m.FadeIn(t_, shift=0.1 * m.DOWN) for t_ in tops], lag_ratio=0.08)),
            m.AnimationGroup(m.FadeIn(l_bot), m.LaggedStart(*[m.FadeIn(b_, shift=0.1 * m.UP) for b_ in bots], lag_ratio=0.08)),
            m.AnimationGroup(m.FadeIn(l_num), m.LaggedStart(*[m.FadeIn(n_) for n_ in nums], lag_ratio=0.08)),
            m.FadeIn(same, shift=0.1 * m.UP),
        )
        self.say(
            "[Across all ten thousand, the median distance is four point five two]; "
            "[for real test digits, which the model never saw, it is four point three four].",
            m.AnimationGroup(m.FadeOut(screen_c), m.FadeIn(plot), m.FadeIn(a_gen), m.Create(c_gen, run_time=1.4),
                             m.Create(v_gen), m.FadeIn(l_gen), m.FadeIn(h_mac)),
            m.AnimationGroup(m.FadeIn(a_test), m.Create(c_test, run_time=1.4), m.Create(v_test), m.FadeIn(l_test)),
        )
        self.say(
            "[Two random digits sit about ten point one apart].",
            m.AnimationGroup(m.Create(v_rand), m.FadeIn(l_rand, shift=0.1 * m.LEFT)),
        )
        self.say(
            "[Copies would pile up near zero]. "
            "[But only zero point four seven percent come closer than the closest one percent of real test digits].",
            m.AnimationGroup(m.Create(v_p1), m.FadeIn(l_p1, shift=0.1 * m.RIGHT)),
            m.AnimationGroup(m.FadeIn(tab[0]), m.FadeIn(t_head), m.FadeIn(t_foot), m.LaggedStart(*[m.FadeIn(r_) for r_ in rows], lag_ratio=0.25)),
        )
        hist = m.VGroup(plot, a_gen, a_test, c_gen, c_test, v_gen, v_test, l_gen, l_test, v_rand, l_rand, v_p1, l_p1, tab,
                        h_mac)
        self.say(
            "[The very closest generated digits are plain ones], [and real test ones, never trained on, come just as close].",
            m.AnimationGroup(m.FadeOut(hist), m.FadeIn(cl_t), m.LaggedStart(*[m.FadeIn(c_, shift=0.1 * m.UP) for c_ in cl],
                                                                            lag_ratio=0.15)),
            m.AnimationGroup(m.FadeIn(tl_t), m.LaggedStart(*[m.FadeIn(t_, shift=0.1 * m.UP) for t_ in tl], lag_ratio=0.2)),
        )
        self.say(
            "[So: no sign of copying, at our scale]: [three point nine million weights, sixty thousand digits], "
            "[not billions of web pictures].",
            m.FadeIn(verdict, shift=0.1 * m.UP),
            m.FadeIn(scale[:4], shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(scale[4], shift=0.1 * m.UP), m.FadeIn(sc_c)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 5 · preference tuning

    def preference(self):
        self.next_section("5 · Preference tuning", notes=(
            "ILLUSTRATIVE swirl: a made-up reward r(x) = −|x − c|² with c = (0.95, 0.72), just outside the swirl. "
            f"'Tuned': the swirl mixture tilted by exp({LAM:g}·r), each of {N_PTS} points sliding along its own arm "
            "(inverse CDF at a fixed quantile): mass moves toward c and stays on the data (all within 0.15 of the arms). "
            "'Pushed too hard': maximising r alone collapses every sample onto c, off the data. Black et al., 'Training "
            "Diffusion Models with Reinforcement Learning' (DDPO), arXiv:2305.13301 (May 2023; ICLR 2024): denoising as a "
            "multi-step decision process, policy gradients on a reward (aesthetic score, JPEG compressibility, prompt "
            "alignment judged by LLaVA); SD v1.4, 50 denoising steps (App. D.5); Fig. 7 / App. A: a typographic attack on "
            "the vision-language judge (writing the number as text instead of drawing that many animals). Wallace et al., "
            "'Diffusion Model Alignment Using Direct Preference Optimization', arXiv:2311.12908 (Nov 2023; CVPR 2024): no "
            "reward model; the loss compares the model's denoising error on the chosen vs the rejected picture, each "
            "relative to a frozen reference; SDXL-1.0 base + Pick-a-Pic (851K pairs): 70.0% general preference vs SDXL "
            "base on PartiPrompts (69% vs base + refiner). SD3 was DPO-aligned (Esser et al. 2024). The chosen/rejected "
            "pair is ILLUSTRATIVE: two of our lighthouses (seeds 0 and 1). Trial and Reward: Ep 21 'Learning from Human "
            "Preferences', Ep 22 'Direct Preference Optimization'."))
        plot = Plot((-1.3, 1.3), (-1.3, 1.3), width=4.9, height=4.9, center=(-2.9, -0.2, 0), axes=False)
        band = swirl_band(plot)
        cloud = plot.cloud(BASE_XY, GEN, radius=0.04, opacity=0.95)
        c_pt = plot.coords(*PREF_C)
        halo = m.VGroup(*[m.Circle(radius=0.22 + 0.11 * i).set_stroke(width=0).set_fill(GOOD, 0.06).move_to(c_pt)
                          for i in range(7)])
        ring = m.Circle(radius=0.16).set_stroke(GOOD, 3).move_to(c_pt)
        pref_l = m.VGroup(text("preferred", 20, GOOD, weight=m.BOLD), cite("a made-up preference"))
        pref_l.arrange(m.DOWN, buff=0.04, aligned_edge=m.LEFT).next_to(ring, m.RIGHT, buff=0.35)
        legend = m.VGroup(m.Dot(radius=0.07, color=GEN), text("the model's samples", 20, INK),
                          m.Line(m.ORIGIN, 0.45 * m.RIGHT).set_stroke(REAL, 8, opacity=0.35), text("the data", 20, INK))
        legend.arrange(m.RIGHT, buff=0.15)
        legend[2].shift(0.2 * m.RIGHT)
        legend[3].shift(0.2 * m.RIGHT)
        legend.move_to([plot.coords(0, 0)[0], -3.2, 0])
        x_r = 0.9
        leash = m.VGroup(pill("tuned toward the preference", GOOD, GOOD_FILL, 22),
                         cite("a leash keeps it near the original"), cite("model, so it stays on the data"))
        leash.arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT).move_to([0, 0.0, 0]).align_to([x_r, 0, 0], m.LEFT)
        hack = m.VGroup(pill("pushed too hard: off the data", BAD, BAD_FILL, 22),
                        cite("asked for a number of animals, it wrote"),
                        cite("the number as text, to fool its judge"),
                        cite("Black et al., 2023, Fig. 7"))
        hack.arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT).move_to([0, -1.75, 0]).align_to([x_r, 0, 0], m.LEFT)
        assert pref_l.get_right()[0] < x_r - 0.1 or pref_l.get_top()[1] > leash.get_top()[1] + 0.1
        assert leash.get_right()[0] < 6.55 and hack.get_right()[0] < 6.55, (leash.get_right(), hack.get_right())
        assert plot.get_top()[1] < 2.62 and legend.get_bottom()[1] > -3.55
        base_cols = [GEN] * N_PTS

        def drift(a):
            cloud.positions(BASE_XY + (TUNED_XY - BASE_XY) * a)

        def collapse(a):
            cloud.positions(TUNED_XY + (HACK_XY - TUNED_XY) * a)
            j = a * (len(FAR) - 1)
            j0 = int(np.floor(j))
            j1 = min(j0 + 1, len(FAR) - 1)
            far = FAR[j0] + (FAR[j1] - FAR[j0]) * (j - j0)
            for d_, f in zip(cloud, far):
                d_.set_color(mix(GEN, BAD, float(f)))

        # the two method cards
        cw, chh, cy = 6.1, 3.8, 0.66
        dd = card(cw, chh, GOOD, CARD, radius=0.18, stroke=2.5).move_to([-3.25, cy, 0])
        dp = card(cw, chh, GOOD, CARD, radius=0.18, stroke=2.5).move_to([3.25, cy, 0])

        def head(name, who, box):
            n_ = code(name, 26, GOOD, weight=m.BOLD)
            w_ = cite(who)
            n_.move_to(box.get_corner(m.UL) + [0.3 + n_.width / 2, -0.4, 0])
            w_.next_to(n_, m.DOWN, buff=0.08).align_to(n_, m.LEFT)
            return m.VGroup(n_, w_)

        dd_h = head("DDPO", "Black et al. · May 2023", dd)
        chain = tex(r"x_T \rightarrow x_{T-1} \rightarrow \cdots \rightarrow x_0", 34, INK)
        chain.move_to(dd.get_corner(m.UL) + [0.3 + chain.width / 2, -1.42, 0])
        rew = pill("reward", GOOD, GOOD_FILL, 22).next_to(chain, m.RIGHT, buff=0.5)
        a_rew = arrow(chain.get_right(), rew.get_left(), GOOD, buff=0.1)
        back = m.CurvedArrow(rew.get_bottom() + [0, -0.1, 0], chain.get_bottom() + [0.5, -0.22, 0], angle=-0.45)
        back.set_stroke(FLOW, 3)
        dd_lines = m.VGroup(text(f"{DDPO_STEPS} denoising steps = {DDPO_STEPS} decisions", 20, INK),
                            text("policy gradients push up what scored well", 20, INK),
                            cite("rewards: aesthetic score, JPEG size,"),
                            cite("a vision-language judge · on SD 1.4")).arrange(m.DOWN, buff=0.09, aligned_edge=m.LEFT)
        dd_lines.next_to(chain, m.DOWN, buff=0.62).align_to(chain, m.LEFT)
        assert rew.get_right()[0] < dd.get_right()[0] - 0.2 and dd_lines.get_bottom()[1] > dd.get_bottom()[1] + 0.12, (rew.get_right(), dd_lines.get_bottom())
        assert dd_lines.get_right()[0] < dd.get_right()[0] - 0.15
        dp_h = head("Diffusion-DPO", "Wallace et al. · Nov 2023", dp)
        th = 0.95
        win = framed_photo(LH_RGB, th, GOOD, 3).move_to(dp.get_corner(m.UL) + [0.3 + th / 2 + 0.04, -1.62, 0])
        lose = framed_photo(LH_B, th, BAD, 3).next_to(win, m.RIGHT, buff=0.3)
        w_l = m.VGroup(mark(True, 22), text("chosen", 20, GOOD)).arrange(m.RIGHT, buff=0.08).next_to(win, m.DOWN, buff=0.1)
        l_l = m.VGroup(mark(False, 22), text("rejected", 20, BAD)).arrange(m.RIGHT, buff=0.08).next_to(lose, m.DOWN, buff=0.1)
        l_l.shift((baseline(w_l[1]) - baseline(l_l[1])) * m.UP)
        p_note = m.VGroup(text("no reward model:", 20, INK, weight=m.BOLD), text("pairs people chose", 20, INK),
                          text("between", 20, INK)).arrange(m.DOWN, buff=0.06, aligned_edge=m.LEFT)
        p_note.next_to(lose, m.RIGHT, buff=0.35).align_to(lose, m.UP)
        loss = m.VGroup(text("denoising error vs a frozen copy:", 20, INK),
                        m.VGroup(text("lower on the chosen,", 20, GOOD), text("higher on the rejected", 20, BAD))
                        .arrange(m.RIGHT, buff=0.15)).arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT)
        loss.next_to(w_l, m.DOWN, buff=0.22).align_to(win, m.LEFT)
        sit(loss[1][1], baseline(loss[1][0]))
        ex_l = cite("pictures: for example").next_to(p_note, m.DOWN, buff=0.12).align_to(p_note, m.LEFT)
        assert p_note.get_right()[0] < dp.get_right()[0] - 0.15 and loss.get_right()[0] < dp.get_right()[0] - 0.15, (p_note.get_right(), loss.get_right())
        assert loss.get_bottom()[1] > dp.get_bottom()[1] + 0.1 and ex_l.get_bottom()[1] > w_l.get_top()[1] - 0.6
        res = m.VGroup(code(f"SDXL + {PICK_PAIRS // 1000} K Pick-a-Pic pairs:", 22, INK),
                       code(f"preferred {DPO_WIN:.1f}% over SDXL base", 22, GOOD, weight=m.BOLD)).arrange(m.RIGHT, buff=0.3)
        res.move_to([0, -1.62, 0])
        res_c = cite(f"PartiPrompts, general preference ({DPO_WIN_REFINER}% vs base + refiner) · "
                     "Stable Diffusion 3 was also tuned with DPO").move_to([0, -2.08, 0])
        tr_t = text("the same ideas for language models, in our sister course Trial and Reward", 20, MUTED).move_to([0, -2.6, 0])
        tr = m.VGroup(pill("Episode 21 · Learning from Human Preferences", INK, CARD_HI, 20),
                      pill("Episode 22 · Direct Preference Optimization", INK, CARD_HI, 20)).arrange(m.RIGHT, buff=0.35)
        tr.move_to([0, -3.15, 0])
        assert res.width < 12.8 and res_c.width < 13.0 and tr.width < 13.1 and dd.get_bottom()[1] > res.get_top()[1] + 0.12, (res.width, res_c.width, tr.width, dd.get_bottom(), res.get_top())
        screen_a = m.VGroup(band, cloud, halo, ring, pref_l, legend, leash, hack)

        self.say(
            "[Last, preference]: [after pretraining, a model can be tuned toward the pictures people like].",
            m.AnimationGroup(self.new_header("05 · preference", "Tuned toward what people prefer"), m.FadeIn(band)),
            m.AnimationGroup(m.FadeIn(cloud), m.FadeIn(legend)),
        )
        self.say(
            "[Here's the idea on our swirl, with a made-up preference]: [say people like points near this corner].",
            m.Circumscribe(band, color=REAL, buff=0.05),
            m.AnimationGroup(m.FadeIn(halo, scale=0.6), m.Create(ring), m.FadeIn(pref_l, shift=0.1 * m.LEFT)),
        )
        self.say(
            "[Tuning shifts the samples toward what scores well], "
            "[while a leash keeps the model close to where it started].",
            drive(cloud, [drift], run_time=3.0, rate=m.smooth),
            m.FadeIn(leash, shift=0.1 * m.UP),
        )
        self.say(
            "[Push too hard, and the model games its score]: [in one study, asked for a number of animals, "
            "it wrote the number into the picture to fool its judge].",
            drive(cloud, [collapse], run_time=2.6, rate=m.smooth),
            m.FadeIn(hack, shift=0.1 * m.UP),
        )
        self.say(
            "[That study, by Black and colleagues in twenty twenty-three, treated the fifty denoising steps as fifty decisions], "
            "[and used policy gradients to raise a reward].",
            m.AnimationGroup(m.FadeOut(screen_a), m.FadeIn(dd), m.FadeIn(dd_h), m.Write(chain), m.GrowArrow(a_rew),
                             m.FadeIn(rew), m.FadeIn(dd_lines[0])),
            m.AnimationGroup(m.Create(back), m.FadeIn(dd_lines[1:], shift=0.1 * m.UP)),
        )
        self.say(
            "[Months later, Diffusion-DPO dropped the reward model]: "
            "[it trains directly on pairs of pictures where people picked one].",
            m.AnimationGroup(m.FadeIn(dp), m.FadeIn(dp_h)),
            m.AnimationGroup(m.FadeIn(win, shift=0.1 * m.UP), m.FadeIn(lose, shift=0.1 * m.UP), m.FadeIn(w_l),
                             m.FadeIn(l_l), m.FadeIn(p_note), m.FadeIn(ex_l)),
        )
        self.say(
            "[Its loss lowers the model's denoising error on the chosen picture, compared with a frozen copy], "
            "[and raises it on the rejected one].",
            m.AnimationGroup(m.FadeIn(loss[0]), m.FadeIn(loss[1][0]), m.Circumscribe(win[0], color=GOOD, buff=0.05)),
            m.AnimationGroup(m.FadeIn(loss[1][1]), m.Circumscribe(lose[0], color=BAD, buff=0.05)),
        )
        self.say(
            "[Tuned on eight hundred fifty-one thousand such choices], "
            "[the new SDXL was preferred seventy percent of the time over the original].",
            m.FadeIn(res[0], shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(res[1], shift=0.1 * m.UP), m.FadeIn(res_c)),
        )
        self.say(
            "[For language models, our sister course Trial and Reward covers the same ideas]: "
            "[in learning from human preferences], [and direct preference optimization].",
            m.FadeIn(tr_t),
            m.FadeIn(tr[0], shift=0.1 * m.UP),
            m.FadeIn(tr[1], shift=0.1 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes=(
            "Data: what the model is shown (LAION-5B's CLIP filter, aesthetic subsets) and whether it copies it (Carlini "
            f"et al.: 109 of 175 M; our digits: median {MED['ddim']:.2f} vs {MED_TEST:.2f} for real test digits, no sign "
            "of copying at 3.9 M weights / 60,000 digits). Captions: what it is told (DALL·E 3: 95% descriptive synthetic "
            "captions). Preference: what it is tuned to like (DDPO, Diffusion-DPO; over-optimising games the score). "
            "Module 5 (Scaling Up) closes: 17 latent diffusion, 18 diffusion transformers, 19 measuring pictures, 20 data "
            "and alignment. Next: Episode 21, Module 6 (video)."))
        halo = glow(DATA, radius=4.0)
        words = m.VGroup(*[title_text(w, 54, c) for w, c in (("Data", DATA), ("Captions", CAPTION), ("Preference", GOOD))])
        subs = m.VGroup(*[code(s, 20, MUTED) for s in ("what it is shown", "what it is told", "what it is tuned to like")])
        words.arrange(m.RIGHT, buff=1.25)
        on_baseline(words)
        for w, s in zip(words, subs):
            s.set_x(w.get_x())
        line_y = min(w.get_bottom()[1] for w in words) - 0.45
        for s in subs:
            sit(s, line_y)
        mid = baseline(words[0]) + 0.3
        dots = m.VGroup(*[m.Dot([(a.get_right()[0] + b.get_left()[0]) / 2, mid, 0], radius=0.07, color=MUTED)
                          for a, b in zip(words, words[1:])])
        row = m.VGroup(words, dots, subs).move_to(1.4 * m.UP)
        assert row.width < 12.6 and all(a.get_right()[0] + 0.3 < b.get_left()[0] for a, b in zip(subs, subs[1:])), (row.width, [w.width for w in words], [s_.width for s_ in subs])
        mod = m.VGroup(*[slim_pill(s, DATA, TEAL_FILL) for s in ("17 · latent diffusion", "18 · transformers",
                                                                "19 · measuring pictures", "20 · data and alignment")])
        mod.arrange(m.RIGHT, buff=0.2).move_to([0, -0.75, 0])
        mod_t = text("Module 5 · Scaling up: complete", 24, INK, weight=m.BOLD).next_to(mod, m.UP, buff=0.3)
        assert mod.width < 12.9, mod.width
        mascot = pixel(0.42)
        nxt = pill("NEXT · MODULE 6 · VIDEO AND YOUR MAC", LIME, LIME_FILL, 20, mono=True)
        foot = m.VGroup(mascot, nxt).arrange(m.RIGHT, buff=0.3).move_to([0, -2.3, 0])

        self.say(
            "[To sum up]: [data], [captions], [preference].",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.3 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(dots[0]), m.FadeIn(words[1], shift=0.3 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(dots[1]), m.FadeIn(words[2], shift=0.3 * m.UP), m.FadeIn(subs[2])),
        )
        self.say(
            "[A model learns what its data shows], [what its captions say], [and what it is tuned to prefer].",
            m.Circumscribe(words[0], color=DATA, buff=0.1),
            m.Circumscribe(words[1], color=CAPTION, buff=0.1),
            m.Circumscribe(words[2], color=GOOD, buff=0.1),
        )
        self.say(
            "[That completes module five]: [latent diffusion], [transformers], [measuring pictures], "
            "[and today, data and alignment].",
            m.FadeIn(mod_t, shift=0.1 * m.UP),
            m.FadeIn(mod[0], shift=0.1 * m.UP),
            m.FadeIn(mod[1], shift=0.1 * m.UP),
            m.FadeIn(mod[2], shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(mod[3], shift=0.1 * m.UP), m.FadeIn(mascot, shift=0.2 * m.UP),
                             m.FadeIn(nxt, shift=0.2 * m.UP)),
        )
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, mod, mod_t, foot)], run_time=0.9)
