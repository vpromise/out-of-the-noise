"""Out of the Noise · Episode 3 · Before Diffusion (Module 1 · Pictures from Noise)

Earlier generators traded quality, diversity and stability. Running example: the swirl (a collapsed generator covering
one arm; the swirl again for diffusion's answer), with real MNIST digits (the digit model's data) for the VAE's blur
and the pixel-by-pixel order. Sections: hook (Episode 2's blurry average seven; a timeline 2013–2021) · autoencoder
(a linear autoencoder, PCA with 32 numbers, computed: rebuilds a seven; random codes decode to a smudge) · VAE (codes
pulled into one round cloud; why it blurs: squared error makes the unsure decoder predict the average — two real sevens
averaged, then the 4 … 256 nearest training sevens) · GANs (generator vs critic on the swirl; the critic's map is the
best critic for the shown points, computed; a collapsed generator on one arm, illustrative; the chase round the arms) ·
pixel by pixel (one network call per value: 784, 3,072, 196,608) · diffusion (the denoising loop; the digit model's
guess starts as a blurry average and sharpens; the swirl MLP's DDPM run covers both arms) · scorecard · 2021 (Dhariwal &
Nichol, Table 5: ADM-G FID 4.59 vs BigGAN-deep 6.95, precision 0.82 vs 0.87, recall 0.52 vs 0.28; 962 V100-days;
StyleGAN-XL 2.30 in 2022) · recap.

Sources (research/notes.md §3, §1 timeline, Appendix A row "3 · Diffusion beats GANs", Appendix C row 3; facts dated
October 2026):
- Kingma & Welling 2013 (VAE, arXiv:1312.6114); Goodfellow et al. 2014 (GAN, arXiv:1406.2661; §6 "the Helvetica
  scenario"; Prop. 1: the best critic for a fixed generator is p_data / (p_data + p_g)); Sohl-Dickstein et al. 2015
  (diffusion, arXiv:1503.03585); van den Oord et al. 2016 (PixelRNN arXiv:1601.06759, PixelCNN arXiv:1606.05328: one
  value at a time, 3,072 for a 32×32 colour image); Ho et al. 2020 (DDPM, 1,000 steps).
- Dhariwal & Nichol 2021, "Diffusion Models Beat GANs on Image Synthesis" (arXiv:2105.05233), Table 5, ImageNet
  256×256: BigGAN-deep FID 6.95 / P 0.87 / R 0.28; ADM-G (250 steps, classifier guidance) FID 4.59 / P 0.82 / R 0.52;
  Table 10: ADM-G 916 + 46 = 962 V100-days; BigGAN-deep 128–256 V100-days (converted from TPU estimates). Checked
  against the paper (ar5iv HTML) on 2026-10-26.
- Sauer, Schwarz & Geiger 2022 (StyleGAN-XL, arXiv:2202.00273), Table 2: ImageNet 256 FID 2.30 (Appendix A: "diffusion
  beats GANs" was a 2021 result; StyleGAN-XL led in early 2022).
- Why diffusion won anyway (notes §3, common misconceptions): stability, coverage, text conditioning and scaling.
- Runs on this Mac: the swirl MLP (45,954 parameters, research/experiments/toy.md; its DDPM 1,000-step run from
  gaussian_noise(2000, seed=5), sampler seed 1, recomputed here and asserted equal to toy.json's stored cloud and
  metrics); the digit model (research/experiments/logs/train_ddpm_config.json: U-Net, 3,940,801 parameters, MLX; its
  DDPM 1,000-step trajectory of a 7 from arrays/digit_trajectory_7.npz and samples from
  arrays/digit_samples_ddpm_ddpm1000_w1.npz). mac_runs.md did not exist yet, so the arrays and config are read directly.

Computed in this file (seeded, asserted): the mean of the 6,265 training sevens; the pair average of training sevens
15 and 42 and its 93 split pixels; averages of the 1/4/16/64/256 training sevens nearest to seven 15 (Euclidean, pixels
in −1…1); a linear autoencoder (PCA on the first 10,000 training digits, 32 components) and a random code's decode; the
critic maps; the collapsed points' share on the swirl and coverage (swirl_coverage, off_swirl); 28·28, 32·32·3, 256·256·3.

Illustrative (said so on screen): the encoder/decoder and code-space drawings; the blurry decode in the VAE diagram (it
is the average of 64 real sevens, revealed in the next lines); the GAN generator's point positions (a blob, half-way,
one arm, the other arm), labelled "illustration"; the pixel-by-pixel reveal of a real MNIST two (the order, not a model's
output); the denoising loop's pictures (kit.denoise_loop).

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403
from diffusion_reference import *  # noqa: E402,F403

HERE = Path(__file__).resolve().parent
PREFIX = "ep03"
ARRAYS = EXPERIMENTS / "arrays"
TOY = lambda *keys: measured("toy", *keys)  # noqa: E731

# ---------------------------------------------------------------------------- real digits (MNIST training set)

IMAGES, LABELS = load_mnist("train")
X = IMAGES / 127.5 - 1.0                                   # the model's range, −1 … 1
SEVENS = np.flatnonzero(LABELS == 7)
assert IMAGES.shape == (60000, 28, 28) and len(SEVENS) == 6265          # "over six thousand sevens"
MEAN7 = X[SEVENS].mean(0)                                  # Episode 2's blur

A_IDX, B_IDX = int(SEVENS[0]), 42                          # the first seven (Episode 4's digit) and a narrow one
assert A_IDX == 15 and LABELS[B_IDX] == 7
PAIR = (X[A_IDX] + X[B_IDX]) / 2                           # the squared-error-best guess between the two
SPLIT = (X[A_IDX] > 0) ^ (X[B_IDX] > 0)                    # ink in one, background in the other
N_SPLIT = int(SPLIT.sum())
BOTH = (X[A_IDX] > 0) & (X[B_IDX] > 0)
assert N_SPLIT == 93
assert abs(PAIR[SPLIT].mean()) < 0.25 and PAIR[BOTH].mean() > 0.6     # split pixels end grey, shared ink stays bright
# the squared error of the average is lower than betting on either seven, whichever one was meant
_err = lambda guess: 0.5 * (((guess - X[A_IDX]) ** 2).sum() + ((guess - X[B_IDX]) ** 2).sum())  # noqa: E731
assert _err(PAIR) < _err(X[A_IDX]) and _err(PAIR) < _err(X[B_IDX])

_flat7 = X[SEVENS].reshape(len(SEVENS), -1)
NEAR = SEVENS[np.argsort(np.linalg.norm(_flat7 - X[A_IDX].reshape(-1), axis=1), kind="stable")]
assert NEAR[0] == A_IDX
KS = [1, 4, 16, 64, 256]
AVG_K = {k: X[NEAR[:k]].mean(0) for k in KS}
GREY = {k: float(np.mean(np.abs(AVG_K[k]) < 0.5)) for k in KS}        # share of half-tone pixels
assert all(GREY[a] < GREY[b] for a, b in zip(KS, KS[1:]))              # "the strokes melt": more grey every time

# a linear autoencoder: PCA on 10,000 training digits, 32 numbers per digit (computed, no training run)
_fit = X[:10000].reshape(10000, -1)
PCA_MU = _fit.mean(0)
_, PCA_S, PCA_VT = np.linalg.svd(_fit - PCA_MU, full_matrices=False)
N_CODE = 32
W_CODE = PCA_VT[:N_CODE]
CODE_STD = PCA_S[:N_CODE] / np.sqrt(len(_fit) - 1)
CODE = (X[A_IDX].reshape(-1) - PCA_MU) @ W_CODE.T
REC = (PCA_MU + CODE @ W_CODE).reshape(28, 28)
RAND_CODE = np.random.default_rng(7).uniform(-3, 3, N_CODE) * CODE_STD
SMUDGE = (PCA_MU + RAND_CODE @ W_CODE).reshape(28, 28)
assert np.mean((REC - X[A_IDX]) ** 2) < 0.1                            # the copy is close to the original
_nn = lambda img: float(np.min(np.linalg.norm(_fit - img.reshape(-1), axis=1)))  # noqa: E731
assert _nn(SMUDGE) > 1.4 * _nn(REC)                                     # the smudge is far from every real digit

TWO_IDX = int(np.flatnonzero(LABELS == 2)[0])
assert TWO_IDX == 5
CALLS = [(28, 28, 1), (32, 32, 3), (256, 256, 3)]
N_CALLS = [h * w * c for h, w, c in CALLS]
assert N_CALLS == [784, 3072, 196608]                                   # "784", "3,072", "almost 200,000"

# ---------------------------------------------------------------------------- the swirl: a collapsed generator

BW = 0.09                                                  # smoothing for both densities in the critic's map
REAL_XY = swirl(600, seed=30)
_xy, _arm = swirl_labelled(560, seed=31)
_ang = lambda p: np.arctan2(p[:, 1], p[:, 0])  # noqa: E731
FAKE_ARM = _xy[_arm == 0]
FAKE_ARM = FAKE_ARM[np.argsort(_ang(FAKE_ARM), kind="stable")]           # illustrative: every fake on arm 1
N_FAKE = len(FAKE_ARM)
BLOB = 0.28 * gaussian_noise(N_FAKE, seed=21)
BLOB = BLOB[np.argsort(_ang(BLOB), kind="stable")]                        # paired by angle: no crossing paths
HALF = BLOB + 0.55 * (FAKE_ARM - BLOB)
ON_SWIRL = 1 - off_swirl(FAKE_ARM)
COVER = swirl_coverage(FAKE_ARM)
assert ON_SWIRL == 1.0 and COVER == 0.5                                   # "every point on the swirl", "half"
assert swirl_coverage(-FAKE_ARM) == 0.5 and swirl_coverage(REAL_XY) == 1.0

REAL_MIX = swirl_mixture(noise=float(np.hypot(SWIRL_NOISE, BW)))
MAP_N, MAP_LIM = 160, 1.2


def critic_rgba(fakes, alpha=0.55):
    """The best possible critic for these fakes (Goodfellow et al. 2014, Prop. 1): D = p_data / (p_data + p_fake),
    both densities smoothed by BW. GREEN where D > ½ (looks real), RED where D < ½ (looks fake), clear where neither
    density reaches (no evidence either way)."""
    g = np.linspace(-MAP_LIM, MAP_LIM, MAP_N)
    pts = np.array([(x, y) for y in g[::-1] for x in g])
    fake_mix = Mixture(np.asarray(fakes), np.full(len(fakes), BW), np.full(len(fakes), 1 / len(fakes)),
                       np.zeros(len(fakes), int))
    pd = np.exp(mixture_posterior(pts, 1.0, 0.0, REAL_MIX).log_density)
    pg = np.exp(mixture_posterior(pts, 1.0, 0.0, fake_mix).log_density)
    d = pd / (pd + pg + 1e-300)
    conf = (pd + pg) / (pd + pg + 0.05)
    s = (np.clip((d - 0.5) * 2, -1, 1) * conf).reshape(MAP_N, MAP_N)
    rgba = np.zeros((MAP_N, MAP_N, 4), np.uint8)
    green = np.array([int(GREEN[i:i + 2], 16) for i in (1, 3, 5)])
    red = np.array([int(RED[i:i + 2], 16) for i in (1, 3, 5)])
    rgba[..., :3] = np.where(s[..., None] >= 0, green, red)
    rgba[..., 3] = (np.abs(s) * 255 * alpha).round().astype(np.uint8)
    return rgba, d.reshape(MAP_N, MAP_N)


MAPS = [critic_rgba(f) for f in (BLOB, HALF, FAKE_ARM, -FAKE_ARM)]
_i0 = MAP_N // 2
assert MAPS[0][1][_i0, _i0] < 0.2                          # the blob's centre: "looks fake"
_arm2 = swirl_curve(40)[1][25]                             # a point on the empty arm
_ij = lambda p: (int(round((MAP_LIM - p[1]) / (2 * MAP_LIM) * (MAP_N - 1))),  # noqa: E731
                 int(round((p[0] + MAP_LIM) / (2 * MAP_LIM) * (MAP_N - 1))))
assert MAPS[2][1][_ij(_arm2)] > 0.95 and MAPS[2][1][_ij(-_arm2)] < 0.45    # the critic catches on: the empty arm is real

# ---------------------------------------------------------------------------- diffusion: runs on this Mac

SW_CACHE = HERE / "ep03_swirl_ddpm.npz"


def swirl_run():
    """The swirl MLP's DDPM run (toy.md §3: noise gaussian_noise(2000, seed=5), sampler seed 1), cached in this folder."""
    if SW_CACHE.exists():
        z = np.load(SW_CACHE)
        return z["path"], z["final"]
    tr = ddpm_sample(load_swirl_model(), gaussian_noise(2000, seed=5), seed=1)
    path, final = tr.xs[::8, :1000], tr.xs[-1]
    np.savez_compressed(SW_CACHE, path=path, final=final)
    return path, final


SW_PATH, SW_FINAL = swirl_run()
_run = TOY("samplers", "mlp", "runs", "ddpm_1000")
assert np.abs(SW_FINAL[:1000] - np.array(TOY("samplers", "mlp", "clouds", "ddpm_1000"))).max() < 1e-3
assert abs(swirl_distance(SW_FINAL) - _run["dist"]) < 5e-5 and abs(off_swirl(SW_FINAL) - _run["off"]) < 1e-9
assert swirl_coverage(SW_FINAL) == _run["coverage"] == 1.0 and _run["nfe"] == 1000
assert np.allclose(SW_PATH[-1], SW_FINAL[:1000]) and len(SW_PATH) == 126
SW_ON = 1 - off_swirl(SW_PATH[-1])
SW_COVER = swirl_coverage(SW_PATH[-1])
assert f"{100 * SW_ON:.1f}" == "99.3" and SW_COVER == 1.0                  # the 1,000 points shown
MLP_PARAMS = TOY("headline", "mlp_params")
assert MLP_PARAMS == 45954

_traj = np.load(ARRAYS / "digit_trajectory_7.npz")
TRAJ_T = _traj["ddpm_ddpm1000_w1_t"].astype(int)
TRAJ_X = _traj["ddpm_ddpm1000_w1_x_t"].astype(float)
TRAJ_X0 = _traj["ddpm_ddpm1000_w1_x0_hat"].astype(float)
TRAJ_FINAL = _traj["ddpm_ddpm1000_w1_final"].astype(float)
assert int(_traj["label"]) == 7 and TRAJ_X.shape == (1000, 28, 28) and TRAJ_T[0] == 999 and TRAJ_T[-1] == 0
_samples = np.load(ARRAYS / "digit_samples_ddpm_ddpm1000_w1.npz")
GEN7 = _samples["samples"][7::10][:6]
assert np.all(_samples["labels"][7::10][:6] == 7)
DIGIT_CFG = json.loads((EXPERIMENTS / "logs" / "train_ddpm_config.json").read_text())
assert DIGIT_CFG["params"] == 3940801 and DIGIT_CFG["arch"] == "unet"                 # "3.9M parameters"
_gen = json.loads((EXPERIMENTS / "results" / "gen_ddpm_ddpm1000_w1.json").read_text())
assert _gen["steps"] == 1000 and _gen["sampler"] == "ddpm"

# ---------------------------------------------------------------------------- the 2021 table (cited)

ADM_G = dict(fid=4.59, p=0.82, r=0.52)        # Dhariwal & Nichol 2021, Table 5, ImageNet 256×256, 250 steps
BIGGAN = dict(fid=6.95, p=0.87, r=0.28)       # same table, BigGAN-deep
STYLEGAN_XL = 2.30                            # Sauer et al. 2022, Table 2, ImageNet 256
ADM_DAYS = 916 + 46                           # Table 10: generator + classifier, V100-days
assert ADM_DAYS == 962 and ADM_G["fid"] < BIGGAN["fid"] and ADM_G["r"] > BIGGAN["r"] and ADM_G["p"] < BIGGAN["p"]
assert STYLEGAN_XL < ADM_G["fid"]

# ---------------------------------------------------------------------------- look

FLAT = set("abcdefhiklmnorstuvwxzABCDEFGHIJKLMNOPRSTUVWXYZ0123456789")
DIFF = VIOLET                 # diffusion's milestones in this episode (the module's accent)
GAN_C = MODEL                 # the generator networks (AMBER)
UNSEEN = mix(BG, FAINT, 0.2)


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


def texture(rgb):
    """The RGBA texture picture() makes from a small image (crisp pixels), for swapping in place."""
    k = max(1, int(np.ceil(540 / max(rgb.shape[:2]))))
    arr = np.repeat(np.repeat(rgb, k, axis=0), k, axis=1)
    rgba = np.concatenate([arr, np.full(arr.shape[:2] + (1,), 255, np.uint8)], axis=-1)
    rgba.flags.writeable = False
    return rgba


class Live:
    """A small picture whose pixels change during an animation (its texture is swapped each frame)."""

    def __init__(self, rgb, height):
        self.image = picture(rgb, height)

    def show(self, rgb):
        self.image.paint = self.image.paint.but(texture=texture(rgb))


def drive(mob, hook, run_time=1.4, rate=m.smooth):
    """Call hook(a) for a = 0 … 1 every frame."""
    return m.UpdateFromAlphaFunc(mob, lambda _, a: hook(a), run_time=run_time, rate_func=rate)


def digit(img, color=DATA, height=2.0, label=None, frame=EDGE, size=20):
    """A real (or generated) digit, tinted, in a thin frame with an optional mono caption."""
    return framed(picture(tint(img, color, -1, 1), height), frame, label=label, size=size)


def funnel(center, squeeze=True, wide=1.9, narrow=0.85, length=1.35, label=""):
    """An encoder (squeeze=True: wide → narrow) or decoder (narrow → wide), AMBER: a network."""
    x, y = center[0], center[1]
    hl, hr = (wide, narrow) if squeeze else (narrow, wide)
    l, r = x - length / 2, x + length / 2
    shape = m.Polygon([l, y + hl / 2, 0], [r, y + hr / 2, 0], [r, y - hr / 2, 0], [l, y - hl / 2, 0],
                      stroke_color=MODEL, stroke_width=2.5, fill_color=AMBER_FILL, fill_opacity=1)
    words = text(label, 20, INK, weight=m.BOLD).move_to([x, y, 0])
    return m.VGroup(shape, words)


def code_cells(values, color, cols=8, rows=4, cell=0.27, gap=0.05):
    """A code as a grid of cells; brightness = |value| (values in units of each number's spread)."""
    grid = cell_grid(rows, cols, cell, gap)
    cells = [sq for row in grid for sq in row]
    for sq, v in zip(cells, values):
        sq.set_fill(mix(CARD, color, 0.18 + 0.72 * min(abs(float(v)) / 2.5, 1.0)), 1).set_stroke(EDGE, 1.2)
    return grid, cells


def tick_mark(color=GREEN, size=0.34, width=6):
    s = size
    return m.VMobject().set_points_as_corners([[-0.5 * s, 0.0, 0], [-0.12 * s, -0.38 * s, 0], [0.55 * s, 0.42 * s, 0]]
                                              ).set_stroke(color, width).set_fill(opacity=0)


def cross_mark(color=RED, size=0.3, width=6):
    s = size / 2
    return m.VGroup(m.Line([-s, -s, 0], [s, s, 0]), m.Line([-s, s, 0], [s, -s, 0])).set_stroke(color, width)


def swatch(value, color=DATA, side=0.62):
    """One pixel of value −1 … 1, drawn as a tinted square."""
    rgb = tint(np.array([[value]]), color, -1, 1)[0, 0]
    return m.Square(side).set_stroke(EDGE, 2).set_fill("#{:02X}{:02X}{:02X}".format(*rgb), 1)


class Counter(m.VGroup):
    """A number (mono, bold) left-aligned at a fixed point, e.g. a step counter t = 999 (BLUE: time)."""

    def __init__(self, left, y, value=0, size=44, color=TIME, prefix="", fmt="{:,}"):
        super().__init__()
        self.anchor = np.array([left, y, 0.0])
        self.size, self.color, self.prefix, self.fmt = size, color, prefix, fmt
        self.words = self._make(value)
        self.add(self.words)

    def _make(self, value):
        return code(self.prefix + self.fmt.format(int(round(value))), self.size, self.color, weight=m.BOLD).move_to(
            self.anchor, aligned_edge=m.LEFT)

    def set_value(self, value):
        self.words.become(self._make(value))


def rot(xy, angle):
    c, s = np.cos(angle), np.sin(angle)
    return xy @ np.array([[c, s], [-s, c]])


# ---------------------------------------------------------------------------- the episode


class Ep03BeforeDiffusion(Episode):
    number = 3

    def lesson(self):
        self.hook()
        self.autoencoder()
        self.vae()
        self.gans()
        self.pixel_by_pixel()
        self.diffusion()
        self.scorecard()
        self.moment()
        self.recap()

    # ------------------------------------------------------------------ 0 · hook

    def hook(self):
        self.next_section("Hook", notes=(
            "Callback to Episode 2: the mean of all 6,265 MNIST training sevens (load_mnist('train'), pixels −1…1) is a "
            "blur. Timeline (arXiv v1 dates, notes §1/§3): VAE Dec 2013 (Kingma & Welling), GAN Jun 2014 (Goodfellow et "
            "al.), diffusion Mar 2015 (Sohl-Dickstein et al.), PixelRNN/PixelCNN 2016 (van den Oord et al.), DDPM Jun 2020 "
            "(Ho et al.), 'Diffusion Models Beat GANs' May 2021 (Dhariwal & Nichol). Running example: the swirl, with real "
            "MNIST digits for the blur."))
        blur = digit(MEAN7, DATA, 2.5).move_to([-4.3, 0.55, 0])
        cap = code("average of 6,265 training sevens", 20, INK).next_to(blur, m.DOWN, buff=0.25)
        ep2 = cite("Episode 2 · Images Are Numbers").next_to(cap, m.DOWN, buff=0.14)

        rows = [("2013", "VAE", INK), ("2014", "GAN", INK), ("2015", "diffusion", DIFF), ("2016", "PixelCNN", INK),
                ("2020", "DDPM", DIFF), ("2021", "“Diffusion Models Beat GANs”", DIFF)]
        x_year, x_dot, x_name = -0.4, 0.8, 1.15
        ys = [1.85 - 0.62 * i for i in range(len(rows))]
        spine = m.Line([x_dot, ys[0] + 0.25, 0], [x_dot, ys[-1] - 0.25, 0]).set_stroke(EDGE, 3)
        items = []
        for (year, name, color), y in zip(rows, ys):
            dot = m.Dot([x_dot, y, 0], radius=0.075, color=color)
            words = text(name, 26, color, weight=m.BOLD).move_to([0, y, 0], aligned_edge=m.LEFT).set_x(x_name, m.LEFT)
            yr = code(year, 22, MUTED).move_to([0, y, 0]).set_x(x_year + 0.6, m.RIGHT)
            sit(yr, baseline(words))
            items.append(m.VGroup(yr, dot, words))
        before = [items[i] for i in (0, 1, 3)]
        diffusion_rows = [items[i] for i in (2, 4, 5)]
        crit = m.VGroup(*[pill(w, c, FILLS[c], 22) for w, c in (("sharp", GREEN), ("varied", GREEN),
                                                                 ("stable training", GREEN))]).arrange(m.RIGHT, buff=0.3)
        crit.move_to([2.95, -2.75, 0])
        crit_lab = cite("the scorecard:").next_to(crit, m.LEFT, buff=0.3)
        sit(crit_lab, baseline(crit[0][1]))

        self.say(
            "[In episode two, the average of all the training sevens] [was a blur that nobody ever wrote].",
            m.AnimationGroup(self.new_header("00 · hook", "Before diffusion"), m.FadeIn(blur, shift=0.2 * m.UP)),
            m.AnimationGroup(m.FadeIn(cap, shift=0.1 * m.UP), m.FadeIn(ep2)),
        )
        self.say(
            "[Before diffusion, other generators fought that blur], and each one paid for its fix somewhere else.",
            m.AnimationGroup(m.Create(spine), m.LaggedStart(*[m.FadeIn(r, shift=0.15 * m.LEFT) for r in before],
                                                            lag_ratio=0.35)),
        )
        self.say(
            "[Today we meet them in order], [score them on sharpness, variety and stable training], "
            "[and watch diffusion take over].",
            m.Circumscribe(m.VGroup(*before), color=MUTED, buff=0.12),
            m.AnimationGroup(m.FadeIn(crit_lab), m.LaggedStart(*[m.FadeIn(c, shift=0.1 * m.UP) for c in crit],
                                                                lag_ratio=0.3)),
            m.LaggedStart(*[m.FadeIn(r, shift=0.15 * m.LEFT) for r in diffusion_rows], lag_ratio=0.35),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 1 · autoencoder

    def autoencoder(self):
        self.next_section("1 · Autoencoder", notes=(
            "Computed, not trained: the linear autoencoder is PCA on the first 10,000 MNIST training digits (pixels −1…1), "
            f"keeping {N_CODE} components (a linear autoencoder trained with squared error learns the same subspace). "
            f"Encoder = projection onto the {N_CODE} directions, decoder = mean + code · directions. The digit is training "
            "image 15 (the first seven). The cells show each number's size in units of its spread over the 10,000 digits. "
            "The random code: uniform in ±3 spreads per number (default_rng(7)); its decode is far from every digit of "
            "the 10,000 (asserted: > 1.4× the copy's distance). Notes §3: an autoencoder compresses but does not generate."))
        y = 0.45
        src = digit(X[A_IDX], DATA, 1.75).move_to([-5.6, y, 0])
        enc = funnel((-3.15, y), True, length=1.2, label="encoder")
        grid, cells = code_cells(CODE / CODE_STD, INK)
        grid.move_to([0, y, 0])
        dec = funnel((3.15, y), False, length=1.2, label="decoder")
        out = digit(REC, DATA, 1.75).move_to([5.6, y, 0])
        a1 = arrow(src.get_right(), enc.get_left(), MUTED, buff=0.12)
        a2 = arrow(enc.get_right(), grid.get_left(), MUTED, buff=0.15)
        a3 = arrow(grid.get_right(), dec.get_left(), MUTED, buff=0.15)
        a4 = arrow(dec.get_right(), out.get_left(), MUTED, buff=0.12)
        ly = -1.05
        l_src = text("a real seven", 22, MUTED).move_to([src.get_x(), ly, 0])
        l_code = code(f"{N_CODE} numbers", 22, INK).move_to([0, ly, 0])
        l_out = text("the copy", 22, MUTED).move_to([out.get_x(), ly, 0])
        for w in (l_code, l_out):
            sit(w, baseline(l_src))
        how = cite(f"computed: a linear autoencoder (PCA on 10,000 MNIST digits, {N_CODE} numbers)").move_to([0, -3.3, 0])
        rand_cells = [mix(CARD, NOISE, 0.18 + 0.72 * min(abs(float(v)) / 2.5, 1.0)) for v in RAND_CODE / CODE_STD]
        smudge = digit(SMUDGE, DATA, 1.75).move_to(out)
        l_rand = code(f"{N_CODE} random numbers", 22, NOISE).move_to([0, ly, 0])
        sit(l_rand, baseline(l_src))
        verdict = pill("a smudge, not a digit", RED, RED_FILL, 22).move_to([0, -2.15, 0])

        self.say(
            "[First, the autoencoder.] [An encoder squeezes a digit into a short code]: [here, just thirty-two numbers].",
            m.AnimationGroup(self.new_header("01 · autoencoder", "Squeeze and rebuild"), m.FadeIn(src, shift=0.2 * m.UP),
                             m.FadeIn(l_src)),
            m.AnimationGroup(m.GrowArrow(a1), m.FadeIn(enc, shift=0.1 * m.RIGHT), m.GrowArrow(a2)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(c, scale=0.6) for c in cells], lag_ratio=0.04), m.FadeIn(l_code)),
        )
        self.say(
            "[A decoder rebuilds the digit from that code], [and both are trained to make the copy match the original].",
            m.AnimationGroup(m.GrowArrow(a3), m.FadeIn(dec, shift=0.1 * m.RIGHT), m.GrowArrow(a4),
                             m.FadeIn(out, shift=0.1 * m.RIGHT), m.FadeIn(l_out)),
            m.AnimationGroup(m.Circumscribe(src[0], color=DATA, buff=0.06), m.Circumscribe(out[0], color=DATA, buff=0.06),
                             m.FadeIn(how)),
        )
        self.say(
            "[But it can't invent new digits]: [feed the decoder thirty-two random numbers], [and out comes a smudge].",
            m.AnimationGroup(m.FadeOut(m.Group(src, a1, enc, a2, l_src), shift=0.1 * m.LEFT), m.FadeOut(l_code)),
            m.AnimationGroup(*[c.animate.set_fill(col, 1) for c, col in zip(cells, rand_cells)], m.FadeIn(l_rand)),
            m.AnimationGroup(m.FadeOut(m.Group(out, l_out)), m.FadeIn(smudge), m.FadeIn(verdict, shift=0.1 * m.UP)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 2 · VAE

    def vae(self):
        self.next_section("2 · Why VAEs blur", notes=(
            "VAE: Kingma & Welling 2013. The KL term pulls each digit's code distribution toward N(0, I), so random codes "
            "from that cloud decode to digit-like pictures (the code-space drawing is a sketch). With a Gaussian pixel "
            "likelihood the decoder pays squared error, so when a code could stand for several digits its best answer is "
            "their average: blur (notes §3, worked example). The 'decoded' picture in the diagram is an illustration: it is "
            f"the average of the 64 training sevens nearest to seven 15, shown again in the strip. Pair: training sevens "
            f"{A_IDX} and {B_IDX}; {N_SPLIT} of 784 pixels have ink (> 0) in exactly one, and their average is grey "
            f"(mean {PAIR[SPLIT].mean():+.2f} on those pixels); the average's squared error, whichever seven was meant, is "
            f"{_err(PAIR):.0f} vs {_err(X[A_IDX]):.0f} for betting on either one. Strip: averages of the k = "
            + ", ".join(map(str, KS)) + " nearest training sevens (Euclidean, −1…1); half-tone pixels (|v| < 0.5): "
            + ", ".join(f"{100 * GREY[k]:.1f}%" for k in KS) + "."))
        # the diagram
        thumbs = m.Group(*[digit(X[i], DATA, 0.82) for i in (A_IDX, int(SEVENS[3]), int(np.flatnonzero(LABELS == 3)[0]))])
        thumbs.arrange(m.DOWN, buff=0.22).move_to([-5.75, 0.25, 0])
        enc = funnel((-4.05, 0.25), True, wide=2.2, narrow=0.9, length=1.2, label="encoder")
        cs = np.array([-0.95, 0.25, 0.0])
        R = 1.55
        disc = m.Circle(radius=R).set_stroke(width=0).set_fill(NOISE, 0.1).move_to(cs)
        ring = m.DashedVMobject(m.Circle(radius=R), num_dashes=48).set_stroke(NOISE, 2.5).move_to(cs)
        ring2 = m.DashedVMobject(m.Circle(radius=R / 2), num_dashes=26).set_stroke(NOISE, 2).move_to(cs)
        prior = tex(r"\mathcal{N}(0,\, I)", 34, NOISE).move_to(cs + [0, R + 0.32, 0])
        space_lab = cite("code space (sketch)").move_to(cs + [0, -R - 0.32, 0])
        code_pts = [cs + np.array(p) for p in ((-0.55, 0.5, 0), (0.35, 0.75, 0), (-0.2, -0.6, 0))]
        code_dots = m.VGroup(*[m.Dot(p, radius=0.085, color=DATA) for p in code_pts])
        links = m.VGroup(*[m.Line(enc.get_right() + [0.05, 0, 0], p, buff=0.1).set_stroke(DATA, 2, opacity=0.6)
                           for p in code_pts])
        pick = m.Dot(cs + np.array([0.75, -0.25, 0]), radius=0.11, color=NOISE)
        pick_ring = m.Circle(radius=0.2).set_stroke(NOISE, 3).move_to(pick)
        dec = funnel((2.15, 0.25), False, wide=2.2, narrow=0.9, length=1.2, label="decoder")
        a_pick = arrow(pick.get_center(), dec.get_left(), NOISE, buff=0.22)
        soft = digit(AVG_K[64], DATA, 1.95).move_to([4.6, 0.25, 0])
        a_out = arrow(dec.get_right(), soft.get_left(), MUTED, buff=0.12)
        soft_lab = text("digit-like, but blurry", 22, INK).next_to(soft, m.DOWN, buff=0.22)
        soft_note = cite("illustration").next_to(soft_lab, m.DOWN, buff=0.1)
        km = cite("Kingma & Welling, 2013").move_to([4.9, -3.35, 0])
        diagram = m.Group(thumbs, enc, disc, ring, ring2, prior, space_lab, code_dots, links, pick, pick_ring, dec, a_pick,
                          soft, a_out, soft_lab, soft_note)

        # why: squared error, two sevens, their average
        loss = tex(r"\mathrm{loss} = \sum_{\mathrm{pixels}} (x - \hat{x})^2", 46).move_to([0, 1.65, 0])
        loss_sub = text("squared error, pixel by pixel", 22, MUTED).next_to(loss, m.DOWN, buff=0.25)
        pa = digit(X[A_IDX], DATA, 2.2, label=f"MNIST #{A_IDX}").move_to([-4.55, -0.85, 0])
        pb = digit(X[B_IDX], DATA, 2.2, label=f"MNIST #{B_IDX}").move_to([-1.75, -0.85, 0])
        pa[2].set_y(pb[2].get_y())
        orr = text("or", 26, MUTED).move_to([(pa.get_x() + pb.get_x()) / 2, pa[0].get_y(), 0])
        avg = digit(PAIR, DATA, 2.2, label="their average").move_to([2.15, -0.85, 0])
        to_avg = arrow(pb[0].get_right(), avg[0].get_left(), MUTED, buff=0.2)
        cell = avg[1].height / 28
        tl = avg[1].get_corner(m.UL)
        marks = m.VGroup(*[m.Square(cell * 0.9).set_stroke(RED, 1.6).set_fill(opacity=0)
                           .move_to(tl + np.array([(c + 0.5) * cell, -(r + 0.5) * cell, 0]))
                           for r, c in zip(*np.nonzero(SPLIT))])
        sw = m.VGroup(swatch(1.0), tex("+", 36), swatch(-1.0), tex(r"\to", 36), swatch(float(PAIR[SPLIT].mean())))
        sw.arrange(m.RIGHT, buff=0.22).move_to([5.25, -0.25, 0])
        sw_lab = m.VGroup(text("ink", 22, MUTED).next_to(sw[0], m.DOWN, buff=0.15),
                          text("blank", 22, MUTED).next_to(sw[2], m.DOWN, buff=0.15),
                          text("grey", 22, INK).next_to(sw[4], m.DOWN, buff=0.15))
        for w in sw_lab[1:]:
            sit(w, baseline(sw_lab[0]))
        n_split = code(f"{N_SPLIT} of 784 pixels", 22, RED).move_to([5.25, -1.55, 0])

        strip_imgs = [digit(AVG_K[k], DATA, 1.75, label=(f"{k} seven" if k == 1 else f"{k} sevens")) for k in KS]
        strip_row = m.Group(*strip_imgs).arrange(m.RIGHT, buff=0.42).move_to([0, -0.95, 0])
        for f in strip_imgs[1:]:
            f[2].set_y(strip_imgs[0][2].get_y())
        strip_src = cite(f"averages of the training sevens closest to MNIST #{A_IDX} · computed").move_to([0, -3.2, 0])

        self.say(
            "[In twenty thirteen came the VAE]: [it trains every code to land inside one round cloud].",
            m.AnimationGroup(self.new_header("02 · VAE", "Why VAEs blur"), m.FadeIn(thumbs, shift=0.1 * m.RIGHT),
                             m.FadeIn(enc), m.FadeIn(km)),
            m.AnimationGroup(m.FadeIn(disc), m.Create(ring), m.Create(ring2), m.FadeIn(prior), m.FadeIn(space_lab),
                             m.LaggedStart(*[m.AnimationGroup(m.Create(l), m.FadeIn(d, scale=0.5))
                                             for l, d in zip(links, code_dots)], lag_ratio=0.3)),
        )
        self.say(
            "[So any random point from the cloud] [decodes into something digit-like], [but blurry].",
            m.AnimationGroup(m.FadeIn(pick, scale=0.4), m.Create(pick_ring)),
            m.AnimationGroup(m.GrowArrow(a_pick), m.FadeIn(dec), m.GrowArrow(a_out), m.FadeIn(soft, shift=0.1 * m.RIGHT)),
            m.AnimationGroup(m.FadeIn(soft_lab, shift=0.1 * m.UP), m.FadeIn(soft_note)),
        )
        self.say(
            "[Here's why.] [The decoder is scored by squared error, pixel by pixel].",
            m.AnimationGroup(m.FadeOut(diagram), m.FadeOut(km)),
            m.AnimationGroup(m.Write(loss), m.FadeIn(loss_sub, shift=0.1 * m.UP)),
        )
        self.say(
            "[Suppose one code could mean either of these two real sevens].",
            m.AnimationGroup(m.FadeIn(pa, shift=0.15 * m.UP), m.FadeIn(orr), m.FadeIn(pb, shift=0.15 * m.UP)),
        )
        self.say(
            "[When it can't tell which], [the lowest-error answer is neither one]: [it's their average].",
            m.AnimationGroup(m.Circumscribe(pa[0], color=MUTED, buff=0.06), m.Circumscribe(pb[0], color=MUTED, buff=0.06)),
            m.GrowArrow(to_avg),
            m.FadeIn(avg, target_position=m.Group(pa, pb)),
        )
        self.say(
            "[Where one seven has ink and the other has none], [the average is grey]: [a stroke that neither one has].",
            m.AnimationGroup(m.FadeIn(sw[:3]), m.FadeIn(sw_lab[:2])),
            m.AnimationGroup(m.FadeIn(sw[3:]), m.FadeIn(sw_lab[2])),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(q) for q in marks], lag_ratio=0.01, run_time=1.2),
                             m.FadeIn(n_split, shift=0.1 * m.UP)),
        )
        self.say(
            "[Average more look-alike sevens], [and the strokes melt into a blur]: [the soft look of VAE samples].",
            m.AnimationGroup(m.FadeOut(m.Group(pa, pb, orr, avg, to_avg, marks, sw, sw_lab, n_split)),
                             m.LaggedStart(*[m.FadeIn(f, shift=0.1 * m.UP) for f in strip_imgs[:2]], lag_ratio=0.4)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(f, shift=0.1 * m.UP) for f in strip_imgs[2:]], lag_ratio=0.4),
                             m.FadeIn(strip_src)),
            m.Circumscribe(strip_imgs[3][0], color=DATA, buff=0.08),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 3 · GANs

    def gans(self):
        self.next_section("3 · GANs and mode collapse", notes=(
            "GANs: Goodfellow et al. 2014 (a generator G maps noise to samples; a discriminator, here 'the critic', tells "
            "real from fake; a minimax game). On the swirl the generator's points are an ILLUSTRATION (no GAN was trained): "
            f"{N_FAKE} points as a blob (0.28 · gaussian_noise(seed=21)), half-way, then all on arm 1 (swirl_labelled(560, "
            "seed=31), arm 0 only), then rotated half a turn onto arm 2. The critic's map is COMPUTED for each of those "
            "states: the best possible critic for fixed fakes, D = p_data / (p_data + p_fake) (Goodfellow et al. 2014, "
            f"Prop. 1), with both densities smoothed by {BW} (the swirl as a mixture of std √(0.03² + {BW}²), the fakes as "
            "a kernel density); GREEN = D > ½, RED = D < ½, transparent where neither density reaches. On arm 1, the "
            f"collapsed points: {100 * ON_SWIRL:.0f}% within 0.1 of the curve (off_swirl), coverage {COVER:.0%} of the 40 "
            "pieces (swirl_coverage; real data 100%). Mode collapse in the paper's words: 'the Helvetica scenario' (§6)."))
        plot = Plot((-1.2, 1.2), (-1.2, 1.2), width=4.7, height=4.7, center=(-0.25, -0.5, 0), axes=False)
        side = plot.coords(1.2, 0)[0] - plot.coords(-1.2, 0)[0]

        def map_img(i):
            """A fresh picture of critic map i (each appearance its own object, so fades never collide)."""
            image = m.ImageMobject(MAPS[i][0])
            image.height = side
            return image.move_to(plot.coords(0, 0)).set_z_index(-2)

        maps = [map_img(0), map_img(1), map_img(2)]
        chase = [map_img(3), map_img(2), map_img(3)]
        real = plot.cloud(REAL_XY, DATA, radius=0.024, opacity=0.6)
        fakes = plot.cloud(BLOB, GAN_C, radius=0.036)
        fakes.set_z_index(2)

        z_img = framed(picture(tint(noise_image(10, 10, seed=4), NOISE, -2, 2), 1.0)).move_to([-5.35, 1.35, 0])
        z_lab = text("noise", 20, NOISE).next_to(z_img, m.UP, buff=0.12)
        gen_box = card(2.15, 1.0, GAN_C, AMBER_FILL)
        gen = m.VGroup(gen_box, text("generator", 24, INK, weight=m.BOLD).move_to(gen_box)).move_to([-5.35, -0.5, 0])
        g1 = arrow(z_img.get_bottom(), gen.get_top(), NOISE, buff=0.1)
        g2 = arrow(gen.get_right(), plot.coords(-1.2, 0) + [-0.05, 0, 0], GAN_C, buff=0.12)
        fake_lab = text("fakes", 22, GAN_C).next_to(gen, m.DOWN, buff=0.25)
        real_lab = text("real data", 22, DATA).next_to(fake_lab, m.DOWN, buff=0.12)

        crit_box = card(2.5, 1.15, INK, CARD_HI)
        crit = m.VGroup(crit_box, text("critic", 26, INK, weight=m.BOLD), text("real or fake?", 20, MUTED))
        m.VGroup(crit[1], crit[2]).arrange(m.DOWN, buff=0.08).move_to(crit_box)
        crit.move_to([4.75, 1.3, 0])
        legend = m.VGroup(*[m.VGroup(m.Square(0.26).set_stroke(width=0).set_fill(c, 0.8), text(w, 22, INK))
                            .arrange(m.RIGHT, buff=0.2) for c, w in ((GREEN, "looks real"), (RED, "looks fake"))])
        legend.arrange(m.DOWN, buff=0.18, aligned_edge=m.LEFT).move_to([4.75, 0.05, 0])
        computed = cite("the best critic for these points,\ncomputed").move_to([4.75, -0.85, 0])
        gw = cite("Goodfellow et al., 2014").move_to([-5.35, -3.35, 0])
        illus = cite("illustration: where the generator's points land").move_to([plot.get_x(), -3.35, 0])

        on_sw = code(f"on the swirl: {100 * ON_SWIRL:.0f}%", 22, GREEN).move_to([4.75, -1.55, 0])
        covered = code(f"swirl covered: {COVER:.0%}", 22, RED).move_to([4.75, -2.05, 0])
        collapse = pill("mode collapse", RED, RED_FILL, 22).move_to([4.75, -2.7, 0])
        empty = m.DashedVMobject(m.VMobject().set_points_as_corners([plot.coords(x, y) for x, y in swirl_curve(160)[1]]),
                                 num_dashes=70).set_stroke(INK, 3, opacity=0.8)
        helv = cite("“the Helvetica scenario”").next_to(collapse, m.DOWN, buff=0.12)
        verdicts = m.VGroup(pill("sharp", GREEN, GREEN_FILL, 22), pill("unstable", RED, RED_FILL, 22),
                            pill("mode collapse", RED, RED_FILL, 22)).arrange(m.DOWN, buff=0.18).move_to([4.75, -1.75, 0])

        def swing(a0, a1):
            def hook(a):
                fakes._put_xy(rot(FAKE_ARM, a0 + (a1 - a0) * a))
            return drive(fakes, hook, run_time=1.5)

        def swap(old, new, run_time=0.8):
            return m.AnimationGroup(m.FadeOut(old), m.FadeIn(new), run_time=run_time)

        self.say(
            "[In twenty fourteen came GANs]: [two networks that train against each other].",
            m.AnimationGroup(self.new_header("03 · GANs", "Generator versus critic"), m.FadeIn(gw)),
            m.AnimationGroup(m.FadeIn(gen, shift=0.15 * m.RIGHT), m.FadeIn(crit, shift=0.15 * m.LEFT)),
        )
        self.say(
            "[The generator turns random noise into fake points]; [the critic learns to tell fakes from real data].",
            m.AnimationGroup(m.FadeIn(z_img), m.FadeIn(z_lab), m.GrowArrow(g1), m.GrowArrow(g2),
                             m.LaggedStart(*[m.FadeIn(d, scale=0.4) for d in fakes], lag_ratio=0.004, run_time=1.2),
                             m.FadeIn(fake_lab)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(d, scale=0.4) for d in real], lag_ratio=0.002, run_time=1.2),
                             m.FadeIn(real_lab), m.Circumscribe(crit, color=INK, buff=0.08)),
        )
        self.say(
            "[On the swirl, the critic paints the plane]: [green where points look real], [red where they look fake].",
            m.AnimationGroup(m.FadeIn(maps[0]), m.FadeIn(computed)),
            m.FadeIn(legend[0], shift=0.1 * m.UP),
            m.FadeIn(legend[1], shift=0.1 * m.UP),
        )
        self.say(
            "[The generator moves its points toward the green], [then the critic paints again]: [a tug of war].",
            fakes.to(HALF, run_time=1.6),
            swap(maps[0], maps[1]),
            m.AnimationGroup(m.Indicate(gen, color=GAN_C, scale_factor=1.08), m.Indicate(crit, color=INK, scale_factor=1.08)),
        )
        self.say(
            "[Nothing gets averaged, so the results are sharp]: [every point lands right on the swirl].",
            fakes.to(FAKE_ARM, run_time=1.6),
            m.FadeIn(on_sw, shift=0.1 * m.UP),
        )
        self.say(
            "[But they all sit on one arm.] [That's mode collapse]: [half of the data is never made].",
            m.AnimationGroup(m.FadeOut(maps[1]), m.Create(empty, run_time=1.4), m.FadeIn(illus)),
            m.AnimationGroup(m.FadeIn(collapse, shift=0.1 * m.UP), m.FadeIn(helv)),
            m.FadeIn(covered, shift=0.1 * m.UP),
        )
        self.say(
            "[When the critic catches on], [the generator swings to the other arm], [and back again], [round and round].",
            m.AnimationGroup(m.FadeIn(maps[2]), m.FadeOut(empty)),
            m.AnimationGroup(swing(0, PI_), swap(maps[2], chase[0], 1.0)),
            m.AnimationGroup(swing(PI_, 2 * PI_), swap(chase[0], chase[1], 1.0)),
            m.AnimationGroup(swing(2 * PI_, 3 * PI_), swap(chase[1], chase[2], 1.0)),
        )
        self.say(
            "[Training is a balancing act]: [sharp results], [but runs that can wobble, collapse, or fall apart].",
            m.AnimationGroup(m.FadeOut(m.Group(on_sw, covered, collapse, helv, computed)),
                             m.Indicate(gen, color=GAN_C, scale_factor=1.08), m.Indicate(crit, color=INK, scale_factor=1.08)),
            m.FadeIn(verdicts[0], shift=0.1 * m.UP),
            m.LaggedStart(*[m.FadeIn(v, shift=0.1 * m.UP) for v in verdicts[1:]], lag_ratio=0.5),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 4 · pixel by pixel

    def pixel_by_pixel(self):
        self.next_section("4 · One pixel at a time", notes=(
            "Autoregressive pixel models: PixelRNN and PixelCNN (van den Oord et al. 2016): p(x) = ∏ p(x_i | x_1 … x_(i−1)); "
            "each value is sampled from a predicted distribution (no averaging), trained by maximum likelihood (stable). "
            "Sampling needs one network call per value, in order: 28·28·1 = 784 for an MNIST digit, 32·32·3 = 3,072 for a "
            "CIFAR-10 image (notes §3), 256·256·3 = 196,608 at 256 × 256 (computed). The reveal is a real MNIST training "
            f"digit (#{TWO_IDX}, a two) uncovered in raster order: an illustration of the order, not a model's sample."))
        H_IMG = 3.0
        live = Live(self.reveal_rgb(0), H_IMG)
        pic = framed(live.image).move_to([-3.4, 0.05, 0])
        note = cite("a real MNIST digit, uncovered in drawing order (illustration)").next_to(pic, m.DOWN, buff=0.22)
        cell = live.image.height / 28
        tl = live.image.get_corner(m.UL)
        cursor = m.Square(cell).set_stroke(MODEL, 3).set_fill(opacity=0).move_to(tl + np.array([cell / 2, -cell / 2, 0]))
        cursor.set_z_index(3)
        calls_lab = text("network calls", 24, MUTED).move_to([0, 1.95, 0]).set_x(0.75, m.LEFT)
        counter = Counter(0.75, 1.3, 0, 60)
        pills = m.VGroup(pill("samples each pixel: sharp", GREEN, GREEN_FILL, 22),
                         pill("plain likelihood: stable", GREEN, GREEN_FILL, 22)).arrange(m.DOWN, buff=0.2,
                                                                                          aligned_edge=m.LEFT)
        pills.move_to([0, 0.05, 0]).set_x(0.75, m.LEFT)
        rows = m.VGroup()
        for (h, w, c), n, lab in zip(CALLS, N_CALLS, ("a digit", "small colour", "large colour")):
            r = m.VGroup(code(f"{h} × {w} × {c}", 20, INK), code(f"{n:,}", 26, TIME, weight=m.BOLD), text(lab, 20, MUTED))
            rows.add(r)
        for i, r in enumerate(rows):
            y = -1.35 - 0.62 * i
            r[0].move_to([0, y, 0]).set_x(0.75, m.LEFT)
            r[1].move_to([0, y, 0]).set_x(4.8, m.RIGHT)
            r[2].move_to([0, y, 0]).set_x(5.05, m.LEFT)
            sit(r[1], baseline(r[0]))
            sit(r[2], baseline(r[0]))
        src = cite("van den Oord et al., 2016 · PixelRNN, PixelCNN").move_to([3.6, -3.35, 0])
        state = {"n": 0}

        def upto(n0, n1):
            def hook(a):
                n = int(round(n0 + (n1 - n0) * a))
                if n != state["n"]:
                    state["n"] = n
                    live.show(self.reveal_rgb(n))
                    k = min(n, 783)
                    cursor.move_to(tl + np.array([(k % 28 + 0.5) * cell, -(k // 28 + 0.5) * cell, 0]))
                    counter.set_value(n)
            return hook

        self.say(
            "[A third family, from twenty sixteen, paints one pixel at a time], [each chosen given all the pixels before it].",
            m.AnimationGroup(self.new_header("04 · pixel by pixel", "One pixel at a time"), m.FadeIn(pic),
                             m.FadeIn(note), m.FadeIn(cursor), m.FadeIn(calls_lab), m.FadeIn(counter), m.FadeIn(src)),
            drive(live.image, upto(0, 300), run_time=3.0, rate=m.linear),
        )
        self.say(
            "[It samples instead of averaging, so it stays sharp], [and its training is stable].",
            m.FadeIn(pills[0], shift=0.1 * m.UP),
            m.FadeIn(pills[1], shift=0.1 * m.UP),
        )
        self.say(
            "[But each pixel costs one network call]: [seven hundred eighty-four for one small digit].",
            m.Circumscribe(counter, color=TIME, buff=0.1),
            m.AnimationGroup(drive(live.image, upto(300, 784), run_time=2.0, rate=m.linear),
                             m.FadeIn(rows[0], shift=0.1 * m.UP)),
        )
        self.say(
            "[A small colour image needs three thousand and seventy-two], [a larger one almost two hundred thousand], "
            "[one after another].",
            m.FadeIn(rows[1], shift=0.1 * m.UP),
            m.FadeIn(rows[2], shift=0.1 * m.UP),
            m.Circumscribe(rows[2][1], color=TIME, buff=0.1),
        )
        self.wait(0.4)
        self.clear()

    @staticmethod
    def reveal_rgb(n):
        rgb = tint(X[TWO_IDX], DATA, -1, 1).reshape(-1, 3).copy()
        unseen = np.array([int(UNSEEN[i:i + 2], 16) for i in (1, 3, 5)], np.uint8)
        rgb[n:] = unseen
        return rgb.reshape(28, 28, 3)

    # ------------------------------------------------------------------ 5 · diffusion

    def diffusion(self):
        self.next_section("5 · Diffusion", notes=(
            "Diffusion: Sohl-Dickstein et al. 2015; made competitive by DDPM (Ho et al. 2020): one network learns to remove "
            "a little noise, by plain regression (stable, no adversary; notes §3). The loop diagram is kit.denoise_loop "
            "(pictures illustrative). The 7: generated on this Mac by the digit model (U-Net, 3,940,801 parameters, MLX; "
            "logs/train_ddpm_config.json), DDPM 1,000 steps, guidance w = 1, from x_T = default_rng(7) "
            "(arrays/digit_trajectory_7.npz): left x_t (violet → teal as t falls, clipped to −1…1), right the model's "
            "clean-image guess x̂0 at each step: blurry early, sharp late. Swirl: the swirl MLP (45,954 parameters, trained on "
            "this Mac, toy.md), DDPM 1,000 steps from gaussian_noise(2000, seed=5), sampler seed 1 (equal to toy.json's "
            f"run: dist {_run['dist']}, off {_run['off']}, coverage {_run['coverage']}); the first 1,000 points shown, every "
            f"8th step; on the swirl {100 * SW_ON:.1f}%, coverage {SW_COVER:.0%}. The view zooms in as the cloud contracts. "
            "Sevens: samples 7, 17, …, 57 of digit_samples_ddpm_ddpm1000_w1.npz (label = index mod 10)."))
        loop = denoise_loop(center=(0, 0.75, 0), width=9.0, size=1.0)
        loop_src = cite("Sohl-Dickstein et al., 2015 · Ho et al., 2020").move_to([0, -3.35, 0])
        good = m.VGroup(*[pill(w, GREEN, GREEN_FILL, 22) for w in ("plain regression", "no critic to chase",
                                                                    "stable training")]).arrange(m.RIGHT, buff=0.35)
        good.move_to([0, -2.2, 0])

        # the digit model's 7, step by step
        frac = lambda i: i / (len(TRAJ_T) - 1)  # noqa: E731
        xt_live = Live(tint(TRAJ_X[0], NOISE, -1, 1), 2.7)
        x0_live = Live(tint(TRAJ_X0[0], DATA, -1, 1), 2.7)
        xt_pic = framed(xt_live.image).move_to([-3.7, 0.4, 0])
        x0_pic = framed(x0_live.image, MODEL).move_to([0.0, 0.4, 0])
        xt_lab = text("the sample", 22, INK).next_to(xt_pic, m.DOWN, buff=0.2)
        x0_lab = text("its guess of the clean digit", 22, MODEL).next_to(x0_pic, m.DOWN, buff=0.2)
        sit(x0_lab, baseline(xt_lab))
        step_lab = text("step", 24, MUTED).move_to([0, 1.5, 0]).set_x(2.75, m.LEFT)
        t_counter = Counter(2.75, 0.9, 999, 48, prefix="t = ", fmt="{}")
        mac = m.VGroup(text("generated on this Mac", 22, INK),
                       code("the digit model · 3.9M parameters · DDPM, 1,000 steps", 20, MUTED))
        mac.arrange(m.DOWN, buff=0.14).move_to([-1.85, -2.45, 0])
        state = {"i": -1}

        def walk_traj(i0, i1):
            def hook(a):
                i = int(round(i0 + (i1 - i0) * a))
                if i != state["i"]:
                    state["i"] = i
                    xt_live.show(tint(TRAJ_X[i], mix(NOISE, DATA, frac(i)), -1, 1))
                    x0_live.show(tint(TRAJ_X0[i], DATA, -1, 1))
                    t_counter.set_value(TRAJ_T[i])
            return hook

        # the swirl, from noise to both arms (the trained MLP's DDPM run)
        sc = np.array([-3.05, -0.45, 0.0])
        R_VIEW = 2.25
        ext = np.array([np.abs(p).max() for p in SW_PATH])
        scales = np.minimum(2.2, R_VIEW / np.minimum.accumulate(ext))       # zoom in as the cloud contracts
        dots = m.VGroup(*[m.Dot(radius=0.024, color=NOISE) for _ in range(len(SW_PATH[0]))])
        ring = m.Circle(radius=scales[0]).set_stroke(FAINT, 2.5).move_to(sc)

        def put(i, a=None):
            s = scales[i]
            for d, (x, y) in zip(dots, SW_PATH[i]):
                d.move_to(sc + s * np.array([x, y, 0.0]))
            ring.become(m.Circle(radius=s).set_stroke(FAINT, 2.5).move_to(sc))
            dots.set_color(mix(NOISE, DATA, i / (len(SW_PATH) - 1)))

        put(0)
        rx = 0.25
        sw_title = m.VGroup(text("trained on this Mac", 22, INK),
                            code(f"the swirl MLP · {MLP_PARAMS:,} parameters", 20, MUTED),
                            code("DDPM · 1,000 steps · 1,000 points", 20, MUTED)).arrange(m.DOWN, buff=0.12,
                                                                                          aligned_edge=m.LEFT)
        sw_title.move_to([0, 1.75, 0]).set_x(rx, m.LEFT)
        ring_note = cite("faint ring: radius 1 (the view zooms in)").move_to([sc[0], -3.35, 0])
        sw_on = code(f"on the swirl: {100 * SW_ON:.1f}%", 22, GREEN).move_to([0, 0.55, 0]).set_x(rx, m.LEFT)
        sw_cov = code(f"swirl covered: {SW_COVER:.0%}", 22, GREEN).next_to(sw_on, m.DOWN, buff=0.16).set_x(rx, m.LEFT)
        sevens = m.Group(*[digit(g / 127.5 - 1, DATA, 0.95) for g in GEN7]).arrange_in_grid(rows=2, cols=3, buff=0.2)
        sevens.move_to([0, -1.55, 0]).set_x(rx, m.LEFT)
        sev_lab = cite("generated on this Mac:\nthe digit model · DDPM, 1,000 steps").next_to(sevens, m.DOWN, buff=0.16).set_x(rx, m.LEFT)

        self.say(
            "[Diffusion, first proposed in twenty fifteen, takes another road]: [remove a little noise], [and repeat].",
            m.AnimationGroup(self.new_header("05 · diffusion", "A little noise at a time"),
                             m.FadeIn(loop.left), m.FadeIn(loop.model), m.FadeIn(loop.right), m.FadeIn(loop_src)),
            m.AnimationGroup(m.GrowArrow(loop.go), m.GrowArrow(loop.out), pulse(loop.go, NOISE), pulse(loop.out, DATA)),
            m.Create(loop.back),
        )
        self.say(
            "[Its training is plain regression], [with no critic to chase], [so it is as stable as the VAE].",
            m.FadeIn(good[0], shift=0.1 * m.UP),
            m.FadeIn(good[1], shift=0.1 * m.UP),
            m.FadeIn(good[2], shift=0.1 * m.UP),
        )
        self.say(
            "[Early on, its guess of the clean digit is a blurry average too], [but each step moves only a little], "
            "[and the seven sharpens].",
            m.AnimationGroup(m.FadeOut(m.Group(loop, good)), m.FadeIn(m.Group(xt_pic, x0_pic, xt_lab, x0_lab)),
                             m.FadeIn(step_lab), m.FadeIn(t_counter), m.FadeIn(mac),
                             m.Succession(m.Wait(0.6), drive(x0_live.image, walk_traj(0, 120), run_time=2.2))),
            drive(xt_live.image, walk_traj(120, 800), run_time=2.4),
            drive(xt_live.image, walk_traj(800, 999), run_time=1.6),
        )
        self.say(
            "[On the swirl, a small network trained on this Mac] [turns a thousand noise points into both arms].",
            m.AnimationGroup(m.FadeOut(m.Group(xt_pic, x0_pic, xt_lab, x0_lab, step_lab, t_counter, mac, loop_src)),
                             m.FadeIn(ring), m.FadeIn(dots), m.FadeIn(sw_title), m.FadeIn(ring_note)),
            drive(dots, lambda a: put(int(round(a * (len(SW_PATH) - 1)))), run_time=3.4, rate=m.linear),
        )
        self.say(
            "[Every piece of the swirl gets covered], [and the digit model makes sevens that are all different, and all sharp].",
            m.AnimationGroup(m.FadeIn(sw_on, shift=0.1 * m.UP), m.FadeIn(sw_cov, shift=0.1 * m.UP)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(s, shift=0.1 * m.UP) for s in sevens], lag_ratio=0.15),
                             m.FadeIn(sev_lab)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 6 · scorecard

    def scorecard(self):
        self.next_section("6 · Scorecard", notes=(
            "Summary of Sections 1–5 (notes §3 mechanisms). Network calls per sample: one decoder or generator pass for "
            "VAEs and GANs; one per value for pixel-by-pixel models (784 for a digit, growing with the picture); 1,000 for "
            "DDPM's original sampler (Ho et al. 2020), cut by faster samplers and distillation in Episodes 9–12 "
            "(e.g. 1–4 steps for distilled models, notes §12)."))
        box = card(12.6, 4.75, EDGE, CARD, radius=0.18, stroke=2).move_to([0, -0.45, 0])
        cols = [-0.85, 1.35, 3.45, 5.3]
        top = box.get_top()[1] - 0.48
        heads = m.VGroup(*[code(h, 22, MUTED) for h in ("sharp", "varied", "stable", "calls")])
        for h, x in zip(heads, cols):
            h.move_to([x, top, 0])
        for h in heads[1:]:
            sit(h, baseline(heads[0]))
        rule = m.Line([box.get_left()[0] + 0.35, top - 0.35, 0], [box.get_right()[0] - 0.35, top - 0.35, 0])
        rule.set_stroke(EDGE, 2)
        names = [("the VAE", "2013", INK), ("GANs", "2014", INK), ("pixel by pixel", "2016", INK),
                 ("diffusion", "2015 · 2020", DIFF)]
        marks = [(0, 1, 1, "1"), (1, 0, 0, "1"), (1, 1, 1, "784+"), (1, 1, 1, "1,000")]
        x_name = box.get_left()[0] + 0.45
        rows = []
        for i, ((name, year, color), (sh, va, st, calls)) in enumerate(zip(names, marks)):
            y = top - 0.95 - 0.88 * i
            nm = text(name, 28, color, weight=m.BOLD).move_to([0, y, 0]).set_x(x_name, m.LEFT)
            yr = code(year, 20, MUTED).next_to(nm, m.RIGHT, buff=0.25)
            sit(yr, baseline(nm))
            ms = [(tick_mark() if ok else cross_mark()).move_to([x, y + 0.08, 0]) for ok, x in zip((sh, va, st), cols)]
            c = code(calls, 26, GREEN if calls == "1" else RED, weight=m.BOLD).move_to([cols[3], y, 0])
            sit(c, baseline(nm))
            rows.append(m.VGroup(nm, yr, *ms, c))
        later = code("→ a few steps · Episodes 9–12", 20, TIME).move_to([0, box.get_bottom()[1] - 0.35, 0])
        later.set_x(box.get_right()[0] - 0.1, m.RIGHT)
        ddpm = cite("DDPM, 2020").next_to(later, m.LEFT, buff=0.6)
        sit(ddpm, baseline(later))

        self.say(
            "[Side by side]: [the VAE blurs], [GANs collapse and wobble], [and pixel-by-pixel models crawl].",
            m.AnimationGroup(self.new_header("06 · scorecard", "What each one gave up"), m.FadeIn(box), m.FadeIn(heads),
                             m.Create(rule)),
            m.FadeIn(rows[0], shift=0.1 * m.UP),
            m.FadeIn(rows[1], shift=0.1 * m.UP),
            m.FadeIn(rows[2], shift=0.1 * m.UP),
        )
        self.say(
            "[Diffusion ticks all three boxes], [but pays in network calls]: [a thousand in the original recipe], "
            "[far fewer by episode twelve].",
            m.FadeIn(rows[3], shift=0.1 * m.UP),
            m.Circumscribe(rows[3][-1], color=RED, buff=0.1),
            m.FadeIn(ddpm),
            m.FadeIn(later, shift=0.1 * m.LEFT),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 7 · 2021

    def moment(self):
        self.next_section("7 · Diffusion beats GANs", notes=(
            "Dhariwal & Nichol, 'Diffusion Models Beat GANs on Image Synthesis', arXiv:2105.05233 (May 2021), Table 5, "
            "ImageNet 256×256 (class-conditional): ADM-G (classifier guidance, 250 steps) FID 4.59, precision 0.82, recall "
            "0.52; BigGAN-deep FID 6.95, precision 0.87, recall 0.28 (checked against the paper 2026-10-26). Table 10: "
            "ADM-G 916 generator + 46 classifier = 962 V100-days; BigGAN-deep 128–256 V100-days (converted from TPU "
            "estimates, 2 TPU-v3 days = 1 V100 day). FID: lower is better (Heusel et al. 2017; Episode 19). Recall = the "
            "share of real data the samples reach (like the swirl's coverage); precision = the share of samples that look "
            "real. StyleGAN-XL (Sauer, Schwarz & Geiger, arXiv:2202.00273, Feb 2022), Table 2: ImageNet 256 FID 2.30 — a GAN "
            "led again in early 2022 (Appendix A). Why diffusion won anyway (notes §3): stability, coverage, text "
            "conditioning, scaling."))
        title = text("“Diffusion Models Beat GANs on Image Synthesis”", 30, INK, weight=m.BOLD).move_to([0, 2.15, 0])
        who = cite("Dhariwal & Nichol · May 2021 · ImageNet 256 × 256, Table 5").next_to(title, m.DOWN, buff=0.16)
        x0, scale = -2.35, 0.62
        fid_head = m.VGroup(code("FID", 24, INK, weight=m.BOLD), text("lower is better", 20, MUTED)).arrange(m.RIGHT, buff=0.3)
        fid_head.move_to([0, 0.75, 0]).set_x(x0, m.LEFT)
        sit(fid_head[1], baseline(fid_head[0]))
        entries = [("ADM-G · diffusion", ADM_G["fid"], DIFF), ("BigGAN-deep · GAN", BIGGAN["fid"], GAN_C),
                   ("StyleGAN-XL · GAN, 2022", STYLEGAN_XL, GAN_C)]
        bars = []
        for i, (name, fid, color) in enumerate(entries):
            y = 0.05 - 0.68 * i
            lab = text(name, 22, INK).move_to([0, y, 0]).set_x(x0 - 0.3, m.RIGHT)
            bar = m.Rectangle(width=fid * scale, height=0.42).set_stroke(color, 2).set_fill(FILLS[color], 1)
            bar.move_to([x0 + fid * scale / 2, y, 0])
            val = code(f"{fid:.2f}", 24, color, weight=m.BOLD).next_to(bar, m.RIGHT, buff=0.18)
            bars.append(m.VGroup(lab, bar, val))
        px, rx = 4.35, 5.75
        pr_head = m.VGroup(code("precision", 20, MUTED).move_to([px, 0.75, 0]), code("recall", 20, MUTED).move_to([rx, 0.75, 0]))
        sit(pr_head[1], baseline(pr_head[0]))
        pr_rows = m.VGroup()
        for i, d in enumerate((ADM_G, BIGGAN)):
            y = 0.05 - 0.68 * i
            pr_rows.add(m.VGroup(code(f"{d['p']:.2f}", 24, INK).move_to([px, y, 0]),
                                 code(f"{d['r']:.2f}", 24, GREEN if d is ADM_G else RED, weight=m.BOLD).move_to([rx, y, 0])))
        p_col = m.VGroup(pr_head[0], *[r[0] for r in pr_rows])
        r_col = m.VGroup(pr_head[1], *[r[1] for r in pr_rows])
        r_note = text("recall: how much of the real\ndata the samples reach\n(like the swirl's coverage)", 20, MUTED)
        r_note.move_to([4.95, -1.6, 0])
        cost = m.VGroup(code(f"{ADM_DAYS} V100-days", 26, INK, weight=m.BOLD),
                        text("to train ADM-G · BigGAN-deep: 128–256", 20, MUTED)).arrange(m.RIGHT, buff=0.3)
        cost.move_to([0, -2.6, 0])
        sit(cost[1], baseline(cost[0]))
        cost_src = cite("Table 10").next_to(cost, m.DOWN, buff=0.12)
        sxl_src = cite("Sauer et al., 2022, Table 2 · same test").move_to([0, -1.85, 0]).set_x(x0, m.LEFT)
        why = m.VGroup(*[pill(w, GREEN, GREEN_FILL, 22) for w in ("trains reliably", "covers the data",
                                                                  "follows text prompts", "improves with scale")])
        why.arrange(m.RIGHT, buff=0.3).move_to([0, -2.75, 0])

        self.say(
            "[In May twenty twenty-one, a paper by Dhariwal and Nichol] [put it in its title]: [Diffusion Models Beat GANs].",
            m.AnimationGroup(self.new_header("07 · 2021", "Diffusion beats GANs"), m.FadeIn(who)),
            m.Write(title, run_time=1.4),
            m.Circumscribe(title, color=DIFF, buff=0.12),
        )
        self.say(
            "[On ImageNet at two hundred fifty-six pixels], [their guided model scored an FID of four point five nine], "
            "[against six point nine five for the leading GAN of the day].",
            m.AnimationGroup(m.FadeIn(fid_head), m.FadeIn(bars[0][0]), m.FadeIn(bars[1][0])),
            m.AnimationGroup(m.GrowFromEdge(bars[0][1], m.LEFT), m.FadeIn(bars[0][2], shift=0.1 * m.RIGHT)),
            m.AnimationGroup(m.GrowFromEdge(bars[1][1], m.LEFT), m.FadeIn(bars[1][2], shift=0.1 * m.RIGHT)),
        )
        self.say(
            "[Lower is better.] [The GAN was a little more precise], [but diffusion covered much more of the data].",
            m.Circumscribe(fid_head[1], color=MUTED, buff=0.08),
            m.FadeIn(p_col, shift=0.1 * m.UP),
            m.FadeIn(r_col, shift=0.1 * m.UP),
        )
        self.say(
            "[Recall measures how much of the real data the samples reach], [like our swirl's coverage]: "
            "[zero point five two, against zero point two eight].",
            m.FadeIn(r_note, shift=0.1 * m.UP),
            m.Circumscribe(r_note, color=MUTED, buff=0.08),
            m.AnimationGroup(m.Circumscribe(pr_rows[0][1], color=GREEN, buff=0.1),
                             m.Circumscribe(pr_rows[1][1], color=RED, buff=0.1)),
        )
        self.say(
            "[It was also more expensive to train]: [nine hundred sixty-two GPU days].",
            m.FadeIn(cost[1], shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(cost[0], shift=0.1 * m.UP), m.FadeIn(cost_src)),
        )
        self.say(
            "[And GANs struck back]: [in early twenty twenty-two, StyleGAN-XL scored two point three zero on the same test].",
            m.AnimationGroup(m.FadeOut(m.VGroup(cost, cost_src)), m.FadeIn(bars[2][0])),
            m.AnimationGroup(m.GrowFromEdge(bars[2][1], m.LEFT), m.FadeIn(bars[2][2], shift=0.1 * m.RIGHT),
                             m.FadeIn(sxl_src)),
        )
        self.say(
            "[Diffusion won anyway]: [it trains reliably], [covers the data], [follows text prompts], "
            "[and keeps improving as it grows].",
            m.AnimationGroup(m.FadeOut(sxl_src), m.Circumscribe(bars[0][1], color=DIFF, buff=0.08)),
            m.FadeIn(why[0], shift=0.1 * m.UP),
            m.FadeIn(why[1], shift=0.1 * m.UP),
            m.FadeIn(why[2], shift=0.1 * m.UP),
            m.FadeIn(why[3], shift=0.1 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes=(
            "Blur (the VAE's squared error averages plausible digits), collapse (a GAN's generator can cover one arm and "
            "the chase can go round), crawl (pixel by pixel: one network call per value). Diffusion: sharp, varied and "
            "stable, paid for in steps. Next: Episode 4, Adding Noise (the forward process)."))
        halo = glow(VIOLET, radius=4.0)
        words = m.VGroup(*[title_text(w, 64, c) for w, c in (("Blur", DATA), ("Collapse", GAN_C), ("Crawl", TIME))])
        subs = m.VGroup(*[code(s, 20, MUTED) for s in ("the VAE averages", "GANs chase a critic", "one pixel at a time")])
        words.arrange(m.RIGHT, buff=1.0)
        on_baseline(words)
        for w, s in zip(words, subs):
            s.set_x(w.get_x())
        line_y = min(w.get_bottom()[1] for w in words) - 0.45
        for s in subs:
            s.set_y(line_y)
        mid = baseline(words[0]) + 0.3
        dots = m.VGroup(*[m.Dot([(l.get_right()[0] + r.get_left()[0]) / 2, mid, 0], radius=0.07, color=MUTED)
                          for l, r in zip(words, words[1:])])
        row = m.VGroup(words, dots, subs).move_to(1.0 * m.UP)
        kept = m.VGroup(*[pill(w, GREEN, GREEN_FILL, 22) for w in ("sharp", "varied", "stable")],
                        pill("paid in steps", TIME, BLUE_FILL, 22)).arrange(m.RIGHT, buff=0.3)
        lead = text("diffusion:", 26, DIFF, weight=m.BOLD)
        kept_row = m.VGroup(lead, kept).arrange(m.RIGHT, buff=0.35).next_to(row, m.DOWN, buff=0.75)
        tag = text("To build it, we start with the easy half: destroying a picture on purpose.", 26, MUTED)
        mascot = pixel(0.42)
        nxt = pill("EPISODE 4 · ADDING NOISE", VIOLET, VIOLET_FILL, 20, mono=True)
        foot = m.VGroup(tag, m.VGroup(mascot, nxt).arrange(m.RIGHT, buff=0.3)).arrange(m.DOWN, buff=0.35)
        foot.next_to(kept_row, m.DOWN, buff=0.6)

        self.say(
            "[To sum up]: [blur], [collapse], [crawl].",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.3 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(dots[0]), m.FadeIn(words[1], shift=0.3 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(dots[1]), m.FadeIn(words[2], shift=0.3 * m.UP), m.FadeIn(subs[2])),
        )
        self.say(
            "[Diffusion kept the sharpness, the variety and the stable training], [and paid for them in steps].",
            m.AnimationGroup(m.FadeIn(lead), m.LaggedStart(*[m.FadeIn(k, shift=0.1 * m.UP) for k in kept[:3]],
                                                           lag_ratio=0.3)),
            m.FadeIn(kept[3], shift=0.1 * m.UP),
        )
        self.say(
            "[To build it, we start with the easy half]: [destroying a picture on purpose].",
            m.Write(tag),
            m.AnimationGroup(m.FadeIn(mascot, shift=0.2 * m.UP), m.FadeIn(nxt, shift=0.2 * m.UP)),
        )
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, kept_row, foot)], run_time=0.9)
