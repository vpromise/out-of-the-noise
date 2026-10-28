"""Out of the Noise · Episode 13 · Asking for a Digit (Module 4 · Steering the Picture — the module's first episode)

A condition tells the denoiser what to make. Running example: the digit model (research/experiments/mac_runs.md §3–4),
one starting noise, ten labels: arrays/digit_same_noise_ten_labels.npz holds 8 noises (seeds 0–7, x_T from
np.random.default_rng(100 + seed)) × labels 0–9, the same x_T for all ten labels of a row; we show DDIM 50 at w = 1 (one
plain conditional pass per step, no guidance) and, once, w = 3 (set up for Episode 14). The label enters as a row of
the model's own 11 × 128 label table (read from weights/digit_ddpm_ema.safetensors, label_emb.weight; row 10 = "no
label"), added to the time embedding and fed to all 17 residual blocks (Episode 8 showed the wiring). The run's own
classifier (99.45% on MNIST test) reads 98.1% of generated digits as the requested label at w = 1, 9.9% (chance) with no
label (mac_runs §4, headline guidance_w1_cond_acc / guidance_w0_cond_acc).

Classifier guidance (Dhariwal & Nichol 2021, arXiv:2105.05233; notes §13): ε̂ = ε_θ(x_t) − √(1 − ᾱ_t)·s·∇ log p_φ(y | x_t).
On the swirl it is computed EXACTLY: a perfect classifier's gradient is ∇ log p_t(arm | x) = score(x | arm) − score(x)
(Bayes), checked against finite differences of log arm_probability; with it, classifier guidance at scale s equals
diffusion_reference.guided_eps(s) to 1e-9 (the toy's guidance runs: DDIM 100, z = gaussian_noise(2000, seed=5)), and the
spreads reproduce toy.json (0.6768 at s = 1, 0.4601 at s = 7; coverage 0.45). Swirl fact (toy.json forward.linear):
at t = 500 a perfect classifier names the right arm only 52.8% of the time.

Every number on screen is computed below (diffusion_reference, the run's arrays and weights, measured(...)) and
asserted, or cited with its source. Illustrative: the in-between frames of the ten digits emerging are
√ᾱ·(final digit) + √(1 − ᾱ)·(starting noise) on the DDIM-50 grid — the run saved only the start and the end of these
samples; the same blend matches the run's real DDIM-50 path for the "7" (arrays/digit_trajectory_7.npz) with correlation
≥ 0.99 at every step (asserted). The time-embedding bar in section 2 is a schematic (its values are not shown).

    VOICE=draft manimgx inspect scene.py      # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import json
import struct
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403
from diffusion_reference import *  # noqa: E402,F403

EP_DIR = Path(__file__).resolve().parent  # set AFTER the import: diffusion_reference exports HERE (the series folder)
warnings.filterwarnings("ignore", message=".*encountered in matmul")  # NumPy/Accelerate's spurious warnings

PREFIX = "ep13"
ARR = EXPERIMENTS / "arrays"
H = lambda *keys: measured("mac_runs", "headline", *keys)  # noqa: E731
TOY = lambda *keys: measured("toy", "headline", *keys)  # noqa: E731

# ---------------------------------------------------------------------------- one noise, ten labels (mac_runs §4)

SN = np.load(ARR / "digit_same_noise_ten_labels.npz")
assert list(SN["labels"]) == list(range(10)) and list(SN["seeds"]) == list(range(8))
NOISES = SN["xT"].astype(np.float64)                                              # (8, 28, 28): one x_T per row


def finals(key):
    return SN[key].astype(np.float64) / 127.5 - 1.0                               # uint8 → the model's [−1, 1]


GRID_W1, GRID_W3 = finals("ddpm_ddim50_w1"), finals("ddpm_ddim50_w3")
assert GRID_W1.shape == GRID_W3.shape == (8, 10, 28, 28)
HERO, MISS = 7, 5                                                                 # the cleanest row; a row with misses

T7 = np.load(ARR / "digit_trajectory_7.npz")
TS50 = T7["ddpm_ddim50_w1_t"].astype(int)                                         # the DDIM-50 grid, code t 999 … 0
assert len(TS50) == 50 and TS50[0] == 999 and TS50[-1] == 0


def ab_code(t):
    """ᾱ at the run's 0-based step t (code t = paper t − 1, mac_runs §3)."""
    return float(ALPHA_BAR[int(t) + 1])


def blend(final, noise, t):
    ab = ab_code(t)
    return np.sqrt(ab) * final + np.sqrt(1 - ab) * noise


_xs7, _f7, _n7 = (T7[k].astype(np.float64) for k in ("ddpm_ddim50_w1_x_t", "ddpm_ddim50_w1_final", "xT"))
assert np.abs(_xs7[0] - _n7).max() < 1e-2
BLEND_R = min(float(np.corrcoef(blend(_f7, _n7, t).ravel(), x.ravel())[0, 1]) for t, x in zip(TS50, _xs7))
assert BLEND_R > 0.99                                                             # the emerging frames are faithful

# ---------------------------------------------------------------------------- does it listen? (mac_runs §3–4)

ACC_W1, ACC_W0, CLF_ACC = H("guidance_w1_cond_acc"), H("guidance_w0_cond_acc"), H("classifier_test_accuracy")
assert (f"{100 * ACC_W1:.1f}", f"{100 * ACC_W0:.1f}", f"{100 * CLF_ACC:.2f}") == ("98.1", "9.9", "99.45")
CLF = measured("mac_runs", "classifier")
assert CLF["params"] == 467_818 and round(CLF["wall_s"]) == 14 and CLF["test_errors"] == 55
TRAIN = json.loads((WEIGHTS / "digit_ddpm_ema.json").read_text())
P_UNCOND = TRAIN["p_uncond"]
assert P_UNCOND == 0.1 and TRAIN["params"] == H("digit_unet_params") == 3_940_801  # "one training step in ten"
MISSES = (4, 6)  # by eye: in row 5 at w = 1 the "4" looks like a 9 and the "6" like a 0 (fixed at w = 3)

# ---------------------------------------------------------------------------- the label table (the model's own weights)


def read_tensor(path, name):
    """One float32 tensor from a .safetensors file (an 8-byte header length, a JSON header, raw little-endian data)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n))
        a, b = head[name]["data_offsets"]
        assert head[name]["dtype"] == "F32"
        f.seek(8 + n + a)
        return np.frombuffer(f.read(b - a), np.float32).reshape(head[name]["shape"]).astype(np.float64)


TABLE = read_tensor(WEIGHTS / "digit_ddpm_ema.safetensors", "label_emb.weight")
ARCH = measured("mac_runs", "architecture", "ddpm")
NRES, LEVELS = ARCH["config"]["nres"], len(ARCH["config"]["mults"])
N_BLOCKS = NRES * LEVELS + 2 + (NRES + 1) * LEVELS
assert TABLE.shape == (11, 128) and ARCH["params_by_part"]["label_emb"] == TABLE.size == 1408
assert ARCH["config"]["num_classes"] == 10 and ARCH["config"]["tdim"] == 128 and N_BLOCKS == 17
TAB_LO, TAB_HI = float(TABLE.min()), float(TABLE.max())
assert -0.4 < TAB_LO < 0 < TAB_HI < 0.4

# ---------------------------------------------------------------------------- the noise sets the handwriting


def slant(img):
    """The digit's slant: the image moment μ11/μ02 (the shear used to deskew MNIST)."""
    w = np.clip((img + 1) / 2, 0, 1)
    w = w / w.sum()
    ys, xs = np.mgrid[0:28, 0:28]
    mx, my = (w * xs).sum(), (w * ys).sum()
    return float((w * (xs - mx) * (ys - my)).sum() / (w * (ys - my) ** 2).sum())


def share(a, axis):
    """The share of a's variance explained by its means along `axis` (rows: axis=1, the noise; columns: axis=0)."""
    return float(a.mean(axis).var() / a.var())


SLANT = np.array([[slant(GRID_W1[r, c]) for c in range(10)] for r in range(8)])
SLANT_NOISE, SLANT_LABEL = share(SLANT, 1), share(SLANT, 0)
assert round(SLANT_NOISE, 2) == 0.41 and round(SLANT_LABEL, 2) == 0.11
assert 3.5 < SLANT_NOISE / SLANT_LABEL < 4.5                                      # "about four times"
INK_FRAC = (GRID_W1 > 0).mean(axis=(2, 3))
REL_INK = (INK_FRAC / INK_FRAC.mean(0)).mean(1)                                             # stroke weight vs the label's average
THIN, BOLD = int(REL_INK.argmin()), int(REL_INK.argmax())
assert (THIN, BOLD) == (3, 2)

# ---------------------------------------------------------------------------- classifier guidance on the swirl (exact)


def classifier_grad(x, t, label=0, mix=SWIRL):
    """∇ log p_t(label | x) of a PERFECT classifier, by Bayes: the labelled score minus the plain score."""
    return mixture_score(x, t, mix.given(label)) - mixture_score(x, t, mix)


def classifier_guided_eps(scale, label=0, mix=SWIRL):
    """Dhariwal & Nichol's guided ε: ε̂ = ε(x_t) − √(1 − ᾱ_t)·s·∇ log p(y | x_t), with exact parts."""
    def eps_model(x, t):
        return exact_eps(x, t, mix) - np.sqrt(1 - alpha_bar_at(t)) * scale * classifier_grad(x, t, label, mix)
    return eps_model


_probe, _h = gaussian_noise(6, seed=3) * 0.7, 1e-5
for _t in (100, 500):
    _fd = np.stack([(np.log(arm_probability(_probe + d, _t)) - np.log(arm_probability(_probe - d, _t))) / (2 * _h)
                    for d in (np.array([_h, 0.0]), np.array([0.0, _h]))], axis=1)
    assert np.abs(_fd - classifier_grad(_probe, _t)).max() < 1e-6                # the gradient is the classifier's

Z = gaussian_noise(2000, seed=5)                                                  # the toy's shared noise (toy.md)
RUN_U = ddim_sample(exact_eps, Z, steps=100)
RUN_1 = ddim_sample(classifier_guided_eps(1.0), Z, steps=100)
RUN_7 = ddim_sample(classifier_guided_eps(7.0), Z, steps=100)
assert np.abs(RUN_7.xs[-1] - ddim_sample(guided_eps(7.0, 0), Z, steps=100).xs[-1]).max() < 1e-9


def on_arm0(xy):
    return float(np.mean(arm_probability(xy, 0) > 0.5))


def spread(xy):
    return float(np.sqrt(np.trace(np.cov(np.asarray(xy).T))))


SHARE_U, SHARE_1, SHARE_7 = on_arm0(RUN_U.xs[-1]), on_arm0(RUN_1.xs[-1]), on_arm0(RUN_7.xs[-1])
assert round(100 * SHARE_U) == 49 and SHARE_1 == SHARE_7 == 1.0                   # "about half" → "every one"
SPREAD_1, SPREAD_7 = spread(RUN_1.xs[-1]), spread(RUN_7.xs[-1])
assert round(SPREAD_1, 4) == TOY("swirl_w1_spread") and round(SPREAD_7, 4) == TOY("swirl_w7_spread")
COVER_7 = swirl_coverage(RUN_7.xs[-1], arm=0)
OUTER_1, OUTER_7 = (float(swirl_profile(r.xs[-1])[0, 10:].sum()) for r in (RUN_1, RUN_7))
assert round(100 * OUTER_1) == 52 and round(100 * OUTER_7) == 98                  # toy.md §5: 52 % → 98 %
assert COVER_7 == TOY("swirl_w7_coverage_arm0") == 0.45

# the push along the s = 1 path (ε units): how hard the classifier pulls at each step
PUSH_TS = RUN_1.ts[:-1].astype(float)
PUSH = np.array([float(np.mean(np.sqrt(1 - alpha_bar_at(t)) * np.linalg.norm(classifier_grad(x, t), axis=1)))
                 for x, t in zip(RUN_1.xs[:-1], PUSH_TS)])
LATE_SHARE = float(PUSH[PUSH_TS <= 300].sum() / PUSH.sum())
PEAK_T = int(PUSH_TS[PUSH.argmax()])
assert 0.70 < LATE_SHARE < 0.75 and PEAK_T == 40 and PUSH[PUSH_TS == 1000][0] < 0.01   # "nearly three quarters"

# how often a perfect classifier names the right arm (toy.json forward.linear; recomputed at t = 500)
ACC_TABLE = [(int(r["t"]), float(r["arm_accuracy"])) for r in measured("toy", "forward", "linear", "table")]
ACC_AT = dict(ACC_TABLE)
_x0l, _arm = swirl_labelled(20000, seed=0)
X500 = q_sample(_x0l, 500, gaussian_noise(20000, seed=1))
ACC500 = float(np.mean((arm_probability(X500, 500) > 0.5) == (_arm == 0)))
assert round(ACC500, 3) == TOY("arm_accuracy_t500_linear") == ACC_AT[500] == 0.528
assert (ACC_AT[100], ACC_AT[50], ACC_AT[1000]) == (0.643, 0.78, 0.502)           # "sixty-four … seventy-eight"
LIM_WIDE = 3.0                                                                    # the plane for t = 500: ±3
_inside = np.flatnonzero(np.abs(X500).max(axis=1) < LIM_WIDE - 0.15)[:1000]
assert len(_inside) == 1000 and _inside[-1] < 1050                                # the first points, nearly all
SHOW_X0, SHOW_ARM, SHOW_X500 = _x0l[_inside], _arm[_inside], X500[_inside]

# the field: exact scores at one step
T_FIELD = 20                                                                      # late: the arms are still apart
LIM, FIELD_N, FIELD_MARGIN = 1.5, 15, 0.06
_ax = np.linspace(-LIM + FIELD_MARGIN * 2 * LIM, LIM - FIELD_MARGIN * 2 * LIM, FIELD_N)
GRID = np.array([(x, y) for y in _ax for x in _ax])
SC_U = mixture_score(GRID, T_FIELD)
SC_C = classifier_grad(GRID, T_FIELD)
SC_G = SC_U + SC_C                                                                # s = 1: the labelled score
assert np.abs(SC_G - mixture_score(GRID, T_FIELD, SWIRL.given(0))).max() < 1e-8

# ---------------------------------------------------------------------------- cited (notes §13; Ho & Salimans 2022)

ADM_ROWS = [("none", 10.94, 0.69, 0.63), ("1.0", 4.59, 0.82, 0.52), ("10.0", 9.11, 0.88, 0.32)]  # ADM Table 4, cond.
ADM_CLF_DAYS, ADM_GEN_DAYS = 46, 916                                              # ADM Table 10, ImageNet 256
assert ADM_CLF_DAYS + ADM_GEN_DAYS == 962

# ---------------------------------------------------------------------------- look

BG_RGB = np.array([int(BG[i:i + 2], 16) for i in (1, 3, 5)], np.uint8)
FLAT = set("abcdefhiklmnorstuvwxzABCDEFGHIJKLMNOPRSTUVWXYZ0123456789")
PLANE_C, PLANE_S = (-3.7, -0.45, 0.0), 4.8


def cite(s, size=20):
    return text(s, size, MUTED)


def baseline(words):
    chars = [c for c in words.text if not c.isspace()]
    if len(words) == len(chars):
        bottoms = sorted(g.get_bottom()[1] for g, c in zip(words, chars) if c in FLAT)
        if bottoms:
            return bottoms[len(bottoms) // 2]
    return float(np.median([g.get_bottom()[1] for g in words]))


def sit(words, y):
    return words.shift((y - baseline(words)) * m.UP)


def texture(rgb):
    k = max(1, int(np.ceil(540 / max(rgb.shape[:2]))))
    arr = np.repeat(np.repeat(rgb, k, axis=0), k, axis=1)
    return np.concatenate([arr, np.full(arr.shape[:2] + (1,), 255, np.uint8)], axis=-1)


def swap(image, rgb):
    image.paint = image.paint.but(texture=texture(rgb))


def frame_rgb(final, noise, i):
    """Frame i of a digit emerging on the DDIM-50 grid (i = 50: the finished digit): VIOLET while noise outweighs the
    picture (SNR < 1), TEAL after, as in Episodes 7 and 8."""
    if i >= len(TS50):
        return tint(final, DATA, -1, 1)
    t = TS50[i]
    return tint(np.clip(blend(final, noise, t), -1, 1), DATA if snr(ab_code(t)) > 1 else NOISE, -1, 1)


def digit(img, height, color=DATA):
    return picture(tint(img, color, -1, 1), height)


def labelled_chip(label, color, fill, size=20, height=0.44, pad=0.36, mono=True, ink=INK, width=None):
    words = code(label, size, ink) if mono else text(label, size, ink)
    box = card(max(words.width + pad, height, width or 0), height, color, fill, radius=0.1, stroke=2)
    return m.VGroup(box, words.move_to(box))


def plus_node(color, radius=0.2):
    ring = m.Circle(radius=radius).set_stroke(color, 3).set_fill(BG, 1)
    bar_h = m.Line([-radius * 0.55, 0, 0], [radius * 0.55, 0, 0]).set_stroke(color, 3)
    bar_v = m.Line([0, -radius * 0.55, 0], [0, radius * 0.55, 0]).set_stroke(color, 3)
    return m.VGroup(ring, bar_h, bar_v)


def flash(mob, color, width=7, time_width=0.8, run_time=0.9):
    return m.ShowPassingFlash(mob.copy().set_stroke(color, width), time_width=time_width, run_time=run_time)


def table_rgb(rows=None, row_px=7, gap_px=1, k=5):
    """The label table as a picture: one column per number, one band per row; brighter PINK = larger."""
    rows = list(range(len(TABLE))) if rows is None else rows
    h = len(rows) * (row_px + gap_px) - gap_px
    out = np.zeros((h, TABLE.shape[1], 3), np.uint8)
    out[:] = BG_RGB
    lo = np.array([int(c, 16) for c in ("1C", "1D", "26")], float)
    hi = np.array([int(PINK[i:i + 2], 16) for i in (1, 3, 5)], float)
    for j, r in enumerate(rows):
        f = ((TABLE[r] - TAB_LO) / (TAB_HI - TAB_LO))[None, :, None]
        out[j * (row_px + gap_px): j * (row_px + gap_px) + row_px] = (lo + (hi - lo) * f).round().astype(np.uint8)
    return np.repeat(np.repeat(out, k, axis=0), k, axis=1)


def grid_rgb(images, pad=2, k=3, color=DATA):
    """A rows × cols block of digits as one picture (one texture instead of eighty)."""
    rows, cols = images.shape[:2]
    out = np.zeros((rows * 28 + (rows - 1) * pad, cols * 28 + (cols - 1) * pad, 3), np.uint8)
    out[:] = BG_RGB
    for r in range(rows):
        for c in range(cols):
            out[r * (28 + pad): r * (28 + pad) + 28, c * (28 + pad): c * (28 + pad) + 28] = tint(images[r, c], color, -1, 1)
    return np.repeat(np.repeat(out, k, axis=0), k, axis=1)


def cell_at(image, rows, cols, r, c, pad=2):
    """Scene centre and side of cell (r, c) in a grid_rgb picture."""
    w_px, h_px = cols * 28 + (cols - 1) * pad, rows * 28 + (rows - 1) * pad
    s = image.width / w_px
    x = image.get_left()[0] + (c * (28 + pad) + 14) * s
    y = image.get_top()[1] - (r * (28 + pad) + 14) * s
    return np.array([x, y, 0.0]), 28 * s


def plane(center=PLANE_C, size=PLANE_S):
    return Plot((-LIM, LIM), (-LIM, LIM), width=size, height=size, center=tuple(center), axes=False)


def glow_rgba(t, n=200, alpha=0.6, gamma=0.7, mix_=SWIRL, rgb=TEAL):
    """The exact density of the noised swirl at step t as a soft glow (alpha ∝ (p/p_max)^γ, round vignette)."""
    ax = np.linspace(-LIM, LIM, n)
    X, Y = np.meshgrid(ax, ax[::-1])
    ab = float(alpha_bar_at(t))
    logp = mixture_posterior(np.stack([X.ravel(), Y.ravel()], 1), np.sqrt(ab), np.sqrt(1 - ab), mix_).log_density
    logp = logp.reshape(n, n)
    d = np.exp(logp - logp.max()) ** gamma
    edge = np.clip((LIM - np.hypot(X, Y)) / 0.35, 0, 1)
    out = np.zeros((n, n, 4), np.uint8)
    out[..., :3] = [int(rgb[i:i + 2], 16) for i in (1, 3, 5)]
    out[..., 3] = (255 * alpha * d * edge * edge * (3 - 2 * edge)).round().astype(np.uint8)
    return out


def glow_of(plot, t, **kw):
    img = m.ImageMobject(glow_rgba(t, **kw))
    img.height = plot.h
    return img.move_to(plot.coords(0, 0)).set_z_index(-2)


def arm_line(plot, arm=0, color=PROMPT, width=3.5, opacity=0.9):
    pts = swirl_curve(120)[arm]
    return m.VMobject().set_points_smoothly([plot.coords(*p) for p in pts]).set_stroke(color, width, opacity=opacity)


def field(plot, vecs, color, k, max_len, width=2.6, min_len=0.07):
    """Arrows from every grid point: direction of vecs, length tanh(k·|v|)·max_len (always an Arrow, so two fields
    built here can be morphed into each other)."""
    arrows = m.VGroup()
    for p, v in zip(GRID, vecs):
        n = float(np.hypot(*v))
        length = max(np.tanh(k * n) * max_len, min_len)
        d = (v / n if n > 1e-12 else np.array([1.0, 0.0])) * length
        a = plot.coords(*p)
        arrows.add(m.Arrow(a, a + np.array([d[0], d[1], 0]), buff=0, color=color, stroke_width=width,
                           max_tip_length_to_length_ratio=0.38))
    return arrows


class Swarm:
    """Points replaying a sampler run (steps + 1, n, 2), VIOLET at high noise turning TEAL with the signal."""

    SHADES = 6

    def __init__(self, plot, n=1000, radius=0.022):
        self.plot, self.n = plot, n
        self.cloud = plot.cloud(Z[:n], NOISE, radius, 0.92)
        self.palette = [mix(VIOLET, TEAL, k / (self.SHADES - 1)) for k in range(self.SHADES)]
        self.shade = 0

    def _paint(self, level):
        if level != self.shade:
            self.cloud.set_color(self.palette[level])
            self.shade = level

    def replay(self, run, seconds=3.2):
        xs = run.xs[:, : self.n]
        ts = np.asarray(run.ts, float)

        def step(_, a):
            i = a * (len(xs) - 1)
            j = int(np.floor(i))
            k = min(j + 1, len(xs) - 1)
            self.cloud.positions(xs[j] + (xs[k] - xs[j]) * (i - j))
            ab = float(alpha_bar_at(ts[k]))
            frac = np.sqrt(ab) / (np.sqrt(ab) + np.sqrt(1 - ab))
            self._paint(int(round(frac * (self.SHADES - 1))))
        return m.UpdateFromAlphaFunc(self.cloud, step, run_time=seconds, rate_func=m.linear)

    def reset(self, start_xy, seconds=0.8):
        here = np.array([self.cloud._data_of(d) for d in self.cloud])

        def step(_, a):
            self.cloud.positions(here + (start_xy[: self.n] - here) * a)
            if a > 0.5:
                self._paint(0)
        return m.UpdateFromAlphaFunc(self.cloud, step, run_time=seconds)

    def to(self, xy, seconds=1.6):
        here = np.array([self.cloud._data_of(d) for d in self.cloud])

        def step(_, a):
            self.cloud.positions(here + (xy[: self.n] - here) * a)
        return m.UpdateFromAlphaFunc(self.cloud, step, run_time=seconds)


# ---------------------------------------------------------------------------- the episode


class Ep13AskingForADigit(Episode):
    number = 13

    def lesson(self):
        self.hook()
        self.the_condition()
        self.the_embedding()
        self.the_handwriting()
        self.does_it_listen()
        self.classifier_guidance()
        self.the_trade()
        self.late_push()
        self.the_catch()
        self.recap()

    # ------------------------------------------------------------------ 0 · hook

    def hook(self):
        self.next_section("Hook", notes=(
            "One starting noise (seed 7 of arrays/digit_same_noise_ten_labels.npz: x_T from np.random.default_rng(107)) "
            "given to the digit model ten times with labels 0–9, DDIM 50 steps at w = 1 (the plain conditional model, no "
            "guidance), generated on this Mac (mac_runs §4). Start and end frames are the run's; the in-between frames "
            "are √ᾱ·digit + √(1 − ᾱ)·noise on the DDIM-50 grid (illustrative; this blend tracks the run's real DDIM-50 "
            f"path for the 7 with correlation ≥ {BLEND_R:.3f} at every step). Colour turns VIOLET → TEAL with the "
            "signal's share."))
        noise = NOISES[HERO]
        big = picture(tint(np.clip(noise, -1, 1), NOISE, -1, 1), 2.5)
        big_f = framed(big, NOISE).move_to([0, 0.2, 0])
        big_cap = code("x_T · 28 × 28 random numbers", 20, MUTED).next_to(big_f, m.DOWN, buff=0.22)

        self.frames = [[frame_rgb(GRID_W1[HERO, d], noise, i) for i in range(len(TS50) + 1)] for d in range(10)]
        tile_h, pitch, ys = 1.38, 1.86, (1.0, -1.32)
        imgs, tiles, chips = [], m.Group(), m.VGroup()
        for d in range(10):
            r, c = divmod(d, 5)
            img = picture(self.frames[d][0], tile_h)
            fr = framed(img, EDGE).move_to([(c - 2) * pitch, ys[r], 0])
            imgs.append(img)
            tiles.add(fr)
            chip = labelled_chip(f"label {d}", PROMPT, PINK_FILL, 20, 0.42, 0.3)
            chips.add(chip.next_to(fr, m.DOWN, buff=0.14))
        self.hero_imgs, self.hero_tiles, self.hero_chips = imgs, tiles, chips
        counter = code("step 0 of 50", 22, TIME, weight=m.BOLD).move_to([0, 2.35, 0])
        counter.align_to(tiles, m.RIGHT)
        cap = cite("generated on this Mac · digit model · DDIM, 50 steps · one starting noise (seed 7) for all ten")
        cap.move_to([0, -3.3, 0])
        mascot = pixel(0.5).move_to([5.95, -0.15, 0])
        assert chips.get_bottom()[1] > -2.95 and cap.get_right()[0] < 6.5 and tiles.get_right()[0] + 0.3 < mascot.get_left()[0]
        state = {"i": -1}

        def grow(_, a):
            i = min(int(a * (len(TS50) + 1)), len(TS50))
            if i == state["i"]:
                return
            state["i"] = i
            for d, img in enumerate(imgs):
                swap(img, self.frames[d][i])
            counter.become(code(f"step {i} of 50", 22, TIME, weight=m.BOLD).move_to(counter.get_center())
                           .align_to(counter, m.RIGHT))

        self.say(
            "[Here is one patch of static]: [twenty-eight by twenty-eight random numbers].",
            m.AnimationGroup(self.new_header("00 · hook", "One noise, ten digits"), m.FadeIn(big_f, scale=0.9)),
            m.FadeIn(big_cap, shift=0.1 * m.UP),
        )
        self.say(
            "[We hand it to our digit model ten times], [and change only one thing]: [the label we ask for].",
            m.AnimationGroup(m.FadeOut(big_cap),
                             m.LaggedStart(*[m.FadeIn(t, target_position=big_f) for t in tiles], lag_ratio=0.06),
                             m.FadeOut(big_f), run_time=1.4),
            m.Wait(0.1),
            m.LaggedStart(*[m.FadeIn(c, shift=0.12 * m.UP) for c in chips], lag_ratio=0.08, run_time=1.2),
        )
        self.say(
            "[Same noise, same fifty steps]... [and out come ten different digits].",
            m.FadeIn(counter),
            m.AnimationGroup(m.UpdateFromAlphaFunc(m.Group(*imgs), grow, run_time=4.0, rate_func=m.linear),
                             m.FadeIn(cap, run_time=1.0)),
        )
        self.say(
            "[Today: how a label steers the denoiser], [whether it listens], [and a second way to steer it, "
            "with a classifier].",
            m.LaggedStart(*[m.Indicate(c, color=PROMPT, scale_factor=1.12) for c in chips], lag_ratio=0.08),
            m.AnimationGroup(*[m.Circumscribe(t[0], color=DATA, buff=0.05, run_time=1.0) for t in tiles]),
            m.FadeIn(mascot, shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 1 · the condition

    def the_condition(self):
        self.next_section("1 · A third input", notes=(
            "The denoiser of Episodes 5–8 took x_t and t; a condition y is one more input, so the network learns p(x | y) "
            "instead of p(x) (notes §13). The noisy 7 is a frame of the hook's label-7 tile; the training pairs are real "
            "MNIST training digits with their labels (load_mnist). The digit model hid the label in "
            f"{P_UNCOND:.0%} of training images (p_uncond = {P_UNCOND}, weights/digit_ddpm_ema.json; mac_runs §3 '10% label "
            "dropout'), so it also runs with y = 'no label' — measured below, and the basis of Episode 14."))
        i_mid = 40
        t_mid = int(TS50[i_mid])
        den = m.VGroup(card(3.5, 1.75, MODEL, AMBER_FILL, radius=0.2, stroke=3),
                       m.VGroup(text("denoiser", 30, INK, weight=m.BOLD), code("3.94 M parameters", 20, MUTED))
                       .arrange(m.DOWN, buff=0.12))
        den[1].move_to(den[0])
        den.move_to([0.75, 0.95, 0])
        xin = picture(self.frames[7][i_mid], 1.15)
        xin_f = framed(xin, NOISE).move_to([-4.45, 1.85, 0])
        xin_l = tex(r"x_t", 34, NOISE).next_to(xin_f, m.LEFT, buff=0.2)
        t_chip = labelled_chip(f"t = {t_mid}", TIME, BLUE_FILL, 22, 0.48, width=1.85).move_to([-4.45, 0.62, 0])
        y_chip = labelled_chip("y = 7", PROMPT, PINK_FILL, 22, 0.48, width=1.85).move_to([-4.45, -0.32, 0])
        y_none = labelled_chip("y = none", PROMPT, CARD, 22, 0.48, width=1.85).move_to(y_chip)
        left_x = den[0].get_left()[0]
        a_x = arrow(xin_f.get_right(), [left_x, 1.42, 0], NOISE, buff=0.12)
        a_t = arrow(t_chip.get_right(), [left_x, 0.95, 0], TIME, buff=0.12)
        a_y = arrow(y_chip.get_right(), [left_x, 0.48, 0], PROMPT, buff=0.12)
        out = arrow(den[0].get_right(), [3.65, 0.95, 0], MODEL, buff=0.12)
        out_l = m.VGroup(tex(r"\hat\varepsilon", 54, MODEL), text("noise guess", 20, MUTED)).arrange(m.RIGHT, buff=0.2)
        out_l.next_to(out, m.RIGHT, buff=0.2)
        sig = code("eps_model(x_t, t)", 22, INK, t2c={r"x_t": NOISE, r"\bt\b": TIME, r"eps_model": TEAL})
        sig.next_to(den, m.DOWN, buff=0.3)
        sig2 = code("eps_model(x_t, t, y)", 22, INK, t2c={r"x_t": NOISE, r"\bt\b": TIME, r"\by\b": PROMPT,
                                                          r"eps_model": TEAL}).move_to(sig)
        hidden = text(f"hidden for 1 in {round(1 / P_UNCOND)} training images", 20, PROMPT)
        hidden.next_to(y_chip, m.DOWN, buff=0.2).align_to(y_chip, m.LEFT).shift(0.1 * m.LEFT)
        assert out_l.get_right()[0] < 6.5 and xin_l.get_left()[0] > -6.5 and xin_f.get_top()[1] < 2.65

        imgs, labs = load_mnist("train", limit=200)
        picks = [int(np.flatnonzero(labs == d)[0]) for d in (3, 7, 0, 4)]
        pairs = m.Group()
        for k in picks:
            pic = framed(picture(tint(imgs[k] / 127.5 - 1, DATA, -1, 1), 0.92), EDGE)
            ch = labelled_chip(str(labs[k]), PROMPT, PINK_FILL, 20, 0.4, 0.3).next_to(pic, m.DOWN, buff=0.12)
            pairs.add(m.Group(pic, ch))
        pairs.arrange(m.RIGHT, buff=0.28).move_to([-3.75, -2.55, 0])
        pairs_l = text("training pairs: a picture and its label", 20, MUTED).next_to(pairs, m.UP, buff=0.2)
        p_any = m.VGroup(tex(r"p(x)", 44, DATA), text("any digit", 20, MUTED)).arrange(m.DOWN, buff=0.16)
        p_cond = m.VGroup(tex(r"p(x \mid y)", 44, DATA), text("the digit it is asked for", 20, MUTED)).arrange(m.DOWN, buff=0.16)
        p_cond[0][0][4].set_color(PROMPT)                                          # the y
        p_any.move_to([1.55, -2.4, 0])
        p_cond.move_to([4.55, -2.4, 0])
        sit(p_cond[1], baseline(p_any[1]))
        to_cond = arrow(p_any[0].get_right(), p_cond[0].get_left(), PROMPT, buff=0.25)
        assert pairs_l.get_top()[1] < hidden.get_bottom()[1] - 0.15 and pairs.get_bottom()[1] > -3.5, (
            pairs_l.get_top(), hidden.get_bottom(), pairs.get_bottom())
        assert p_cond.get_right()[0] < 6.5 and pairs.get_right()[0] < p_any.get_left()[0] - 0.3, (
            p_cond.get_right(), pairs.get_right(), p_any.get_left())

        self.say(
            "[Until now, our denoiser took two inputs]: [the noisy picture], [and the step, t].",
            m.AnimationGroup(self.new_header("01 · the condition", "A third input"), m.FadeIn(den, scale=0.95),
                             m.GrowArrow(out), m.FadeIn(out_l, shift=0.1 * m.RIGHT), m.FadeIn(sig, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(m.Group(xin_f, xin_l), shift=0.15 * m.RIGHT), m.GrowArrow(a_x)),
            m.AnimationGroup(m.FadeIn(t_chip, shift=0.15 * m.RIGHT), m.GrowArrow(a_t)),
        )
        self.say(
            "[A condition is one more input]: [here, the label of the digit we want].",
            m.AnimationGroup(m.FadeIn(y_chip, shift=0.15 * m.RIGHT), m.GrowArrow(a_y)),
            m.AnimationGroup(m.ReplacementTransform(sig, sig2), m.Circumscribe(y_chip, color=PROMPT, buff=0.08)),
        )
        self.say(
            "[Trained on digits paired with their labels], [it learns to draw the digit it is asked for], "
            "[not just any digit].",
            m.AnimationGroup(m.FadeIn(pairs_l), m.LaggedStart(*[m.FadeIn(p, shift=0.12 * m.UP) for p in pairs],
                                                              lag_ratio=0.2)),
            m.AnimationGroup(m.FadeIn(p_cond, shift=0.1 * m.LEFT)),
            m.AnimationGroup(m.FadeIn(p_any, shift=0.1 * m.RIGHT), m.GrowArrow(to_cond)),
        )
        self.say(
            "[One time in ten during training, the label was hidden], [so the same model can also run with no label].",
            m.AnimationGroup(m.FadeIn(hidden, shift=0.1 * m.UP), m.Circumscribe(y_chip, color=PROMPT, buff=0.08)),
            m.ReplacementTransform(y_chip, y_none),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 2 · the embedding

    def the_embedding(self):
        self.next_section("2 · How the label gets in", notes=(
            "The table is the trained model's own label_emb.weight (11 × 128, weights/digit_ddpm_ema.safetensors, read "
            f"with numpy; values {TAB_LO:.2f} … {TAB_HI:.2f}, brighter PINK = larger). Row 10 means 'no label' (used for "
            "the unconditional runs and for classifier-free guidance, Episode 14). The chosen row is added to the time "
            "embedding (sinusoidal 32 → Linear 128 → SiLU → Linear 128), then SiLU, and every one of the 17 residual "
            "blocks turns that vector into one number per channel (mac_runs §3; Episode 8 showed the wiring). The time "
            "bar here is a schematic: its values are not shown. ADM fed labels the same way, through AdaGN (notes §13)."))
        tab_h = 3.4
        tab = picture(table_rgb(), tab_h, pixelated=False)
        tab.move_to([-3.05, 0.45, 0])
        n_rows, row_px, gap_px = 11, 7, 1
        tot_px = n_rows * (row_px + gap_px) - gap_px
        unit = tab.height / tot_px

        def row_y(r):
            return tab.get_top()[1] - (r * (row_px + gap_px) + row_px / 2) * unit

        row_h = row_px * unit
        names = m.VGroup()
        for r in range(n_rows):
            names.add(code(str(r) if r < 10 else "none", 20, MUTED).move_to([0, row_y(r), 0])
                      .align_to([tab.get_left()[0] - 0.15, 0, 0], m.RIGHT))
        tab_cap = cite("label table · 11 × 128 numbers · from our trained model")
        tab_cap.next_to(tab, m.DOWN, buff=0.28).set_x(tab.get_x())
        hl = m.Rectangle(width=tab.width + 0.12, height=row_h + 0.08).set_stroke(PROMPT, 3).move_to([tab.get_x(), row_y(7), 0])
        hl_none = m.Rectangle(width=tab.width + 0.12, height=row_h + 0.08).set_stroke(PROMPT, 3)
        hl_none.move_to([tab.get_x(), row_y(10), 0])
        none_note = text("no label · Episode 14 builds on it", 20, PROMPT).next_to(tab_cap, m.DOWN, buff=0.16)
        none_note.align_to(tab_cap, m.LEFT)
        assert names.get_left()[0] > -6.6 and tab_cap.get_left()[0] > -6.6 and none_note.get_bottom()[1] > -3.5

        bx, bw = 3.85, 4.2
        row7 = picture(table_rgb([7], k=5), row_h * bw / tab.width, pixelated=False)
        row7.move_to([bx, 2.05, 0])
        lab_w = text("label 7", 22, PROMPT, weight=m.BOLD).next_to(row7, m.LEFT, buff=0.25)
        plus = plus_node(INK, 0.2).move_to([bx, 1.45, 0])
        tbar = card(bw, row7.height, TIME, BLUE_FILL, radius=0.04, stroke=2).move_to([bx, 0.85, 0])
        lab_t = text("time t", 22, TIME, weight=m.BOLD).next_to(tbar, m.LEFT, buff=0.25).align_to(lab_w, m.RIGHT)
        u = UNet(sizes=(28, 14, 7), channels=(32, 64, 128), center=(bx, -1.72, 0), scale=0.5, labels=False)
        blocks = list(u.down) + [u.mid] + list(u.up)
        xs = [b.get_x() for b in blocks]
        bus_y = -0.42
        down = m.Arrow([bx, tbar.get_bottom()[1] - 0.05, 0], [bx, bus_y + 0.03, 0], buff=0, color=PROMPT,
                       stroke_width=4, max_tip_length_to_length_ratio=0.2)
        bus = m.Line([min(xs), bus_y, 0], [max(xs), bus_y, 0]).set_stroke(PROMPT, 4)
        ticks = m.VGroup(*[m.Line([b.get_x(), bus_y, 0], [b.get_x(), b.get_top()[1] + 0.03, 0]).set_stroke(PROMPT, 2.5)
                           for b in blocks])
        sum_l = code("sum · SiLU", 20, MUTED).next_to(down, m.RIGHT, buff=0.18)
        ep8 = pill(f"Episode 8 · into all {N_BLOCKS} residual blocks", MODEL, AMBER_FILL, 20).move_to([bx, -3.3, 0])
        assert lab_w.get_left()[0] > tab.get_right()[0] + 0.3 and row7.get_right()[0] < 6.5
        assert u.get_bottom()[1] > ep8.get_top()[1] + 0.1 and u.get_top()[1] < bus_y - 0.2 and ep8.get_right()[0] < 6.5, (
            u.get_bottom(), ep8.get_top(), u.get_top())
        assert u.get_right()[0] < 6.5 and names.get_left()[0] > -6.55

        self.say(
            "[The label is an index into a table]: [eleven rows of a hundred twenty-eight learned numbers].",
            m.AnimationGroup(self.new_header("02 · embedding", "How the label gets in"),
                             m.LaggedStart(*[m.FadeIn(n, shift=0.1 * m.RIGHT) for n in names], lag_ratio=0.05)),
            m.AnimationGroup(m.FadeIn(tab, shift=0.1 * m.RIGHT), m.FadeIn(tab_cap)),
        )
        self.say(
            "[Ask for a seven], [and row seven is looked up].",
            m.AnimationGroup(m.Create(hl), names[7].animate.set_color(PROMPT)),
            m.AnimationGroup(m.FadeIn(row7, target_position=hl), m.FadeIn(lab_w, shift=0.1 * m.RIGHT)),
        )
        self.say(
            "[It is added to the time embedding], [and the sum reaches every residual block], "
            "[just as episode eight showed].",
            m.AnimationGroup(m.FadeIn(plus), m.FadeIn(tbar, shift=0.1 * m.UP), m.FadeIn(lab_t)),
            m.Succession(m.AnimationGroup(m.FadeIn(u), m.GrowArrow(down), m.FadeIn(sum_l), m.Create(bus),
                                          m.Create(ticks), run_time=1.1),
                         m.AnimationGroup(m.LaggedStart(*[flash(t, PROMPT, 7, 0.9, 0.7) for t in ticks], lag_ratio=0.1),
                                          *[b.animate(rate_func=m.there_and_back, run_time=1.3).set_stroke(PROMPT, 4)
                                            for b in blocks])),
            m.AnimationGroup(m.FadeIn(ep8, shift=0.1 * m.UP)),
        )
        self.say(
            "[The last row means no label]: [the setting episode fourteen builds on].",
            m.AnimationGroup(m.Create(hl_none), names[10].animate.set_color(PROMPT)),
            m.FadeIn(none_note, shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 3 · the handwriting

    def the_handwriting(self):
        self.next_section("3 · The noise writes, the label decides", notes=(
            "All 8 rows × 10 labels of the run's DDIM-50, w = 1 grid (mac_runs §4; images/digit_same_noise_ten_labels_"
            "ddpm_ddim50_w1.png), each row from one starting noise (the VIOLET tile). Slant = the image moment μ11/μ02 of "
            f"each digit; of its variation across the grid, the row means (the noise) explain {SLANT_NOISE:.0%} and the "
            f"column means (the label) {SLANT_LABEL:.0%} (the DDPM-1000 and flow-50 grids agree: 28% vs 12%, 40% vs "
            f"13%). Stroke weight relative to each label's average: seed {BOLD} {REL_INK[BOLD]:.2f}× (boldest), seed "
            f"{THIN} {REL_INK[THIN]:.2f}× (thinnest)."))
        rgb = grid_rgb(GRID_W1)
        g = picture(rgb, 4.55, pixelated=False).move_to([-2.45, -0.6, 0])
        cell = cell_at(g, 8, 10, 0, 0)[1]
        thumbs = m.Group()
        for r in range(8):
            c, _ = cell_at(g, 8, 10, r, 0)
            th = picture(tint(np.clip(NOISES[r], -1, 1), NOISE, -1, 1), cell)
            thumbs.add(th.move_to([g.get_left()[0] - 0.5, c[1], 0]))
        heads = m.VGroup(*[code(str(d), 20, PROMPT).move_to(cell_at(g, 8, 10, 0, d)[0] + np.array([0, cell / 2 + 0.24, 0]))
                           for d in range(10)])
        n_head = code("x_T", 20, NOISE).move_to([thumbs.get_x(), heads[0].get_y(), 0])
        sit(n_head, baseline(heads[0]))
        cap = cite("generated on this Mac · digit model · DDIM, 50 steps · 8 noises × 10 labels").move_to([-2.45, -3.3, 0])

        def row_box(r, color):
            c0, _ = cell_at(g, 8, 10, r, 0)
            return m.Rectangle(width=g.width + 0.95, height=cell + 0.08).set_stroke(color, 3).move_to(
                [g.get_x() - 0.43, c0[1], 0])

        col_c, _ = cell_at(g, 8, 10, 0, 2)
        col_box = m.Rectangle(width=cell + 0.08, height=g.height + 0.08).set_stroke(PROMPT, 3).move_to([col_c[0], g.get_y(), 0])
        boxes = m.VGroup(row_box(BOLD, NOISE), row_box(THIN, NOISE))
        rx = 1.75
        key1 = text("down a column: one label", 22, INK)
        key2 = text("along a row: one noise", 22, INK)
        k1 = m.VGroup(m.Square(0.2).set_stroke(PROMPT, 3), key1).arrange(m.RIGHT, buff=0.2)
        k2 = m.VGroup(m.Square(0.2).set_stroke(NOISE, 3), key2).arrange(m.RIGHT, buff=0.2)
        keys = m.VGroup(k1, k2).arrange(m.DOWN, buff=0.3, aligned_edge=m.LEFT).move_to([0, 1.25, 0]).align_to([rx, 0, 0], m.LEFT)
        notes_ = m.VGroup(text(f"seed {BOLD}: bold strokes", 20, MUTED), text(f"seed {THIN}: thin strokes", 20, MUTED))
        notes_.arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT).next_to(keys, m.DOWN, buff=0.35).align_to(keys, m.LEFT)
        q = text("what sets the slant?", 24, INK, weight=m.BOLD).move_to([0, -1.05, 0]).align_to([rx, 0, 0], m.LEFT)
        rows = m.VGroup()
        for name, v, col in (("the noise", SLANT_NOISE, NOISE), ("the label", SLANT_LABEL, PROMPT)):
            words = text(name, 22, INK)
            bar = Meter(2.2, 0.0, col, height=0.2)
            num = code(f"{v:.0%}", 22, INK)
            rows.add(m.VGroup(words, bar, num))
        for i, r_ in enumerate(rows):
            y = q.get_bottom()[1] - 0.45 - 0.5 * i
            r_[0].move_to([0, y, 0]).align_to([rx, 0, 0], m.LEFT)
            r_[1].move_to([rx + 1.55 + 1.1, y, 0])
            r_[2].next_to(r_[1], m.RIGHT, buff=0.2)
            sit(r_[2], baseline(r_[0]))
        assert all(r_[0].get_right()[0] < r_[1].get_left()[0] - 0.2 for r_ in rows)
        foot = cite("share of the slant's variation\nacross this grid").next_to(rows, m.DOWN, buff=0.25).align_to(q, m.LEFT)
        assert rows.get_right()[0] < 6.5 and foot.get_bottom()[1] > -3.5 and g.get_right()[0] < rx - 0.3
        assert thumbs.get_left()[0] > -6.6 and heads.get_top()[1] < 2.65

        self.say(
            "[Here are eight different noises], [each asked for all ten digits].",
            m.AnimationGroup(self.new_header("03 · style", "The noise writes, the label decides"),
                             m.LaggedStart(*[m.FadeIn(t, shift=0.1 * m.RIGHT) for t in thumbs], lag_ratio=0.1),
                             m.FadeIn(n_head)),
            m.AnimationGroup(m.FadeIn(g, shift=0.1 * m.RIGHT), m.FadeIn(heads), m.FadeIn(cap)),
        )
        self.say(
            "[Down a column, the label fixes the digit]. [Along a row, the noise sets the handwriting]: "
            "[its slant, and how bold the strokes are].",
            m.AnimationGroup(m.Create(col_box), m.FadeIn(k1, shift=0.1 * m.UP)),
            m.AnimationGroup(m.Create(boxes[0]), m.FadeIn(k2, shift=0.1 * m.UP)),
            m.AnimationGroup(m.Create(boxes[1]), m.FadeIn(notes_, shift=0.1 * m.UP)),
        )
        self.say(
            "[Measured across this grid], [the starting noise explains about four times more of the slant than "
            "the label does].",
            m.AnimationGroup(m.FadeIn(q, shift=0.1 * m.UP), m.FadeIn(rows), m.FadeIn(foot)),
            m.AnimationGroup(rows[0][1].fill_to(SLANT_NOISE, run_time=1.2), rows[1][1].fill_to(SLANT_LABEL, run_time=1.2)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 4 · does it listen?

    def does_it_listen(self):
        self.next_section("4 · Does it listen?", notes=(
            f"The run's own classifier (a small CNN, {CLF['params']:,} parameters, trained on this Mac in "
            f"{CLF['wall_s']:.0f} s): {CLF_ACC:.2%} on the 10,000 MNIST test digits ({CLF['test_errors']} errors). "
            f"Conditional accuracy = classifier label == requested label, over 10,000 generated digits, DDIM 50 "
            f"(mac_runs §4 guidance table): w = 1 {ACC_W1:.1%}; w = 0 (the 'no label' row) {ACC_W0:.2%} — chance, since "
            "the label is ignored (its label histogram is 8–11% per class). The row shown is seed 5 at w = 1; that its "
            "4 looks like a 9 and its 6 like a 0 is judged by eye (the classifier was not re-run)."))
        box = card(4.7, 2.75, EDGE, CARD, radius=0.2, stroke=2).move_to([-3.75, 0.85, 0])
        net = NeuralNet(layers=(6, 8, 8, 4), width=2.4, height=1.0, center=(-3.75, 1.6, 0), color=MUTED, radius=0.07)
        c_name = text("a separate digit classifier", 24, INK, weight=m.BOLD).move_to([-3.75, 0.66, 0])
        c_acc = code(f"{100 * CLF_ACC:.2f}% on test digits", 22, GOOD).move_to([-3.75, 0.17, 0])
        c_src = cite(f"trained on this Mac in {CLF['wall_s']:.0f} s").move_to([-3.75, -0.24, 0])
        clf = m.VGroup(box, net, c_name, c_acc, c_src)
        assert c_src.width < box.width - 0.3, c_src.width

        bars = Bars([100 * ACC_W1, 100 * ACC_W0], labels=["label given", "no label"], top=100, width=3.4, height=2.3,
                    colors=[GOOD, MUTED], gap=0.9, fmt="{:.1f}%", size=22, center=(2.45, 0.42, 0))
        chance = m.DashedLine([bars.xs[0] - 0.8, bars.level(10), 0], [bars.xs[1] + 0.8, bars.level(10), 0],
                              dash_length=0.08).set_stroke(INK, 2.5)
        chance_l = code("chance · 10%", 20, INK).next_to(chance, m.RIGHT, buff=0.15)
        b_title = text("read as the digit we asked for", 22, MUTED).move_to([2.45, 2.35, 0])
        assert chance_l.get_right()[0] < 6.6 and bars.numbers[0].get_top()[1] < b_title.get_bottom()[1] - 0.1, (chance_l.get_right(), bars.numbers[0].get_top(), b_title.get_bottom())

        row = m.Group()
        for d in range(10):
            fr = framed(digit(GRID_W1[MISS, d], 0.8), EDGE)
            row.add(fr)
        row.arrange(m.RIGHT, buff=0.14).move_to([0, -2.45, 0])
        heads = m.VGroup(*[code(str(d), 20, PROMPT).next_to(row[d], m.UP, buff=0.2) for d in range(10)])
        rings = m.VGroup(*[m.SurroundingRectangle(row[d], buff=0.06).set_stroke(BAD, 4) for d in MISSES])
        row_cap = cite("one noise (seed 5), labels 0–9 · generated on this Mac · DDIM, 50 steps").move_to([0, -3.32, 0])
        assert heads.get_top()[1] < bars.labels.get_bottom()[1] - 0.15

        self.say(
            "[To check, we use a separate digit classifier], [right on ninety-nine point four five percent of real "
            "test digits].",
            m.AnimationGroup(self.new_header("04 · does it listen?", "Checking with a classifier"),
                             m.FadeIn(m.VGroup(box, net, c_name), shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(c_acc, shift=0.1 * m.UP), m.FadeIn(c_src)),
        )
        self.say(
            "[With the label], [it reads ninety-eight point one percent of our generated digits as the one we asked for].",
            m.AnimationGroup(m.FadeIn(b_title), m.FadeIn(bars.zero), m.FadeIn(bars.labels[0])),
            m.AnimationGroup(m.GrowFromEdge(bars.bars[0], m.DOWN), m.FadeIn(bars.numbers[0], shift=0.1 * m.UP)),
        )
        self.say(
            "[With no label], [nine point nine percent]: [chance, one in ten].",
            m.FadeIn(bars.labels[1]),
            m.AnimationGroup(m.GrowFromEdge(bars.bars[1], m.DOWN), m.FadeIn(bars.numbers[1], shift=0.1 * m.UP)),
            m.AnimationGroup(m.Create(chance), m.FadeIn(chance_l)),
        )
        self.say(
            "[Not every one lands], though: [in this row, the four came out looking like a nine], [and the six like "
            "a zero].",
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(f, shift=0.1 * m.UP) for f in row], lag_ratio=0.06),
                             m.FadeIn(heads), m.FadeIn(row_cap)),
            m.Create(rings[0]),
            m.Create(rings[1]),
        )
        self.miss_row = (row, heads, rings, row_cap)
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 5 · classifier guidance

    def classifier_guidance(self):
        self.next_section("5 · Classifier guidance", notes=(
            "Dhariwal & Nichol 2021 ('Diffusion Models Beat GANs on Image Synthesis', arXiv:2105.05233), ADM Alg. 2 / "
            "eq. 14: ε̂ = ε_θ(x_t) − √(1 − ᾱ_t)·s·∇ log p_φ(y | x_t), with a classifier p_φ trained on noisy images; "
            "Bayes: ∇ log p(x | y) = ∇ log p(x) + ∇ log p(y | x) (eqs. 12–13; notes §13). On the swirl both parts are "
            "exact: the arms are the two labels and a perfect classifier's gradient is score(x | arm) − score(x) "
            f"(checked against finite differences). Arrows at t = {T_FIELD} (linear schedule) on a 15 × 15 grid; "
            "lengths are squashed (tanh(√(1 − ᾱ)·|v|)) so every arrow fits; directions are exact. LIME = the score "
            "(Episode 6), PINK = the classifier's push toward arm 0 (the PINK line), then their sum at s = 1."))
        pl = plane()
        frame_sq = m.Square(PLANE_S).set_stroke(EDGE, 2).move_to(PLANE_C)
        glow = glow_of(pl, T_FIELD)
        line0 = arm_line(pl)
        ab = float(alpha_bar_at(T_FIELD))
        k = float(np.sqrt(1 - ab))
        step_ = PLANE_S * (1 - 2 * FIELD_MARGIN) / (FIELD_N - 1)
        f_u = field(pl, SC_U, SCORE, k, 0.85 * step_)
        f_c = field(pl, SC_C, PROMPT, k, 0.85 * step_, width=3.0)
        f_g = field(pl, SC_G, SCORE, k, 0.85 * step_)
        p_cap = cite(f"the swirl · exact scores · step t = {T_FIELD}").next_to(frame_sq, m.DOWN, buff=0.16)
        arm_tag = labelled_chip("asked for: this arm", PROMPT, PINK_FILL, 20, 0.42, 0.3)
        arm_tag.move_to([PLANE_C[0], frame_sq.get_top()[1] + 0.32, 0])

        rx = 3.45
        # first centred on the stage, then moved to the right-hand panel when the swirl arrives
        title = title_text("classifier guidance", 50, PROMPT).move_to([0, 1.85, 0])
        src = cite("Dhariwal & Nichol, 2021").next_to(title, m.DOWN, buff=0.16)
        f1 = m.MathTex(r"\hat\varepsilon", "=", r"\varepsilon_\theta(x_t)", r"-\sqrt{1-\bar\alpha_t}\,", "s",
                       r"\,\nabla \log p_\phi(y \mid x_t)", font_size=42)
        f1[2].set_color(MODEL)
        f1[4].set_color(PROMPT)
        f1[5].set_color(PROMPT)
        f1.move_to([0, 0.15, 0])
        f1_a = text("the denoiser's guess", 22, MODEL).next_to(f1[2], m.UP, buff=0.22)
        f1_b = text("a classifier's push, scaled by s", 22, PROMPT).next_to(f1[5], m.DOWN, buff=0.24)
        f1_b.align_to(f1, m.RIGHT)
        f2 = m.MathTex(r"\nabla \log p(x \mid y)", "=", r"\nabla \log p(x)", "+", r"\nabla \log p(y \mid x)", font_size=42)
        f2[2].set_color(SCORE)
        f2[4].set_color(PROMPT)
        f2.move_to([0, -1.75, 0])
        f2_l = cite("Bayes' rule, in score form").next_to(f2, m.DOWN, buff=0.22)
        assert f1.width < 12.4 and f2.width < 12.4 and f1_a.get_top()[1] < src.get_bottom()[1] - 0.1, (
            f1.width, f2.width, f1_a.get_top(), src.get_bottom())
        assert f1_b.get_bottom()[1] > f2.get_top()[1] + 0.25 and f2_l.get_bottom()[1] > -3.5
        k_title, k_f1 = 0.88, min(0.95, 5.6 / f1.width)
        title_to = np.array([rx, 1.95, 0])
        src_to = title_to - np.array([0, k_title * title.height / 2 + 0.14 + src.height / 2, 0])
        f1_to = np.array([rx, 0.8, 0])
        assert rx + k_f1 * f1.width / 2 < 6.5 and f1_to[1] + k_f1 * f1.height / 2 < src_to[1] - src.height / 2 - 0.1

        key_s = m.VGroup(m.Arrow([0, 0, 0], [0.45, 0, 0], buff=0, color=SCORE, stroke_width=3,
                                 max_tip_length_to_length_ratio=0.35), text("the score: toward likely points", 20, INK))
        key_c = m.VGroup(m.Arrow([0, 0, 0], [0.45, 0, 0], buff=0, color=PROMPT, stroke_width=3,
                                 max_tip_length_to_length_ratio=0.35), text("the classifier's push", 20, INK))
        for kk in (key_s, key_c):
            kk.arrange(m.RIGHT, buff=0.2)
        keys = m.VGroup(key_s, key_c).arrange(m.DOWN, buff=0.22, aligned_edge=m.LEFT)
        keys.move_to([0, -2.6, 0]).align_to([0.85, 0, 0], m.LEFT)
        bend = pill("the sum bends toward the pink arm", SCORE, LIME_FILL, 20).move_to([rx, -0.75, 0])
        assert keys.get_bottom()[1] > -3.5 and bend.get_right()[0] < 6.5

        self.say(
            "[There is a second way to steer], [published by Dhariwal and Nichol in twenty twenty-one]: "
            "[classifier guidance].",
            m.AnimationGroup(self.new_header("05 · classifier guidance", "Steering with a classifier")),
            m.FadeIn(src, shift=0.1 * m.UP),
            m.Write(title),
        )
        self.say(
            "[Train a classifier on noisy pictures]. [Then, at every step], [nudge the sample along its gradient], "
            "[toward the label you want].",
            m.FadeIn(m.VGroup(f1[0], f1[1], f1[2]), shift=0.1 * m.UP),
            m.FadeIn(f1_a, shift=0.1 * m.UP),
            m.FadeIn(m.VGroup(f1[3], f1[4], f1[5]), shift=0.1 * m.UP),
            m.FadeIn(f1_b, shift=0.1 * m.UP),
        )
        self.say(
            "[It works because of Bayes' rule]: [the score of the labelled data] [is the plain score, "
            "plus the classifier's gradient].",
            m.FadeIn(f2[1]),
            m.FadeIn(f2[0], shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(m.VGroup(f2[2], f2[3], f2[4]), shift=0.1 * m.UP), m.FadeIn(f2_l)),
        )
        self.say(
            "[On the swirl, we can compute both exactly]. [These arrows point toward more likely points, "
            "at step twenty].",
            m.AnimationGroup(m.FadeOut(m.VGroup(f2, f2_l, f1_a, f1_b)), f1.animate.scale(k_f1).move_to(f1_to),
                             title.animate.scale(k_title).move_to(title_to), src.animate.move_to(src_to),
                             m.FadeIn(frame_sq), m.FadeIn(glow), m.FadeIn(p_cap)),
            m.AnimationGroup(m.LaggedStart(*[m.GrowArrow(a) for a in f_u], lag_ratio=0.004, run_time=1.6),
                             m.FadeIn(key_s, shift=0.1 * m.UP)),
        )
        self.say(
            "[Ask for this arm], [and a perfect classifier adds its own push]: [pink arrows, toward the arm we asked for].",
            m.AnimationGroup(m.Create(line0, run_time=1.2), m.FadeIn(arm_tag, shift=0.1 * m.DOWN)),
            m.AnimationGroup(f_u.animate(run_time=0.6).set_opacity(0.18), m.FadeIn(key_c, shift=0.1 * m.UP)),
            m.LaggedStart(*[m.GrowArrow(a) for a in f_c], lag_ratio=0.004, run_time=1.6),
        )
        self.say(
            "[Add the two], [and the whole field bends toward the requested arm].",
            m.AnimationGroup(m.FadeOut(f_c, run_time=0.8), m.ReplacementTransform(f_u, f_g, run_time=1.8)),
            m.FadeIn(bend, shift=0.1 * m.UP),
        )
        self.plane_parts = (pl, frame_sq, glow, line0, p_cap, arm_tag, f_g, f1, title, src, keys, bend)
        self.wait(0.3)

    # ------------------------------------------------------------------ 6 · the trade

    def the_trade(self):
        self.next_section("6 · Quality against variety", notes=(
            "Sampling with the exact parts, DDIM 100 steps from z = gaussian_noise(2000, seed=5) (the toy's shared noise; "
            "1,000 of the points shown). Unguided: "
            f"{SHARE_U:.1%} land on the pink arm (a perfect classifier's verdict on the samples); classifier-guided at "
            f"s = 1: {SHARE_1:.0%}; at s = 7: {SHARE_7:.0%}, but crowded onto the arm's outer half (spread √trace cov "
            f"{SPREAD_1:.4f} → {SPREAD_7:.4f}, coverage {COVER_7:.2f}; toy.md §5 lists the same numbers for the "
            "equivalent CFG runs). ADM Table 4 (ImageNet 256 × 256, the class-conditional model; FID lower is better; "
            "precision ≈ fidelity, recall ≈ variety): no guidance 10.94 / 0.69 / 0.63; classifier scale 1.0 4.59 / 0.82 / "
            "0.52; scale 10.0 9.11 / 0.88 / 0.32 (Dhariwal & Nichol 2021, notes §13)."))
        pl, frame_sq, glow, line0, p_cap, arm_tag, f_g, f1, title, src, keys, bend = self.plane_parts
        glow0 = glow_of(pl, 0)
        sw = Swarm(pl, 1000)
        p_cap2 = cite("the swirl · DDIM, 100 steps · exact parts").move_to(p_cap)
        rx = 3.45
        r_lab = text("of 2,000 points, on the pink arm", 20, MUTED).move_to([rx, 1.35, 0])
        r_num = title_text(f"{SHARE_U:.0%}", 60, INK).next_to(r_lab, m.UP, buff=0.15)
        r_sub = code("no push", 20, MUTED).next_to(r_lab, m.DOWN, buff=0.14)
        r_num1 = title_text(f"{SHARE_1:.0%}", 60, GOOD).move_to(r_num)
        r_sub1 = code("guided · s = 1", 20, PROMPT).move_to(r_sub)
        r_sub7 = code("guided · s = 7", 20, PROMPT).move_to(r_sub)
        sp_rows = m.VGroup(
            m.VGroup(text("on its outer half", 22, INK), code(f"{OUTER_1:.0%} → {OUTER_7:.0%}", 24, BAD, weight=m.BOLD)),
            m.VGroup(text("spread", 22, INK), code(f"{SPREAD_1:.2f} → {SPREAD_7:.2f}", 24, BAD, weight=m.BOLD)))
        for i, (lab, val) in enumerate(sp_rows):
            y = r_sub.get_y() - 0.5 - 0.42 * i
            lab.move_to([0, y, 0]).align_to([rx - 0.15, 0, 0], m.RIGHT)
            val.next_to(lab, m.RIGHT, buff=0.3)
            sit(val, baseline(lab))
        spread_l = sp_rows
        assert r_num.get_top()[1] < 2.6

        rows = [("scale", "FID", "precision", "recall")] + [(a, f"{b:.2f}", f"{c:.2f}", f"{d:.2f}") for a, b, c, d in ADM_ROWS]
        col_x, row_dy = (0.0, 1.0, 2.05, 3.75), 0.38                                     # one Text per cell
        cells = m.VGroup()
        for i, r in enumerate(rows):
            line = m.VGroup()
            for j, cell_s in enumerate(r):
                word = code(cell_s, 20, MUTED if i == 0 else INK)
                line.add(word.move_to([col_x[j] + word.width / 2, -i * row_dy, 0]))
            for w in line[1:]:
                sit(w, baseline(line[0]))
            cells.add(line)
        tcap = text("ImageNet, class-conditional model", 20, INK, weight=m.BOLD)
        tsrc = cite("Dhariwal & Nichol, 2021 · Table 4")
        inner = m.VGroup(tcap, cells, tsrc).arrange(m.DOWN, buff=0.2, aligned_edge=m.LEFT)
        tbox = card(inner.width + 0.6, inner.height + 0.5, PROMPT, CARD, radius=0.16, stroke=2)
        tcard = m.VGroup(tbox, inner.move_to(tbox))
        tcard.move_to([rx, -1.95, 0])

        def col_rect(row, col, color):
            return m.SurroundingRectangle(cells[row][col], buff=0.07).set_stroke(color, 2.5)

        fid_hl = col_rect(2, 1, GOOD)
        rec_hl = col_rect(3, 3, BAD)
        prec_hl = col_rect(3, 2, GOOD)
        assert tcard.get_right()[0] < 6.5 and tcard.get_bottom()[1] > -3.5 and tcard.get_top()[1] < spread_l.get_bottom()[1] - 0.1

        self.say(
            "[Without the push], [points sampled from noise split about evenly between the two arms].",
            m.AnimationGroup(self.new_header("06 · the trade", "Quality against variety"),
                             m.FadeOut(m.VGroup(f_g, f1, title, src, keys, bend)), m.FadeOut(glow), m.FadeIn(glow0),
                             m.FadeOut(p_cap), m.FadeIn(p_cap2), m.FadeIn(sw.cloud)),
            m.AnimationGroup(sw.replay(RUN_U, 3.2),
                             m.Succession(m.Wait(2.2), m.AnimationGroup(m.FadeIn(r_num, shift=0.1 * m.UP),
                                                                        m.FadeIn(r_lab), m.FadeIn(r_sub)))),
        )
        self.say(
            "[With the classifier's push], [every one of two thousand lands on the pink arm].",
            sw.reset(Z),
            m.AnimationGroup(sw.replay(RUN_1, 3.2),
                             m.Succession(m.Wait(2.4), m.AnimationGroup(m.Transform(r_num, r_num1),
                                                                        m.Transform(r_sub, r_sub1)))),
        )
        self.say(
            "[Turn the guidance scale up to seven], [and they crowd onto the arm's outer half], "
            "[where it looks least like the other arm].",
            m.Transform(r_sub, r_sub7),
            sw.to(RUN_7.xs[-1], 1.8),
            m.FadeIn(spread_l, shift=0.1 * m.UP),
        )
        self.say(
            "[The paper found the same trade on ImageNet]: [at scale one, the FID score improved "
            "from ten point nine to four point six].",
            m.FadeIn(tcard, shift=0.1 * m.UP),
            m.Create(fid_hl),
        )
        self.say(
            "[At scale ten, precision rose further], [but recall, a measure of variety, fell to zero point three two].",
            m.AnimationGroup(m.FadeOut(fid_hl), m.Create(prec_hl)),
            m.Create(rec_hl),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 7 · when it pushes

    def late_push(self):
        self.next_section("7 · Guidance works late", notes=(
            "The swirl's 20,000 labelled points (swirl_labelled(20000, seed=0); arm 0 PINK, arm 1 TEAL) noised to t = 500 "
            "on the linear schedule (gaussian_noise(20000, seed=1)); shown: the first 1,000 that stay inside the ±3 "
            f"plane. A perfect classifier (arm_probability > 0.5) names the right arm for {ACC500:.1%} of all 20,000 "
            "(toy.json headline arm_accuracy_t500_linear). Accuracy by step from toy.json forward.linear: "
            + ", ".join(f"t = {t}: {a:.1%}" for t, a in ACC_TABLE)
            + ". Push = the mean size of √(1 − ᾱ_t)·|∇ log p(arm | x_t)| along the s = 1 guided DDIM-100 path (ε units): "
            f"it peaks at t = {PEAK_T}; {LATE_SHARE:.1%} of it comes at t ≤ 300."))
        pl = Plot((-LIM_WIDE, LIM_WIDE), (-LIM_WIDE, LIM_WIDE), width=PLANE_S, height=PLANE_S, center=PLANE_C, axes=False)
        frame_sq = m.Square(PLANE_S).set_stroke(EDGE, 2).move_to(PLANE_C)
        cl = pl.cloud(SHOW_X0, PROMPT, 0.024, 0.92)
        for d, a in zip(cl, SHOW_ARM):
            if a == 1:
                d.set_color(DATA)
        t_lab = code("t = 0", 22, TIME, weight=m.BOLD).move_to([PLANE_C[0], frame_sq.get_top()[1] + 0.3, 0])
        t_lab5 = code("t = 500", 22, TIME, weight=m.BOLD).move_to(t_lab)
        key = m.VGroup(m.Dot(radius=0.07, color=PROMPT), text("the arm we ask for", 20, INK),
                       m.Dot(radius=0.07, color=DATA), text("the other arm", 20, INK)).arrange(m.RIGHT, buff=0.15)
        key[2].shift(0.25 * m.RIGHT)
        key[3].shift(0.25 * m.RIGHT)
        key.next_to(frame_sq, m.DOWN, buff=0.2)

        rx = 3.55
        big = title_text(f"{100 * ACC500:.1f}%", 64, INK).move_to([rx, 0.95, 0])
        big_l = text("a perfect classifier names the right arm", 20, MUTED).next_to(big, m.DOWN, buff=0.15)
        big_c = code("at t = 500 · a coin flip is 50%", 20, MUTED).next_to(big_l, m.DOWN, buff=0.12)

        order = sorted(ACC_TABLE, key=lambda r: -r[0])                                  # as sampling runs: 1000 → 0
        bars = Bars([a - 0.5 for _, a in order], labels=[str(t) for t, _ in order], top=0.5, width=4.6, height=1.35,
                    colors=[PROMPT] * len(order), gap=0.16, show_values=False, size=20, center=(rx, 0.95, 0))
        nums = m.VGroup(*[code(f"{a:.0%}", 20, INK).move_to([x, bars.level(a - 0.5) + 0.2, 0])
                          for x, (_, a) in zip(bars.xs, order)])
        b_title = text("a perfect classifier is right, by step t", 20, MUTED).move_to([rx, 2.32, 0])
        b_floor = code("50%", 20, MUTED).next_to(bars.zero, m.LEFT, buff=0.12)
        i100, i50 = [t for t, _ in order].index(100), [t for t, _ in order].index(50)
        push = Plot((1000, 0), (0, 0.5), width=4.6, height=1.15, center=(rx, -2.05, 0), x_ticks=[1000, 500, 0],
                    y_ticks=[0, 0.5], y_label="the push, guidance scale 1", x_label="step t, as sampling runs")
        push_line = push.line(PUSH_TS, PUSH, PROMPT, 4)
        late = push.area([t for t in PUSH_TS if t <= 300], [q for t, q in zip(PUSH_TS, PUSH) if t <= 300], PROMPT, 0.35)
        late_l = code(f"{LATE_SHARE:.0%} of the push", 20, INK, weight=m.BOLD).move_to(push.coords(600, 0.3))
        assert nums.get_top()[1] < b_title.get_bottom()[1] - 0.08 and bars.labels.get_bottom()[1] > push.get_top()[1] + 0.2, (
            nums.get_top(), b_title.get_bottom(), bars.labels.get_bottom(), push.get_top())
        assert push.get_bottom()[1] > -3.5 and push.get_left()[0] > frame_sq.get_right()[0] + 0.3 and b_floor.get_left()[0] > 0.4
        assert key.get_bottom()[1] > -3.5 and t_lab.get_top()[1] < 2.65

        self.say(
            "[One more honest fact]. [Colour the swirl by arm], [and add noise up to step five hundred].",
            m.AnimationGroup(self.new_header("07 · timing", "Guidance works late"), m.FadeIn(frame_sq)),
            m.AnimationGroup(m.FadeIn(cl), m.FadeIn(t_lab), m.FadeIn(key)),
            m.AnimationGroup(cl.to(SHOW_X500, run_time=2.4), m.Transform(t_lab, t_lab5)),
        )
        self.say(
            "[Even a perfect classifier now names the right arm only fifty-two point eight percent of the time].",
            m.AnimationGroup(m.FadeIn(big, shift=0.15 * m.UP), m.FadeIn(big_l), m.FadeIn(big_c)),
        )
        self.say(
            "[Its accuracy climbs only near the end]: [about sixty-four percent at step one hundred], "
            "[seventy-eight at step fifty].",
            m.AnimationGroup(m.FadeOut(m.VGroup(big, big_l, big_c)), m.FadeIn(b_title), m.FadeIn(bars.zero),
                             m.FadeIn(bars.labels), m.FadeIn(b_floor),
                             m.LaggedStart(*[m.AnimationGroup(m.GrowFromEdge(b, m.DOWN), m.FadeIn(n_))
                                             for b, n_ in zip(bars.bars, nums)], lag_ratio=0.15, run_time=1.6)),
            m.Circumscribe(m.VGroup(bars.bars[i100], nums[i100]), color=INK, buff=0.08),
            m.Circumscribe(m.VGroup(bars.bars[i50], nums[i50]), color=INK, buff=0.08),
        )
        self.say(
            "[So the push is tiny at high noise], [and nearly three quarters of it comes in the last three hundred steps].",
            m.AnimationGroup(m.FadeIn(push), m.Create(push_line, run_time=1.6)),
            m.AnimationGroup(m.FadeIn(late), m.FadeIn(late_l, shift=0.1 * m.UP)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 8 · the catch

    def the_catch(self):
        self.next_section("8 · The catch", notes=(
            "Classifier guidance needs a second network trained on noisy images at every noise level; on ImageNet "
            f"256 × 256 ADM's classifier cost {ADM_CLF_DAYS} V100-days on top of {ADM_GEN_DAYS} for the diffusion model "
            "(Dhariwal & Nichol 2021, Table 10; notes §3, §13). Ho & Salimans (2022, arXiv:2207.12598, §1): "
            "'this classifier must be trained on noisy data so it is generally not possible to plug in a pre-trained "
            "classifier' and 'classifier-guided diffusion sampling can be interpreted as attempting to confuse an image "
            "classifier with a gradient-based adversarial attack'. The rows: seed 5 at w = 1 and w = 3, DDIM 50, the "
            "same starting noise (mac_runs §4); w is the classifier-free guidance scale of Episode 14 (w = 1 = no "
            "guidance)."))
        noisy = m.Group()
        for k, i in enumerate((8, 20, 32, 44)):
            fr = framed(picture(self.frames[3][i], 0.95), EDGE).move_to([-5.45 + 1.12 * k, 1.75, 0])
            noisy.add(fr)
        n_lab = text("noisy pictures, every noise level", 20, MUTED).next_to(noisy, m.UP, buff=0.16)
        box = card(4.3, 1.5, PROMPT, PINK_FILL, radius=0.18, stroke=2.5).move_to([-3.77, -0.35, 0])
        b_txt = m.VGroup(text("a second network", 24, INK, weight=m.BOLD), text("a classifier for noisy pictures", 20, MUTED))
        b_txt.arrange(m.DOWN, buff=0.12).move_to(box)
        into = arrow([noisy.get_x(), noisy.get_bottom()[1], 0], box.get_top(), MUTED, buff=0.12)
        cost = code(f"+{ADM_CLF_DAYS} V100-days", 26, BAD, weight=m.BOLD).move_to([2.95, 1.85, 0])
        cost_l = cite(f"on top of {ADM_GEN_DAYS} for the diffusion model\nImageNet · Dhariwal & Nichol, 2021, Table 10")
        cost_l.next_to(cost, m.DOWN, buff=0.16)
        q_txt = m.VGroup(*[text(s, 22, INK) for s in (
            "“classifier-guided diffusion sampling",
            "can be interpreted as attempting to",
            "confuse an image classifier with a",
            "gradient-based adversarial attack”")]).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        q_src = cite("Ho & Salimans, 2022")
        q_in = m.VGroup(q_txt, q_src).arrange(m.DOWN, buff=0.22, aligned_edge=m.LEFT)
        q_box = card(q_in.width + 0.7, q_in.height + 0.6, EDGE, CARD, radius=0.16, stroke=2)
        quote = m.VGroup(q_box, q_in.move_to(q_box)).move_to([2.95, -1.0, 0])
        assert quote.get_top()[1] < cost_l.get_bottom()[1] - 0.3, (quote.get_top(), cost_l.get_bottom())
        assert quote.get_right()[0] < 6.5 and quote.get_left()[0] > box.get_right()[0] + 0.3
        assert cost_l.get_bottom()[1] > -3.5

        # the knob our own model has (Episode 14)
        rows = m.Group()
        for grid in (GRID_W1, GRID_W3):
            r = m.Group(*[framed(digit(grid[MISS, d], 0.74), EDGE) for d in range(10)]).arrange(m.RIGHT, buff=0.12)
            rows.add(r)
        rows.arrange(m.DOWN, buff=0.4).move_to([0.55, -0.65, 0])
        r_labs = m.VGroup(code("w = 1", 22, MUTED), code("w = 3", 22, PROMPT, weight=m.BOLD))
        for lab, r in zip(r_labs, rows):
            lab.next_to(r, m.LEFT, buff=0.3)
        heads = m.VGroup(*[code(str(d), 20, PROMPT).next_to(rows[0][d], m.UP, buff=0.2) for d in range(10)])
        bad = m.VGroup(*[m.SurroundingRectangle(rows[0][d], buff=0.05).set_stroke(BAD, 4) for d in MISSES])
        good = m.VGroup(*[m.SurroundingRectangle(rows[1][d], buff=0.05).set_stroke(GOOD, 4) for d in MISSES])
        knob = pill("Episode 14 explains this knob", PROMPT, PINK_FILL, 22).move_to([0.55, -2.85, 0])
        k_cap = cite("one noise (seed 5) · generated on this Mac · DDIM, 50 steps").move_to([0.55, -3.38, 0])
        assert r_labs.get_left()[0] > -6.6 and heads.get_top()[1] < 1.1 and knob.get_bottom()[1] > k_cap.get_top()[1]

        self.say(
            "[The catch]: [you need a second network], [a classifier trained on noisy pictures at every noise level].",
            m.AnimationGroup(self.new_header("08 · the catch", "Why not always a classifier?")),
            m.FadeIn(box, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(b_txt), m.LaggedStart(*[m.FadeIn(n, shift=0.1 * m.DOWN) for n in noisy],
                                                            lag_ratio=0.15), m.FadeIn(n_lab), m.GrowArrow(into)),
        )
        self.say(
            "[On ImageNet, that classifier cost forty-six extra graphics-card days], "
            "[on top of nine hundred sixteen for the diffusion model].",
            m.FadeIn(cost, shift=0.1 * m.UP),
            m.FadeIn(cost_l, shift=0.1 * m.UP),
        )
        self.say(
            "[And its push can work like an adversarial attack]: [it can please a classifier], "
            "[without making a better picture].",
            m.FadeIn(quote, shift=0.1 * m.LEFT),
            m.Circumscribe(q_txt[3], color=BAD, buff=0.06),
            m.Wait(0.2),
        )
        self.say(
            "[Our model has a cheaper knob built in]. [Turned up], [the stray four and six fall into line]: "
            "[episode fourteen explains this knob].",
            m.AnimationGroup(m.FadeOut(m.Group(noisy, n_lab, box, b_txt, into, cost, cost_l, quote)),
                             m.FadeIn(rows[0], shift=0.1 * m.UP), m.FadeIn(r_labs[0]), m.FadeIn(heads),
                             m.FadeIn(k_cap), m.Create(bad)),
            m.AnimationGroup(m.FadeIn(rows[1], shift=0.1 * m.UP), m.FadeIn(r_labs[1])),
            m.Create(good),
            m.FadeIn(knob, shift=0.1 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes=(
            "Ask (the label is one more input to the denoiser), embed (a learned row of an 11 × 128 table, added to the "
            "time embedding, into every residual block), steer (classifier guidance: push along ∇ log p(y | x_t), "
            "scaled; stronger = more certain, less varied). Module 4: 13 class labels, 14 classifier-free guidance, "
            "15 text encoders and cross-attention, 16 editing and control. Next: Episode 14."))
        halo = glow(PROMPT, radius=4.0)
        words = m.VGroup(*[title_text(w, 58, c) for w, c in (("Ask", PROMPT), ("Embed", MODEL), ("Steer", SCORE))])
        subs = m.VGroup(*[code(s, 20, MUTED) for s in ("the label is an input", "a learned row, plus time",
                                                       "push toward the label")])
        words.arrange(m.RIGHT, buff=2.0)
        on_baseline(words)
        for w, s in zip(words, subs):
            s.set_x(w.get_x())
        line_y = min(w.get_bottom()[1] for w in words) - 0.45
        for s in subs:
            sit(s, line_y)
        mid = baseline(words[0]) + 0.3
        dots = m.VGroup(*[m.Dot([(a.get_right()[0] + b.get_left()[0]) / 2, mid, 0], radius=0.07, color=MUTED)
                          for a, b in zip(words, words[1:])])
        row = m.VGroup(words, dots, subs).move_to(1.55 * m.UP)
        assert row.width < 12.6 and all(a.get_right()[0] + 0.3 < b.get_left()[0] for a, b in zip(subs, subs[1:]))
        strip_ = m.Group(*[framed(digit(GRID_W1[HERO, d], 0.66), EDGE) for d in range(10)]).arrange(m.RIGHT, buff=0.1)
        strip_.move_to([0, -0.15, 0])
        noise_t = framed(picture(tint(np.clip(NOISES[HERO], -1, 1), NOISE, -1, 1), 0.66), NOISE)
        noise_t.next_to(strip_, m.LEFT, buff=0.55)
        to_s = arrow(noise_t.get_right(), strip_.get_left(), MUTED, buff=0.1)
        whole = m.Group(noise_t, to_s, strip_)
        whole.move_to([0, -0.15, 0])
        mod = m.VGroup(*[pill(s, PROMPT, PINK_FILL if i == 0 else CARD, 20) for i, s in enumerate(
            ("13 · labels", "14 · free guidance", "15 · words", "16 · editing"))])
        mod.arrange(m.RIGHT, buff=0.25).move_to([0, -1.65, 0])
        for p_ in mod[1:]:
            p_[1].set_color(MUTED)
        mascot = pixel(0.42)
        nxt = pill("NEXT · CLASSIFIER-FREE GUIDANCE", PROMPT, PINK_FILL, 20, mono=True)
        foot = m.VGroup(mascot, nxt).arrange(m.RIGHT, buff=0.3).move_to([0, -2.75, 0])
        assert whole.width < 12.6 and mod.width < 12.6

        self.say(
            "[To sum up]: [ask], [embed], [steer].",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.3 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(dots[0]), m.FadeIn(words[1], shift=0.3 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(dots[1]), m.FadeIn(words[2], shift=0.3 * m.UP), m.FadeIn(subs[2])),
        )
        self.say(
            "[One noise, ten labels, ten digits]: [the condition decides what the denoiser makes].",
            m.AnimationGroup(m.FadeIn(noise_t, shift=0.1 * m.RIGHT), m.GrowArrow(to_s),
                             m.LaggedStart(*[m.FadeIn(f, shift=0.1 * m.UP) for f in strip_], lag_ratio=0.08)),
            m.Circumscribe(words[0], color=PROMPT, buff=0.1),
        )
        self.say(
            "[This opens module four]: [from labels], [to guidance], [to whole sentences], [to editing].",
            m.FadeIn(mod[0], shift=0.1 * m.UP),
            m.Indicate(mod[0], color=PROMPT, scale_factor=1.08),
            m.FadeIn(mod[1], shift=0.1 * m.UP),
            m.FadeIn(mod[2], shift=0.1 * m.UP),
            m.FadeIn(mod[3], shift=0.1 * m.UP),
        )
        self.say(
            "[Next, we drop the classifier], [and let one model guide itself].",
            m.FadeIn(mascot, shift=0.2 * m.UP),
            m.FadeIn(nxt, shift=0.2 * m.UP),
        )
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, whole, mod, foot)], run_time=0.9)
