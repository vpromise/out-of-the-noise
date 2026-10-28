"""Out of the Noise · Episode 19 · Measuring Pictures (Module 5 · Scaling Up)

Judging generators is hard; every metric has a blind spot. Every earlier episode quoted "cFID": this one explains it.
Running example: the digit model's FID-like score ("cFID", research/experiments/mac_runs.md §3–4), measured against
steps (Episode 9) and guidance (Episode 14); the swirl (diffusion_reference + toy.md §7) is the 2D stand-in for the
features, where every number is computed exactly.

The story:
- FID (Heusel et al. 2017): features of real and generated pictures (Inception-v3, 2,048 numbers each), a Gaussian
  (mean, covariance) fitted to each cloud, the Fréchet distance between the two Gaussians:
  ‖μ_r − μ_g‖² + Tr(Σ_r + Σ_g − 2(Σ_r·Σ_g)^½) — diffusion_reference.frechet_distance, shown live on N(0, I) vs
  N((1, 0), diag(2, 0.5)) (notes §19's worked example: 1.257).
- The blind spot (toy.md §7, recomputed here exactly with the experiment's own random stream): a Gaussian with the
  swirl's mean and covariance scores 0.0003315 against 10,000 swirl points, yet 48% of its points are off the swirl.
- Small-sample bias (toy.md §7, recomputed exactly): identical distributions score 0.03698 (2-D, n = 100), 0.0003503
  (2-D, n = 10,000), 1.119 (64-D, n = 1,000).
- Our cFID: the same formula on the 128-d penultimate features of our own MNIST classifier (99.45% test accuracy),
  10,000 generated vs the 10,000 test digits — NOT Inception FID, not comparable with published numbers. Floor 2.06
  (2.10 ± 0.10 over five subsets), bootstrap std 0.355 at DDIM 50 ("differences under about 0.7 are not meaningful").
  Steps: DDPM-1000 2.64; DDIM 100/50/20/10/5: 3.50/3.35/3.30/4.57/21.6. Guidance (DDIM 50): 3.35/21.4/42.0/66.9/85.4
  at w = 1/2/3/5/8. Precision/recall (Kynkäänniemi et al. 2019, k = 3) at w = 1/3/8: 0.83/0.89, 0.89/0.70,
  0.87/0.38; label accuracy 98.10% → 99.99%.
- CLIP score (Hessel et al. 2021): 2.5·max(cos, 0) — prompt following, not quality; Imagen: more guidance improves
  alignment but damages fidelity. Human preference studies and learned preference models (PickScore, ImageReward,
  HPS v2; 2023). Recap: Statistics · Samples · People.

Every number on screen is computed below (diffusion_reference, measured(...), the run's arrays) and asserted, or cited
with its source. Illustrative ("for example"): the 2-D feature clouds of section 1 (two of the 2,048 numbers, drawn as
Gaussians), the precision/recall toy generators (one arm; extra noise), the CLIP angle, the vote split.

    VOICE=draft manimgx inspect scene.py      # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403
from diffusion_reference import *  # noqa: E402,F403

EP_DIR = Path(__file__).resolve().parent  # set AFTER the import: diffusion_reference exports HERE (the series folder)
warnings.filterwarnings("ignore", message=".*encountered in matmul")  # NumPy/Accelerate's spurious warnings

ARR = EXPERIMENTS / "arrays"
IMG = EXPERIMENTS / "images"
H = lambda *keys: measured("mac_runs", "headline", *keys)  # noqa: E731
MD = (EXPERIMENTS / "mac_runs.md").read_text()
TOY_MD = (EXPERIMENTS / "toy.md").read_text()
NOTES = (SERIES / "research" / "notes.md").read_text()
REF_SRC = (SERIES / "diffusion_reference.py").read_text()

REAL, GEN = DATA, MODEL          # real pictures TEAL, generated pictures AMBER (the course's colours)
SIG = lambda v, d=4: float(f"{float(v):.{d}g}")  # noqa: E731  (the experiment script's rounding)

# ---------------------------------------------------------------------------- 1 · FID's formula (notes §19 worked example)

MU_R, COV_R = np.zeros(2), np.eye(2)
MU_G, COV_G = np.array([1.0, 0.0]), np.diag([2.0, 0.5])
FD_SHIFT = frechet_distance(MU_R, COV_R, MU_G, COV_R)
FD_BOTH = frechet_distance(MU_R, COV_R, MU_G, COV_G)
assert abs(FD_SHIFT - 1.0) < 1e-12 and f"{FD_BOTH:.3f}" == "1.257" and frechet_distance(MU_R, COV_R, MU_R, COV_R) == 0
assert "FID = 1 + 4.5 − 2·2.121 = **1.257**" in NOTES
_closed = 1.0 + 2 + 2.5 - 2 * (np.sqrt(2) + np.sqrt(0.5))                       # ‖Δμ‖² + Tr C_r + Tr C_g − 2 Tr √(C_r C_g)
assert abs(_closed - FD_BOTH) < 1e-12
assert "‖μ1 − μ2‖² + Tr(C1) + Tr(C2) − 2·Tr((C1·C2)^½)" in REF_SRC
assert "the 2,048-d activations of the last pooling layer" in NOTES and "we generate 50,000 images" in NOTES
FEAT_Z = np.random.default_rng(19).standard_normal((900, 2))
FEAT_Z = FEAT_Z[np.all(np.abs(FEAT_Z) < 2.3, axis=1)][:240]                    # two clouds of 240 dots, inside the plane
FEAT_R = np.random.default_rng(20).standard_normal((900, 2))
FEAT_R = FEAT_R[np.all(np.abs(FEAT_R) < 2.3, axis=1)][:240]
assert len(FEAT_Z) == len(FEAT_R) == 240

# ---------------------------------------------------------------------------- 2–3 · the swirl's blind spot and the bias (toy.md §7)
# The experiment's own random stream, replayed exactly (scripts/toy_experiments.py, fid()): rng(21) feeds the bias
# table (2-D: 200 repeats, 64-D: 40, n = 100 / 1,000 / 10,000, two draws per repeat), then the matched Gaussian.

_rng = np.random.default_rng(21)
BIAS = {}
for _d, _reps in ((2, 200), (64, 40)):
    for _n in (100, 1000, 10000):
        _vals = []
        for _ in range(_reps):
            _s1 = _rng.standard_normal((_n, _d))
            _rng.standard_normal((_n, _d))                                          # the second draw (two-sample column)
            _vals.append(frechet_distance(*gaussian_stats(_s1), np.zeros(_d), np.eye(_d)))
        BIAS[(_d, _n)] = SIG(np.mean(_vals))
SW_DATA = swirl(10000, seed=31)
SW_MU, SW_COV = gaussian_stats(SW_DATA)
SW_GAUSS = _rng.multivariate_normal(SW_MU, SW_COV, 10000)
FD_BLOB = fd_from_samples(SW_DATA, SW_GAUSS)
FD_REAL2 = fd_from_samples(SW_DATA, swirl(10000, seed=32))
OFF_BLOB, OFF_REAL = off_swirl(SW_GAUSS), off_swirl(SW_DATA)
_fj = lambda *k: measured("toy", "fid", *k)  # noqa: E731
for (_d, _n), _v in BIAS.items():
    assert _v == _fj("bias", f"d{_d}", "rows", str(_n), "vs_truth_mean"), (_d, _n, _v)
assert SIG(FD_BLOB) == _fj("blind_spot", "fd_swirl_vs_matched_gaussian") == measured("toy", "headline", "fd_swirl_vs_matched_gaussian")
assert SIG(FD_REAL2) == _fj("blind_spot", "fd_swirl_vs_swirl") and SIG(OFF_BLOB) == _fj("blind_spot", "gaussian_quality", "off")
assert np.abs(np.round(SW_GAUSS[:1000], 3) - np.array(_fj("blind_spot", "gaussian_cloud"))).max() == 0
assert BIAS[(2, 100)] == measured("toy", "headline", "fd_bias_2d_n100") and BIAS[(64, 1000)] == measured("toy", "headline", "fd_bias_64d_n1000")
assert BIAS[(2, 10000)] == measured("toy", "headline", "fd_bias_2d_n10000")
BLOB_S, BLOB_OFF = f"{FD_BLOB:.4f}", f"{100 * OFF_BLOB:.0f}%"
assert (BLOB_S, BLOB_OFF, f"{FD_REAL2:.5f}") == ("0.0003", "48%", "0.00002") and OFF_REAL < 0.002
B_TXT = {(2, 100): "0.037", (2, 1000): "0.0036", (2, 10000): "0.00035", (64, 100): "12.3", (64, 1000): "1.12", (64, 10000): "0.11"}
for _k, _s in B_TXT.items():
    assert abs(float(_s) - BIAS[_k]) / BIAS[_k] < 0.02, (_k, _s)
assert "In 64 dimensions\nthe bias is 1.119 even with 1,000 samples" in TOY_MD and "48 % of the Gaussian's points are off the swirl" in TOY_MD
# one draw of 100 points for the picture (chosen: its own score is near the 200-draw mean)
_draws = {s: np.random.default_rng(s).standard_normal((100, 2)) for s in range(40)}
_fd = {s: frechet_distance(*gaussian_stats(x), np.zeros(2), np.eye(2)) for s, x in _draws.items()}
DRAW_SEED = min(_fd, key=lambda s: abs(_fd[s] - BIAS[(2, 100)]))
DRAW100 = _draws[DRAW_SEED]
DRAW10K = np.random.default_rng(100).standard_normal((10000, 2))
assert abs(_fd[DRAW_SEED] - BIAS[(2, 100)]) < 0.005, _fd[DRAW_SEED]

# ---------------------------------------------------------------------------- 4–6 · our digit score (mac_runs.md §3–4)

CLS = measured("mac_runs", "classifier")
assert CLS["feature_dim"] == 128 and CLS["test_accuracy"] == H("classifier_test_accuracy") == 0.9945 and CLS["params"] == 467_818
assert "fc128(features)" in CLS["arch"] and "Its 128-d penultimate features are used for" in MD and "**This is not Inception FID.**" in MD
CLS_S = f"{CLS['wall_s']:.0f} s"
assert CLS_S == "14 s"
CFG = measured("mac_runs", "digit_sampling", "configs")
REFR = measured("mac_runs", "digit_sampling", "real_reference")
FLOOR = REFR["cfid"]
UNC = measured("mac_runs", "cfid_uncertainty")
SUBSETS = UNC["real_floor_5_subsets"]["values"]
assert abs(FLOOR - H("cfid_real_data_floor")) < 1e-3 and f"{FLOOR:.2f}" == "2.06" and abs(SUBSETS[0] - FLOOR) < 1e-12
SUB_STD = UNC["real_floor_5_subsets"]["std"]
BOOT = {k: UNC[f"ddpm_{k}_w1"]["bootstrap_std"] for k in ("ddpm1000", "ddim100", "ddim50", "ddim20")}
BOOT_S = f"{BOOT['ddim50']:.2f}"
assert f"{SUB_STD:.2f}" == "0.10" and BOOT_S == "0.35" and abs(BOOT["ddim50"] - H("cfid_bootstrap_std_ddim50")) < 1e-3
assert f"{max(BOOT.values()):.2f}" == "0.36" and "bootstrap std of cFID ≈ **0.3** (0.27–0.36)" in MD
assert (f"{min(SUBSETS):.2f}", f"{max(SUBSETS):.2f}") == ("1.99", "2.28") and "Differences under about 0.7 are not meaningful" in MD
STEPS = (5, 10, 20, 50, 100)
SC = {n: CFG[f"ddpm_ddim{n}_w1"] for n in STEPS}
SC[1000] = CFG["ddpm_ddpm1000_w1"]
for _n in SC:
    assert SC[_n]["steps"] == _n and SC[_n]["w"] == 1
    assert abs(SC[_n]["cfid"] - H("cfid_ddpm1000" if _n == 1000 else f"cfid_ddim{_n}")) < 1e-3
S_TXT = {n: (f"{SC[n]['cfid']:.2f}" if SC[n]["cfid"] < 10 else f"{SC[n]['cfid']:.1f}") for n in SC}
assert [S_TXT[n] for n in (5, 10, 20, 50, 100, 1000)] == ["21.6", "4.57", "3.30", "3.35", "3.50", "2.64"]
assert max(SC[n]["cfid"] for n in (20, 50, 100)) - min(SC[n]["cfid"] for n in (20, 50, 100)) < min(BOOT[f"ddim{n}"] for n in (20, 50, 100))
WS = (1, 2, 3, 5, 8)
GC = {w: CFG[f"ddpm_ddim50_w{w}"] for w in WS}
for _w in WS:
    assert GC[_w]["w"] == _w and GC[_w]["steps"] == 50 and abs(GC[_w]["cfid"] - H(f"guidance_w{_w}_cfid")) < 1e-3
G_TXT = {w: (f"{GC[w]['cfid']:.2f}" if GC[w]["cfid"] < 10 else f"{GC[w]['cfid']:.1f}") for w in WS}
assert [G_TXT[w] for w in WS] == ["3.35", "21.4", "42.0", "66.9", "85.4"]
PR = {w: (GC[w]["precision"], GC[w]["recall"]) for w in (1, 3, 8)}
for _w in (1, 3, 8):
    assert abs(PR[_w][0] - H(f"guidance_w{_w}_precision")) < 1e-3 and abs(PR[_w][1] - H(f"guidance_w{_w}_recall")) < 1e-3
PR_TXT = {w: (f"{PR[w][0]:.2f}", f"{PR[w][1]:.2f}") for w in PR}
assert PR_TXT == {1: ("0.83", "0.89"), 3: ("0.89", "0.70"), 8: ("0.87", "0.38")}
ACC = {w: f"{100 * GC[w]['cond_acc']:.2f}%" for w in (1, 3, 8)}
assert ACC == {1: "98.10%", 3: "100.00%", 8: "99.99%"}
REAL_PR = (f"{REFR['precision']:.2f}", f"{REFR['recall']:.2f}")
assert REAL_PR == ("0.87", "0.89")
DIT = H("cfid_dit_ddim50")
assert f"{DIT:.1f}" == "15.9"


def samples(key):
    f = np.load(ARR / f"digit_samples_{key}.npz")
    assert "initial noise seed 1234" in str(f["note"]) and np.all(f["labels"][:20] == np.arange(20) % 10)
    return f["samples"]


DIG = {key: samples(f"ddpm_{key}")[10:20] for key in ("ddim50_w1", "ddim50_w3", "ddim50_w8", "ddim5_w1", "ddim10_w1",
                                                      "ddpm1000_w1")}
TEST_X, TEST_Y = load_mnist("test", limit=200)
REAL_DIGITS = [TEST_X[i] for i in range(10)]
SEVENS = [TEST_X[i] for i in np.flatnonzero(TEST_Y == 7)[:5]]
assert len(SEVENS) == 5

# ---------------------------------------------------------------------------- 6 · precision and recall on the swirl (for example)


def knn_radii(points, k=3):
    """Distance from each point to its k-th nearest neighbour in the same set (Kynkäänniemi et al.'s balls, k = 3)."""
    d = np.sqrt(((points[:, None] - points[None]) ** 2).sum(-1))
    return np.sort(d, 1)[:, k]


def inside(points, centres, radii):
    d = np.sqrt(((points[:, None] - centres[None]) ** 2).sum(-1))
    return (d <= radii[None]).any(1)


PR_N = 200
PR_REAL = swirl(PR_N, seed=1)
_x, _lab = swirl_labelled(3 * PR_N, seed=2)
PR_GEN = {"narrow": _x[_lab == 0][:PR_N], "sloppy": swirl(PR_N, seed=3, noise=0.12)}
R_REAL = knn_radii(PR_REAL)
PR_TOY = {}
for _k, _g in PR_GEN.items():
    _in_real = inside(_g, PR_REAL, R_REAL)                                         # precision: generated inside real balls
    _rg = knn_radii(_g)
    _cov = inside(PR_REAL, _g, _rg)                                                # recall: real inside generated balls
    PR_TOY[_k] = dict(inside=_in_real, radii=_rg, covered=_cov, p=float(_in_real.mean()), r=float(_cov.mean()))
PR_TOY_TXT = {k: (f"{v['p']:.2f}", f"{v['r']:.2f}") for k, v in PR_TOY.items()}
assert PR_TOY_TXT == {"narrow": ("0.98", "0.49"), "sloppy": ("0.64", "0.99")}, PR_TOY_TXT
assert '"k = 3 is a robust choice"' in NOTES

# ---------------------------------------------------------------------------- 7 · prompts and people (notes §19, cited)

PROMPT_TEXT = "a lighthouse on a rocky coast at dusk, oil painting"
assert f"Prompt: **\"{PROMPT_TEXT}\"**" in MD
LH = IMG / "lighthouse"
LH_A, LH_B = LH / "lighthouse_sd21_cfg7.5_steps50_seed0.png", LH / "lighthouse_sd21_cfg7.5_steps50_seed2.png"
assert LH_A.exists() and LH_B.exists()
assert "CLIP-S(c, v) = w·max(cos(c, v), 0), w = 2.5" in NOTES
PREF = (("PickScore", "Pick-a-Pic: 583,747 choices", "Kirstain et al., 2023"),
        ("ImageReward", "137k expert comparisons", "Xu et al., 2023"),
        ("HPS v2", "798,090 choices", "Wu et al., 2023"))
assert "(training set version: 583,747 examples, 37,523 prompts, 4,375 users)" in NOTES
assert "\"collects 137k expert comparisons to date\"" in NOTES and "HPD v2 comprises 798,090 human preference choices" in NOTES
assert "\"DrawBench comprises 200 prompts in total\"" in NOTES
assert "the improvements of SDXL over previous methods are not reflected" in NOTES
# Imagen §2.3 (Saharia et al. 2022, arXiv:2205.11487, read via ar5iv, October 2026):
IMAGEN_QUOTE = "increasing the classifier-free guidance weight\nimproves image-text alignment,\nbut damages image fidelity"

# ---------------------------------------------------------------------------- look

FLAT = set("abcdefhiklmnorstuvwxzABCDEFGHIJKLMNOPRSTUVWXYZ0123456789")


def cite(s, size=20, color=MUTED):
    return text(s, size, color)


def baseline(words):
    """The baseline of a Text: the median bottom of its flat-bottomed letters."""
    chars = [c for c in words.text if not c.isspace()]
    if len(words) == len(chars):
        bottoms = sorted(g.get_bottom()[1] for g, c in zip(words, chars) if c in FLAT)
        if bottoms:
            return bottoms[len(bottoms) // 2]
    return float(np.median([g.get_bottom()[1] for g in words]))


def sit(words, y):
    return words.shift((y - baseline(words)) * m.UP)


def digit_row(images, color, height, gap_px=6, k=8):
    """Several 28×28 digits (uint8) as ONE picture with dark gaps, `height` units tall."""
    n = len(images)
    canvas = np.zeros((28, n * 28 + (n - 1) * gap_px))
    for i, img in enumerate(images):
        canvas[:, i * (28 + gap_px): i * (28 + gap_px) + 28] = np.asarray(img, dtype=float) / 255
    rgb = np.repeat(np.repeat(tint(canvas, color, 0, 1), k, axis=0), k, axis=1)
    return picture(rgb, height, pixelated=False)


def digit_col(images, color, height, gap_px=6, k=8):
    """Digits stacked vertically as one picture; `height` is ONE digit's height."""
    n = len(images)
    canvas = np.zeros((n * 28 + (n - 1) * gap_px, 28))
    for i, img in enumerate(images):
        canvas[i * (28 + gap_px): i * (28 + gap_px) + 28] = np.asarray(img, dtype=float) / 255
    rgb = np.repeat(np.repeat(tint(canvas, color, 0, 1), k, axis=0), k, axis=1)
    return picture(rgb, height * canvas.shape[0] / 28, pixelated=False)


class Readout(m.VGroup):
    """A mono number display, left-aligned at a fixed point; .show(v) updates it."""

    def __init__(self, left, y, fmt, value, size=40, color=INK):
        super().__init__()
        self.anchor, self.fmt, self.size, self.color = np.array([left, y, 0.0]), fmt, size, color
        self.value = value
        self.words = self._make(value)
        self.add(self.words)

    def _make(self, value):
        return code(self.fmt.format(value), self.size, self.color, weight=m.BOLD).move_to(self.anchor, aligned_edge=m.LEFT)

    def show(self, value):
        if value != self.value:
            self.value = value
            self.words.become(self._make(value))
        return self


def boxed(group, color=EDGE, buff=0.3, radius=0.16, fill=CARD):
    box = card(group.width + 2 * buff, group.height + 2 * buff, color, fill, radius=radius, stroke=2).move_to(group)
    return m.VGroup(box, group)


def ellipse_xy(mu, cov, k=2.0, n=96):
    """Points of the k-standard-deviation ellipse of N(mu, cov) (a closed loop, data coordinates)."""
    lam, vec = np.linalg.eigh(np.asarray(cov, dtype=float))
    a = np.linspace(0, 2 * np.pi, n)
    return np.asarray(mu, dtype=float) + (k * np.stack([np.cos(a), np.sin(a)], 1) * np.sqrt(np.clip(lam, 0, None))) @ vec.T


def ellipse(plot, mu, cov, color, width=3.5, k=2.0, opacity=1.0):
    pts = [plot.coords(float(x), float(y)) for x, y in ellipse_xy(mu, cov, k)]
    return m.VMobject().set_points_as_corners(pts).set_stroke(color, width, opacity=opacity).set_fill(opacity=0)


def cross(p, color, r=0.13, width=4):
    p = np.asarray(p, dtype=float)
    return m.VGroup(m.Line(p + [-r, 0, 0], p + [r, 0, 0]), m.Line(p + [0, -r, 0], p + [0, r, 0])).set_stroke(color, width)


def plane(center, width, height, xr, yr):
    return Plot(xr, yr, width=width, height=height, center=(center[0], center[1], 0), axes=False)


def dots(plot, xy, color, radius=0.03, opacity=0.9):
    return m.VGroup(*[m.Dot(plot.coords(float(x), float(y)), radius=radius, color=color).set_opacity(opacity) for x, y in xy])


def swirl_band(plot, color=REAL, width=10, opacity=0.13):
    band = m.VGroup()
    for arm in swirl_curve(200):
        band.add(m.VMobject().set_points_smoothly([plot.coords(*p) for p in arm[::3]] + [plot.coords(*arm[-1])])
                 .set_stroke(color, width, opacity=opacity))
    return band.set_z_index(-1)


def arrow_to(a, b, color=MUTED, width=4, buff=0.1, tip=0.2):
    return m.Arrow(a, b, buff=buff, color=color, stroke_width=width, max_tip_length_to_length_ratio=tip)


def label_pair(name, value, color, size=24):
    """A muted name and a bold mono value, on one baseline."""
    n_ = text(name, size, MUTED)
    v_ = code(value, size, color, weight=m.BOLD)
    row = m.VGroup(n_, v_).arrange(m.RIGHT, buff=0.2)
    sit(v_, baseline(n_))
    return row


def recolour(group, idx, start, end, seconds=1.0):
    """Turn the dots group[i] for i in idx from colour `start` to `end` (one animation on the whole group)."""
    idx = list(idx)

    def step(_, a):
        col = mix(start, end, a)
        for i in idx:
            group[i].set_color(col)
    return m.UpdateFromAlphaFunc(group, step, run_time=seconds)


# ---------------------------------------------------------------------------- the episode


class Ep19MeasuringPictures(Episode):
    number = 19

    def lesson(self):
        self.hook()
        self.two_clouds()
        self.blind_spot()
        self.sample_size()
        self.our_score()
        self.against_steps()
        self.precision_recall()
        self.prompts_and_people()
        self.recap()

    # ------------------------------------------------------------------ 0 · hook

    def hook(self):
        self.next_section("Hook", notes=(
            "Generated on this Mac (mac_runs.md §4): the digit model (3.94 M-parameter U-Net), DDIM 50 steps, samples 10–19 "
            "(they ask for 0 … 9), the same starting noise (seed 1234) in both rows. Batch A: guidance w = 1 (no "
            f"guidance), cFID {G_TXT[1]}; batch B: w = 8, cFID {G_TXT[8]}. cFID = FID's formula on our own digit "
            f"classifier's features (floor {FLOOR:.2f}), explained in this episode. Earlier quotes: Episode 9 (DDPM 1,000 "
            f"steps {S_TXT[1000]}), Episode 14 (w = 8: {G_TXT[8]}), Episode 18 (tiny DiT, DDIM 50: {DIT:.1f}). The three "
            "goals (fidelity, diversity, alignment): notes §19, 'There is no single accuracy for pictures'. The real "
            "digits on the cards are MNIST test digits."))
        hgt = 0.7
        cx = -0.35
        rows = {}
        for key, y, name in (("ddim50_w1", 1.15, "batch A"), ("ddim50_w8", -0.35, "batch B")):
            pic = digit_row(DIG[key], GEN, hgt).move_to([cx, y, 0])
            frame = m.SurroundingRectangle(pic, buff=0.1).set_stroke(EDGE, 2)
            lab = code(name, 24, INK, weight=m.BOLD).next_to(frame, m.LEFT, buff=0.3)
            rows[key] = m.Group(pic, frame, lab)
        cap = cite("generated on this Mac · the digit model · 50 steps · the same starting noise in both rows").move_to([0, -1.45, 0])
        q = title_text("Which batch is better?", 40, INK).move_to([0, 2.2, 0])
        score_a = label_pair("score", G_TXT[1], GREEN).next_to(rows["ddim50_w1"][1], m.RIGHT, buff=0.35)
        score_b = label_pair("score", G_TXT[8], RED).next_to(rows["ddim50_w8"][1], m.RIGHT, buff=0.35)
        bold = text("bolder, cleaner strokes", 22, GEN, weight=m.BOLD).next_to(rows["ddim50_w8"][1], m.DOWN, buff=0.12)
        bold.align_to(rows["ddim50_w8"][1], m.RIGHT)
        assert score_a.get_right()[0] < 6.6 and rows["ddim50_w1"][2].get_left()[0] > -6.6, (score_a.get_right(), rows["ddim50_w1"][2].get_left())
        assert bold.get_bottom()[1] > cap.get_top()[1] + 0.05, (bold.get_bottom(), cap.get_top())

        quotes = (("Episode 9", "1,000 steps", S_TXT[1000]), ("Episode 14", "guidance 8", G_TXT[8]),
                  ("Episode 18", "a tiny DiT", f"{DIT:.1f}"), ("real digits", "the floor", f"{FLOOR:.2f}"))
        chips = m.VGroup()
        for i, (ep, what, val) in enumerate(quotes):
            body = m.VGroup(text(ep, 20, MUTED), text(what, 20, INK), code(val, 26, REAL if i == 3 else INK, weight=m.BOLD))
            body.arrange(m.DOWN, buff=0.07)
            box = card(2.3, body.height + 0.32, REAL if i == 3 else EDGE, CARD, radius=0.14, stroke=2).move_to(body)
            chips.add(m.VGroup(box, body))
        chips.arrange(m.RIGHT, buff=0.3).move_to([-0.55, -2.6, 0])
        mascot = pixel(0.55).next_to(chips, m.RIGHT, buff=0.45)
        assert chips.get_bottom()[1] > -3.6 and mascot.get_right()[0] < 6.6 and chips.get_top()[1] < cap.get_bottom()[1] - 0.08, (chips.get_bottom(), mascot.get_right(), chips.get_top())

        # the three goals
        tiles = m.Group()
        goal = (("look real", "fidelity", REAL), ("cover the variety", "diversity", REAL), ("follow the prompt", "alignment", PROMPT))
        visuals = [picture(tint(REAL_DIGITS[0], REAL, 0, 255), 1.05),
                   digit_row(SEVENS, REAL, 0.62, gap_px=5),
                   pill("“a lighthouse at dusk”", PROMPT, PINK_FILL, 22)]
        for (title, sub, color), vis, x in zip(goal, visuals, (-4.45, 0.0, 4.45)):
            box = card(3.95, 2.75, color, CARD, stroke=2.5).move_to([x, 0.35, 0])
            t_ = text(title, 28, INK, weight=m.BOLD).move_to([x, -0.1, 0])
            s_ = code(sub, 22, color).move_to([x, -0.6, 0])
            vis.move_to([x, 0.95, 0])
            tiles.add(m.Group(box, vis, t_, s_))
        part = text("Every metric sees only part of it.", 30, INK, weight=m.BOLD).move_to([-0.4, -1.95, 0])
        mascot2 = pixel(0.5).next_to(part, m.RIGHT, buff=0.4)
        g_cite = cite("notes §19: there is no single accuracy for pictures").move_to([0, -3.3, 0])
        assert tiles[0][0].get_left()[0] > -6.6 and tiles[0][0].get_top()[1] < 2.6

        self.say(
            "[Two batches of digits from the same model, generated on this Mac]. [Which batch is better]?",
            m.AnimationGroup(self.new_header("00 · hook", "Two batches, one score"),
                             m.FadeIn(rows["ddim50_w1"], shift=0.1 * m.RIGHT), m.FadeIn(rows["ddim50_w8"], shift=0.1 * m.RIGHT),
                             m.FadeIn(cap)),
            m.Write(q),
        )
        self.say(
            "[The bold ones look cleaner], [yet our score rates them far worse]: [eighty-five against three point three five].",
            m.FadeIn(bold, shift=0.1 * m.UP),
            m.FadeIn(score_b, shift=0.1 * m.LEFT),
            m.AnimationGroup(m.FadeIn(score_a, shift=0.1 * m.LEFT), m.Circumscribe(score_b, color=RED, buff=0.08)),
        )
        self.say(
            "[Every episode so far has quoted that score]. [Today: what it measures], [and what it misses].",
            m.LaggedStart(*[m.FadeIn(c_, shift=0.1 * m.UP) for c_ in chips], lag_ratio=0.25, run_time=1.6),
            m.Circumscribe(chips[3], color=REAL, buff=0.06),
            m.FadeIn(mascot, shift=0.2 * m.UP),
        )
        self.wait(0.3)
        screen = m.Group(rows["ddim50_w1"], rows["ddim50_w8"], cap, q, score_a, score_b, bold, chips, mascot)
        self.say(
            "[A picture has no single right answer]: [we want samples that look real], [that cover the variety], "
            "[and that follow the prompt].",
            m.FadeOut(screen),
            m.FadeIn(tiles[0], shift=0.15 * m.UP),
            m.FadeIn(tiles[1], shift=0.15 * m.UP),
            m.FadeIn(tiles[2], shift=0.15 * m.UP),
        )
        self.say(
            "[Every metric sees only part of that]: [today, we find each one's blind spot].",
            m.Write(part),
            m.AnimationGroup(m.FadeIn(mascot2, shift=0.2 * m.UP), m.FadeIn(g_cite)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 1 · two clouds

    def two_clouds(self):
        self.next_section("1 · Features, then two Gaussians", notes=(
            "FID (Heusel et al., arXiv:1706.08500, Jun 2017; NeurIPS 2017): Inception-v3's last pooling layer (2,048 numbers "
            "per picture; the '2,048' is standard Inception-v3 knowledge, notes Appendix B), a Gaussian fitted to each set, "
            "the Fréchet (Wasserstein-2) distance between them: ‖μ_r − μ_g‖² + Tr(Σ_r + Σ_g − 2(Σ_r·Σ_g)^½) (notes §19, eq. 6). "
            "The paper used 50,000 generated images. The 2-D clouds are ILLUSTRATIVE (two of the 2,048 numbers, drawn as "
            "Gaussians, 240 dots each). The live score is diffusion_reference.frechet_distance on the exact Gaussians: "
            f"N(0, I) vs N(0, I) = 0; vs N((1, 0), I) = {FD_SHIFT:.3f}; vs N((1, 0), diag(2, 0.5)) = {FD_BOTH:.4f} (notes §19 "
            "worked example: 1 + 4.5 − 2·2.121 = 1.257). Ellipses: two standard deviations."))
        real_col = digit_col(REAL_DIGITS[:3], REAL, 0.6, gap_px=5).move_to([-5.7, 1.0, 0])
        gen_col = digit_col(DIG["ddim50_w1"][4:7], GEN, 0.6, gap_px=5).move_to([-5.7, -1.55, 0])
        real_l = code("real", 22, REAL, weight=m.BOLD).next_to(real_col, m.UP, buff=0.12)
        gen_l = code("generated", 22, GEN, weight=m.BOLD).next_to(gen_col, m.DOWN, buff=0.12)
        net_t = m.VGroup(text("image classifier", 24, INK, weight=m.BOLD), code("Inception-v3", 20, MUTED),
                         cite("trained on ImageNet")).arrange(m.DOWN, buff=0.08)
        net_box = card(net_t.width + 0.5, net_t.height + 0.45, INK, CARD_HI)
        net = m.VGroup(net_box, net_t.move_to(net_box)).move_to([-3.05, -0.28, 0])
        a_r = arrow_to(real_col.get_right() + [0.08, 0, 0], net_box.get_left() + [0, 0.3, 0], REAL)
        a_g = arrow_to(gen_col.get_right() + [0.08, 0, 0], net_box.get_left() + [0, -0.3, 0], GEN)
        vec_t = code("0.31\n1.72\n0.05\n ⋮\n0.88", 20, INK)
        vec_box = m.RoundedRectangle(corner_radius=0.1, width=0.95, height=vec_t.height + 0.35).set_stroke(MUTED, 2).set_fill(CARD, 1)
        vec = m.VGroup(vec_box, vec_t.move_to(vec_box)).move_to([-0.7, -0.28, 0])
        vec_l = m.VGroup(code("2,048", 24, INK, weight=m.BOLD), cite("numbers")).arrange(m.DOWN, buff=0.06).next_to(vec, m.DOWN, buff=0.15)
        a_v = arrow_to(net_box.get_right(), vec_box.get_left(), MUTED, buff=0.12)
        src = cite("Heusel et al., 2017").next_to(net, m.UP, buff=0.2)

        pl = plane((3.85, -0.28), 5.0, 3.46, (-3.2, 4.4), (-2.63, 2.63))
        frame = m.Rectangle(width=5.2, height=3.66).set_stroke(EDGE, 2).move_to(pl.coords(0.6, 0))
        pl_t = cite("two of the 2,048 numbers, for example").next_to(frame, m.UP, buff=0.14)
        a_p = arrow_to(vec_box.get_right(), frame.get_left(), MUTED, buff=0.12)
        assert frame.get_right()[0] < 6.6 and pl_t.get_top()[1] < 2.6 and frame.get_bottom()[1] > -3.6, (frame.get_right(), pl_t.get_top())
        assert real_col.get_left()[0] > -6.6 and real_l.get_top()[1] < 2.6 and gen_l.get_bottom()[1] > -3.6, (real_col.get_left(), real_l.get_top(), gen_l.get_bottom())

        sq = lambda c: np.sqrt(np.diag(c))  # noqa: E731
        r_dots = dots(pl, MU_R + FEAT_R * sq(COV_R), REAL, 0.032).set_z_index(2)
        g_dots = dots(pl, MU_G + FEAT_Z * sq(COV_G), GEN, 0.032).set_z_index(2)
        r_ell = ellipse(pl, MU_R, COV_R, REAL, 4).set_z_index(4)
        g_ell = ellipse(pl, MU_G, COV_G, GEN, 4).set_z_index(4)
        r_mu = cross(pl.coords(*MU_R), REAL).set_z_index(5)
        g_mu = cross(pl.coords(*MU_G), GEN).set_z_index(5)

        f = m.MathTex(r"d^2", "=", r"\lVert", r"\mu_r", "-", r"\mu_g", r"\rVert^2", "+",
                      r"\mathrm{Tr}\big(", r"\Sigma_r", "+", r"\Sigma_g", r"-\, 2\,(", r"\Sigma_r", r"\Sigma_g", r")^{1/2}\big)",
                      font_size=32, color=INK)
        for i in (3, 9, 13):
            f[i].set_color(REAL)
        for i in (5, 11, 14):
            f[i].set_color(GEN)
        f.move_to([0, 1.55, 0]).align_to([-6.3, 0, 0], m.LEFT)
        b_mean = m.Brace(f[2:7], m.DOWN, buff=0.08).set_color(MUTED)
        b_spread = m.Brace(f[8:], m.DOWN, buff=0.08).set_color(MUTED)
        l_mean = text("averages", 20, INK).next_to(b_mean, m.DOWN, buff=0.06)
        l_spread = text("spreads", 20, INK).next_to(b_spread, m.DOWN, buff=0.06)
        f_name = text("the Fréchet distance between two Gaussians", 22, INK, weight=m.BOLD).next_to(f, m.UP, buff=0.3).align_to(f, m.LEFT)
        assert f.get_right()[0] < frame.get_left()[0] - 0.15 and f_name.get_top()[1] < 2.6, (f.get_right(), frame.get_left(), f_name.get_top())
        score_l = text("score", 24, MUTED).move_to([0, -0.5, 0]).align_to([-6.3, 0, 0], m.LEFT)
        score = Readout(-5.15, -0.5, "{:.3f}", float(round(FD_BOTH, 3)), 46, INK)
        sit(score_l, baseline(score.words))
        call = m.VGroup(code("frechet_distance(mu_r, cov_r, mu_g, cov_g)", 20, INK, t2c={r"frechet_distance": TEAL, r"_r\b": REAL, r"_g\b": GEN}),
                        cite("the course's code, diffusion_reference.py")).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        call.move_to([0, -1.45, 0]).align_to([-6.3, 0, 0], m.LEFT)
        steps_t = m.VGroup(*[code(s, 22, c) for s, c in (("identical            0", MUTED), ("shifted by 1         1.000", MUTED),
                                                          (f"shifted, stretched   {FD_BOTH:.3f}", INK))])
        steps_t.arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT).move_to([0, -2.85, 0]).align_to([-6.3, 0, 0], m.LEFT)
        assert call.get_right()[0] < frame.get_left()[0] - 0.1 and steps_t.get_bottom()[1] > -3.6, (call.get_right(), steps_t.get_bottom())
        assert l_spread.get_bottom()[1] > score.get_top()[1] + 0.15, (l_spread.get_bottom(), score.get_top())

        def morph(mu0, cov0, mu1, cov1, seconds=1.4):
            def step(_, a):
                mu = mu0 + (mu1 - mu0) * a
                cov = cov0 + (cov1 - cov0) * a
                for d_, z in zip(g_dots, FEAT_Z):
                    d_.move_to(pl.coords(*(mu + z * sq(cov))))
                g_ell.become(ellipse(pl, mu, cov, GEN, 4).set_z_index(4))
                g_mu.move_to(pl.coords(*mu))
                score.show(round(frechet_distance(MU_R, COV_R, mu, cov), 3))
            return m.UpdateFromAlphaFunc(m.VGroup(g_dots, g_ell, g_mu, score), step, run_time=seconds, rate_func=m.smooth)

        pipeline = m.Group(real_col, gen_col, real_l, gen_l, net, a_r, a_g, vec, vec_l, a_v, a_p, src)
        self.say(
            "[The standard score is called FID]. [Every picture goes through an image classifier, Inception], "
            "[which hands back two thousand and forty-eight numbers].",
            m.AnimationGroup(self.new_header("01 · two clouds", "Features, then two Gaussians"),
                             m.FadeIn(real_col), m.FadeIn(real_l), m.FadeIn(gen_col), m.FadeIn(gen_l)),
            m.AnimationGroup(m.GrowArrow(a_r), m.GrowArrow(a_g), m.FadeIn(net), m.FadeIn(src)),
            m.AnimationGroup(m.GrowArrow(a_v), m.FadeIn(vec, shift=0.1 * m.RIGHT), m.FadeIn(vec_l)),
        )
        self.say(
            "[So each picture becomes a point]: [real pictures make one cloud], [generated pictures make another].",
            m.AnimationGroup(m.GrowArrow(a_p), m.Create(frame), m.FadeIn(pl_t)),
            m.LaggedStart(*[m.FadeIn(d_, scale=0.5) for d_ in r_dots], lag_ratio=0.004, run_time=1.2),
            m.LaggedStart(*[m.FadeIn(d_, scale=0.5) for d_ in g_dots], lag_ratio=0.004, run_time=1.2),
        )
        self.say(
            "[Then it keeps only each cloud's average and spread]: [a Gaussian, drawn here as an ellipse].",
            m.AnimationGroup(m.FadeIn(r_mu, scale=0.5), m.FadeIn(g_mu, scale=0.5)),
            m.AnimationGroup(m.Create(r_ell), m.Create(g_ell), r_dots.animate.set_opacity(0.3), g_dots.animate.set_opacity(0.3)),
        )
        self.say(
            "[The score is the Fréchet distance between the two Gaussians]: [one term for the averages], [one for the spreads].",
            m.AnimationGroup(m.FadeOut(pipeline), m.FadeIn(f_name), m.Write(f)),
            m.AnimationGroup(m.FadeIn(b_mean), m.FadeIn(l_mean)),
            m.AnimationGroup(m.FadeIn(b_spread), m.FadeIn(l_spread)),
        )
        self.say(
            "[With the course's own code]: [make the two Gaussians identical], [and the score is zero].",
            m.AnimationGroup(m.FadeIn(call, shift=0.1 * m.UP), m.FadeIn(score_l), m.FadeIn(score)),
            morph(MU_G, COV_G, MU_R, COV_R, 1.5),
            m.FadeIn(steps_t[0], shift=0.1 * m.UP),
        )
        self.say(
            "[Slide the generated one a unit to the right]: [one]. [Stretch it and squeeze it]: [one point two six].",
            morph(MU_R, COV_R, MU_G, COV_R, 1.3),
            m.FadeIn(steps_t[1], shift=0.1 * m.UP),
            morph(MU_G, COV_R, MU_G, COV_G, 1.3),
            m.AnimationGroup(m.FadeIn(steps_t[2], shift=0.1 * m.UP), m.Circumscribe(score, color=GEN, buff=0.1)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 2 · the blind spot

    def blind_spot(self):
        self.next_section("2 · Same statistics, wrong shape", notes=(
            "toy.md §7, recomputed here with the experiment's own random stream and asserted against toy.json: the swirl, "
            "10,000 points (diffusion_reference.swirl, seed 31); a Gaussian with exactly their mean and covariance, 10,000 "
            f"points; Fréchet distance {FD_BLOB:.7f} (another 10,000 real swirl points, seed 32: {FD_REAL2:.2e}); "
            f"{100 * OFF_BLOB:.1f}% of the Gaussian's points are more than 0.1 from the swirl's centre line (off_swirl; real "
            f"data: {100 * OFF_REAL:.1f}%). 1,000 points of each are drawn; ellipses: two standard deviations of the shared "
            "covariance."))
        lim = 1.6
        size = 4.3
        pls = [plane((x, -0.05), size, size, (-lim, lim), (-lim, lim)) for x in (-3.35, 3.35)]
        frames = m.VGroup(*[m.Square(size).set_stroke(EDGE, 2).move_to(p.coords(0, 0)) for p in pls])
        ok_rows = np.all(np.abs(SW_DATA[:1300]) < 0.97 * lim, 1) & np.all(np.abs(SW_GAUSS[:1300]) < 0.97 * lim, 1)
        keep = np.flatnonzero(ok_rows)[:1000]
        assert len(keep) == 1000
        real = dots(pls[0], SW_DATA[keep], REAL, 0.024).set_z_index(2)
        blob = Cloud(pls[1], SW_DATA[keep], GEN, 0.024, 0.9).set_z_index(2)
        band = swirl_band(pls[1], REAL, 11, 0.16)
        ells = m.VGroup(*[ellipse(p, SW_MU, SW_COV, INK, 3, opacity=0.8).set_z_index(4) for p in pls])
        mus = m.VGroup(*[cross(p.coords(*SW_MU), INK, 0.11, 3).set_z_index(5) for p in pls])
        t_l = m.VGroup(code("the swirl", 24, REAL, weight=m.BOLD), cite("10,000 real points"))
        t_r = m.VGroup(code("a blob", 24, GEN, weight=m.BOLD), cite("the same mean and covariance"))
        for t_, fr in ((t_l, frames[0]), (t_r, frames[1])):
            t_.arrange(m.RIGHT, buff=0.25)
            sit(t_[1], baseline(t_[0]))
            t_.next_to(fr, m.UP, buff=0.16)
        mid = m.VGroup(text("score", 24, MUTED), code(BLOB_S, 36, GREEN, weight=m.BOLD)).arrange(m.DOWN, buff=0.12).move_to([0, 0.3, 0])
        ok = pill("near perfect", GREEN, GREEN_FILL, 20).next_to(mid, m.UP, buff=0.25)
        mid_r = m.VGroup(cite("real vs real"), code(f"{FD_REAL2:.5f}", 20, MUTED)).arrange(m.DOWN, buff=0.08).move_to([0, -0.95, 0])
        off_idx = np.flatnonzero(nearest_on_swirl(SW_GAUSS[keep])[0] > 0.1)
        off_l = m.VGroup(code(BLOB_OFF, 30, RED, weight=m.BOLD), text("of the blob's points lie off the swirl", 24, INK)).arrange(m.RIGHT, buff=0.25)
        sit(off_l[1], baseline(off_l[0]))
        off_l.move_to([0, -2.72, 0])
        moral = pill("matching statistics ≠ matching the data", RED, RED_FILL, 22).move_to([0, -3.38, 0])
        gap_w = frames[1].get_left()[0] - frames[0].get_right()[0]
        assert mid.width < gap_w - 0.2 and ok.width < gap_w - 0.1 and mid_r.width < gap_w - 0.2, (mid.width, ok.width, gap_w)
        assert t_r.get_right()[0] < 6.6 and t_l.get_top()[1] < 2.6 and off_l.get_top()[1] < frames.get_bottom()[1] - 0.08, (t_r.get_right(), t_l.get_top(), off_l.get_top())
        assert abs(len(off_idx) / len(keep) - OFF_BLOB) < 0.03, len(off_idx)

        self.say(
            "[Now the swirl]: [ten thousand real points on two thin arms].",
            m.AnimationGroup(self.new_header("02 · the blind spot", "Same statistics, wrong shape"), m.FadeIn(frames[0]), m.FadeIn(t_l)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(d_, scale=0.5) for d_ in real], lag_ratio=0.001, run_time=1.4),
                             m.Succession(m.Wait(0.8), m.AnimationGroup(m.Create(ells[0]), m.FadeIn(mus[0])))),
        )
        self.say(
            "[Make a shapeless blob with exactly the same average and spread].",
            m.AnimationGroup(m.FadeIn(frames[1]), m.FadeIn(t_r), m.FadeIn(blob, target_position=real),
                             m.Succession(m.Wait(1.0), blob.to(SW_GAUSS[keep], run_time=2.0)),
                             m.Succession(m.Wait(2.3), m.AnimationGroup(m.Create(ells[1]), m.FadeIn(mus[1])))),
        )
        self.say(
            "[Its score against the swirl]: [zero point zero zero zero three]. [Near perfect].",
            m.Circumscribe(ells[1], color=INK, buff=0.04),
            m.AnimationGroup(m.FadeIn(mid, shift=0.1 * m.UP), m.FadeIn(mid_r)),
            m.FadeIn(ok, shift=0.1 * m.UP),
        )
        self.say(
            "[Yet forty-eight percent of its points lie off the swirl].",
            m.AnimationGroup(m.FadeIn(band), recolour(blob, off_idx, GEN, RED, 1.2), m.FadeIn(off_l, shift=0.1 * m.UP)),
        )
        self.say(
            "[The score only sees averages and spreads]: [matching the statistics is not matching the data].",
            m.AnimationGroup(m.Circumscribe(ells[0], color=INK, buff=0.04), m.Circumscribe(ells[1], color=INK, buff=0.04)),
            m.FadeIn(moral, shift=0.1 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 3 · sample size

    def sample_size(self):
        self.next_section("3 · Few samples, a biased score", notes=(
            "toy.md §7, recomputed here exactly (the experiment's random stream, asserted against toy.json): samples of "
            "N(0, I) scored against N(0, I) itself (true value 0), mean over 200 draws (2-D) and 40 draws (64-D): "
            + "; ".join(f"{d}-D, n = {n:,}: {BIAS[(d, n)]}" for d, n in sorted(BIAS)) +
            ". For N(0, I) the expected bias is ≈ (d + d(d + 1)/4)/n: it falls as 1/n and grows like d². The picture's "
            f"100 points are one draw (seed {DRAW_SEED}, its own score {_fd[DRAW_SEED]:.3f}, chosen near the mean); 2,000 of "
            "10,000 points are drawn for n = 10,000. Chong & Forsyth (arXiv:1911.07023, CVPR 2020): FID and IS are biased, "
            "and the bias depends on the model, so compare only at equal sample counts. Heusel et al. (2017) computed FID "
            "on 50,000 generated images."))
        lim = 3.1
        size = 3.9
        pl = plane((-3.85, 0.1), size, size, (-lim, lim), (-lim, lim))
        frame = m.Square(size).set_stroke(EDGE, 2).move_to(pl.coords(0, 0))
        truth = m.DashedVMobject(ellipse(pl, np.zeros(2), np.eye(2), REAL, 3.5), num_dashes=40).set_z_index(4)
        d100 = dots(pl, DRAW100, GEN, 0.045).set_z_index(2)
        e100 = ellipse(pl, *gaussian_stats(DRAW100), GEN, 4).set_z_index(5)
        shown = DRAW10K[:2000][np.all(np.abs(DRAW10K[:2000]) < 0.97 * lim, 1)]
        d10k = dots(pl, shown, GEN, 0.018, 0.75).set_z_index(2)
        e10k = ellipse(pl, *gaussian_stats(DRAW10K), GEN, 4).set_z_index(5)
        n_read = Readout(frame.get_left()[0], frame.get_bottom()[1] - 0.36, "n = {}", "100", 26, GEN)
        n_note = cite("one draw · 2,000 of 10,000 shown").move_to(n_read.get_center()).align_to([frame.get_left()[0] + 2.5, 0, 0], m.LEFT)
        sit(n_note, baseline(n_read.words))
        leg = m.VGroup(m.VGroup(m.DashedLine([0, 0, 0], [0.45, 0, 0], dash_length=0.08).set_stroke(REAL, 3.5), cite("true Gaussian")).arrange(m.RIGHT, buff=0.15),
                       m.VGroup(m.Line([0, 0, 0], [0.45, 0, 0]).set_stroke(GEN, 4), cite("fitted to the draw")).arrange(m.RIGHT, buff=0.15))
        leg.arrange(m.RIGHT, buff=0.4).next_to(frame, m.UP, buff=0.15).align_to(frame, m.LEFT)
        assert frame.get_left()[0] > -6.6 and leg.get_top()[1] < 2.6 and leg.get_left()[0] > -6.6, (frame.get_left(), leg.get_top(), leg.get_left())

        bp = Plot((60, 18000), (1e-4, 40), width=4.4, height=3.2, center=(3.3, -0.12, 0), x_ticks=[100, 1000, 10000],
                  y_ticks=[1e-4, 1e-2, 1], x_label="samples", log_x=True, log_y=True, x_fmt="{:,.0f}", y_fmt="{:g}")
        y_t = cite("score of identical distributions (should be 0)").next_to(bp, m.UP, buff=0.12).align_to(bp, m.LEFT)
        ns = (100, 1000, 10000)
        line2 = bp.line(ns, [BIAS[(2, n)] for n in ns], INK, 4)
        line64 = bp.line(ns, [BIAS[(64, n)] for n in ns], RED, 4)
        dots2 = m.VGroup(*[bp.dot(n, BIAS[(2, n)], INK, 0.07) for n in ns])
        dots64 = m.VGroup(*[bp.dot(n, BIAS[(64, n)], RED, 0.07) for n in ns])
        v2a = code(B_TXT[(2, 100)], 22, INK, weight=m.BOLD).next_to(dots2[0], m.UP, buff=0.14).shift(0.15 * m.RIGHT)
        v2c = code(B_TXT[(2, 10000)], 22, INK, weight=m.BOLD).next_to(dots2[2], m.UP, buff=0.14).shift(0.2 * m.RIGHT)
        v64 = code(B_TXT[(64, 1000)], 22, RED, weight=m.BOLD).next_to(dots64[1], m.DOWN, buff=0.16)
        mid_y = lambda d: float(np.sqrt(BIAS[(d, 1000)] * BIAS[(d, 10000)]))  # noqa: E731  (the line's height at n ≈ 3,162)
        lab2 = code("2 numbers", 20, INK).next_to(bp.coords(3162, mid_y(2)), m.DOWN, buff=0.3).shift(0.15 * m.LEFT)
        lab64 = code("64 numbers", 20, RED).next_to(bp.coords(3162, mid_y(64)), m.UP, buff=0.3).shift(0.35 * m.RIGHT)
        assert y_t.get_top()[1] < 2.6 and bp.get_right()[0] < 6.6 and lab64.get_right()[0] < 6.6, (y_t.get_top(), bp.get_right(), lab64.get_right())
        assert leg.get_right()[0] < y_t.get_left()[0] - 0.3, (leg.get_right(), y_t.get_left())
        foot = m.VGroup(pill("Inception: 2,048 numbers  →  FID on 50,000 pictures", INK, CARD_HI, 22),
                        cite("Heusel et al., 2017 · compare only at equal sample counts: Chong & Forsyth, 2020")).arrange(m.DOWN, buff=0.1)
        foot.move_to([0, -3.0, 0])
        assert foot.get_bottom()[1] > -3.6 and foot.width < 13.2 and foot.get_top()[1] < n_read.get_bottom()[1] - 0.05, (foot.get_bottom(), foot.get_top(), n_read.get_bottom())
        assert bp.get_bottom()[1] > foot.get_top()[1] + 0.05, (bp.get_bottom(), foot.get_top())

        self.say(
            "[A second catch]: [draw a hundred points from one Gaussian], [and score them against that very Gaussian].",
            m.AnimationGroup(self.new_header("03 · sample size", "Few samples, a biased score"), m.FadeIn(frame), m.Create(truth),
                             m.FadeIn(leg[0])),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(d_, scale=0.5) for d_ in d100], lag_ratio=0.02, run_time=1.2), m.FadeIn(n_read)),
            m.AnimationGroup(m.Create(e100), m.FadeIn(leg[1])),
        )
        self.say(
            "[The answer should be zero]. [Averaged over many draws, it is zero point zero three seven].",
            m.AnimationGroup(m.FadeIn(bp), m.FadeIn(y_t)),
            m.AnimationGroup(m.FadeIn(dots2[0], scale=0.5), m.FadeIn(v2a)),
        )
        self.say(
            "[With ten thousand points], [it falls to zero point zero zero zero three five], [and the fitted ellipse finds the truth].",
            m.AnimationGroup(m.FadeOut(d100), m.LaggedStart(*[m.FadeIn(d_) for d_ in d10k], lag_ratio=0.0005, run_time=1.0),
                             m.Transform(e100, e10k), m.FadeIn(n_note), n_read.animate.show("10,000")),
            m.AnimationGroup(m.Create(line2, run_time=1.2), m.FadeIn(dots2[1:]), m.FadeIn(v2c), m.FadeIn(lab2)),
            m.Circumscribe(frame, color=GEN, buff=0.04),
        )
        self.say(
            "[In sixty-four dimensions], [even a thousand points score one point one two]: [more numbers, more bias].",
            m.AnimationGroup(m.Create(line64, run_time=1.2), m.FadeIn(dots64), m.FadeIn(lab64)),
            m.AnimationGroup(m.FadeIn(v64), m.Circumscribe(dots64[1], color=RED, buff=0.08)),
            m.Indicate(lab64, color=RED, scale_factor=1.12),
        )
        self.say(
            "[Inception gives two thousand and forty-eight], [so FID is computed on fifty thousand pictures], "
            "[and compared only at equal sample counts].",
            m.FadeIn(foot[0], shift=0.1 * m.UP),
            m.Circumscribe(foot[0], color=INK, buff=0.06),
            m.FadeIn(foot[1]),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 4 · our score

    def our_score(self):
        self.next_section("4 · A digit classifier's FID", notes=(
            "mac_runs.md §3: our classifier (conv32-conv32-pool-conv64-conv64-pool-fc128-fc10, 467,818 parameters, 6 "
            f"epochs, {CLS_S} on this Mac) reads 99.45% of the MNIST test digits correctly (55 errors). Its 128-d penultimate "
            "features feed the same formula: 10,000 generated digits vs the 10,000 test digits. NOT Inception FID; only "
            f"comparable with itself. Floor: 10,000 real training digits vs test = {FLOOR:.3f}; five random subsets "
            f"{', '.join(f'{v:.2f}' for v in SUBSETS)} (mean {UNC['real_floor_5_subsets']['mean']:.2f}, std {SUB_STD:.3f}). "
            f"Bootstrap (20 resamples of the generated features vs the fixed test set): std {BOOT['ddim50']:.3f} at DDIM 50; "
            + ", ".join(f"{k} {v:.2f}" for k, v in BOOT.items()) + " across samplers (mac_runs: 'Differences under about "
            "0.7 are not meaningful'). Standard FID: Inception-v3 pool features, 2,048-d, 50,000 samples (Heusel et al. "
            "2017); even the reference set changes it (DDPM: 3.17 vs the training set, 5.24 vs test, notes §19)."))
        real_col = digit_col(REAL_DIGITS[3:5], REAL, 0.55, gap_px=5).move_to([-5.95, 1.55, 0])
        gen_col = digit_col(DIG["ddim50_w1"][7:9], GEN, 0.55, gap_px=5).move_to([-5.95, -0.05, 0])
        net_t = m.VGroup(text("our digit classifier", 24, INK, weight=m.BOLD), text("467,818 parameters", 20, MUTED),
                         text(f"trained on this Mac in {CLS_S}", 20, MUTED)).arrange(m.DOWN, buff=0.08)
        net_box = card(net_t.width + 0.5, net_t.height + 0.45, REAL, TEAL_FILL)
        net = m.VGroup(net_box, net_t.move_to(net_box)).move_to([-2.85, 0.85, 0])
        a_r = arrow_to(real_col.get_right() + [0.08, 0, 0], net_box.get_left() + [0, 0.3, 0], REAL)
        a_g = arrow_to(gen_col.get_right() + [0.08, 0, 0], net_box.get_left() + [0, -0.3, 0], GEN)
        acc = m.VGroup(code(f"{100 * CLS['test_accuracy']:.2f}%", 28, GREEN, weight=m.BOLD), text("of test digits read correctly", 22, INK)).arrange(m.RIGHT, buff=0.2)
        sit(acc[1], baseline(acc[0]))
        acc.next_to(net, m.DOWN, buff=0.22)
        feat = m.VGroup(code("128", 30, INK, weight=m.BOLD), cite("numbers per digit")).arrange(m.DOWN, buff=0.06).move_to([0.95, 0.85, 0])
        a_f = arrow_to(net_box.get_right(), feat.get_left() + [-0.05, 0, 0], MUTED, buff=0.12)
        form = m.VGroup(text("the same formula", 22, INK), code("our digit score", 24, REAL, weight=m.BOLD)).arrange(m.DOWN, buff=0.1).move_to([4.3, 0.85, 0])
        form_b = boxed(form, REAL, 0.25)
        a_s = arrow_to(feat.get_right() + [0.05, 0, 0], form_b.get_left(), MUTED, buff=0.12)
        assert form_b.get_right()[0] < 6.6 and real_col.get_left()[0] > -6.6 and real_col.get_top()[1] < 2.6, (form_b.get_right(), real_col.get_left(), real_col.get_top())
        colx = (-6.1, -2.55, 1.65)
        rows_data = (("", "standard FID", "our digit score"),
                     ("features from", "Inception-v3, ImageNet", "our digit classifier"),
                     ("numbers per picture", "2,048", "128"),
                     ("pictures compared", "50,000", "10,000 vs 10,000"))
        trows = m.VGroup()
        for i, row in enumerate(rows_data):
            cells = m.VGroup()
            for j, (s, x) in enumerate(zip(row, colx)):
                if not s:
                    continue
                if i == 0:
                    w_ = text(s, 22, INK if j == 1 else REAL, weight=m.BOLD)
                elif j == 0:
                    w_ = text(s, 22, MUTED)
                else:
                    w_ = code(s, 22, INK if j == 1 else REAL)
                cells.add(w_.move_to([0, -1.1 - 0.5 * i, 0]).align_to([x, 0, 0], m.LEFT))
            trows.add(cells)
        rule = m.Line([-6.1, -1.37, 0], [5.9, -1.37, 0]).set_stroke(EDGE, 2)
        verdict = pill("not Inception FID · never compare it with published numbers", RED, RED_FILL, 22).move_to([0, -3.35, 0])
        assert trows.get_right()[0] < 6.6 and trows.get_bottom()[1] > verdict.get_top()[1] + 0.08, (trows.get_right(), trows.get_bottom())
        assert acc.get_bottom()[1] > trows.get_top()[1] + 0.1, (acc.get_bottom(), trows.get_top())

        # the ruler: how big is a meaningful gap?
        rx0, rx1, v0, v1 = -5.6, 5.6, 1.6, 4.0
        X = lambda v: rx0 + (rx1 - rx0) * (v - v0) / (v1 - v0)  # noqa: E731
        ry = -0.55
        axis = m.Line([rx0 - 0.2, ry, 0], [rx1 + 0.2, ry, 0]).set_stroke(FAINT, 3)
        ticks = m.VGroup()
        for v in (2.0, 2.5, 3.0, 3.5, 4.0):
            ticks.add(m.Line([X(v), ry - 0.09, 0], [X(v), ry + 0.09, 0]).set_stroke(FAINT, 2),
                      code(f"{v:.1f}", 20, MUTED).move_to([X(v), ry - 0.38, 0]))
        floor_dot = m.Dot([X(FLOOR), ry, 0], radius=0.11, color=REAL).set_z_index(3)
        sub_dots = m.VGroup(*[m.Dot([X(v), ry + 0.42, 0], radius=0.07, color=REAL).set_opacity(0.85) for v in SUBSETS])
        sub_span = m.Line([X(min(SUBSETS)), ry + 0.42, 0], [X(max(SUBSETS)), ry + 0.42, 0]).set_stroke(REAL, 3, opacity=0.6)
        floor_l = m.VGroup(text("real vs real: the floor", 22, REAL, weight=m.BOLD), code(f"{FLOOR:.2f}", 26, REAL, weight=m.BOLD)).arrange(m.RIGHT, buff=0.2)
        sit(floor_l[1], baseline(floor_l[0]))
        floor_l.move_to([0, ry + 1.65, 0]).align_to([X(FLOOR) - 0.5, 0, 0], m.LEFT)
        sub_l = cite(f"five subsets: {min(SUBSETS):.2f} to {max(SUBSETS):.2f}").move_to([0, ry + 1.05, 0]).align_to([X(FLOOR) - 0.5, 0, 0], m.LEFT)
        c50 = SC[50]["cfid"]
        lo, hi = c50 - BOOT["ddim50"], c50 + BOOT["ddim50"]
        bar = m.Line([X(lo), ry, 0], [X(hi), ry, 0]).set_stroke(GEN, 8).set_z_index(2)
        caps = m.VGroup(*[m.Line([X(v), ry - 0.16, 0], [X(v), ry + 0.16, 0]).set_stroke(GEN, 4) for v in (lo, hi)])
        d50 = m.Dot([X(c50), ry, 0], radius=0.11, color=GEN).set_z_index(3)
        d50_l = m.VGroup(text("fifty steps", 22, GEN, weight=m.BOLD), code(f"{S_TXT[50]} ± {BOOT_S}", 26, GEN, weight=m.BOLD)).arrange(m.RIGHT, buff=0.2)
        sit(d50_l[1], baseline(d50_l[0]))
        d50_l.move_to([X(c50), ry + 0.75, 0])
        gap = m.VGroup(text("gaps under about 0.7 mean nothing", 24, INK, weight=m.BOLD),
                       cite("bootstrap: 20 resamples of the generated digits · mac_runs.md")).arrange(m.DOWN, buff=0.1)
        gap.move_to([0, -2.55, 0])
        assert d50_l.get_right()[0] < 6.6 and floor_l.get_left()[0] > -6.6 and gap.get_bottom()[1] > -3.6, (d50_l.get_right(), floor_l.get_left())
        assert d50_l.get_left()[0] > sub_l.get_right()[0] + 0.2 or d50_l.get_bottom()[1] > sub_l.get_top()[1] + 0.05, (d50_l.get_left(), sub_l.get_right())

        top = m.Group(real_col, gen_col, net, a_r, a_g, acc, feat, a_f, form_b, a_s)
        self.say(
            "[Our digit score uses the same formula], [with Inception swapped for a small digit classifier, trained on this Mac].",
            m.AnimationGroup(self.new_header("04 · our score", "A digit classifier's FID"), m.FadeIn(form_b, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(real_col), m.FadeIn(gen_col), m.GrowArrow(a_r), m.GrowArrow(a_g), m.FadeIn(net)),
        )
        self.say(
            "[It reads ninety-nine point four five percent of test digits correctly], [and gives a hundred and twenty-eight numbers per digit].",
            m.FadeIn(acc, shift=0.1 * m.UP),
            m.AnimationGroup(m.GrowArrow(a_f), m.FadeIn(feat, shift=0.1 * m.RIGHT), m.GrowArrow(a_s)),
        )
        self.say(
            "[We compare ten thousand generated digits with ten thousand real ones]. "
            "[So it is not Inception FID]: [never compare it with published numbers].",
            m.AnimationGroup(m.FadeIn(rule), m.LaggedStart(*[m.FadeIn(r_, shift=0.05 * m.UP) for r_ in trows], lag_ratio=0.25, run_time=1.6)),
            m.Circumscribe(m.VGroup(trows[1][1], trows[1][2]), color=INK, buff=0.08),
            m.FadeIn(verdict, shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.say(
            "[Real digits against real digits score two point oh six]: [that is the floor], [give or take zero point one].",
            m.AnimationGroup(m.FadeOut(m.Group(top, trows, rule, verdict)), m.FadeIn(axis), m.FadeIn(ticks),
                             m.FadeIn(floor_dot, scale=0.5), m.FadeIn(floor_l, shift=0.1 * m.UP)),
            m.Circumscribe(floor_dot, color=REAL, buff=0.06),
            m.AnimationGroup(m.FadeIn(sub_span), m.LaggedStart(*[m.FadeIn(d_, scale=0.5) for d_ in sub_dots], lag_ratio=0.2), m.FadeIn(sub_l)),
        )
        self.say(
            "[Resampling the same fifty-step digits moves the score by about zero point three five], "
            "[so gaps smaller than about zero point seven mean nothing].",
            m.AnimationGroup(m.FadeIn(d50, scale=0.5), m.FadeIn(d50_l, shift=0.1 * m.UP), m.GrowFromCenter(bar), m.FadeIn(caps)),
            m.FadeIn(gap, shift=0.1 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 5 · against steps

    def against_steps(self):
        self.next_section("5 · The score against steps", notes=(
            "Measured on this Mac (mac_runs.md §4, Episode 9): the digit model sampled with DDIM 5 / 10 / 20 / 50 / 100 steps "
            "and DDPM ancestral 1,000 steps (fresh noise each step), 10,000 digits each, the same starting noise (seed "
            "1234). cFID: " + ", ".join(f"{n}: {S_TXT[n]}" for n in (5, 10, 20, 50, 100, 1000)) + ". Error bars: bootstrap "
            "std " + ", ".join(f"{k} {v:.2f}" for k, v in BOOT.items()) + " (none measured for 5 and 10 steps). Floor band: "
            f"the five real-vs-real subsets, {min(SUBSETS):.2f}–{max(SUBSETS):.2f}. Digits: samples 10–14 (labels 0–4), "
            "generated on this Mac."))
        bp = Plot((3.6, 1600), (1.7, 30), width=6.3, height=3.6, center=(-2.6, -0.3, 0), x_ticks=[5, 10, 20, 50, 100, 1000],
                  y_ticks=[2, 3, 5, 10, 20], x_label="sampling steps (log scale)", log_x=True, log_y=True, x_fmt="{:,.0f}")
        y_t = cite("our digit score, lower is better (log scale)").next_to(bp, m.UP, buff=0.12).align_to(bp, m.LEFT)
        band = bp.band([3.6, 1600], [min(SUBSETS)] * 2, [max(SUBSETS)] * 2, REAL, 0.16)
        fl = bp.hline(FLOOR, REAL, 2.5)
        fl_l = text(f"real {FLOOR:.2f}", 20, REAL).next_to(bp.coords(3.75, max(SUBSETS)), m.UR, buff=0.05)
        xs = list(STEPS)
        ys = [SC[n]["cfid"] for n in xs]
        line = bp.line(xs, ys, GEN, 4)
        pts = m.VGroup(*[bp.dot(n, SC[n]["cfid"], GEN, 0.08) for n in xs])
        vals = m.VGroup()
        for n in xs:
            p = bp.coords(n, SC[n]["cfid"])
            if n == 5:
                vals.add(code(S_TXT[n], 20, GEN).next_to(p, m.RIGHT, buff=0.15))
            elif n == 10:
                vals.add(code(S_TXT[n], 20, GEN).next_to(p, m.LEFT, buff=0.14))
            else:
                vals.add(code(S_TXT[n], 20, GEN).move_to(p + np.array([0, 0.5, 0])))
        errs = m.VGroup()
        for n, k in ((20, "ddim20"), (50, "ddim50"), (100, "ddim100"), (1000, "ddpm1000")):
            c_ = SC[n]["cfid"]
            a, b = bp.coords(n, c_ - BOOT[k]), bp.coords(n, c_ + BOOT[k])
            errs.add(m.VGroup(m.Line(a, b), m.Line(a + [-0.07, 0, 0], a + [0.07, 0, 0]), m.Line(b + [-0.07, 0, 0], b + [0.07, 0, 0]))
                     .set_stroke(VIOLET if n == 1000 else GEN, 2.5))
        tie = m.SurroundingRectangle(m.VGroup(pts[2:5], errs[:3], vals[2:5]), buff=0.1, corner_radius=0.1).set_stroke(INK, 2)
        tie_l = text("a tie", 20, INK, weight=m.BOLD).next_to(tie, m.UP, buff=0.08)
        p1000 = bp.dot(1000, SC[1000]["cfid"], VIOLET, 0.1)
        v1000 = code(S_TXT[1000], 22, VIOLET, weight=m.BOLD).next_to(errs[3], m.UP, buff=0.1)
        assert y_t.get_top()[1] < 2.6 and bp.get_left()[0] > -6.6 and tie_l.get_top()[1] < y_t.get_bottom()[1] - 0.02, (y_t.get_top(), bp.get_left(), tie_l.get_top())

        rx = 4.5
        dig_rows = m.Group()
        for i, (key, n, col) in enumerate((("ddim5_w1", 5, GEN), ("ddim10_w1", 10, GEN), ("ddim50_w1", 50, GEN), ("ddpm1000_w1", 1000, VIOLET))):
            pic = digit_row(DIG[key][:5], GEN, 0.5, gap_px=5).move_to([rx, 1.45 - 0.98 * i, 0])
            lab = code(f"{n:,} steps", 20, col, weight=m.BOLD).next_to(pic, m.UP, buff=0.08).align_to(pic, m.LEFT)
            val = code(S_TXT[n], 20, col).next_to(lab, m.RIGHT, buff=0.25)
            sit(val, baseline(lab))
            dig_rows.add(m.Group(pic, lab, val))
        d_cap = cite("generated on this Mac").next_to(dig_rows, m.DOWN, buff=0.2)
        assert dig_rows.get_right()[0] < 6.6 and dig_rows.get_top()[1] < 2.6 and d_cap.get_bottom()[1] > -3.6, (dig_rows.get_right(), dig_rows.get_top(), d_cap.get_bottom())
        assert dig_rows.get_left()[0] > bp.get_right()[0] + 0.1, (dig_rows.get_left(), bp.get_right())

        steps_read = Readout(-0.6, 2.15, "steps: {}", "5", 22, GEN)
        lx = np.log10(np.array(xs, dtype=float))

        def draw(seconds=2.4):
            def step(_, a):
                pos = a * (len(xs) - 1)
                j = min(int(np.floor(pos)), len(xs) - 2)
                u = pos - j
                pts_now = [bp.coords(x, y) for x, y in zip(xs[: j + 1], ys[: j + 1])]
                xn = 10 ** (lx[j] + (lx[j + 1] - lx[j]) * u)
                yn = 10 ** (np.log10(ys[j]) + (np.log10(ys[j + 1]) - np.log10(ys[j])) * u)
                pts_now.append(bp.coords(xn, yn))
                line.set_points_as_corners(pts_now)
                steps_read.show(f"{int(round(xn)):,}")
            return m.UpdateFromAlphaFunc(m.VGroup(line, steps_read), step, run_time=seconds, rate_func=m.linear)

        line.set_points_as_corners([bp.coords(xs[0], ys[0]), bp.coords(xs[0], ys[0]) + np.array([0.001, 0, 0])])
        self.say(
            "[Here are the scores from episode nine], [as the same model takes more steps].",
            m.AnimationGroup(self.new_header("05 · steps", "The score against steps"), m.FadeIn(bp), m.FadeIn(y_t),
                             m.FadeIn(band), m.Create(fl), m.FadeIn(fl_l)),
            m.AnimationGroup(m.FadeIn(steps_read), m.FadeIn(d_cap), m.FadeIn(dig_rows[0], shift=0.1 * m.LEFT),
                             m.FadeIn(pts[0], scale=0.5), m.FadeIn(vals[0]), m.FadeIn(line)),
        )
        self.say(
            "[Five steps score twenty-one point six]. [Ten steps, four point five seven].",
            m.Circumscribe(dig_rows[0][0], color=GEN, buff=0.05),
            m.AnimationGroup(draw(2.6), m.Succession(m.Wait(0.55), m.AnimationGroup(m.FadeIn(pts[1], scale=0.5), m.FadeIn(vals[1]),
                                                                                    m.FadeIn(dig_rows[1], shift=0.1 * m.LEFT))),
                             m.Succession(m.Wait(1.7), m.AnimationGroup(m.FadeIn(pts[2:], scale=0.5), m.FadeIn(vals[2:]),
                                                                        m.FadeIn(dig_rows[2], shift=0.1 * m.LEFT)))),
        )
        self.say(
            "[From twenty to a hundred steps], [the scores sit inside each other's error bars]: [a tie].",
            m.FadeIn(errs[:3]),
            m.Create(tie),
            m.FadeIn(tie_l, shift=0.1 * m.UP),
        )
        self.say(
            "[The thousand-step sampler, with fresh noise at every step, comes closest to the floor], [at two point six four].",
            m.AnimationGroup(m.FadeIn(p1000, scale=0.5), m.FadeIn(errs[3]), m.FadeIn(dig_rows[3], shift=0.1 * m.LEFT),
                             steps_read.animate.show("1,000")),
            m.AnimationGroup(m.FadeIn(v1000, shift=0.1 * m.DOWN), m.Circumscribe(dig_rows[3][2], color=VIOLET, buff=0.06)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 6 · precision and recall

    def precision_recall(self):
        self.next_section("6 · Two scores pull apart", notes=(
            "Guidance (mac_runs.md §4, Episode 14): DDIM 50, the same noise, w = 1 / 2 / 3 / 5 / 8: cFID "
            + " / ".join(G_TXT[w] for w in WS) + ". Precision/recall: Kynkäänniemi et al. (arXiv:1904.06991, NeurIPS "
            "2019), k-NN manifold estimate with k = 3 ('k = 3 is a robust choice'); our run computes them on the same 128-d "
            "classifier features. The swirl panels are ILLUSTRATIVE generators, computed here: real = 200 swirl points "
            "(seed 1); 'one arm' = 200 points of arm 0 only (seed 2); 'too much noise' = 200 points of a swirl with "
            f"noise 0.12 instead of 0.03 (seed 3). Precision / recall: one arm {PR_TOY_TXT['narrow'][0]} / "
            f"{PR_TOY_TXT['narrow'][1]}; too much noise {PR_TOY_TXT['sloppy'][0]} / {PR_TOY_TXT['sloppy'][1]}. Digits "
            f"(mac_runs.json): w = 1 {PR_TXT[1]}, w = 3 {PR_TXT[3]}, w = 8 {PR_TXT[8]}; real digits {REAL_PR}; label "
            f"accuracy {ACC[1]} → {ACC[3]} → {ACC[8]}."))
        # A: the score against guidance
        gp = Plot((0, 8.6), (0, 95), width=5.2, height=3.3, center=(-3.0, -0.4, 0), x_ticks=[1, 2, 3, 5, 8],
                  y_ticks=[0, 30, 60, 90], x_label="guidance scale w")
        g_t = cite("our digit score, lower is better · 50 steps").next_to(gp, m.UP, buff=0.12).align_to(gp, m.LEFT)
        g_line = gp.line(list(WS), [GC[w]["cfid"] for w in WS], RED, 4)
        g_pts = m.VGroup(*[gp.dot(w, GC[w]["cfid"], RED if w > 1 else GREEN, 0.08) for w in WS])
        g_v1 = code(G_TXT[1], 20, GREEN).next_to(g_pts[0], m.UP, buff=0.2).shift(0.22 * m.LEFT)
        g_v8 = code(G_TXT[8], 22, RED, weight=m.BOLD).next_to(g_pts[4], m.UP, buff=0.16)
        g_fl = gp.hline(FLOOR, REAL, 2)
        grows = m.Group()
        for i, w in enumerate((1, 3, 8)):
            pic = digit_row(DIG[f"ddim50_w{w}"][:6], GEN, 0.55, gap_px=5).move_to([3.8, 1.45 - 1.15 * i, 0])
            lab = code(f"w = {w}", 22, PROMPT, weight=m.BOLD).next_to(pic, m.UP, buff=0.08).align_to(pic, m.LEFT)
            grows.add(m.Group(pic, lab))
        gcap = cite("generated on this Mac · 50 steps").next_to(grows, m.DOWN, buff=0.18)
        why = text("One number cannot say what went wrong.", 26, INK, weight=m.BOLD).move_to([0, -3.3, 0])
        assert g_t.get_top()[1] < 2.6 and grows.get_right()[0] < 6.6 and grows.get_top()[1] < 2.6, (g_t.get_top(), grows.get_right(), grows.get_top())
        assert gcap.get_bottom()[1] > why.get_top()[1] + 0.05 and gp.get_bottom()[1] > why.get_top()[1] + 0.05, (gcap.get_bottom(), gp.get_bottom())

        # B: the swirl toy
        lim = 1.3
        size = 3.9
        pls = [plane((x, 0.05), size, size, (-lim, lim), (-lim, lim)) for x in (-3.3, 3.3)]
        frames = m.VGroup(*[m.Square(size).set_stroke(EDGE, 2).move_to(p.coords(0, 0)) for p in pls])
        unit = size / (2 * lim)
        keys = ("narrow", "sloppy")
        real_d, real_b, gen_d, gen_b = [], [], [], []
        for p, k in zip(pls, keys):
            real_d.append(dots(p, PR_REAL, REAL, 0.032).set_z_index(3))
            real_b.append(m.VGroup(*[m.Circle(radius=max(r, 1e-3) * unit).set_stroke(width=0).set_fill(REAL, 0.13).move_to(p.coords(*q))
                                     for q, r in zip(PR_REAL, R_REAL)]).set_z_index(1))
            g = np.clip(PR_GEN[k], -0.98 * lim, 0.98 * lim)
            gen_d.append(dots(p, g, GEN, 0.032).set_z_index(4))
            gen_b.append(m.VGroup(*[m.Circle(radius=max(r, 1e-3) * unit).set_stroke(width=0).set_fill(GEN, 0.11).move_to(p.coords(*q))
                                    for q, r in zip(g, PR_TOY[k]["radii"])]).set_z_index(1))
        t_k = {"narrow": "draws one arm only", "sloppy": "draws with too much noise"}
        heads = m.VGroup()
        for k, fr in zip(keys, frames):
            h_ = m.VGroup(code("a model that", 22, GEN, weight=m.BOLD), text(t_k[k], 22, INK)).arrange(m.RIGHT, buff=0.18)
            sit(h_[1], baseline(h_[0]))
            heads.add(h_.next_to(fr, m.UP, buff=0.15))
        p_read = m.VGroup(*[label_pair("precision", PR_TOY_TXT[k][0], GREEN if PR_TOY[k]["p"] > 0.9 else RED).next_to(fr, m.DOWN, buff=0.18).align_to(fr, m.LEFT)
                            for k, fr in zip(keys, frames)])
        r_read = m.VGroup(*[label_pair("recall", PR_TOY_TXT[k][1], GREEN if PR_TOY[k]["r"] > 0.9 else RED).next_to(pr_, m.DOWN, buff=0.12).align_to(fr, m.LEFT)
                            for k, fr, pr_ in zip(keys, frames, p_read)])
        legend = cite("for example · 200 swirl points · each ball reaches the third-nearest neighbour · Kynkäänniemi et al., 2019").move_to([0, -3.4, 0])
        q_p = text("look real?", 22, GREEN, weight=m.BOLD).move_to([0, 0.45, 0])
        q_r = text("cover it?", 22, GREEN, weight=m.BOLD).move_to([0, -0.35, 0])
        gap_w = frames[1].get_left()[0] - frames[0].get_right()[0]
        assert q_p.width < gap_w - 0.1 and heads.get_top()[1] < 2.6 and r_read.get_bottom()[1] > legend.get_top()[1] + 0.05, (q_p.width, gap_w, heads.get_top(), r_read.get_bottom())
        assert legend.width < 13.2 and heads[1].get_right()[0] < 6.6, (legend.width, heads[1].get_right())

        # C: the digits
        dp = Plot((0, 8.6), (0.3, 1.0), width=4.6, height=3.1, center=(3.55, -0.3, 0), x_ticks=[1, 3, 8],
                  y_ticks=[0.4, 0.6, 0.8, 1.0], x_label="guidance scale w", y_fmt="{:.1f}")
        p_line = dp.line([1, 3, 8], [PR[w][0] for w in (1, 3, 8)], GREEN, 4)
        r_line = dp.line([1, 3, 8], [PR[w][1] for w in (1, 3, 8)], RED, 4)
        p_pts = m.VGroup(*[dp.dot(w, PR[w][0], GREEN, 0.08) for w in (1, 3, 8)])
        r_pts = m.VGroup(*[dp.dot(w, PR[w][1], RED, 0.08) for w in (1, 3, 8)])
        p_name = text("precision", 22, GREEN, weight=m.BOLD).next_to(dp.coords(8, PR[8][0]), m.UP, buff=0.22)
        r_name = text("recall", 22, RED, weight=m.BOLD).next_to(dp.coords(8, PR[8][1]), m.UP, buff=0.25)
        d_t = cite("our digits, 10,000 per scale").next_to(dp, m.UP, buff=0.15).align_to(dp, m.LEFT)
        tx = (-6.3, -5.15, -3.3, -1.75)
        t_head = m.VGroup(*[text(h_, 22, MUTED).move_to([0, 1.25, 0]).align_to([x, 0, 0], m.LEFT)
                            for h_, x in zip(("scale", "precision", "recall", "label right"), tx)])
        t_rule = m.Line([tx[0], 0.95, 0], [0.3, 0.95, 0]).set_stroke(EDGE, 2)
        t_rows = []
        for i, w in enumerate((1, 3, 8)):
            y = 0.55 - 0.55 * i
            cols = (str(w), PR_TXT[w][0], PR_TXT[w][1], ACC[w])
            colours = (PROMPT, INK if w == 1 else GREEN, INK if w == 1 else RED, INK if w == 1 else GREEN)
            t_rows.append(m.VGroup(*[code(c_, 24, col, weight=m.BOLD).move_to([0, y, 0]).align_to([x, 0, 0], m.LEFT)
                                     for c_, x, col in zip(cols, tx, colours)]))
        t_real = m.VGroup(*[code(c_, 22, REAL).move_to([0, -1.25, 0]).align_to([x, 0, 0], m.LEFT)
                            for c_, x in zip(("real", REAL_PR[0], REAL_PR[1], f"{100 * REFR['cond_acc']:.2f}%"), tx)])
        t_cap = cite("label right: our classifier reads the asked-for digit").move_to([0, -1.85, 0]).align_to([tx[0], 0, 0], m.LEFT)
        pr_cols = m.VGroup(t_head[:3], *[r_[:3] for r_ in t_rows], t_real[:3])
        acc_col = m.VGroup(t_head[3], *[r_[3] for r_ in t_rows], t_real[3])
        trade = pill("one metric rises while another falls", INK, CARD_HI, 22).move_to([0, -3.35, 0])
        assert t_head.get_left()[0] > -6.6 and acc_col.get_right()[0] < dp.get_left()[0] - 0.2, (t_head.get_left(), acc_col.get_right(), dp.get_left())
        assert d_t.get_top()[1] < 2.6 and dp.get_right()[0] < 6.6 and p_name.get_top()[1] < d_t.get_bottom()[1], (d_t.get_top(), dp.get_right(), p_name.get_top())
        assert trade.get_top()[1] < dp.get_bottom()[1] - 0.05 and t_cap.get_bottom()[1] > trade.get_top()[1] + 0.1, (trade.get_top(), dp.get_bottom())
        assert t_real[3].text == "99.55%"

        self.say(
            "[Turn guidance up, as in episode fourteen], [and the score climbs to eighty-five]. [One number cannot say what went wrong].",
            m.AnimationGroup(self.new_header("06 · precision and recall", "Two scores pull apart"), m.FadeIn(gp), m.FadeIn(g_t),
                             m.FadeIn(g_fl), m.FadeIn(grows), m.FadeIn(gcap), m.FadeIn(g_pts[0]), m.FadeIn(g_v1)),
            m.AnimationGroup(m.Create(g_line, run_time=1.4), m.FadeIn(g_pts[1:]), m.FadeIn(g_v8)),
            m.Write(why),
        )
        self.wait(0.3)
        screen_a = m.Group(gp, g_t, g_fl, grows, gcap, g_pts, g_v1, g_v8, g_line, why)
        self.say(
            "[Precision and recall split it in two]. [Draw a ball around each real sample], [out to its third-nearest neighbour].",
            m.AnimationGroup(m.FadeOut(screen_a), m.FadeIn(frames), m.FadeIn(legend),
                             *[m.LaggedStart(*[m.FadeIn(d_, scale=0.5) for d_ in rd], lag_ratio=0.004, run_time=1.0) for rd in real_d]),
            m.AnimationGroup(*[m.LaggedStart(*[m.GrowFromCenter(c_) for c_ in rb], lag_ratio=0.004, run_time=1.4) for rb in real_b]),
            m.Wait(0.2),
        )
        self.say(
            "[Precision is the share of generated samples inside the real balls]: [do they look real]?",
            m.AnimationGroup(m.FadeIn(heads), *[m.LaggedStart(*[m.FadeIn(d_, scale=0.5) for d_ in gd], lag_ratio=0.004, run_time=1.2) for gd in gen_d]),
            m.AnimationGroup(*[recolour(gd, np.flatnonzero(~PR_TOY[k]["inside"]), GEN, RED) for gd, k in zip(gen_d, keys)],
                             m.FadeIn(p_read), m.FadeIn(q_p)),
        )
        self.say(
            "[Recall turns it around]: [how much of the real data do the generated balls cover]?",
            m.AnimationGroup(*[m.FadeOut(rb) for rb in real_b],
                             *[recolour(gd, np.flatnonzero(~PR_TOY[k]["inside"]), RED, GEN, 0.6) for gd, k in zip(gen_d, keys)]),
            m.AnimationGroup(*[m.LaggedStart(*[m.GrowFromCenter(c_) for c_ in gb], lag_ratio=0.004, run_time=1.2) for gb in gen_b],
                             *[recolour(rd, np.flatnonzero(~PR_TOY[k]["covered"]), REAL, RED, 1.2) for rd, k in zip(real_d, keys)],
                             m.FadeIn(r_read), m.FadeIn(q_r)),
        )
        self.say(
            "[The one-arm model: high precision, low recall]. [The noisy one: the opposite].",
            m.Circumscribe(m.VGroup(p_read[0], r_read[0]), color=INK, buff=0.08),
            m.Circumscribe(m.VGroup(p_read[1], r_read[1]), color=INK, buff=0.08),
        )
        self.wait(0.3)
        screen_b = m.VGroup(frames, legend, *real_d, *gen_d, *gen_b, heads, p_read, r_read, q_p, q_r)
        self.say(
            "[On our digits, guidance raises precision from zero point eight three to as much as zero point eight nine].",
            m.AnimationGroup(m.FadeOut(screen_b), m.FadeIn(dp), m.FadeIn(d_t), m.FadeIn(t_rule), m.FadeIn(pr_cols[0]),
                             m.LaggedStart(*[m.FadeIn(r_[:2], shift=0.05 * m.UP) for r_ in t_rows], lag_ratio=0.25),
                             m.Create(p_line, run_time=1.4), m.FadeIn(p_pts), m.FadeIn(p_name)),
        )
        self.say(
            "[Recall falls from zero point eight nine to zero point three eight], "
            "[and label accuracy climbs to ninety-nine point nine nine percent].",
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(r_[2], shift=0.05 * m.UP) for r_ in t_rows], lag_ratio=0.25),
                             m.Create(r_line, run_time=1.4), m.FadeIn(r_pts), m.FadeIn(r_name)),
            m.AnimationGroup(m.FadeIn(acc_col, shift=0.05 * m.UP), m.FadeIn(t_real[:3]), m.FadeIn(t_cap)),
        )
        self.say(
            "[One metric rises while another falls]: [the bold digits look real], [but the variety is gone].",
            m.FadeIn(trade, shift=0.1 * m.UP),
            m.Circumscribe(p_name, color=GREEN, buff=0.06),
            m.Circumscribe(r_name, color=RED, buff=0.06),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 7 · prompts and people

    def prompts_and_people(self):
        self.next_section("7 · The prompt, and people", notes=(
            "CLIPScore (Hessel et al., arXiv:2104.08718, EMNLP 2021): CLIP-S(c, v) = 2.5·max(cos(c, v), 0) with CLIP ViT-B/32 "
            "(notes §19); CLIP itself: Episode 15. The angle drawn is ILLUSTRATIVE (for example; we did not run CLIP). "
            "Imagen (Saharia et al. 2022, arXiv:2205.11487, §2.3, read via ar5iv October 2026): 'increasing the "
            "classifier-free guidance weight improves image-text alignment, but damages image fidelity'; Imagen reports "
            "trade-off curves of CLIP score vs FID across guidance weights. SDXL (Podell et al. 2023, §2.5): 'when using "
            "classical performance metrics such as FID and CLIP scores the improvements of SDXL over previous methods are "
            "not reflected'; it leads with human preference. Preference models [DATED: 2023 papers, read October 2026]: "
            "PickScore (Pick-a-Pic, 583,747 training examples), ImageReward (137k expert comparisons), HPS v2 (798,090 "
            "choices on 433,760 pairs). The two lighthouses: generated on this Mac, SD 2.1 base, 50 steps, guidance 7.5, "
            "seeds 0 and 2. The vote split is ILLUSTRATIVE."))
        pic = picture(LH_A, 1.9).move_to([-5.2, 1.0, 0])
        pic_f = m.Rectangle(width=1.98, height=1.98).set_stroke(EDGE, 2).move_to(pic)
        pic_c = cite("generated on this Mac").next_to(pic_f, m.DOWN, buff=0.1)
        p_words = code("a lighthouse on a\nrocky coast at dusk,\noil painting", 20, INK)
        prompt = boxed(p_words, PROMPT, 0.2, fill=PINK_FILL).move_to([-4.75, -1.65, 0])
        enc_i = chip("image encoder", INK, CARD_HI, width=2.2, height=0.62, size=20).move_to([-2.05, 1.0, 0])
        enc_t = chip("text encoder", INK, CARD_HI, width=2.0, height=0.62, size=20).move_to([-1.5, -1.65, 0])
        a_i = arrow_to(pic_f.get_right(), enc_i.get_left(), REAL, buff=0.1)
        a_t = arrow_to(prompt.get_right(), enc_t.get_left(), PROMPT, buff=0.08)
        O = np.array([0.9, -1.0, 0.0])
        R_ = 2.0
        th_t, th_i = np.deg2rad(16), np.deg2rad(64)
        tip = lambda th: O + R_ * np.array([np.cos(th), np.sin(th), 0])  # noqa: E731
        v_t = m.Arrow(O, tip(th_t), buff=0, color=PROMPT, stroke_width=6, max_tip_length_to_length_ratio=0.12)
        v_i = m.Arrow(O, tip(th_i), buff=0, color=REAL, stroke_width=6, max_tip_length_to_length_ratio=0.12)
        arc = m.Arc(radius=0.8, start_angle=th_t, angle=th_i - th_t, arc_center=O).set_stroke(INK, 3)
        theta = tex(r"\theta", 34, INK).move_to(O + 1.25 * np.array([np.cos((th_t + th_i) / 2), np.sin((th_t + th_i) / 2), 0]))
        l_t = text("the prompt", 20, PROMPT, weight=m.BOLD).next_to(tip(th_t), m.RIGHT, buff=0.12)
        l_i = text("the picture", 20, REAL, weight=m.BOLD).next_to(tip(th_i), m.RIGHT, buff=0.12)
        b_i = arrow_to(enc_i.get_right(), O + [-0.1, 0.2, 0], MUTED, buff=0.1)
        b_t = arrow_to(enc_t.get_right(), O + [-0.12, -0.05, 0], MUTED, buff=0.1)
        f = tex(r"\text{CLIP score} = 2.5 \cdot \max(\cos\theta,\ 0)", 34, INK).move_to([3.3, 2.1, 0])
        f_c = cite("Hessel et al., 2021 · the angle drawn is an example").next_to(f, m.DOWN, buff=0.12)
        meas = text("prompt following,\nnot quality", 24, PROMPT, weight=m.BOLD).move_to([5.12, 0.88, 0])
        assert f.get_right()[0] < 6.6 and f.get_top()[1] < 2.6 and l_t.get_right()[0] < 6.6 and meas.get_bottom()[1] > 0.45 and meas.get_top()[1] < f_c.get_bottom()[1] - 0.1, (f.get_right(), f.get_top(), l_t.get_right())
        assert tip(th_i)[1] + 0.3 < f_c.get_bottom()[1] and meas.get_right()[0] < 6.6 and prompt.get_left()[0] > -6.6, (tip(th_i), f_c.get_bottom(), meas.get_right(), prompt.get_left())
        assert prompt.get_right()[0] < enc_t.get_left()[0] - 0.3 and pic_c.get_bottom()[1] > prompt.get_top()[1] + 0.1, (prompt.get_right(), enc_t.get_left(), pic_c.get_bottom(), prompt.get_top())
        th_g = np.deg2rad(42)
        quote = m.VGroup(text("“" + IMAGEN_QUOTE + "”", 20, INK), cite("Saharia et al., 2022 (Imagen)")).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        q_box = boxed(quote, PROMPT, 0.22)
        q_box.move_to([6.5 - q_box.width / 2, -2.68, 0])
        assert q_box.get_bottom()[1] > -3.6 and q_box.get_left()[0] > enc_t.get_right()[0] + 0.2 and q_box.get_top()[1] < O[1] - 0.25, (q_box.get_bottom(), q_box.get_left(), q_box.get_top())

        def turn(seconds=1.4):
            def step(_, a):
                th = th_i + (th_g - th_i) * a
                v_i.put_start_and_end_on(O, tip(th))
                arc.become(m.Arc(radius=0.8, start_angle=th_t, angle=th - th_t, arc_center=O).set_stroke(INK, 3))
                theta.move_to(O + 1.25 * np.array([np.cos((th_t + th) / 2), np.sin((th_t + th) / 2), 0]))
                l_i.next_to(tip(th), m.RIGHT, buff=0.12)
            return m.UpdateFromAlphaFunc(m.VGroup(v_i, arc, theta, l_i), step, run_time=seconds, rate_func=m.smooth)

        # people
        pa = picture(LH_A, 2.5).move_to([-4.85, 0.75, 0])
        pb = picture(LH_B, 2.5).move_to([-1.95, 0.75, 0])
        fa = m.Rectangle(width=2.58, height=2.58).set_stroke(EDGE, 2).move_to(pa)
        fb = m.Rectangle(width=2.58, height=2.58).set_stroke(EDGE, 2).move_to(pb)
        la = code("A", 26, INK, weight=m.BOLD).next_to(fa, m.UP, buff=0.1)
        lb = code("B", 26, INK, weight=m.BOLD).next_to(fb, m.UP, buff=0.1)
        q_ = text("which one is better?", 24, PROMPT, weight=m.BOLD).move_to([-3.4, -1.0, 0])
        vote_a = m.Rectangle(width=5.4 * 0.62, height=0.3).set_stroke(width=0).set_fill(GREEN, 0.8)
        vote_b = m.Rectangle(width=5.4 * 0.38, height=0.3).set_stroke(width=0).set_fill(MUTED, 0.5)
        votes = m.VGroup(vote_a, vote_b).arrange(m.RIGHT, buff=0).move_to([-3.4, -1.6, 0])
        v_l = m.VGroup(code("A 62%", 20, GREEN).next_to(votes, m.DOWN, buff=0.1).align_to(votes, m.LEFT),
                       code("B 38%", 20, MUTED).next_to(votes, m.DOWN, buff=0.1).align_to(votes, m.RIGHT))
        p_cap = cite("for example · two seeds, generated on this Mac").move_to([-3.4, -2.45, 0])
        rows = m.VGroup()
        for name, what, who in PREF:
            r_ = m.VGroup(code(name, 22, PROMPT, weight=m.BOLD), text(what, 20, INK), cite(who)).arrange(m.DOWN, buff=0.06, aligned_edge=m.LEFT)
            rows.add(r_)
        rows.arrange(m.DOWN, buff=0.2, aligned_edge=m.LEFT)
        pref_t = text("scoring models learned from votes", 22, INK, weight=m.BOLD)
        pref_in = m.VGroup(pref_t, rows).arrange(m.DOWN, buff=0.22, aligned_edge=m.LEFT)
        pref = boxed(pref_in, PROMPT, 0.28).move_to([3.6, 0.3, 0])
        caveat = text("raters, prompts and the question asked all change the verdict", 22, INK).move_to([0, -3.0, 0])
        sdxl = cite("SDXL's paper leads with people's votes: FID and CLIP score missed its gains (Podell et al., 2023)").move_to([0, -3.45, 0])
        assert pref.get_right()[0] < 6.6 and pref.get_top()[1] < 2.6 and pref.get_bottom()[1] > caveat.get_top()[1] + 0.05, (pref.get_right(), pref.get_top(), pref.get_bottom())
        assert fa.get_left()[0] > -6.6 and la.get_top()[1] < 2.6 and fb.get_right()[0] < pref.get_left()[0] - 0.2, (fa.get_left(), la.get_top(), fb.get_right())
        assert p_cap.get_bottom()[1] > caveat.get_top()[1] + 0.05 and sdxl.width < 13.2, (p_cap.get_bottom(), caveat.get_top(), sdxl.width)

        self.say(
            "[None of these scores ever reads the prompt]. [The CLIP score does]: [the angle between the prompt's embedding and the picture's].",
            m.AnimationGroup(self.new_header("07 · the prompt", "Does it follow the prompt?"), m.FadeIn(pic), m.Create(pic_f),
                             m.FadeIn(pic_c), m.FadeIn(prompt, shift=0.1 * m.UP)),
            m.AnimationGroup(m.GrowArrow(a_i), m.GrowArrow(a_t), m.FadeIn(enc_i), m.FadeIn(enc_t), m.Write(f), m.FadeIn(f_c)),
            m.AnimationGroup(m.GrowArrow(b_i), m.GrowArrow(b_t), m.GrowArrow(v_t), m.GrowArrow(v_i), m.FadeIn(l_t),
                             m.FadeIn(l_i), m.Create(arc), m.FadeIn(theta)),
        )
        self.say(
            "[It measures prompt following, not quality]: [more guidance raises it], [even as the pictures lose fidelity].",
            m.FadeIn(meas, shift=0.1 * m.UP),
            turn(1.4),
            m.FadeIn(q_box, shift=0.1 * m.UP),
        )
        self.wait(0.3)
        screen_a = m.Group(pic, pic_f, pic_c, prompt, enc_i, enc_t, a_i, a_t, b_i, b_t, v_t, v_i, arc, theta, l_t, l_i, f, f_c, meas, q_box)
        self.say(
            "[So the final judge is people]: [two pictures, one prompt], [which one is better]?",
            m.AnimationGroup(self.new_header("07 · people", "Ask people"), m.FadeOut(screen_a)),
            m.AnimationGroup(m.FadeIn(pa), m.FadeIn(pb), m.Create(fa), m.Create(fb), m.FadeIn(la), m.FadeIn(lb)),
            m.AnimationGroup(m.FadeIn(q_, shift=0.1 * m.UP), m.GrowFromEdge(votes, m.LEFT), m.FadeIn(v_l), m.FadeIn(p_cap)),
        )
        self.say(
            "[Hundreds of thousands of such votes now train scoring models], "
            "[but raters, prompts and the question asked all change the verdict].",
            m.AnimationGroup(m.FadeIn(pref[0]), m.FadeIn(pref_t), m.LaggedStart(*[m.FadeIn(r_, shift=0.1 * m.UP) for r_ in rows], lag_ratio=0.3, run_time=1.6)),
            m.AnimationGroup(m.FadeIn(caveat, shift=0.1 * m.UP), m.FadeIn(sdxl)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes=(
            f"Statistics: FID compares only averages and spreads of features (the matched blob scores {BLOB_S} with "
            f"{BLOB_OFF} of its points off the swirl); our digit score is FID's formula on our own classifier's features "
            f"(floor {FLOOR:.2f}, ± about 0.35; not Inception FID). Samples: the score is biased at small n (2-D: "
            f"{B_TXT[(2, 100)]} at 100; 64-D: {B_TXT[(64, 1000)]} at 1,000), so FID uses 50,000; precision and recall "
            f"split fidelity from coverage (w = 1 → 8: {PR_TXT[1]} → {PR_TXT[8]}). People: the CLIP score checks the "
            "prompt; human votes and learned preference models judge the rest. Next: Episode 20, Data and Alignment."))
        halo = glow(REAL, radius=4.0)
        words = m.VGroup(*[title_text(w, 58, c) for w, c in (("Statistics", REAL), ("Samples", GEN), ("People", PROMPT))])
        subs = m.VGroup(*[code(s, 20, MUTED) for s in ("averages and spreads,", "many of them,", "the prompt,")])
        subs2 = m.VGroup(*[code(s, 20, MUTED) for s in ("not shapes", "scored two ways", "and the verdict")])
        words.arrange(m.RIGHT, buff=1.6)
        on_baseline(words)
        line_y = min(w_.get_bottom()[1] for w_ in words) - 0.45
        for w_, s, s2 in zip(words, subs, subs2):
            s.set_x(w_.get_x())
            s2.set_x(w_.get_x())
            sit(s, line_y)
            sit(s2, line_y - 0.36)
        mid = baseline(words[0]) + 0.3
        dts = m.VGroup(*[m.Dot([(l_.get_right()[0] + r_.get_left()[0]) / 2, mid, 0], radius=0.07, color=MUTED)
                         for l_, r_ in zip(words, words[1:])])
        row = m.VGroup(words, dts, subs, subs2).move_to(0.9 * m.UP)
        assert row.width < 12.8 and all(a.get_right()[0] + 0.3 < b.get_left()[0] for a, b in zip(subs, subs[1:])), [(s_.get_left()[0], s_.get_right()[0]) for s_ in subs]
        tag = text("Every metric has a blind spot: read several, and look at the pictures.", 28, MUTED)
        mascot = pixel(0.42)
        nxt = pill("NEXT · DATA AND ALIGNMENT", REAL, TEAL_FILL, 20, mono=True)
        foot = m.VGroup(tag, m.VGroup(mascot, nxt).arrange(m.RIGHT, buff=0.3)).arrange(m.DOWN, buff=0.4)
        foot.next_to(row, m.DOWN, buff=0.7)
        assert foot.get_bottom()[1] > -3.6 and tag.width < 13.0, (foot.get_bottom(), tag.width)

        self.say(
            "[To sum up]: [statistics], [samples], [people].",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.3 * m.UP), m.FadeIn(subs[0]), m.FadeIn(subs2[0])),
            m.AnimationGroup(m.FadeIn(dts[0]), m.FadeIn(words[1], shift=0.3 * m.UP), m.FadeIn(subs[1]), m.FadeIn(subs2[1])),
            m.AnimationGroup(m.FadeIn(dts[1]), m.FadeIn(words[2], shift=0.3 * m.UP), m.FadeIn(subs[2]), m.FadeIn(subs2[2])),
        )
        self.say(
            "[Every metric has a blind spot]: [read several at once], [and always look at the pictures].",
            m.Write(tag),
            m.Wait(0.1),
            m.AnimationGroup(m.FadeIn(mascot, shift=0.2 * m.UP), m.FadeIn(nxt, shift=0.2 * m.UP)),
        )
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, foot)], run_time=0.9)
