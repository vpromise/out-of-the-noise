"""Out of the Noise · Episode 14 · Classifier-Free Guidance (Module 4 · Steering the Picture)

Push away from "anything" toward "what I asked for". Running examples: the digit model (the measured trade-off,
mac_runs.md §4) and the lighthouse (why everyone uses guidance on text prompts, mac_runs.md §5), with the swirl and the
three-blob toy (diffusion_reference: exact scores, toy.md §5) for the mechanism.

ONE CONVENTION ON SCREEN: the guidance SCALE w, ε̃ = ε_u + w·(ε_c − ε_u) (diffusion_reference.cfg; Stable Diffusion,
diffusers, our digit run and our lighthouse run all use it): w = 0 ignores the label, w = 1 is the plain conditional
model (no guidance), w > 1 pushes past it. Ho & Salimans write ε̃ = (1 + w′)·ε_c − w′·ε_u, so their w′ = our w − 1
(Stable Diffusion's 7.5 is their 6.5; their best ImageNet-128 FID, 2.43 at w′ = 0.3, is scale 1.3).

The honest story:
- The swirl's arms only CONCENTRATE guided samples (arm coverage 1.00 → 0.45 at w = 7, 98 % on the arm's outer half,
  spread 0.68 → 0.46, 0 % off the swirl): the thin arm catches them again at low noise. Samples pushed BEYOND the data
  are shown on the three-blob toy (after Ho & Salimans' Fig. 2): at w = 7 the class's centre moves from height 0.60
  to 0.96 and 86 % of samples lie beyond 2 std of the class (real data 14 %).
- Our digit model (10 % label dropout): guidance raises label accuracy 98.1 % → 99.99 % and precision 0.83 → 0.89 (at
  w = 3; 0.87 at w = 8), but recall falls 0.89 → 0.38, diversity 1.01 → 0.74, and cFID gets WORSE: 3.35 / 21.4 / 42.0 /
  66.9 / 85.4 at w = 1 / 2 / 3 / 5 / 8. cFID = FID's formula on our own MNIST classifier's features, NOT Inception FID
  (floor 2.06). On these digits the best cFID is with no guidance.
- The lighthouse (SD 2.1 base, 50 steps): guidance 1 is washed out; 3–7.5 coherent; 10–15 over-saturated. Mean
  saturation (seeds 0–3) 0.19 → 0.28 → 0.36 at 1 / 7.5 / 15; 9.3 % of seed 0's pixels clip at 15. Guided steps take
  0.141 s vs 0.083 s (1.7×, the two passes share one batch).

Every number on screen is computed below (diffusion_reference, the run's arrays and PNGs, measured(...)) and asserted,
or cited with its source. Illustrative: the two prediction arrows of section 2 (directions chosen, "for example").

    VOICE=draft manimgx inspect scene.py      # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403
from diffusion_reference import *  # noqa: E402,F403

EP_DIR = Path(__file__).resolve().parent  # set AFTER the import: diffusion_reference exports HERE (the series folder)
warnings.filterwarnings("ignore", message=".*encountered in matmul")  # NumPy/Accelerate's spurious warnings

PREFIX = "ep14"
ARR = EXPERIMENTS / "arrays"
IMG = EXPERIMENTS / "images"
H = lambda *keys: measured("mac_runs", "headline", *keys)  # noqa: E731
TOY = lambda *keys: measured("toy", "headline", *keys)  # noqa: E731
GJ = lambda *keys: measured("toy", "guidance", *keys)  # noqa: E731
MD = (EXPERIMENTS / "mac_runs.md").read_text()
TOY_MD = (EXPERIMENTS / "toy.md").read_text()
NOTES = (SERIES / "research" / "notes.md").read_text()

UNC = MODEL        # the unconditional prediction, "any digit" (AMBER: the model on its own)
CON = PROMPT       # the conditional prediction, "what I asked for" (PINK: conditioning)
GUI = FLOW         # the guided prediction, the push (LIME: scores and arrows)
ORIGIN_ = np.zeros(3)

# ---------------------------------------------------------------------------- the convention (diffusion_reference.cfg)

assert cfg(0.2, 0.5, 0) == 0.2 and cfg(0.2, 0.5, 1) == 0.5                     # w = 0: unlabelled; w = 1: labelled
assert abs(cfg(0.2, 0.5, 7.5) - (6.5 + 1) * 0.5 + 6.5 * 0.2) < 1e-12              # scale 7.5 = Ho & Salimans' w′ = 6.5
assert abs(cfg(0.2, 0.5, 7.5) - 2.45) < 1e-12 and "| **7.5 (w = 6.5)** | **2.45** |" in NOTES   # notes §14 example
CONV = measured("mac_runs", "digit_sampling", "guidance_convention")
assert "pred = uncond + w (cond - uncond)" in CONV and "w_HS = w - 1" in CONV and "w=1 plain conditional" in CONV
assert "`eps_neg + w·(eps_text − eps_neg)` with CFG off for w ≤ 1" in MD          # the lighthouse run: the same form
CFG_SRC = "def cfg(eps_uncond, eps_cond, w):\n    return eps_uncond + w * (eps_cond - eps_uncond)"
assert "return eps_uncond + w * (eps_cond - eps_uncond)" in (SERIES / "diffusion_reference.py").read_text()
assert "SD's default 7.5 is w = 6.5" in NOTES and "s = 1 + w" in NOTES

# ---------------------------------------------------------------------------- label dropout (the run's own config)

TRAIN = json.loads((WEIGHTS / "digit_ddpm_ema.json").read_text())
P_UNCOND = TRAIN["p_uncond"]
assert P_UNCOND == 0.1 and TRAIN["params"] == H("digit_unet_params") == 3_940_801 and "10% label dropout (for CFG)" in MD
ARCH = measured("mac_runs", "architecture", "ddpm", "config")
N_ROWS = ARCH["num_classes"] + 1
assert N_ROWS == 11 and "11x128 table" in MD and "(index 10 = \"no label\")" in MD
T7 = np.load(ARR / "digit_trajectory_7.npz")
assert int(T7["label"]) == 7
XT_K = 25
XT_T = int(T7["ddpm_ddim50_w1_t"][XT_K])
XT7 = T7["ddpm_ddim50_w1_x_t"][XT_K].astype(np.float64)
assert XT_T == 489
assert "c ← ∅ with probability p_uncond" in NOTES and "\"10% dropping of the text-conditioning\"" in NOTES

# ---------------------------------------------------------------------------- the swirl, guided with exact scores (toy.md §5)

Z = gaussian_noise(2000, seed=5)                                                  # the toy's shared noise
SW_WS = (0, 1, 3, 7)
SW_RUN = {w: ddim_sample(guided_eps(w, 0), Z, steps=100) for w in SW_WS}
SW_STATS = {}
for _w in SW_WS:
    _x = SW_RUN[_w].xs[-1]
    _d, _arm, _u = nearest_on_swirl(_x)
    _prof = swirl_profile(_x)[0]
    SW_STATS[_w] = dict(on0=float(np.mean((_arm == 0) & (_d < 0.1))), cover=swirl_coverage(_x, arm=0),
                        spread=float(np.sqrt(np.trace(np.cov(_x, rowvar=False)))), off=off_swirl(_x),
                        outer=float(_prof[10:].sum() / _prof.sum()))
    _j = GJ("runs", str(_w))
    assert abs(SW_STATS[_w]["on0"] - _j["on_arm0"]) < 1e-3 and abs(SW_STATS[_w]["cover"] - _j["coverage_arm0"]) < 1e-9
    assert abs(SW_STATS[_w]["spread"] - _j["spread"]) < 1e-4 and abs(SW_STATS[_w]["off"] - _j["off"]) < 1e-3
assert [f"{100 * SW_STATS[w]['outer']:.0f}" for w in SW_WS] == ["51", "52", "77", "98"]   # "98% on the outer half"
assert f"{SW_STATS[7]['cover']:.2f}" == "0.45" == f"{TOY('swirl_w7_coverage_arm0'):.2f}"
assert (f"{SW_STATS[1]['spread']:.2f}", f"{SW_STATS[7]['spread']:.2f}") == ("0.68", "0.46")
assert TOY("swirl_w1_spread") == 0.6768 and TOY("swirl_w7_spread") == 0.4601
assert SW_STATS[7]["off"] == 0 and SW_STATS[3]["off"] == 0 and round(100 * SW_STATS[0]["on0"]) == 49
SW_LIM = 1.55
_inside = np.all(np.abs(np.concatenate([SW_RUN[w].xs for w in SW_WS])) < 0.97 * SW_LIM, axis=(0, 2))
SW_SHOW = np.flatnonzero(_inside)[:450]
assert len(SW_SHOW) == 450

# ---------------------------------------------------------------------------- the three blobs (toy.md §5, after Ho & Salimans' Fig. 2)

BM = blobs_mixture()
M0, S0 = BM.means[0], float(BM.stds[0])
assert np.allclose(M0, [0, 0.6]) and S0 == 0.15
BL_WS = (0, 1, 3, 7)
BL = {w: ddim_sample(guided_eps(w, 0, BM), Z, steps=100).xs[-1] for w in BL_WS}
BL_STATS = {}
for _w in BL_WS:
    _x = BL[_w]
    _k = np.argmin(((_x[:, None] - BM.means[None]) ** 2).sum(-1), 1)
    _r = np.linalg.norm(_x - M0, axis=1) / S0
    BL_STATS[_w] = dict(class0=float(np.mean(_k == 0)), centre=_x[_k == 0].mean(0),
                        beyond=float(np.mean(_r[_k == 0] > 2)))
    _j = GJ("blobs", "runs", str(_w))
    assert abs(BL_STATS[_w]["centre"][1] - _j["centre_y"]) < 1e-4 and abs(BL_STATS[_w]["beyond"] - _j["beyond_2sd"]) < 1e-3
_data0 = BM.given(0).sample(2000, seed=3)[0]
BL_DATA_BEYOND = float(np.mean(np.linalg.norm(_data0 - M0, axis=1) / S0 > 2))
BL_DATA_CENTRE = float(_data0[:, 1].mean())
assert abs(BL_DATA_BEYOND - GJ("blobs", "data_class0", "beyond_2sd")) < 1e-3 and TOY("blobs_data_beyond_2sd") == 0.143
assert TOY("blobs_w1_centre_y") == 0.6032 and TOY("blobs_w7_centre_y") == 0.9634 and TOY("blobs_w7_beyond_2sd") == 0.859
assert BL_STATS[1]["class0"] == BL_STATS[7]["class0"] == 1.0
BL_ROWS = [("data", f"{BL_DATA_CENTRE:.2f}", f"{100 * BL_DATA_BEYOND:.0f}%")] + \
          [(str(w), f"{BL_STATS[w]['centre'][1]:.2f}", f"{100 * BL_STATS[w]['beyond']:.0f}%") for w in (1, 3, 7)]
assert BL_ROWS == [("data", "0.60", "14%"), ("1", "0.60", "9%"), ("3", "0.79", "16%"), ("7", "0.96", "86%")], BL_ROWS
BL_X, BL_Y = (-1.2, 1.2), (-0.85, 1.55)
_bl_in = np.all([(np.abs(BL[w][:, 0]) < 1.15) & (BL[w][:, 1] > -0.8) & (BL[w][:, 1] < 1.5) for w in BL_WS], axis=0)
BL_SHOW = np.flatnonzero(_bl_in)[:1000]
assert len(BL_SHOW) == 1000

# ---------------------------------------------------------------------------- the digit model at w = 0 … 8 (mac_runs.md §4)

DW = (0, 1, 2, 3, 5, 8)
SAMPLES = {}
for _w in DW:
    _f = np.load(ARR / f"digit_samples_ddpm_ddim50_w{_w}.npz")
    assert "initial noise seed 1234" in str(_f["note"]) and np.all(_f["labels"][:20] == np.arange(20) % 10)
    SAMPLES[_w] = _f["samples"]
    assert SAMPLES[_w].shape == (10000, 28, 28)
GRID = {w: SAMPLES[w][10:20] for w in DW}                                         # samples 10 … 19 ask for 0 … 9
INK_FR = {w: float((SAMPLES[w] > 127).mean()) for w in DW}
INK_S = {w: f"{INK_FR[w]:.3f}" for w in DW}
assert [INK_S[w] for w in DW] == ["0.133", "0.133", "0.146", "0.155", "0.169", "0.182"]   # mac_runs.md §4 ink column
assert isinstance(INK, str)                                                       # the kit colour, not shadowed
assert "0.133 → 0.182" in MD
SEVENS = [7 + 10 * k for k in range(12)]                                          # twelve 7s, the same noises
CFGS = measured("mac_runs", "digit_sampling", "configs")
MET = {w: CFGS[f"ddpm_ddim50_w{w}"] for w in DW}
for _w in DW:
    assert MET[_w]["w"] == _w and MET[_w]["steps"] == 50 and MET[_w]["sampler"] == "ddim"
    assert abs(MET[_w]["cfid"] - H(f"guidance_w{_w}_cfid")) < 1e-3 and abs(MET[_w]["cond_acc"] - H(f"guidance_w{_w}_cond_acc")) < 1e-4
ACC = {w: f"{100 * MET[w]['cond_acc']:.2f}%" for w in DW}
PREC = {w: f"{MET[w]['precision']:.2f}" for w in DW}
REC = {w: f"{MET[w]['recall']:.2f}" for w in DW}
DIV = {w: (f"{MET[w]['diversity']:.2f}" if w else "n/a") for w in DW}
CFID = {w: MET[w]["cfid"] for w in DW}
CF = {w: (f"{CFID[w]:.2f}" if CFID[w] < 10 else f"{CFID[w]:.1f}") for w in DW}
assert [ACC[w] for w in (1, 8)] == ["98.10%", "99.99%"] and ACC[0] == "9.93%"
assert [PREC[w] for w in DW] == ["0.76", "0.83", "0.88", "0.89", "0.88", "0.87"]
assert [REC[w] for w in DW] == ["0.90", "0.89", "0.79", "0.70", "0.54", "0.38"]
assert [DIV[w] for w in DW] == ["n/a", "1.01", "0.87", "0.83", "0.78", "0.74"]
assert [CF[w] for w in DW] == ["22.3", "3.35", "21.4", "42.0", "66.9", "85.4"]
assert min(CFID, key=CFID.get) == 1                                               # best with no guidance
FLOOR = H("cfid_real_data_floor")
assert f"{FLOOR:.2f}" == "2.06" and "**This is not Inception FID.**" in MD
SEC_1000 = {w: MET[w]["s_per_1000"] for w in (1, 3)}
assert (f"{SEC_1000[1]:.1f}", f"{SEC_1000[3]:.1f}") == ("8.9", "19.4")

# Ho & Salimans, Table 2: ImageNet 128×128, 256 sampling steps, FID (Inception), their w′ = scale − 1
# (arXiv:2207.12598 via ar5iv, read October 2026; four of these rows are quoted in notes §14).
HS_WP = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0)
HS_FID = (7.27, 4.53, 3.03, 2.43, 2.49, 2.98, 3.76, 4.67, 5.74, 6.8, 7.86, 15.93, 19.77, 21.53)
for _s in ("w = 0: FID 8.11 / 7.27 / 7.22", "w = 0.3: FID 3.04 / 2.43 / 2.43", "w = 1.0: 7.88 / 7.86 / 7.8",
           "w = 4.0: 21.55 / 21.53 / 21.45"):
    assert _s in NOTES, _s
HS_SCALE = tuple(round(1 + w, 1) for w in HS_WP)
HS_BEST = HS_SCALE[int(np.argmin(HS_FID))]
assert HS_BEST == 1.3 and min(HS_FID) == 2.43

# ---------------------------------------------------------------------------- the lighthouse (mac_runs.md §5)

LH = IMG / "lighthouse"
LH_SCALES = ("1", "3", "5", "7.5", "10", "15")


def colour_stats(path):
    """Mean HSV saturation and the share of pixels with a channel at 0 or 255 (gpu/lighthouse.py's own definitions)."""
    a = np.asarray(Image.open(path).convert("RGB"))
    im = a.astype(np.float32) / 255
    hi, lo = im.max(-1), im.min(-1)
    sat = np.where(hi > 0, (hi - lo) / np.maximum(hi, 1e-6), 0)
    return float(sat.mean()), float(((a == 0) | (a == 255)).any(-1).mean())


LH_FILE = {s: LH / f"lighthouse_sd21_cfg{s}_steps50_seed0.png" for s in LH_SCALES}
LH_STATS = {s: [colour_stats(LH / f"lighthouse_sd21_cfg{s}_steps50_seed{k}.png") for k in range(4)] for s in LH_SCALES}
SAT = {s: float(np.mean([v[0] for v in LH_STATS[s]])) for s in LH_SCALES}
CLIP0 = {s: LH_STATS[s][0][1] for s in LH_SCALES}
_sweep = {(r["cfg"], r["seed"]): r for r in measured("mac_runs", "lighthouse", "guidance_sweep")}
for _s in LH_SCALES:
    for _k in range(4):
        _r = _sweep[(float(_s), _k)]
        assert abs(_r["mean_saturation"] - LH_STATS[_s][_k][0]) < 1e-5 and abs(_r["frac_clipped_pixels"] - LH_STATS[_s][_k][1]) < 1e-7
assert (round(SAT["1"], 2), round(SAT["7.5"], 2), round(SAT["15"], 2)) == (0.19, 0.28, 0.36)
assert abs(SAT["1"] - H("lighthouse_cfg1_mean_saturation")) < 1e-3 and abs(SAT["15"] - H("lighthouse_cfg15_mean_saturation")) < 1e-3
assert f"{100 * CLIP0['15']:.1f}" == "9.3" and CLIP0["1"] == 0


def clip_label(v):
    return "0%" if v == 0 else "<0.01%" if v < 5e-5 else f"{100 * v:.2f}%" if v < 1e-3 else f"{100 * v:.1f}%"


CLIP_S = {s: clip_label(CLIP0[s]) for s in LH_SCALES}
assert [CLIP_S[s] for s in LH_SCALES] == ["0%", "0%", "<0.01%", "0.01%", "4.0%", "9.3%"]
STEP_CFG, STEP_ONE = H("lighthouse_unet_step_s_cfg"), H("lighthouse_unet_step_s_nocfg")
assert (STEP_CFG, STEP_ONE) == (0.141, 0.083) and f"{STEP_CFG / STEP_ONE:.1f}" == "1.7"
assert "0.083 s/step, 4.43 s total. CFG costs 1.7x per step on this GPU, not 2x" in MD and "865.9 M" in MD and "866 M-parameter UNet" in MD
LH_NOISY = IMG / "lighthouse_frames30" / "xt_08.png"
assert LH_NOISY.exists()
PROMPT_TEXT = "a lighthouse on a rocky coast at dusk, oil painting"
assert f"Prompt: **\"{PROMPT_TEXT}\"**" in MD
DEFAULTS = (("Stable Diffusion 1.x and 2.x", "7.5"), ("SDXL", "5.0"), ("Stable Diffusion 3", "7.0"),
            ("FLUX.1 [dev]", "3.5"))
assert "diffusers v0.42: SD 7.5, SDXL 5.0, SD3 7.0 (28 steps), FLUX [dev] distilled guidance 3.5" in NOTES
assert "By default, this uses a guidance scale of `--scale 7.5`" in NOTES

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
    """Put the baseline of Text `words` at height y."""
    return words.shift((y - baseline(words)) * m.UP)


def formula(*parts, size=44, color=INK):
    return m.MathTex(*parts, font_size=size, color=color)


def digit_row(images, color, height, gap_px=5, k=8):
    """Several 28×28 digits (uint8) as ONE picture with dark gaps, `height` units tall (crisp: nearest neighbour)."""
    n = len(images)
    canvas = np.zeros((28, n * 28 + (n - 1) * gap_px))
    for i, img in enumerate(images):
        canvas[:, i * (28 + gap_px): i * (28 + gap_px) + 28] = np.asarray(img, dtype=float) / 255
    rgb = np.repeat(np.repeat(tint(canvas, color, 0, 1), k, axis=0), k, axis=1)
    return picture(rgb, height, pixelated=False)


def row_width(n, height, gap_px=5):
    return height * (n * 28 + (n - 1) * gap_px) / 28


class Readout(m.VGroup):
    """A mono number display left-aligned at a fixed point; .show(v) updates it."""

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


def plane(center, size, lim):
    """A bare square plane for the swirl (no axes), data range [−lim, lim]²."""
    return Plot((-lim, lim), (-lim, lim), width=size, height=size, center=tuple(center), axes=False)


def at(plot, xy):
    return plot.coords(float(xy[0]), float(xy[1]))


def arm_band(plot, arm, color, width=9, opacity=0.2):
    pts = swirl_curve(200)[arm]
    return m.VMobject().set_points_smoothly([at(plot, p) for p in pts[::3]] + [at(plot, pts[-1])]) \
        .set_stroke(color, width, opacity=opacity).set_z_index(-1)


def vec(a, b, color, width=6, tip=0.22):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return m.Arrow(a, b, buff=0, color=color, stroke_width=width, max_tip_length_to_length_ratio=tip,
                   max_stroke_width_to_length_ratio=100)


def boxed(group, color=EDGE, buff=0.3, radius=0.16, fill=CARD):
    box = card(group.width + 2 * buff, group.height + 2 * buff, color, fill, radius=radius, stroke=2).move_to(group)
    return m.VGroup(box, group)


def fly(dots, plot, path, seconds, c0=NOISE, c1=DATA):
    """Dots walk a sampler's path (steps × n × 2) and turn from noise to data colour on the way."""
    path = np.asarray(path, dtype=float)
    ramp = [mix(c0, c1, s / 12) for s in range(13)]

    def step(_, a):
        i = a * (len(path) - 1)
        j = int(np.floor(i))
        k = min(j + 1, len(path) - 1)
        now = path[j] + (path[k] - path[j]) * (i - j)
        col = ramp[int(round(12 * a ** 1.5))]
        for d, p in zip(dots, now):
            d.move_to(at(plot, p))
            d.set_color(col)

    return m.UpdateFromAlphaFunc(dots, step, run_time=seconds, rate_func=m.linear)


def glide(dots, plot, a_xy, b_xy, seconds=1.4):
    a_xy, b_xy = np.asarray(a_xy, dtype=float), np.asarray(b_xy, dtype=float)

    def step(_, a):
        now = a_xy + (b_xy - a_xy) * a
        for d, p in zip(dots, now):
            d.move_to(at(plot, p))

    return m.UpdateFromAlphaFunc(dots, step, run_time=seconds, rate_func=m.smooth)


def cross(p, color, r=0.16, width=4):
    p = np.asarray(p, dtype=float)
    return m.VGroup(m.Line(p + [-r, 0, 0], p + [r, 0, 0]), m.Line(p + [0, -r, 0], p + [0, r, 0])).set_stroke(color, width)


def chip_box(label, color, fill, size=24, width=None, height=0.62):
    words = code(label, size, INK)
    box = card(width or words.width + 0.5, height, color, fill, radius=0.14, stroke=2.5)
    return m.VGroup(box, words.move_to(box))


def cells(labels, size=0.46, color=EDGE, fill=CARD, text_size=20, text_color=MUTED):
    out = m.VGroup()
    for lab in labels:
        sq = m.Square(size).set_stroke(color, 2).set_fill(fill, 1)
        out.add(m.VGroup(sq, code(lab, text_size, text_color).move_to(sq)))
    return out.arrange(m.RIGHT, buff=0.06)


# ---------------------------------------------------------------------------- the episode


class Ep14ClassifierFreeGuidance(Episode):
    number = 14

    def lesson(self):
        self.hook()
        self.two_questions()
        self.extrapolate()
        self.beyond_the_data()
        self.digits()
        self.lighthouse()
        self.the_cost()
        self.recap()

    # ------------------------------------------------------------------ 0 · hook

    def hook(self):
        self.next_section("Hook", notes=(
            "The lighthouse, generated on this Mac (mac_runs.md §5): Stable Diffusion 2.1 base (sd2-community mirror) "
            "through ml-explore/mlx-examples, deterministic Euler, 50 steps, 512×512, seed 0, the prompt "
            f"\"{PROMPT_TEXT}\", empty negative prompt. Left: guidance scale 1 (one pass, CFG off); right: 7.5 "
            "(Stable Diffusion's default). Files: images/lighthouse/lighthouse_sd21_cfg{1,7.5}_steps50_seed0.png. "
            "One convention throughout: the guidance scale w (1 = no guidance)."))
        h = 3.4
        imgs = {}
        for key, x, lab, color in (("1", -2.4, "guidance scale 1 (none)", MUTED),
                                   ("7.5", 2.4, "guidance scale 7.5", CON)):
            pic = picture(LH_FILE[key], h).move_to([x, -0.1, 0])
            frame = m.Rectangle(width=h + 0.12, height=h + 0.12).set_stroke(EDGE, 2.5).move_to(pic)
            words = code(lab, 22, color).next_to(frame, m.DOWN, buff=0.18)
            imgs[key] = dict(pic=pic, frame=frame, lab=words)
        p_words = code(PROMPT_TEXT, 22, INK)
        p_box = card(p_words.width + 0.6, 0.6, CON, PINK_FILL, radius=0.14, stroke=2.5)
        prompt = m.VGroup(p_box, p_words.move_to(p_box)).move_to([0, 2.2, 0])
        cap = cite("generated on this Mac · Stable Diffusion 2.1 base · 50 steps · seed 0 · the same prompt and noise").move_to([0, -3.35, 0])
        washed = pill("washed out", RED, RED_FILL, 20).next_to(imgs["1"]["lab"], m.DOWN, buff=0.14)
        clear = pill("a clear painting", GREEN, GREEN_FILL, 20).next_to(imgs["7.5"]["lab"], m.DOWN, buff=0.14)
        washed.align_to(imgs["1"]["frame"], m.LEFT)
        clear.align_to(imgs["7.5"]["frame"], m.LEFT)
        cap.next_to(washed, m.DOWN, buff=0.2).set_x(0)
        mascot = pixel(0.5).move_to([5.6, -1.6, 0])
        assert prompt.get_top()[1] < 2.6 and prompt.get_bottom()[1] > imgs["1"]["frame"].get_top()[1] + 0.05, prompt.get_bottom()
        assert cap.get_bottom()[1] > -3.6 and imgs["7.5"]["frame"].get_right()[0] < mascot.get_left()[0] - 0.2

        def show(key):
            d = imgs[key]
            return m.AnimationGroup(m.FadeIn(d["pic"], scale=0.96), m.Create(d["frame"]), m.FadeIn(d["lab"], shift=0.1 * m.UP))

        self.say(
            "[Same prompt, same seed, same fifty steps]: [two very different lighthouses].",
            m.AnimationGroup(self.new_header("00 · hook", "Same prompt, two pictures"), m.FadeIn(prompt, shift=0.1 * m.DOWN)),
            m.AnimationGroup(show("1"), show("7.5"), m.FadeIn(cap)),
        )
        self.say(
            "[On the left, the model's plain answer]: [washed out, the lighthouse barely there].",
            m.Circumscribe(imgs["1"]["frame"], color=MUTED, buff=0.06),
            m.FadeIn(washed, shift=0.1 * m.UP),
        )
        self.say(
            "[On the right, the same model, with guidance turned up to seven point five].",
            m.AnimationGroup(m.Circumscribe(imgs["7.5"]["frame"], color=CON, buff=0.06), m.FadeIn(clear, shift=0.1 * m.UP)),
        )
        self.say(
            "[Today: how that push works], [what it does to variety], [and what it costs].",
            m.FadeIn(mascot, shift=0.2 * m.UP),
            m.Indicate(washed[1], color=RED, scale_factor=1.15),
            m.Indicate(clear[1], color=GREEN, scale_factor=1.15),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 1 · one network, two questions

    def two_questions(self):
        self.next_section("1 · One network, two questions", notes=(
            "The digit model (mac_runs.md §3): a class-conditional U-Net, 3,940,801 parameters, whose label embedding is "
            f"an {N_ROWS}×128 table; index 10 = \"no label\" (Episode 13). Training replaced the label by index 10 with "
            f"probability p_uncond = {P_UNCOND} (weights/digit_ddpm_ema.json; mac_runs: \"10% label dropout (for CFG)\"), "
            "as in Ho & Salimans' Algorithm 1 (\"c ← ∅ with probability p_uncond\"; they tried 0.1 / 0.2 / 0.5) and "
            "Stable Diffusion v1 (\"10% dropping of the text-conditioning\", notes §14). The noisy 7 is the run's DDIM-50 "
            f"path at code step t = {XT_T} (arrays/digit_trajectory_7.npz). Ho & Salimans, \"Classifier-Free Diffusion "
            "Guidance\": NeurIPS 2021 Workshop on Deep Generative Models; arXiv:2207.12598, 26 Jul 2022 (notes Appendix A)."))
        xt_pic = picture(tint(np.clip(XT7, -1, 1), mix(VIOLET, TEAL, 0.3), -1, 1), 1.35).move_to([-5.15, 1.25, 0])
        xt_frame = m.Rectangle(width=1.45, height=1.45).set_stroke(NOISE, 2.5).move_to(xt_pic)
        xt_lab = tex("x_t", 32, NOISE).next_to(xt_frame, m.LEFT, buff=0.15)
        t_chip = chip_box(f"t = {XT_T}", TIME, BLUE_FILL, 22, width=1.6).move_to([-5.15, -0.05, 0])
        y7 = chip_box("y = 7", CON, PINK_FILL, 22, width=1.6).move_to([-5.15, -0.85, 0])
        yn = chip_box("y = none", MUTED, CARD, 22, width=1.6).move_to([-5.15, -1.65, 0])
        box = card(3.0, 1.75, MODEL, AMBER_FILL)
        box_t = m.VGroup(text("denoiser", 30, INK, weight=m.BOLD), code("3.94 M parameters", 20, MUTED)).arrange(m.DOWN, buff=0.12)
        net = m.VGroup(box, box_t.move_to(box)).move_to([-1.35, 0.15, 0])
        left = box.get_left()
        ins = m.VGroup(*[m.Arrow(src.get_right() + [0.08, 0, 0], left + [-0.05, dy, 0], buff=0.05, color=c, stroke_width=4,
                                 max_tip_length_to_length_ratio=0.12)
                         for src, dy, c in ((xt_frame, 0.55, NOISE), (t_chip, 0.18, TIME), (y7, -0.2, CON), (yn, -0.55, MUTED))])
        out_c = m.VGroup(tex(r"\varepsilon_c", 46, CON), text("the noise, if it is a seven", 24, CON)).arrange(m.RIGHT, buff=0.3)
        out_u = m.VGroup(tex(r"\varepsilon_u", 46, UNC), text("the noise, if it is any digit", 24, UNC)).arrange(m.RIGHT, buff=0.3)
        out_c.move_to([0, 0.75, 0]).align_to([1.15, 0, 0], m.LEFT)
        out_u.move_to([0, -0.45, 0]).align_to([1.15, 0, 0], m.LEFT)
        a_c = m.Arrow(box.get_right() + [0, 0.3, 0], out_c.get_left() + [-0.1, 0, 0], buff=0.08, color=CON, stroke_width=5,
                      max_tip_length_to_length_ratio=0.25)
        a_u = m.Arrow(box.get_right() + [0, -0.3, 0], out_u.get_left() + [-0.1, 0, 0], buff=0.08, color=UNC, stroke_width=5,
                      max_tip_length_to_length_ratio=0.25)
        assert out_c.get_right()[0] < 6.6 and out_u.get_right()[0] < 6.6, (out_c.get_right(), out_u.get_right())

        # the label table: 11 rows, the last means "no label"
        tab = cells([str(i) for i in range(10)] + ["∅"])
        tab[7][0].set_stroke(CON, 3).set_fill(PINK_FILL, 1)
        tab[7][1].set_color(INK)
        tab[10][0].set_stroke(MUTED, 3)
        tab[10][1].set_color(INK)
        tab_cap = cite(f"the label table (Episode 13): {N_ROWS} rows,\nthe last one means no label")
        table = m.VGroup(tab, tab_cap).arrange(m.DOWN, buff=0.16, aligned_edge=m.LEFT).move_to([0, -2.75, 0]).align_to([-6.3, 0, 0], m.LEFT)
        # ten training steps, one with the label hidden
        steps = cells(["3", "7", "0", "4", "1", "9", "∅", "2", "8", "5"], fill=CARD)
        for c_ in steps:
            c_[1].set_color(DATA)
        steps[6][0].set_stroke(CON, 3).set_fill(PINK_FILL, 1)
        steps[6][1].set_color(INK)
        st_cap = cite(f"ten training steps:\nthe label hidden in one (p = {P_UNCOND})")
        training = m.VGroup(steps, st_cap).arrange(m.DOWN, buff=0.16, aligned_edge=m.LEFT).move_to([0, -2.75, 0]).align_to([0.85, 0, 0], m.LEFT)
        assert table.get_right()[0] < training.get_left()[0] - 0.3 and training.get_right()[0] < 6.6, (table.get_right(), training.get_right())
        name = title_text("classifier-free guidance", 34, CON)
        name_c = cite("Ho & Salimans · workshop 2021, arXiv 2022")
        title = m.VGroup(name, name_c).arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT).move_to([0, 2.05, 0]).align_to([1.15, 0, 0], m.LEFT)
        assert title.get_right()[0] < 6.6 and title.get_bottom()[1] > out_c.get_top()[1] + 0.25 and title.get_top()[1] < 2.6, (title.get_right(), title.get_top())

        self.say(
            "[Episode thirteen gave our digit model a label to ask for], [and a spare row in its table that means no label].",
            m.AnimationGroup(self.new_header("01 · two questions", "One network, two questions"),
                             m.FadeIn(xt_pic), m.Create(xt_frame), m.FadeIn(xt_lab), m.FadeIn(t_chip), m.FadeIn(y7),
                             m.FadeIn(net), m.GrowArrow(ins[0]), m.GrowArrow(ins[1]), m.GrowArrow(ins[2])),
            m.AnimationGroup(m.FadeIn(table, shift=0.1 * m.UP), m.FadeIn(yn), m.GrowArrow(ins[3])),
        )
        self.say(
            "[In training, the label was hidden one step in ten], [so one network learned to answer with and without it].",
            m.FadeIn(training, shift=0.1 * m.UP),
            m.AnimationGroup(m.Circumscribe(y7, color=CON, buff=0.06), m.Circumscribe(yn, color=MUTED, buff=0.06)),
        )
        self.say(
            "[Given the label, it guesses the noise as if the picture were a seven].",
            m.AnimationGroup(m.GrowArrow(a_c), m.FadeIn(out_c, shift=0.1 * m.RIGHT), m.Indicate(y7[1], color=CON, scale_factor=1.25)),
        )
        self.say(
            "[Given no label, it guesses the noise for any digit at all].",
            m.AnimationGroup(m.GrowArrow(a_u), m.FadeIn(out_u, shift=0.1 * m.RIGHT), m.Indicate(yn[1], color=INK, scale_factor=1.25)),
        )
        self.say(
            "[That is all classifier-free guidance needs]: [one model, asked twice], [a trick from Ho and Salimans].",
            m.Write(name),
            m.AnimationGroup(m.Circumscribe(out_c, color=CON, buff=0.08), m.Circumscribe(out_u, color=UNC, buff=0.08)),
            m.FadeIn(name_c, shift=0.1 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 2 · extrapolate

    def extrapolate(self):
        self.next_section("2 · Push past the labelled guess", notes=(
            "Illustrative geometry (for example): two noise predictions drawn as arrows from the noisy point; the guided "
            "prediction ε̃ = ε_u + w·(ε_c − ε_u) moves along the line through their tips, w = 0 at ε_u, w = 1 at ε_c, "
            "past it for w > 1. The code card is diffusion_reference.cfg verbatim (its docstring states both "
            "conventions). Conventions: guidance scale w (GLIDE §2.3, CompVis --scale, diffusers, our digit run and the "
            "lighthouse run: mac_runs.md §4–5) vs Ho & Salimans' eq. 6, ε̃ = (1 + w′)·ε_c − w′·ε_u, so w′ = w − 1: "
            "Stable Diffusion's default 7.5 is their 6.5 (notes §14, Appendix C)."))
        P0 = np.array([-5.9, -2.3, 0.0])
        EU = np.array([0.9, 3.4, 0.0])
        D = np.array([0.65, 0.0, 0.0])
        U, C = P0 + EU, P0 + EU + D
        G = lambda w: U + w * D  # noqa: E731
        assert G(7.5)[0] < 0.0 and G(7.5)[1] > -2.4
        x_dot = m.Dot(P0, radius=0.09, color=NOISE).set_z_index(6)
        x_lab = tex("x_t", 32, NOISE).next_to(x_dot, m.LEFT, buff=0.14)
        a_u = vec(P0, U, UNC, 6).set_z_index(3)
        a_c = vec(P0, C, CON, 6).set_z_index(3)
        nrm = np.array([0.0, 1.0, 0.0])                                                 # perpendicular to the ruler, up
        l_u = tex(r"\varepsilon_u", 40, UNC).move_to(U + 0.66 * nrm + [-0.06, 0, 0])
        l_c = tex(r"\varepsilon_c", 40, CON).move_to(C + 0.66 * nrm + [0.06, 0, 0])
        diff = vec(U, C, CON, 4, 0.45).set_z_index(4)
        ruler = m.DashedLine(G(-0.7), G(8.3), dash_length=0.09).set_stroke(MUTED, 2.5).set_z_index(1)
        ticks = m.VGroup()
        tick_l = m.VGroup()
        for w in (0, 1, 3, 7.5):
            p = G(w)
            ticks.add(m.Line(p - 0.1 * nrm, p + 0.1 * nrm).set_stroke(MUTED, 2.5))
            tick_l.add(code(f"{w:g}", 20, MUTED).move_to(p + 0.27 * nrm))
        l_u.shift(0.12 * nrm)
        l_c.shift(0.12 * nrm)
        g_arrow = vec(P0, G(1), GUI, 7).set_z_index(5)
        g_dot = m.Dot(G(0), radius=0.08, color=GUI).set_z_index(7)
        g_lab = m.VGroup(tex(r"\tilde\varepsilon", 40, GUI), text("guided", 22, GUI, weight=m.BOLD)).arrange(m.RIGHT, buff=0.15)
        g_lab.move_to([-2.0, -0.9, 0])
        ex = cite("for example: two noise guesses, drawn as arrows").move_to([-3.4, -3.3, 0])
        assert g_lab.get_right()[0] < 0.6 and ex.get_left()[0] > -6.6, (g_lab.get_right(), ex.get_left())

        rx = 1.0
        f = formula(r"\tilde\varepsilon", "=", r"\varepsilon_u", "+", "w", r"\,(", r"\varepsilon_c", "-", r"\varepsilon_u", ")", size=48)
        f[0].set_color(GUI)
        f[2].set_color(UNC)
        f[4].set_color(CON)
        f[6].set_color(CON)
        f[8].set_color(UNC)
        f.move_to([0, 2.05, 0]).align_to([rx, 0, 0], m.LEFT)
        brace = m.Brace(f[5:], m.DOWN, buff=0.08).set_color(CON)
        brace_l = text("what the label adds", 20, CON).next_to(brace, m.DOWN, buff=0.08)
        src = m.VGroup(code("cfg(eps_uncond, eps_cond, w)", 22, INK, t2c={r"\bcfg\b": TEAL, r"\bw\b": CON}),
                       cite("the course's code, diffusion_reference.py")).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        src.move_to([0, 0.4, 0]).align_to([rx, 0, 0], m.LEFT)
        w_read = Readout(rx, -0.75, "w = {:.1f}", 0.0, 40, GUI)
        w_lab = text("the guidance scale", 22, CON).move_to([0, -0.75, 0]).align_to([rx + 2.75, 0, 0], m.LEFT)
        st0 = text("ignores the label", 24, UNC)
        st1 = text("the plain labelled answer", 24, CON)
        st2 = text("pushed past it", 24, GUI, weight=m.BOLD)
        for s in (st0, st1, st2):
            s.move_to([0, -1.45, 0]).align_to([rx, 0, 0], m.LEFT)
        status = st0.copy()
        none_pill = pill("1 = no guidance", GREEN, GREEN_FILL, 22).move_to([0, -2.25, 0]).align_to([rx, 0, 0], m.LEFT)
        assert src.get_right()[0] < 6.6 and f.get_right()[0] < 6.6 and src.get_top()[1] < brace_l.get_bottom()[1] - 0.1, (src.get_right(), f.get_right(), src.get_top(), brace_l.get_bottom())

        def slide(w0, w1, seconds=1.2, arrow=True):
            def step(_, a):
                w = w0 + (w1 - w0) * a
                g_dot.move_to(G(w))
                if arrow:
                    g_arrow.become(vec(P0, G(w), GUI, 7).set_z_index(5))
                w_read.show(round(w, 1))
            group = m.VGroup(g_dot, g_arrow) if arrow else m.VGroup(g_dot)
            return m.UpdateFromAlphaFunc(group, step, run_time=seconds, rate_func=m.smooth)

        # the two conventions
        xs = lambda v: 0.95 + 0.6 * v  # noqa: E731
        conv_rows = m.VGroup()
        for y, title, vals, shown, color in ((-0.15, "guidance scale: ours and Stable Diffusion's", (0, 1, 3, 7.5), (0, 1, 3, 7.5), CON),
                                             (-1.75, "w in the paper by Ho & Salimans", (0, 1, 3, 7.5), (-1, 0, 2, 6.5), MUTED)):
            line = m.Line([xs(-0.4), y, 0], [xs(8.1), y, 0]).set_stroke(color, 3)
            tks = m.VGroup(*[m.Line([xs(v), y - 0.1, 0], [xs(v), y + 0.1, 0]).set_stroke(color, 3) for v in vals])
            nums = m.VGroup(*[code(f"{s:g}".replace("-", "−"), 24, color, weight=m.BOLD).move_to([xs(v), y - 0.38, 0]) for v, s in zip(vals, shown)])
            cap_ = text(title, 20, color, weight=m.BOLD).move_to([0, y + 0.45 if y > -1 else y - 0.97, 0]).align_to([xs(-0.4), 0, 0], m.LEFT)
            conv_rows.add(m.VGroup(line, tks, nums, cap_))
        col1 = m.RoundedRectangle(corner_radius=0.12, width=0.7, height=2.5).set_stroke(GREEN, 3).set_fill(opacity=0)
        col1.move_to([xs(1), -1.16, 0])
        col1_l = text("no guidance", 22, GREEN, weight=m.BOLD).move_to([0, -1.1, 0]).align_to([xs(1) + 0.5, 0, 0], m.LEFT)
        hl_a = m.SurroundingRectangle(conv_rows[0][2][3], buff=0.08).set_stroke(CON, 2.5)
        hl_b = m.SurroundingRectangle(conv_rows[1][2][3], buff=0.08).set_stroke(MUTED, 2.5)
        ours = text("our digit and lighthouse runs: the same scale", 20, CON)
        ours.move_to([0, -3.3, 0]).align_to([rx, 0, 0], m.LEFT)
        assert ours.get_right()[0] < 6.6 and ours.get_top()[1] < conv_rows[1][3].get_bottom()[1] - 0.1, (ours.get_right(), ours.get_top(), conv_rows[1][3].get_bottom())
        assert conv_rows.get_right()[0] < 6.6 and conv_rows.get_bottom()[1] > -3.55, (conv_rows.get_right(), conv_rows.get_bottom())
        assert col1.get_top()[1] < conv_rows[0][3].get_bottom()[1] - 0.03 and col1.get_bottom()[1] > conv_rows[1][3].get_top()[1] + 0.03
        assert conv_rows[0][3].get_right()[0] < 6.6

        self.say(
            "[Draw the two guesses as arrows from the noisy point]: [one for any digit], [one for a seven].",
            m.AnimationGroup(self.new_header("02 · extrapolate", "Push past the labelled guess"), m.FadeIn(x_dot, scale=0.5),
                             m.FadeIn(x_lab), m.FadeIn(ex)),
            m.AnimationGroup(m.GrowArrow(a_u), m.FadeIn(l_u)),
            m.AnimationGroup(m.GrowArrow(a_c), m.FadeIn(l_c)),
        )
        self.say(
            "[Their difference is what the label adds].",
            m.AnimationGroup(m.GrowArrow(diff), m.Write(f), m.FadeIn(brace), m.FadeIn(brace_l)),
        )
        self.say(
            "[Guidance starts at the unlabelled guess], [and steps along that difference], [times one number: the guidance scale].",
            m.AnimationGroup(m.Create(ruler), m.FadeIn(ticks), m.FadeIn(tick_l), m.FadeIn(g_dot, scale=0.5), m.FadeIn(w_read),
                             m.FadeIn(status)),
            m.AnimationGroup(slide(0.0, 1.0, 1.4, arrow=False), m.Transform(status, st1)),
            m.AnimationGroup(m.Indicate(f[4], color=CON, scale_factor=1.4), m.FadeIn(w_lab)),
        )
        self.say(
            "[At scale one, it lands on the labelled guess]: [that is no guidance at all].",
            m.AnimationGroup(m.Circumscribe(m.VGroup(l_c, tick_l[1]), color=CON, buff=0.08), m.FadeIn(src, shift=0.1 * m.UP)),
            m.FadeIn(none_pill, shift=0.1 * m.UP),
        )
        self.say(
            "[Above one, it is pushed past]: [to three], [and on to seven point five].",
            m.AnimationGroup(m.GrowArrow(g_arrow), m.Transform(status, st2), m.FadeIn(g_lab)),
            slide(1.0, 3.0, 0.45),                                        # its bracket lasts ~0.5 s: done before the next
            slide(3.0, 7.5, 1.6),
        )
        self.say(
            "[Papers count this two ways]: [for Ho and Salimans, zero means no guidance], "
            "[so seven point five in Stable Diffusion is six point five in their paper].",
            m.AnimationGroup(m.FadeOut(src), m.FadeOut(w_read), m.FadeOut(status), m.FadeOut(none_pill), m.FadeOut(w_lab),
                             m.FadeIn(conv_rows[0], shift=0.1 * m.UP)),
            m.FadeIn(conv_rows[1], shift=0.1 * m.UP),
            m.AnimationGroup(m.Create(hl_a), m.Create(hl_b)),
        )
        self.say(
            "[Our runs use the scale too]: [here, one always means no guidance].",
            m.FadeIn(ours, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeOut(hl_a), m.FadeOut(hl_b), m.Create(col1), m.FadeIn(col1_l)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 3 · beyond the data

    def beyond_the_data(self):
        self.next_section("3 · Concentrate, then overshoot", notes=(
            "toy.md §5, recomputed here and asserted against toy.json: exact ε for both the unlabelled swirl and arm 0 "
            "(diffusion_reference.guided_eps), DDIM 100 steps, η = 0, the same 2,000 noise points (seed 5) for every "
            "scale; 450 drawn per panel (those whose paths stay inside the panels). Asked-for arm: arm 0 (PINK). On the "
            + "; ".join(f"w = {w}: {100 * SW_STATS[w]['on0']:.0f}% on arm 0, {100 * SW_STATS[w]['outer']:.0f}% of those on "
                        f"its outer half, spread {SW_STATS[w]['spread']:.4f}, coverage {SW_STATS[w]['cover']:.2f}, off "
                        f"{100 * SW_STATS[w]['off']:.1f}%" for w in SW_WS)
            + ". With exact scores, CFG equals Episode 13's classifier guidance with a perfect classifier (Bayes). "
            "Three blobs (blobs_mixture: std 0.15, centres at radius 0.6 at 90/210/330°, class 0 on top), after Ho & "
            "Salimans' Figure 2: asking for class 0, DDIM 100, same noise; 1,000 of 2,000 drawn. Discs: radius 2 std "
            "(in 2D, 1 − e⁻² ≈ 86% of a blob's data lies inside). Centre height = mean y of the samples nearest class 0; "
            "beyond = farther than 2 std from the class centre: "
            + "; ".join(f"{r[0]}: {r[1]}, {r[2]}" for r in BL_ROWS) + "."))
        size, gap = 2.75, 0.55
        xs = [(-1.5 + i) * (size + gap) for i in range(4)]
        pls = [plane((x, 0.55, 0), size, SW_LIM) for x in xs]
        frames = m.VGroup(*[m.Square(size).set_stroke(EDGE, 2).move_to(p.coords(0, 0)) for p in pls])
        bands = m.VGroup(*[m.VGroup(arm_band(p, 0, CON, 11, 0.5), arm_band(p, 1, DATA, 10, 0.14)) for p in pls])
        clouds = []
        for p, w in zip(pls, SW_WS):
            start = SW_RUN[w].xs[0][SW_SHOW]
            clouds.append(m.VGroup(*[m.Dot(at(p, q), radius=0.02, color=NOISE).set_opacity(0.85) for q in start]).set_z_index(3))
        tops = m.VGroup()
        for fr, lab, sub in zip(frames, ("w = 0", "w = 1", "w = 3", "w = 7"), ("no label", "no guidance", "", "")):
            words = code(lab, 24, CON, weight=m.BOLD)
            row = m.VGroup(words, cite(sub)).arrange(m.RIGHT, buff=0.2) if sub else m.VGroup(words)
            if sub:
                sit(row[1], baseline(row[0]))
            tops.add(row.next_to(fr, m.UP, buff=0.18))
        stat_rows = m.VGroup()
        for fr, w in zip(frames, SW_WS):
            s = SW_STATS[w]
            first = (f"{100 * s['on0']:.0f}% on the pink arm" if w <= 1 else f"{100 * s['outer']:.0f}% on its outer half")
            l1 = text(first, 20, INK if w <= 1 else CON)
            l2 = code(f"spread {s['spread']:.2f}", 20, MUTED if w < 7 else RED)
            stat_rows.add(m.VGroup(l1, l2).arrange(m.DOWN, buff=0.1).next_to(fr, m.DOWN, buff=0.18))
        cov = pill(f"arm coverage {SW_STATS[7]['cover']:.2f}", RED, RED_FILL, 20).next_to(stat_rows[3], m.DOWN, buff=0.16)
        off = text(f"off the swirl at every scale: {100 * SW_STATS[7]['off']:.0f}%", 22, GREEN, weight=m.BOLD)
        off_why = cite("the thin arm catches them again at low noise · as with Episode 13's classifier")
        why = m.VGroup(off, off_why).arrange(m.DOWN, buff=0.1).move_to([-1.45, -2.45, 0])
        assert why.get_right()[0] < cov.get_left()[0] - 0.2 or why.get_top()[1] < cov.get_bottom()[1] - 0.1, (why.get_right(), cov.get_left())
        sw_cap = cite("exact scores · DDIM, 100 steps · the same 2,000 starting points at every scale, 450 drawn").move_to([0, -3.4, 0])
        assert frames.get_left()[0] > -6.6 and frames.get_right()[0] < 6.6 and tops.get_top()[1] < 2.6
        assert why.get_left()[0] > -6.6 and why.get_bottom()[1] > sw_cap.get_top()[1] + 0.05, (why.get_left(), why.get_bottom())

        # the blobs
        bp = Plot(BL_X, BL_Y, width=5.0, height=5.0, center=(-3.55, -0.45, 0), axes=False)
        b_frame = m.Rectangle(width=5.0, height=5.0).set_stroke(EDGE, 2).move_to(bp.coords(0, 0.35))
        unit = bp.w / (bp.x1 - bp.x0)
        discs = m.VGroup()
        for k, mu in enumerate(BM.means):
            col = CON if k == 0 else DATA
            disc = m.Circle(radius=2 * S0 * unit).set_stroke(width=0).set_fill(col if k == 0 else DATA, 0.13).move_to(at(bp, mu))
            ring = m.DashedVMobject(m.Circle(radius=2 * S0 * unit), num_dashes=30).set_stroke(col, 2.5).move_to(at(bp, mu))
            discs.add(m.VGroup(disc, ring))
        disc_l = code("class 0", 20, CON, weight=m.BOLD).next_to(discs[0], m.LEFT, buff=0.6)
        b_dots = m.VGroup(*[m.Dot(at(bp, q), radius=0.022, color=DATA).set_opacity(0.9) for q in BL[0][BL_SHOW]]).set_z_index(3)
        own = cross(at(bp, M0), DATA, 0.14, 3).set_z_index(5)
        cen = {w: at(bp, BL_STATS[w]["centre"]) for w in (1, 3, 7)}
        mark = cross(cen[1], CON, 0.2, 5).set_z_index(6)
        mark_l = text("centre", 20, CON, weight=m.BOLD).set_z_index(8)
        mark_l.next_to(mark, m.RIGHT, buff=0.7)
        marker = m.VGroup(mark, mark_l)
        mk_at = lambda w: marker.get_center() + (cen[w] - cen[1])  # noqa: E731
        b_cap = cite("exact scores · DDIM, 100 steps · 1,000 of 2,000 samples · after Ho & Salimans, Fig. 2").move_to([0, -3.4, 0])
        assert b_frame.get_left()[0] > -6.6 and b_frame.get_top()[1] < 2.6 and mark_l.get_right()[0] + 0.0 < b_frame.get_right()[0] - 0.1, (b_frame.get_top(), mark_l.get_right())

        tx = 0.85
        t_title = text("asking for class 0", 24, CON, weight=m.BOLD).move_to([0, 2.0, 0]).align_to([tx, 0, 0], m.LEFT)
        colx = (tx, tx + 1.65, tx + 3.4)
        heads = m.VGroup(*[code(s, 22, MUTED) for s in ("scale", "centre", "beyond 2 std")])
        rows = [heads]
        for r in BL_ROWS:
            rows.append(m.VGroup(*[code(s, 26, DATA if r[0] == "data" else INK, weight=m.BOLD) for s in r]))
        for i, row in enumerate(rows):
            y = 1.25 - 0.62 * i - (0.12 if i else 0)
            for cell, x in zip(row, colx):
                cell.move_to([0, y, 0]).align_to([x, 0, 0], m.LEFT)
        rows[4][1].set_color(RED)
        rows[4][2].set_color(RED)
        tbl_rule = m.Line([tx, 0.95, 0], [colx[2] + heads[2].width, 0.95, 0]).set_stroke(EDGE, 2)
        past = pill("pushed past the data", RED, RED_FILL, 22).move_to([0, -2.1, 0]).align_to([tx, 0, 0], m.LEFT)
        far = text("away from the other classes", 22, INK).next_to(past, m.DOWN, buff=0.15).align_to(past, m.LEFT)
        assert heads[2].get_right()[0] < 6.6 and far.get_bottom()[1] > b_cap.get_top()[1] + 0.05, (heads[2].get_right(), far.get_bottom())

        self.say(
            "[What does the push do to samples]? [On the swirl, with exact scores], [ask for one arm at four scales].",
            m.AnimationGroup(self.new_header("03 · beyond the data", "Concentrate, then overshoot"), m.FadeIn(frames), m.FadeIn(bands)),
            m.AnimationGroup(m.FadeIn(tops, shift=0.1 * m.DOWN), m.FadeIn(sw_cap), *[m.FadeIn(c) for c in clouds]),
            m.AnimationGroup(*[fly(c, p, SW_RUN[w].xs[:, SW_SHOW], 3.0) for c, p, w in zip(clouds, pls, SW_WS)]),
        )
        self.say(
            "[At zero, both arms]. [At one, the asked-for arm, all of it]. [At three and seven, the samples crowd onto its outer half].",
            m.FadeIn(stat_rows[0], shift=0.1 * m.UP),
            m.FadeIn(stat_rows[1], shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(stat_rows[2], shift=0.1 * m.UP), m.FadeIn(stat_rows[3], shift=0.1 * m.UP)),
        )
        self.say(
            "[Coverage of the arm falls to zero point four five], [yet nothing leaves it]: [the thin arm catches the samples again].",
            m.FadeIn(cov, shift=0.1 * m.UP),
            m.FadeIn(off, shift=0.1 * m.UP),
            m.FadeIn(off_why),
        )
        self.wait(0.3)
        swirl_part = m.VGroup(frames, bands, *clouds, tops, stat_rows, cov, why, sw_cap)
        self.say(
            "[So take a toy with nothing to catch them]: [three round blobs, after a figure by Ho and Salimans].",
            m.FadeOut(swirl_part),
            m.AnimationGroup(m.FadeIn(b_frame), m.FadeIn(discs), m.FadeIn(disc_l), m.FadeIn(own), m.FadeIn(b_cap),
                             m.LaggedStart(*[m.FadeIn(d, scale=0.5) for d in b_dots], lag_ratio=0.002, run_time=1.4)),
        )
        self.say(
            "[Ask for the top blob at scale one], [and the samples gather in it], [centred at height zero point six].",
            m.AnimationGroup(glide(b_dots, bp, BL[0][BL_SHOW], BL[1][BL_SHOW], 1.6), m.FadeIn(t_title), m.FadeIn(heads),
                             m.Create(tbl_rule)),
            m.AnimationGroup(m.FadeIn(rows[1], shift=0.1 * m.UP), m.FadeIn(rows[2], shift=0.1 * m.UP)),
            m.FadeIn(marker, scale=0.8),
        )
        self.say(
            "[At three, they tighten and rise]. [At seven, their centre climbs to zero point nine six], [above the blob itself].",
            m.AnimationGroup(glide(b_dots, bp, BL[1][BL_SHOW], BL[3][BL_SHOW], 1.4), marker.animate(run_time=1.4).move_to(mk_at(3)),
                             m.FadeIn(rows[3], shift=0.1 * m.UP)),
            m.AnimationGroup(glide(b_dots, bp, BL[3][BL_SHOW], BL[7][BL_SHOW], 1.6), marker.animate(run_time=1.6).move_to(mk_at(7)),
                             m.FadeIn(rows[4], shift=0.1 * m.UP)),
            m.Circumscribe(discs[0], color=CON, buff=0.06),
        )
        self.say(
            "[Eighty-six percent now lie beyond two standard deviations of the class]: [for real data, fourteen]. "
            "[Strong guidance pushes samples past the data].",
            m.Circumscribe(rows[4][2], color=RED, buff=0.08),
            m.Circumscribe(m.VGroup(rows[1][0], rows[1][2]), color=DATA, buff=0.08),
            m.AnimationGroup(m.FadeIn(past, shift=0.1 * m.UP), m.FadeIn(far, shift=0.1 * m.UP)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 4 · the digits

    def digits(self):
        self.next_section("4 · The digits: the honest trade", notes=(
            "Generated on this Mac (mac_runs.md §4): the digit model, DDIM 50 steps, 10,000 samples per scale, sample i asks "
            "for label i mod 10, the same starting noise for every scale (numpy seed 1234); rows show samples 10–19. "
            "w = 0 is the unconditional model (the 'no label' row, trained 10% of the time). Ink = share of pixels above "
            "127 over all 10,000: " + ", ".join(f"w = {w}: {INK_S[w]}" for w in DW) + ". Twelve 7s: samples 7, 17, …, 117 "
            "at w = 1 and 8 (the same noises). Metrics (mac_runs.json digit_sampling.configs): label accuracy by our own "
            "classifier (99.45% on MNIST test); precision/recall: k-NN estimate, k = 3 (Kynkäänniemi et al. 2019); "
            "diversity: per class, mean pairwise feature distance ÷ real test digits' (n/a at w = 0). "
            + "; ".join(f"w = {w}: {ACC[w]}, {PREC[w]}, {REC[w]}, {DIV[w]}, cFID {CF[w]}" for w in DW)
            + f". cFID = FID's formula on our classifier's 128-d features, 10,000 generated vs 10,000 test digits — NOT "
            f"Inception FID; floor {FLOOR:.2f} (real vs real); bootstrap std ≈ 0.3. ImageNet 128 × 128: Ho & Salimans, "
            "Table 2, 256 sampling steps, Inception FID, read from arXiv:2207.12598 (ar5iv, October 2026); their w′ = "
            "scale − 1. The flow model shows the same trend (mac_runs §4). Guided sampling of 1,000 digits took "
            f"{SEC_1000[3]:.1f} s vs {SEC_1000[1]:.1f} s (batch 1,000, the GPU already full)."))
        hgt, pitch = 0.6, 0.74
        rows_y = [1.9 - i * pitch for i in range(6)]
        gx = 0.1
        grid = m.Group()
        labs = m.VGroup()
        inks = m.VGroup()
        for w, y in zip(DW, rows_y):
            grid.add(digit_row(GRID[w], DATA, hgt, gap_px=6).move_to([gx, y, 0]))
            lab = code(f"w = {w}", 22, CON, weight=m.BOLD)
            sub = cite("no label" if w == 0 else "no guidance" if w == 1 else "")
            row = m.VGroup(lab, sub).arrange(m.RIGHT, buff=0.18) if w <= 1 else m.VGroup(lab)
            if w <= 1:
                sit(sub, baseline(lab))
            labs.add(row.move_to([0, y, 0]).align_to([gx - row_width(10, hgt, 6) / 2 - 0.3, 0, 0], m.RIGHT))
            inks.add(code(INK_S[w], 22, RED if w == 8 else INK).move_to([gx + row_width(10, hgt, 6) / 2 + 1.1, y, 0]))
        ink_h = text("inked pixels", 20, MUTED).next_to(inks, m.UP, buff=0.18)
        grid_frame = m.SurroundingRectangle(grid, buff=0.1).set_stroke(EDGE, 2)
        g_cap = cite("generated on this Mac · the digit model · DDIM, 50 steps · same starting noise per column (seed 1234)").move_to([0, -3.35, 0])
        hl0 = m.SurroundingRectangle(m.Group(grid[0], labs[0]), buff=0.06).set_stroke(UNC, 2.5)
        hl_bold = m.SurroundingRectangle(inks[2:], buff=0.1).set_stroke(RED, 2.5)
        bold_l = text("bolder\nstrokes", 20, RED, weight=m.BOLD).next_to(hl_bold, m.DOWN, buff=0.12)
        assert bold_l.get_left()[0] > grid_frame.get_right()[0] + 0.1, (bold_l.get_left(), grid_frame.get_right())
        assert labs.get_left()[0] > -6.6 and inks.get_right()[0] < 6.6 and ink_h.get_top()[1] < 2.6, (labs.get_left(), ink_h.get_top())
        assert bold_l.get_bottom()[1] > g_cap.get_top()[1] + 0.05 and grid_frame.get_bottom()[1] > g_cap.get_top()[1] + 0.1

        # twelve sevens
        sh = 0.86
        sev = {}
        for w, y in ((1, 0.95), (8, -1.35)):
            pic = digit_row(SAMPLES[w][SEVENS], DATA, sh, gap_px=5).move_to([0, y, 0])
            fr = m.SurroundingRectangle(pic, buff=0.08).set_stroke(EDGE, 2)
            lab = code(f"w = {w}", 24, CON, weight=m.BOLD)
            sub = text("each in its own hand" if w == 1 else "nearly copies", 22, GREEN if w == 1 else RED, weight=m.BOLD)
            head = m.VGroup(lab, sub).arrange(m.RIGHT, buff=0.35)
            sit(sub, baseline(lab))
            head.next_to(fr, m.UP, buff=0.18).align_to(fr, m.LEFT)
            sev[w] = m.Group(pic, fr, head)
        div_l = m.VGroup(text("diversity, all ten digits:", 22, INK), code(f"{DIV[1]} → {DIV[8]}", 24, RED, weight=m.BOLD),
                         cite("(1.0 = as varied as real digits)")).arrange(m.RIGHT, buff=0.25)
        div_l.move_to([0, -2.55, 0])
        s_cap = cite("generated on this Mac · twelve 7s · the same twelve starting noises in both rows").move_to([0, -3.35, 0])
        assert sev[1][1].get_left()[0] > -6.6 and sev[1][1].get_right()[0] < 6.6 and div_l.get_right()[0] < 6.6

        # the metrics table
        cols = ("scale w", "label right", "precision", "recall", "diversity")
        colx = [-5.6, -3.55, -0.85, 1.55, 3.6]
        trow = lambda vals, y, colors, size=24: m.VGroup(*[code(v, size, c, weight=m.BOLD if size > 22 else m.NORMAL)  # noqa: E731
                                                           .move_to([0, y, 0]).align_to([x, 0, 0], m.LEFT)
                                                           for v, x, c in zip(vals, colx, colors)])
        head = trow(cols, 1.85, [MUTED] * 5, 22)
        rule = m.Line([-5.7, 1.5, 0], [5.5, 1.5, 0]).set_stroke(EDGE, 2)
        trs = {}
        for i, w in enumerate(DW):
            y = 1.05 - 0.6 * i
            better = lambda a, b: GREEN if a > b else RED if a < b else INK  # noqa: E731
            if w == 0:
                colors = [CON, MUTED, MUTED, MUTED, MUTED]
            elif w == 1:
                colors = [CON, INK, INK, INK, INK]
            else:
                colors = [CON, GREEN, better(float(PREC[w]), float(PREC[1])), RED, RED]
            trs[w] = trow((f"{w}", ACC[w], PREC[w], REC[w], DIV[w]), y, colors)
        real = trow(("real", f"{100 * measured('mac_runs', 'digit_sampling', 'real_reference', 'cond_acc'):.2f}%",
                     f"{measured('mac_runs', 'digit_sampling', 'real_reference', 'precision'):.2f}",
                     f"{measured('mac_runs', 'digit_sampling', 'real_reference', 'recall'):.2f}",
                     f"{measured('mac_runs', 'digit_sampling', 'real_reference', 'diversity'):.2f}"), 1.05 - 0.6 * 6 - 0.1, [DATA] * 5, 22)
        assert real[1].text == "99.55%" and real[3].text == "0.89"
        t_cap = cite("10,000 digits per scale · label right: our classifier reads the asked-for digit · real: 10,000 training digits").move_to([0, -3.35, 0])
        assert head.get_right()[0] < 6.6 and t_cap.width < 13.0, t_cap.width
        assert real.get_bottom()[1] > t_cap.get_top()[1] + 0.05, (real.get_bottom(), t_cap.get_top())
        hl_acc = m.SurroundingRectangle(m.VGroup(*[trs[w][1] for w in DW[1:]]), buff=0.12).set_stroke(GREEN, 2.5)
        hl_prec = m.SurroundingRectangle(m.VGroup(*[trs[w][2] for w in DW[1:]]), buff=0.12).set_stroke(GREEN, 2.5)
        hl_rec = m.SurroundingRectangle(m.VGroup(*[trs[w][3] for w in DW[1:]]), buff=0.12).set_stroke(RED, 2.5)
        hl_div = m.SurroundingRectangle(m.VGroup(*[trs[w][4] for w in DW[1:]]), buff=0.12).set_stroke(RED, 2.5)

        # two FID plots
        ours = Plot((1, 8.4), (0, 90), width=4.6, height=2.4, center=(3.35, -0.55, 0), x_ticks=[1, 2, 3, 5, 8],
                    y_ticks=[0, 30, 60, 90], x_label="guidance scale")
        hs = Plot((1, 8.4), (0, 25), width=4.6, height=2.4, center=(-3.35, -0.55, 0), x_ticks=[1, 2, 3, 5, 8],
                  y_ticks=[0, 10, 20], x_label="guidance scale")
        o_line = ours.line([1, 2, 3, 5, 8], [CFID[w] for w in (1, 2, 3, 5, 8)], DATA, 4)
        o_dots = m.VGroup(*[ours.dot(w, CFID[w], DATA, 0.07) for w in (1, 2, 3, 5, 8)])
        o_best = ours.dot(1, CFID[1], GREEN, 0.11)
        o_floor = ours.hline(FLOOR, MUTED)
        o_vals = m.VGroup(*[code(CF[w], 20, RED).next_to(ours.coords(w, CFID[w]), m.DR if w == 2 else m.UL, buff=0.08) for w in (2, 3, 5, 8)])
        h_line = hs.line(HS_SCALE, HS_FID, INK, 4)
        h_best = hs.dot(HS_BEST, min(HS_FID), GREEN, 0.11)
        h_val = code(f"{min(HS_FID)} at scale {HS_BEST}", 20, GREEN).next_to(hs.coords(HS_BEST, min(HS_FID)), m.RIGHT, buff=0.45)
        h_end = code(f"{HS_FID[-1]}", 20, INK).next_to(hs.coords(HS_SCALE[-1], HS_FID[-1]), m.RIGHT, buff=0.12)
        o_t = m.VGroup(text("our digits · FID-like", 24, DATA, weight=m.BOLD), cite("our classifier's features, not Inception's"),
                       cite(f"on this Mac · lower is better · real: {FLOOR:.2f}")).arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT)
        h_t = m.VGroup(text("ImageNet, 128 × 128", 24, INK, weight=m.BOLD), cite("FID (Inception) · lower is better"),
                       cite("Ho & Salimans, 2022, Table 2 · 256 steps")).arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT)
        o_t.next_to(ours, m.UP, buff=0.3).align_to(ours, m.LEFT)
        h_t.next_to(hs, m.UP, buff=0.3).align_to(hs, m.LEFT)
        o_none = pill(f"best: {CF[1]}, with no guidance", GREEN, GREEN_FILL, 20).move_to([3.35, -2.95, 0])
        h_some = pill("best: a little guidance", GREEN, GREEN_FILL, 20).move_to([-3.35, -2.95, 0])
        assert o_t.get_top()[1] < 2.6 and h_t.get_top()[1] < 2.6 and ours.get_right()[0] < 6.6 and hs.get_left()[0] > -6.6, (o_t.get_top(), ours.get_right(), hs.get_left())
        assert o_none.get_top()[1] < ours.get_bottom()[1] - 0.05 and o_none.get_bottom()[1] > -3.6, (o_none.get_top(), ours.get_bottom())
        ours_all = m.VGroup(ours, o_t, o_floor, o_line, o_dots, o_best, o_vals)
        ours_home = ours_all.get_center()
        ours_all.shift(3.35 * m.LEFT)                                                 # centred until ImageNet joins

        self.say(
            "[Now the digit model, trained on this Mac with that one-in-ten dropout]: [the same noise at six guidance scales].",
            m.AnimationGroup(self.new_header("04 · the digits", "The honest trade"), m.FadeIn(g_cap), m.Create(grid_frame)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(m.Group(g, l_), shift=0.08 * m.RIGHT) for g, l_ in zip(grid, labs)],
                                           lag_ratio=0.25, run_time=2.0), m.FadeIn(ink_h), m.FadeIn(inks)),
        )
        self.say(
            "[At zero, the label is ignored]: [any digit]. [From two up, the strokes grow bolder].",
            m.Create(hl0),
            m.Indicate(labs[0], color=UNC, scale_factor=1.15),
            m.AnimationGroup(m.Create(hl_bold), m.FadeIn(bold_l)),
        )
        self.wait(0.3)
        screen_a = m.Group(grid, labs, inks, ink_h, grid_frame, g_cap, hl0, hl_bold, bold_l)
        self.say(
            "[Look at twelve sevens]. [At scale one, each has its own hand]; [at eight, they could be copies].",
            m.AnimationGroup(m.FadeOut(screen_a), m.FadeIn(s_cap)),
            m.FadeIn(sev[1], shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(sev[8], shift=0.1 * m.UP), m.FadeIn(div_l, shift=0.1 * m.UP)),
        )
        self.wait(0.3)
        self.say(
            "[Over ten thousand digits], [label accuracy rises from ninety-eight point one to ninety-nine point nine nine percent].",
            m.AnimationGroup(m.FadeOut(m.Group(sev[1], sev[8], div_l, s_cap)), m.FadeIn(head), m.Create(rule),
                             m.LaggedStart(*[m.FadeIn(trs[w], shift=0.05 * m.UP) for w in DW], lag_ratio=0.15, run_time=1.4),
                             m.FadeIn(real), m.FadeIn(t_cap)),
            m.Create(hl_acc),
        )
        self.say(
            "[Precision rises too], [from zero point eight three to as much as zero point eight nine].",
            m.Create(hl_prec),
            m.Indicate(trs[3][2], color=GREEN, scale_factor=1.15),
        )
        self.say(
            "[But recall falls from zero point eight nine to zero point three eight], [and diversity from one point zero one to zero point seven four].",
            m.Create(hl_rec),
            m.Create(hl_div),
        )
        self.wait(0.3)
        table = m.VGroup(head, rule, *trs.values(), real, t_cap, hl_acc, hl_prec, hl_rec, hl_div)
        self.say(
            "[And the FID-like score gets worse at every step up]: [three point three five with no guidance], [about eighty-five at eight].",
            m.AnimationGroup(m.FadeOut(table), m.FadeIn(ours), m.FadeIn(o_t), m.FadeIn(o_floor)),
            m.AnimationGroup(m.FadeIn(o_dots[0]), m.FadeIn(o_best, scale=0.5)),
            m.AnimationGroup(m.Create(o_line, run_time=1.4), m.FadeIn(o_dots[1:]), m.FadeIn(o_vals)),
        )
        self.say(
            "[It uses our own digit classifier's features, not Inception's], [so compare it only with itself].",
            m.Circumscribe(o_t[1], color=DATA, buff=0.06),
            m.Indicate(o_floor, color=INK, scale_factor=1.0),
        )
        self.say(
            "[On ImageNet, Ho and Salimans found the best score with a little guidance], [at scale one point three]. "
            "[On our digits, the best is none].",
            m.AnimationGroup(ours_all.animate(run_time=1.0).move_to(ours_home),
                             m.Succession(m.Wait(0.8), m.AnimationGroup(m.FadeIn(hs), m.FadeIn(h_t), m.Create(h_line, run_time=1.4),
                                                                        m.FadeIn(h_end)))),
            m.AnimationGroup(m.FadeIn(h_best, scale=0.5), m.FadeIn(h_val), m.FadeIn(h_some, shift=0.1 * m.UP)),
            m.FadeIn(o_none, shift=0.1 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 5 · the lighthouse

    def lighthouse(self):
        self.next_section("5 · The lighthouse: why everyone guides", notes=(
            "Generated on this Mac (mac_runs.md §5): SD 2.1 base, 50 steps, seed 0, guidance 1 / 3 / 5 / 7.5 / 10 / 15 "
            "(images/lighthouse/lighthouse_sd21_cfgN_steps50_seed0.png). Mean HSV saturation (max − min)/max over all "
            "pixels, averaged over seeds 0–3, recomputed here from the PNGs and asserted against mac_runs.json: "
            + ", ".join(f"{s}: {SAT[s]:.3f}" for s in LH_SCALES) + ". Clipped = a channel at 0 or 255, seed 0: "
            + ", ".join(f"{s}: {CLIP_S[s]}" for s in LH_SCALES) + " (seeds 1–3 clip less: 0 / 1.5 / 0.5% at 15). "
            f"Digit model: label right {ACC[1]} at w = 1 (no guidance). Stable Diffusion's default 7.5: CompVis SD v1 README "
            "(notes §14)."))
        c1 = m.VGroup(text("the digit model", 26, DATA, weight=m.BOLD), text("a label: one of ten", 24, INK),
                      text(f"right {ACC[1]} of the time,", 22, MUTED), text("with no guidance at all", 22, MUTED))
        c1.arrange(m.DOWN, buff=0.14, aligned_edge=m.LEFT)
        c2 = m.VGroup(text("the lighthouse", 26, CON, weight=m.BOLD), text("a prompt: open-ended", 24, INK),
                      code(PROMPT_TEXT[:29], 20, CON), code(PROMPT_TEXT[29:].strip(), 20, CON))
        c2.arrange(m.DOWN, buff=0.14, aligned_edge=m.LEFT)
        k1 = boxed(c1, DATA, 0.35).move_to([-3.2, 0.2, 0])
        k2 = boxed(c2, CON, 0.35).move_to([3.2, 0.2, 0])
        assert k1.get_left()[0] > -6.6 and k2.get_right()[0] < 6.6

        ph, pgap = 1.9, 0.2
        pxs = [(-2.5 + i) * (ph + pgap) for i in range(6)]
        pics = m.Group()
        frames = m.VGroup()
        labs = m.VGroup()
        for s, x in zip(LH_SCALES, pxs):
            pic = picture(LH_FILE[s], ph).move_to([x, 1.2, 0])
            pics.add(pic)
            frames.add(m.Rectangle(width=ph + 0.08, height=ph + 0.08).set_stroke(EDGE, 2).move_to(pic))
            labs.add(code(f"scale {s}", 22, CON, weight=m.BOLD).next_to(pic, m.UP, buff=0.16))
        verdict = {0: RED, 1: GREEN, 2: GREEN, 3: GREEN, 4: RED, 5: RED}
        v_frames = m.VGroup(*[m.Rectangle(width=ph + 0.08, height=ph + 0.08).set_stroke(verdict[i], 4).move_to(pics[i]) for i in range(6)])
        base_y, unit = -2.0, 3.3
        bars = m.VGroup()
        vals = m.VGroup()
        for s, x in zip(LH_SCALES, pxs):
            hh = SAT[s] * unit
            col = MUTED if s == "1" else GREEN if float(s) <= 7.5 else RED
            bars.add(m.Rectangle(width=0.8, height=hh).set_stroke(col, 2).set_fill(FILLS.get(col, CARD), 1).move_to([x, base_y + hh / 2, 0]))
            vals.add(code(f"{SAT[s]:.2f}", 20, col).move_to([x, base_y + hh + 0.2, 0]))
        axis = m.Line([pxs[0] - 0.95, base_y, 0], [pxs[-1] + 0.95, base_y, 0]).set_stroke(FAINT, 2)
        bar_t = cite("mean colour saturation, seeds 0–3").move_to([0, -0.45, 0]).align_to([pxs[0] - ph / 2, 0, 0], m.LEFT)
        clips = m.VGroup(*[code(CLIP_S[s], 20, RED if CLIP0[s] > 0.01 else MUTED).move_to([x, base_y - 0.32, 0]) for s, x in zip(LH_SCALES, pxs)])
        clip_t = cite("pixels clipped to pure black or white, seed 0 (the pictures shown)").move_to([0, base_y - 0.72, 0])
        cap = cite("generated on this Mac · Stable Diffusion 2.1 base · 50 steps · seed 0").move_to([0, -3.4, 0])
        assert labs.get_top()[1] < 2.6 and pics.get_left()[0] > -6.6 and pics.get_right()[0] < 6.6, (labs.get_top(), pics.get_left())
        assert all(v.get_top()[1] < bar_t.get_bottom()[1] - 0.05 for v, x in zip(vals, pxs) if x - 0.5 < bar_t.get_right()[0])
        assert bar_t.get_top()[1] < pics.get_bottom()[1] - 0.2
        assert clip_t.get_bottom()[1] > cap.get_top()[1] + 0.03, (clip_t.get_bottom(), cap.get_top())
        # the clipped pixels of the scale-15 picture, painted RED on a dimmed copy
        a15 = np.asarray(Image.open(LH_FILE["15"]).convert("RGB"))
        clipped = ((a15 == 0) | (a15 == 255)).any(-1)
        assert abs(clipped.mean() - CLIP0["15"]) < 1e-9
        dim = (a15.astype(float) * 0.35).round().astype(np.uint8)
        dim[clipped] = [int(RED[i:i + 2], 16) for i in (1, 3, 5)]
        mask = picture(dim, ph).move_to(pics[5]).set_z_index(2)

        self.say(
            "[So why does everyone turn guidance up]? [A ten-label digit model is already right ninety-eight percent of the time].",
            m.AnimationGroup(self.new_header("05 · the lighthouse", "Why everyone guides"), m.FadeIn(k2, shift=0.1 * m.UP)),
            m.FadeIn(k1, shift=0.1 * m.UP),
        )
        self.say(
            "[A text prompt is open-ended]. [At scale one], [the lighthouse comes out washed out and muddled].",
            m.AnimationGroup(m.FadeOut(k1), m.FadeOut(k2), m.FadeIn(cap),
                             m.LaggedStart(*[m.FadeIn(m.Group(p, f_, l_), shift=0.08 * m.UP) for p, f_, l_ in zip(pics, frames, labs)],
                                           lag_ratio=0.12, run_time=1.6)),
            m.Wait(0.1),
            m.Create(v_frames[0]),
        )
        self.say(
            "[From three to seven point five, a clear painting]: [Stable Diffusion's default is seven point five].",
            m.LaggedStart(*[m.Create(v_frames[i]) for i in (1, 2, 3)], lag_ratio=0.25),
            m.Indicate(labs[3], color=CON, scale_factor=1.15),
        )
        self.say(
            "[Past that, colours burn]. [Mean saturation climbs from zero point one nine to zero point three six].",
            m.LaggedStart(*[m.Create(v_frames[i]) for i in (4, 5)], lag_ratio=0.3),
            m.AnimationGroup(m.FadeIn(axis), m.FadeIn(bar_t),
                             m.LaggedStart(*[m.GrowFromEdge(b, m.DOWN) for b in bars], lag_ratio=0.12, run_time=1.4),
                             m.FadeIn(vals)),
        )
        self.say(
            "[And in this picture at fifteen, nine point three percent of the pixels clip]: [the blobs' overshoot, in colour].",
            m.AnimationGroup(m.FadeIn(mask), m.FadeIn(clips), m.FadeIn(clip_t)),
            m.Circumscribe(clips[5], color=RED, buff=0.08),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 6 · the cost

    def the_cost(self):
        self.next_section("6 · Two passes per step", notes=(
            "Measured on this Mac (mac_runs.md §5, timing breakdown, SD 2.1 base, 512 × 512, 50 steps, median of 3 runs): "
            f"guided step (two passes batched) {STEP_CFG} s, unguided step (guidance 1, one pass) {STEP_ONE} s: "
            f"{STEP_CFG / STEP_ONE:.2f}×, not 2×, because the two passes share one batch (Apple's own benchmark: "
            "\"classifier-free guidance (batch size of 2 for unet)\", notes §14). The dashed bar is 2 × 0.083 s, for "
            "comparison (not measured). The noisy picture is the lighthouse run's decoded x_t at step 8 of 30 "
            "(images/lighthouse_frames30/xt_08.png). Defaults [DATED, read October 2026]: CompVis SD v1 README "
            "(--scale 7.5); diffusers v0.42: SD 7.5, SDXL 5.0, SD3 7.0, FLUX.1 [dev] distilled guidance 3.5 (notes §14). "
            "FLUX.1 [dev] is guidance-distilled: the scale is an input to one pass (Meng et al. 2022's guidance "
            "distillation, Episode 12)."))
        lat = lambda: picture(LH_NOISY, 1.05)  # noqa: E731
        b1, b2 = lat(), lat()
        b1.move_to([-5.3, 1.85, 0])
        b2.move_to([-5.3, 0.55, 0])
        f1 = m.Rectangle(width=1.13, height=1.13).set_stroke(CON, 3).move_to(b1)
        f2 = m.Rectangle(width=1.13, height=1.13).set_stroke(MUTED, 3).move_to(b2)
        l1 = text("+ the prompt", 20, CON, weight=m.BOLD).next_to(f1, m.RIGHT, buff=0.15)
        l2 = text("+ an empty prompt", 20, MUTED, weight=m.BOLD).next_to(f2, m.RIGHT, buff=0.15)
        brace = m.Brace(m.VGroup(f1, f2), m.LEFT, buff=0.1).set_color(INK)
        br_l = text("one batch of two", 20, INK).rotate(PI_ / 2).next_to(brace, m.LEFT, buff=0.08)
        box = card(2.9, 1.3, MODEL, AMBER_FILL)
        box_t = m.VGroup(text("denoiser", 26, INK, weight=m.BOLD), code("866 M parameters", 20, MUTED)).arrange(m.DOWN, buff=0.1)
        net = m.VGroup(box, box_t.move_to(box)).move_to([-0.25, 1.2, 0])
        go = m.Arrow([f1.get_right()[0] + 0.05, 1.2, 0], box.get_left(), buff=0.12, color=INK, stroke_width=5, max_tip_length_to_length_ratio=0.12)
        o_c = tex(r"\varepsilon_c", 44, CON).move_to([1.95, 1.75, 0])
        o_u = tex(r"\varepsilon_u", 44, UNC).move_to([1.95, 0.65, 0])
        a_c = m.Arrow(box.get_right() + [0, 0.25, 0], o_c.get_left(), buff=0.12, color=CON, stroke_width=4, max_tip_length_to_length_ratio=0.25)
        a_u = m.Arrow(box.get_right() + [0, -0.25, 0], o_u.get_left(), buff=0.12, color=UNC, stroke_width=4, max_tip_length_to_length_ratio=0.25)
        mixf = formula(r"\varepsilon_u", "+", "w", r"\,(", r"\varepsilon_c", "-", r"\varepsilon_u", ")", size=36)
        mixf[0].set_color(UNC)
        mixf[2].set_color(CON)
        mixf[4].set_color(CON)
        mixf[6].set_color(UNC)
        mixf.move_to([4.85, 1.2, 0])
        mix_box = m.SurroundingRectangle(mixf, buff=0.18).set_stroke(GUI, 2.5)
        j_c = m.Arrow(o_c.get_right(), mix_box.get_left() + [0, 0.2, 0], buff=0.12, color=CON, stroke_width=4, max_tip_length_to_length_ratio=0.25)
        j_u = m.Arrow(o_u.get_right(), mix_box.get_left() + [0, -0.2, 0], buff=0.12, color=UNC, stroke_width=4, max_tip_length_to_length_ratio=0.25)
        assert mix_box.get_right()[0] < 6.6 and br_l.get_left()[0] > -6.6, (mix_box.get_right(), br_l.get_left())
        assert l2.get_right()[0] < box.get_left()[0] - 0.4, (l2.get_right(), box.get_left())

        # timing bars (seconds per denoising step)
        bx0, span = -1.0, 5.2
        sec = lambda s: span * s / 0.17  # noqa: E731
        rows = (("guidance 1 · one pass", STEP_ONE, MUTED, False), ("guidance 7.5 · one batch of two", STEP_CFG, GUI, False),
                ("two passes, one after another", 2 * STEP_ONE, FAINT, True))
        tbars = m.VGroup()
        for i, (lab, s, col, ghost) in enumerate(rows):
            y = -1.05 - 0.72 * i
            r_ = m.Rectangle(width=sec(s), height=0.42).move_to([bx0 + sec(s) / 2, y, 0])
            if ghost:
                r_ = m.DashedVMobject(r_, num_dashes=40).set_stroke(MUTED, 2)
            else:
                r_.set_stroke(col, 2).set_fill(FILLS.get(col, CARD), 1)
            words = text(lab, 22, INK if not ghost else MUTED).move_to([0, y, 0]).align_to([bx0 - 0.25, 0, 0], m.RIGHT)
            num = code(f"{s:.3f} s", 22, col if not ghost else MUTED,
                       weight=m.BOLD if not ghost else m.NORMAL).next_to(r_, m.RIGHT, buff=0.18)
            tbars.add(m.VGroup(r_, words, num))
        t_title = text("seconds per denoising step, the lighthouse on this Mac", 22, INK, weight=m.BOLD).move_to([0, -0.35, 0])
        ratio = pill(f"{STEP_CFG / STEP_ONE:.1f}× per step", GUI, LIME_FILL, 22).move_to([0, -3.35, 0])
        assert tbars.get_left()[0] > -6.6 and tbars.get_right()[0] < 6.6, (tbars.get_left(), tbars.get_right())

        dlines = m.VGroup(*[m.VGroup(text(name, 22, INK), code(val, 24, CON, weight=m.BOLD)) for name, val in DEFAULTS])
        for row in dlines:
            row[1].move_to([0, 0, 0])
        dlines.arrange(m.DOWN, buff=0.16, aligned_edge=m.LEFT)
        for row in dlines:
            row[1].move_to([0, row[0].get_y(), 0]).align_to([dlines.get_left()[0] + 3.9, 0, 0], m.LEFT)
        d_t = text("typical default guidance scales", 22, INK, weight=m.BOLD)
        d_c = cite("library defaults (diffusers v0.42), October 2026")
        one_pass = text("guidance distilled in: one pass (Episode 12's idea)", 20, GUI, weight=m.BOLD)
        one_pass.next_to(dlines[3][1], m.RIGHT, buff=0.35)
        d_in = m.VGroup(d_t, m.VGroup(dlines, one_pass), d_c).arrange(m.DOWN, buff=0.2, aligned_edge=m.LEFT)
        dcard = boxed(d_in, CON, 0.3).move_to([0, -1.75, 0])
        assert dcard.get_bottom()[1] > -3.6 and dcard.get_top()[1] < min(mix_box.get_bottom()[1], f2.get_bottom()[1]) - 0.08, (dcard.get_bottom(), dcard.get_top())
        assert one_pass.get_right()[0] < 6.6, one_pass.get_right()

        self.say(
            "[The price: the network runs twice at every step].",
            m.AnimationGroup(self.new_header("06 · the cost", "Two passes per step"), m.FadeIn(net), m.FadeIn(b1), m.Create(f1),
                             m.FadeIn(l1), m.FadeIn(b2), m.Create(f2), m.FadeIn(l2), m.GrowArrow(go)),
        )
        self.say(
            "[In practice, both share one batch of two]: [the picture with the prompt, and with an empty one], [mixed at the end].",
            m.AnimationGroup(m.FadeIn(brace), m.FadeIn(br_l)),
            m.AnimationGroup(m.GrowArrow(a_c), m.FadeIn(o_c), m.GrowArrow(a_u), m.FadeIn(o_u)),
            m.AnimationGroup(m.GrowArrow(j_c), m.GrowArrow(j_u), m.Write(mixf), m.Create(mix_box)),
        )
        self.say(
            "[On this Mac, a guided step takes zero point one four one seconds], [against zero point zero eight three without]: "
            "[one point seven times, not two].",
            m.AnimationGroup(m.FadeIn(t_title), m.FadeIn(tbars[1][1]), m.GrowFromEdge(tbars[1][0], m.LEFT), m.FadeIn(tbars[1][2])),
            m.AnimationGroup(m.FadeIn(tbars[0][1]), m.GrowFromEdge(tbars[0][0], m.LEFT), m.FadeIn(tbars[0][2])),
            m.AnimationGroup(m.FadeIn(tbars[2]), m.FadeIn(ratio, shift=0.1 * m.UP)),
        )
        self.say(
            "[Typical defaults]: [seven point five for Stable Diffusion], [five for SDXL], [seven for Stable Diffusion three].",
            m.AnimationGroup(m.FadeOut(tbars), m.FadeOut(t_title), m.FadeOut(ratio), m.FadeIn(dcard[0]), m.FadeIn(d_t), m.FadeIn(d_c)),
            m.FadeIn(dlines[0], shift=0.1 * m.UP),
            m.FadeIn(dlines[1], shift=0.1 * m.UP),
            m.FadeIn(dlines[2], shift=0.1 * m.UP),
        )
        self.say(
            "[And FLUX dev takes three point five], [baked into a single pass]: [the idea from episode twelve, applied to guidance].",
            m.FadeIn(dlines[3], shift=0.1 * m.UP),
            m.FadeIn(one_pass, shift=0.1 * m.LEFT),
            m.Circumscribe(one_pass, color=GUI, buff=0.08),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes=(
            f"Drop (hide the label in training: p_uncond = {P_UNCOND} in our run), ask (two predictions per step, one batch of "
            "two: 1.7× per step on this Mac), push (ε̃ = ε_u + w·(ε_c − ε_u), guidance scale w, 1 = no guidance; Ho & "
            "Salimans' w′ = w − 1). A little guidance helps open-ended prompts (the lighthouse) and ImageNet's FID (best "
            f"at scale {HS_BEST}); on our 10-class digits any guidance worsened cFID ({CF[1]} → {CF[8]}) while label accuracy "
            f"rose ({ACC[1]} → {ACC[8]}); high scales cost variety and burn colours. Next: Episode 15, Words to Pictures."))
        halo = glow(CON, radius=4.0)
        words = m.VGroup(*[title_text(w, 58, c) for w, c in (("Drop", CON), ("Ask", UNC), ("Push", GUI))])
        subs = m.VGroup(*[code(s, 20, MUTED) for s in ("the label, 1 in 10", "with and without it", "past the labelled guess")])
        words.arrange(m.RIGHT, buff=2.7)
        on_baseline(words)
        for w_, s in zip(words, subs):
            s.set_x(w_.get_x())
        line_y = min(w_.get_bottom()[1] for w_ in words) - 0.45
        for s in subs:
            sit(s, line_y)
        mid = baseline(words[0]) + 0.3
        dots = m.VGroup(*[m.Dot([(l_.get_right()[0] + r_.get_left()[0]) / 2, mid, 0], radius=0.07, color=MUTED)
                          for l_, r_ in zip(words, words[1:])])
        row = m.VGroup(words, dots, subs).move_to(1.0 * m.UP)
        assert row.width < 12.6 and all(a.get_right()[0] + 0.3 < b.get_left()[0] for a, b in zip(subs, subs[1:])), [(s_.get_left()[0], s_.get_right()[0]) for s_ in subs]
        f = formula(r"\tilde\varepsilon", "=", r"\varepsilon_u", "+", "w", r"\,(", r"\varepsilon_c", "-", r"\varepsilon_u", ")", size=46)
        f[0].set_color(GUI)
        f[2].set_color(UNC)
        f[4].set_color(CON)
        f[6].set_color(CON)
        f[8].set_color(UNC)
        f.next_to(row, m.DOWN, buff=0.55)
        f_n = cite("w: the guidance scale · 1 = no guidance").next_to(f, m.DOWN, buff=0.14)
        tag = text("A little push sharpens; a big one costs variety and burns colour.", 28, MUTED)
        mascot = pixel(0.42)
        nxt = pill("NEXT · WORDS TO PICTURES", CON, PINK_FILL, 20, mono=True)
        foot = m.VGroup(tag, m.VGroup(mascot, nxt).arrange(m.RIGHT, buff=0.3)).arrange(m.DOWN, buff=0.35)
        foot.next_to(f_n, m.DOWN, buff=0.4)
        assert foot.get_bottom()[1] > -3.6 and tag.width < 13.0, (foot.get_bottom(), tag.width)

        self.say(
            "[To sum up]: [drop the label], [ask twice], [push past].",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.3 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(dots[0]), m.FadeIn(words[1], shift=0.3 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(dots[1]), m.FadeIn(words[2], shift=0.3 * m.UP), m.FadeIn(subs[2]), m.FadeIn(f), m.FadeIn(f_n)),
        )
        self.say(
            "[A little push sharpens what you asked for]; [a big one costs variety], [and burns the colours].",
            m.Circumscribe(f, color=GUI, buff=0.1),
            m.Write(tag),
            m.Wait(0.1),
        )
        self.say(
            "[Next, prompts made of words instead of labels].",
            m.AnimationGroup(m.FadeIn(mascot, shift=0.2 * m.UP), m.FadeIn(nxt, shift=0.2 * m.UP)),
        )
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, f, f_n, foot)], run_time=0.9)
