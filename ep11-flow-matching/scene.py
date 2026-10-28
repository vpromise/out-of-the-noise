"""Out of the Noise · Episode 11 · Flow Matching (Module 3 · Faster Sampling)

Train on straight lines from data to noise and learn the velocity along them. Running examples: the swirl (the
mechanism: diffusion_reference.py and the two trained 45,954-parameter swirl MLPs — one trained with DDPM's ε-loss,
one with flow matching) and the digit model (the measured result: the same 3.94 M-parameter network trained twice on
this Mac, mac_runs.md §3–4).

The honest story (toy.md §4, "Caveats (important for Ep 11)"): flow matching's TRAINING pairs are straight lines
(length ÷ chord 1.00 vs 1.22 on average for DDPM's noising arcs, 5,000 pairs), but the SAMPLING paths of the learned,
averaged velocity field bend (median 1.56 for the flow model vs 1.14 for the DDPM model's ODE in its x_t picture —
each in its own coordinates; exact fields 1.56 vs 1.08). With perfect models, flow Euler IS DDIM on the σ grid
σ = t/(1 − t) (max difference 0.0005 after 16 steps), so its per-step advantage on the swirl is step placement: 16
even flow steps put 7 step starts where signal beats noise (σ < 1), 16 even DDPM steps only 4 (Episode 10's lesson).
Stranded (> 0.1 from the swirl, trained models, same 2,000 noise points): 25.1 / 6.8 / 2.4 % (flow) vs 46.5 / 35.5 /
15.6 % (DDIM) at 8 / 16 / 32 steps; at 2 and 4 steps both are poor (40.4 / 46.2 % vs 42.2 / 38.2 %). Reflow (Liu et
al. 2022) straightens the sampling paths: its new pairs, computed here with our trained flow, do not cross (0 of 435
line pairs among 30 lines, vs 95 for random pairs); we did not retrain. On the digit model (cFID = FID's formula on our
own MNIST classifier's features, NOT Inception FID; floor 2.06): 10 steps 4.57 vs 4.57; 5 steps 9.28 vs 21.6; 2 steps
85.3 vs 939; 1 step 886 (a blurry average digit) vs 943 (speckles).

Every number on screen is computed below (diffusion_reference.py, the trained swirl MLPs, the run's arrays) and
asserted, or loaded with measured(...), or cited with its source.

Sources (research/notes.md §11, Appendix A rows 4/11 and 5/11 and 11/18/23 FLUX size, Appendix C row 11; read
October 2026):
- Lipman, Chen, Ben-Hamu, Nickel & Le, "Flow Matching for Generative Modeling", arXiv:2210.02747 (6 Oct 2022), ICLR
  2023 (t = 0 noise in their notation). Liu, Gong & Liu, "Flow Straight and Fast" (rectified flow, reflow),
  arXiv:2209.03003 (7 Sep 2022), ICLR 2023 (t = 0 noise): Table 1, CIFAR-10, one Euler step: 1-rectified flow FID 378,
  2-rectified flow (one reflow) 12.21, 2-rectified + distillation 4.85; full ODE solve (RK45) 2.58 → 3.36 (read from
  the arXiv HTML, October 2026; notes §11 quotes 378 and 4.85).
- Salimans & Ho 2022 (v-prediction, Episode 5): v = √ᾱ·ε − √(1 − ᾱ)·x₀ — not the flow-matching velocity.
- Esser et al. (Stability AI), "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis",
  arXiv:2403.03206 (5 Mar 2024), ICML 2024: Stable Diffusion 3, "we scale our models up to 8B parameters",
  logit-normal timestep sampling. Black Forest Labs, FLUX.1 announcement (1 Aug 2024): "scaled to 12B parameters",
  "building on flow matching" (HF metadata: 11.9 B counted). Fast-moving facts dated October 2026.
- research/experiments/toy.md §4 + toy.json (flow/*, training/flow) and mac_runs.md §3–4 + mac_runs.json (headline
  cfid_*, digit_models.flow/ddpm wall_min; arrays digit_samples_*_w1.npz, digit_trajectory_7.npz).

Illustrative (computed, but chosen): the single example pair in section 1; the 12 pairs drawn (the first 12 of toy.md's
recipe); the query point (0.5, 0.3) at t = 0.5 and the 24 lines drawn through it; the 8 traced particles (the first 8
of the shared noise); the 30 lines of the reflow picture; digits 10–19 of the grids.

    VOICE=draft manimgx inspect scene.py      # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import sys
import warnings
from pathlib import Path

import numpy as np

EP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EP_DIR.parent))
from kit import *  # noqa: E402,F403
from diffusion_reference import *  # noqa: E402,F403

warnings.filterwarnings("ignore", message=".*encountered in matmul")  # NumPy/Accelerate's spurious warnings
warnings.filterwarnings("ignore", message=".*divide by zero.*")

PREFIX = "ep11"
ARR = EXPERIMENTS / "arrays"
MAC = lambda *keys: measured("mac_runs", "headline", *keys)  # noqa: E731
FLOWJ = lambda *keys: measured("toy", "flow", *keys)  # noqa: E731
TOY = lambda *keys: measured("toy", "headline", *keys)  # noqa: E731
AB = ALPHA_BAR
assert T == 1000

FM_C = FLOW        # flow matching: straight lines, its velocity, its samples (LIME)
DF_C = TIME        # the diffusion (DDPM-trained) model: paths shaped by the noise schedule (BLUE)
ORIGIN_ = np.zeros(3)


def ratio(xs):
    """Path length ÷ straight distance between the ends, per path (xs: steps × n × 2)."""
    xs = np.asarray(xs, dtype=float)
    return np.linalg.norm(np.diff(xs, axis=0), axis=2).sum(0) / np.linalg.norm(xs[-1] - xs[0], axis=1)


# ---------------------------------------------------------------------------- the two swirl networks

V_NET = load_swirl_flow()
E_NET = load_swirl_model()
MLP_PARAMS = TOY("mlp_params")
assert MLP_PARAMS == count_params(load_weights("swirl_flow.npz")) == 45954
FLOW_TRAIN = measured("toy", "training", "flow")
assert FLOW_TRAIN["steps"] == 40000 and FLOW_TRAIN["batch"] == 512 and round(FLOW_TRAIN["train_seconds"]) == 94
Z = gaussian_noise(2000, seed=5)                                    # toy.md's shared noise for every comparison

# ---------------------------------------------------------------------------- section 1: the straight line, one pair

PAIR_X0 = swirl_curve(200)[0][195]                                  # an example data point: the outer arm's tip
PAIR_EPS = np.array([1.8, 0.8])                                     # an example noise point ("for example")
LINE_TS = np.linspace(0, 1, 101)
PAIR_LINE = np.array([flow_interpolant(PAIR_X0[None], t, PAIR_EPS[None])[0] for t in LINE_TS])
PAIR_ARC = np.array([q_sample(PAIR_X0[None], t, PAIR_EPS[None])[0] for t in np.linspace(0, 1000, 201)])
assert abs(ratio(PAIR_LINE[:, None])[0] - 1) < 1e-9 and ratio(PAIR_ARC[:, None])[0] > 1.08

# many pairs: toy.md §4's recipe (x0 = swirl(5000, seed=3), eps = gaussian_noise(5000, seed=4))
TP_X0, TP_EPS = swirl(5000, seed=3), gaussian_noise(5000, seed=4)
TP_DDPM = np.array([q_sample(TP_X0, t, TP_EPS) for t in np.linspace(1000, 0, 101)])
TP_FLOW = np.array([flow_interpolant(TP_X0, t, TP_EPS) for t in np.linspace(1, 0, 101)])
RATIO_ARC, RATIO_LINE = float(ratio(TP_DDPM).mean()), float(ratio(TP_FLOW).mean())
assert abs(RATIO_ARC - FLOWJ("training_paths", "straightness_ddpm", "mean")) < 1e-3
assert RATIO_LINE == 1.0 or abs(RATIO_LINE - 1) < 1e-9
assert TOY("training_path_ratio_ddpm_mean") == 1.217 and f"{RATIO_ARC:.2f}" == "1.22"   # "twenty-two percent longer"
assert np.abs(np.array(FLOWJ("training_paths", "paths_ddpm")) - TP_DDPM[:, :8].transpose(1, 0, 2)).max() < 1e-3
N_PAIRS = 12
PAIRS_IN = [i for i in range(40) if np.abs(TP_EPS[i]).max() < 2.15][:N_PAIRS]         # inside the drawn plane

# ---------------------------------------------------------------------------- section 2: training, the averaged velocity

TRAIN_SRC = "\n".join([
    "x0 = swirl(512, seed=step)",
    "eps = gaussian_noise(512, seed=step)",
    "t = rng.uniform(0, 1, 512)",
    "x_t = flow_interpolant(x0, t, eps)  # on the line",
    "target = flow_velocity(x0, eps)     # eps - x0",
    "v_hat = v_model(x_t, t)             # the network",
    "loss = np.mean((v_hat - target) ** 2)",
])
_ns = dict(swirl=swirl, gaussian_noise=gaussian_noise, flow_interpolant=flow_interpolant, flow_velocity=flow_velocity,
           np=np, rng=np.random.default_rng(0), step=0, v_model=V_NET)
exec(TRAIN_SRC, _ns)
assert len(TRAIN_SRC.split("\n")) <= 10 and 0.4 < _ns["loss"] < 1.2, _ns["loss"]   # the excerpt runs as shown

Q_PT, Q_T = np.array([0.5, 0.3]), 0.5
_bx0, _beps = swirl(200000, seed=11), gaussian_noise(200000, seed=12)
_bxt = flow_interpolant(_bx0, Q_T, _beps)
_near = np.flatnonzero(np.linalg.norm(_bxt - Q_PT, axis=1) < 0.05)
N_NEAR = len(_near)
V_EXACT = exact_velocity(Q_PT[None], Q_T)[0]                        # E[ε − x0 | x_t]: the average over every line
V_NETQ = V_NET(Q_PT[None], Q_T)[0]                                  # what the trained network answers there
V_SAMPLE = (_beps - _bx0)[_near].mean(0)
assert N_NEAR == 468 and np.linalg.norm(V_SAMPLE - V_EXACT) < 0.06 and np.linalg.norm(V_NETQ - V_EXACT) < 0.03
_vel = (_beps - _bx0)[_near]
_ang = np.arctan2(_vel[:, 1], _vel[:, 0])
_order = _near[np.argsort(_ang)]
STAR = _order[np.linspace(0, len(_order) - 1, 24).round().astype(int)]    # 24 of them, spread round the circle
STAR_X0, STAR_EPS = _bx0[STAR], _beps[STAR]
# the highlighted pair: the one whose velocity points farthest from the average
_cos = ((STAR_EPS - STAR_X0) @ V_EXACT) / (np.linalg.norm(STAR_EPS - STAR_X0, axis=1) * np.linalg.norm(V_EXACT))
HERO = int(np.argmin(np.abs(_cos + 0.2)))                           # a clearly different direction (≈ 100°)
assert np.ptp(np.degrees(_ang)) > 300                               # "lines from every direction"

# ---------------------------------------------------------------------------- section 3: sampling

RUN32 = flow_euler(V_NET, Z, 32)
OFF_F = {s: off_swirl(flow_euler(V_NET, Z, s).xs[-1]) for s in (1, 2, 4, 8, 16)}
OFF_F[32] = off_swirl(RUN32.xs[-1])
DDIM_RUNS = {s: ddim_sample(E_NET, Z, s) for s in (2, 4, 8, 16, 32)}
OFF_D = {s: off_swirl(DDIM_RUNS[s].xs[-1]) for s in DDIM_RUNS}
for _s in (2, 4, 8, 16, 32):
    _q = FLOWJ("trained", "quality_vs_steps", str(_s))
    assert abs(OFF_F[_s] - _q["flow"]["off"]) < 1e-6 and abs(OFF_D[_s] - _q["ddim"]["off"]) < 1e-6, _s
assert TOY("flow8_trained_off") == 0.2505 and TOY("ddim8_trained_off") == 0.465
assert TOY("flow16_trained_off") == 0.068 and TOY("ddim16_trained_off") == 0.355
assert 0.24 < OFF_F[8] < 0.26 and 0.45 < OFF_D[8] < 0.5                  # "a quarter … nearly half"
assert OFF_F[16] < 0.07 and OFF_D[16] > 1 / 3                           # "under seven percent … over a third"
assert 0.015 < OFF_F[32] < 0.025 and 0.155 < OFF_D[32] < 0.165          # "about two … sixteen"
assert OFF_F[2] > 0.4 and OFF_D[2] > 0.4 and OFF_F[4] > 0.4 and 0.35 < OFF_D[4] < OFF_F[4]   # 2 and 4: both poor
PCT = lambda v: f"{100 * v:.1f}%"  # noqa: E731
ARROW_K = 0.9                                                        # velocity arrows drawn at 0.9 × their length

SHOW = [i for i in range(2000) if np.abs(Z[i]).max() < 2.55][:1000]    # 1,000 starting points inside the plane
SHOW_RACE = [i for i in range(2000) if np.abs(Z[i]).max() < 1.95][:1000]
TRACE = np.arange(8)                                                 # the first 8 of the shared noise
FLOW200 = flow_euler(V_NET, Z, 200).xs
DDIM200 = ddim_sample(E_NET, Z, 200).xs
STRAIGHT_F, STRAIGHT_D = float(np.median(ratio(FLOW200))), float(np.median(ratio(DDIM200)))
assert abs(STRAIGHT_F - FLOWJ("trained", "straightness_flow", "median")) < 2e-3
assert abs(STRAIGHT_D - FLOWJ("trained", "straightness_ddpm_ode", "median")) < 2e-3
assert (f"{STRAIGHT_F:.2f}", f"{STRAIGHT_D:.2f}") == ("1.56", "1.14")
EXACT_STRAIGHT = (FLOWJ("exact", "straightness_flow", "median"), FLOWJ("exact", "straightness_ddpm_ode", "median"))
SIGMA_VIEW = FLOWJ("trained", "straightness_ddpm_ode_sigma_view", "median")
assert EXACT_STRAIGHT == (1.56, 1.075) and SIGMA_VIEW == 1.0
DATA_MEAN = SWIRL.weights @ SWIRL.means
assert np.allclose(flow_euler(exact_velocity, Z[:50], 1).xs[-1], DATA_MEAN)      # one exact step: the data's average
assert np.abs(FLOW200[:, TRACE]).max() < 2.0 and np.abs(DDIM200[:, TRACE]).max() < 2.0

# the digit 7 under 10 flow steps (arrays/digit_trajectory_7.npz, noise np.random.default_rng(7))
TR7 = np.load(ARR / "digit_trajectory_7.npz")
assert int(TR7["label"]) == 7
F10_T = TR7["flow_euler10_w1_t"].astype(float)
assert np.allclose(F10_T, np.linspace(1, 0, 11)[:-1])
F10_X = TR7["flow_euler10_w1_x_t"].astype(float)
F10_G = TR7["flow_euler10_w1_x0_hat"].astype(float)
assert np.allclose(F10_X[0], TR7["xT"].astype(float))
D10_FINAL = TR7["ddpm_ddim10_w1_final"].astype(float)
SAME7 = float(np.corrcoef(TR7["flow_euler10_w1_final"].astype(float).ravel(), D10_FINAL.ravel())[0, 1])
assert SAME7 > 0.97

# ---------------------------------------------------------------------------- section 5: the twist (placement)

with np.errstate(divide="ignore"):
    _tf = np.linspace(1, 0, 17)
    _fe = flow_euler(exact_velocity, Z, 16).xs[-1]
    _dd = ddim_sample(exact_eps, Z, timestep_from_sigma(_tf / (1 - _tf))).xs[-1]
    SIG_FLOW = _tf[:-1] / (1 - _tf[:-1])                              # the noise level each flow step starts at (∞ first)
IDENTITY = float(np.abs(_fe - _dd).max())
assert abs(IDENTITY - FLOWJ("exact_flow16_vs_ddim_on_sigma_grid_max_diff")) < 1e-6 and IDENTITY < 6e-4
SIG_DDIM = sigma_from_alpha_bar(AB[ddim_timesteps(16)[:-1]])
LOW_F, LOW_D = int((SIG_FLOW < 1).sum()), int((SIG_DDIM < 1).sum())
assert (LOW_F, LOW_D) == (7, 4)                                     # "seven of sixteen … only four"
assert f"{SIG_FLOW[-1]:.3f}" == "0.067" and f"{SIG_DDIM[-1]:.2f}" == "0.21"

# ---------------------------------------------------------------------------- section 6: the digit model

GRID_IDX = list(range(10, 20))                                      # samples 10 … 19 ask for labels 0 … 9
N_GRID = 8                                                          # the digits section shows labels 0 … 7
DIGITS = {}
for _cfg in ("ddpm_ddim10", "ddpm_ddim5", "ddpm_ddim2", "ddpm_ddim1", "flow_euler10", "flow_euler5", "flow_euler2",
             "flow_euler1"):
    _f = np.load(ARR / f"digit_samples_{_cfg}_w1.npz")
    assert "seed 1234" in str(_f["note"]) and list(_f["labels"][GRID_IDX]) == list(range(10))
    DIGITS[_cfg] = _f["samples"][GRID_IDX]
_corr = lambda a, b: float(np.corrcoef(a.ravel().astype(float), b.ravel().astype(float))[0, 1])  # noqa: E731
SAME_DIGITS = float(np.median([_corr(DIGITS["ddpm_ddim10"][i], DIGITS["flow_euler10"][i]) for i in range(10)]))
assert SAME_DIGITS > 0.9                                            # "the two networks draw the same digits"
CFID = {(m_, s): MAC(f"cfid_{m_}{s}") for m_ in ("ddim", "flow") for s in (10, 5, 2, 1)}
for (_m, _s), _v in CFID.items():
    _key = f"ddpm_ddim{_s}_w1" if _m == "ddim" else f"flow_euler{_s}_w1"
    assert abs(measured("mac_runs", "digit_sampling", "configs", _key, "cfid") - _v) < 1e-3
FMT = lambda v: f"{v:.2f}" if v < 10 else f"{v:.1f}" if v < 100 else f"{v:.0f}"  # noqa: E731
CF = {k: FMT(v) for k, v in CFID.items()}
assert [CF[("ddim", s)] for s in (10, 5, 2, 1)] == ["4.57", "21.6", "939", "943"]
assert [CF[("flow", s)] for s in (10, 5, 2, 1)] == ["4.57", "9.28", "85.3", "886"]
FLOOR = MAC("cfid_real_data_floor")
assert f"{FLOOR:.2f}" == "2.06"
MIN_FLOW, MIN_DDPM = MAC("digit_flow_train_minutes"), MAC("digit_ddpm_train_minutes")
assert (MIN_FLOW, MIN_DDPM) == (29.2, 31.8)
DIGIT_PARAMS = MAC("digit_unet_params")
assert DIGIT_PARAMS == measured("mac_runs", "digit_models", "flow", "params") == 3940801
assert measured("mac_runs", "digit_models", "flow", "objective") == "flow"

# ---------------------------------------------------------------------------- section 7: reflow


def _crossings(a, b):
    """Every crossing point between the segments a[i]–b[i] (and how many pairs of segments there are)."""
    pts, n = [], len(a)
    for i in range(n):
        for j in range(i + 1, n):
            p, r = a[i], b[i] - a[i]
            q, s = a[j], b[j] - a[j]
            den = r[0] * s[1] - r[1] * s[0]
            if abs(den) < 1e-12:
                continue
            u = ((q - p)[0] * s[1] - (q - p)[1] * s[0]) / den
            w = ((q - p)[0] * r[1] - (q - p)[1] * r[0]) / den
            if 0 < u < 1 and 0 < w < 1:
                pts.append(p + u * r)
    return np.array(pts).reshape(-1, 2), n * (n - 1) // 2


RF_N = 30
RF_X0, RF_Z = swirl(RF_N, seed=3), gaussian_noise(RF_N, seed=4)     # random pairs, as flow matching trains on
RF_END = flow_euler(V_NET, RF_Z, 200).xs[-1]                         # reflow's pairs: where our flow carries each z
CROSS_RAND, N_LINE_PAIRS = _crossings(RF_X0, RF_Z)
CROSS_REFLOW, _ = _crossings(RF_END, RF_Z)
assert (len(CROSS_RAND), len(CROSS_REFLOW), N_LINE_PAIRS) == (95, 0, 435)
_b300 = swirl(300, seed=3), gaussian_noise(300, seed=4)
SHARE_RAND = len(_crossings(*_b300)[0]) / 44850
SHARE_REFLOW = len(_crossings(flow_euler(V_NET, _b300[1], 200).xs[-1], _b300[1])[0]) / 44850
assert f"{100 * SHARE_RAND:.0f}" == "23" and f"{100 * SHARE_REFLOW:.1f}" == "0.1"
assert np.abs(RF_Z).max() < 2.3
RF_TABLE = (("first flow, one step", "378"), ("after one reflow", "12.2"), ("reflow + distillation", "4.85"))
_notes = (EP_DIR.parent / "research" / "notes.md").read_text()
assert "1-rectified flow (no distillation) FID **378**" in _notes and "**2-rectified + distill 4.85**" in _notes
assert "we scale our models up to 8B parameters" in _notes and "scaled to 12B parameters" in _notes
assert "Rectified flow is a recent generative model formulation that connects data and noise in a straight line." in _notes

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


def formula(*parts, size=44, color=INK):
    return m.MathTex(*parts, font_size=size, color=color)


def flow_weight(t):
    """0 while the noise wins, 1 once the picture wins (log SNR of the straight path, (1 − t)²/t²)."""
    if t <= 0:
        return 1.0
    if t >= 1:
        return 0.0
    w = float(np.clip((2 * np.log((1 - t) / t) + 0.6) / 1.2, 0, 1))
    return w * w * (3 - 2 * w)


def state_rgb(x, t):
    return tint(np.clip(x, -1, 1), mix(VIOLET, TEAL, flow_weight(t)), -1, 1)


def guess_rgb(x):
    return tint(np.clip(x, -1, 1), MODEL, -1, 1)


def digit_row(images, color, height, gap_px=5, k=8):
    """Several 28×28 digits (uint8) as ONE picture with dark gaps, `height` units tall (crisp: nearest neighbour)."""
    n = len(images)
    canvas = np.zeros((28, n * 28 + (n - 1) * gap_px))
    for i, img in enumerate(images):
        canvas[:, i * (28 + gap_px): i * (28 + gap_px) + 28] = np.asarray(img, dtype=float) / 255
    rgb = tint(canvas, color, 0, 1)
    rgb = np.repeat(np.repeat(rgb, k, axis=0), k, axis=1)
    return picture(rgb, height, pixelated=False)


class Readout(m.VGroup):
    """A mono number display left-aligned at a fixed point; .show(v) updates it."""

    def __init__(self, left, y, fmt, value, size=40, color=TIME):
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


def swirl_band(plot, width=9, opacity=0.2):
    """The swirl's two arms as wide faint TEAL strokes: where the data lives."""
    return m.VGroup(*[m.VMobject().set_points_smoothly([at(plot, p) for p in arm[::3]] + [at(plot, arm[-1])])
                      .set_stroke(DATA, width, opacity=opacity) for arm in swirl_curve(200)]).set_z_index(-1)


def path_line(plot, pts, color, width=2.6, opacity=0.95):
    return m.VMobject().set_points_as_corners([at(plot, p) for p in pts]).set_stroke(color, width, opacity=opacity)


def chord(plot, a, b, color=INK, width=2.5):
    return m.DashedLine(at(plot, a), at(plot, b), dash_length=0.09).set_stroke(color, width, opacity=0.75)


def grow_paths(plot, paths, lines, heads, seconds):
    """Draw paths (steps × n × 2) as they are walked: each line follows its head."""
    paths = np.asarray(paths, dtype=float)

    def step(_, a):
        i = a * (len(paths) - 1)
        j = int(np.floor(i))
        k = min(j + 1, len(paths) - 1)
        now = paths[j] + (paths[k] - paths[j]) * (i - j)
        for n, (line, head) in enumerate(zip(lines, heads)):
            pts = [at(plot, p) for p in paths[: j + 1, n]] + [at(plot, now[n])]
            if len(pts) < 2:
                pts = pts + pts
            line.set_points_as_corners(pts)
            head.move_to(at(plot, now[n]))

    return m.UpdateFromAlphaFunc(m.VGroup(lines, heads), step, run_time=seconds, rate_func=m.linear)


def arrow_to(a, b, color=MUTED, width=4, buff=0.12):
    return m.Arrow(a, b, buff=buff, color=color, stroke_width=width, max_tip_length_to_length_ratio=0.25)


def vec_arrow(plot, p, v, color, scale=0.6, width=6):
    """A velocity drawn at point p (data units), `scale` × its data length."""
    a = at(plot, p)
    b = at(plot, np.asarray(p) + scale * np.asarray(v))
    return m.Arrow(a, b, buff=0, color=color, stroke_width=width, max_tip_length_to_length_ratio=0.3,
                   max_stroke_width_to_length_ratio=100)


def field_arrows(plot, v_of, t, n=11, color=FM_C, max_len=0.36, opacity=0.55, margin=0.08):
    """The direction of travel −v̂(x, t) on an n × n grid (one network call for all), longest = max_len units."""
    xs = np.linspace(plot.x0 + margin * (plot.x1 - plot.x0), plot.x1 - margin * (plot.x1 - plot.x0), n)
    pts = np.array([(x, y) for y in xs for x in xs])
    vec = -np.asarray(v_of(pts, t), dtype=float) * plot.w / (plot.x1 - plot.x0)
    longest = np.max(np.linalg.norm(vec, axis=1)) or 1.0
    out = m.VGroup()
    for p, v in zip(pts, vec * (max_len / longest)):
        a = at(plot, p)
        out.add(m.Arrow(a, a + np.array([v[0], v[1], 0]), buff=0, color=color, stroke_width=2.6,
                        max_tip_length_to_length_ratio=0.35).set_opacity(opacity))
    return out


class Swarm(m.VGroup):
    """Swirl samples as dots: VIOLET while they are noise, then TEAL on an arm or RED stranded (> 0.1 away).
    `.go(xy)` moves every dot to its new place and recolours it on the way."""

    def __init__(self, plot, xy, radius=0.024, noise=True):
        super().__init__()
        self.plot, self.xy = plot, np.asarray(xy, dtype=float)
        self.cols = [NOISE] * len(self.xy) if noise else self._colours(self.xy)
        for p, c in zip(self.xy, self.cols):
            self.add(m.Dot(at(plot, p), radius=radius, color=c).set_opacity(0.9))

    @staticmethod
    def _colours(xy):
        far = nearest_on_swirl(xy)[0] > 0.1
        return [RED if f else DATA for f in far]

    def go(self, xy, run_time=1.2):
        start, end = self.xy.copy(), np.asarray(xy, dtype=float)
        c0, c1 = list(self.cols), self._colours(end)
        ramps = {}
        for a in (RED, DATA, NOISE):
            for b in (RED, DATA):
                ramps[(a, b)] = [mix(a, b, s / 8) for s in range(9)]
        self.xy, self.cols = end, c1

        def step(_, a):
            now = start + (end - start) * a
            s = int(round(8 * a))
            for d, p, ca, cb in zip(self, now, c0, c1):
                d.move_to(at(self.plot, p))
                if ca != cb:
                    d.set_color(ramps[(ca, cb)][s])

        return m.UpdateFromAlphaFunc(self, step, run_time=run_time)


def legend_line(color, label, dashed=False):
    mark = (m.DashedLine(ORIGIN_, [0.45, 0, 0], dash_length=0.08) if dashed else m.Line(ORIGIN_, [0.45, 0, 0]))
    return m.VGroup(mark.set_stroke(color, 4), text(label, 20, color if not dashed else MUTED)).arrange(m.RIGHT, buff=0.15)


def boxed(group, color=EDGE, buff=0.3, radius=0.16):
    box = card(group.width + 2 * buff, group.height + 2 * buff, color, CARD, radius=radius, stroke=2).move_to(group)
    return m.VGroup(box, group)


# ---------------------------------------------------------------------------- the episode


class Ep11FlowMatching(Episode):
    number = 11

    def lesson(self):
        self.hook()
        self.the_line()
        self.training()
        self.sampling()
        self.the_race()
        self.the_twist()
        self.digits()
        self.reflow()
        self.in_the_wild()
        self.recap()

    # ------------------------------------------------------------------ 0 · hook

    def hook(self):
        self.next_section("Hook", notes=(
            "The digit model's network (3,940,801 parameters, mac_runs.md §3) trained twice on this Mac with the same "
            f"recipe: DDPM's ε-loss ({MIN_DDPM} min) and rectified flow, target v = ε − x₀, t ~ U(0, 1) ({MIN_FLOW} min; the "
            "time difference is not a property of the objectives, mac_runs.md §8). Two steps each (DDIM η = 0 / flow "
            "Euler), samples 10–19 (labels 0–9), the same starting noise per column (numpy seed 1234): "
            f"arrays/digit_samples_*_w1.npz. cFID 2 steps: {CF[('ddim', 2)]} vs {CF[('flow', 2)]}."))
        h = 0.92
        rows = {}
        for key, y, color, name, sub in (("ddpm_ddim2", 0.85, DF_C, "trained as a diffusion model · 2 steps",
                                          f"noise target · {MIN_DDPM} min on this Mac"),
                                         ("flow_euler2", -1.55, FM_C, "trained with flow matching · 2 steps",
                                          f"velocity target · {MIN_FLOW} min on this Mac")):
            row = digit_row(DIGITS[key], DATA, h, gap_px=4).move_to([0, y, 0])
            frame = m.SurroundingRectangle(row, buff=0.08).set_stroke(EDGE, 2)
            lab = text(name, 24, color, weight=m.BOLD)
            sub_l = cite(sub)
            lab.next_to(frame, m.UP, buff=0.18).align_to(frame, m.LEFT)
            sub_l.next_to(lab, m.RIGHT, buff=0.35)
            sit(sub_l, baseline(lab))
            rows[key] = dict(row=row, frame=frame, lab=lab, sub=sub_l)
        chip_w = text("the same network · 3.9 M parameters", 22, INK, weight=m.BOLD)
        net = m.VGroup(card(chip_w.width + 0.5, 0.62, MODEL, AMBER_FILL, radius=0.14), chip_w)
        net[1].move_to(net[0])
        net.move_to([0, 2.35, 0])
        cap = cite("generated on this Mac · the digit model · labels 0–9 · same starting noise per column (seed 1234)")
        cap.move_to([0, -3.3, 0])
        speck = pill("speckles", RED, RED_FILL, 20).next_to(rows["ddpm_ddim2"]["frame"], m.DOWN, buff=0.14).align_to(rows["ddpm_ddim2"]["frame"], m.RIGHT)
        good = pill("digits", GREEN, GREEN_FILL, 20).next_to(rows["flow_euler2"]["frame"], m.DOWN, buff=0.14).align_to(rows["flow_euler2"]["frame"], m.RIGHT)
        for r in rows.values():
            assert r["frame"].get_left()[0] > -6.6 and r["sub"].get_right()[0] < 6.6, (r["frame"].get_left(), r["sub"].get_right())

        def labels(key):
            r = rows[key]
            return m.AnimationGroup(m.FadeIn(r["lab"], shift=0.1 * m.RIGHT), m.FadeIn(r["sub"]), m.Create(r["frame"]))

        self.say(
            "[The same network, trained twice on this Mac]: [once as a diffusion model], [once with flow matching].",
            m.AnimationGroup(self.new_header("00 · hook", "Same network, a new target"), m.FadeIn(net, shift=0.1 * m.DOWN)),
            labels("ddpm_ddim2"),
            labels("flow_euler2"),
        )
        self.say(
            "[Give both the same noise and only two steps]: [the diffusion model draws speckles], [flow matching draws digits].",
            m.FadeIn(cap),
            m.AnimationGroup(m.FadeIn(rows["ddpm_ddim2"]["row"]), m.FadeIn(speck, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(rows["flow_euler2"]["row"]), m.FadeIn(good, shift=0.1 * m.UP)),
        )
        self.say(
            "[Today: what flow matching learns], [and, honestly, where its edge comes from].",
            m.Circumscribe(net, color=MODEL, buff=0.08),
            m.Circumscribe(rows["flow_euler2"]["frame"], color=FM_C, buff=0.06),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 1 · the straight line

    def the_line(self):
        self.next_section("1 · A straight line from data to noise", notes=(
            "flow_interpolant: x_t = (1 − t)·x₀ + t·ε, t = 0 data, t = 1 noise (our convention = SD3's; Lipman et al. and "
            "Liu et al. put noise at t = 0, notes Appendix A). flow_velocity: v = dx_t/dt = ε − x₀, constant along the line. "
            "The example pair is illustrative (x₀ on the outer arm, ε = (1.55, −1.3)). The arc is q_sample (Episode 4's "
            "x_t = √ᾱ·x₀ + √(1 − ᾱ)·ε, linear β) for t = 0 … 1000 with the same pair. Many pairs: toy.md §4's recipe "
            f"(x₀ = swirl(5000, seed=3), ε = gaussian_noise(5000, seed=4)); {N_PAIRS} drawn. Length ÷ straight distance, "
            f"mean of 5,000: arcs {RATIO_ARC:.3f} (toy.json 1.217; median 1.056, 10% above 1.494), lines {RATIO_LINE:.0f}."))
        lim, size = 2.0, 5.2
        pl = plane((-3.45, -0.5, 0), size, lim)
        band = swirl_band(pl, 12, 0.22)
        frame = m.Square(size).set_stroke(EDGE, 2).move_to(pl.coords(0, 0))
        x0d = m.Dot(at(pl, PAIR_X0), radius=0.1, color=DATA).set_z_index(5)
        epd = m.Dot(at(pl, PAIR_EPS), radius=0.1, color=NOISE).set_z_index(5)
        x0l = tex("x_0", 36, DATA).next_to(x0d, m.LEFT, buff=0.14).shift(0.12 * m.UP)
        epl = tex(r"\varepsilon", 38, NOISE).next_to(epd, m.DOWN, buff=0.14)
        lab_d = m.VGroup(m.Dot(radius=0.08, color=DATA), tex("x_0", 30, DATA), text("a data point", 20, DATA)).arrange(m.RIGHT, buff=0.15)
        lab_n = m.VGroup(m.Dot(radius=0.08, color=NOISE), tex(r"\varepsilon", 32, NOISE), text("a noise point", 20, NOISE)).arrange(m.RIGHT, buff=0.15)
        m.VGroup(lab_d, lab_n).arrange(m.RIGHT, buff=0.6).next_to(frame, m.DOWN, buff=0.16)
        line = m.Line(at(pl, PAIR_X0), at(pl, PAIR_EPS)).set_stroke(FM_C, 5).set_z_index(3)
        marker = m.Dot(at(pl, PAIR_X0), radius=0.085, color=INK).set_z_index(6)
        ring = m.Circle(radius=0.15).set_stroke(INK, 2.5).move_to(marker).set_z_index(6)
        arc = path_line(pl, PAIR_ARC, DF_C, 4.5).set_z_index(2)

        rx = 0.55
        f_line = formula("x_t", "=", "(1-t)", r"\,x_0", "+", "t", r"\,\varepsilon", size=46)
        f_line[2].set_color(TIME)
        f_line[3].set_color(DATA)
        f_line[5].set_color(TIME)
        f_line[6].set_color(NOISE)
        f_line.move_to([0, 2.15, 0]).align_to([rx, 0, 0], m.LEFT)
        conv = m.VGroup(text("our convention: t = 0 is data, t = 1 is noise,", 20, INK),
                        cite("as in Stable Diffusion 3. Lipman et al. and"),
                        cite("Liu et al., 2022, count t the other way.")).arrange(m.DOWN, buff=0.06, aligned_edge=m.LEFT)
        conv.move_to([0, 1.3, 0]).align_to([rx, 0, 0], m.LEFT)
        t_read = Readout(rx, 0.45, "t = {:.2f}", 0.0, 34, TIME)
        f_vel = formula("v", "=", r"\frac{\mathrm{d}x_t}{\mathrm{d}t}", "=", r"\varepsilon", "-", "x_0", size=44)
        f_vel[0].set_color(FM_C)
        f_vel[4].set_color(NOISE)
        f_vel[6].set_color(DATA)
        f_vel.move_to([0, -0.45, 0]).align_to([rx, 0, 0], m.LEFT)
        f_vel.shift(0.15 * m.UP)
        same = text("the same at every t", 22, FM_C, weight=m.BOLD).next_to(f_vel, m.DOWN, buff=0.12).align_to(f_vel, m.LEFT)
        vel_arrows = m.VGroup(*[vec_arrow(pl, PAIR_LINE[k], (PAIR_EPS - PAIR_X0) / np.linalg.norm(PAIR_EPS - PAIR_X0), FM_C,
                                          0.45, 5).set_z_index(7) for k in (18, 50, 82)])
        f_arc = formula("x_t", "=", r"\sqrt{\bar\alpha_t}", r"\,x_0", "+", r"\sqrt{1-\bar\alpha_t}", r"\,\varepsilon", size=40)
        f_arc[2].set_color(TIME)
        f_arc[5].set_color(TIME)
        f_arc[3].set_color(DATA)
        f_arc[6].set_color(NOISE)
        arc_lab = text("Episode 4's noising path: an arc", 22, DF_C, weight=m.BOLD)
        arc_g = m.VGroup(arc_lab, f_arc).arrange(m.DOWN, buff=0.15, aligned_edge=m.LEFT)
        arc_g.move_to([0, -2.1, 0]).align_to([rx, 0, 0], m.LEFT)
        assert conv.get_right()[0] < 6.6 and same.get_right()[0] < 6.6 and f_line.get_right()[0] < 6.6, (conv.get_right(), same.get_right())

        # many pairs
        lines_many = m.VGroup(*[path_line(pl, TP_FLOW[:, i], FM_C, 2.4, 0.9) for i in PAIRS_IN]).set_z_index(3)
        arcs_many = m.VGroup(*[path_line(pl, TP_DDPM[:, i], DF_C, 2.4, 0.85) for i in PAIRS_IN]).set_z_index(2)
        ends = m.VGroup(*[m.Dot(at(pl, TP_X0[i]), radius=0.05, color=DATA) for i in PAIRS_IN],
                        *[m.Dot(at(pl, TP_EPS[i]), radius=0.05, color=NOISE) for i in PAIRS_IN]).set_z_index(5)
        s_line = m.VGroup(title_text(f"{RATIO_LINE:.2f}", 60, FM_C), text("straight lines", 22, FM_C, weight=m.BOLD))
        s_arc = m.VGroup(title_text(f"{RATIO_ARC:.2f}", 60, DF_C), text("noising arcs", 22, DF_C, weight=m.BOLD))
        for g in (s_line, s_arc):
            g.arrange(m.DOWN, buff=0.1)
        stats = m.VGroup(s_line, s_arc).arrange(m.RIGHT, buff=1.0).move_to([2.9, 0.35, 0])
        st_t = text("path length ÷ straight distance", 22, INK).next_to(stats, m.UP, buff=0.35)
        st_c = cite("mean of 5,000 random pairs · 12 drawn").next_to(stats, m.DOWN, buff=0.3)
        assert st_c.get_right()[0] < 6.6 and st_c.get_left()[0] > frame.get_right()[0] + 0.1, st_c.get_left()

        def slide(a0, a1, seconds=1.4):
            def step(_, a):
                t = a0 + (a1 - a0) * a
                p = flow_interpolant(PAIR_X0[None], t, PAIR_EPS[None])[0]
                marker.move_to(at(pl, p))
                ring.move_to(marker)
                t_read.show(round(t, 2))
                marker.set_color(mix(DATA, NOISE, t))
            return m.UpdateFromAlphaFunc(m.VGroup(marker, ring), step, run_time=seconds, rate_func=m.smooth)

        self.say(
            "[On the swirl, take one data point], [and one point of pure noise].",
            m.AnimationGroup(self.new_header("01 · the line", "A straight line from data to noise"), m.FadeIn(frame),
                             m.FadeIn(band), m.FadeIn(x0d, scale=0.5), m.FadeIn(x0l), m.FadeIn(lab_d)),
            m.AnimationGroup(m.FadeIn(epd, scale=0.5), m.FadeIn(epl), m.FadeIn(lab_n)),
        )
        self.say(
            "[Join them with a straight line]: [x t equals one minus t, times x zero, plus t times epsilon].",
            m.Create(line),
            m.Write(f_line),
        )
        self.say(
            "[At t equals zero we are on the data]; [at t equals one, on pure noise].",
            m.AnimationGroup(m.FadeIn(marker), m.FadeIn(ring), m.FadeIn(t_read), m.FadeIn(conv, shift=0.1 * m.UP)),
            slide(0.0, 1.0, 1.6),
        )
        self.say(
            "[Along the line, the velocity never changes]: [it is epsilon minus x zero].",
            m.AnimationGroup(m.LaggedStart(*[m.GrowArrow(a) for a in vel_arrows], lag_ratio=0.25), slide(1.0, 0.5, 1.2)),
            m.AnimationGroup(m.Write(f_vel), m.FadeIn(same, shift=0.1 * m.LEFT)),
        )
        self.say(
            "[Episode four's noising path for the same pair is an arc instead].",
            m.AnimationGroup(m.Create(arc, run_time=1.6), m.FadeIn(arc_g, shift=0.1 * m.UP)),
        )
        single = m.VGroup(x0d, epd, x0l, epl, lab_d, lab_n, line, marker, ring, arc, vel_arrows)
        right = m.VGroup(f_line, conv, t_read, f_vel, same, arc_g)
        self.say(
            "[Over many pairs], [the arcs run twenty-two percent longer than the straight line between their ends].",
            m.AnimationGroup(m.FadeOut(single), m.FadeOut(right), m.FadeIn(ends, scale=0.5),
                             m.LaggedStart(*[m.Create(p) for p in lines_many], lag_ratio=0.08, run_time=1.4),
                             m.LaggedStart(*[m.Create(p) for p in arcs_many], lag_ratio=0.08, run_time=1.6)),
            m.AnimationGroup(m.FadeIn(st_t), m.FadeIn(s_line, shift=0.1 * m.UP), m.FadeIn(s_arc, shift=0.1 * m.UP),
                             m.FadeIn(st_c)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 2 · training

    def training(self):
        self.next_section("2 · Learn the velocity", notes=(
            "The excerpt (6 lines) is executed at import with the trained swirl flow model as v_model (it runs; loss "
            f"{_ns['loss']:.3f} on one batch). The swirl flow MLP: {MLP_PARAMS:,} parameters, {FLOW_TRAIN['steps']:,} Adam "
            f"steps × {FLOW_TRAIN['batch']} pairs, {FLOW_TRAIN['train_seconds']:.1f} s on this Mac's CPU (toy.json training.flow). "
            "Names: flow matching (Lipman, Chen, Ben-Hamu, Nickel & Le, arXiv:2210.02747, 6 Oct 2022) and rectified flow "
            "(Liu, Gong & Liu, arXiv:2209.03003, 7 Sep 2022). v-prediction (Salimans & Ho 2022, Episode 5): "
            "v = √ᾱ·ε − √(1 − ᾱ)·x₀, not ε − x₀ (notes Appendix A). The averaged velocity: at the query point "
            f"({Q_PT[0]}, {Q_PT[1]}), t = {Q_T}, {N_NEAR} of 200,000 random pairs (swirl seed 11, noise seed 12) pass within 0.05; "
            f"24 drawn, spread round the circle. Their mean ({V_SAMPLE[0]:.2f}, {V_SAMPLE[1]:.2f}); the exact average "
            f"E[ε − x₀ | x_t] = exact_velocity = ({V_EXACT[0]:.2f}, {V_EXACT[1]:.2f}); the trained network "
            f"({V_NETQ[0]:.2f}, {V_NETQ[1]:.2f}). Arrows drawn at 0.6 × their length."))
        pl = Plot((-1.15, 2.15), (-1.3, 2.0), width=4.6, height=4.6, center=(4.0, -0.5, 0), axes=False)
        band = swirl_band(pl, 12, 0.2)
        frame = m.Rectangle(width=4.6, height=4.6).set_stroke(EDGE, 2).move_to(pl.coords(0.5, 0.35))
        hx0, heps = STAR_X0[HERO], STAR_EPS[HERO]
        hq = flow_interpolant(hx0[None], Q_T, heps[None])[0]
        h_line = m.Line(at(pl, hx0), at(pl, heps)).set_stroke(FM_C, 4).set_z_index(3)
        h_ends = m.VGroup(m.Dot(at(pl, hx0), radius=0.08, color=DATA), m.Dot(at(pl, heps), radius=0.08, color=NOISE)).set_z_index(5)
        q_dot = m.Dot(at(pl, hq), radius=0.08, color=INK).set_z_index(8)
        target = vec_arrow(pl, hq, heps - hx0, FM_C, ARROW_K, 6).set_z_index(7)
        guess = vec_arrow(pl, Q_PT, V_NETQ, MODEL, ARROW_K, 4).set_z_index(11)
        assert pl.x0 < (hq + ARROW_K * (heps - hx0))[0] < pl.x1 and pl.y0 < (hq + ARROW_K * (heps - hx0))[1] < pl.y1
        gap = m.DashedLine(target.get_end(), guess.get_end(), dash_length=0.06).set_stroke(RED, 3).set_z_index(9)
        lab_target = legend_line(FM_C, "this pair's target: ε − x₀")
        lab_guess = legend_line(MODEL, "the network's answer")
        lab_gap = legend_line(RED, "the error, squared: the loss", dashed=True)
        lab_gap[1].set_color(RED)
        keys = m.VGroup(lab_target, lab_guess, lab_gap).arrange(m.DOWN, buff=0.14, aligned_edge=m.LEFT)
        keys.next_to(frame, m.UP, buff=0.18).align_to(frame, m.LEFT)
        pl_cap = cite(f"t = {Q_T} · arrows at {ARROW_K} × length").next_to(frame, m.DOWN, buff=0.18)

        src = code_block(TRAIN_SRC, size=18, title="one training step, 512 pairs", accent=EDGE)
        src.move_to([-2.55, 1.02, 0])
        assert src.get_left()[0] > -6.6 and src.get_right()[0] < frame.get_left()[0] - 0.2, (src.get_left(), src.get_right(), frame.get_left())
        net_note = cite(f"the swirl's network: {MLP_PARAMS:,} parameters, trained on this Mac:\n"
                        f"{FLOW_TRAIN['steps']:,} steps × {FLOW_TRAIN['batch']} pairs, {FLOW_TRAIN['train_seconds']:.0f} s")
        net_note.next_to(src, m.DOWN, buff=0.2).align_to(src, m.LEFT)
        names = m.VGroup(
            m.VGroup(text("flow matching", 26, FM_C, weight=m.BOLD), cite("Lipman et al., Oct 2022")),
            m.VGroup(text("rectified flow", 26, FM_C, weight=m.BOLD), cite("Liu, Gong & Liu, Sep 2022")))
        for nm in names:
            nm.arrange(m.RIGHT, buff=0.3)
            sit(nm[1], baseline(nm[0]))
        names.arrange(m.DOWN, buff=0.25, aligned_edge=m.LEFT).move_to([0, -2.25, 0]).align_to(src, m.LEFT)
        assert names.get_left()[0] > -6.6 and names.get_right()[0] < frame.get_left()[0] - 0.15, (names.get_left(), names.get_right())

        # v-prediction is not this velocity
        vp_f = formula("v", "=", r"\sqrt{\bar\alpha_t}", r"\,\varepsilon", "-", r"\sqrt{1-\bar\alpha_t}", r"\,x_0", size=40)
        vp_f[2].set_color(TIME)
        vp_f[5].set_color(TIME)
        vp_f[3].set_color(NOISE)
        vp_f[6].set_color(DATA)
        fm_f = formula("v", "=", r"\varepsilon", "-", "x_0", size=40)
        fm_f[0].set_color(FM_C)
        fm_f[2].set_color(NOISE)
        fm_f[4].set_color(DATA)
        vp = m.VGroup(text("Episode 5: v-prediction", 22, INK, weight=m.BOLD), vp_f,
                      cite("weights change with t · Salimans & Ho, 2022")).arrange(m.DOWN, buff=0.14, aligned_edge=m.LEFT)
        fm = m.VGroup(text("flow matching's velocity", 22, FM_C, weight=m.BOLD), fm_f,
                      cite("the same all along a line")).arrange(m.DOWN, buff=0.14, aligned_edge=m.LEFT)
        neq = title_text("≠", 60, RED)
        cmp_row = m.VGroup(vp, neq, fm).arrange(m.DOWN, buff=0.22, aligned_edge=m.LEFT)
        neq.shift(0.6 * m.RIGHT)
        cmp = boxed(cmp_row, RED, 0.3).move_to([-2.55, 0.0, 0])
        assert cmp.get_left()[0] > -6.6 and cmp.get_right()[0] < frame.get_left()[0] - 0.15, (cmp.get_left(), cmp.get_right())

        # many lines cross here: the average
        others = [i for i in range(len(STAR)) if i != HERO]
        star = m.VGroup(*[m.Line(at(pl, STAR_X0[i]), at(pl, STAR_EPS[i])).set_stroke(FM_C, 1.4, opacity=0.35) for i in others]).set_z_index(2)
        star_ends = m.VGroup(*[m.Dot(at(pl, STAR_X0[i]), radius=0.04, color=DATA) for i in others],
                             *[m.Dot(at(pl, STAR_EPS[i]), radius=0.04, color=NOISE) for i in others]).set_z_index(4)
        avg = m.VGroup(vec_arrow(pl, Q_PT, V_EXACT, BG, ARROW_K, 22).set_z_index(9),
                       vec_arrow(pl, Q_PT, V_EXACT, FM_C, ARROW_K, 14).set_z_index(10))
        many_t = m.VGroup(code(f"{N_NEAR}", 44, FM_C, weight=m.BOLD),
                          text(f"of 200,000 training lines pass\nwithin 0.05 of this point at t = {Q_T}", 22, INK))
        many_t.arrange(m.RIGHT, buff=0.3).move_to([0, 1.3, 0]).align_to([-6.3, 0, 0], m.LEFT)
        avg_k = m.VGroup(legend_line(FM_C, "their average: the exact E[ε − x₀ | x_t]"),
                         text("the network's answer lands almost on it", 20, MODEL)).arrange(m.DOWN, buff=0.18, aligned_edge=m.LEFT)
        avg_k.next_to(many_t, m.DOWN, buff=0.45).align_to(many_t, m.LEFT)
        assert avg_k.get_right()[0] < frame.get_left()[0] - 0.1 and many_t.get_right()[0] < frame.get_left()[0] - 0.1, (avg_k.get_right(), many_t.get_right())

        self.say(
            "[Training is episode five's recipe with a new target]: [pick a data point, a noise point and a time].",
            m.AnimationGroup(self.new_header("02 · training", "Learn the velocity"), m.FadeIn(src, shift=0.1 * m.UP),
                             m.FadeIn(frame), m.FadeIn(band)),
            m.AnimationGroup(m.FadeIn(h_ends, scale=0.5), m.Create(h_line), m.FadeIn(q_dot), m.FadeIn(pl_cap)),
        )
        self.say(
            "[The network sees the point on the line], [and learns to predict the velocity], [by squared error].",
            m.AnimationGroup(m.Indicate(q_dot, color=INK, scale_factor=1.6), m.FadeIn(net_note)),
            m.AnimationGroup(m.GrowArrow(target), m.FadeIn(lab_target), m.GrowArrow(guess), m.FadeIn(lab_guess)),
            m.AnimationGroup(m.Create(gap), m.FadeIn(lab_gap)),
        )
        self.say(
            "[In twenty twenty-two, two papers a month apart introduced it], [under two names]: [flow matching], [and rectified flow].",
            m.Wait(0.1),
            m.Wait(0.1),
            m.FadeIn(names[0], shift=0.1 * m.UP),
            m.FadeIn(names[1], shift=0.1 * m.UP),
        )
        self.say(
            "[Careful: this velocity is not episode five's v-prediction], [which mixes noise and picture with weights that change over time].",
            m.AnimationGroup(m.FadeOut(src), m.FadeOut(net_note), m.FadeOut(names), m.FadeIn(cmp[0]), m.FadeIn(vp),
                             m.FadeIn(neq, scale=0.7), m.FadeIn(fm)),
            m.AnimationGroup(m.Circumscribe(vp_f[2], color=TIME, buff=0.06), m.Circumscribe(vp_f[5], color=TIME, buff=0.06)),
        )
        self.say(
            "[One catch: at every point, lines from many pairs cross], [and the network cannot tell them apart].",
            m.AnimationGroup(m.FadeOut(gap), m.FadeOut(lab_gap), m.FadeOut(cmp[0]), m.FadeOut(vp),
                             m.FadeOut(neq), m.FadeOut(fm),
                             m.LaggedStart(*[m.Create(s) for s in star], lag_ratio=0.05, run_time=1.6), m.FadeIn(star_ends)),
            m.FadeIn(many_t, shift=0.1 * m.UP),
        )
        self.say(
            "[So it learns their average]: [one arrow for each point and time].",
            m.AnimationGroup(m.GrowArrow(avg[0]), m.GrowArrow(avg[1]), m.FadeIn(avg_k[0])),
            m.AnimationGroup(m.FadeIn(avg_k[1]), m.Circumscribe(guess, color=MODEL, buff=0.06)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 3 · sampling

    def sampling(self):
        self.next_section("3 · Follow the field", notes=(
            "flow_euler with the trained swirl flow model, 32 equal steps from t = 1 to 0 (x ← x + (t_next − t)·v̂), from "
            "toy.md's shared noise z = gaussian_noise(2000, seed=5) (1,000 shown, those inside the plane). Arrows: the "
            "direction of travel −v̂(x, t) of the trained network, each field scaled to its own longest arrow. Eight traced "
            "paths: the first 8 of z, 200 Euler steps; dashed: the straight line between each path's ends. At t = 1 the "
            "exact velocity is x − (the data's mean) (diffusion_reference self-check), so one exact step lands every point "
            f"on the mean. Length ÷ straight distance, median of all 2,000 paths (200 steps): flow {STRAIGHT_F:.3f} "
            f"(toy.json 1.561), the DDPM model's ODE drawn in its x_t picture (DDIM, 200 steps) {STRAIGHT_D:.3f} (toy 1.137); "
            f"exact fields {EXACT_STRAIGHT[0]} vs {EXACT_STRAIGHT[1]}; in the σ view (x/√ᾱ) the DDPM paths measure "
            f"{SIGMA_VIEW:.2f} — straightness depends on the coordinates. The 7: arrays/digit_trajectory_7.npz, flow Euler 10, "
            "noise np.random.default_rng(7); top x_t (input to each step), bottom the clean guess x̂₀ = x_t − t·v̂."))
        lim, size = 2.6, 5.2
        pl = plane((-3.05, -0.35, 0), size, lim)
        band = swirl_band(pl, 12, 0.2)
        frame = m.Square(size).set_stroke(EDGE, 2).move_to(pl.coords(0, 0))
        xs32 = RUN32.xs[:, SHOW]
        cloud = m.VGroup(*[m.Dot(at(pl, p), radius=0.022, color=NOISE).set_opacity(0.85) for p in xs32[0]]).set_z_index(3)
        fields = {}

        def field_at(k):
            if k not in fields:
                fields[k] = field_arrows(pl, V_NET, float(RUN32.ts[k]), 11, FM_C, 0.34, 0.5).set_z_index(1)
            return fields[k]

        field = field_at(0).copy()
        rx = 0.75
        t_read = Readout(rx, 2.05, "t = {:.2f}", 1.0, 40, TIME)
        k_read = Readout(rx, 1.45, "Euler step {} of 32", 0, 22, MUTED)
        f_eul = formula("x", r"\leftarrow", "x", "+", r"(t_{\mathrm{next}} - t)", r"\,\hat v(x, t)", size=42)
        f_eul[4].set_color(TIME)
        f_eul[5].set_color(FM_C)
        f_eul.move_to([0, 0.55, 0]).align_to([rx, 0, 0], m.LEFT)
        key_a = m.VGroup(m.Arrow(ORIGIN_, [0.5, 0, 0], buff=0, color=FM_C, stroke_width=3, max_tip_length_to_length_ratio=0.35),
                         text("where the network sends each point", 20, FM_C)).arrange(m.RIGHT, buff=0.15)
        key_a.move_to([0, -0.35, 0]).align_to([rx, 0, 0], m.LEFT)
        cap = cite(f"the swirl's flow-matching network · {MLP_PARAMS:,} parameters · 1,000 of 2,000 points").move_to([0, -3.35, 0])
        assert f_eul.get_right()[0] < 6.6 and key_a.get_right()[0] < 6.6

        def run(seconds=3.6):
            state = {"k": -1}

            def step(_, a):
                i = a * 32
                j = min(int(np.floor(i)), 31)
                now = xs32[j] + (xs32[j + 1] - xs32[j]) * (i - j)
                tt = 1 - i / 32
                col = mix(NOISE, DATA, 1 - tt)
                for d, p in zip(cloud, now):
                    d.move_to(at(pl, p))
                    d.set_color(col)
                k = min(int(round(i)), 32)
                if k != state["k"]:
                    state["k"] = k
                    field.become(field_at(min(k, 31)))
                    field.set_opacity(0.12 + 0.43 * (1 - k / 32))
                    k_read.show(k)
                t_read.show(round(tt, 2))
            return m.UpdateFromAlphaFunc(m.VGroup(cloud, field), step, run_time=seconds, rate_func=m.linear)

        # traced paths
        P = FLOW200[:, TRACE]
        lines = m.VGroup(*[m.VMobject().set_stroke(FM_C, 3.2) for _ in TRACE]).set_z_index(5)
        heads = m.VGroup(*[m.Dot(at(pl, q), radius=0.065, color=NOISE).set_z_index(6) for q in P[0]])
        starts = m.VGroup(*[m.Dot(at(pl, q), radius=0.055, color=NOISE).set_z_index(6) for q in P[0]])
        chords = m.VGroup(*[chord(pl, P[0, n], P[-1, n]) for n in range(len(TRACE))]).set_z_index(4)
        key_p = m.VGroup(legend_line(FM_C, "a sampling path, 200 small steps"),
                         legend_line(INK, "the straight line between its ends", dashed=True)).arrange(m.DOWN, buff=0.16, aligned_edge=m.LEFT)
        key_p.move_to([0, -0.35, 0]).align_to([rx, 0, 0], m.LEFT)
        mean_mark = m.VGroup(m.Circle(radius=0.16).set_stroke(INK, 3), m.Dot(radius=0.05, color=INK)).move_to(at(pl, DATA_MEAN)).set_z_index(7)
        bend = pill("they bend", RED, RED_FILL, 22).move_to([rx + 1.2, -1.8, 0])
        f1_note = m.VGroup(text("at t = 1, every arrow points", 22, INK), text("to the data's average", 22, INK)).arrange(m.DOWN, buff=0.06, aligned_edge=m.LEFT)
        f1_note.move_to([0, -2.55, 0]).align_to([rx, 0, 0], m.LEFT)

        self.say(
            "[To sample, start from noise at t equals one], [and step against the learned velocity, down to zero].",
            m.AnimationGroup(self.new_header("03 · sampling", "Follow the field"), m.FadeIn(frame), m.FadeIn(band),
                             m.FadeIn(cloud), m.FadeIn(t_read), m.FadeIn(cap)),
            m.AnimationGroup(m.FadeIn(field), m.Write(f_eul), m.FadeIn(k_read), m.FadeIn(key_a)),
        )
        self.say(
            "[Thirty-two Euler steps carry the cloud onto the swirl].",
            run(3.6),
        )
        self.say(
            "[Now trace eight points]. [The training lines were straight], [but these paths bend].",
            m.AnimationGroup(cloud.animate.set_opacity(0.12), field.animate.set_opacity(0.15), m.FadeOut(key_a),
                             m.FadeIn(starts)),
            m.AnimationGroup(m.FadeIn(key_p), grow_paths(pl, P, lines, heads, 2.6)),
            m.AnimationGroup(m.LaggedStart(*[m.Create(c) for c in chords], lag_ratio=0.1, run_time=1.2),
                             m.FadeIn(bend, scale=0.8)),
        )
        f1 = field_arrows(pl, V_NET, 1.0, 11, FM_C, 0.34, 0.5).set_z_index(1)
        self.say(
            "[At first, the average pulls every point toward the middle of the data]; [only later does it turn toward an arm].",
            m.AnimationGroup(m.Transform(field, f1), m.FadeIn(mean_mark, scale=0.6), m.FadeIn(f1_note),
                             m.Transform(t_read.words, t_read._make(1.0)), m.Transform(k_read.words, k_read._make(0))),
            m.LaggedStart(*[m.Indicate(h, color=DATA, scale_factor=1.8) for h in heads], lag_ratio=0.1),
        )
        self.wait(0.3)
        screen = m.VGroup(frame, band, cloud, field, t_read, k_read, f_eul, cap, lines, heads, starts, chords, key_p,
                          mean_mark, bend, f1_note)

        # the digit 7, same story
        hh, pitch = 0.98, 1.12
        x_first = -5.05
        tops, bots, tl = m.Group(), m.Group(), m.VGroup()
        for i in range(10):
            x = x_first + i * pitch
            tops.add(framed(picture(state_rgb(F10_X[i], F10_T[i]), hh), stroke=2).move_to([x, 0.95, 0]))
            bots.add(framed(picture(guess_rgb(F10_G[i]), hh), MODEL, stroke=2).move_to([x, -0.75, 0]))
            tl.add(code(f"{F10_T[i]:.1f}", 20, TIME).move_to([x, 1.75, 0]))
        assert tops.get_right()[0] < 6.6 and tops.get_left()[0] > -6.6
        t_lab = code("t =", 20, TIME).next_to(tl[0], m.LEFT, buff=0.2)
        top_l = text("the input at each step, x_t", 20, MUTED).next_to(tops, m.DOWN, buff=0.1).align_to(tops, m.LEFT)
        f_g = formula(r"\hat x_0", "=", "x_t", "-", "t", r"\,\hat v", size=34)
        f_g[0].set_color(MODEL)
        f_g[4].set_color(TIME)
        f_g[5].set_color(FM_C)
        bot_l = m.VGroup(text("the network's clean guess", 22, MODEL, weight=m.BOLD), f_g).arrange(m.RIGHT, buff=0.3)
        bot_l.next_to(bots, m.DOWN, buff=0.22).align_to(bots, m.LEFT)
        blur = m.SurroundingRectangle(bots[0], buff=0.1).set_stroke(RED, 3)
        blur_l = text("a blurry average seven", 22, RED, weight=m.BOLD).next_to(blur, m.DOWN, buff=0.12).align_to(blur, m.LEFT)
        bot_l.next_to(blur_l, m.DOWN, buff=0.2).align_to(bots, m.LEFT)
        cap7 = cite("generated on this Mac · the digit model trained with flow matching · 10 Euler steps · label 7").move_to([0, -3.35, 0])

        self.say(
            "[Digits do the same]: [the first clean guess is a blurry average seven], [which sharpens as t falls].",
            m.AnimationGroup(m.FadeOut(screen), m.FadeIn(tl), m.FadeIn(t_lab), m.FadeIn(cap7),
                             m.LaggedStart(*[m.FadeIn(f, shift=0.05 * m.DOWN) for f in tops], lag_ratio=0.08, run_time=1.4),
                             m.FadeIn(top_l)),
            m.AnimationGroup(m.FadeIn(bots[0]), m.Create(blur), m.FadeIn(blur_l), m.FadeIn(bot_l)),
            m.LaggedStart(*[m.FadeIn(f, shift=0.05 * m.RIGHT) for f in bots[1:]], lag_ratio=0.15, run_time=1.8),
        )
        self.wait(0.3)
        seven = m.Group(tops, bots, tl, t_lab, top_l, bot_l, blur, blur_l, cap7)

        # straightness, both models
        lim2, size2 = 2.0, 4.0
        pa = plane((-4.45, -0.35, 0), size2, lim2)
        pb = plane((-0.15, -0.35, 0), size2, lim2)
        frames = m.VGroup(*[m.Square(size2).set_stroke(EDGE, 2).move_to(p.coords(0, 0)) for p in (pa, pb)])
        bands = m.VGroup(swirl_band(pa, 9, 0.2), swirl_band(pb, 9, 0.2))
        Pd = DDIM200[:, TRACE]
        la = m.VGroup(*[path_line(pa, P[:, n], FM_C, 3) for n in range(len(TRACE))]).set_z_index(3)
        lb = m.VGroup(*[path_line(pb, Pd[:, n], DF_C, 3) for n in range(len(TRACE))]).set_z_index(3)
        ca = m.VGroup(*[chord(pa, P[0, n], P[-1, n]) for n in range(len(TRACE))]).set_z_index(2)
        cb = m.VGroup(*[chord(pb, Pd[0, n], Pd[-1, n]) for n in range(len(TRACE))]).set_z_index(2)
        ea = m.VGroup(*[m.Dot(at(pa, q), radius=0.05, color=NOISE) for q in P[0]]).set_z_index(4)
        eb = m.VGroup(*[m.Dot(at(pb, q), radius=0.05, color=NOISE) for q in Pd[0]]).set_z_index(4)
        ta = text("flow matching", 24, FM_C, weight=m.BOLD).next_to(frames[0], m.UP, buff=0.15)
        tb = text("the diffusion model (DDIM)", 24, DF_C, weight=m.BOLD).next_to(frames[1], m.UP, buff=0.15)
        sa = m.VGroup(title_text(f"{STRAIGHT_F:.2f}", 56, FM_C), text("flow matching", 20, FM_C))
        sb = m.VGroup(title_text(f"{STRAIGHT_D:.2f}", 56, DF_C), text("diffusion", 20, DF_C))
        for g in (sa, sb):
            g.arrange(m.DOWN, buff=0.08)
        sg = m.VGroup(sa, sb).arrange(m.RIGHT, buff=0.55).move_to([4.6, 0.35, 0])
        sg_t = text("length ÷ straight distance,\nmedian of 2,000 paths", 20, INK).next_to(sg, m.UP, buff=0.3)
        own = cite("each in its own coordinates;\nin Episode 10's σ view the\ndiffusion paths measure "
                   f"{SIGMA_VIEW:.2f}").next_to(sg, m.DOWN, buff=0.35)
        assert own.get_right()[0] < 6.6 and sg_t.get_right()[0] < 6.6 and frames[1].get_right()[0] < own.get_left()[0] - 0.1, (own.get_left(), own.get_right(), sg_t.get_right())
        ex_note = cite(f"exact fields: {EXACT_STRAIGHT[0]:.2f} vs {EXACT_STRAIGHT[1]:.2f} · the same 8 starting points").move_to([-2.3, -2.75, 0])

        self.say(
            "[Length over straight distance: one point five six for these paths], [and one point one four for the diffusion model's], "
            "[each in its own coordinates].",
            m.AnimationGroup(m.FadeOut(seven), m.FadeIn(frames[0]), m.FadeIn(bands[0]), m.FadeIn(ta), m.FadeIn(ea),
                             m.FadeIn(la), m.FadeIn(ca), m.FadeIn(sg_t), m.FadeIn(sa, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(frames[1]), m.FadeIn(bands[1]), m.FadeIn(tb), m.FadeIn(eb),
                             m.LaggedStart(*[m.Create(l_) for l_ in lb], lag_ratio=0.1, run_time=1.4), m.FadeIn(cb),
                             m.FadeIn(sb, shift=0.1 * m.UP), m.FadeIn(ex_note)),
            m.FadeIn(own, shift=0.1 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 4 · the race

    def the_race(self):
        self.next_section("4 · Euler steps on both", notes=(
            "toy.md §4, recomputed here and asserted against toy.json flow/trained/quality_vs_steps: the trained flow "
            "model with flow_euler (S equal steps in t from 1 to 0) vs the DDPM-trained model with DDIM (η = 0, S steps "
            "evenly spaced in DDPM time, 'trailing'; DDIM = Euler in σ, Episode 9), from the same 2,000 noise points "
            "(seed 5). Stranded = off_swirl, farther than 0.1 from the swirl's centre line (the data: 0.1%). Shown: 1,000 "
            "starting points inside the planes; percentages over all 2,000. "
            + "; ".join(f"{s} steps: {PCT(OFF_F[s])} vs {PCT(OFF_D[s])}" for s in (2, 4, 8, 16, 32))
            + ". One step: flow 96.8%, DDIM 95.8% (both send points to the middle)."))
        lim, size = 2.0, 4.0
        pa = plane((-4.5, -0.6, 0), size, lim)
        pb = plane((-0.3, -0.6, 0), size, lim)
        frames = m.VGroup(*[m.Square(size).set_stroke(EDGE, 2).move_to(p.coords(0, 0)) for p in (pa, pb)])
        bands = m.VGroup(swirl_band(pa, 9, 0.2), swirl_band(pb, 9, 0.2))
        z0 = Z[SHOW_RACE]
        sw_a, sw_b = Swarm(pa, z0), Swarm(pb, z0)
        ta = text("flow matching · Euler", 24, FM_C, weight=m.BOLD).next_to(frames[0], m.UP, buff=0.55).align_to(frames[0], m.LEFT)
        tb = text("diffusion model · DDIM", 24, DF_C, weight=m.BOLD).next_to(frames[1], m.UP, buff=0.55).align_to(frames[1], m.LEFT)
        ra = Readout(frames[0].get_left()[0], frames[0].get_top()[1] + 0.24, "{}", "starting noise", 22, MUTED)
        rb = Readout(frames[1].get_left()[0], frames[1].get_top()[1] + 0.24, "{}", "starting noise", 22, MUTED)
        legend = m.VGroup(m.VGroup(m.Dot(radius=0.07, color=NOISE), cite("noise")).arrange(m.RIGHT, buff=0.12),
                          m.VGroup(m.Dot(radius=0.07, color=DATA), cite("on the swirl")).arrange(m.RIGHT, buff=0.12),
                          m.VGroup(m.Dot(radius=0.07, color=RED), cite("stranded")).arrange(m.RIGHT, buff=0.12))
        legend.arrange(m.RIGHT, buff=0.45).move_to([-2.4, -2.92, 0])
        same = cite("the same 2,000 starting points · the two trained swirl networks · 1,000 shown").move_to([-2.4, -3.35, 0])

        # the table
        cx = (3.1, 4.5, 5.85)
        head = m.VGroup(code("steps", 20, MUTED), code("flow", 22, FM_C, weight=m.BOLD), code("diffusion", 22, DF_C, weight=m.BOLD))
        for w, x in zip(head, cx):
            w.move_to([x, 1.75, 0])
        for w in head[1:]:
            sit(w, baseline(head[0]))
        t_title = text("stranded: more than 0.1 from the swirl", 20, INK).move_to([4.45, 2.35, 0])
        rule = m.Line([2.45, 1.45, 0], [6.5, 1.45, 0]).set_stroke(EDGE, 2)
        ys = {2: 1.0, 4: 0.45, 8: -0.1, 16: -0.65, 32: -1.2}
        trow = {}
        for s, y in ys.items():
            win_f, win_d = OFF_F[s] < OFF_D[s] - 0.05, OFF_D[s] < OFF_F[s] - 0.05
            row = m.VGroup(code(f"{s}", 24, INK, weight=m.BOLD),
                           code(PCT(OFF_F[s]), 24, GOOD if win_f else INK, weight=m.BOLD if win_f else m.NORMAL),
                           code(PCT(OFF_D[s]), 24, GOOD if win_d else INK, weight=m.BOLD if win_d else m.NORMAL))
            for w, x in zip(row, cx):
                w.move_to([x, y, 0])
            for w in row[1:]:
                sit(w, baseline(row[0]))
            trow[s] = row
        poor = m.SurroundingRectangle(m.VGroup(trow[2], trow[4]), buff=0.12, corner_radius=0.08).set_stroke(RED, 2.5)
        poor_l = text("2 and 4 steps: both poor", 20, RED, weight=m.BOLD).move_to([4.45, -2.35, 0])
        data_l = cite("over all 2,000 points · the data: 0.1%").move_to([4.3, -1.85, 0])
        assert data_l.get_right()[0] < 6.6 and frames[1].get_right()[0] < 2.3 and poor_l.get_left()[0] > frames[1].get_right()[0] + 0.05, (data_l.get_right(), poor_l.get_left())

        def to(s, run_time=1.2):
            return m.AnimationGroup(sw_a.go(flow_euler(V_NET, z0, s).xs[-1], run_time), sw_b.go(ddim_sample(E_NET, z0, s).xs[-1], run_time),
                                    m.Transform(ra.words, ra._make(f"{s} steps")), m.Transform(rb.words, rb._make(f"{s} steps")))

        self.say(
            "[Yet flow matching needs fewer steps]. [Race the two trained networks on the swirl], [from the same noise].",
            m.AnimationGroup(self.new_header("04 · the race", "Euler steps on both"), m.FadeIn(t_title)),
            m.AnimationGroup(m.FadeIn(frames), m.FadeIn(bands), m.FadeIn(ta), m.FadeIn(tb), m.FadeIn(ra), m.FadeIn(rb)),
            m.AnimationGroup(m.FadeIn(sw_a), m.FadeIn(sw_b), m.FadeIn(legend), m.FadeIn(same), m.FadeIn(head), m.FadeIn(rule)),
        )
        self.say(
            "[With eight Euler steps, a quarter of the flow model's points are stranded], [against nearly half for the diffusion model].",
            m.AnimationGroup(to(8, 1.6), m.FadeIn(trow[8][:2])),
            m.FadeIn(trow[8][2], shift=0.1 * m.LEFT),
        )
        self.say(
            "[With sixteen: under seven percent, against over a third]. [With thirty-two: about two against sixteen].",
            m.AnimationGroup(to(16), m.FadeIn(trow[16])),
            m.AnimationGroup(to(32), m.FadeIn(trow[32]), m.FadeIn(data_l)),
        )
        self.say(
            "[To be fair: at two and four steps both are poor], [and at four, the diffusion model is ahead].",
            m.AnimationGroup(to(2), m.FadeIn(trow[2]), m.FadeIn(trow[4]), m.Create(poor), m.FadeIn(poor_l)),
            m.AnimationGroup(to(4), m.Circumscribe(trow[4][2], color=GOOD, buff=0.08)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 5 · the twist: placement

    def the_twist(self):
        self.next_section("5 · Same steps, placed differently", notes=(
            "With exact models, flow Euler IS DDIM: x_t^flow/(1 − t) = x₀ + σ·ε with σ = t/(1 − t), the same σ view as "
            "Episodes 9–10, so flow_euler(exact_velocity, z, 16) and ddim_sample(exact_eps, z, timestep_from_sigma(σ grid)) "
            f"agree to {IDENTITY:.4f} on 2,000 points (toy.json exact_flow16_vs_ddim_on_sigma_grid_max_diff; the first flow "
            "level σ = ∞ is clipped to DDPM's t = 1000, σ = 157). Dots: the noise level σ at which each of 16 steps starts: "
            "flow (t = 1, 15/16, …, 1/16 → σ = ∞, 15, 7, …, 0.067) and DDIM (t = 1000, 938, …, 62 under DDPM's linear β → "
            f"σ = 157, 86, …, 0.21). Below σ = 1 (signal beats noise, SNR > 1): flow {LOW_F} of 16, DDIM {LOW_D} of 16. "
            "toy.md §4: 'With perfect models the per-step advantage comes entirely from where the steps fall … the trained "
            "models track the exact results closely.' Episode 10: placement (Karras et al. 2022)."))
        x_l, x_r = -4.75, 5.55
        s_hi, s_lo = 200.0, 0.05
        xpos = lambda s: x_l + (np.log10(s_hi) - np.log10(s)) / (np.log10(s_hi) - np.log10(s_lo)) * (x_r - x_l)  # noqa: E731
        y_ax, y_f, y_d = -0.1, 0.55, -0.75
        axis = m.Line([x_l, y_ax, 0], [x_r, y_ax, 0]).set_stroke(EDGE, 3)
        inf_x = -5.85
        brk = m.DashedLine([inf_x + 0.15, y_ax, 0], [x_l, y_ax, 0], dash_length=0.07).set_stroke(EDGE, 3)
        ticks = m.VGroup()
        for s in (100, 10, 1, 0.1):
            ticks.add(m.Line([xpos(s), y_ax - 0.1, 0], [xpos(s), y_ax + 0.1, 0]).set_stroke(FAINT, 2))
            ticks.add(code(f"{s:g}", 20, MUTED).move_to([xpos(s), y_ax - 0.32, 0]))
        ticks.add(code("∞", 30, MUTED).move_to([inf_x, y_ax - 0.34, 0]))
        sig_l = text("noise level σ (log scale)", 20, MUTED).move_to([0.4, -1.75, 0])
        ends_l = m.VGroup(text("pure noise", 20, NOISE, weight=m.BOLD).move_to([inf_x + 0.35, 2.05, 0]),
                          text("data", 20, DATA, weight=m.BOLD).move_to([x_r - 0.1, 2.05, 0]))
        go = arrow_to([inf_x + 1.2, 2.05, 0], [x_r - 0.6, 2.05, 0], MUTED, 3, 0.1)
        go_l = cite("sampling runs this way").next_to(go, m.UP, buff=0.08)
        band = m.Rectangle(width=x_r + 0.25 - xpos(1), height=2.25).set_stroke(width=0).set_fill(GREEN, 0.07)
        band.move_to([(xpos(1) + x_r + 0.25) / 2, -0.1, 0])
        band_edge = m.DashedLine([xpos(1), y_ax + 0.18, 0], [xpos(1), 1.0, 0], dash_length=0.08).set_stroke(GREEN, 2)
        band_l = text("signal beats noise", 20, GREEN, weight=m.BOLD).move_to([(xpos(1) + x_r) / 2, 1.3, 0])
        fdots = m.VGroup(*[m.Dot([inf_x if not np.isfinite(s) else xpos(s), y_f, 0], radius=0.085, color=FM_C) for s in SIG_FLOW])
        ddots = m.VGroup(*[m.Dot([xpos(s), y_d, 0], radius=0.085, color=DF_C) for s in SIG_DDIM])
        fl = text("flow matching: 16 even steps in t", 22, FM_C, weight=m.BOLD).move_to([0, 1.0, 0]).align_to([inf_x - 0.1, 0, 0], m.LEFT)
        dl = text("diffusion: 16 even steps in DDPM time", 22, DF_C, weight=m.BOLD).move_to([0, -1.2, 0]).align_to([inf_x - 0.1, 0, 0], m.LEFT)
        cf = code(f"{LOW_F} of 16", 24, FM_C, weight=m.BOLD).move_to([(xpos(1) + x_r) / 2, 0.95, 0])
        cd = code(f"{LOW_D} of 16", 24, DF_C, weight=m.BOLD).move_to([(xpos(1) + x_r) / 2, -1.25, 0])
        band_l.move_to([(xpos(1) + x_r) / 2, 1.42, 0])
        assert fl.get_right()[0] < xpos(1) - 0.2 and dl.get_right()[0] < xpos(1) - 0.2, (fl.get_right(), dl.get_right(), xpos(1))
        same = m.VGroup(text("with perfect models: flow Euler = DDIM on the levels σ = t / (1 − t)", 22, INK),
                        cite(f"checked on the swirl, exact velocity, 16 steps, 2,000 points: they differ by at most {IDENTITY:.4f}"))
        same.arrange(m.DOWN, buff=0.1).move_to([0, -2.45, 0])
        lesson = pill("Episode 10: placement matters", TIME, BLUE_FILL, 22).move_to([0, -3.22, 0])
        assert same.get_right()[0] < 6.6 and same.get_left()[0] > -6.6, (same.get_left(), same.get_right())

        self.say(
            "[Here is the twist]: [with perfect models, these flow steps are exactly DDIM steps], [placed at other noise levels].",
            m.AnimationGroup(self.new_header("05 · the twist", "Same steps, placed differently"), m.FadeIn(axis), m.FadeIn(brk),
                             m.FadeIn(ticks), m.FadeIn(sig_l), m.FadeIn(ends_l), m.GrowArrow(go), m.FadeIn(go_l)),
            m.FadeIn(same, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(fl), m.LaggedStart(*[m.FadeIn(d, scale=0.4) for d in fdots], lag_ratio=0.08, run_time=1.4),
                             m.FadeIn(dl), m.LaggedStart(*[m.FadeIn(d, scale=0.4) for d in ddots], lag_ratio=0.08, run_time=1.4)),
        )
        self.say(
            "[Even steps in flow time put seven of sixteen where signal beats noise]; [even steps in the diffusion model's time, only four].",
            m.AnimationGroup(m.FadeIn(band), m.Create(band_edge), m.FadeIn(band_l), m.FadeIn(cf),
                             m.LaggedStart(*[m.Indicate(d, color=INK, scale_factor=1.6) for d in fdots[-LOW_F:]], lag_ratio=0.1)),
            m.AnimationGroup(m.FadeIn(cd), m.LaggedStart(*[m.Indicate(d, color=INK, scale_factor=1.6) for d in ddots[-LOW_D:]], lag_ratio=0.1)),
        )
        self.say(
            "[That is episode ten's lesson: placement matters]. [On the swirl, it is the whole advantage, not straighter paths].",
            m.FadeIn(lesson, shift=0.1 * m.UP),
            m.Circumscribe(m.VGroup(cf, cd), color=GREEN, buff=0.12),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 6 · the digit model, measured

    def digits(self):
        self.next_section("6 · Measured on this Mac", notes=(
            "mac_runs.md §3–4: the digit model's network (3,940,801 parameters) trained twice, same recipe (10,000 steps × "
            f"256, Adam, EMA 0.999): DDPM ε-loss {MIN_DDPM} min, rectified flow (target ε − x₀, t ~ U(0, 1)) {MIN_FLOW} min. "
            "Sampled 10,000 digits per configuration, the same starting noise per sample (numpy seed 1234), w = 1; DDIM "
            "(η = 0) for the DDPM model, flow Euler for the flow model. Shown: samples 10–19 (labels 0–9). cFID = FID's "
            "formula on the 128-d features of our own MNIST classifier (99.45%), 10,000 generated vs 10,000 test digits — "
            f"NOT Inception FID; floor {FLOOR:.3f}; bootstrap std ≈ 0.3. Median pixel correlation of the 10-step digits "
            f"between the two networks: {SAME_DIGITS:.3f}. 1-step flow: 'a smooth, blurry average digit' (diversity 0.08); "
            "1–2-step DDIM: speckles (x̂₀ divides by √ᾱ_1000 = 0.0064, Episode 9)."))
        steps = (10, 5, 2, 1)
        ys = {10: 1.12, 5: 0.34, 2: -0.44, 1: -1.22}
        dh = 0.46
        gx_d, gx_f = -3.55, 1.05
        num_d, num_f = 4.25, 5.65
        head_y, sub_y = 2.2, 1.78
        rows = {}
        for s in steps:
            y = ys[s]
            rd = digit_row(DIGITS[f"ddpm_ddim{s}"][:N_GRID], DATA, dh, gap_px=5).move_to([gx_d, y, 0])
            rf = digit_row(DIGITS[f"flow_euler{s}"][:N_GRID], DATA, dh, gap_px=5).move_to([gx_f, y, 0])
            nd = code(CF[("ddim", s)], 24, INK).move_to([num_d, y, 0])
            nf = code(CF[("flow", s)], 24, INK).move_to([num_f, y, 0])
            sl = code(f"{s}", 24, INK, weight=m.BOLD).move_to([-6.2, y, 0])
            sit(nf, baseline(nd))
            sit(sl, baseline(nd))
            rows[s] = dict(d=rd, f=rf, nd=nd, nf=nf, sl=sl)
        r10 = rows[10]
        assert r10["d"].get_left()[0] > r10["sl"].get_right()[0] + 0.2 and r10["d"].get_right()[0] < r10["f"].get_left()[0] - 0.3, (r10["d"].get_left(), r10["d"].get_right(), r10["f"].get_left())
        assert r10["f"].get_right()[0] < num_d - 0.6 and num_f + 0.45 < 6.6, (r10["f"].get_right(),)
        hd = m.VGroup(text("diffusion model · DDIM", 22, DF_C, weight=m.BOLD).move_to([gx_d, head_y, 0]),
                      text("flow matching · Euler", 22, FM_C, weight=m.BOLD).move_to([gx_f, head_y, 0]),
                      code("steps", 20, MUTED).move_to([-6.2, sub_y, 0]),
                      text("cFID, lower is better", 20, INK).move_to([(num_d + num_f) / 2, head_y, 0]))
        sit(hd[1], baseline(hd[0]))
        sit(hd[3], baseline(hd[0]))
        sub = m.VGroup(cite(f"trained {MIN_DDPM} min").move_to([gx_d, sub_y, 0]),
                       cite(f"trained {MIN_FLOW} min").move_to([gx_f, sub_y, 0]),
                       code("diffusion", 20, DF_C).move_to([num_d, sub_y, 0]), code("flow", 20, FM_C).move_to([num_f, sub_y, 0]))
        for w in list(sub[1:]) + [hd[2]]:
            sit(w, baseline(sub[0]))
        rule = m.Line([-6.5, sub_y - 0.3, 0], [6.5, sub_y - 0.3, 0]).set_stroke(EDGE, 2)
        assert hd[3].get_right()[0] < 6.6 and hd[2].get_left()[0] > -6.6, (hd[3].get_right(), hd[2].get_left())
        cap = cite(f"generated on this Mac · the digit model, trained twice · labels 0–{N_GRID - 1} · same noise per column (seed 1234)")
        cap.move_to([0, -3.35, 0])
        not_inc = cite(f"cFID: FID's formula on our own digit classifier's features, not Inception FID · real digits {FLOOR:.2f}")
        not_inc.move_to([0, -2.92, 0])
        assert cap.get_left()[0] > -6.6 and not_inc.get_left()[0] > -6.6, (cap.get_left(), not_inc.get_left())
        tie_d = m.SurroundingRectangle(rows[10]["nd"], buff=0.1, corner_radius=0.08).set_stroke(MUTED, 2.5)
        tie_f = m.SurroundingRectangle(rows[10]["nf"], buff=0.1, corner_radius=0.08).set_stroke(MUTED, 2.5)
        win = {s: m.SurroundingRectangle(rows[s]["nf"], buff=0.1, corner_radius=0.08).set_stroke(GOOD, 2.5) for s in (5, 2)}
        blurry = pill("blurry average digits", MODEL, AMBER_FILL, 20).move_to([gx_f, -1.82, 0])
        speck = pill("speckles", RED, RED_FILL, 20).move_to([gx_d, -1.82, 0])
        verdict = text("flow matching wins at very few steps · from 10 steps up, they tie", 24, INK, weight=m.BOLD).move_to([0, -2.42, 0])
        same_d = m.VGroup(m.SurroundingRectangle(rows[10]["d"], buff=0.06).set_stroke(TEAL, 2.5),
                          m.SurroundingRectangle(rows[10]["f"], buff=0.06).set_stroke(TEAL, 2.5))

        def row_in(s, nums=True):
            r = rows[s]
            parts = [m.FadeIn(r["sl"]), m.FadeIn(r["d"]), m.FadeIn(r["f"])]
            if nums:
                parts += [m.FadeIn(r["nd"]), m.FadeIn(r["nf"])]
            return m.AnimationGroup(*parts)

        self.say(
            "[Back to the digit model]: [the same three point nine million parameters, trained for twenty-nine minutes with flow matching].",
            m.AnimationGroup(self.new_header("06 · the digits", "Measured on this Mac"), m.FadeIn(hd[0]), m.FadeIn(hd[2]),
                             m.FadeIn(hd[3]), m.FadeIn(sub[0]), m.FadeIn(sub[2]), m.FadeIn(rule), m.FadeIn(cap)),
            m.AnimationGroup(m.FadeIn(hd[1]), m.FadeIn(sub[1]), m.FadeIn(sub[3])),
        )
        self.say(
            "[With ten steps, both score four point five seven], [on our classifier-based FID, not the Inception one].",
            m.AnimationGroup(row_in(10), m.Create(tie_d), m.Create(tie_f)),
            m.FadeIn(not_inc, shift=0.1 * m.UP),
        )
        self.say(
            "[And from the same noise, the two networks draw the same digits].",
            m.AnimationGroup(m.Create(same_d[0]), m.Create(same_d[1])),
        )
        self.say(
            "[With five steps: about nine point three for flow matching], [twenty-one point six for diffusion].",
            m.AnimationGroup(m.FadeOut(same_d), row_in(5, False), m.FadeIn(rows[5]["nf"]), m.Create(win[5])),
            m.FadeIn(rows[5]["nd"]),
        )
        self.say(
            "[With two: eighty-five against nine hundred thirty-nine].",
            m.AnimationGroup(row_in(2), m.Create(win[2])),
        )
        self.say(
            "[One step gives flow matching a blurry average digit], [and diffusion only speckles]: "
            "[both score terribly, eight hundred eighty-six and nine hundred forty-three].",
            m.AnimationGroup(m.FadeIn(rows[1]["sl"]), m.FadeIn(rows[1]["f"]), m.FadeIn(blurry, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(rows[1]["d"]), m.FadeIn(speck, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(rows[1]["nf"]), m.FadeIn(rows[1]["nd"])),
        )
        self.say(
            "[So flow matching wins at very few steps], [and from ten steps up, the two tie].",
            m.AnimationGroup(m.FadeIn(verdict, shift=0.1 * m.UP), m.Circumscribe(m.VGroup(rows[5]["nf"], rows[2]["nf"]), color=GOOD, buff=0.1)),
            m.Circumscribe(m.VGroup(rows[10]["nd"], rows[10]["nf"]), color=MUTED, buff=0.12),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 7 · reflow

    def reflow(self):
        self.next_section("7 · Straightening the paths", notes=(
            "Reflow (Liu, Gong & Liu 2022, 'rectified flow'): (Z₀, Z₁) = (noise, the trained flow's ODE output from it), "
            "then retrain on those pairs ('2-rectified flow'). Left: 30 random pairs as flow matching trains on them "
            "(x₀ = swirl(30, seed=3), ε = gaussian_noise(30, seed=4)); right: the same 30 noise points joined to where our "
            f"trained swirl flow carries them (200 Euler steps). Crossings among the 435 pairs of lines: {len(CROSS_RAND)} vs "
            f"{len(CROSS_REFLOW)} (red dots: computed crossing points); over 300 lines {100 * SHARE_RAND:.1f}% vs "
            f"{100 * SHARE_REFLOW:.2f}% of line pairs cross. We did NOT retrain (no reflow model exists here). The paper's "
            "Table 1 (CIFAR-10, one Euler step, FID): 1-rectified flow 378; 2-rectified flow 12.21; 2-rectified + "
            "distillation 4.85 (Episode 12); with a full adaptive ODE solve the FID moves 2.58 → 3.36 (read from the arXiv "
            "HTML, Oct 2026)."))
        lim, size = 2.4, 4.0
        pa = plane((-4.55, -0.55, 0), size, lim)
        pb = plane((-0.35, -0.55, 0), size, lim)
        frames = m.VGroup(*[m.Square(size).set_stroke(EDGE, 2).move_to(p.coords(0, 0)) for p in (pa, pb)])
        bands = m.VGroup(swirl_band(pa, 8, 0.2), swirl_band(pb, 8, 0.2))
        la = m.VGroup(*[m.Line(at(pa, a), at(pa, b)).set_stroke(FM_C, 1.8, opacity=0.7) for a, b in zip(RF_X0, RF_Z)]).set_z_index(2)
        lb = m.VGroup(*[m.Line(at(pb, a), at(pb, b)).set_stroke(FM_C, 1.8, opacity=0.8) for a, b in zip(RF_END, RF_Z)]).set_z_index(2)
        da = m.VGroup(*[m.Dot(at(pa, p), radius=0.045, color=DATA) for p in RF_X0], *[m.Dot(at(pa, p), radius=0.045, color=NOISE) for p in RF_Z]).set_z_index(4)
        db = m.VGroup(*[m.Dot(at(pb, p), radius=0.045, color=DATA) for p in RF_END], *[m.Dot(at(pb, p), radius=0.045, color=NOISE) for p in RF_Z]).set_z_index(4)
        xa = m.VGroup(*[m.Dot(at(pa, p), radius=0.04, color=RED) for p in CROSS_RAND]).set_z_index(5)
        ta = m.VGroup(text("random pairs", 24, FM_C, weight=m.BOLD), cite("what flow matching trains on")).arrange(m.DOWN, buff=0.06)
        tb = m.VGroup(text("reflow's pairs", 24, FM_C, weight=m.BOLD), cite("noise → where our flow takes it")).arrange(m.DOWN, buff=0.06)
        ta.next_to(frames[0], m.UP, buff=0.12)
        tb.next_to(frames[1], m.UP, buff=0.12)
        rx = 4.55
        ca = m.VGroup(title_text(f"{len(CROSS_RAND)}", 56, RED), text("crossings", 20, RED))
        cb = m.VGroup(title_text(f"{len(CROSS_REFLOW)}", 56, GOOD), text("crossings", 20, GOOD))
        for g in (ca, cb):
            g.arrange(m.DOWN, buff=0.06)
        cg = m.VGroup(ca, cb).arrange(m.RIGHT, buff=0.8).move_to([rx, 1.55, 0])
        c_note = cite(f"among the {N_LINE_PAIRS} pairs of these 30 lines").next_to(cg, m.DOWN, buff=0.15)
        notrun = pill("not run here: no retraining", MUTED, CARD, 20).move_to([rx, 0.35, 0])
        rows_t = m.VGroup()
        for i, (lab, val) in enumerate(RF_TABLE):
            row = m.VGroup(text(lab, 20, INK), code(val, 24, GOOD if i else RED, weight=m.BOLD))
            row[0].move_to([0, -1.15 - 0.48 * i, 0]).align_to([2.7, 0, 0], m.LEFT)
            row[1].move_to([6.2, row[0].get_y(), 0], aligned_edge=m.RIGHT)
            sit(row[1], baseline(row[0]))
            rows_t.add(row)
        t_head = text("the paper: small photos, FID", 20, INK, weight=m.BOLD).move_to([0, -0.62, 0]).align_to([2.7, 0, 0], m.LEFT)
        t_src = cite("Liu et al., 2022 · Table 1 · CIFAR-10").move_to([4.4, -2.65, 0])
        ep12 = cite("distillation: Episode 12").move_to([4.4, -3.0, 0])
        tbox = card(4.36, 3.05, EDGE, CARD, radius=0.16, stroke=2).move_to([4.4, -1.8, 0])
        assert tbox.get_right()[0] < 6.6 and tbox.get_left()[0] > frames[1].get_right()[0] + 0.1, (tbox.get_left(), tbox.get_right(), frames[1].get_right())
        assert all(r[1].get_left()[0] > r[0].get_right()[0] + 0.15 for r in rows_t), [(r[0].get_right(), r[1].get_left()) for r in rows_t]
        assert cg.get_right()[0] < 6.6 and notrun.get_right()[0] < 6.6, (cg.get_right(), notrun.get_right())

        self.say(
            "[To straighten the sampling paths themselves], [rectified flow adds a step called reflow].",
            m.AnimationGroup(self.new_header("07 · reflow", "Straightening the paths"), m.FadeIn(frames[0]), m.FadeIn(bands[0]),
                             m.FadeIn(ta), m.FadeIn(da), m.FadeIn(la)),
            m.AnimationGroup(m.FadeIn(frames[1]), m.FadeIn(bands[1]), m.FadeIn(tb)),
        )
        self.say(
            "[Pair each noise point with the point our trained flow carries it to].",
            m.AnimationGroup(m.FadeIn(db), m.LaggedStart(*[m.Create(l_) for l_ in lb], lag_ratio=0.04, run_time=1.6)),
        )
        self.say(
            "[Random pairs cross ninety-five times among these thirty lines]; [the new pairs, not once].",
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(d, scale=0.3) for d in xa], lag_ratio=0.02, run_time=1.4),
                             m.FadeIn(ca, shift=0.1 * m.UP), m.FadeIn(c_note)),
            m.FadeIn(cb, shift=0.1 * m.UP),
        )
        self.say(
            "[Retrain on such pairs, and the paths come out nearly straight]. [We did not run that step here].",
            m.Circumscribe(frames[1], color=FM_C, buff=0.06),
            m.FadeIn(notrun, scale=0.9),
        )
        self.say(
            "[In the paper, one Euler step on small photos scored an FID of three hundred seventy-eight]; [after one reflow, twelve].",
            m.AnimationGroup(m.FadeIn(tbox), m.FadeIn(t_head), m.FadeIn(rows_t[0]), m.FadeIn(t_src)),
            m.AnimationGroup(m.FadeIn(rows_t[1], shift=0.1 * m.UP), m.FadeIn(rows_t[2], shift=0.1 * m.UP), m.FadeIn(ep12)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 8 · in the wild

    def in_the_wild(self):
        self.next_section("8 · Who uses it", notes=(
            "Dated October 2026 (notes §11). Stable Diffusion 3: Esser et al. (Stability AI), 'Scaling Rectified Flow "
            "Transformers for High-Resolution Image Synthesis', arXiv:2403.03206, 5 Mar 2024, ICML 2024 — 'we scale our "
            "models up to 8B parameters'; training times sampled from a logit-normal (more effort mid-path); Euler sampling. "
            "FLUX.1: Black Forest Labs blog, 1 Aug 2024 — 'scaled to 12B parameters', 'building on flow matching' (HF "
            "metadata 11.9 B; notes Appendix A: say 12 billion); FLUX.1 [schnell] model card: 'a 12 billion parameter "
            "rectified flow transformer', 1–4 steps after latent adversarial diffusion distillation. Episode 18: the "
            "transformers (MM-DiT)."))

        def model_card(name, maker, date, facts, src, x):
            title = title_text(name, 40, INK)
            who = text(maker, 22, FM_C, weight=m.BOLD)
            when = code(date, 20, TIME)
            items = m.VGroup(*[m.VGroup(m.Dot(radius=0.05, color=FM_C), text(f, 21, INK)).arrange(m.RIGHT, buff=0.18)
                               for f in facts]).arrange(m.DOWN, buff=0.2, aligned_edge=m.LEFT)
            s = cite(src)
            body = m.VGroup(title, who, when, items, s).arrange(m.DOWN, buff=0.22, aligned_edge=m.LEFT)
            box = card(6.0, 4.6, EDGE, CARD, radius=0.18, stroke=2)
            body.move_to(box).align_to(box, m.LEFT).shift(0.4 * m.RIGHT)
            assert body.width < 5.3, (name, body.width)
            return m.VGroup(box, body).move_to([x, -0.45, 0])

        sd3 = model_card("Stable Diffusion 3", "Stability AI", "March 2024",
                         ("rectified-flow transformers", "up to 8 billion parameters", "more training on the middle of the line"),
                         "Esser et al., 2024 · arXiv:2403.03206", -3.2)
        flux = model_card("FLUX.1", "Black Forest Labs", "August 2024",
                          ("12 billion parameters", "\"building on flow matching\"", "a distilled version: 1 to 4 steps"),
                          "Black Forest Labs, 1 Aug 2024 · model cards", 3.2)
        dated = cite("as of October 2026 · Episode 18 opens up these transformers").move_to([0, -3.3, 0])
        mascot = pixel(0.42).move_to([6.05, -3.3, 0])

        self.say(
            "[Flow matching now trains big image models]: [Stable Diffusion three, in March twenty twenty-four], "
            "[rectified-flow transformers up to eight billion parameters].",
            m.AnimationGroup(self.new_header("08 · in the wild", "Who uses it"), m.FadeIn(dated)),
            m.FadeIn(sd3, shift=0.15 * m.UP),
            m.Circumscribe(sd3[1][3][:2], color=FM_C, buff=0.08),
        )
        self.say(
            "[And that August, the FLUX models from Black Forest Labs]: [twelve billion parameters, built on flow matching].",
            m.AnimationGroup(m.FadeIn(flux, shift=0.15 * m.UP), m.FadeIn(mascot, shift=0.1 * m.UP)),
            m.Circumscribe(flux[1][3][:2], color=FM_C, buff=0.08),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes=(
            "Draw (straight training lines x_t = (1 − t)·x₀ + t·ε), learn (the velocity ε − x₀; the network learns the "
            "average over crossing lines, so sampling paths bend), follow (Euler steps from t = 1 to 0; on the digit model "
            f"{CF[('flow', 5)]} vs {CF[('ddim', 5)]} at 5 steps, tied at 10). Next: Episode 12, One Step at a Time? — "
            "distillation (progressive distillation, consistency models, SDXL Turbo)."))
        halo = glow(TIME, radius=4.0)
        words = m.VGroup(*[title_text(w, 58, c) for w, c in (("Draw", FM_C), ("Learn", MODEL), ("Follow", TIME))])
        subs = m.VGroup(*[code(s, 20, MUTED) for s in ("straight lines", "the velocity ε − x₀", "Euler steps, 1 → 0")])
        words.arrange(m.RIGHT, buff=2.0)
        on_baseline(words)
        for w, s in zip(words, subs):
            s.set_x(w.get_x())
        line_y = min(w.get_bottom()[1] for w in words) - 0.45
        for s in subs:
            sit(s, line_y)
        mid = baseline(words[0]) + 0.3
        dots = m.VGroup(*[m.Dot([(l_.get_right()[0] + r_.get_left()[0]) / 2, mid, 0], radius=0.07, color=MUTED)
                          for l_, r_ in zip(words, words[1:])])
        row = m.VGroup(words, dots, subs).move_to(1.0 * m.UP)
        assert row.width < 12.6 and all(a.get_right()[0] + 0.3 < b.get_left()[0] for a, b in zip(subs, subs[1:])), (row.width, [(s_.get_left()[0], s_.get_right()[0]) for s_ in subs])
        f = formula("x_t", "=", "(1-t)", r"\,x_0", "+", "t", r"\,\varepsilon", r"\qquad", "v", "=", r"\varepsilon", "-", "x_0", size=44)
        f[2].set_color(TIME)
        f[3].set_color(DATA)
        f[5].set_color(TIME)
        f[6].set_color(NOISE)
        f[8].set_color(FM_C)
        f[10].set_color(NOISE)
        f[12].set_color(DATA)
        f.next_to(row, m.DOWN, buff=0.6)
        tag = text("Straight lines to train on; a curved, averaged field to follow.", 28, MUTED)
        mascot = pixel(0.42)
        nxt = pill("NEXT · ONE STEP AT A TIME?", TIME, BLUE_FILL, 20, mono=True)
        foot = m.VGroup(tag, m.VGroup(mascot, nxt).arrange(m.RIGHT, buff=0.3)).arrange(m.DOWN, buff=0.35)
        foot.next_to(f, m.DOWN, buff=0.5)

        self.say(
            "[To sum up]: [draw], [learn], [follow].",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.3 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(dots[0]), m.FadeIn(words[1], shift=0.3 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(dots[1]), m.FadeIn(words[2], shift=0.3 * m.UP), m.FadeIn(subs[2]), m.FadeIn(f)),
        )
        self.say(
            "[Straight lines to train on], [an averaged velocity to learn], [and a few Euler steps to follow it].",
            m.Circumscribe(f[:7], color=FM_C, buff=0.08),
            m.Circumscribe(f[8:], color=FM_C, buff=0.08),
            m.Write(tag),
        )
        self.say(
            "[Next, squeezing many steps into one].",
            m.AnimationGroup(m.FadeIn(mascot, shift=0.2 * m.UP), m.FadeIn(nxt, shift=0.2 * m.UP)),
        )
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, f, foot)], run_time=0.9)
