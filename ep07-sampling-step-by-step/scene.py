"""Out of the Noise · Episode 7 · Sampling Step by Step (Module 2 · Learning to Denoise)

Start from pure noise and apply the denoiser a thousand times: DDPM ancestral sampling (Ho et al. 2020, Algorithm 2).
Running example: the digit model (research/experiments/mac_runs.md) — one "7" generated on this Mac in 1,000 steps
(DDPM, w = 1, starting noise np.random.default_rng(57): arrays/digit_trajectory_7_more_seeds.npz, every x_t and the
network's clean guess x̂₀ recorded). The swirl (diffusion_reference) shows the same loop on 2,000 points.

The episode takes one real step of that run apart (t = 500: the network's ε̂ and the step's fresh noise z are
recovered from the saved x_t and x̂₀, asserted to be unit-size noise); shows the loop as a 6-line excerpt in
diffusion_reference's names (executed below and asserted equal to ddpm_sample); runs it on the swirl with the exact
denoiser and the trained MLP (toy.md §3: 0.02454 / 0.02534 vs the data's 0.02405, recomputed by precompute.py and
asserted); watches the 7 emerge with its x̂₀ beside it; deletes the fresh noise (the points pile up at the arms' inner
tips) and runs 500 chains from ONE starting point (they cover the swirl); quotes toy.md's determinism numbers (two
runs from the same noise: same arm 49.9%, 0.90 apart vs 0.91 for random pairs) to set up Episode 9; and the cost:
1,000 network calls, 4.3 s per digit and 172 s per 1,000 digits on this Mac, cFID 2.64 vs the floor 2.06
(classifier-feature FID, not Inception FID).

Every number on screen is computed below (diffusion_reference.py, the run's arrays, precompute.py's runs) and asserted,
or loaded with measured(...), or cited with its source.

Sources (research/notes.md §7, Appendix C row 7; read October 2026):
- Ho, Jain & Abbeel 2020 (DDPM, arXiv:2006.11239): Algorithm 2 (x_{t−1} = (1/√α_t)(x_t − β_t/√(1 − ᾱ_t)·ε_θ) + σ_t·z,
  z = 0 at t = 1), §3.2 (σ_t² = β_t or β̃_t "had similar results"; "Algorithm 2, resembles Langevin dynamics with εθ as
  a learned gradient of the data density"). notes §7 worked example: at t = 500, 1/√α_t = 1.00506, β_t/√(1 − ᾱ_t) =
  0.01046, σ_t = 0.1002 (both variances).
- notes §7 mechanism and misconceptions: deleting σ_t·z while keeping DDPM's mean is not a derived sampler; DDIM with
  η = 0 is the principled deterministic version (Episode 9); the same x_T gives different samples.
- research/experiments/toy.md §3 and toy.json (headline ddpm1000_exact_dist, ddpm1000_mlp_dist, data_dist,
  ddpm_seed1_vs_seed2_same_arm_mlp; samplers/mlp/determinism).
- research/experiments/mac_runs.md §3–4 and mac_runs.json (headline cfid_ddpm1000, cfid_real_data_floor,
  seconds_per_1000_digits_ddpm1000, digit_unet_params; latency rows: one digit, batch 1). The Mac was shared while
  timing: ±5–10% jitter (mac_runs.md §8). cFID uses our own MNIST classifier's features: not Inception FID.

Note on the brief: the six "more seeds" 7s (seeds 17 … 67) each start from their OWN noise (np.random.default_rng(seed)),
not from the same noise, so they are shown as six different starting noises in "The cost", not as evidence of
stochasticity. Stochasticity from one start is shown on the swirl (computed) and with toy.md's measured numbers.

Illustrative (computed, but chosen): the step at t = 500; the start point z[115] of the 500 chains; the six traced
particles; the strip's time steps.

    python3 precompute.py                     # once (~20 s): writes swirl_runs.npz
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

PREFIX = "ep07"
ARR = EXPERIMENTS / "arrays"
TOY = lambda *keys: measured("toy", "headline", *keys)  # noqa: E731
MAC = lambda *keys: measured("mac_runs", "headline", *keys)  # noqa: E731
BETAS = linear_betas()
AB = ALPHA_BAR

assert T == 1000 and np.isclose(BETAS[1], 1e-4) and np.isclose(BETAS[T], 0.02)

# ---------------------------------------------------------------------------- the digit model's 7 (mac_runs.md)

TRAJ = np.load(ARR / "digit_trajectory_7_more_seeds.npz")
SEED = 57
SEEDS = [int(s) for s in TRAJ["seeds"]]
assert int(TRAJ["label"]) == 7 and SEED in SEEDS and SEEDS == [17, 27, 37, 47, 57, 67]
CODE_T = TRAJ["ddpm_ddpm1000_w1_t"].astype(int)
assert np.array_equal(CODE_T, np.arange(999, -1, -1))         # the run's code t = 999 … 0 is the paper's t − 1
_XT = TRAJ[f"seed{SEED}_ddpm_ddpm1000_w1_x_t"].astype(np.float64)
_X0 = TRAJ[f"seed{SEED}_ddpm_ddpm1000_w1_x0_hat"].astype(np.float64)
FINAL = TRAJ[f"seed{SEED}_ddpm_ddpm1000_w1_final"].astype(np.float64)
STATES = np.concatenate([_XT, FINAL[None]])                       # STATES[k] = x_t with t = 1000 − k (paper)
GUESSES = np.concatenate([_X0, FINAL[None]])                      # GUESSES[k] = x̂₀ at that step (the output at t = 0)
_start = np.random.default_rng(SEED).standard_normal((1, 28, 28, 1), dtype=np.float32)[0, ..., 0]
assert np.allclose(STATES[0], _start.astype(np.float16))         # x_1000 is the seed's noise, exactly
FWD = np.load(ARR / "digit_forward_process.npz")
assert np.allclose(FWD["abar"], AB[1:])                           # the run's ᾱ (code t) = the course's ᾱ (t + 1)
DIGIT_PARAMS = MAC("digit_unet_params")
assert DIGIT_PARAMS == 3940801 and f"{DIGIT_PARAMS / 1e6:.1f}" == "3.9"


def corr(a, b):
    return float(np.corrcoef(np.ravel(a), np.ravel(b))[0, 1])


C_XT = {t: corr(STATES[1000 - t], FINAL) for t in (600, 400, 100)}
C_X0 = {t: corr(np.clip(GUESSES[1000 - t], -1, 1), FINAL) for t in (800, 450)}
assert C_XT[600] < 0.15 and C_XT[400] < 0.4 and C_XT[100] > 0.8      # the picture emerges late …
assert C_X0[800] > 0.5 and C_X0[450] > 0.9                            # … while the guess settles early

# one real step at t = 500, taken apart (the network's ε̂ and the fresh noise z, recovered from the saved arrays)
STEP_T = 500
_k = 1000 - STEP_T
AB_T, AB_S = AB[STEP_T], AB[STEP_T - 1]
BETA = 1 - AB_T / AB_S
assert np.isclose(BETA, BETAS[STEP_T])
assert np.abs(GUESSES[_k]).max() < 2.99                                       # x̂₀ was not clipped at this step
EPS_HAT = (STATES[_k] - np.sqrt(AB_T) * GUESSES[_k]) / np.sqrt(1 - AB_T)      # x̂₀ = (x_t − √(1 − ᾱ)·ε̂)/√ᾱ, inverted
MEAN = (STATES[_k] - BETA / np.sqrt(1 - AB_T) * EPS_HAT) / np.sqrt(1 - BETA)
SIGMA = float(np.sqrt(BETA * (1 - AB_S) / (1 - AB_T)))                         # the digit model's σ² = β̃
Z = (STATES[_k + 1] - MEAN) / SIGMA
assert abs(EPS_HAT.std() - 1) < 0.05 and abs(Z.std() - 1) < 0.05 and abs(Z.mean()) < 0.1   # both are unit-size noise
COEF_SCALE, COEF_EPS = 1 / np.sqrt(1 - BETA), BETA / np.sqrt(1 - AB_T)
assert f"{COEF_SCALE:.3f}" == "1.005" and f"{COEF_EPS:.4f}" == "0.0105" and f"{SIGMA:.3f}" == "0.100"
assert f"{np.sqrt(BETA):.3f}" == f"{SIGMA:.3f}"                               # σ² = β or β̃: the same here
assert round(100 * COEF_EPS) == 1 and round(1000 * (COEF_SCALE - 1)) == 5    # "one percent", "half a percent"

# ---------------------------------------------------------------------------- the loop on screen (≤ 6 lines)

LOOP_SRC = "\n".join([
    "x = gaussian_noise(n, seed)                     # pure noise",
    "for t in range(T, 0, -1):                       # t = 1000 … 1",
    "    eps = eps_model(x, t)                       # predict the noise",
    "    x = (x - beta[t] / np.sqrt(1 - ab[t]) * eps) / np.sqrt(1 - beta[t])",
    "    if t > 1:                                   # never on the last step",
    "        x = x + np.sqrt(beta[t]) * rng.standard_normal(x.shape)",
])
LOOP_LINES = LOOP_SRC.split("\n")
assert len(LOOP_LINES) == 6


def run_loop(eps_model, n, seed):
    """Execute the excerpt exactly as shown."""
    names = dict(gaussian_noise=gaussian_noise, eps_model=eps_model, T=T, np=np, beta=BETAS, ab=AB,
                 rng=np.random.default_rng(seed), n=n, seed=seed)
    exec(LOOP_SRC, names)
    return names["x"]


assert np.allclose(run_loop(exact_eps, 48, 7), ddpm_sample(exact_eps, gaussian_noise(48, seed=7), seed=7).xs[-1],
                   atol=1e-8)                                                 # the excerpt IS ddpm_sample (σ² = β)

# ---------------------------------------------------------------------------- the swirl runs (precompute.py)

RUNS = np.load(EP_DIR / "swirl_runs.npz")
N_PTS, EVERY = len(RUNS["z"]), 5
assert N_PTS == 2000 and np.allclose(RUNS["z"], gaussian_noise(2000, seed=5))      # toy.md's shared noise
D_DATA, D_EXACT, D_MLP = TOY("data_dist"), TOY("ddpm1000_exact_dist"), TOY("ddpm1000_mlp_dist")
assert round(swirl_distance(RUNS["exact_final"]), 5) == D_EXACT == 0.02454
assert round(swirl_distance(RUNS["mlp_final"]), 5) == D_MLP == 0.02534
assert (f"{D_DATA:.4f}", f"{D_EXACT:.4f}", f"{D_MLP:.4f}") == ("0.0240", "0.0245", "0.0253")
MLP_PARAMS = TOY("mlp_params")
assert MLP_PARAMS == count_params(load_weights()) == 45954

EX_STATES, EX_DIST = RUNS["exact_states"].astype(float), RUNS["exact_dist"].astype(float)
_d = EX_DIST.mean(1)
_sd = EX_STATES.std(axis=(1, 2))
st_of = lambda t: (1000 - t) // EVERY  # noqa: E731   (stored state index of step t)
assert _sd[st_of(500)] > 0.95 and _d[st_of(500)] > 0.9 * _d[0]               # "the first 500 steps: hardly changes"
assert _d[st_of(50)] > 3 * _d[-1] and _d[st_of(200)] > 0.2                   # "the arms snap sharp in the last 50"

NN_STATES, NN_DIST = RUNS["nonoise_states"].astype(float), RUNS["nonoise_dist"].astype(float)
NN_COVER = swirl_coverage(RUNS["nonoise_final"])
assert NN_COVER == 0.05 and np.mean(np.linalg.norm(RUNS["nonoise_final"], axis=1) < 0.2) > 0.95   # the inner tips
OS_STATES, OS_DIST = RUNS["one_start_states"].astype(float), RUNS["one_start_dist"].astype(float)
START = RUNS["z"][int(RUNS["start_index"])]
assert np.allclose(OS_STATES[0], START) and len(OS_STATES[0]) == 500
OS_COVER = swirl_coverage(RUNS["one_start_final"])
OS_ARM0 = float(np.mean(nearest_on_swirl(RUNS["one_start_final"])[1] == 0))
assert OS_COVER == 1.0 and 0.45 < OS_ARM0 < 0.55                              # all over the swirl, both arms

DET = measured("toy", "samplers", "mlp", "determinism")
SAME_ARM, GAP, RAND_GAP = TOY("ddpm_seed1_vs_seed2_same_arm_mlp"), DET["ddpm_seed1_vs_seed2"], DET["random_pair_baseline"]
assert SAME_ARM == DET["ddpm_seed1_vs_seed2_same_arm"] == 0.499
_mlp1, _mlp2 = RUNS["mlp_final"], RUNS["mlp_final_seed2"]
assert round(float(np.mean(nearest_on_swirl(_mlp1)[1] == nearest_on_swirl(_mlp2)[1])), 3) == SAME_ARM
assert round(float(np.linalg.norm(_mlp1 - _mlp2, axis=1).mean()), 4) == GAP == 0.9003
assert f"{GAP:.2f}" == "0.90" and f"{RAND_GAP:.2f}" == "0.91"

# ---------------------------------------------------------------------------- the cost (mac_runs.md §4)

CFID, FLOOR = MAC("cfid_ddpm1000"), MAC("cfid_real_data_floor")
S_PER_1000 = MAC("seconds_per_1000_digits_ddpm1000")
_lat = [r for r in measured("mac_runs", "latency", "rows") if r["cfg"] == "ddpm_ddpm1000_w1" and r["batch"] == 1]
S_ONE = _lat[0]["s"]
assert (f"{CFID:.2f}", f"{FLOOR:.2f}", f"{S_PER_1000:.0f}", f"{S_ONE:.1f}") == ("2.64", "2.06", "172", "4.3")
assert measured("mac_runs", "digit_sampling", "configs", "ddpm_ddpm1000_w1", "nfe_per_sample") == 1000
MANY = np.load(ARR / "digit_trajectory_7.npz")
MANY_XT = MANY["many20_ddpm1000_w1_x_t"].astype(np.float64)                    # (40 frames, 20 digits, 28, 28)
MANY_T = MANY["many20_ddpm1000_w1_t"].astype(int) + 1                           # paper t of each frame
MANY_FINAL = MANY["many20_ddpm1000_w1_final"].astype(np.float64)
assert MANY_XT.shape == (40, 20, 28, 28) and MANY_T[0] == 1000 and MANY_T[-1] == 1
assert list(MANY["many20_labels"]) == list(range(10)) * 2
DDIM_SAME = TRAJ[f"seed{SEED}_ddpm_ddim50_w1_final"].astype(np.float64)        # Episode 9's sampler, same x_T
assert np.allclose(TRAJ[f"seed{SEED}_ddpm_ddim50_w1_x_t"][0], TRAJ[f"seed{SEED}_ddpm_ddpm1000_w1_x_t"][0])
assert corr(DDIM_SAME, FINAL) < 0.6                                            # a different 7 from the same noise

# ---------------------------------------------------------------------------- look

PC = np.array([-3.25, -0.45, 0.0])     # the swirl plane's centre
PS = 5.6                               # … and its size
RX = 3.6                               # the right column's centre
LEFT = 1.15                            # the right column's left edge
FLAT = set("abcdefhiklmnorstuvwxzABCDEFGHIJKLMNOPRSTUVWXYZ0123456789")
PALE = mix(TEAL, INK, 0.6)


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


def step_formula(size=46):
    """Algorithm 2's step: [0] x_{t−1} [1] = [2] 1/√α_t [3] ( [4] x_t [5] − [6] β_t/√(1 − ᾱ_t) [7] ε̂ [8] ) [9] + [10] σ_t
    [11] z — the schedule BLUE, the network's guess AMBER, the fresh noise VIOLET."""
    f = formula(r"x_{t-1}", "=", r"\frac{1}{\sqrt{\alpha_t}}", "(", "x_t", "-",
                r"\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}", r"\,\hat\varepsilon", ")", "+", r"\sigma_t", r"\,z", size=size)
    for i in (2, 6, 10):
        f[i].set_color(TIME)
    f[7].set_color(MODEL)
    f[11].set_color(NOISE)
    return f


def late(p):
    """A rate that hurries through the early (noise-only) steps and slows down near the end."""
    return lambda a: float(a) ** p


def picture_weight(t):
    """0 while the noise wins, 1 once the picture wins: a quick smooth step in log SNR around SNR = 1 (t ≈ 259)."""
    if t <= 0:
        return 1.0
    w = float(np.clip((log_snr(AB[int(round(t))]) + 0.6) / 1.2, 0, 1))
    return w * w * (3 - 2 * w)


def state_rgb(x, t):
    """A noisy picture: clipped to −1 … 1, VIOLET while noise wins, TEAL once the picture does (Episodes 4, 5)."""
    return tint(np.clip(x, -1, 1), mix(VIOLET, TEAL, picture_weight(t)), -1, 1)


def guess_rgb(x):
    return tint(np.clip(x, -1, 1), MODEL, -1, 1)


def texture(rgb):
    """The RGBA texture picture() makes from a small image (crisp pixels), for swapping in place."""
    k = max(1, int(np.ceil(540 / max(rgb.shape[:2]))))
    arr = np.repeat(np.repeat(rgb, k, axis=0), k, axis=1)
    rgba = np.concatenate([arr, np.full(arr.shape[:2] + (1,), 255, np.uint8)], axis=-1)
    rgba.flags.writeable = False
    return rgba


class Flip:
    """A picture whose pixels are swapped for frame k of a sequence (only when k changes)."""

    def __init__(self, rgb_of, k, height):
        self.rgb_of, self.k = rgb_of, int(k)
        self.image = picture(rgb_of(self.k), height)

    def show(self, k):
        k = int(k)
        if k != self.k:
            self.k = k
            self.image.paint = self.image.paint.but(texture=texture(self.rgb_of(k)))


def drive(mob, t0, t1, hooks, run_time=1.4, rate=m.smooth):
    """Animate a step t from t0 to t1, calling every hook(t) each frame."""

    def step(_, a):
        t = t0 + (t1 - t0) * a
        for hook in hooks:
            hook(t)

    return m.UpdateFromAlphaFunc(mob, step, run_time=run_time, rate_func=rate)


class Readout(m.VGroup):
    """A mono number display ("t = 250"), left-aligned at a fixed point; .show(v) updates it."""

    def __init__(self, left, y, fmt, value, size=40, color=TIME):
        super().__init__()
        self.anchor, self.fmt, self.size, self.color = np.array([left, y, 0.0]), fmt, size, color
        self.words = self._make(value)
        self.add(self.words)

    def _make(self, value):
        return code(self.fmt.format(value), self.size, self.color, weight=m.BOLD).move_to(self.anchor, aligned_edge=m.LEFT)

    def show(self, value):
        self.words.become(self._make(value))
        return self


class Progress(m.VGroup):
    """A thin BLUE bar: the share of the 1,000 steps done."""

    def __init__(self, center, span=4.2, h=0.16):
        super().__init__()
        self.c, self.span, self.h = np.array(center, dtype=float), span, h
        self.track = m.RoundedRectangle(corner_radius=h / 2, width=span, height=h).set_stroke(EDGE, 2).set_fill(CARD, 1)
        self.track.move_to(self.c)
        self.bar = self._bar(0.0)
        self.add(self.track, self.bar)

    def _bar(self, f):
        w = max(self.span * float(np.clip(f, 0, 1)), 0.02)
        bar = m.RoundedRectangle(corner_radius=self.h / 2, width=w, height=self.h).set_stroke(width=0).set_fill(TIME, 1)
        return bar.move_to([self.c[0] - self.span / 2 + w / 2, self.c[1], 0])

    def show(self, f):
        self.bar.become(self._bar(f))


def plane(center=PC, size=PS, lim=1.5):
    """A bare square plane for the swirl (no axes), data range [−lim, lim]²."""
    return Plot((-lim, lim), (-lim, lim), width=size, height=size, center=tuple(center), axes=False)


def at(plot, xy):
    return plot.coords(float(xy[0]), float(xy[1]))


def zoom_of(states, lo=1.5, hi=3.7, win=4):
    """The plane's half-width for each stored state: the cloud's extent (99.5th percentile), smoothed."""
    ext = np.percentile(np.abs(states).max(axis=2), 99.5, axis=1)
    ext = np.array([ext[max(0, j - win):j + win + 1].max() for j in range(len(ext))])
    ext = np.convolve(np.pad(ext, win, mode="edge"), np.ones(2 * win + 1) / (2 * win + 1), mode="valid")
    return np.clip(1.08 * ext, lo, hi)


class Swarm:
    """Stored swirl states (every 5th step), replayed; a point turns from VIOLET (noise) toward TEAL (data) as it
    reaches the swirl, and only once the picture is winning (t below ≈ 300). The view zooms with the cloud."""

    SHADES = 5

    def __init__(self, plot, states, dists, radius=0.026, opacity=0.95):
        self.plot, self.xs, self.ds = plot, states, dists
        self.lims = zoom_of(states)
        self.ring = m.Circle(radius=1.0).set_stroke(FAINT, 2.5)
        self._view(0)
        self.cloud = plot.cloud(states[0], NOISE, radius, opacity)
        self.palette = [mix(VIOLET, TEAL, k / (self.SHADES - 1)) for k in range(self.SHADES)]
        self.shade = np.zeros(len(self.cloud), int)
        self.hooks = []

    def t_of(self, i):
        return 1000 - EVERY * i

    def _view(self, i):
        lim = float(np.interp(i, np.arange(len(self.lims)), self.lims))
        p = self.plot
        p.x0, p.x1, p.y0, p.y1 = -lim, lim, -lim, lim
        self.ring.become(m.Circle(radius=PS / 2 / lim).set_stroke(FAINT, 2.5).move_to(p.coords(0, 0)))

    def _recolor(self, i):
        j = int(round(i))
        cap = int(round((self.SHADES - 1) * picture_weight(self.t_of(j))))
        lv = np.clip(np.round((0.16 - self.ds[j]) / 0.11 * (self.SHADES - 1)), 0, cap).astype(int)
        for k in np.flatnonzero(lv != self.shade):
            self.cloud[k].set_color(self.palette[lv[k]])
        self.shade = lv

    def state(self, i):
        j = int(np.floor(i))
        k = min(j + 1, len(self.xs) - 1)
        return self.xs[j] + (self.xs[k] - self.xs[j]) * (i - j)

    def put(self, i):
        self._view(i)
        self.cloud.positions(self.state(i))
        self._recolor(i)
        for hook in self.hooks:
            hook(i)

    def replay(self, t0, t1, seconds, rate=m.linear):
        i0, i1 = (1000 - t0) / EVERY, (1000 - t1) / EVERY

        def step(_, a):
            self.put(i0 + (i1 - i0) * a)
        return m.UpdateFromAlphaFunc(self.cloud, step, run_time=seconds, rate_func=rate)


def line_glyphs(block, k):
    """The glyphs of line k of LOOP_SRC inside a code_block (whitespace draws none)."""
    body = block[1]
    start = sum(len("".join(line.split())) for line in LOOP_LINES[:k])
    return body[start:start + len("".join(LOOP_LINES[k].split()))]


def op_sign(kind, size=0.42, color=MUTED, width=6):
    """−, + or = drawn as strokes (a text minus is too flat to read as text)."""
    h = m.Line([-size / 2, 0, 0], [size / 2, 0, 0])
    if kind == "+":
        g = m.VGroup(h, m.Line([0, -size / 2, 0], [0, size / 2, 0]))
    elif kind == "=":
        g = m.VGroup(h.shift(0.1 * m.UP), m.Line([-size / 2, -0.1, 0], [size / 2, -0.1, 0]))
    else:
        g = m.VGroup(h)
    return g.set_stroke(color, width)


def labelled(image, words, buff=0.16):
    """A framed picture with a caption under it."""
    return m.Group(image, words.next_to(image, m.DOWN, buff=buff))


# ---------------------------------------------------------------------------- the episode


class Ep07SamplingStepByStep(Episode):
    number = 7

    def lesson(self):
        self.hook()
        self.one_step()
        self.the_loop()
        self.swirl_unmelts()
        self.digit_emerges()
        self.why_fresh_noise()
        self.it_forgets()
        self.the_cost()
        self.recap()

    # ------------------------------------------------------------------ 0 · hook

    def hook(self):
        self.next_section("Hook", notes=(
            f"One 7 generated on this Mac by the digit model ({DIGIT_PARAMS:,}-parameter U-Net, mac_runs.md §3): DDPM "
            f"ancestral sampling, 1,000 steps, w = 1, label 7, starting noise np.random.default_rng({SEED}) "
            "(arrays/digit_trajectory_7_more_seeds.npz). Shown: x_t at every step, clipped to −1 … 1, VIOLET while the "
            "noise wins and TEAL once the picture does (a smooth step around SNR = 1, t ≈ 259). Episode 5: the network "
            "predicts the noise; Episode 6: that prediction is the score, an arrow toward more likely data."))
        flip = Flip(lambda k: state_rgb(STATES[k], 1000 - k), 0, 3.2)
        pic = framed(flip.image).move_to([-4.3, -0.15, 0])
        counter = Readout(-5.45, -2.3, "t = {:d}", 1000, 36)
        net = m.VGroup(card(2.2, 0.8, MODEL, AMBER_FILL, radius=0.16), text("the network", 24, INK, weight=m.BOLD))
        net[1].move_to(net[0])
        net.move_to([-0.75, -0.15, 0])
        go = m.CurvedArrow(pic[0].get_right() + [0.12, 0.75, 0], net[0].get_top() + [0, 0.1, 0], angle=-1.0)
        back = m.CurvedArrow(net[0].get_bottom() + [0, -0.1, 0], pic[0].get_right() + [0.12, -0.75, 0], angle=-1.0)
        loop_arrow = m.VGroup(go.set_stroke(MODEL, 4), back.set_stroke(MODEL, 4))
        times = code("× 1,000", 26, TIME, weight=m.BOLD).move_to([-1.2, -1.95, 0])

        title = title_text("sampling", 66, MODEL).move_to([RX, 1.75, 0])
        sub = text("from pure noise, one small step at a time", 24, MUTED).next_to(title, m.DOWN, buff=0.28)
        ep5 = pill("Episode 5 · it predicts the noise", MODEL, AMBER_FILL, 22)
        ep6 = pill("Episode 6 · which way is more likely", SCORE, LIME_FILL, 22)
        links = m.VGroup(ep5, ep6).arrange(m.DOWN, buff=0.22).move_to([RX, -0.35, 0])
        rows = m.VGroup()
        for k, words in enumerate(("one step back", "the loop that repeats it", "why every step adds fresh noise")):
            rows.add(m.VGroup(code(f"{k + 1}", 26, MODEL, weight=m.BOLD), text(words, 26, INK)).arrange(m.RIGHT, buff=0.3))
        rows.arrange(m.DOWN, buff=0.3, aligned_edge=m.LEFT).move_to([RX, -1.0, 0])
        for r in rows:
            sit(r[0], baseline(r[1]))

        def at_t(t):
            flip.show(1000 - int(round(t)))
            counter.show(int(round(t)))

        self.say(
            "[Every picture a diffusion model makes starts here]: [pure noise].",
            self.new_header("00 · hook", "From noise to a seven"),
            m.AnimationGroup(m.FadeIn(pic, scale=0.9), m.FadeIn(counter)),
        )
        self.say(
            "[A network removes a little of it], [a thousand times in a row], [and a seven appears].",
            m.AnimationGroup(m.FadeIn(net, shift=0.15 * m.LEFT), m.Create(loop_arrow)),
            m.FadeIn(times, shift=0.1 * m.LEFT),
            drive(pic, 1000, 0, [at_t], run_time=4.2, rate=late(0.55)),
        )
        self.say(
            "[Episode five trained that network to predict the noise]; [episode six showed which way the prediction points].",
            m.AnimationGroup(m.FadeIn(title, shift=0.2 * m.UP), m.FadeIn(sub), m.FadeIn(ep5, shift=0.1 * m.UP)),
            m.FadeIn(ep6, shift=0.1 * m.UP),
        )
        self.say(
            "Today: [one step back], [the loop that repeats it], [and why every step adds fresh noise].",
            m.AnimationGroup(m.FadeOut(links, shift=0.1 * m.DOWN), m.FadeIn(rows[0], shift=0.1 * m.UP)),
            m.FadeIn(rows[1], shift=0.1 * m.UP),
            m.FadeIn(rows[2], shift=0.1 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 1 · one step back

    def one_step(self):
        self.next_section("1 · One step back", notes=(
            f"One real step of the run (seed {SEED}) at t = {STEP_T} (the paper's t; the run's code t = {STEP_T - 1}). "
            "From the saved arrays: x_t and x_{t−1} directly; the network's ε̂ = (x_t − √ᾱ_t·x̂₀)/√(1 − ᾱ_t) (x̂₀ unclipped "
            f"here); the fresh noise z = (x_(t−1) − mean)/σ_t, std {Z.std():.3f} (ε̂ std {EPS_HAT.std():.3f}): unit-size, "
            f"as they should be. Coefficients (linear β, notes §7 table): 1/√α_t = {COEF_SCALE:.5f}, "
            f"β_t/√(1 − ᾱ_t) = {COEF_EPS:.5f}, σ_t = {SIGMA:.4f} (the digit model uses σ² = β̃; β gives the same to 3 "
            "decimals at t = 500). Algorithm 2 (Ho et al. 2020); α_t = 1 − β_t. Pictures clipped to −1 … 1."))
        f = step_formula(46).move_to([0, 2.05, 0])
        notes = m.VGroup(code(f"{COEF_SCALE:.3f}", 22, TIME), code(f"{COEF_EPS:.4f}", 22, TIME), code(f"{SIGMA:.3f}", 22, TIME))
        for n_, i in zip(notes, (2, 6, 10)):
            n_.next_to(f[i], m.DOWN, buff=0.2)
        y_n = min(n_.get_y() for n_ in notes)
        for n_ in notes:
            n_.set_y(y_n)
        at_lab = code(f"at t = {STEP_T}", 22, MUTED).next_to(notes[2], m.RIGHT, buff=0.45).set_y(y_n)
        src = cite("Ho et al., 2020 · Algorithm 2 · α = 1 − β").move_to([3.9, -3.35, 0])

        h = 1.85
        imgs = [framed(picture(state_rgb(STATES[_k], STEP_T), h)),
                framed(picture(tint(np.clip(EPS_HAT, -1, 1), NOISE, -1, 1), h), MODEL),
                framed(picture(tint(np.clip(Z, -1, 1), NOISE, -1, 1), h), NOISE),
                framed(picture(state_rgb(STATES[_k + 1], STEP_T - 1), h))]
        ops = [op_sign(s) for s in "-+="]
        row = m.Group(imgs[0], ops[0], imgs[1], ops[1], imgs[2], ops[2], imgs[3]).arrange(m.RIGHT, buff=0.42)
        row.move_to([0, -0.75, 0])
        labs = [tex(f"x_{{{STEP_T}}}", 34), m.VGroup(tex(r"\hat\varepsilon", 34, MODEL), text("the network's guess", 20, MUTED)),
                m.VGroup(tex("z", 34, NOISE), text("fresh noise", 20, MUTED)), tex(f"x_{{{STEP_T - 1}}}", 34)]
        for lab, im in zip(labs, imgs):
            if isinstance(lab, m.VGroup):
                lab.arrange(m.RIGHT, buff=0.15)
            lab.next_to(im, m.DOWN, buff=0.2)
        y_l = min(lab.get_y() for lab in labs)
        for lab in labs:
            lab.set_y(y_l)
        arc = m.CurvedArrow(imgs[0].get_top() + [0.2, 0.1, 0], imgs[1].get_top() + [-0.2, 0.1, 0], angle=-0.9)
        arc.set_stroke(MODEL, 4)
        arc_lab = text("the network", 22, MODEL, weight=m.BOLD).next_to(arc, m.UP, buff=0.08)
        cap = cite(f"one step of the seven, generated on this Mac · seed {SEED}").move_to([-3.2, -3.35, 0])
        same = text("to the eye, nothing changed", 22, INK).next_to(labs[3], m.DOWN, buff=0.22).set_x(imgs[3].get_x())
        if same.get_right()[0] > 6.5:
            same.shift((6.5 - same.get_right()[0]) * m.RIGHT)

        self.say(
            "[Here is one real step from that run], [halfway down]: [the noisy seven at t equals five hundred].",
            m.AnimationGroup(self.new_header("01 · one step", "One step back"), m.FadeIn(cap)),
            m.FadeIn(imgs[0], shift=0.1 * m.UP),
            m.FadeIn(labs[0]),
        )
        self.say(
            "[The network looks at it and predicts the noise inside]: [epsilon hat].",
            m.AnimationGroup(m.Create(arc), m.FadeIn(arc_lab)),
            m.AnimationGroup(m.FadeIn(imgs[1], target_position=imgs[0]), m.FadeIn(labs[1]), m.FadeIn(ops[0])),
        )
        self.say(
            "[Subtract about one percent of that prediction], [and scale up by half a percent]: [a small step toward the data].",
            m.AnimationGroup(m.Write(f[:9]), m.FadeIn(notes[1], shift=0.1 * m.UP), m.FadeIn(src)),
            m.AnimationGroup(m.Circumscribe(f[2], color=TIME, buff=0.06), m.FadeIn(notes[0], shift=0.1 * m.UP)),
            m.Circumscribe(m.VGroup(f[4], f[5], f[6], f[7]), color=MODEL, buff=0.08),
        )
        self.say(
            "[Then add back a little fresh noise]: [brand-new random numbers, about a tenth in size].",
            m.AnimationGroup(m.Write(f[9:]), m.FadeIn(ops[1]), m.FadeIn(imgs[2], shift=0.1 * m.UP), m.FadeIn(labs[2])),
            m.AnimationGroup(m.FadeIn(notes[2], shift=0.1 * m.UP), m.FadeIn(at_lab)),
        )
        self.say(
            "[The result is x t minus one]. [To the eye, nothing changed]: [one step is tiny].",
            m.AnimationGroup(m.FadeIn(ops[2]), m.FadeIn(imgs[3], shift=0.1 * m.LEFT), m.FadeIn(labs[3])),
            m.FadeIn(same, shift=0.1 * m.UP),
            m.AnimationGroup(m.Circumscribe(imgs[0], color=INK, buff=0.06), m.Circumscribe(imgs[3], color=INK, buff=0.06)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 2 · the loop

    def the_loop(self):
        self.next_section("2 · The loop", notes=(
            "DDPM's Algorithm 2 (Ho et al. 2020) as six lines in diffusion_reference's names (beta = linear_betas(), "
            "ab = ALPHA_BAR, rng = np.random.default_rng(seed)). This exact text is executed in scene.py and asserted "
            "equal to diffusion_reference.ddpm_sample (σ² = β) on 48 swirl points with the exact ε. z = 0 at t = 1: the "
            "last step adds no noise. One network call per step: 1,000 calls per picture."))
        block = code_block(LOOP_SRC, 20, title="ddpm_sample, simplified · beta, ab = linear_betas(), ALPHA_BAR")
        block.move_to([0, 1.15, 0])
        block[0].set_z_index(0)
        block[1].set_z_index(2)
        assert block.width < 13.0, block.width
        boxes = []
        for k in range(6):
            g = line_glyphs(block, k)
            hl = m.RoundedRectangle(corner_radius=0.06, width=block[0].width - 0.3, height=0.26).set_stroke(width=0)
            hl.set_fill(MODEL, 0.16).move_to([block[0].get_x(), g.get_y(), 0]).set_z_index(1)
            boxes.append(hl)
        cursor = boxes[0]
        last = m.RoundedRectangle(corner_radius=0.06, width=block[0].width - 0.3, height=0.26).set_stroke(width=0)
        last.set_fill(NOISE, 0.16).move_to(boxes[4]).set_z_index(1)
        steps_ = [("predict the noise", MODEL), ("step back", DATA), ("add fresh noise", NOISE)]
        chips = m.VGroup(*[chip(w_, c_, FILLS[c_], width=text(w_, 24, INK, weight=m.BOLD).width + 0.6, height=0.7, size=24)
                           for w_, c_ in steps_])
        chips.arrange(m.RIGHT, buff=1.0).move_to([0, -1.25, 0])
        links = m.VGroup(*[arrow(a.get_right(), b.get_left(), MUTED) for a, b in zip(chips, chips[1:])])
        ret = m.CurvedArrow(chips[2].get_bottom() + [0, -0.1, 0], chips[0].get_bottom() + [0, -0.1, 0], angle=-0.55)
        ret.set_stroke(MUTED, 3)
        ring = m.VGroup(chips, m.VGroup(links, ret))
        mid = code("× 1,000", 26, TIME, weight=m.BOLD).move_to([0, -2.75, 0])
        src = cite("Ho et al., 2020 · Algorithm 2").move_to([4.9, -3.3, 0])
        calls = m.VGroup(code("network calls", 20, MUTED), title_text("1,000", 44, MODEL)).arrange(m.DOWN, buff=0.12)
        calls.move_to([-5.0, -2.75, 0])

        def to_line(k):
            return cursor.animate.move_to(boxes[k].get_center())

        self.say(
            "[Put that step in a loop], [and you have the whole sampler], [in six lines].",
            m.AnimationGroup(self.new_header("02 · the loop", "Algorithm 2"), m.FadeIn(block[0]), m.FadeIn(block[2])),
            m.FadeIn(block[1], lag_ratio=0.02, run_time=1.4),
            m.FadeIn(src),
        )
        self.say(
            "[Start from pure noise]. [For t from a thousand down to one], [predict the noise], [step back], "
            "[and add fresh noise].",
            m.FadeIn(cursor),
            m.AnimationGroup(to_line(1), m.Create(ret), m.FadeIn(mid)),
            m.AnimationGroup(to_line(2), m.FadeIn(chips[0], scale=0.8)),
            m.AnimationGroup(to_line(3), m.GrowArrow(links[0]), m.FadeIn(chips[1], scale=0.8)),
            m.AnimationGroup(to_line(5), m.GrowArrow(links[1]), m.FadeIn(chips[2], scale=0.8)),
        )
        self.say(
            "[On the very last step, no noise is added], [so the picture comes out clean].",
            m.AnimationGroup(m.FadeOut(cursor), m.FadeIn(last), m.Circumscribe(chips[2], color=NOISE, buff=0.08)),
            m.Circumscribe(chips[1], color=DATA, buff=0.08),
        )
        self.say(
            "[This is Algorithm two of the DDPM paper, from twenty twenty]: [a thousand network calls for one picture].",
            m.Circumscribe(src, color=MUTED, buff=0.08),
            m.AnimationGroup(m.FadeIn(calls, shift=0.1 * m.UP), m.Circumscribe(mid, color=TIME, buff=0.1)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 3 · the swirl un-melts

    def swirl_unmelts(self):
        self.next_section("3 · The swirl un-melts", notes=(
            "toy.md §3: z = gaussian_noise(2000, seed=5); DDPM ancestral sampling, 1,000 steps, σ² = β, per-step noise "
            "seed 1, with the exact ε (exact_eps) and with the trained MLP (load_swirl_model, 45,954 parameters, ε-loss). "
            "Recomputed by precompute.py and asserted against toy.json (headline ddpm1000_exact_dist 0.02454, "
            "ddpm1000_mlp_dist 0.02534, data_dist 0.02405 = 0.03·√(2/π)). Drawn: the first 1,000 points, every 5th step; "
            "the view zooms with the cloud (the faint ring has radius 1). A point turns TEAL as it nears the swirl once "
            "the picture is winning (t below ≈ 300). The traces: six drawn points' paths over the last 150 steps."))
        plot = plane()
        swarm = Swarm(plot, EX_STATES, EX_DIST)
        self.swarm = swarm
        counter = Readout(LEFT + 0.5, 1.75, "t = {:d}", 1000, 46)
        swarm.hooks.append(lambda i: counter.show(int(round(swarm.t_of(i)))))
        setting = cite("2,000 points · 1,000 drawn · the view zooms;\nthe faint ring has radius 1").move_to([RX, 0.85, 0])

        box = card(5.0, 2.45, EDGE, CARD, radius=0.18, stroke=2).move_to([RX, -1.35, 0])
        head = text("mean distance to the curve", 22, MUTED).move_to(box.get_top() + [0, -0.38, 0])
        rows = m.VGroup()
        for k, (name, v, col) in enumerate((("the data", D_DATA, DATA), ("exact denoiser", D_EXACT, INK),
                                            ("trained network", D_MLP, MODEL))):
            y = box.get_top()[1] - 0.95 - 0.47 * k
            lab = text(name, 24, col).move_to([box.get_left()[0] + 0.4, y, 0], aligned_edge=m.LEFT)
            num = code(f"{v:.4f}", 26, col, weight=m.BOLD).move_to([box.get_right()[0] - 0.4, y, 0], aligned_edge=m.RIGHT)
            sit(num, baseline(lab))
            rows.add(m.VGroup(lab, num))
        mlp_src = cite(f"{MLP_PARAMS:,} parameters · trained on this Mac · toy.md §3").move_to([RX, -3.0, 0])

        # six traced points: drawn ones whose last 150 steps stay inside the final view and travel
        paths = RUNS["exact_paths"].astype(float)
        tail = paths[850:]
        ok = [p for p in range(paths.shape[1]) if np.abs(tail[:, p]).max() < 1.35
              and np.linalg.norm(tail[0, p] - tail[-1, p]) > 0.3 and 0.5 < np.linalg.norm(tail[-1, p]) < 0.95]
        ends = paths[-1, ok]
        angle = np.arctan2(ends[:, 1], ends[:, 0])
        picks = [ok[int(np.argmin(np.abs(np.angle(np.exp(1j * (angle - a))))))] for a in np.linspace(-np.pi, np.pi, 6, endpoint=False)]
        picks = list(dict.fromkeys(picks))
        assert len(picks) >= 5, picks
        traces = m.VGroup(*[m.VMobject().set_points_as_corners([at(plot, xy) for xy in paths[850::2, p]] + [at(plot, paths[-1, p])])
                            .set_stroke(SCORE, 2.4, opacity=0.9) for p in picks]).set_z_index(4)
        heads = m.VGroup(*[m.Dot(at(plot, paths[-1, p]), radius=0.06, color=SCORE).set_z_index(5) for p in picks])

        self.say(
            "[Now run the loop on the swirl]: [two thousand points of pure noise], [one round Gaussian blob].",
            m.AnimationGroup(self.new_header("03 · the swirl", "The swirl un-melts"), m.FadeIn(counter)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(d, scale=0.5) for d in swarm.cloud], lag_ratio=0.0015, run_time=1.4),
                             m.FadeIn(setting)),
            m.Create(swarm.ring),
        )
        self.say(
            "[For the first five hundred steps, the blob hardly changes].",
            swarm.replay(1000, 500, 3.4),
        )
        self.say(
            "[Then it tightens], [the two arms appear], [and in the last fifty steps they snap sharp].",
            swarm.replay(500, 200, 0.8),
            swarm.replay(200, 50, 0.8),
            swarm.replay(50, 0, 0.8),
        )
        self.say(
            "[Follow six points]: [each path wanders, kicked by fresh noise], [yet ends on an arm].",
            swarm.cloud.animate(run_time=0.8).set_opacity(0.35),
            m.LaggedStart(*[m.Create(tr) for tr in traces], lag_ratio=0.15, run_time=2.4),
            m.FadeIn(heads, scale=0.5),
        )
        self.say(
            "[With the exact denoiser, the samples sit zero point zero two four five from the curve on average]; "
            "[the data itself, zero point zero two four].",
            m.AnimationGroup(m.FadeOut(traces), m.FadeOut(heads), m.FadeOut(setting),
                             swarm.cloud.animate.set_opacity(0.95),
                             m.FadeIn(box), m.FadeIn(head), m.FadeIn(rows[1], shift=0.1 * m.UP)),
            m.FadeIn(rows[0], shift=0.1 * m.UP),
        )
        self.say(
            "[Swap in the small trained network], [and they sit zero point zero two five three away]: [almost as close].",
            swarm.cloud.to(RUNS["mlp_final"][:len(swarm.cloud)], run_time=1.6),
            m.AnimationGroup(m.FadeIn(rows[2], shift=0.1 * m.UP), m.FadeIn(mlp_src)),
            m.Circumscribe(m.VGroup(rows[1], rows[2]), color=MODEL, buff=0.1),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 4 · a digit emerges

    def digit_emerges(self):
        self.next_section("4 · A seven, a thousand steps", notes=(
            f"The 7 generated on this Mac by the digit model ({DIGIT_PARAMS:,}-parameter U-Net trained on MNIST, "
            f"mac_runs.md §3): DDPM ancestral sampling, 1,000 steps, σ² = β̃, w = 1, label 7, starting noise "
            f"np.random.default_rng({SEED}). Left: x_t (clipped to −1 … 1); right: the network's clean guess "
            "x̂₀ = (x_t − √(1 − ᾱ_t)·ε̂)/√ᾱ_t (AMBER, clipped). Correlation with the final 7: x_t "
            f"{C_XT[600]:.2f} at t = 600, {C_XT[400]:.2f} at 400, {C_XT[100]:.2f} at 100; x̂₀ {C_X0[800]:.2f} at 800, "
            f"{C_X0[450]:.2f} at 450 (the guess settles early, the picture emerges late). t is the paper's (the run's "
            "code t + 1); at t = 1000 x̂₀ is speckled (÷ √ᾱ ≈ 0.0064, Episode 5)."))
        H = 3.0
        xt = Flip(lambda k: state_rgb(STATES[k], 1000 - k), 0, H)
        x0 = Flip(lambda k: guess_rgb(GUESSES[k]), 0, H)
        left = framed(xt.image).move_to([-4.35, -0.4, 0])
        right = framed(x0.image, MODEL).move_to([-0.6, -0.4, 0])
        l_lab = m.VGroup(tex("x_t", 36), text("what the sampler holds", 22, MUTED)).arrange(m.RIGHT, buff=0.2)
        r_lab = m.VGroup(tex(r"\hat x_0", 36, MODEL), text("the network's guess", 22, MUTED)).arrange(m.RIGHT, buff=0.2)
        l_lab.next_to(left, m.UP, buff=0.25)
        r_lab.next_to(right, m.UP, buff=0.25)
        r_lab.align_to(l_lab, m.DOWN)
        counter = Readout(2.35, 1.35, "t = {:d}", 1000, 44)
        prog = Progress((4.45, 0.55, 0), span=4.2)
        prog_lab = code("steps done", 20, MUTED).next_to(prog, m.DOWN, buff=0.16).align_to(prog, m.LEFT)
        ask = pill("label: 7", PROMPT, PINK_FILL, 22, mono=True).move_to([3.15, -0.75, 0])
        cap = cite(f"generated on this Mac · digit model,\n{DIGIT_PARAMS / 1e6:.1f} M-parameter U-Net ·\n"
                   f"DDPM, 1,000 steps · seed {SEED}").move_to([4.45, -2.25, 0])

        def at_t(t):
            k = 1000 - int(round(t))
            xt.show(k)
            x0.show(k)
            counter.show(int(round(t)))
            prog.show(k / 1000)

        self.say(
            "[Now a real digit, from the model we trained on this Mac]: [ask for a seven, and start from noise].",
            m.AnimationGroup(self.new_header("04 · a digit", "A seven in a thousand steps"), m.FadeIn(cap)),
            m.AnimationGroup(m.FadeIn(ask, shift=0.1 * m.UP), m.FadeIn(left, scale=0.92), m.FadeIn(counter),
                             m.FadeIn(prog), m.FadeIn(prog_lab)),
        )
        self.say(
            "[On the left, what the sampler holds]. [On the right, the network's guess of the finished picture].",
            m.FadeIn(l_lab, shift=0.1 * m.DOWN),
            m.AnimationGroup(m.FadeIn(right, target_position=left), m.FadeIn(r_lab, shift=0.1 * m.DOWN)),
        )
        self.say(
            "[For hundreds of steps the left is still static], [but the guess already shows a blurry seven].",
            drive(left, 1000, 750, [at_t], run_time=1.0, rate=m.linear),
            drive(left, 750, 560, [at_t], run_time=1.0, rate=m.linear),
        )
        self.say(
            "[Around step four hundred, the stroke shows through], [and the guess turns sharp].",
            drive(left, 560, 400, [at_t], run_time=1.0, rate=m.linear),
            drive(left, 400, 250, [at_t], run_time=1.0, rate=m.linear),
        )
        self.say(
            "[The last steps only polish the edges]. [At step zero: a clean seven].",
            drive(left, 250, 0, [at_t], run_time=1.2, rate=m.smooth),
            m.Circumscribe(left, color=DATA, buff=0.06),
        )
        self.wait(0.3)
        ts = [1000, 800, 600, 450, 300, 200, 100, 0]
        h = 1.05
        top = m.Group(*[framed(picture(state_rgb(STATES[1000 - t], t), h)) for t in ts]).arrange(m.RIGHT, buff=0.2)
        bot = m.Group(*[framed(picture(guess_rgb(GUESSES[1000 - t]), h), MODEL) for t in ts]).arrange(m.RIGHT, buff=0.2)
        grid = m.Group(top, bot).arrange(m.DOWN, buff=0.55).move_to([0.45, -0.05, 0])
        t_labs = m.VGroup(*[code(f"{t}", 20, TIME).next_to(c, m.UP, buff=0.14) for t, c in zip(ts, top)])
        t_head = code("t =", 20, TIME).next_to(t_labs[0], m.LEFT, buff=0.3)
        r1 = tex("x_t", 34).next_to(top, m.LEFT, buff=0.35)
        r2 = tex(r"\hat x_0", 34, MODEL).next_to(bot, m.LEFT, buff=0.35)
        late = text("the picture emerges late", 22, INK).next_to(top, m.DOWN, buff=0.1).align_to(top, m.RIGHT)
        early = text("the guess settles early", 22, MODEL).next_to(bot, m.DOWN, buff=0.16).align_to(bot, m.LEFT)
        strip_cap = cite(f"generated on this Mac · digit model · DDPM, 1,000 steps · seed {SEED} · label 7")
        strip_cap.move_to([0.45, -3.3, 0])
        self.say(
            "[In a row]: [the picture emerges late], [while the guess settles early].",
            m.AnimationGroup(m.FadeOut(m.Group(left, right, l_lab, r_lab, counter, prog, prog_lab, ask, cap)),
                             m.FadeIn(m.Group(top, bot), lag_ratio=0.05), m.FadeIn(t_labs), m.FadeIn(t_head),
                             m.FadeIn(r1), m.FadeIn(r2), m.FadeIn(strip_cap)),
            m.FadeIn(late, shift=0.1 * m.UP),
            m.FadeIn(early, shift=0.1 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 5 · why fresh noise

    def why_fresh_noise(self):
        self.next_section("5 · Why fresh noise?", notes=(
            "The same 2,000 noise points (first 1,000 drawn) and the exact ε, with DDPM's mean step only — the fresh "
            "noise σ_t·z deleted (precompute.py, nonoise). Not a derived sampler (notes §7): every point slides inward "
            f"and ends within 0.2 of the centre, at the arms' inner tips; swirl_coverage = {NN_COVER:.2f} (data: 1.0). "
            "Each step's mean is a posterior average, and averaging shrinks the spread; the fresh noise restores the "
            "variance each step is derived for. DDPM §3.2: Algorithm 2 'resembles Langevin dynamics' (Episode 6)."))
        plot = plane()
        guide = m.VGroup(*[m.VMobject().set_points_smoothly([at(plot, p) for p in arm[::3]] + [at(plot, arm[-1])])
                           .set_stroke(DATA, 14, opacity=0.2) for arm in swirl_curve(201)]).set_z_index(-1)
        swarm = Swarm(plot, NN_STATES, NN_DIST)
        g_scale = {"s": 1.0}

        def regrid(i):          # the guide (drawn at half-width 1.5) follows the zoom
            s_ = 1.5 / (plot.x1 - plot.x0) * 2
            guide.scale(s_ / g_scale["s"], about_point=PC)
            g_scale["s"] = s_

        regrid(0)
        swarm.hooks.append(regrid)
        f = step_formula(34).move_to([RX, 1.85, 0])
        gone = text("deleted: no fresh noise", 22, RED, weight=m.BOLD).next_to(f, m.DOWN, buff=0.3)
        cover = m.VGroup(title_text(f"{100 * NN_COVER:.0f}%", 54, RED), text("of the swirl covered", 22, MUTED))
        cover.arrange(m.DOWN, buff=0.12).move_to([RX, -0.35, 0])
        why1 = text("each step moves to an average guess,\nand averaging shrinks the spread", 22, INK)
        why2 = text("the fresh noise puts the spread back", 22, GREEN, weight=m.BOLD)
        why = m.VGroup(why1, why2).arrange(m.DOWN, buff=0.25).move_to([RX, -2.25, 0])
        src = cite("Ho et al., 2020 §3.2 · like Langevin dynamics, Episode 6").move_to([RX, -3.35, 0])
        tips = m.VGroup(*[m.Circle(radius=0.32).set_stroke(RED, 3).move_to(at(plot, SWIRL_R0 * np.array([s, 0])))
                          for s in (1, -1)])

        self.say(
            "[Why add noise back], [when the goal is to remove it]? [Delete that line and run the swirl again].",
            m.AnimationGroup(self.new_header("05 · fresh noise", "Why add noise back?"), m.FadeIn(f)),
            m.AnimationGroup(f[9:].animate.set_color(RED).set_opacity(0.35), m.FadeIn(gone)),
            m.AnimationGroup(m.FadeIn(guide), m.Create(swarm.ring),
                             m.LaggedStart(*[m.FadeIn(d, scale=0.5) for d in swarm.cloud], lag_ratio=0.0015, run_time=1.4)),
        )
        self.say(
            "[Every point slides inward], [and piles up at the two inner tips of the arms].",
            swarm.replay(1000, 150, 1.0, rate=m.smooth),
            m.AnimationGroup(swarm.replay(150, 0, 1.0, rate=m.smooth), m.FadeIn(cover, shift=0.1 * m.UP)),
        )
        self.say(
            "[Each step moves toward an average guess], [and averaging shrinks the spread]. [The fresh noise puts it back].",
            m.AnimationGroup(m.Create(tips), m.FadeIn(why1)),
            m.Circumscribe(why1, color=INK, buff=0.08),
            m.AnimationGroup(m.FadeIn(why2, shift=0.1 * m.UP), m.FadeIn(src)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 6 · it forgets where it started

    def it_forgets(self):
        self.next_section("6 · It forgets where it started", notes=(
            f"Five hundred DDPM chains (exact ε, σ² = β, seed 3) from ONE starting point, z[115] = ({START[0]:.2f}, "
            f"{START[1]:.2f}) of toy.md's noise: they end covering the whole swirl (coverage {OS_COVER:.1f}, "
            f"{100 * OS_ARM0:.1f}% on arm 1). Measured on 2,000 points with the trained MLP (toy.json samplers/mlp/"
            f"determinism): two runs from the same noise (per-step seeds 1 and 2) end on the same arm {100 * SAME_ARM:.1f}% "
            f"of the time, {GAP:.4f} apart on average; two random swirl samples are {RAND_GAP:.4f} apart (exact ε: "
            f"{100 * DET['ddpm_seed1_vs_seed2_same_arm']:.1f}%, 0.9002). The digit pair: the same starting noise (seed "
            f"{SEED}) through DDPM 1,000 (ours) and DDIM 50 steps, η = 0 (Episode 9) — a different 7."))
        plot = plane()
        swarm = Swarm(plot, OS_STATES, OS_DIST, radius=0.03)
        start = m.VGroup(m.Circle(radius=0.2).set_stroke(NOISE, 3), m.Dot(radius=0.08, color=NOISE))
        start.move_to(at(plot, START)).set_z_index(6)

        def follow(i):
            start.move_to(at(plot, START))

        swarm.hooks.append(follow)
        counter = Readout(LEFT + 0.5, 2.0, "t = {:d}", 1000, 40)
        swarm.hooks.append(lambda i: counter.show(int(round(swarm.t_of(i)))))
        runs = text("500 runs, one starting point", 24, INK).move_to([RX, 1.25, 0])

        box = card(5.2, 3.0, EDGE, CARD, radius=0.18, stroke=2).move_to([RX, -0.55, 0])
        b_head = text("two runs from the same noise", 22, MUTED).move_to(box.get_top() + [0, -0.38, 0])
        pct = title_text(f"{100 * SAME_ARM:.1f}%", 50, INK)
        pct_l = text("on the same arm", 24, INK)
        r_a = m.VGroup(pct, pct_l).arrange(m.RIGHT, buff=0.25).move_to(box.get_top() + [0, -1.05, 0])
        sit(pct_l, baseline(pct))
        coin = code("a coin flip: 50%", 20, MUTED).next_to(r_a, m.DOWN, buff=0.2)
        gap = m.VGroup(code(f"{GAP:.2f} apart", 24, INK, weight=m.BOLD), code(f"two random points: {RAND_GAP:.2f}", 20, MUTED))
        gap.arrange(m.DOWN, buff=0.14).move_to(box.get_bottom() + [0, 0.62, 0])
        stat_src = cite("trained network · 2,000 points · toy.md §3").next_to(box, m.DOWN, buff=0.18)

        hp = 1.45
        same = framed(picture(state_rgb(STATES[0], 1000), hp)).move_to([LEFT + 0.95, -1.25, 0])
        ours = framed(picture(state_rgb(FINAL, 0), hp), DATA).move_to([5.55, -0.25, 0])
        ep9 = framed(picture(state_rgb(DDIM_SAME, 0), hp), TIME).move_to([5.55, -2.45, 0])
        a1 = arrow(same.get_right() + [0, 0.25, 0], ours.get_left() + [0, -0.2, 0], DATA)
        a2 = arrow(same.get_right() + [0, -0.25, 0], ep9.get_left() + [0, 0.2, 0], TIME)
        l_same = code(f"same noise · seed {SEED}", 20, MUTED).next_to(same, m.DOWN, buff=0.16)
        l_ours = text("with fresh noise: ours", 20, DATA).next_to(ours, m.LEFT, buff=0.25).shift(0.55 * m.UP)
        l_ep9 = text("without (Episode 9)", 20, TIME).next_to(ep9, m.LEFT, buff=0.25).shift(0.55 * m.DOWN)

        self.say(
            "[With the noise in], [start five hundred runs from one single point].",
            m.AnimationGroup(self.new_header("06 · it forgets", "Same start, anywhere"), m.FadeIn(start, scale=0.6),
                             m.Create(swarm.ring)),
            m.AnimationGroup(m.FadeIn(swarm.cloud), m.FadeIn(runs), m.FadeIn(counter)),
        )
        self.say(
            "[Watch them spread out], [and settle all over the swirl, on both arms].",
            swarm.replay(1000, 300, 1.0, rate=m.smooth),
            swarm.replay(300, 0, 1.0, rate=m.smooth),
        )
        self.say(
            "[Two runs from the same starting noise land on the same arm only forty-nine point nine percent of the time].",
            m.AnimationGroup(m.FadeOut(runs), m.FadeOut(counter), m.FadeIn(box), m.FadeIn(b_head),
                             m.FadeIn(r_a, shift=0.1 * m.UP), m.FadeIn(coin), m.FadeIn(stat_src)),
        )
        self.say(
            "[They end zero point nine apart, as far as two random points]: [the sampler forgets where it started].",
            m.FadeIn(gap, shift=0.1 * m.UP),
            m.Circumscribe(start, color=NOISE, shape=m.Circle, buff=0.1),
        )
        self.say(
            "[Episode nine drops the fresh noise the right way]: [then the same noise always gives the same picture].",
            m.AnimationGroup(m.FadeOut(m.VGroup(box, b_head, r_a, coin, gap, stat_src)), m.FadeIn(same), m.FadeIn(l_same),
                             m.GrowArrow(a1), m.FadeIn(ours), m.FadeIn(l_ours)),
            m.AnimationGroup(m.GrowArrow(a2), m.FadeIn(ep9, shift=0.25 * m.LEFT), m.FadeIn(l_ep9)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 7 · the cost

    def the_cost(self):
        self.next_section("7 · The cost", notes=(
            f"One network call per step: 1,000 calls per sample (mac_runs.json digit_sampling nfe_per_sample). Measured "
            f"on this Mac (M4 Max, MLX): one digit alone {S_ONE:.2f} s (latency, batch 1); {S_PER_1000:.1f} s per 1,000 "
            f"digits in batches of 1,000 (mean of batches 2–10). The Mac was shared: ±5–10% jitter (mac_runs.md §8). "
            f"Quality: cFID {CFID:.3f} vs the real-data floor {FLOOR:.3f} (10,000 vs 10,000; bootstrap std ≈ 0.3) — the "
            "FID formula on our own MNIST classifier's 128-d features, NOT Inception FID. The six sevens: seeds "
            f"{', '.join(map(str, SEEDS))}, each from its own starting noise (np.random.default_rng(seed)), DDPM 1,000 "
            "steps. The batch: arrays/digit_trajectory_7.npz many20 (20 digits, labels 0–9 twice, 40 recorded frames)."))
        cols, rows_n = 40, 25
        cell, gap_ = 0.13, 0.035
        grid = m.VGroup(*[m.Square(cell).set_stroke(width=0).set_fill(EDGE, 1) for _ in range(cols * rows_n)])
        grid.arrange_in_grid(rows=rows_n, cols=cols, buff=gap_).move_to([-2.6, -0.55, 0])
        counter = Readout(2.3, 1.2, "{:,}", 0, 54, MODEL)
        c_lab = text("network calls for one picture", 22, MUTED).next_to(counter, m.DOWN, buff=0.2).align_to(counter, m.LEFT)
        c_lab.set_x(2.3 + c_lab.width / 2)
        unet = cite(f"each: the {DIGIT_PARAMS / 1e6:.1f} M-parameter U-Net, once").move_to([4.35, -0.3, 0])
        lit = {"n": 0}

        def fill(_, a):
            n = int(round(a * cols * rows_n))
            for k in range(lit["n"], n):
                grid[k].set_fill(MODEL, 1)
            lit["n"] = max(lit["n"], n)
            counter.show(n)

        # six sevens, and a batch of twenty emerging together
        sevens = m.Group(*[framed(picture(state_rgb(TRAJ[f"seed{s}_ddpm_ddpm1000_w1_final"].astype(float), 0), 0.85))
                           for s in SEEDS]).arrange(m.RIGHT, buff=0.18).move_to([-2.6, 1.95, 0])
        sev_cap = cite("six starting noises, six sevens · 1,000 calls each").next_to(sevens, m.DOWN, buff=0.16)
        batch_flips = [Flip(lambda k, d=d: state_rgb(MANY_XT[k, d] if k < 40 else MANY_FINAL[d], 0 if k >= 40 else MANY_T[k]),
                            0, 0.6) for d in range(20)]
        batch = m.Group(*[framed(fl.image, buff=0.04, stroke=2) for fl in batch_flips])
        batch.arrange_in_grid(rows=4, cols=5, buff=0.12).move_to([-2.6, -0.8, 0])
        batch_cap = cite("a batch: twenty digits, one thousand steps together").next_to(batch, m.DOWN, buff=0.16)

        def run_batch(_, a):
            k = int(np.floor(a * 40.999))
            for fl in batch_flips:
                fl.show(k)

        s1 = stat(f"{S_ONE:.1f}", "s", "one seven, on its own", INK, 60)
        s2 = stat(f"{S_PER_1000:.0f}", "s", "a thousand, in batches", INK, 60)
        stats = m.VGroup(s1, s2).arrange(m.DOWN, buff=0.35, aligned_edge=m.LEFT).move_to([4.2, 1.3, 0])
        mac = cite("measured on this Mac · M4 Max").next_to(stats, m.DOWN, buff=0.22).align_to(stats, m.LEFT)

        bars = Bars([FLOOR, CFID], labels=["real digits", "ours"], top=3.0, width=3.0, height=1.15, colors=[DATA, MODEL],
                    gap=0.5, fmt="{:.2f}", center=(4.3, -2.05, 0))
        b_lab = text("classifier-based score · lower is better", 20, MUTED).next_to(bars, m.UP, buff=0.18)
        not_fid = cite("our classifier's features, not Inception FID").next_to(bars, m.DOWN, buff=0.12)
        module3 = pill("Module 3 · Faster Sampling", TIME, BLUE_FILL, 24).move_to([-2.6, -3.15, 0])
        assert module3.get_top()[1] < batch.get_bottom()[1] - 0.1, (module3.get_top(), batch.get_bottom())

        self.say(
            "[Now the price]: [every step is a full pass through the network], [so one picture costs a thousand calls].",
            m.AnimationGroup(self.new_header("07 · the cost", "A thousand calls per picture"), m.FadeIn(grid),
                             m.FadeIn(counter), m.FadeIn(c_lab)),
            m.FadeIn(unet, shift=0.1 * m.UP),
            m.UpdateFromAlphaFunc(grid, fill, run_time=3.0, rate_func=m.linear),
        )
        self.say(
            "[On this Mac, one seven takes four point three seconds]. [In batches, a thousand take a hundred seventy-two].",
            m.AnimationGroup(m.FadeOut(m.VGroup(grid, counter, c_lab, unet)), m.FadeIn(sevens, lag_ratio=0.1),
                             m.FadeIn(sev_cap), m.FadeIn(s1, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(batch), m.FadeIn(batch_cap), m.FadeIn(s2, shift=0.1 * m.UP), m.FadeIn(mac),
                             m.UpdateFromAlphaFunc(batch, run_batch, run_time=3.2, rate_func=m.linear)),
        )
        self.say(
            "[And they are good]: [a classifier-based score of two point six four, where real digits score two point zero six].",
            m.AnimationGroup(m.FadeIn(bars.zero), m.FadeIn(b_lab), m.FadeIn(not_fid)),
            m.AnimationGroup(*[m.GrowFromEdge(b, m.DOWN) for b in bars.bars], m.FadeIn(bars.labels), m.FadeIn(bars.numbers)),
        )
        self.say(
            "[A thousand calls per picture was diffusion's first big weakness]. [Module three is about cutting it down].",
            m.Circumscribe(stats, color=MODEL, buff=0.12),
            m.AnimationGroup(m.FadeOut(batch_cap), m.FadeIn(module3, shift=0.1 * m.UP)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes=(
            "Predict (the network's ε̂), step (the mean step back, plus fresh noise σ_t·z except on the last step), repeat "
            f"(1,000 times). The strip: the seed-{SEED} seven at t = 1000, 600, 400, 200, 0. Next: Episode 8, The U-Net."))
        halo = glow(MODEL, radius=4.0)
        words = m.VGroup(*[title_text(w, 58, c) for w, c in (("Predict", MODEL), ("Step", DATA), ("Repeat", TIME))])
        subs = m.VGroup(*[code(s, 20, MUTED) for s in ("the noise", "back, plus fresh noise", "1,000 times")])
        words.arrange(m.RIGHT, buff=1.15)
        on_baseline(words)
        for w, s in zip(words, subs):
            s.set_x(w.get_x())
        line_y = min(w.get_bottom()[1] for w in words) - 0.45
        for s in subs:
            sit(s, line_y)
        mid = baseline(words[0]) + 0.3
        dots = m.VGroup(*[m.Dot([(l.get_right()[0] + r.get_left()[0]) / 2, mid, 0], radius=0.07, color=MUTED)
                          for l, r in zip(words, words[1:])])
        row = m.VGroup(words, dots, subs).move_to(1.35 * m.UP)
        assert row.width < 12.6 and all(a.get_right()[0] + 0.3 < b.get_left()[0] for a, b in zip(subs, subs[1:]))
        ts = [1000, 600, 400, 200, 0]
        strip_ = m.Group(*[framed(picture(state_rgb(STATES[1000 - t], t), 1.05)) for t in ts]).arrange(m.RIGHT, buff=0.5)
        strip_.next_to(row, m.DOWN, buff=0.55)
        arrows = m.VGroup(*[arrow(a.get_right(), b.get_left(), MUTED, buff=0.08) for a, b in zip(strip_, strip_[1:])])
        tag = text("From static to a seven, one small noisy step at a time.", 28, MUTED).next_to(strip_, m.DOWN, buff=0.45)
        mascot = pixel(0.42)
        nxt = pill("NEXT · THE U-NET", MODEL, AMBER_FILL, 20, mono=True)
        foot = m.VGroup(mascot, nxt).arrange(m.RIGHT, buff=0.3).next_to(tag, m.DOWN, buff=0.35)

        self.say(
            "[To sum up]: [predict], [step], [repeat].",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.3 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(dots[0]), m.FadeIn(words[1], shift=0.3 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(dots[1]), m.FadeIn(words[2], shift=0.3 * m.UP), m.FadeIn(subs[2])),
        )
        self.say(
            "[Predict the noise, step back, add a little fresh noise]: [a thousand times, from static to a seven].",
            m.LaggedStart(*[m.FadeIn(p, shift=0.1 * m.RIGHT) for p in strip_], lag_ratio=0.25, run_time=1.6),
            m.AnimationGroup(m.LaggedStart(*[m.GrowArrow(a) for a in arrows], lag_ratio=0.2), m.Write(tag)),
        )
        self.say(
            "[Next, the network that makes every one of those predictions].",
            m.AnimationGroup(m.FadeIn(mascot, shift=0.2 * m.UP), m.FadeIn(nxt, shift=0.2 * m.UP)),
        )
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, strip_, arrows, tag, foot)], run_time=0.9)
