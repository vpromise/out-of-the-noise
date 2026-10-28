"""Out of the Noise · Episode 9 · Fewer Steps (Module 3 · Faster Sampling)

The same trained model can sample deterministically in far fewer steps: DDIM (Song, Meng & Ermon, arXiv Oct 2020,
ICLR 2021) with η = 0. Running example: the digit model (research/experiments/mac_runs.md), one starting noise sampled
with Episode 7's 1,000-step ancestral sampler and with DDIM 50 and 10 (arrays/digit_trajectory_7.npz, noise
np.random.default_rng(7)); then the swirl (diffusion_reference + the trained swirl MLP) for the mechanism.

The episode: a real DDIM step of that 7 (x̂₀ and ε̂ recovered from the saved arrays; every stored step obeys
x_s = √ᾱ_s·x̂₀ + √(1 − ᾱ_s)·ε̂ to float16 precision, asserted); skipping (10 hops); the swirl: same noise → same point
(toy.md §3: DDIM-100 vs DDIM-50 same arm 97.2%, two DDPM runs 49.9%; recomputed by precompute.py and asserted); the
probability-flow ODE (Episode 6's twin) and the identities DDIM(η = 0) = Euler in σ, DDIM(η = 1) = DDPM with β̃
(computed below, max difference ~1e-15); the digit model's same-seed grid at 1,000 / 100 / 50 / 20 / 10 / 5 / 2 / 1
steps and the measured cFID vs time per 1,000 digits (mac_runs.json; cFID = FID's formula on our own MNIST
classifier's features, NOT Inception FID; floor 2.06; bootstrap error ≈ 0.3); DDIM's own CIFAR-10 table (notes §9);
the price on the swirl (stranded 2.6 / 7.5 / 31 / 45 % at 100 / 50 / 20 / 10 steps; DDIM-100 "too tidy": 0.020 vs the
data's 0.024); inversion (deterministic = invertible; Episode 16 uses it).

Every number on screen is computed below (diffusion_reference.py, the run's arrays, precompute.py's runs) and asserted,
or loaded with measured(...), or cited with its source.

Sources (research/notes.md §9, §6, Appendix A rows 9, Appendix C row 9; read October 2026):
- Song, Meng & Ermon (DDIM, arXiv:2010.02502, 6 Oct 2020; ICLR 2021): eq. 12 (the update; "predicted x0", "direction
  pointing to xt", "random noise"), eq. 16 (η), eq. 14 + Proposition 1 (an ODE; Euler in σ̄), Table 1 (CIFAR-10 FID by
  η and steps), Table 2 (reconstruction error), §5.2 (consistency).
- Song et al. 2021 (score SDE, ICLR 2021): eq. 13, the probability-flow ODE ("same marginal probability densities").
- research/experiments/toy.md §3 and toy.json (samplers/mlp: runs, determinism; headline ddim*_mlp_off, data_dist,
  ddim100_mlp_dist, ddpm1000_mlp_dist).
- research/experiments/mac_runs.md §4 and mac_runs.json (digit_sampling configs, headline cfid_*, seconds_per_1000_*,
  cfid_uncertainty). The Mac was shared while timing: ±5–10% jitter (mac_runs.md §8).

Illustrative (computed, but chosen): the step shown (t = 556 → 445 of DDIM-10); the six traced swirl particles and the
three stranded ones (chosen by precompute.py's stated rules); the twelve inverted points; digits 10–19 of the grid.

    ~/.local/share/uv/tools/manimgx/bin/python precompute.py   # once (~12 s): writes swirl_runs.npz
    VOICE=draft manimgx inspect scene.py                       # silent draft
    manimgx render scene.py                                    # with the narrator serving from the series folder
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

PREFIX = "ep09"
ARR = EXPERIMENTS / "arrays"
MAC = lambda *keys: measured("mac_runs", "headline", *keys)  # noqa: E731
CONF = lambda c, *keys: measured("mac_runs", "digit_sampling", "configs", f"ddpm_{c}_w1", *keys)  # noqa: E731
TOY = lambda *keys: measured("toy", "headline", *keys)  # noqa: E731
TOYM = lambda *keys: measured("toy", "samplers", "mlp", *keys)  # noqa: E731
AB = ALPHA_BAR
assert T == 1000


def corr(a, b):
    return float(np.corrcoef(np.ravel(a), np.ravel(b))[0, 1])


# ---------------------------------------------------------------------------- one starting noise, three samplers (the 7)

TR7 = np.load(ARR / "digit_trajectory_7.npz")
assert int(TR7["label"]) == 7
XT7 = TR7["xT"].astype(np.float64)
_start = np.random.default_rng(7).standard_normal((1, 28, 28, 1), dtype=np.float32)[0, ..., 0]
assert np.allclose(XT7, _start.astype(np.float16))                       # "seed 7"
SEVEN = {}
for _key, _n in (("ddpm_ddpm1000_w1", 1000), ("ddpm_ddim50_w1", 50), ("ddpm_ddim10_w1", 10)):
    _code_t = TR7[f"{_key}_t"].astype(int)
    assert np.array_equal(_code_t, np.round(np.linspace(999, 0, _n)).astype(int))   # S steps, t = 999 … 0 (code)
    _xs = np.concatenate([TR7[f"{_key}_x_t"].astype(np.float64), TR7[f"{_key}_final"].astype(np.float64)[None]])
    assert np.allclose(_xs[0], XT7)                                       # every row starts from the same noise
    SEVEN[_n] = dict(xs=_xs, x0s=TR7[f"{_key}_x0_hat"].astype(np.float64),
                     ts=np.concatenate([_code_t + 1, [0]]))                # the paper's t (code t + 1); 0 = the output
assert corr(SEVEN[50]["xs"][-1], SEVEN[10]["xs"][-1]) > 0.99              # "the very same seven"
assert corr(SEVEN[1000]["xs"][-1], SEVEN[50]["xs"][-1]) < 0.5             # the noisy sampler drew another one

# one real DDIM step of that 7, taken apart: step k = 4 of DDIM-10 (paper t = 556 → 445)
D10 = SEVEN[10]
STEP_K = 4
T_FROM, T_TO = int(D10["ts"][STEP_K]), int(D10["ts"][STEP_K + 1])
assert (T_FROM, T_TO) == (556, 445)


def _parts(run, k):
    ab_t, ab_s = AB[run["ts"][k]], AB[run["ts"][k + 1]]
    eps = (run["xs"][k] - np.sqrt(ab_t) * run["x0s"][k]) / np.sqrt(1 - ab_t)   # x̂₀ = (x_t − √(1 − ᾱ)·ε̂)/√ᾱ, inverted
    return eps, np.sqrt(ab_s), np.sqrt(1 - ab_s)


for _k in range(1, 9):                                                     # step 0's x̂₀ was clipped to ±3 when saved
    assert np.abs(D10["x0s"][_k]).max() < 2.99
    _eps, _a, _b = _parts(D10, _k)
    assert np.abs(_a * D10["x0s"][_k] + _b * _eps - D10["xs"][_k + 1]).max() < 0.01   # x_s = √ᾱ_s·x̂₀ + √(1 − ᾱ_s)·ε̂
EPS_HAT, C_X0, C_EPS = _parts(D10, STEP_K)
assert abs(EPS_HAT.std() - 1) < 0.05 and f"{C_X0:.3f}" == "0.365" and f"{C_EPS:.3f}" == "0.931"

T_PER_1000 = {n: CONF("ddpm1000" if n == 1000 else f"ddim{n}", "s_per_1000") for n in (1000, 100, 50, 20, 10, 5, 2, 1)}
assert MAC("seconds_per_1000_digits_ddpm1000") == round(T_PER_1000[1000], 2) == 172.11
assert f"{T_PER_1000[50]:.1f}" == "8.9" and f"{T_PER_1000[10]:.1f}" == "1.8"
# speed-ups as mac_runs.md states them (its headline seconds): 172.11 / 8.87 and 172.11 / 1.78
SPEED50 = MAC("seconds_per_1000_digits_ddpm1000") / MAC("seconds_per_1000_digits_ddim50")
SPEED10 = MAC("seconds_per_1000_digits_ddpm1000") / MAC("seconds_per_1000_digits_ddim10")
assert round(SPEED50) == 19 and round(SPEED10) == 97                       # "nineteen", "almost a hundred times faster"
assert 96 < T_PER_1000[1000] / T_PER_1000[10] < 97                        # unrounded seconds: 96.5×
DIGIT_PARAMS = MAC("digit_unet_params")
assert f"{DIGIT_PARAMS / 1e6:.1f}" == "3.9"

# ---------------------------------------------------------------------------- the same-seed grid and its scores

ROWS = (1000, 100, 50, 20, 10, 5, 2, 1)
GRID_IDX = list(range(10, 20))                                             # samples 10 … 19 ask for labels 0 … 9
SAMPLES = {}
for _n in ROWS:
    _f = np.load(ARR / f"digit_samples_ddpm_{'ddpm1000' if _n == 1000 else f'ddim{_n}'}_w1.npz")
    assert "seed 1234" in str(_f["note"]) and list(_f["labels"][GRID_IDX]) == list(range(10))
    SAMPLES[_n] = _f["samples"]
_sim = lambda a, b: float(np.median([corr(SAMPLES[a][i], SAMPLES[b][i]) for i in range(1000)]))  # noqa: E731
assert _sim(10, 100) > 0.98 and _sim(1000, 100) < 0.6                     # same noise → same digits (DDIM only)
SQRT_AB_T = float(np.sqrt(AB[T]))
assert f"{SQRT_AB_T:.4f}" == "0.0064" and round(1 / SQRT_AB_T) == 157     # "a hundred fifty-seven times"

CFID = {n: CONF("ddpm1000" if n == 1000 else f"ddim{n}", "cfid") for n in ROWS}
assert [MAC("cfid_ddpm1000")] + [MAC(f"cfid_ddim{n}") for n in ROWS[1:]] == [round(CFID[n], 3) for n in ROWS]
CFID_TXT = {n: (f"{CFID[n]:.2f}" if CFID[n] < 10 else f"{CFID[n]:.1f}" if CFID[n] < 100 else f"{CFID[n]:.0f}") for n in ROWS}
assert [CFID_TXT[n] for n in ROWS] == ["2.64", "3.50", "3.35", "3.30", "4.57", "21.6", "939", "943"]
FLOOR = MAC("cfid_real_data_floor")
assert f"{FLOOR:.2f}" == "2.06"
_unc = measured("mac_runs", "cfid_uncertainty")
BOOT = [_unc[f"ddpm_ddim{n}_w1"]["bootstrap_std"] for n in (100, 50, 20)]
assert all(0.25 < b < 0.4 for b in BOOT) and max(CFID[n] for n in (100, 50, 20)) - min(CFID[n] for n in (100, 50, 20)) < 0.3

# DDIM's own table (notes §9, Table 1 of the paper: CIFAR-10, Inception FID; rows η, columns steps)
CIFAR_STEPS = (10, 20, 50, 100, 1000)
CIFAR = {0: (13.36, 6.84, 4.67, 4.16, 4.04), 1: (41.07, 18.36, 8.01, 5.78, 4.73)}
CIFAR_DDPM_1000 = 3.17
_notes = (EP_DIR.parent / "research" / "notes.md").read_text()
assert "| η = 0.0 (DDIM) | **13.36** | **6.84** | **4.67** | **4.16** | 4.04 |" in _notes
assert "| η = 1.0 (DDPM, β̃) | 41.07 | 18.36 | 8.01 | 5.78 | 4.73 |" in _notes and "**3.17**" in _notes

# ---------------------------------------------------------------------------- the swirl (precompute.py, toy.md §3)

MLP = load_swirl_model()
MLP_PARAMS = TOY("mlp_params")
assert MLP_PARAMS == count_params(load_weights()) == 45954
RUNS = np.load(EP_DIR / "swirl_runs.npz")
assert np.array_equal(RUNS["z"], gaussian_noise(2000, seed=5))
OFF = {s: off_swirl(RUNS[f"ddim{s}_final"]) for s in (100, 50, 20, 10)}
for _s in OFF:
    assert abs(OFF[_s] - TOYM("runs", f"ddim_{_s}", "off")) < 2e-3
assert [f"{100 * OFF[s]:.1f}" for s in (100, 50, 20, 10)] == ["2.6", "7.5", "31.2", "44.9"]
assert TOY("ddim50_mlp_off") == 0.075 and TOY("ddim10_mlp_off") == 0.449
DIST = dict(data=TOY("data_dist"), ddpm=swirl_distance(RUNS["ddpm1_final"]), ddim100=swirl_distance(RUNS["ddim100_final"]))
assert abs(DIST["ddpm"] - TOY("ddpm1000_mlp_dist")) < 2e-4 and abs(DIST["ddim100"] - TOY("ddim100_mlp_dist")) < 2e-4
assert (f"{DIST['data']:.3f}", f"{DIST['ddpm']:.3f}", f"{DIST['ddim100']:.3f}") == ("0.024", "0.025", "0.020")
_arm = lambda x: nearest_on_swirl(x)[1]  # noqa: E731
SAME_DDIM = float(np.mean(_arm(RUNS["ddim100_final"]) == _arm(RUNS["ddim50_final"])))
SAME_DDPM = float(np.mean(_arm(RUNS["ddpm1_final"]) == _arm(RUNS["ddpm2_final"])))
assert round(SAME_DDIM, 3) == TOY("ddim100_vs_ddim50_same_arm_mlp") == 0.972
assert round(SAME_DDPM, 3) == TOY("ddpm_seed1_vs_seed2_same_arm_mlp") == 0.499
PICKS = RUNS["picks"]
assert len(PICKS) == 6 and np.allclose(RUNS["ddim100_paths"][0], RUNS["z"][PICKS])

# the identities (diffusion_reference's self-checks, recomputed here on 300 noise points with the exact denoiser)
_z = gaussian_noise(300, seed=5)
ID_EULER = float(np.abs(ddim_sample(exact_eps, _z, 20).xs - euler_ode(exact_eps, _z, 20).xs).max())
ID_DDPM = float(np.abs(ddim_sample(exact_eps, _z, 40, eta=1.0, seed=9).xs
                       - ddpm_sample(exact_eps, _z, variance="beta_tilde", seed=9, steps=40).xs).max())
assert ID_EULER < 1e-9 and ID_DDPM < 1e-9                                  # "agree to within a billionth"

# inversion: the excerpt on screen, executed as shown
INVERT_SRC = "\n".join([
    "def invert(eps_model, x, steps=100):",
    "    ts = ddim_timesteps(steps)[::-1]          # 0 → 1000",
    "    for t, s in zip(ts[:-1], ts[1:]):",
    "        ab_t, ab_s = alpha_bar_at(t), alpha_bar_at(s)",
    "        eps = eps_model(x, t)                 # the noise it sees",
    "        x0 = x0_from_eps(x, eps, ab_t)        # its clean guess",
    "        x = np.sqrt(ab_s) * x0 + np.sqrt(1 - ab_s) * eps",
    "    return x",
])
_ns = {k: globals()[k] for k in ("ddim_timesteps", "alpha_bar_at", "x0_from_eps", "np")}
exec(INVERT_SRC, _ns)
assert len(INVERT_SRC.split("\n")) <= 10
assert np.allclose(_ns["invert"](MLP, RUNS["inv_points"], 100), RUNS["inv_up"][-1])
assert np.allclose(ddim_sample(MLP, RUNS["inv_up"][-1], 100).xs, RUNS["inv_down"])
INV_MED = {s: float(np.median(RUNS[f"inv_err{s}"])) for s in (100, 1000)}
assert f"{INV_MED[100]:.3f}" == "0.013" and f"{INV_MED[1000]:.4f}" == "0.0013"   # "about a hundredth"

# ---------------------------------------------------------------------------- look

FLAT = set("abcdefhiklmnorstuvwxzABCDEFGHIJKLMNOPRSTUVWXYZ0123456789")
DDPM_C = VIOLET                        # the noisy sampler: fresh noise every step
DDPM_C2 = mix(VIOLET, INK, 0.45)       # … its second run
DDIM_C = LIME                          # the noise-free sampler: a flow
DDIM_C2 = mix(LIME, INK, 0.55)


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


def picture_weight(t):
    """0 while the noise wins, 1 once the picture wins: a quick smooth step in log SNR around SNR = 1 (Episode 7)."""
    if t <= 0:
        return 1.0
    w = float(np.clip((log_snr(AB[int(round(t))]) + 0.6) / 1.2, 0, 1))
    return w * w * (3 - 2 * w)


def state_rgb(x, t):
    """A noisy picture: clipped to −1 … 1, VIOLET while noise wins, TEAL once the picture does (Episodes 4, 7)."""
    return tint(np.clip(x, -1, 1), mix(VIOLET, TEAL, picture_weight(t)), -1, 1)


def guess_rgb(x):
    return tint(np.clip(x, -1, 1), MODEL, -1, 1)


def noise_rgb(x):
    return tint(np.clip(x, -1, 1), NOISE, -1, 1)


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


def drive(mob, hooks, run_time=1.4, rate=m.linear):
    """Animate a progress a from 0 to 1, calling every hook(a) each frame."""

    def step(_, a):
        for hook in hooks:
            hook(a)

    return m.UpdateFromAlphaFunc(mob, step, run_time=run_time, rate_func=rate)


class Readout(m.VGroup):
    """A mono number display ("t = 250"), left-aligned at a fixed point; .show(v) updates it."""

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


def grow_paths(plot, paths, lines, heads, seconds, every=1):
    """Draw paths (steps × n × 2) as they are walked: each line follows its head."""
    paths = np.asarray(paths, dtype=float)[::every]

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


class Swarm(m.VGroup):
    """Swirl samples as dots coloured by distance to the curve: TEAL on an arm, RED stranded (> 0.1 away).
    `.go(xy)` moves every dot to its new place and recolours it on the way."""

    def __init__(self, plot, xy, radius=0.026):
        super().__init__()
        self.plot, self.xy = plot, np.asarray(xy, dtype=float)
        for p, c in zip(self.xy, self._colours(self.xy)):
            self.add(m.Dot(at(plot, p), radius=radius, color=c).set_opacity(0.9))

    @staticmethod
    def _colours(xy):
        far = nearest_on_swirl(xy)[0] > 0.1
        return [RED if f else DATA for f in far]

    def go(self, xy, run_time=1.2):
        start, end = self.xy.copy(), np.asarray(xy, dtype=float)
        c0, c1 = self._colours(start), self._colours(end)
        mixed = {(a, b): [mix(a, b, s / 8) for s in range(9)] for a in (RED, DATA) for b in (RED, DATA)}
        self.xy = end

        def step(_, a):
            now = start + (end - start) * a
            s = int(round(8 * a))
            for d, p, ca, cb in zip(self, now, c0, c1):
                d.move_to(at(self.plot, p))
                if ca != cb:
                    d.set_color(mixed[(ca, cb)][s])

        return m.UpdateFromAlphaFunc(self, step, run_time=run_time)


def arrow_to(a, b, color=MUTED, width=4, buff=0.12):
    return m.Arrow(a, b, buff=buff, color=color, stroke_width=width, max_tip_length_to_length_ratio=0.25)


# ---------------------------------------------------------------------------- the episode


class Ep09FewerSteps(Episode):
    number = 9

    def lesson(self):
        self.hook()
        self.one_step()
        self.swirl_paths()
        self.the_ode()
        self.digit_grid()
        self.measured_cost()
        self.paper_table()
        self.the_catch()
        self.inversion()
        self.recap()

    # ------------------------------------------------------------------ 0 · hook

    def hook(self):
        self.next_section("Hook", notes=(
            "One starting noise (np.random.default_rng(7), asserted) for every row: Episode 7's ancestral sampler with "
            "1,000 steps (σ² = β̃), DDIM (η = 0) with 50 and with 10 steps — mac_runs.md §4, arrays/digit_trajectory_7.npz "
            "(each row's x_t at its own steps; t shown as the paper's t = code t + 1). Times: s per 1,000 digits, batch "
            f"1,000 on this Mac's GPU (mac_runs.json digit_sampling): {T_PER_1000[1000]:.1f} / {T_PER_1000[50]:.2f} / "
            f"{T_PER_1000[10]:.2f} s. DDIM-50 and DDIM-10 finals correlate "
            f"{corr(SEVEN[50]['xs'][-1], SEVEN[10]['xs'][-1]):.3f}; DDPM-1000 vs DDIM-50 "
            f"{corr(SEVEN[1000]['xs'][-1], SEVEN[50]['xs'][-1]):.2f} (a different 7)."))
        ys = (1.75, 0.0, -1.75)
        h = 1.35
        rows = {}
        for (n, sub, color), y in zip(((1000, "fresh noise every step", DDPM_C), (50, "no fresh noise · DDIM", DDIM_C),
                                       (10, "no fresh noise · DDIM", DDIM_C)), ys):
            run = SEVEN[n]
            name = title_text(f"{n:,} steps", 34, INK)
            tag = text(sub, 20, color, weight=m.SEMIBOLD)
            label = m.VGroup(name, tag).arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT)
            label.move_to([0, y, 0]).align_to([-6.45, 0, 0], m.LEFT)
            start = framed(picture(state_rgb(XT7, 1000), h)).move_to([-2.35, y, 0])
            flip = Flip(lambda k, run=run: state_rgb(run["xs"][k], run["ts"][k]), 0, h)
            out = m.Group(m.Rectangle(width=h + 0.12, height=h + 0.12).set_stroke(EDGE, 2.5).move_to([0.35, y, 0]),
                          flip.image.move_to([0.35, y, 0]))
            go = arrow_to(start.get_right(), out.get_left() + [-0.12, 0, 0], MUTED, 4, 0.12)
            counter = Readout(1.4, y + 0.22, "t = {:d}", 1000, 30)
            calls = code(f"{n:,} network calls", 20, MUTED).move_to([1.4, y - 0.38, 0], aligned_edge=m.LEFT)
            secs = T_PER_1000[n]
            stat_n = title_text(f"{secs:.0f} s" if secs > 100 else f"{secs:.1f} s", 44, INK)
            stat_l = text("per 1,000 digits", 20, MUTED)
            stat_g = m.VGroup(stat_n, stat_l).arrange(m.DOWN, buff=0.1).move_to([5.65, y, 0])
            rows[n] = dict(label=label, start=start, out=out, flip=flip, go=go, counter=counter, calls=calls,
                           stat=stat_g, run=run)
        cap = cite("generated on this Mac · digit model (3.9 M parameters) · label 7 · the same starting noise in every row")
        cap.move_to([0, -3.3, 0])

        def runner(n, rate_pow=1.0):
            row = rows[n]
            last = len(row["run"]["xs"]) - 1

            def hook(a):
                k = min(last, int(np.floor(a ** rate_pow * (last + 1))))
                row["flip"].show(k)
                row["counter"].show(int(row["run"]["ts"][k]))
            return hook

        def static(n):
            r = rows[n]
            return m.AnimationGroup(m.FadeIn(r["label"], shift=0.1 * m.RIGHT), m.FadeIn(r["start"]), m.GrowArrow(r["go"]),
                                    m.FadeIn(r["out"]), m.FadeIn(r["counter"]), m.FadeIn(r["calls"]))

        self.say(
            "[Episode seven's sampler] [turns noise into a seven in a thousand small, noisy steps].",
            m.AnimationGroup(self.new_header("00 · hook", "A thousand steps is a lot"), static(1000), m.FadeIn(cap)),
            drive(rows[1000]["out"][1], [runner(1000, 1.0)], run_time=3.6, rate=lambda a: float(a) ** 0.55),
        )
        self.say(
            "[On this Mac, a thousand digits take a hundred seventy-two seconds].",
            m.FadeIn(rows[1000]["stat"], shift=0.1 * m.UP),
        )
        self.say(
            "[Today, the same trained network starts from the same noise], [and takes fifty steps], [or only ten].",
            m.AnimationGroup(static(50), static(10)),
            m.AnimationGroup(drive(rows[50]["out"][1], [runner(50)], run_time=1.8), m.FadeIn(rows[50]["stat"], shift=0.1 * m.UP)),
            m.AnimationGroup(drive(rows[10]["out"][1], [runner(10)], run_time=1.6), m.FadeIn(rows[10]["stat"], shift=0.1 * m.UP)),
        )
        same = m.SurroundingRectangle(m.Group(rows[50]["out"], rows[10]["out"]), buff=0.14, corner_radius=0.12)
        same.set_stroke(DDIM_C, 3)
        same_l = text("the same seven", 22, DDIM_C, weight=m.BOLD).next_to(same, m.DOWN, buff=0.16)
        self.say(
            "[Fifty and ten steps draw the very same seven]: [with no fresh noise, the same start always gives the same picture].",
            m.AnimationGroup(m.Create(same), m.FadeIn(same_l)),
            m.AnimationGroup(m.Circumscribe(m.Group(rows[1000]["out"]), color=DDPM_C, buff=0.1)),
        )
        self.wait(0.4)
        self.hook_rows = rows
        self.clear()

    # ------------------------------------------------------------------ 1 · one step

    def one_step(self):
        self.next_section("1 · Guess, then re-noise", notes=(
            f"Step {STEP_K + 1} of the DDIM-10 run above (paper t = {T_FROM} → {T_TO}; code t = {T_FROM - 1} → {T_TO - 1}). "
            "x_t, x_s and x̂₀ are the saved arrays; ε̂ = (x_t − √ᾱ_t·x̂₀)/√(1 − ᾱ_t) (std "
            f"{EPS_HAT.std():.3f}). DDIM eq. 12 with η = 0: x_s = √ᾱ_s·x̂₀ + √(1 − ᾱ_s)·ε̂ — here {C_X0:.3f} and {C_EPS:.3f}. "
            "Every stored step 1–8 obeys it to < 0.01 (float16 storage; step 0's x̂₀ was saved clipped to ±3). "
            "Ten network calls at t = 1000, 889, …, 112, 1 (the run's 'S steps evenly spaced from t = 999 to 0' in code "
            "time); the last call lands on the clean picture. DDIM: Song, Meng & Ermon, arXiv:2010.02502 (6 Oct 2020), "
            "ICLR 2021; 'denoising diffusion implicit models'. Pictures clipped to −1 … 1."))
        h = 1.45
        k = STEP_K
        x_t = framed(picture(state_rgb(D10["xs"][k], T_FROM), h)).move_to([-5.55, -0.6, 0])
        chip_words = text("the network", 22, INK, weight=m.BOLD)
        chip_ = m.VGroup(card(chip_words.width + 0.5, 0.8, MODEL, AMBER_FILL, radius=0.14), chip_words)
        chip_[1].move_to(chip_[0])
        chip_.move_to([-3.05, -0.6, 0])
        x0 = framed(picture(guess_rgb(D10["x0s"][k]), h), MODEL).move_to([-0.75, 0.45, 0])
        ep = framed(picture(noise_rgb(EPS_HAT), h), MODEL).move_to([-0.75, -1.65, 0])
        x_s = framed(picture(state_rgb(D10["xs"][k + 1], T_TO), h)).move_to([2.75, -0.6, 0])
        lab_t = tex(f"x_{{{T_FROM}}}", 34).next_to(x_t, m.DOWN, buff=0.2)
        lab_s = tex(f"x_{{{T_TO}}}", 34).next_to(x_s, m.DOWN, buff=0.2)
        lab_x0 = m.VGroup(tex(r"\hat x_0", 34, MODEL), text("the clean guess", 20, MUTED)).arrange(m.RIGHT, buff=0.15)
        lab_x0.next_to(x0, m.UP, buff=0.16)
        lab_ep = m.VGroup(tex(r"\hat\varepsilon", 34, MODEL), text("the noise it sees", 20, MUTED)).arrange(m.RIGHT, buff=0.15)
        lab_ep.next_to(ep, m.DOWN, buff=0.16)
        a_in = arrow_to(x_t.get_right(), chip_.get_left(), MUTED)
        a_ep = arrow_to(chip_.get_right(), ep.get_left() + [0, 0.25, 0], MODEL)
        a_x0 = arrow_to(chip_.get_right(), x0.get_left() + [0, -0.25, 0], MODEL)
        b_x0 = arrow_to(x0.get_right(), x_s.get_left() + [0, 0.3, 0], TIME)
        b_ep = arrow_to(ep.get_right(), x_s.get_left() + [0, -0.3, 0], TIME)
        k_x0 = code(f"× {C_X0:.3f}", 22, TIME, weight=m.BOLD).move_to(b_x0.get_center() + [0.2, 0.5, 0])
        k_ep = code(f"× {C_EPS:.3f}", 22, TIME, weight=m.BOLD).move_to(b_ep.get_center() + [0.2, -0.5, 0])

        f = formula(r"x_s", "=", r"\sqrt{\bar\alpha_s}", r"\,\hat x_0", "+", r"\sqrt{1-\bar\alpha_s}", r"\,\hat\varepsilon", size=46)
        f[2].set_color(TIME)
        f[5].set_color(TIME)
        f[3].set_color(MODEL)
        f[6].set_color(MODEL)
        f.move_to([-0.6, 2.25, 0])
        fz = formula("+", r"\,0\cdot z", size=46, color=NOISE).next_to(f, m.RIGHT, buff=0.22)
        fz.align_to(f[4], m.DOWN)
        steps = Readout(3.9, 0.6, "t = {}", f"{T_FROM} → {T_TO}", 23)
        step_of = code(f"step {k + 1} of 10", 20, MUTED).move_to([3.9, 0.15, 0], aligned_edge=m.LEFT)
        nofresh = pill("no fresh noise", NOISE, VIOLET_FILL, 22).move_to([5.0, -1.7, 0])
        cap = cite("one step of the seven above, DDIM with 10 steps · generated on this Mac").move_to([-2.6, -3.35, 0])
        src = cite("Song, Meng & Ermon, 2020 · eq. 12").move_to([4.65, -3.35, 0])
        assert steps.get_right()[0] < 6.6 and nofresh.get_right()[0] < 6.6 and fz.get_right()[0] < 6.6, (steps.get_right(), nofresh.get_right(), fz.get_right())

        self.say(
            "[Each step begins as in episode seven]: [the network predicts the noise in the picture], "
            "[and from it, a guess of the clean seven].",
            m.AnimationGroup(self.new_header("01 · one step", "Guess, then re-noise"), m.FadeIn(x_t), m.FadeIn(lab_t),
                             m.FadeIn(cap)),
            m.AnimationGroup(m.GrowArrow(a_in), m.FadeIn(chip_), m.GrowArrow(a_ep), m.FadeIn(ep), m.FadeIn(lab_ep)),
            m.AnimationGroup(m.GrowArrow(a_x0), m.FadeIn(x0), m.FadeIn(lab_x0)),
        )
        self.say(
            "[The new step keeps that guess], [adds the predicted noise back in], "
            "[and scales both for the next, lower noise level].",
            m.AnimationGroup(m.FadeIn(f[:4], shift=0.1 * m.UP), m.GrowArrow(b_x0), m.FadeIn(src)),
            m.AnimationGroup(m.FadeIn(f[4:], shift=0.1 * m.UP), m.GrowArrow(b_ep)),
            m.AnimationGroup(m.FadeIn(k_x0), m.FadeIn(k_ep), m.FadeIn(x_s, target_position=m.Group(x0, ep)),
                             m.FadeIn(lab_s), m.FadeIn(steps), m.FadeIn(step_of)),
        )
        self.say(
            "[No new random numbers are drawn]: [what goes back in is the very noise the network saw].",
            m.AnimationGroup(m.FadeIn(fz, shift=0.1 * m.LEFT), m.FadeIn(nofresh, shift=0.1 * m.UP)),
            m.AnimationGroup(m.Circumscribe(ep, color=MODEL, buff=0.08), m.Circumscribe(f[6], color=MODEL, buff=0.08)),
        )
        self.wait(0.3)
        screen_a = m.Group(x_t, chip_, x0, ep, x_s, lab_t, lab_s, lab_x0, lab_ep, a_in, a_ep, a_x0, b_x0, b_ep, k_x0, k_ep,
                           f, fz, steps, step_of, nofresh, cap, src)

        # the skip: ten hops along the clock, with the seven at each
        y_line, x_l, x_r = -0.3, -5.6, 5.6
        xpos = lambda t: x_l + (1000 - t) / 1000 * (x_r - x_l)  # noqa: E731
        axis = m.Line([x_l, y_line, 0], [x_r, y_line, 0]).set_stroke(EDGE, 3)
        comb = m.VGroup(*[m.Line([xpos(t), y_line - 0.13, 0], [xpos(t), y_line + 0.13, 0]).set_stroke(FAINT, 1)
                          for t in range(1000, 0, -1)])
        ends = m.VGroup(code("t = 1000", 20, TIME).next_to([x_l, y_line, 0], m.DOWN, buff=0.25),
                        code("0", 20, TIME).next_to([x_r, y_line, 0], m.DOWN, buff=0.25))
        sit(ends[1], baseline(ends[0]))
        comb_l = text("Episode 7: 1,000 small steps", 22, MUTED).move_to([-2.6, -1.05, 0])
        hop_ts = [int(t) for t in D10["ts"][:-1]]                                       # 1000, 889, …, 112, 1
        assert hop_ts[0] == 1000 and hop_ts[-1] == 1 and len(hop_ts) == 10
        arcs = m.VGroup()
        for t0, t1 in zip(hop_ts[:-1], hop_ts[1:]):
            arcs.add(m.ArcBetweenPoints([xpos(t0), y_line + 0.05, 0], [xpos(t1), y_line + 0.05, 0], angle=-1.6)
                     .set_stroke(TIME, 4))
        arcs.add(m.Dot([xpos(1), y_line, 0], radius=0.07, color=TIME))
        hop_l = text("DDIM: 10 hops, 10 network calls", 22, TIME, weight=m.BOLD).move_to([3.2, -1.05, 0])
        sit(hop_l, baseline(comb_l))
        fh = 0.92
        frames = m.Group()
        tlabs = m.VGroup()
        shown = list(range(9)) + [10]                                                    # t = 1000 … 112, then the output
        for i in shown:
            t = int(D10["ts"][i])
            x = xpos(t) if i < 10 else x_r
            fr = framed(picture(state_rgb(D10["xs"][i], t), fh), stroke=2).move_to([x, 1.1, 0])
            frames.add(fr)
            tlabs.add(code(f"{t}", 20, MUTED).next_to(fr, m.UP, buff=0.12))
        for lab in tlabs[1:]:
            sit(lab, baseline(tlabs[0]))
        name = title_text("DDIM", 52, INK)
        name_l = m.VGroup(text("denoising diffusion implicit models", 26, INK),
                          cite("Song, Meng & Ermon, 2020 · ICLR 2021 · the same trained network")).arrange(
            m.DOWN, buff=0.12, aligned_edge=m.LEFT)
        name_card = m.VGroup(name, name_l).arrange(m.RIGHT, buff=0.45).move_to([0, -2.45, 0])
        rule = m.Line(name.get_corner(m.UR) + [0.22, 0.05, 0], name.get_corner(m.DR) + [0.22, -0.05, 0]).set_stroke(TIME, 4)
        name_card.add(rule)

        self.say(
            "[Since each step lands straight on a chosen level], [the levels can be far apart]: "
            "[ten hops, from a thousand down to zero].",
            m.AnimationGroup(m.FadeOut(screen_a), m.FadeIn(axis), m.FadeIn(ends)),
            m.AnimationGroup(m.FadeIn(comb), m.FadeIn(comb_l)),
            m.AnimationGroup(m.LaggedStart(m.AnimationGroup(m.FadeIn(frames[0], shift=0.1 * m.DOWN), m.FadeIn(tlabs[0])),
                                           *[m.AnimationGroup(m.Create(arcs[i]), m.FadeIn(frames[i + 1], shift=0.1 * m.DOWN),
                                                              m.FadeIn(tlabs[i + 1])) for i in range(8)],
                                           m.AnimationGroup(m.Create(arcs[8]), m.FadeIn(arcs[9]), m.FadeIn(frames[9], shift=0.1 * m.DOWN),
                                                            m.FadeIn(tlabs[9])),
                                           lag_ratio=0.35, run_time=2.6),
                             m.FadeIn(hop_l)),
        )
        self.say(
            "[This is DDIM, from Song, Meng and Ermon in twenty twenty]: [the same trained network, a new way to sample].",
            m.FadeIn(name_card, shift=0.15 * m.UP),
            m.Circumscribe(name_l[1], color=TIME, buff=0.08),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 2 · the swirl: same noise, same point

    def swirl_paths(self):
        self.next_section("2 · Same noise, same point", notes=(
            "The swirl's trained MLP (toy.md §2: 45,954 parameters, DDPM's ε-loss, trained on this Mac) as the denoiser. "
            "All runs from toy.md §3's shared noise z = gaussian_noise(2000, seed=5), recomputed by precompute.py and "
            "asserted against toy.json: DDPM ancestral 1,000 steps (σ² = β) with fresh-noise seeds 1 and 2; DDIM 100 and 50 "
            f"(η = 0). Same arm (nearest_on_swirl): DDIM-100 vs DDIM-50 {SAME_DDIM:.1%} (toy 0.972); DDPM seed 1 vs seed 2 "
            f"{SAME_DDPM:.1%} (toy 0.499). The six traced particles: starts with 0.8 < |z| < 2, both DDPM runs on the swirl but "
            "on different arms, DDIM runs on one arm, paths inside the view, spread round the circle (precompute.py) — "
            "chosen to show the contrast; the percentages are over all 2,000. DDPM paths drawn every 20th step."))
        lim, size = 2.15, 4.3
        pl = plane((-4.3, -0.6, 0), size, lim)
        pr = plane((0.4, -0.6, 0), size, lim)
        bands = m.VGroup(swirl_band(pl), swirl_band(pr))
        frames = m.VGroup(*[m.Square(size).set_stroke(EDGE, 2).move_to(p.coords(0, 0)) for p in (pl, pr)])
        t_l = m.VGroup(text("fresh noise every step", 24, DDPM_C, weight=m.BOLD), cite("Episode 7 · 1,000 steps"))
        t_r = m.VGroup(text("no fresh noise", 24, DDIM_C, weight=m.BOLD), cite("DDIM · 100 steps"))
        for tt, fr in zip((t_l, t_r), frames):
            tt.arrange(m.DOWN, buff=0.08).next_to(fr, m.UP, buff=0.15)
        z = RUNS["z"][PICKS]
        starts = m.VGroup(*[m.VGroup(*[m.Dot(at(p, q), radius=0.075, color=NOISE).set_z_index(4) for q in z]) for p in (pl, pr)])

        def lines(color, n=6, width=2.6, opacity=0.95):
            return m.VGroup(*[m.VMobject().set_stroke(color, width, opacity=opacity) for _ in range(n)]).set_z_index(2)

        def heads(plot, color, start):
            return m.VGroup(*[m.Dot(at(plot, q), radius=0.06, color=color).set_z_index(5) for q in start])

        l1, l2 = lines(DDPM_C, width=1.8, opacity=0.8), lines(DDPM_C2, width=1.6, opacity=0.65)
        h1, h2 = heads(pl, DDPM_C, z), heads(pl, DDPM_C2, z)
        l3, l4 = lines(DDIM_C, width=3.2), lines(DDIM_C2, width=2.2, opacity=0.9)
        h3, h4 = heads(pr, DDIM_C, z), heads(pr, DDIM_C2, z)
        leg_l = m.VGroup(m.VGroup(m.Line(ORIGIN_, [0.45, 0, 0]).set_stroke(DDPM_C, 4), code("run 1", 20, MUTED)).arrange(m.RIGHT, buff=0.15),
                         m.VGroup(m.Line(ORIGIN_, [0.45, 0, 0]).set_stroke(DDPM_C2, 4), code("run 2", 20, MUTED)).arrange(m.RIGHT, buff=0.15))
        leg_l.arrange(m.RIGHT, buff=0.5).next_to(frames[0], m.DOWN, buff=0.22)
        leg_r = m.VGroup(m.VGroup(m.Line(ORIGIN_, [0.45, 0, 0]).set_stroke(DDIM_C, 4), code("100 steps", 20, MUTED)).arrange(m.RIGHT, buff=0.15),
                         m.VGroup(m.Line(ORIGIN_, [0.45, 0, 0]).set_stroke(DDIM_C2, 4), code("50 steps", 20, MUTED)).arrange(m.RIGHT, buff=0.15))
        leg_r.arrange(m.RIGHT, buff=0.5).next_to(frames[1], m.DOWN, buff=0.22)
        rx = 4.65
        net = cite(f"the denoiser: a small trained\nnetwork, {MLP_PARAMS:,} parameters").move_to([rx, 1.75, 0])
        s1 = title_text(f"{100 * SAME_DDIM:.0f}%", 64, DDIM_C).move_to([rx, 0.55, 0])
        s1_l = text("same arm: 100 vs 50 steps", 22, MUTED).next_to(s1, m.DOWN, buff=0.15)
        s2 = title_text(f"{100 * SAME_DDPM:.1f}%", 64, DDPM_C).move_to([rx, -1.35, 0])
        s2_l = text("same arm: two runs of\nEpisode 7's sampler", 22, MUTED).next_to(s2, m.DOWN, buff=0.15)
        over = cite("over all 2,000 points").move_to([rx, -3.15, 0])
        assert max(g.get_right()[0] for g in (net, s1_l, s2_l)) < 6.6 and frames[1].get_right()[0] < rx - 1.6, ([g.get_right()[0] for g in (net, s1_l, s2_l)], frames[1].get_right())

        self.say(
            "[Back on the swirl, take six starting points], [with the small trained network as the denoiser].",
            m.AnimationGroup(self.new_header("02 · the swirl", "Same noise, same point"), m.FadeIn(frames), m.FadeIn(bands),
                             m.FadeIn(t_l), m.FadeIn(t_r), m.FadeIn(starts, scale=0.5)),
            m.FadeIn(net),
        )
        self.say(
            "[Episode seven's sampler wanders], [and two runs from the same start end on different arms].",
            m.AnimationGroup(m.FadeIn(leg_l[0]), grow_paths(pl, RUNS["ddpm1_paths"], l1, h1, 2.4, every=20)),
            m.AnimationGroup(m.FadeIn(leg_l[1]), grow_paths(pl, RUNS["ddpm2_paths"], l2, h2, 2.4, every=20)),
        )
        self.say(
            "[The new sampler glides along smooth curves].",
            m.AnimationGroup(m.FadeIn(leg_r[0]), grow_paths(pr, RUNS["ddim100_paths"], l3, h3, 2.4)),
        )
        self.say(
            "[Halve the steps to fifty], [and the paths land in almost the same spots].",
            m.AnimationGroup(m.FadeIn(leg_r[1]), grow_paths(pr, RUNS["ddim50_paths"], l4, h4, 2.0)),
            m.LaggedStart(*[m.Circumscribe(d, shape=m.Circle, color=DDIM_C, buff=0.1, run_time=0.9) for d in h3], lag_ratio=0.12),
        )
        self.say(
            "[Over two thousand points], [runs with a hundred steps and with fifty end on the same arm ninety-seven percent of the time].",
            m.FadeIn(over),
            m.AnimationGroup(m.FadeIn(s1, shift=0.15 * m.UP), m.FadeIn(s1_l)),
        )
        self.say(
            "[Two runs of episode seven's sampler agree only forty-nine point nine percent of the time]: [a coin flip].",
            m.AnimationGroup(m.FadeIn(s2, shift=0.15 * m.UP), m.FadeIn(s2_l)),
            m.Circumscribe(s2, color=DDPM_C, buff=0.1),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 3 · the ODE

    def the_ode(self):
        self.next_section("3 · An equation in disguise", notes=(
            "Song et al. 2021 (score SDE): reverse SDE eq. 6 (Episode 6) and the probability-flow ODE eq. 13, 'whose "
            "trajectories share the same marginal probability densities'. DDIM eq. 14 / Proposition 1: with σ = √((1 − ᾱ)/ᾱ) "
            "and y = x/√ᾱ = x₀ + σ·ε, dy/dσ = ε̂; DDIM (η = 0) is Euler on it in σ. Algebra: x_s/√ᾱ_s = x̂₀ + σ_s·ε̂ and "
            "x̂₀ = y_t − σ_t·ε̂, so y_s = y_t + (σ_s − σ_t)·ε̂. Checked here with the swirl's exact denoiser on 300 noise "
            f"points: ddim_sample vs euler_ode (20 steps), max difference {ID_EULER:.1e}; DDIM with η = 1 vs ddpm_sample with "
            f"σ² = β̃ (40 steps, same noise seed), max difference {ID_DDPM:.1e}. η (DDIM eq. 16) scales the fresh noise; the "
            "paper's Table 1 also has η = 0.2 and 0.5. Episode 7's digit run used σ² = β̃ (mac_runs.md §4)."))
        lx, fx = -6.45, -2.55
        ys = (1.75, 0.45, -0.85, -2.15)
        f1 = formula(r"\mathrm{d}x", "=", r"\big[f(x,t) - g(t)^2", r"\nabla_x \log p_t(x)", r"\big]\,\mathrm{d}t", "+",
                     r"g(t)\,\mathrm{d}\bar w", size=36)
        f1[3].set_color(SCORE)
        f1[6].set_color(NOISE)
        f2 = formula(r"\mathrm{d}x", "=", r"\big[f(x,t) - \frac{1}{2}\,g(t)^2", r"\nabla_x \log p_t(x)", r"\big]\,\mathrm{d}t", size=36)
        f2[3].set_color(SCORE)
        f3 = formula(r"\frac{\mathrm{d}y}{\mathrm{d}\sigma}", "=", r"\hat\varepsilon(x_t, t)", size=42)
        f3[2].set_color(MODEL)
        f4 = formula(r"y_s", "=", r"y_t", "+", r"(\sigma_s - \sigma_t)", r"\,\hat\varepsilon", size=42)
        f4[4].set_color(TIME)
        f4[5].set_color(MODEL)
        l1 = text("Episode 6: undoing noise,\nwith fresh noise", 22, MUTED)
        l2 = text("its noise-free twin:\nthe probability flow ODE", 22, INK, t2c={"probability flow ODE": DDIM_C})
        l3 = m.VGroup(text("in the σ view", 22, INK), tex(r"y = x_0 + \sigma\,\varepsilon", 30))
        l3.arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT)
        l4 = text("one DDIM step\n= one Euler step", 22, INK)
        rows = []
        for lab, f, y in zip((l1, l2, l3, l4), (f1, f2, f3, f4), ys):
            lab.move_to([0, y, 0]).align_to([lx, 0, 0], m.LEFT)
            f.move_to([0, y, 0]).align_to([fx, 0, 0], m.LEFT)
            assert f.get_right()[0] < 6.6, f.get_right()
            rows.append(m.VGroup(lab, f))
        no_w = text("no fresh noise", 22, DDIM_C, weight=m.BOLD).next_to(f2, m.RIGHT, buff=0.4)
        src2 = cite("Song et al., 2021 · eq. 13").move_to([4.9, 2.35, 0])
        src4 = cite("Song, Meng & Ermon, 2020 · eq. 14").next_to(f4, m.RIGHT, buff=0.4)
        assert no_w.get_right()[0] < 6.6 and src4.get_right()[0] < 6.6
        check = code(f"checked on the swirl: DDIM (η = 0) and Euler in σ differ by {ID_EULER:.0e}", 20, GOOD)
        check.move_to([0, -3.25, 0])

        self.say(
            "[Why can it take such big steps]? [Recall episode six's equation for undoing noise], [with fresh noise in every step].",
            self.new_header("03 · the ODE", "An equation in disguise"),
            m.AnimationGroup(m.FadeIn(l1), m.Write(f1)),
            m.Circumscribe(f1[6], color=NOISE, buff=0.08),
        )
        self.say(
            "[Every such equation has a noise-free twin], [the probability flow ODE]: "
            "[the same noise levels at every time, but no randomness].",
            m.AnimationGroup(m.FadeIn(l2), m.FadeIn(src2)),
            m.Write(f2),
            m.FadeIn(no_w, shift=0.1 * m.LEFT),
        )
        self.say(
            "[Write the noisy picture as clean picture plus sigma times noise], [and the twin is simply this]: "
            "[the picture changes along epsilon hat as sigma shrinks].",
            m.FadeIn(l3),
            m.Write(f3),
            m.Circumscribe(f3[2], color=MODEL, buff=0.08),
        )
        self.say(
            "[One step of the new sampler is one Euler step along it]: [we checked, they agree to within a billionth].",
            m.AnimationGroup(m.FadeIn(l4), m.Write(f4), m.FadeIn(src4)),
            m.FadeIn(check, shift=0.1 * m.UP),
        )
        # the η dial
        y_d, x0_, x1_ = -1.15, -3.6, 3.6
        ex = lambda e: x0_ + e * (x1_ - x0_)  # noqa: E731
        dial = m.Line([x0_, y_d, 0], [x1_, y_d, 0]).set_stroke(TIME, 5)
        ticks = m.VGroup()
        for e in (0, 0.2, 0.5, 1):
            ticks.add(m.Line([ex(e), y_d - 0.13, 0], [ex(e), y_d + 0.13, 0]).set_stroke(TIME, 3),
                      code(f"η = {e:g}" if e in (0, 1) else f"{e:g}", 22, TIME).move_to([ex(e), y_d + 0.45, 0]))
        knob = m.Dot([ex(0), y_d, 0], radius=0.15, color=INK).set_z_index(3)

        def end_card(head, body, color, x):
            words = m.VGroup(text(head, 22, color, weight=m.BOLD), text(body, 20, MUTED)).arrange(m.DOWN, buff=0.1)
            box = card(words.width + 0.5, words.height + 0.4, color, CARD, radius=0.14, stroke=2).move_to(words)
            return m.VGroup(box, words).move_to([x, -2.35, 0])

        c0 = end_card("no fresh noise: DDIM", "= Euler in σ, exactly", DDIM_C, ex(0))
        c1 = end_card("fresh noise every step", f"= Episode 7's sampler (σ² = β̃), exactly", DDPM_C, ex(1))
        for c_ in (c0, c1):
            assert c_.get_left()[0] > -6.6 and c_.get_right()[0] < 6.6, c_.get_right()
        both = cite(f"both identities checked on the swirl: differences below 10⁻⁹ · DDIM paper eq. 16").move_to([0, -3.35, 0])
        up = ys[0] - ys[2]
        self.say(
            "[A dial called eta brings fresh noise back]: [at eta equals one, it is exactly the noisy sampler of episode seven].",
            m.AnimationGroup(m.FadeOut(rows[0]), m.FadeOut(rows[1]), m.FadeOut(no_w), m.FadeOut(src2), m.FadeOut(check),
                             rows[2].animate.shift(up * m.UP), rows[3].animate.shift(up * m.UP), src4.animate.shift(up * m.UP),
                             m.FadeIn(dial), m.FadeIn(ticks), m.FadeIn(knob), m.FadeIn(c0, shift=0.1 * m.UP)),
            m.AnimationGroup(knob.animate(run_time=1.4).move_to([ex(1), y_d, 0]), m.FadeIn(c1, shift=0.1 * m.UP), m.FadeIn(both)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 4 · the digit model, every step count

    def digit_grid(self):
        self.next_section("4 · Same seeds, every step count", notes=(
            "mac_runs.md §4: 10,000 samples per configuration, sample i asks for label i mod 10 and starts from the same "
            "noise in every configuration (numpy seed 1234); shown: samples 10–19 (labels 0–9) from "
            "arrays/digit_samples_ddpm_<config>_w1.npz. DDPM ancestral 1,000 steps (σ² = β̃) vs DDIM (η = 0) 100 … 1 steps. "
            f"Median pixel correlation with DDIM-100 over the first 1,000 samples: DDIM-10 {_sim(10, 100):.3f}, DDPM-1000 "
            f"{_sim(1000, 100):.2f}. Why 1–2 steps give speckles (mac_runs.md): x̂₀ = (x_t − √(1 − ᾱ)·ε̂)/√ᾱ and at t = 1000 "
            f"√ᾱ = {SQRT_AB_T:.5f}, so the network's error is multiplied by 1/√ᾱ = {1 / SQRT_AB_T:.1f}."))
        pitch, dh = 0.66, 0.56
        y_top = 2.15
        x_first = -4.55
        grid = m.Group()
        labels = m.VGroup()
        for r, n in enumerate(ROWS):
            y = y_top - r * pitch
            row = m.Group(*[picture(tint(SAMPLES[n][i], DATA), dh).move_to([x_first + c * pitch, y, 0])
                            for c, i in enumerate(GRID_IDX)])
            grid.add(row)
            labels.add(code(f"{n:,}", 24, DDPM_C if n == 1000 else DDIM_C, weight=m.BOLD)
                       .move_to([-5.35, y, 0], aligned_edge=m.RIGHT))
        steps_l = cite("steps").next_to(labels, m.DOWN, buff=0.2).align_to(labels, m.RIGHT)
        assert grid.get_bottom()[1] > -2.95 and labels.get_left()[0] > -6.6
        cap = cite("generated on this Mac · digit model · samples 10–19 (labels 0–9) · the same noise in every row (seed 1234)")
        cap.move_to([0, -3.35, 0])
        rx = 4.5
        p1 = pill("1,000 steps: fresh noise", DDPM_C, VIOLET_FILL, 20).move_to([rx, 1.95, 0])
        p2 = pill("DDIM: no fresh noise", DDIM_C, LIME_FILL, 20).move_to([rx, 1.3, 0])
        rows_ok = m.SurroundingRectangle(m.Group(grid[1], grid[4]), buff=0.06, corner_radius=0.08).set_stroke(DDIM_C, 3)
        rows_bad = m.SurroundingRectangle(m.Group(grid[6], grid[7]), buff=0.06, corner_radius=0.08).set_stroke(RED, 3)
        fx0 = formula(r"\hat x_0", "=", r"\frac{x_t - \sqrt{1-\bar\alpha_t}\,\hat\varepsilon}{\sqrt{\bar\alpha_t}}", size=38)
        fx0[0].set_color(MODEL)
        fx0.move_to([rx, 0.05, 0])
        at_t = code(f"t = 1000:  √ᾱ = {SQRT_AB_T:.4f}", 22, TIME).move_to([rx, -1.0, 0])
        mag = title_text(f"{1 / SQRT_AB_T:.0f}×", 64, RED).move_to([rx, -1.95, 0])
        mag_l = text("every error, magnified", 22, MUTED).next_to(mag, m.DOWN, buff=0.12)
        assert grid.get_right()[0] < rx - 2.0 and at_t.get_right()[0] < 6.6, (grid.get_right(), at_t.get_right(), fx0.get_left())

        self.say(
            "[Now ten digits from the digit model, each from its own starting noise], [sampled at every step count].",
            m.AnimationGroup(self.new_header("04 · the digits", "Same seeds, every step count"), m.FadeIn(cap),
                             m.FadeIn(p1), m.FadeIn(p2)),
            m.AnimationGroup(m.LaggedStart(*[m.AnimationGroup(m.FadeIn(lab), m.FadeIn(row)) for lab, row in zip(labels, grid)],
                                           lag_ratio=0.25, run_time=2.4), m.FadeIn(steps_l)),
        )
        self.say(
            "[The thousand noisy steps drew different digits]; [from a hundred steps down to ten, the same noise gives the same digits].",
            m.Circumscribe(grid[0], color=DDPM_C, buff=0.06),
            m.Create(rows_ok),
        )
        self.say(
            "[At five steps, the strokes start to smear]; [at two steps, and at one, only speckled noise is left].",
            m.Circumscribe(grid[5], color=MODEL, buff=0.06),
            m.Create(rows_bad),
        )
        self.say(
            "[Why speckles]? [From almost pure noise, the clean guess divides by a tiny number], "
            "[which magnifies the network's error a hundred fifty-seven times].",
            m.FadeOut(m.VGroup(p1, p2)),
            m.AnimationGroup(m.Write(fx0), m.FadeIn(at_t, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(mag, scale=0.8), m.FadeIn(mag_l)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 5 · measured: quality against time

    def measured_cost(self):
        self.next_section("5 · Quality against time", notes=(
            "mac_runs.json digit_sampling (10,000 samples per row; s per 1,000 = mean of batches 2–10 at batch 1,000 on "
            "this Mac's GPU; the Mac was shared: ±5–10% jitter). cFID = the FID formula on the 128-d features of our own "
            "MNIST classifier (99.45% test accuracy), 10,000 generated vs the 10,000 test digits — NOT Inception FID; only "
            f"comparable within this course. Floor (real training digits): {FLOOR:.3f}. Bootstrap std ≈ 0.3 "
            f"({', '.join(f'{b:.2f}' for b in BOOT)} for DDIM 100/50/20), so 20, 50 and 100 steps are tied. "
            f"Speed-ups: {SPEED50:.1f}× (50 steps), {SPEED10:.1f}× (10 steps)."))
        pl = Plot((0.1, 400), (1.5, 1500), width=6.3, height=4.0, center=(-2.8, -0.2, 0), log_x=True, log_y=True,
                  x_ticks=[0.1, 1, 10, 100], y_ticks=[10, 100, 1000], x_fmt="{:g}", y_fmt="{:g}",
                  x_label="seconds per 1,000 digits (log scale)", y_label="cFID, lower is better (log scale)")
        floor = pl.hline(FLOOR, DATA, 2.5)
        floor_l = text(f"real digits {FLOOR:.2f}", 20, DATA).next_to(pl.coords(0.1, FLOOR), m.UR, buff=0.08)
        floor_l.shift(0.12 * m.RIGHT)
        pts = {n: (T_PER_1000[n], CFID[n]) for n in ROWS}
        dot = {n: pl.dot(*pts[n], DDPM_C if n == 1000 else (RED if n <= 5 else DDIM_C), 0.09) for n in ROWS}
        lab = {}
        for n in ROWS:
            words = code(f"{n:,}", 20, INK if n > 5 else RED)
            if n <= 5:      # the cliff's dots are listed in the RED note instead
                lab[n] = m.VGroup()
            elif n == 10:
                lab[n] = words.next_to(dot[n], m.LEFT, buff=0.12)
            else:
                lab[n] = words.next_to(dot[n], m.UP, buff=0.12)
        order = (100, 50, 20, 10, 5, 2, 1)
        seg = {b: pl.line([pts[a][0], pts[b][0]], [pts[a][1], pts[b][1]], DDIM_C if b > 5 else RED, 4)
               for a, b in zip(order[:-1], order[1:])}
        cliff = code("5 steps  21.6\n2 steps   939\n1 step    943", 20, RED).move_to(pl.coords(4.5, 250))
        not_inc = cite("cFID: FID's formula on our own digit classifier's features, not Inception FID").move_to([-2.5, -3.45, 0])
        # the table
        tx = 3.85
        cols = (-1.5, -0.35, 0.8, 1.8)
        head = m.VGroup(*[code(w, 20, MUTED) for w in ("steps", "cFID", "time", "speed")])
        rows_t = []
        for n, speed in ((1000, ""), (50, f"{SPEED50:.0f}×"), (10, f"{SPEED10:.0f}×")):
            secs = T_PER_1000[n]
            rows_t.append(m.VGroup(code(f"{n:,}", 24, DDPM_C if n == 1000 else DDIM_C, weight=m.BOLD),
                                   code(CFID_TXT[n], 24, INK), code(f"{secs:.0f} s" if secs > 100 else f"{secs:.1f} s", 24, INK),
                                   code(speed or "1×", 24, GOOD if speed else MUTED, weight=m.BOLD)))
        for r, row in enumerate([head] + rows_t):
            y = (1.35, 0.7, 0.1, -0.5)[r]
            for c_, w in zip(cols, row):
                w.move_to([tx + c_, y, 0])
            ref = baseline(row[0])
            for w in row[1:]:
                sit(w, ref)
        table_box = card(4.6, 2.75, EDGE, CARD, radius=0.16, stroke=2).move_to([tx + 0.15, 0.25, 0])
        rule = m.Line([tx - 2.0, 1.08, 0], [tx + 2.3, 1.08, 0]).set_stroke(EDGE, 2)
        per = cite("time per 1,000 digits, this Mac\nnumbers by the dots: steps").move_to([tx + 0.15, -1.6, 0])
        tie = m.SurroundingRectangle(m.VGroup(dot[100], dot[50], dot[20], lab[100], lab[50], lab[20]), buff=0.12,
                                     corner_radius=0.1).set_stroke(MUTED, 2)
        tie_l = text("tied: error ≈ ±0.3", 20, MUTED).next_to(tie, m.UP, buff=0.12)
        honest = text("1,000 noisy steps still score best", 22, INK).move_to([tx + 0.15, -2.35, 0])
        assert table_box.get_right()[0] < 6.6 and pl.get_right()[0] < table_box.get_left()[0] - 0.2, (table_box.get_left(), table_box.get_right(), pl.get_right())

        self.say(
            "[Measured on this Mac]: [quality against time], [both on log scales].",
            m.AnimationGroup(self.new_header("05 · measured", "Quality against time"), m.FadeIn(pl.axes)),
            m.FadeIn(pl.ticks),
            m.AnimationGroup(m.FadeIn(pl.x_label), m.FadeIn(pl.y_label)),
        )
        self.say(
            "[The score is a classifier-based version of FID, not the standard Inception one]: "
            "[lower is better], [and real digits score two point zero six].",
            m.FadeIn(not_inc, shift=0.1 * m.UP),
            m.Circumscribe(pl.y_label, color=INK, buff=0.08),
            m.AnimationGroup(m.Create(floor), m.FadeIn(floor_l)),
        )
        self.say(
            "[A thousand noisy steps score two point six four], [in a hundred seventy-two seconds per thousand digits].",
            m.AnimationGroup(m.FadeIn(dot[1000], scale=0.5), m.FadeIn(lab[1000]), m.FadeIn(table_box), m.FadeIn(head),
                             m.FadeIn(rule), m.FadeIn(rows_t[0][:2])),
            m.AnimationGroup(m.FadeIn(rows_t[0][2:]), m.FadeIn(per)),
        )
        self.say(
            "[Fifty steps score three point three five in under nine seconds]: [nineteen times faster].",
            m.AnimationGroup(m.FadeIn(dot[100], scale=0.5), m.FadeIn(lab[100]), m.Create(seg[50]), m.FadeIn(dot[50], scale=0.5),
                             m.FadeIn(lab[50]), m.FadeIn(rows_t[1][:3])),
            m.FadeIn(rows_t[1][3], scale=0.8),
        )
        self.say(
            "[Ten steps score four point five seven in under two seconds]: [almost a hundred times faster].",
            m.AnimationGroup(m.Create(seg[20]), m.FadeIn(dot[20], scale=0.5), m.FadeIn(lab[20]), m.Create(seg[10]),
                             m.FadeIn(dot[10], scale=0.5), m.FadeIn(lab[10]), m.FadeIn(rows_t[2][:3])),
            m.FadeIn(rows_t[2][3], scale=0.8),
        )
        self.say(
            "[Below ten steps, quality falls off a cliff].",
            m.AnimationGroup(m.Succession(*[m.AnimationGroup(m.Create(seg[b]), m.FadeIn(dot[b], scale=0.5), m.FadeIn(lab[b]),
                                                             run_time=0.5) for b in (5, 2, 1)]),
                             m.FadeIn(cliff, shift=0.1 * m.UP)),
        )
        self.say(
            "[To be fair]: [twenty, fifty and a hundred steps are tied within the error bars], "
            "[and a thousand noisy steps still score best].",
            m.Wait(0.1),
            m.AnimationGroup(m.Create(tie), m.FadeIn(tie_l)),
            m.AnimationGroup(m.Circumscribe(dot[1000], shape=m.Circle, color=DDPM_C, buff=0.08), m.FadeIn(honest)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 6 · the paper's table

    def paper_table(self):
        self.next_section("6 · The paper's own numbers", notes=(
            "Song, Meng & Ermon (DDIM), Table 1, CIFAR-10 (32 × 32 colour photos), Inception FID (lower is better), the "
            "same trained model sampled with S steps; η = 0 is DDIM, η = 1 is DDPM-like with β̃ (notes §9). At 1,000 steps "
            f"the original DDPM sampler (the larger variance σ̂) scores {CIFAR_DDPM_1000} — better than DDIM's 4.04 "
            "(notes Appendix A: DDPM wins at 1,000 steps, DDIM wins at ≤ 100). The paper: 'high quality samples 10× to 50× "
            "faster in terms of wall-clock time'."))
        cx = [-2.6, -0.7, 0.85, 2.4, 3.95, 5.5]
        y_head, y0, y1 = 1.55, 0.6, -0.45
        box = card(12.6, 3.2, EDGE, CARD, radius=0.18, stroke=2).move_to([0, 0.35, 0])
        head = m.VGroup(text("steps", 22, MUTED).move_to([cx[0], y_head, 0]),
                        *[code(f"{s:,}", 24, TIME, weight=m.BOLD).move_to([x, y_head, 0]) for s, x in zip(CIFAR_STEPS, cx[1:])])
        head[0].align_to([-5.9, 0, 0], m.LEFT)
        for w in head[1:]:
            sit(w, baseline(head[1]))
        sit(head[0], baseline(head[1]))
        rule = m.Line([-5.95, y_head - 0.4, 0], [5.95, y_head - 0.4, 0]).set_stroke(EDGE, 2)

        def row(label, color, values, y):
            words = text(label, 22, color, weight=m.BOLD).move_to([0, y, 0]).align_to([-5.9, 0, 0], m.LEFT)
            nums = m.VGroup(*[code(f"{v:.2f}", 26, INK).move_to([x, y, 0]) for v, x in zip(values, cx[1:])])
            for nn in nums[1:]:
                sit(nn, baseline(nums[0]))
            return m.VGroup(words, nums)

        r0 = row("no fresh noise · η = 0", DDIM_C, CIFAR[0], y0)
        r1 = row("fresh noise · η = 1", DDPM_C, CIFAR[1], y1)
        assert r0[0].get_right()[0] < cx[1] - 0.5 and r1[0].get_right()[0] < cx[1] - 0.5
        ten_bad = m.SurroundingRectangle(r1[1][0], buff=0.1, corner_radius=0.08).set_stroke(RED, 3)
        ddpm_note = m.VGroup(text("the original noisy sampler at 1,000 steps:", 22, MUTED),
                             code(f"{CIFAR_DDPM_1000:.2f}", 26, GOOD, weight=m.BOLD)).arrange(m.RIGHT, buff=0.25)
        ddpm_note.move_to([0, -1.9, 0])
        sit(ddpm_note[1], baseline(ddpm_note[0]))
        title = text("CIFAR-10: 32 × 32 colour photos · Inception FID, lower is better", 22, INK).move_to([0, 2.45, 0])
        src = cite("Song, Meng & Ermon, 2020 · Table 1").move_to([0, -3.35, 0])
        assert box.get_top()[1] < title.get_bottom()[1] + 0.05 or True

        self.say(
            "[The original paper saw the same pattern on small colour photos]: [ten steps scored thirteen point three six], "
            "[fifty, four point six seven], [a thousand, four point zero four].",
            m.AnimationGroup(self.new_header("06 · the paper", "Small photos, same story"), m.FadeIn(box), m.FadeIn(head),
                             m.FadeIn(rule), m.FadeIn(title), m.FadeIn(src), m.FadeIn(r0[0])),
            m.FadeIn(r0[1][0], shift=0.1 * m.UP),
            m.FadeIn(r0[1][1:3], shift=0.1 * m.UP),
            m.FadeIn(r0[1][3:], shift=0.1 * m.UP),
        )
        self.say(
            "[With fresh noise, ten steps scored forty-one]; "
            "[only at a full thousand did the original noisy sampler win, at three point one seven].",
            m.AnimationGroup(m.FadeIn(r1, shift=0.1 * m.UP), m.Create(ten_bad)),
            m.FadeIn(ddpm_note, shift=0.1 * m.UP),
        )
        self.wait(0.5)
        self.clear()

    # ------------------------------------------------------------------ 7 · the catch on the swirl

    def the_catch(self):
        self.next_section("7 · Stranded between the arms", notes=(
            "toy.md §3 (MLP, the same 2,000 starting points; recomputed by precompute.py and asserted): the share of samples "
            "farther than 0.1 from the swirl's centre line (off_swirl) after DDIM with 100 / 50 / 20 / 10 steps: "
            f"{', '.join(f'{100 * OFF[s]:.1f}%' for s in (100, 50, 20, 10))}; the data: 0.1%. Shown: the first 1,000 of the "
            "2,000 samples (TEAL within 0.1, RED beyond). Paths: three particles DDIM-10 strands (> 0.15 away) whose "
            "1,000-step DDIM path ends on an arm (precompute.py's rule): the last big jump lands on the network's average "
            "guess x̂₀ (toy.md caveat: 'DDIM's big last jump … drops them between the arms'). Mean distance to the curve "
            f"(swirl_distance): data {DIST['data']:.5f}, DDPM-1000 {DIST['ddpm']:.5f}, DDIM-100 {DIST['ddim100']:.5f} — "
            "below the data: too tidy (the last step returns an average; for images, the slightly smooth look)."))
        PC, PS, lim = np.array([-3.25, -0.45, 0.0]), 5.6, 1.5
        pl = plane(PC, PS, lim)
        band = swirl_band(pl, 14, 0.16)
        n_show = 1000
        swarm = Swarm(pl, RUNS["ddim100_final"][:n_show])
        rx = 3.6
        bars = Bars([100 * OFF[s] for s in (100, 50, 20, 10)], labels=["100", "50", "20", "10"], top=50, width=4.0,
                    height=1.9, colors=[RED] * 4, gap=0.4, fmt="{:.1f}%", center=(rx, 0.75, 0))
        steps_l = cite("steps").next_to(bars.labels, m.LEFT, buff=0.2)
        bar_t = text("stranded: more than 0.1 from the curve", 22, MUTED).move_to([rx, 2.1, 0])
        legend = m.VGroup(m.VGroup(m.Dot(radius=0.08, color=DATA), cite("on the swirl", 22)).arrange(m.RIGHT, buff=0.14),
                          m.VGroup(m.Dot(radius=0.08, color=RED), cite("stranded", 22)).arrange(m.RIGHT, buff=0.14))
        legend.arrange(m.RIGHT, buff=0.45).move_to([PC[0], -3.4, 0])
        steps_now = Readout(-6.4, 2.2, "DDIM · {} steps", 100, 26, DDIM_C)

        # the stranded paths
        fine = RUNS["stranded_fine"][::10]
        hop = RUNS["stranded_ddim10"]
        p_fine = m.VGroup(*[path_line(pl, fine[:, j], DDIM_C, 4) for j in range(3)]).set_z_index(4)
        p_hop = m.VGroup(*[path_line(pl, hop[:-1, j], TIME, 2.6) for j in range(3)]).set_z_index(3)
        p_last = m.VGroup(*[path_line(pl, hop[-2:, j], RED, 4) for j in range(3)]).set_z_index(4)
        corners = m.VGroup(*[m.Dot(at(pl, q), radius=0.04, color=TIME) for j in range(3) for q in hop[:, j]]).set_z_index(4)
        e_fine = m.VGroup(*[m.Dot(at(pl, fine[-1, j]), radius=0.08, color=DATA).set_z_index(5) for j in range(3)])
        e_hop = m.VGroup(*[m.Dot(at(pl, hop[-1, j]), radius=0.08, color=RED).set_z_index(5) for j in range(3)])
        key = m.VGroup(m.VGroup(m.Line(ORIGIN_, [0.45, 0, 0]).set_stroke(TIME, 4), text("10 big steps", 20, TIME)).arrange(m.RIGHT, buff=0.15),
                       m.VGroup(m.Line(ORIGIN_, [0.45, 0, 0]).set_stroke(DDIM_C, 4), text("1,000 small steps", 20, DDIM_C)).arrange(m.RIGHT, buff=0.15))
        key.arrange(m.RIGHT, buff=0.5).move_to([PC[0], -3.4, 0])

        # the distance table
        dist_rows = m.VGroup()
        for label, v, color in (("the data itself", DIST["data"], DATA), ("1,000 noisy steps", DIST["ddpm"], DDPM_C),
                                ("DDIM, 100 steps", DIST["ddim100"], DDIM_C)):
            dist_rows.add(m.VGroup(text(label, 22, color), code(f"{v:.3f}", 24, INK, weight=m.BOLD)))
        for r, rw in enumerate(dist_rows):
            y = -1.6 - 0.48 * r
            rw[0].move_to([0, y, 0]).align_to([1.25, 0, 0], m.LEFT)
            rw[1].move_to([4.05, y, 0])
            sit(rw[1], baseline(rw[0]))
        dist_t = text("mean distance to the curve", 22, MUTED).move_to([rx, -1.08, 0])
        tidy = pill("too tidy", RED, RED_FILL, 20).next_to(dist_rows[2][1], m.RIGHT, buff=0.25)
        read = text("read it with the stranded share", 22, INK).move_to([rx, -3.2, 0])
        assert tidy.get_right()[0] < 6.6 and bars.get_bottom()[1] > dist_t.get_top()[1], (tidy.get_right(), bars.get_bottom(), dist_t.get_top())

        def to_steps(s, run_time=1.1):
            return m.AnimationGroup(swarm.go(RUNS[f"ddim{s}_final"][:n_show], run_time), m.Transform(steps_now.words, steps_now._make(s)))

        def bar(i):
            return m.AnimationGroup(m.GrowFromEdge(bars.bars[i], m.DOWN), m.FadeIn(bars.labels[i]), m.FadeIn(bars.numbers[i]))

        self.say(
            "[Back on the swirl, fewer steps have a price]: [points get stranded between the arms].",
            m.AnimationGroup(self.new_header("07 · the catch", "Stranded between the arms"), m.FadeIn(band),
                             m.FadeIn(steps_now), m.LaggedStart(*[m.FadeIn(d, scale=0.5) for d in swarm], lag_ratio=0.001, run_time=1.2)),
            m.AnimationGroup(m.FadeIn(legend), m.FadeIn(bar_t), m.FadeIn(bars.zero), m.FadeIn(steps_l)),
        )
        self.say(
            "[A hundred steps strand two point six percent], [fifty, seven and a half], [twenty, thirty-one], "
            "[and ten, forty-five percent].",
            bar(0),
            m.AnimationGroup(to_steps(50), bar(1)),
            m.AnimationGroup(to_steps(20), bar(2)),
            m.AnimationGroup(to_steps(10), bar(3)),
        )
        self.say(
            "[Why? Each big jump lands on the network's average guess], [and between two arms, that average falls in the gap].",
            m.AnimationGroup(swarm.animate.set_opacity(0.15), m.FadeOut(legend), m.FadeIn(key),
                             m.LaggedStart(*[m.Create(p) for p in p_fine], lag_ratio=0.2, run_time=1.4), m.FadeIn(e_fine)),
            m.AnimationGroup(m.LaggedStart(*[m.Create(p) for p in p_hop], lag_ratio=0.2, run_time=1.4), m.FadeIn(corners),
                             m.Succession(m.Wait(1.0), m.AnimationGroup(*[m.Create(p) for p in p_last], m.FadeIn(e_hop, scale=0.5)))),
        )
        self.say(
            "[One more trap]: [at a hundred steps, the average distance to the curve is zero point zero two zero], "
            "[below the data's own zero point zero two four].",
            m.AnimationGroup(m.FadeOut(m.VGroup(p_fine, p_hop, p_last, corners, e_fine, e_hop, key)), swarm.animate.set_opacity(0.9),
                             m.FadeIn(legend)),
            m.AnimationGroup(to_steps(100), m.FadeIn(dist_t), m.FadeIn(dist_rows[2])),
            m.AnimationGroup(m.FadeIn(dist_rows[0]), m.FadeIn(dist_rows[1])),
        )
        self.say(
            "[That is not better, it is too tidy]: [so always read it next to the stranded fraction].",
            m.FadeIn(tidy, scale=0.8),
            m.AnimationGroup(m.FadeIn(read, shift=0.1 * m.UP), m.Circumscribe(m.VGroup(bars.bars[0], bars.numbers[0]), color=RED, buff=0.08)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 8 · inversion

    def inversion(self):
        self.next_section("8 · Run it backwards", notes=(
            "Deterministic = invertible (DDIM §5.4, eq. 14 run from t = 0 to T). On screen: the DDIM step with s above t. INVERT_SRC (scene.py) is executed (asserted "
            "equal to precompute.py's run): 100 Euler steps from t = 0 up to t = 1000 with the swirl MLP, then ddim_sample "
            f"back down with the same 100 steps. Round-trip error on 300 swirl points: median {INV_MED[100]:.4f} with 100 "
            f"steps each way, {INV_MED[1000]:.5f} with 1,000 (the swirl's own thickness: noise std 0.03). Approximate: "
            "Euler error; DDIM Table 2 (CIFAR-10 reconstruction MSE): 0.014 at 10 steps … 0.0001 at 500–1,000. "
            "The twelve points: swirl(12, seed=21). Episode 16 (Editing and Control) inverts real pictures to edit them."))
        lim, size = 2.0, 4.4
        pl = plane((-4.4, -0.45, 0), size, lim)
        band = swirl_band(pl, 10, 0.18)
        frame = m.Square(size).set_stroke(EDGE, 2).move_to(pl.coords(0, 0))
        up, down = RUNS["inv_up"], RUNS["inv_down"]
        n = up.shape[1]
        pts = m.VGroup(*[m.Dot(at(pl, q), radius=0.07, color=DATA).set_z_index(5) for q in up[0]])
        rings = m.VGroup(*[m.Circle(radius=0.14).set_stroke(DATA, 2.5).move_to(at(pl, q)).set_z_index(4) for q in up[0]])
        trails = m.VGroup(*[m.VMobject().set_stroke(DDIM_C, 2.2, opacity=0.85) for _ in range(n)]).set_z_index(2)
        up_t = text("the same step, run upward", 24, INK, weight=m.BOLD)
        up_f = formula(r"x_s", "=", r"\sqrt{\bar\alpha_s}", r"\,\hat x_0", "+", r"\sqrt{1-\bar\alpha_s}", r"\,\hat\varepsilon",
                       size=40)
        up_f[2].set_color(TIME)
        up_f[5].set_color(TIME)
        up_f[3].set_color(MODEL)
        up_f[6].set_color(MODEL)
        up_s = code("s above t:  t = 0 → 1000, 100 steps", 20, TIME)
        up_body = m.VGroup(up_t, up_f, up_s).arrange(m.DOWN, buff=0.25)
        block = m.VGroup(card(up_body.width + 0.7, up_body.height + 0.6, EDGE, CARD, radius=0.16, stroke=2).move_to(up_body), up_body)
        block.move_to([2.2, 1.15, 0])
        assert block.get_right()[0] < 6.6 and block.get_left()[0] > frame.get_right()[0] + 0.2, block.get_left()
        res = m.VGroup(m.VGroup(code(f"{INV_MED[100]:.3f}", 30, INK, weight=m.BOLD), text("median miss, 100 steps each way", 22, MUTED)),
                       m.VGroup(code(f"{INV_MED[1000]:.4f}", 30, INK, weight=m.BOLD), text("with 1,000 steps", 22, MUTED)))
        for r, rw in enumerate(res):
            rw[0].move_to([0, -0.9 - 0.55 * r, 0]).align_to([-0.6, 0, 0], m.LEFT)
            rw[1].next_to(rw[0], m.RIGHT, buff=0.3)
            sit(rw[1], baseline(rw[0]))
        ep16 = pill("Episode 16 · Editing and Control: invert a picture, then edit it", PROMPT, PINK_FILL, 20).move_to([2.2, -2.45, 0])
        assert ep16.get_right()[0] < 6.6 and ep16.get_left()[0] > frame.get_right()[0] + 0.1, ep16.get_left()
        src = cite("approximate: the miss shrinks with more steps · DDIM paper, Table 2").move_to([2.2, -3.35, 0])
        assert src.get_left()[0] > frame.get_right()[0] + 0.1, src.get_left()
        cap = cite("the swirl's trained network · 12 data points").next_to(frame, m.DOWN, buff=0.2)

        def recolour(a, start, end):
            c = mix(start, end, a)
            for d in pts:
                d.set_color(c)

        def walk(path, seconds, start_c, end_c, draw):
            path = np.asarray(path, dtype=float)

            def step(_, a):
                i = a * (len(path) - 1)
                j = int(np.floor(i))
                k = min(j + 1, len(path) - 1)
                now = path[j] + (path[k] - path[j]) * (i - j)
                for idx, d in enumerate(pts):
                    d.move_to(at(pl, now[idx]))
                    if draw:
                        trail = [at(pl, q) for q in path[: j + 1, idx]] + [at(pl, now[idx])]
                        trails[idx].set_points_as_corners(trail if len(trail) > 1 else trail * 2)
                recolour(a, start_c, end_c)

            return m.UpdateFromAlphaFunc(m.VGroup(pts, trails) if draw else pts, step, run_time=seconds, rate_func=m.smooth)

        self.say(
            "[Deterministic also means reversible]: [run the same steps upward], [and each point turns into its own noise].",
            m.AnimationGroup(self.new_header("08 · inversion", "Run it backwards"), m.FadeIn(frame), m.FadeIn(band),
                             m.FadeIn(pts, scale=0.5), m.FadeIn(rings), m.FadeIn(cap)),
            m.FadeIn(block, shift=0.1 * m.UP),
            m.AnimationGroup(walk(up, 2.4, DATA, NOISE, True), m.FadeIn(src)),
        )
        self.say(
            "[Run them back down], [and the median point lands about a hundredth from where it started].",
            m.AnimationGroup(trails.animate.set_stroke(opacity=0.25), walk(down, 2.4, NOISE, DATA, False)),
            m.AnimationGroup(m.FadeIn(res, shift=0.1 * m.UP), m.LaggedStart(*[m.Indicate(r_, color=INK, scale_factor=1.3) for r_ in rings], lag_ratio=0.05)),
        )
        self.say(
            "[Episode sixteen uses this to edit real pictures]: [turn one into noise, then sample it back with a change].",
            m.FadeIn(ep16, shift=0.1 * m.UP),
            m.Circumscribe(ep16, color=PROMPT, buff=0.08),
        )
        self.wait(0.5)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes=(
            "Deterministic (η = 0: no fresh noise, the same noise gives the same picture), fast (DDIM-50: "
            f"{SPEED50:.0f}× faster than 1,000 noisy steps on this Mac at cFID {CFID_TXT[50]} vs {CFID_TXT[1000]}), reversible "
            "(inversion). Next: Episode 10, Solvers and Schedules — better ODE steps (Heun, DPM-Solver) and step placement."))
        halo = glow(TIME, radius=4.0)
        words = m.VGroup(*[title_text(w, 54, c) for w, c in (("Deterministic", DDIM_C), ("Fast", TIME), ("Reversible", DATA))])
        subs = m.VGroup(*[code(s, 20, MUTED) for s in ("same noise, same picture", f"50 steps, {SPEED50:.0f}× faster",
                                                      "picture ↔ noise")])
        words.arrange(m.RIGHT, buff=0.8)
        on_baseline(words)
        for w, s in zip(words, subs):
            s.set_x(w.get_x())
        line_y = min(w.get_bottom()[1] for w in words) - 0.45
        for s in subs:
            sit(s, line_y)
        mid = baseline(words[0]) + 0.3
        dots = m.VGroup(*[m.Dot([(l_.get_right()[0] + r_.get_left()[0]) / 2, mid, 0], radius=0.07, color=MUTED)
                          for l_, r_ in zip(words, words[1:])])
        row = m.VGroup(words, dots, subs).move_to(0.95 * m.UP)
        assert row.width < 12.6 and all(a.get_right()[0] + 0.3 < b.get_left()[0] for a, b in zip(subs, subs[1:])), (row.width, [(s_.get_left()[0], s_.get_right()[0]) for s_ in subs])
        f = formula(r"x_s", "=", r"\sqrt{\bar\alpha_s}", r"\,\hat x_0", "+", r"\sqrt{1-\bar\alpha_s}", r"\,\hat\varepsilon", size=46)
        f[2].set_color(TIME)
        f[5].set_color(TIME)
        f[3].set_color(MODEL)
        f[6].set_color(MODEL)
        f.next_to(row, m.DOWN, buff=0.6)
        tag = text("The same network, sampled as an ODE in big steps.", 28, MUTED)
        mascot = pixel(0.42)
        nxt = pill("NEXT · SOLVERS AND SCHEDULES", TIME, BLUE_FILL, 20, mono=True)
        foot = m.VGroup(tag, m.VGroup(mascot, nxt).arrange(m.RIGHT, buff=0.3)).arrange(m.DOWN, buff=0.35)
        foot.next_to(f, m.DOWN, buff=0.5)

        self.say(
            "[To sum up]: [deterministic], [fast], [reversible].",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.3 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(dots[0]), m.FadeIn(words[1], shift=0.3 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(dots[1]), m.FadeIn(words[2], shift=0.3 * m.UP), m.FadeIn(subs[2]), m.FadeIn(f)),
        )
        self.say(
            "[Same trained network, no fresh noise]: [the ODE, solved in fifty steps instead of a thousand].",
            m.Circumscribe(f[6], color=MODEL, buff=0.08),
            m.Write(tag),
        )
        self.say(
            "[Next, better ways to take those big steps].",
            m.AnimationGroup(m.FadeIn(mascot, shift=0.2 * m.UP), m.FadeIn(nxt, shift=0.2 * m.UP)),
        )
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, f, foot)], run_time=0.9)


ORIGIN_ = np.zeros(3)
