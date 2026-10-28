"""Out of the Noise · Episode 6 · The Score (Module 2 · Learning to Denoise)

Denoising is the same as learning which way is "more likely". Running example: the swirl and its EXACT score (the swirl
is a mixture of 256 round Gaussians, so diffusion_reference.mixture_score_sigma gives ∇log p_σ in closed form at every
noise level σ, no training). The episode draws the score as a LIME arrow field over the swirl's density (a TEAL glow):
sharp at σ = 0.05, smooth at σ = 1; shows why the log removes the normalising constant; runs Langevin dynamics
(langevin_sample) with one small noise level and with Song & Ermon's annealed ladder (10 levels σ = 1 → 0.01, 100
steps each, ε = 2e-5: their own recipe) — the very runs in research/experiments/toy.json (§6), recomputed by
precompute.py and asserted; then links the score to the denoiser: Tweedie's formula x̂0 = x + σ²·score, ε̂ = −σ·score
(DDPM units: score = −ε̂/√(1 − ᾱ_t), score_from_eps), denoising score matching, the trained swirl MLP's arrows vs the
exact ones, and the SDE view (Song et al. 2021). Episode 5 ("predicting the noise") is linked by number, not re-taught.

Every number on screen is computed below from diffusion_reference.py / the precomputed runs and asserted against
measured("toy", …), or cited with its source.

Sources (research/notes.md §6, Appendix C row 6; read October 2026):
- Song & Ermon 2019 (NCSN, arXiv:1907.05600): the Langevin step (eq. 4), §2.2 "Langevin dynamics can produce samples
  … using only the score function", §3 low-density regions, Algorithm 1 and §5: L = 10, σ 1 → 0.01 geometric,
  T = 100, ε = 2 × 10⁻⁵, uniform initial noise. Denoising score matching as stated there (their eq. 2), credited to
  Vincent 2011 (Neural Computation 23(7); the PDF itself was not retrieved — notes Appendix B).
- Efron 2011 (JASA 106(496)), eq. 1.4: Tweedie's formula, "first reported by Robbins in 1956".
- Ho et al. 2020 §3.2; Dhariwal & Nichol 2021 eq. 11: ∇log p = −ε/√(1 − ᾱ_t) (the sign: notes §6 misconceptions).
- Song, Sohl-Dickstein, Kingma, Kumar, Ermon & Poole 2021 (score SDE, ICLR 2021): eq. 5 (forward), eq. 6 (reverse,
  Anderson 1982), VE = NCSN, VP = DDPM, eq. 13 (probability-flow ODE).
- research/experiments/toy.md §2 (MLP vs exact arrows) and §6 (Langevin); toy.json headline langevin_*.

Illustrative choices (not data): the noisy point (0.1, 1.29) and its two readings σ = 0.2 / 0.7; the arrow lengths on
screen are squashed (tanh(σ·|score|), then scaled so the longest is fixed) — directions are exact; the glow is the exact
density p_σ, brightness ∝ (p/p_max)^0.6.

    python3 precompute.py                     # once (~10 s): writes langevin_runs.npz
    VOICE=draft manimgx inspect scene.py      # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import sys
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np

EP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EP_DIR.parent))
from kit import *  # noqa: E402,F403
from diffusion_reference import *  # noqa: E402,F403

warnings.filterwarnings("ignore", message=".*encountered in matmul")  # NumPy/Accelerate's spurious warnings

PREFIX = "ep06"
H = lambda *keys: measured("toy", "headline", *keys)  # noqa: E731
LV = lambda *keys: measured("toy", "langevin", *keys)  # noqa: E731

# ---------------------------------------------------------------------------- the swirl and its exact score

K = len(SWIRL.means)
assert K == 256 and np.allclose(SWIRL.stds, 0.03)                                  # "two hundred fifty-six tiny Gaussians"
SW_X0 = swirl_labelled(20000, seed=0)[0][:1000]                                    # Episode 4's thousand points
LIM = 1.5                                                                          # the plane shown: [−1.5, 1.5]²
FIELD_N, FIELD_MARGIN = 15, 0.06                                                   # Plot.field's grid
_ax = np.linspace(-LIM + FIELD_MARGIN * 2 * LIM, LIM - FIELD_MARGIN * 2 * LIM, FIELD_N)
GRID = np.array([(x, y) for y in _ax for x in _ax])


def typical(sigma):
    """The median length of the true score over the shown grid (the arrows on screen are squashed)."""
    return float(np.median(np.linalg.norm(mixture_score_sigma(GRID, sigma), axis=1)))


SIG_SHOW = (0.05, 0.3, 1.0)
TYP = {s: typical(s) for s in SIG_SHOW}
assert f"{TYP[0.05]:.1f}" == "87.9" and f"{TYP[0.3]:.1f}" == "4.2" and f"{TYP[1.0]:.2f}" == "0.89"
assert 90 < TYP[0.05] / TYP[1.0] < 110                                             # "about a hundred times shorter"

# the log removes the normaliser: ∇log(c·p) = ∇log p (checked on the swirl's density with finite differences)
_pt, _h = np.array([[0.3, -0.2]]), 1e-5
_lp = lambda q, c=1.0: np.log(c) + mixture_posterior(q, 1.0, 0.3).log_density[0]  # noqa: E731
_fd = [(_lp(_pt + _h * e, 7.0) - _lp(_pt - _h * e, 7.0)) / (2 * _h) for e in np.eye(2)]
assert np.allclose(_fd, mixture_score_sigma(_pt, 0.3)[0], rtol=1e-4)

# ---------------------------------------------------------------------------- Langevin (toy.json §6, precompute.py)

RUNS = np.load(EP_DIR / "langevin_runs.npz")
for _name in ("annealed", "plain_small"):
    assert np.abs(np.round(RUNS[f"{_name}_final"][:1000], 3) - np.array(LV("clouds", _name))).max() < 1e-9


def outer_share(points):
    """Share of the on-swirl particles on the outer half (by length) of the arms — toy_experiments.py's definition."""
    p = swirl_profile(points)
    return float(p[:, 10:].sum() / p.sum())


OUT_PLAIN, OUT_ANNEAL = outer_share(RUNS["plain_small_final"]), outer_share(RUNS["annealed_final"])
OUT_DATA = LV("data_outer_half_share")
assert round(OUT_PLAIN, 3) == H("langevin_plain_outer_share") == 0.796               # "eighty percent"
assert round(OUT_ANNEAL, 3) == H("langevin_annealed_outer_share") == 0.481           # "forty-eight percent"
assert OUT_DATA == 0.504 and round(100 * OUT_DATA) == 50                             # "the data's fifty"
ON_PLAIN = 1 - off_swirl(RUNS["plain_small_final"])
assert abs((1 - ON_PLAIN) - LV("runs", "plain_small", "off")) < 1e-9 and f"{100 * ON_PLAIN:.1f}" == "99.8"
SIG_LEVELS = np.geomspace(1.0, 0.01, 10)
assert np.allclose(np.unique(RUNS["annealed_sigma"])[::-1], SIG_LEVELS)              # "ten levels, one to a hundredth"
assert np.allclose(RUNS["plain_small_sigma"], 0.01) and len(RUNS["annealed_sigma"]) == 201  # 1,000 steps, every 5th kept
N_SHOWN, N_RUN, EVERY = RUNS["plain_small_states"].shape[1], len(RUNS["start"]), 5
assert (N_SHOWN, N_RUN) == (1000, 2000)


def level_of(i):
    """The annealing level (0 … 9) of stored state i (state i is step 5·i; step k ≥ 1 runs at level ⌈k/100⌉ − 1)."""
    return int(np.clip(np.ceil(EVERY * i / 100) - 1, 0, 9))


# ---------------------------------------------------------------------------- Tweedie, ε and the score

XN = np.array([0.1, 1.29])                                                         # one noisy point (illustrative)
SIG_A, SIG_B = 0.2, 0.7
POST = {s: mixture_posterior(XN[None], 1.0, s) for s in (SIG_A, SIG_B)}
for _s, _p in POST.items():
    assert np.allclose(_p.x0, XN + _s ** 2 * _p.score)                             # Tweedie: x̂0 = x + σ²·score
    assert np.allclose((XN - _p.x0) / _s, -_s * _p.score)                          # ε̂ = E[ε | x] = −σ·score
_d_a, _d_b = nearest_on_swirl(POST[SIG_A].x0)[0][0], nearest_on_swirl(POST[SIG_B].x0)[0][0]
_arm_b = POST[SIG_B].resp[0][SWIRL.labels == 0].sum()
assert _d_a < 0.02 and _d_b > 0.12 and 0.4 < _arm_b < 0.6                         # on an arm / "in the gap", both arms
_t, _ab = 100, alpha_bar_at(100)
_pts = q_sample(swirl(500, seed=3), _t, gaussian_noise(500, seed=4))
assert np.allclose(score_from_eps(exact_eps(_pts, _t), _ab), mixture_score(_pts, _t))  # score = −ε̂/√(1 − ᾱ)

# the trained swirl MLP (ε-prediction, toy.md §2): its arrows vs the exact ones on toy.json's 21 × 21 grid
MLP = load_swirl_model()
_g = np.linspace(-1.5, 1.5, 21)
_G21 = np.array([(x, y) for y in _g for x in _g])
assert np.allclose(_g, measured("toy", "score", "grid", "axis"))


def median_angle(t):
    ex, ml = mixture_score(_G21, t), score_from_eps(MLP(_G21, t), alpha_bar_at(t))
    cos = (ex * ml).sum(1) / np.linalg.norm(ex, axis=1) / np.linalg.norm(ml, axis=1)
    return float(np.median(np.degrees(np.arccos(np.clip(cos, -1, 1)))))


ANG = {t: median_angle(t) for t in (10, 100)}
_grid_err = {r["t"]: r["median_angle_deg"] for r in measured("toy", "score", "grid_error")}
assert all(abs(ANG[t] - _grid_err[t]) < 0.005 for t in ANG) and ANG[100] < 1.0     # "less than a degree off"
MLP_PARAMS, MLP_SECONDS = H("mlp_params"), H("mlp_train_seconds")
assert MLP_PARAMS == count_params(load_weights()) == 45954

# ---------------------------------------------------------------------------- look

PC = np.array([-3.25, -0.45, 0.0])     # the swirl plane's centre
PS = 5.6                               # … and its size
RX = 3.6                               # the right column's centre
LEFT = 1.15                            # the right column's left edge
FLAT = set("abcdefhiklmnorstuvwxzABCDEFGHIJKLMNOPRSTUVWXYZ0123456789")
OUTER = swirl_curve(201)[:, 100:]      # the outer half (by length) of each arm


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


def formula(*parts, size=44, color=INK):
    return m.MathTex(*parts, font_size=size, color=color)


def plane(center=PC, size=PS, lim=LIM):
    """A bare square plane for the swirl (no axes), data range [−lim, lim]²."""
    return Plot((-lim, lim), (-lim, lim), width=size, height=size, center=tuple(center), axes=False)


def squash(score_fn, noise):
    """Plot.field's fn: the score's direction, its length squashed to tanh(noise·|score|) so every level fits."""
    def fn(x, y):
        s = score_fn(np.array([[x, y]]))[0]
        n = float(np.hypot(*s))
        return s * (np.tanh(noise * n) / n) if n > 1e-12 else np.zeros(2)
    return fn


def field_at(plot, sigma, color=SCORE, width=2.6, max_len=None):
    step = plot.w * (1 - 2 * FIELD_MARGIN) / (FIELD_N - 1)
    return plot.field(squash(lambda p: mixture_score_sigma(p, sigma), sigma), nx=FIELD_N, ny=FIELD_N,
                      color=color, max_len=max_len or 0.82 * step, width=width, margin=FIELD_MARGIN)


@lru_cache(maxsize=16)
def glow_rgba(sigma, n=220, gamma=0.7, alpha=0.62):
    """The exact density p_σ over the plane as a TEAL glow (alpha ∝ (p/p_max)^γ), with a soft round vignette
    (radius 1.15 → 1.5) so a broad density does not show the square's edge."""
    ax = np.linspace(-LIM, LIM, n)
    X, Y = np.meshgrid(ax, ax[::-1])
    logp = mixture_posterior(np.stack([X.ravel(), Y.ravel()], 1), 1.0, sigma).log_density.reshape(n, n)
    d = np.exp(logp - logp.max()) ** gamma
    edge = np.clip((LIM - np.hypot(X, Y)) / 0.35, 0, 1)
    rgba = np.zeros((n, n, 4), np.uint8)
    rgba[..., :3] = [int(TEAL[i:i + 2], 16) for i in (1, 3, 5)]
    rgba[..., 3] = (255 * alpha * d * edge * edge * (3 - 2 * edge)).round().astype(np.uint8)
    rgba.flags.writeable = False
    return rgba


def glow_of(plot, sigma):
    img = m.ImageMobject(glow_rgba(sigma))
    img.height = plot.h
    return img.move_to(plot.coords(0, 0)).set_z_index(-2)


def at(plot, xy):
    return plot.coords(float(xy[0]), float(xy[1]))


def outer_band(plot):
    """The outer half (by length) of both arms as a wide, faint stroke behind the particles."""
    return m.VGroup(*[m.VMobject().set_points_smoothly([at(plot, p) for p in arm[::4]] + [at(plot, arm[-1])])
                      .set_stroke(INK, 26, opacity=0.2) for arm in OUTER]).set_z_index(-1)


class Readout(m.VGroup):
    """A mono number display (e.g. "σ = 0.05", "step 250"), left-aligned at a fixed point; .show(v) updates it."""

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


class Swarm:
    """The first 1,000 particles of a stored Langevin run, replayed; a particle turns from VIOLET (noise) toward
    TEAL (data) as it reaches the swirl (its distance to the centre line, from precompute.py)."""

    SHADES = 5

    def __init__(self, plot, run, radius=0.027):
        self.plot = plot
        self.xs = RUNS[f"{run}_states"].astype(float)
        self.ds = RUNS[f"{run}_dist"].astype(float)
        self.cloud = plot.cloud(self.xs[0], NOISE, radius, 0.95)
        self.palette = [mix(VIOLET, TEAL, k / (self.SHADES - 1)) for k in range(self.SHADES)]
        self.shade = np.zeros(len(self.cloud), int)

    def state(self, i):
        j = int(np.floor(i))
        k = min(j + 1, len(self.xs) - 1)
        return self.xs[j] + (self.xs[k] - self.xs[j]) * (i - j)

    def _recolor(self, i):
        j = int(round(i))
        lv = np.clip(np.round((0.18 - self.ds[j]) / 0.12 * (self.SHADES - 1)), 0, min(self.SHADES - 1, j // 2))
        lv = lv.astype(int)
        for k in np.flatnonzero(lv != self.shade):
            self.cloud[k].set_color(self.palette[lv[k]])
        self.shade = lv

    def put(self, i):
        self.cloud.positions(self.state(i))
        self._recolor(i)

    def replay(self, i0, i1, seconds, hooks=()):
        def step(_, a):
            i = i0 + (i1 - i0) * a
            for hook in hooks:
                hook(i)
            self.put(i)
        return m.UpdateFromAlphaFunc(self.cloud, step, run_time=seconds, rate_func=m.linear)


class Ladder(m.VGroup):
    """Song & Ermon's ten noise levels as dots on a line, σ = 1 (left) to 0.01 (right); .at(level) lights one."""

    def __init__(self, center, span=4.4):
        super().__init__()
        c = np.array(center, dtype=float)
        self.xs = c[0] - span / 2 + span * np.arange(10) / 9
        self.line = m.Line([self.xs[0], c[1], 0], [self.xs[-1], c[1], 0]).set_stroke(EDGE, 3)
        self.dots = m.VGroup(*[m.Dot([x, c[1], 0], radius=0.09, color=FAINT) for x in self.xs])
        self.lo = code("σ = 1", 22, TIME).next_to(self.dots[0], m.DOWN, buff=0.2)
        self.hi = code("0.01", 22, TIME).next_to(self.dots[-1], m.DOWN, buff=0.2)
        sit(self.hi, baseline(self.lo))
        self.cursor = m.Circle(radius=0.2).set_stroke(TIME, 3).move_to(self.dots[0])
        self.add(self.line, self.dots, self.lo, self.hi)
        self.level = -1

    def at(self, level):
        if level != self.level:
            for k, d in enumerate(self.dots):
                d.set_color(TIME if k == level else mix(FAINT, TIME, 0.45) if k < level else FAINT)
            self.cursor.move_to(self.dots[level])
            self.level = level
        return self


# ---------------------------------------------------------------------------- the episode


class Ep06TheScore(Episode):
    number = 6

    def lesson(self):
        self.hook()
        self.the_score()
        self.noise_levels()
        self.langevin()
        self.one_level()
        self.annealing()
        self.tweedie()
        self.network_check()
        self.sde_view()
        self.recap()

    # ------------------------------------------------------------------ 0 · hook

    def hook(self):
        self.next_section("Hook", notes=(
            "The swirl's 1,000 points are Episode 4's (the first 1,000 of swirl_labelled(20000, seed=0)). The noisy point "
            f"({XN[0]}, {XN[1]}) is illustrative. Its arrow and the field are the exact score ∇log p_σ at σ = 0.05 "
            "(mixture_score_sigma, the swirl blurred by a little noise): directions exact, lengths squashed to fit."))
        plot = plane()
        self.plot = plot
        dots = plot.cloud(SW_X0, DATA, 0.026, 0.9)
        dot = m.Dot(at(plot, XN), radius=0.1, color=NOISE).set_z_index(6)
        ring = m.Circle(radius=0.22).set_stroke(NOISE, 3).move_to(dot).set_z_index(6)
        point = m.VGroup(ring, dot)
        s = mixture_score_sigma(XN[None], 0.05)[0]
        hero = m.Arrow(dot.get_center(), dot.get_center() + 0.95 * np.array([*(s / np.linalg.norm(s)), 0]), buff=0.12,
                       color=SCORE, stroke_width=7, max_tip_length_to_length_ratio=0.3).set_z_index(5)
        fld = field_at(plot, 0.05)
        self.fld = fld
        title = title_text("the score", 66, SCORE).move_to([RX, 1.45, 0])
        sub = text("an arrow toward more likely data", 26, MUTED).next_to(title, m.DOWN, buff=0.3)
        rows = m.VGroup()
        for k, words in enumerate(("what the score is", "how arrows become samples", "why every denoiser knows it")):
            rows.add(m.VGroup(code(f"{k + 1}", 26, MODEL, weight=m.BOLD), text(words, 26, INK)).arrange(m.RIGHT, buff=0.3))
        rows.arrange(m.DOWN, buff=0.32, aligned_edge=m.LEFT).move_to([RX, -1.35, 0])
        for r in rows:
            sit(r[0], baseline(r[1]))
        self.hook_rows = m.VGroup(title, sub, rows)

        self.say(
            "[Here is the swirl again]: [a thousand points on two spiral arms], [and one noisy point, out in the empty space beside it].",
            self.new_header("00 · hook", "Which way is more likely?"),
            m.LaggedStart(*[m.FadeIn(d, scale=0.5) for d in dots], lag_ratio=0.002, run_time=1.4),
            m.FadeIn(point, scale=0.6),
        )
        self.say(
            "Ask the data one question about it: [which way is more likely]? [The answer is an arrow].",
            m.Circumscribe(point, shape=m.Circle, color=NOISE, buff=0.12),
            m.GrowArrow(hero),
        )
        self.say(
            "[Ask the same question everywhere, and you get a whole field of arrows]. [This field is called the score].",
            m.LaggedStart(*[m.GrowArrow(a) if isinstance(a, m.Arrow) else m.FadeIn(a) for a in fld], lag_ratio=0.008,
                          run_time=2.2),
            m.AnimationGroup(m.FadeIn(title, shift=0.2 * m.UP), m.FadeIn(sub)),
        )
        self.say(
            "Today: [what the score is], [how arrows become samples], [and why every denoiser already knows it].",
            m.FadeIn(rows[0], shift=0.1 * m.UP),
            m.FadeIn(rows[1], shift=0.1 * m.UP),
            m.FadeIn(rows[2], shift=0.1 * m.UP),
        )
        self.wait(0.4)
        self.swirl_dots, self.point, self.hero = dots, point, hero

    # ------------------------------------------------------------------ 1 · the score

    def the_score(self):
        self.next_section("1 · The score", notes=(
            "Score = ∇ₓ log p(x) (Hyvärinen 2005; Song & Ermon 2019 §2). The glow is the exact density of the swirl blurred "
            "by σ = 0.05 (mixture_posterior(...).log_density), brightness ∝ (p/p_max)^0.6. Why the log: p = p̃/Z, and "
            "∇log p = ∇log p̃ − ∇log Z = ∇log p̃ because Z is a constant (checked: ∇log(7·p) = ∇log p by finite differences). "
            "The swirl is a mixture of 256 round Gaussians (std 0.03) along the arms (diffusion_reference.SWIRL), so "
            "mixture_score_sigma(x, σ) is exact — no training."))
        plot = self.plot
        glow = glow_of(plot, 0.05)
        self.glow = glow
        legend = m.VGroup(
            m.VGroup(m.Dot(radius=0.1, color=TEAL), text("bright: likely", 22, MUTED)).arrange(m.RIGHT, buff=0.15),
            m.VGroup(m.Dot(radius=0.1, color=CARD_HI).set_stroke(EDGE, 2), text("dark: unlikely", 22, MUTED))
            .arrange(m.RIGHT, buff=0.15),
        ).arrange(m.RIGHT, buff=0.6).move_to([RX, 2.15, 0])
        sit(legend[1][1], baseline(legend[0][1]))
        f = formula(r"s(x)", "=", r"\nabla_x", r"\log", r"\,p(x)", size=60).move_to([RX, 1.2, 0])
        f[0].set_color(SCORE)
        f[2].set_color(SCORE)
        f[4].set_color(DATA)
        f_sub = text("the direction in which p rises fastest", 24, INK).next_to(f, m.DOWN, buff=0.3)

        z1 = formula(r"p(x)", "=", r"\frac{\tilde p(x)}{Z}", size=44).move_to([RX - 1.75, -0.55, 0])
        z1[0].set_color(DATA)
        z_lab = text("Z: the total that\nmakes p add up to one", 22, MUTED).next_to(z1, m.RIGHT, buff=0.35)
        assert z_lab.get_right()[0] < 6.6
        z2 = formula(r"\nabla_x \log p(x)", "=", r"\nabla_x \log \tilde p(x)", "-", r"\nabla_x \log Z", size=40)
        z2.move_to([RX, -1.75, 0])
        z2[0].set_color(SCORE)
        z2[4].set_color(BAD)
        zero = code("= 0", 24, BAD, weight=m.BOLD).next_to(z2[4], m.DOWN, buff=0.18)
        ok = pill("the score never needs Z", GOOD, GOOD_FILL, 22).move_to([RX, -3.0, 0])

        means = m.VGroup(*[m.Dot(at(plot, mu), radius=0.024, color=INK).set_opacity(0.9) for mu in SWIRL.means])
        means.set_z_index(3)
        mc1 = text(f"the swirl = {K} round Gaussians", 24, INK)
        mc1b = text("each of std 0.03, along the arms", 22, MUTED)
        mc2 = code("mixture_score_sigma(x, sigma)  # exact", 20, MUTED, t2c={r"#.*": FAINT, r"mixture_score_sigma": TEAL})
        body = m.VGroup(mc1, mc1b, mc2).arrange(m.DOWN, buff=0.18, aligned_edge=m.LEFT)
        mix_card = m.VGroup(card(body.width + 0.7, body.height + 0.6, EDGE, CARD, radius=0.16, stroke=2).move_to(body), body)
        mix_card.move_to([RX, -1.35, 0])

        self.say(
            "[Picture the swirl's density as a glow]: [bright where points are likely, dark where they are not].",
            m.AnimationGroup(self.new_header("01 · the score", "An arrow toward more likely"),
                             m.FadeOut(self.swirl_dots), m.FadeOut(self.hook_rows), m.FadeOut(self.hero),
                             m.FadeIn(glow)),
            m.FadeIn(legend, shift=0.1 * m.DOWN),
        )
        self.say(
            "[The score is the gradient of the log of that density]: [at every point, the direction in which it rises fastest].",
            m.Write(f),
            m.AnimationGroup(m.FadeIn(f_sub, shift=0.1 * m.UP),
                             self.fld.animate(rate_func=m.there_and_back, run_time=1.2).set_stroke(width=5)),
        )
        self.say(
            "[Why the log?] [A density must add up to one], [and for images, that total is impossible to compute].",
            m.Circumscribe(f[3], color=INK, buff=0.08),
            m.FadeIn(z1, shift=0.1 * m.UP),
            m.AnimationGroup(z1[2].animate.set_color(BAD), m.FadeIn(z_lab)),
        )
        self.say(
            "[The log splits it off as a separate constant], [and the slope of a constant is zero]. [So the score never needs it].",
            m.FadeIn(z2, shift=0.1 * m.UP),
            m.AnimationGroup(m.Circumscribe(z2[4], color=BAD, buff=0.08), m.FadeIn(zero), z2[4].animate.set_opacity(0.45)),
            m.FadeIn(ok, shift=0.1 * m.UP),
        )
        self.say(
            "[On the swirl, we can skip learning altogether]: [it is a mixture of two hundred fifty-six tiny Gaussians], "
            "[so its score has an exact formula].",
            m.FadeOut(m.VGroup(z1, z_lab, z2, zero, ok), shift=0.1 * m.DOWN),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(d, scale=0.4) for d in means], lag_ratio=0.004, run_time=1.4),
                             m.FadeIn(mix_card[0]), m.FadeIn(mc1), m.FadeIn(mc1b)),
            m.FadeIn(mc2, shift=0.1 * m.UP),
        )
        self.wait(0.4)
        self.play(m.FadeOut(m.VGroup(legend, f, f_sub, mix_card, means, self.point)), run_time=0.8)

    # ------------------------------------------------------------------ 2 · noise levels

    def noise_levels(self):
        self.next_section("2 · Sharp or smooth", notes=(
            "The score of the swirl blurred by Gaussian noise of std σ (the σ view, x = x0 + σ·ε; Song & Ermon 2019), exact "
            "at σ = 0.05, 0.3, 1 (mixture_score_sigma). The table is the median true |score| over the 15 × 15 grid of arrows "
            f"shown: {TYP[0.05]:.1f}, {TYP[0.3]:.2f}, {TYP[1.0]:.3f} (ratio {TYP[0.05] / TYP[1.0]:.0f}×). On screen every "
            "level's arrows are squashed (tanh(σ·|score|)) and scaled to one length: directions exact."))
        plot = self.plot
        sig = Readout(LEFT + 0.6, 2.05, "σ = {:g}", 0.05, 46)
        fx = formula("x", "=", r"x_0", "+", r"\sigma", r"\,\varepsilon", size=44).move_to([RX, 1.2, 0])
        fx[2].set_color(DATA)
        fx[4].set_color(TIME)
        fx[5].set_color(NOISE)

        box = card(4.8, 2.55, EDGE, CARD, radius=0.18, stroke=2).move_to([RX, -0.85, 0])
        cols = [box.get_left()[0] + 1.1, box.get_left()[0] + 3.3]
        top = box.get_top()[1] - 0.42
        head = m.VGroup(code("σ", 22, TIME, weight=m.BOLD), text("typical true arrow length", 22, MUTED))
        head[0].move_to([cols[0], top, 0])
        head[1].move_to([cols[1] - 0.25, top, 0])
        sit(head[0], baseline(head[1]))
        rule = m.Line([box.get_left()[0] + 0.3, top - 0.3, 0], [box.get_right()[0] - 0.3, top - 0.3, 0]).set_stroke(EDGE, 2)
        rows = m.VGroup()
        for k, s in enumerate(SIG_SHOW):
            y = top - 0.8 - 0.5 * k
            v = TYP[s]
            rows.add(m.VGroup(code(f"{s:g}", 24, TIME).move_to([cols[0], y, 0]),
                              code(f"{v:.1f}" if v >= 1 else f"{v:.2f}", 24, SCORE, weight=m.BOLD).move_to([cols[1], y, 0])))
        table = m.VGroup(box, head, rule)
        note = cite("on screen: lengths squashed to fit, directions exact").move_to([RX, -2.75, 0])

        # a gap where two neighbouring arrows point away from each other, toward different arms (σ = 0.05)
        sc = mixture_score_sigma(GRID, 0.05)
        best, pair = -1e9, None
        for a in range(len(GRID)):
            for b in (a + 1, a + FIELD_N):
                if b >= len(GRID) or (b == a + 1 and (a + 1) % FIELD_N == 0):
                    continue
                d = (GRID[b] - GRID[a]) / np.linalg.norm(GRID[b] - GRID[a])
                ua, ub = sc[a] / np.linalg.norm(sc[a]), sc[b] / np.linalg.norm(sc[b])
                mid = (GRID[a] + GRID[b]) / 2
                if ua @ d < -0.8 and ub @ d > 0.8 and min(np.linalg.norm(sc[a]), np.linalg.norm(sc[b])) * 0.05 > 0.7:
                    rank = -abs(float(np.hypot(*mid)) - 0.6) + 0.3 * mid[0]
                    if rank > best:
                        best, pair = rank, (a, b)
        assert pair is not None
        flip = m.SurroundingRectangle(m.VGroup(self.fld[pair[0]], self.fld[pair[1]]), buff=0.08, corner_radius=0.08)
        flip.set_stroke(INK, 3)
        glows = {s: glow_of(plot, s) for s in SIG_SHOW}
        fields = {s: field_at(plot, s) for s in SIG_SHOW[1:]}

        def to_sigma(s, run_time=1.4):
            old = self.glow
            self.glow = glows[s]
            return m.AnimationGroup(m.FadeOut(old), m.FadeIn(glows[s]), m.Transform(self.fld, fields[s]),
                                    m.Transform(sig.words, sig._make(s)), run_time=run_time)

        self.say(
            "[Which density, exactly?] [The data plus noise of size sigma], [so every noise level has its own score].",
            m.AnimationGroup(self.new_header("02 · noise levels", "Sharp or smooth"), m.FadeIn(sig)),
            m.Write(fx),
            m.AnimationGroup(m.FadeIn(table), m.FadeIn(rows[0], shift=0.1 * m.UP)),
        )
        self.say(
            "[At low noise, the field is sharp]: [arrows snap onto the nearest arm], [and split apart in the gaps between arms].",
            m.Circumscribe(sig, color=TIME, buff=0.1),
            self.fld.animate(rate_func=m.there_and_back, run_time=1.2).set_stroke(width=5),
            m.Create(flip),
        )
        self.say(
            "Turn the noise up: [at sigma point three, the arms blur together]; [at sigma one, the swirl is a single smooth hill], "
            "[and every arrow points to the middle].",
            m.AnimationGroup(m.FadeOut(flip), to_sigma(0.3), m.FadeIn(rows[1], shift=0.1 * m.UP)),
            m.AnimationGroup(to_sigma(1.0), m.FadeIn(rows[2], shift=0.1 * m.UP)),
            m.Flash(plot.coords(0, 0), color=SCORE, flash_radius=0.4, line_length=0.25),
        )
        self.say(
            "[On screen, the arrows are squashed to fit]: [the real ones at sigma one are about a hundred times shorter "
            "than at sigma point zero five].",
            m.FadeIn(note, shift=0.1 * m.UP),
            m.AnimationGroup(m.Circumscribe(rows[0], color=SCORE, buff=0.08), m.Circumscribe(rows[2], color=SCORE, buff=0.08)),
        )
        self.wait(0.4)
        self.play(m.FadeOut(m.VGroup(sig, fx, table, rows, note)), run_time=0.8)

    # ------------------------------------------------------------------ 3 · Langevin dynamics

    def langevin(self):
        self.next_section("3 · Langevin dynamics", notes=(
            "Langevin dynamics (Song & Ermon 2019 eq. 4): x ← x + (α/2)·∇log p_σ(x) + √α·z, z fresh noise. The run is "
            "toy.json's 'plain_small' (σ = 0.01 only, 1,000 steps, α = 2e-5, exact score), from uniform noise in "
            "[−1.5, 1.5]² (2,000 particles, rng 11; langevin_sample seed 12), recomputed by precompute.py and asserted "
            "equal to toy.json's stored cloud; the first 1,000 particles are drawn, every 5th step, coloured by distance to "
            "the swirl (VIOLET far → TEAL on it). Traces: 6 of the drawn particles. Without the noise term it is gradient "
            "ascent and collapses onto the modes (notes §6, misconceptions)."))
        plot = self.plot
        swarm = Swarm(plot, "plain_small")
        self.swarm = swarm
        f0 = field_at(plot, 0.01).set_opacity(0.45)
        f = formula("x", r"\leftarrow", "x", "+", r"\frac{\alpha}{2}", r"\nabla_x \log p_\sigma(x)", "+",
                    r"\sqrt{\alpha}\,z", size=42).move_to([RX - 0.2, 1.75, 0])
        f[5].set_color(SCORE)
        f[7].set_color(NOISE)
        l_score = text("a step along the score", 22, SCORE).next_to(f[5], m.DOWN, buff=0.3)
        l_noise = text("fresh noise", 22, NOISE).next_to(f[7], m.DOWN, buff=0.3)
        sit(l_noise, baseline(l_score))
        if l_noise.get_left()[0] < l_score.get_right()[0] + 0.25:
            l_noise.shift((l_score.get_right()[0] + 0.25 - l_noise.get_left()[0]) * m.RIGHT)
        if l_noise.get_right()[0] > 6.55:
            l_noise.shift((6.55 - l_noise.get_right()[0]) * m.RIGHT)
            l_score.next_to(l_noise, m.LEFT, buff=0.3)
        assert f.get_right()[0] < 6.6, f.get_right()
        name = title_text("Langevin dynamics", 40, INK).move_to([RX, 0.05, 0])
        counter = Readout(LEFT + 0.45, -0.85, "step {:d}", 0, 40)
        sett = cite("σ = 0.01 · α = 2 × 10⁻⁵ · 1,000 of 2,000 particles shown").move_to([RX, -1.55, 0])
        hill = text("no noise: plain hill climbing,\nevery particle piles onto a peak", 22, MUTED).move_to([RX, -2.55, 0])
        src = cite("Song & Ermon, 2019, §2.2").move_to([RX, -3.4, 0])

        start = swarm.xs[0]
        picks = []
        for target in ((-1.35, 0.3), (1.35, -0.35), (0.35, 1.35), (-0.4, -1.35), (0.05, 0.25), (-0.6, 0.05)):
            picks.append(int(np.argmin(np.linalg.norm(start - np.array(target), axis=1))))
        traces = m.VGroup(*[m.VMobject().set_stroke(SCORE, 2.2, opacity=0.75) for _ in picks]).set_z_index(4)

        def trace(i):
            j = max(1, int(np.ceil(i)))
            for tr, p in zip(traces, picks):
                pts = [at(plot, xy) for xy in swarm.xs[:j, p]] + [at(plot, swarm.state(i)[p])]
                tr.set_points_as_corners(pts)

        trace(0.001)

        def count(i):
            counter.show(int(round(EVERY * i)))

        self.say(
            "[To turn arrows into samples], [scatter particles at random].",
            m.AnimationGroup(self.new_header("03 · langevin", "From arrows to samples"), m.FadeOut(self.glow),
                             m.Transform(self.fld, f0)),
            m.LaggedStart(*[m.FadeIn(d, scale=0.4) for d in swarm.cloud], lag_ratio=0.0015, run_time=1.4),
        )
        self.say(
            "[Then repeat one small move]: [a step along the score], [plus a little fresh noise].",
            m.Write(f),
            m.AnimationGroup(m.Circumscribe(f[5], color=SCORE, buff=0.08), m.FadeIn(l_score, shift=0.1 * m.UP)),
            m.AnimationGroup(m.Circumscribe(f[7], color=NOISE, buff=0.08), m.FadeIn(l_noise, shift=0.1 * m.UP)),
        )
        self.say(
            "[This is Langevin dynamics.] [Watch the particles slide onto the spiral].",
            m.AnimationGroup(m.FadeIn(name, shift=0.1 * m.UP), m.FadeIn(counter), m.FadeIn(sett), m.FadeIn(traces)),
            swarm.replay(0, len(swarm.xs) - 1, 5.5, hooks=(count, trace)),
        )
        self.say(
            "[Without the fresh noise, it would be plain hill climbing], [every particle piling onto a peak]. "
            "[The noise is what makes it sample].",
            m.Circumscribe(m.VGroup(f[7], l_noise), color=NOISE, buff=0.1),
            m.FadeIn(hill, shift=0.1 * m.UP),
            m.FadeIn(src),
        )
        self.wait(0.4)
        self.play(m.FadeOut(m.VGroup(f, l_score, l_noise, name, counter, sett, hill, src, traces, self.fld)), run_time=0.8)
        self.picks = picks

    # ------------------------------------------------------------------ 4 · one noise level

    def one_level(self):
        self.next_section("4 · One noise level", notes=(
            f"The plain run's final 2,000 particles: {100 * ON_PLAIN:.1f}% within 0.1 of the swirl (off_swirl); of those, "
            f"{100 * OUT_PLAIN:.1f}% on the outer half of the arms by length (swirl_profile pieces 10–19 of 20; toy.json "
            f"headline langevin_plain_outer_share = 0.796). The data: {100 * OUT_DATA:.1f}% (toy.json data_outer_half_share, "
            "2,000 points of swirl(…, seed=1)). The paths are 6 drawn particles from start to finish. Song & Ermon 2019 §3: "
            "'the scarcity of training data in low data density regions … hinders the accuracy of score estimation and "
            "slows down the mixing of Langevin dynamics'. Our toy uses the EXACT score, so it shows the mixing problem "
            "(proportions) but not the learned-score problem (toy.md §6 caveats)."))
        plot, swarm = self.plot, self.swarm
        stat_n = title_text(f"{100 * ON_PLAIN:.1f}%", 60, INK)
        stat_l = text("of the particles end on the swirl", 24, MUTED)
        stat = m.VGroup(stat_n, stat_l).arrange(m.DOWN, buff=0.15).move_to([RX, 1.75, 0])
        done = text("Job done?", 26, INK, weight=m.BOLD).move_to([RX, -2.75, 0])
        band = outer_band(plot)
        bars = Bars([OUT_PLAIN, OUT_ANNEAL, OUT_DATA], labels=["one level", "annealed", "data"], top=1.0, width=4.4,
                    height=2.0, colors=[BAD, GOOD, DATA], gap=0.45, fmt="{:.0%}", center=(RX, -0.75, 0))
        self.bars = bars
        half = m.DashedLine([bars.xs[0] - bars.bar_w / 2 - 0.15, bars.level(0.5), 0],
                            [bars.xs[-1] + bars.bar_w / 2 + 0.15, bars.level(0.5), 0], dash_length=0.08).set_stroke(DATA, 2)
        self.half = half
        swatch = m.Line([0, 0, 0], [0.55, 0, 0]).set_stroke(INK, 14, opacity=0.25)
        bar_cap = m.VGroup(swatch, text("share on the outer half of the arms", 22, MUTED)).arrange(m.RIGHT, buff=0.2)
        bar_cap.move_to([RX, 0.75, 0])
        self.bar_cap = bar_cap
        part = lambda k: m.VGroup(bars.bars[k], bars.labels[k], bars.numbers[k])  # noqa: E731
        self.part = part

        ends = swarm.xs[-1]
        paths = m.VGroup()
        for p in self.picks:
            pts = [at(plot, xy) for xy in swarm.xs[::2, p]] + [at(plot, ends[p])]
            paths.add(m.VMobject().set_points_as_corners(pts).set_stroke(SCORE, 3, opacity=0.9))
        starts = m.VGroup(*[m.Dot(at(plot, swarm.xs[0, p]), radius=0.07, color=NOISE).set_z_index(5) for p in self.picks])
        square = m.DashedVMobject(m.Square(PS).move_to(PC), num_dashes=60).set_stroke(MUTED, 2)
        corners = m.VGroup(*[title_text("?", 48, BAD).move_to(plot.coords(sx * 1.25, sy * 1.25))
                             for sx in (-1, 1) for sy in (-1, 1)])
        learned = text("far from the data, a learned score\nhas never seen an example", 22, INK).move_to([RX, -2.75, 0])
        learned_src = cite("Song & Ermon, 2019, §3 · our swirl uses the exact score").next_to(learned, m.DOWN, buff=0.16)

        self.say(
            "[After a thousand steps at one small noise level], [almost every particle is on the swirl]. [Job done?]",
            self.new_header("04 · one level", "Right place, wrong amounts"),
            m.FadeIn(stat, shift=0.1 * m.UP),
            m.FadeIn(done, shift=0.1 * m.UP),
        )
        self.say(
            "[Count where they landed]: [eighty percent sit on the outer half of the arms]. [The data puts just half there].",
            m.AnimationGroup(m.FadeOut(done), m.Create(band, run_time=1.2), m.FadeIn(bar_cap),
                             m.FadeIn(bars.zero)),
            m.AnimationGroup(m.GrowFromEdge(bars.bars[0], m.DOWN), m.FadeIn(bars.labels[0]), m.FadeIn(bars.numbers[0])),
            m.AnimationGroup(m.GrowFromEdge(bars.bars[2], m.DOWN), m.FadeIn(bars.labels[2]), m.FadeIn(bars.numbers[2]),
                             m.Create(half)),
        )
        self.say(
            "[The outer turns are closest to most starting points], [and the arrows only ever say: go to the nearest arm].",
            m.AnimationGroup(m.Create(square), m.FadeIn(starts, scale=0.5)),
            m.LaggedStart(*[m.Create(p) for p in paths], lag_ratio=0.15, run_time=2.0),
        )
        self.say(
            "[A learned score is worse off]: [far from the data, it never saw an example], [so its arrows out there are guesses].",
            m.AnimationGroup(m.FadeOut(stat), m.FadeIn(learned, shift=0.1 * m.UP)),
            m.LaggedStart(*[m.FadeIn(c, scale=0.6) for c in corners], lag_ratio=0.2),
            m.FadeIn(learned_src),
        )
        self.wait(0.4)
        self.play(m.FadeOut(m.VGroup(paths, starts, square, corners, learned, learned_src, band, swarm.cloud,
                                     bars.zero, part(0), part(2), half, bar_cap)), run_time=0.8)

    # ------------------------------------------------------------------ 5 · annealing

    def annealing(self):
        self.next_section("5 · Annealing", notes=(
            "Annealed Langevin dynamics (Song & Ermon 2019, Algorithm 1; §5: L = 10 levels, σ 1 → 0.01 geometric, "
            "T = 100 steps each, ε = 2 × 10⁻⁵, α_i = ε·σ_i²/σ_L², uniform initial noise) — exactly toy.json's 'annealed' run "
            "(same start, same seed, exact score), recomputed by precompute.py and asserted. The view zooms out while the "
            "big-noise levels spread the cloud (scale = 2.68/max|x| of the drawn particles, smoothed, capped at the normal "
            "view); the faint ring has radius 1. The arrows are the exact score at the current level (squashed). Result: "
            f"{100 * OUT_ANNEAL:.1f}% on the outer half (headline langevin_annealed_outer_share 0.481) vs one level "
            f"{100 * OUT_PLAIN:.1f}% and the data {100 * OUT_DATA:.1f}%. NCSN: 10 × 100 = 1,000 network calls per sample."))
        plot = self.plot
        swarm = Swarm(plot, "annealed")
        xs = swarm.xs
        S0 = PS / 2 / LIM
        ext = np.abs(xs).max(axis=(1, 2))
        win = 6
        ext = np.array([ext[max(0, j - win):j + win + 1].max() for j in range(len(ext))])
        ext = np.convolve(np.pad(ext, win, mode="edge"), np.ones(2 * win + 1) / (2 * win + 1), mode="valid")
        scale_of = np.minimum(S0, (PS / 2 - 0.12) / ext)
        zoom = {"scale": S0, "level": -1, "built": S0}
        ring = m.Circle(radius=S0).set_stroke(FAINT, 2.5).move_to(PC)
        fld = field_at(plot, SIG_LEVELS[0]).set_opacity(0.45)

        c_title = text("Song & Ermon, 2019 · NCSN", 24, INK, weight=m.BOLD)
        c_rows = m.VGroup(code("10 noise levels, σ = 1 → 0.01", 20, MUTED), code("100 Langevin steps at each", 20, MUTED),
                          code("= 1,000 network calls per sample", 20, MUTED))
        c_rows.arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT)
        c_body = m.VGroup(c_title, c_rows).arrange(m.DOWN, buff=0.2, aligned_edge=m.LEFT)
        card_box = card(c_body.width + 0.7, c_body.height + 0.5, EDGE, CARD, radius=0.16, stroke=2).move_to(c_body)
        ncsn = m.VGroup(card_box, c_body).move_to([RX, 1.6, 0])
        ladder = Ladder((RX, -0.25, 0), span=4.4)
        steps_note = code("× 100 steps each", 22, TIME).move_to([RX, -1.15, 0])
        same = code("= the same 1,000 steps", 22, INK).next_to(steps_note, m.DOWN, buff=0.18)
        counter = Readout(LEFT + 1.2, -2.85, "step {:d}", 0, 34)
        sig_now = Readout(LEFT + 1.2, -2.2, "σ = {:.2f}", 1.0, 34)
        ladder.at(0)

        def look(i):
            s = float(np.interp(i, np.arange(len(scale_of)), scale_of))
            r = PS / 2 / s
            plot.x0, plot.x1, plot.y0, plot.y1 = -r, r, -r, r
            if abs(s - zoom["scale"]) > 1e-9:
                ring.become(m.Circle(radius=s).set_stroke(FAINT, 2.5).move_to(PC))
                zoom["scale"] = s
            lv = level_of(i)
            if lv != zoom["level"] or abs(s / zoom["built"] - 1) > 0.025:   # the arrows always fill the window
                fld.become(field_at(plot, SIG_LEVELS[lv]).set_opacity(0.45))
                zoom["built"] = s
            if lv != zoom["level"]:
                ladder.at(lv)
                sig_now.show(SIG_LEVELS[lv])
                zoom["level"] = lv
            counter.show(int(round(EVERY * i)))

        def reset():
            plot.x0, plot.x1, plot.y0, plot.y1 = -LIM, LIM, -LIM, LIM

        split = 4 * 100 / EVERY      # the end of level 4 (σ = 0.13)
        self.say(
            "[The fix, from Song and Ermon in twenty nineteen]: [start at big noise, then turn it down level by level].",
            m.AnimationGroup(self.new_header("05 · annealing", "Big noise first"), m.FadeIn(ncsn, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(ladder.line), m.FadeIn(ladder.lo), m.FadeIn(ladder.hi),
                             m.LaggedStart(*[m.FadeIn(d, scale=0.5) for d in ladder.dots], lag_ratio=0.12)),
        )
        self.say(
            "[Ten levels, from sigma one down to one hundredth], [a hundred steps at each]: [the same thousand steps as before].",
            m.AnimationGroup(m.Create(ladder.cursor), m.Circumscribe(c_rows[0], color=TIME, buff=0.06)),
            m.FadeIn(steps_note, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(same, shift=0.1 * m.UP), m.FadeIn(ring), m.FadeIn(fld),
                             m.LaggedStart(*[m.FadeIn(d, scale=0.4) for d in swarm.cloud], lag_ratio=0.0015, run_time=1.2)),
        )
        self.say(
            "[At big noise, the smooth field gathers everyone into the right broad shape]; [then each smaller level sharpens it].",
            m.AnimationGroup(m.FadeIn(counter), m.FadeIn(sig_now),
                             swarm.replay(0, split, 4.2, hooks=(look,))),
            swarm.replay(split, len(xs) - 1, 4.4, hooks=(look,)),
        )
        reset()
        bars, part = self.bars, self.part
        band = outer_band(plot)
        good = pill("big noise first", GOOD, GOOD_FILL, 22).move_to([RX, -2.85, 0])
        self.say(
            "[Now forty-eight percent land on the outer half], [right next to the data's fifty].",
            m.AnimationGroup(m.FadeOut(m.VGroup(ncsn, ladder, ladder.cursor, steps_note, same, counter, sig_now, fld, ring)),
                             m.Create(band), m.FadeIn(self.bar_cap), m.FadeIn(bars.zero), m.FadeIn(part(0)),
                             m.FadeIn(part(2)), m.FadeIn(self.half),
                             m.GrowFromEdge(bars.bars[1], m.DOWN), m.FadeIn(bars.labels[1]), m.FadeIn(bars.numbers[1])),
            m.AnimationGroup(m.Circumscribe(part(1), color=GOOD, buff=0.08), m.Circumscribe(part(2), color=DATA, buff=0.08)),
        )
        self.say(
            "[Same exact score, same thousand steps]: [the only difference is starting at big noise].",
            m.Circumscribe(m.VGroup(part(0), part(1)), color=INK, buff=0.1),
            m.FadeIn(good, shift=0.1 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 6 · Tweedie

    def tweedie(self):
        self.next_section("6 · The denoiser knows the score", notes=(
            f"One noisy point x = ({XN[0]}, {XN[1]}) (illustrative), read at σ = {SIG_A} and {SIG_B} (σ view, x = x0 + σ·ε). "
            "The 256 component centres are lit by their exact posterior weight p(component | x) (mixture_posterior; "
            "brightness ∝ √(w/w_max)). The best guess E[x0 | x] (the exact denoiser) and Tweedie's formula "
            "x̂0 = x + σ²·∇log p_σ(x) are asserted equal (Efron 2011 eq. 1.4; 'first reported by Robbins in 1956', who "
            f"credits Tweedie). At σ = {SIG_B} the weights split {_arm_b:.0%} / {1 - _arm_b:.0%} between the arms and the guess "
            f"lands {_d_b:.2f} from the curve (in a gap). ε̂ = E[ε | x] = (x − x̂0)/σ = −σ·score (asserted); in DDPM units "
            "(Episode 5): ∇log p_t(x_t) = −ε̂/√(1 − ᾱ_t) (Dhariwal & Nichol 2021 eq. 11; score_from_eps, asserted against "
            "exact_eps at t = 100). Denoising score matching: Vincent 2011, as stated by Song & Ermon 2019 (eq. 2): target "
            "∇log q_σ(x̃ | x) = −(x̃ − x)/σ² = −ε/σ."))
        plot = plane()
        self.plot = plot
        means = m.VGroup(*[m.Dot(at(plot, mu), radius=0.03, color=DATA).set_opacity(0.35) for mu in SWIRL.means])
        dot = m.Dot(at(plot, XN), radius=0.1, color=NOISE).set_z_index(6)
        ring = m.Circle(radius=0.22).set_stroke(NOISE, 3).move_to(dot).set_z_index(6)
        x_lab = tex("x", 36, NOISE).next_to(ring, m.LEFT, buff=0.12)
        point = m.VGroup(ring, dot, x_lab)
        ep5 = pill("Episode 5 · Learning to Undo Noise", MODEL, AMBER_FILL, 22).move_to([RX, 2.3, 0])
        sig = Readout(LEFT, 1.45, "σ = {:g}", SIG_A, 40)

        def weighted(s_):
            w = POST[s_].resp[0]
            w = np.sqrt(w / w.max())
            return m.VGroup(*[m.Dot(at(plot, mu), radius=0.03 + 0.012 * wk, color=mix(FAINT, DATA, wk))
                              .set_opacity(0.2 + 0.8 * wk) for mu, wk in zip(SWIRL.means, w)])

        def guess(s_):
            return at(plot, POST[s_].x0[0])

        g_dot = m.Dot(guess(SIG_A), radius=0.1, color=MODEL).set_z_index(7)

        def score_arrow(s_):
            return m.Arrow(dot.get_center(), guess(s_), buff=0.14, color=SCORE, stroke_width=6,
                           max_tip_length_to_length_ratio=0.3).set_z_index(5)

        s_arrow = score_arrow(SIG_A)

        # the legend (nothing is written over the swirl)
        def key(marker, words):
            return m.VGroup(marker, words).arrange(m.RIGHT, buff=0.22)

        short = lambda c: m.Arrow([0, 0, 0], [0.5, 0, 0], buff=0, color=c, stroke_width=5,  # noqa: E731
                                  max_tip_length_to_length_ratio=0.35)
        keys = m.VGroup(key(m.Dot(radius=0.09, color=NOISE), text("the noisy point", 22, INK)),
                        key(m.Dot(radius=0.09, color=MODEL), text("the best guess", 22, INK)),
                        key(short(SCORE), tex(r"\sigma^2 \cdot \mathrm{score}", 30, SCORE)),
                        key(short(NOISE), tex(r"\sigma \cdot \hat\varepsilon", 30, NOISE)))
        keys.arrange(m.DOWN, buff=0.16, aligned_edge=m.LEFT).move_to([5.25, 1.3, 0]).align_to([4.35, 0, 0], m.LEFT)
        for k in keys:
            k[0].set_x(4.25)
            k[1].next_to(k[0], m.RIGHT, buff=0.22)
        assert max(k.get_right()[0] for k in keys) < 6.6

        tw = formula(r"\hat x_0", "=", "x", "+", r"\sigma^2", r"\,\nabla_x \log p_\sigma(x)", size=46).move_to([RX, -0.25, 0])
        tw[0].set_color(MODEL)
        tw[2].set_color(NOISE)
        tw[4].set_color(TIME)
        tw[5].set_color(SCORE)
        tw_src = cite("Tweedie’s formula · Robbins, 1956 · Efron, 2011").next_to(tw, m.DOWN, buff=0.2)

        eps_f = formula(r"\hat\varepsilon", "=", "-", r"\sigma", r"\cdot", r"\nabla_x \log p_\sigma(x)", size=44)
        eps_f.move_to([RX, -1.35, 0])
        eps_f[0].set_color(NOISE)
        eps_f[3].set_color(TIME)
        eps_f[5].set_color(SCORE)
        src = "\n".join(["def score_from_eps(eps, ab):", "    return -eps / np.sqrt(1 - ab)"])
        block = code_block(src, 20, title="diffusion_reference.py · DDPM").move_to([RX, -2.62, 0])

        dsm_t = text("denoising score matching", 24, INK, weight=m.BOLD)
        dsm_f = formula(r"\Big\| s_\theta(\tilde x) + \frac{\tilde x - x}{\sigma^2} \Big\|^2", size=30)
        dsm_s = cite("Vincent, 2011 · as stated by Song & Ermon, 2019")
        dsm_body = m.VGroup(dsm_t, dsm_f, dsm_s).arrange(m.DOWN, buff=0.08)
        dsm_box = card(dsm_body.width + 0.6, dsm_body.height + 0.34, MODEL, AMBER_FILL, radius=0.16, stroke=2).move_to(dsm_body)
        dsm = m.VGroup(dsm_box, dsm_body).move_to([RX, -2.65, 0])
        assert dsm.get_bottom()[1] > -3.55 and block.get_bottom()[1] > -3.55, (dsm.get_bottom(), block.get_bottom())
        assert dsm.get_top()[1] < eps_f.get_bottom()[1] and keys.get_bottom()[1] > tw.get_top()[1]

        g_b = guess(SIG_B)
        new_arrow = score_arrow(SIG_B)
        perp = np.array([-(g_b - dot.get_center())[1], (g_b - dot.get_center())[0], 0.0])
        perp = 0.22 * perp / np.linalg.norm(perp)
        e_arrow = m.Arrow(g_b + perp, dot.get_center() + perp, buff=0.14, color=NOISE, stroke_width=6,
                          max_tip_length_to_length_ratio=0.3).set_z_index(5)

        self.say(
            "[So where does the score come from for real pictures?] [From the denoiser of episode five].",
            m.AnimationGroup(self.new_header("06 · tweedie", "The denoiser knows the score"),
                             m.LaggedStart(*[m.FadeIn(d, scale=0.5) for d in means], lag_ratio=0.003, run_time=1.2)),
            m.FadeIn(ep5, shift=0.1 * m.UP),
        )
        self.say(
            "[Take one noisy point.] [Every clean point that could have made it gets a weight], [and the best guess is their average].",
            m.AnimationGroup(m.FadeOut(ep5), m.FadeIn(point, scale=0.6), m.FadeIn(sig), m.FadeIn(keys[0])),
            m.Transform(means, weighted(SIG_A), run_time=1.4),
            m.AnimationGroup(m.FadeIn(g_dot, scale=0.5), m.FadeIn(keys[1], shift=0.1 * m.LEFT)),
        )
        self.say(
            "[That guess is the noisy point, plus sigma squared times the score]: [this is Tweedie's formula].",
            m.AnimationGroup(m.GrowArrow(s_arrow), m.FadeIn(keys[2], shift=0.1 * m.LEFT)),
            m.AnimationGroup(m.Write(tw), m.FadeIn(tw_src)),
        )
        self.say(
            "[Read the same point at more noise]: [the weights spread over both arms], [and the guess lands in the gap between them].",
            m.Transform(sig.words, sig._make(SIG_B)),
            m.Transform(means, weighted(SIG_B), run_time=1.4),
            m.AnimationGroup(g_dot.animate.move_to(g_b), m.Transform(s_arrow, new_arrow)),
        )
        self.say(
            "[In episode five, the network predicts the noise, epsilon]. [The score is that prediction, flipped and scaled].",
            m.AnimationGroup(m.FadeIn(eps_f[0]), m.FadeIn(eps_f[1])),
            m.AnimationGroup(m.FadeIn(eps_f[2:], shift=0.1 * m.LEFT), m.FadeIn(block, shift=0.1 * m.UP)),
        )
        self.say(
            "[Mind the minus sign]: [epsilon points from the clean guess out to the noisy point], [and the score points back].",
            m.Circumscribe(eps_f[2], color=BAD, buff=0.08),
            m.AnimationGroup(m.GrowArrow(e_arrow), m.FadeIn(keys[3], shift=0.1 * m.LEFT)),
            m.Circumscribe(s_arrow, color=SCORE, buff=0.08),
        )
        self.say(
            "[This is denoising score matching, from twenty eleven]: [learning to remove noise is learning the score].",
            m.AnimationGroup(m.FadeOut(block, shift=0.1 * m.DOWN), m.FadeIn(dsm, shift=0.1 * m.UP)),
            m.Circumscribe(dsm_f, color=MODEL, buff=0.08),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 7 · the network's arrows

    def network_check(self):
        self.next_section("7 · A network draws the same arrows", notes=(
            "The swirl's tiny MLP (toy.md §2: 45,954 parameters, trained on this Mac in NumPy with DDPM's simple ε-loss, "
            "40,000 steps, 95 s), turned into a score with score_from_eps, vs the exact score mixture_score, both at DDPM "
            f"step t = 100 (σ = 0.34), on 15 × 15 arrows (squashed the same way). Median angle on toy.json's 21 × 21 grid "
            f"over [−1.5, 1.5]²: {ANG[100]:.3f}° at t = 100, {ANG[10]:.2f}° at t = 10 (recomputed, asserted against "
            "toy.json score/grid_error)."))
        t = 100
        ab = alpha_bar_at(t)
        noise = float(np.sqrt(1 - ab))
        size = 4.2
        pl = plane((-4.45, -0.75, 0), size)
        pr = plane((-0.1, -0.75, 0), size)
        step = size * (1 - 2 * FIELD_MARGIN) / (FIELD_N - 1)
        exact = pl.field(squash(lambda p: mixture_score(p, t), noise), nx=FIELD_N, ny=FIELD_N, color=SCORE, width=2.4,
                         max_len=0.82 * step, margin=FIELD_MARGIN)
        net = pr.field(squash(lambda p: score_from_eps(MLP(p, t), ab), noise), nx=FIELD_N, ny=FIELD_N, color=MODEL,
                       width=2.4, max_len=0.82 * step, margin=FIELD_MARGIN)
        frames = m.VGroup(*[m.Square(size).set_stroke(EDGE, 2).move_to(p.coords(0, 0)) for p in (pl, pr)])
        lt = text("exact score", 24, SCORE, weight=m.BOLD).next_to(frames[0], m.UP, buff=0.22)
        rt = text("the network, as a score", 24, MODEL, weight=m.BOLD).next_to(frames[1], m.UP, buff=0.22)
        sit(rt, baseline(lt))
        tt = code(f"both at step t = {t}", 20, MUTED).next_to(frames, m.DOWN, buff=0.22)
        rx = 4.6
        ang = title_text(f"{ANG[100]:.2f}°", 64, INK).move_to([rx, 0.55, 0])
        ang_l = text("median angle between\nthe two arrows", 22, MUTED).next_to(ang, m.DOWN, buff=0.25)
        ang10 = code(f"{ANG[10]:.1f}° at t = 10", 22, INK).next_to(ang_l, m.DOWN, buff=0.35)
        mac = cite(f"{MLP_PARAMS:,} parameters · trained on\nthis Mac in {MLP_SECONDS:.0f} s · toy.md §2").next_to(ang10, m.DOWN, buff=0.4)

        self.say(
            "[We checked on the swirl]: [a small network trained on this Mac to predict noise, turned into arrows], "
            "[is typically less than a degree off the exact ones].",
            m.AnimationGroup(self.new_header("07 · check", "A network draws the same arrows"), m.FadeIn(frames[0]),
                             m.FadeIn(lt), m.LaggedStart(*[m.FadeIn(a) for a in exact], lag_ratio=0.004, run_time=1.2)),
            m.AnimationGroup(m.FadeIn(frames[1]), m.FadeIn(rt), m.FadeIn(tt),
                             m.LaggedStart(*[m.FadeIn(a) for a in net], lag_ratio=0.004, run_time=1.6)),
            m.AnimationGroup(m.FadeIn(ang, shift=0.15 * m.UP), m.FadeIn(ang_l), m.FadeIn(ang10), m.FadeIn(mac)),
        )
        self.wait(0.6)
        self.clear()

    # ------------------------------------------------------------------ 8 · the SDE view

    def sde_view(self):
        self.next_section("8 · One equation", notes=(
            "Song, Sohl-Dickstein, Kingma, Kumar, Ermon & Poole 2021 (score SDE, ICLR 2021): forward SDE dx = f(x,t)dt + "
            "g(t)dw (eq. 5); reverse-time SDE dx = [f − g²·∇log p_t(x)]dt + g dw̄ (eq. 6, Anderson 1982) — the only unknown "
            "is the score. VE SDE (eq. 9) = Song & Ermon's noise-level ladder (NCSN); VP SDE (eq. 11) = DDPM. Every SDE has "
            "a probability-flow ODE (eq. 13) with the same marginals and no noise (Episode 9). The mini clouds: 300 swirl "
            "points and the same points at t = 1000 (q_sample, linear schedule)."))
        x0 = swirl(300, seed=2)
        xT = q_sample(x0, 1000, gaussian_noise(300, seed=3))

        def mini(xy, color, center, s):
            c = np.array(center, dtype=float)
            return m.VGroup(*[m.Dot(c + s * np.array([x, y, 0]), radius=0.018, color=color) for x, y in xy])

        y1, y2 = 1.3, -0.8
        f1 = formula(r"\mathrm{d}x", "=", r"f(x,t)\,\mathrm{d}t", "+", r"g(t)\,\mathrm{d}w", size=40)
        f2 = formula(r"\mathrm{d}x", "=", r"\big[f(x,t) - g(t)^2", r"\nabla_x \log p_t(x)", r"\big]\,\mathrm{d}t",
                     "+", r"g(t)\,\mathrm{d}\bar w", size=40)
        f2[3].set_color(SCORE)
        f1.move_to([0, y1, 0]).align_to([-2.85, 0, 0], m.LEFT)
        f2.move_to([0, y2, 0]).align_to([-2.85, 0, 0], m.LEFT)
        assert f2.get_right()[0] < 6.6
        sw, bl = -6.0, -4.05
        m1 = m.VGroup(mini(x0, DATA, (sw, y1, 0), 0.42), mini(xT, NOISE, (bl, y1, 0), 0.19))
        m2 = m.VGroup(mini(xT, NOISE, (sw, y2, 0), 0.19), mini(x0, DATA, (bl, y2, 0), 0.42))
        assert m1.get_left()[0] > -6.6 and m2.get_right()[0] < -3.0
        a1 = arrow([sw + 0.5, y1, 0], [bl - 0.62, y1, 0], MUTED)
        a2 = arrow([sw + 0.62, y2, 0], [bl - 0.5, y2, 0], MUTED)
        l1 = text("adding noise", 22, MUTED).move_to([(sw + bl) / 2, y1 + 0.85, 0])
        l2 = text("running it backwards", 22, MUTED).move_to([(sw + bl) / 2, y2 + 0.85, 0])
        under = text("the score", 24, SCORE, weight=m.BOLD).next_to(f2[3], m.DOWN, buff=0.3)
        pills = m.VGroup(pill("noise-level ladder (Song & Ermon, 2019)", INK, CARD, 20),
                         pill("DDPM (Ho et al., 2020)", INK, CARD, 20)).arrange(m.RIGHT, buff=0.4).move_to([0, -2.4, 0])
        ode = pill("noise-free twin: the probability flow ODE · Episode 9", TIME, BLUE_FILL, 20).move_to([0, -3.15, 0])
        src = cite("Song et al., 2021").move_to([5.6, 2.25, 0])

        self.say(
            "[In twenty twenty-one, Song and colleagues joined both views into one]: [adding noise is a stochastic differential equation].",
            m.AnimationGroup(self.new_header("08 · the SDE view", "One equation for both"), m.FadeIn(src),
                             m.FadeIn(m1[0]), m.FadeIn(l1)),
            m.AnimationGroup(m.GrowArrow(a1), m.FadeIn(m1[1], target_position=m1[0]), m.Write(f1)),
        )
        self.say(
            "[Run it backwards], [and the only thing you need to know is the score].",
            m.AnimationGroup(m.FadeIn(m2[0]), m.FadeIn(l2), m.GrowArrow(a2), m.FadeIn(m2[1], target_position=m2[0]),
                             m.Write(f2)),
            m.AnimationGroup(m.Circumscribe(f2[3], color=SCORE, buff=0.08), m.FadeIn(under, shift=0.1 * m.UP)),
        )
        self.say(
            "[The noise-level ladder and DDPM are two versions of it]; [its noise-free twin, the probability flow ODE, "
            "returns in episode nine].",
            m.LaggedStart(*[m.FadeIn(p, shift=0.1 * m.UP) for p in pills], lag_ratio=0.4),
            m.FadeIn(ode, shift=0.1 * m.UP),
        )
        self.wait(0.5)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes=(
            "Arrows (the score ∇log p points toward more likely data), annealing (Langevin at big noise first, then small: "
            "48% vs 80% on the outer half, data 50%), denoising (Tweedie: x̂0 = x + σ²·score; ε̂ = −σ·score). Next: Episode 7, "
            "Sampling Step by Step."))
        halo = glow(MODEL, radius=4.0)
        words = m.VGroup(*[title_text(w, 58, c) for w, c in (("Arrows", SCORE), ("Annealing", TIME), ("Denoising", MODEL))])
        subs = m.VGroup(*[code(s, 20, MUTED) for s in ("toward likely data", "big noise first", "it knows the score")])
        words.arrange(m.RIGHT, buff=0.85)
        on_baseline(words)
        for w, s in zip(words, subs):
            s.set_x(w.get_x())
        line_y = min(w.get_bottom()[1] for w in words) - 0.45
        for s in subs:
            sit(s, line_y)
        mid = baseline(words[0]) + 0.3
        dots = m.VGroup(*[m.Dot([(l.get_right()[0] + r.get_left()[0]) / 2, mid, 0], radius=0.07, color=MUTED)
                          for l, r in zip(words, words[1:])])
        row = m.VGroup(words, dots, subs).move_to(0.95 * m.UP)
        assert row.width < 12.6 and all(a.get_right()[0] + 0.3 < b.get_left()[0] for a, b in zip(subs, subs[1:]))
        f = formula(r"\hat\varepsilon", "=", r"-\sigma", r"\cdot", r"\nabla_x \log p_\sigma(x)", size=48)
        f[0].set_color(NOISE)
        f[2].set_color(TIME)
        f[4].set_color(SCORE)
        f.next_to(row, m.DOWN, buff=0.6)
        tag = text("Predicting the noise is estimating the score.", 28, MUTED)
        mascot = pixel(0.42)
        nxt = pill("NEXT · SAMPLING STEP BY STEP", MODEL, AMBER_FILL, 20, mono=True)
        foot = m.VGroup(tag, m.VGroup(mascot, nxt).arrange(m.RIGHT, buff=0.3)).arrange(m.DOWN, buff=0.35)
        foot.next_to(f, m.DOWN, buff=0.5)

        self.say(
            "[To sum up]: [arrows], [annealing], [denoising].",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.3 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(dots[0]), m.FadeIn(words[1], shift=0.3 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(dots[1]), m.FadeIn(words[2], shift=0.3 * m.UP), m.FadeIn(subs[2]), m.FadeIn(f)),
        )
        self.say(
            "[The score is an arrow toward more likely data], [and a network that predicts noise already draws it].",
            m.Circumscribe(f[4], color=SCORE, buff=0.08),
            m.Write(tag),
        )
        self.say(
            "[Next, we put those arrows to work], [and walk from pure noise to a picture].",
            m.FadeIn(mascot, shift=0.2 * m.UP),
            m.FadeIn(nxt, shift=0.2 * m.UP),
        )
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, f, foot)], run_time=0.9)
