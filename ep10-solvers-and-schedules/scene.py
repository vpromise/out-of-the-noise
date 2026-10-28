"""Out of the Noise · Episode 10 · Solvers and Schedules (Module 3 · Faster Sampling)

Better ODE solvers and better step placement buy quality per network call. Running example: the swirl
(diffusion_reference: its exact denoiser and the trained 45,954-parameter MLP), then the lighthouse generated on this
Mac as the real-world version.

The episode: sampling without fresh noise follows the probability-flow ODE (Episode 9); a sampler integrates it in
steps and big straight steps cut the corners (six particles, DDIM-5 vs the exact path). Euler (one network call per
step; = DDIM in the σ view, Episode 9) and its first-order error (mean miss vs the exact ODE solution, toy.md §3:
0.32 / 0.15 / 0.079 / 0.015 at 5 / 10 / 20 / 100 steps). Heun taken apart on one big step (σ 0.743 → 0.120): trial
Euler step, a second look, the average slope (2 calls per step, 1 on the last: N steps = 2N − 1 calls). Step
placement: the noise levels of 20 steps evenly spaced in DDPM time under the linear and the cosine schedule
(Episode 4) and Karras et al.'s σ^(1/ρ) spacing with ρ = 7 (karras_sigmas, σ 157 → 0.01 = our schedule's range);
the same solver (Euler in σ) and the same trained network with 20 calls strand 31.2% / 13.2% / 4.8% of the samples.
The race by network calls (exact denoiser, Karras grid): Euler's miss halves per doubling; Heun leads from 49 calls
(0.014 vs 0.030; 0.0027 vs 0.015 at 99/100) and strands 0.8% with the trained network (DDPM 1,000: 0.7%), but
diverges at 9 and 5 calls (0.79 and 4.8 from the swirl even with the exact denoiser). Published: EDM (FID 1.79 at
35 network calls), DPM-Solver (10–20 calls) and DPM-Solver++ (15–20 guided steps). The lighthouse at 1 / 2 / 4 / 8 /
20 / 50 Euler steps, guidance 7.5, with measured times (median of 4 seeds).

Every number on screen is computed below (diffusion_reference.py, precompute.py's runs, the run's images) and
asserted, or loaded with measured(...), or cited with its source.

Sources (research/notes.md §10, §9, §4, Appendix A row 10, Appendix C row 10; abstracts re-read October 2026):
- Karras, Aittala, Aila & Laine 2022 (EDM, arXiv:2206.00364, NeurIPS 2022): eq. 5 (σ_i, ρ = 7, σ 0.002 → 80),
  Algorithm 1 (Heun; the last step to σ = 0 is Euler), abstract "new state-of-the-art FID of 1.79 for CIFAR-10 in a
  class-conditional setting … (35 network evaluations per image)"; 35 = 18 Heun steps (notes, derived).
- Lu, Zhou, Bao, Chen, Li & Zhu 2022 (DPM-Solver, arXiv:2206.00927): "high-quality samples in only 10 to 20 function
  evaluations"; "4.70 FID in 10 function evaluations and 2.87 FID in 20 … on the CIFAR10 dataset".
- Lu et al. 2022 (DPM-Solver++, arXiv:2211.01095, 2 Nov 2022): "previous high-order fast samplers suffer from
  instability issues"; "high-quality samples within only 15 to 20 steps for guided sampling".
- Nichol & Dhariwal 2021: the cosine schedule (Episode 4). Song et al. 2021: the probability-flow ODE (Episode 9).
- research/experiments/toy.md §3 and toy.json (samplers/exact/runs, samplers/mlp/runs; headline euler_karras20_mlp_off,
  heun_karras49_mlp_off, ddim20_mlp_off), recomputed by precompute.py and asserted.
- research/experiments/mac_runs.md §5 and mac_runs.json (lighthouse/steps_sweep: SD 2.1-base, 512×512, guidance 7.5,
  seeds 0–3, the mlx-examples SimpleEulerSampler: Euler in σ on evenly spaced timesteps, no fresh noise; CFG = two
  network passes per step, batched). The Mac was shared while timing: ±5–10% jitter (mac_runs.md §8).

Not claimed (toy.md caveats): that the σ coordinate is what makes DDIM good (Euler written in DDPM time is about as
accurate on this toy); here the lesson is step PLACEMENT. Heun is not "always better": it diverges at ≤ 9 calls.

Illustrative (computed, but chosen): the six hook particles (one per 60° of the swirl, DDIM-5 strands them), the hero of
section 1, the particle and the step taken apart in section 2 (its Euler/Heun miss ratio is 3.7; across all 2,000
points at that step the median ratio is 1.9).

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

PREFIX = "ep10"
IMAGES = EXPERIMENTS / "images" / "lighthouse"
EXR = lambda key: measured("toy", "samplers", "exact", "runs", key)  # noqa: E731
MLR = lambda key: measured("toy", "samplers", "mlp", "runs", key)  # noqa: E731
TOY = lambda *keys: measured("toy", "headline", *keys)  # noqa: E731
SMIN, SMAX = sigma_from_alpha_bar(ALPHA_BAR[[1, T]])
assert f"{SMIN:.2f}" == "0.01" and f"{SMAX:.0f}" == "157"
HOOK = [652, 587, 1837, 1639, 1485, 223]
HERO = 1485


def pct(x):
    return f"{100 * x:.1f}%"


# ---------------------------------------------------------------------------- the swirl runs (precompute.py)

RUNS = np.load(EP_DIR / "swirl_runs.npz")
Z = RUNS["z"]
assert np.allclose(Z, gaussian_noise(2000, seed=5))                                    # toy.md's shared noise
REF = RUNS["ref_final"]
miss = lambda x: float(np.linalg.norm(x - REF, axis=1).mean())  # noqa: E731

# section 1: Euler's mean miss vs the exact ODE solution (DDIM = Euler in σ, evenly spaced; exact denoiser)
EULER_MISS = {s: EXR(f"ddim_{s}")["ode_error"] for s in (5, 10, 20, 100)}
for _s, _v in EULER_MISS.items():
    assert abs(miss(RUNS[f"exact_ddim_{_s}"]) - _v) < 1e-4 * max(1, _v)
MISS_TXT = {5: "0.32", 10: "0.15", 20: "0.079", 100: "0.015"}
assert all(f"{EULER_MISS[s]:.2g}" == MISS_TXT[s] for s in MISS_TXT), EULER_MISS
assert 1.8 < EULER_MISS[5] / EULER_MISS[10] < 2.2 and 1.8 < EULER_MISS[10] / EULER_MISS[20] < 2.2   # "halves"
_z3 = gaussian_noise(300, seed=5)
assert np.abs(ddim_sample(exact_eps, _z3, 10).xs - euler_ode(exact_eps, _z3, 10).xs).max() < 1e-9    # Euler in σ = DDIM

# section 2: one big step taken apart
STEP_SIG = RUNS["step_sigmas"]
STEP = {k: RUNS[f"step_{k}"] for k in ("start", "trial", "other", "heun")}
SEG = RUNS["step_seg"]
STEP_END = SEG[-1]
MISS_E, MISS_H = float(np.linalg.norm(STEP["trial"] - STEP_END)), float(np.linalg.norm(STEP["heun"] - STEP_END))
assert (f"{STEP_SIG[0]:.2f}", f"{STEP_SIG[1]:.2f}") == ("0.74", "0.12")
assert (f"{MISS_E:.2f}", f"{MISS_H:.3f}") == ("0.20", "0.055") and 3.5 < MISS_E / MISS_H < 4
assert np.allclose(STEP["heun"], (STEP["trial"] + STEP["other"]) / 2)

# section 3: where 20 steps fall, and what that does (same solver, same trained network, same noise)
SIG_LIN = sigma_from_alpha_bar(ALPHA_BAR[ddim_timesteps(20)[:-1]])
SIG_COS = sigma_from_alpha_bar(alpha_bar_at(RUNS["cos_grid_20"][:-1]))     # the noise levels the run visited
SIG_KAR = karras_sigmas(20, SMIN, SMAX)[:-1]
assert np.allclose(SIG_COS[1:], sigma_from_alpha_bar(ALPHA_BAR_COSINE[ddim_timesteps(20)[1:-1]]))
assert np.isclose(SIG_COS[0], SMAX) and np.isclose(SIG_LIN[0], SMAX) and np.isclose(SIG_KAR[0], SMAX)
LAST = {"lin": SIG_LIN[-1], "cos": SIG_COS[-1], "kar": SIG_KAR[-1]}
assert (f"{LAST['lin']:.2f}", f"{LAST['cos']:.2f}", f"{LAST['kar']:.2f}") == ("0.17", "0.09", "0.01")
assert (SIG_COS < 1).sum() > (SIG_LIN < 1).sum()                                    # cosine: more steps at low noise
assert np.all(np.diff(SIG_KAR[-7:]) > np.diff(SIG_KAR[-8:-1]))                      # Karras: shrinking steps at the end
STRAND = {"lin": off_swirl(RUNS["mlp_ddim_20"]), "cos": off_swirl(RUNS["mlp_cos_20"]),
          "kar": off_swirl(RUNS["mlp_euler_karras_20"])}
assert STRAND["lin"] == MLR("ddim_20")["off"] == TOY("ddim20_mlp_off")
assert STRAND["kar"] == MLR("euler_karras_20")["off"] == TOY("euler_karras20_mlp_off")
assert (pct(STRAND["lin"]), pct(STRAND["cos"]), pct(STRAND["kar"])) == ("31.2%", "13.2%", "4.8%")

# section 4: the race (exact denoiser, Karras grid): mean miss vs network calls (toy.json, recomputed by precompute.py)
E_NFE, H_NFE = [5, 10, 20, 50, 100], [5, 9, 19, 49, 99]
E_ERR = [EXR(f"euler_karras_{n}")["ode_error"] for n in E_NFE]
H_ERR = [EXR(f"heun_karras_{n}")["ode_error"] for n in H_NFE]
for _n, _v in zip(E_NFE, E_ERR):
    assert abs(miss(RUNS[f"exact_euler_karras_{_n}"]) - _v) < 1e-4 * max(1, _v)
for _n, _v in zip(H_NFE, H_ERR):
    assert abs(miss(RUNS[f"exact_heun_karras_{_n}"]) - _v) < 1e-4 * max(1, _v)
assert H_ERR[2] > E_ERR[2] and H_ERR[3] < 0.5 * E_ERR[3]                            # behind at 19, leads at 49
LEAD99 = E_ERR[4] / H_ERR[4]
assert 5 < LEAD99 < 6 and f"{LEAD99:.1f}" == "5.6"                                   # "more than five times smaller"
assert all(1.6 < a / b < 2.6 for a, b in zip(E_ERR[1:3], E_ERR[2:4]))               # Euler: ÷2 per doubling (2.5× at 20→50)
H49_OFF, DDPM_OFF = off_swirl(RUNS["mlp_heun_karras_49"]), off_swirl(RUNS["mlp_ddpm_1000"])
assert H49_OFF == TOY("heun_karras49_mlp_off") and DDPM_OFF == MLR("ddpm_1000")["off"]
assert (pct(H49_OFF), pct(DDPM_OFF)) == ("0.8%", "0.7%")
DIV9, DIV5 = swirl_distance(RUNS["exact_heun_karras_9"]), swirl_distance(RUNS["exact_heun_karras_5"])
assert abs(DIV9 - EXR("heun_karras_9")["dist"]) < 1e-4 and abs(DIV5 - EXR("heun_karras_5")["dist"]) < 1e-3
assert (f"{DIV9:.2f}", f"{DIV5:.1f}") == ("0.79", "4.8")
OFF9 = off_swirl(RUNS["exact_heun_karras_9"])
assert pct(OFF9) == "93.3%"
DATA_DIST = TOY("data_dist")
assert f"{DATA_DIST:.3f}" == "0.024"

# section 6: the lighthouse, generated on this Mac (mac_runs.md §5)
SWEEP = [r for r in measured("mac_runs", "lighthouse", "steps_sweep") if "stable-diffusion-2-1" in r["model"]]
LH_STEPS = [1, 2, 4, 8, 20, 50]
LH_TIME = {s: float(np.median([r["total_s"] for r in SWEEP if r["steps"] == s])) for s in LH_STEPS}
assert all(len([r for r in SWEEP if r["steps"] == s]) == 4 for s in LH_STEPS)       # median of 4 seeds
assert [f"{LH_TIME[s]:.2f}" for s in LH_STEPS] == ["0.33", "0.48", "0.75", "1.34", "2.93", "6.90"]
assert all(r["cfg"] == 7.5 for r in SWEEP) and LH_TIME[20] < 0.5 * LH_TIME[50]
LH_FILES = {s: IMAGES / f"lighthouse_sd21_steps{s:02d}_cfg7.5_seed0.png" for s in LH_STEPS}
assert all(p.exists() for p in LH_FILES.values())
assert any(r["file"] == LH_FILES[50].name and r["seed"] == 0 for r in SWEEP)

# ---------------------------------------------------------------------------- look

FLAT = set("abcdefhiklmnorstuvwxzABCDEFGHIJKLMNOPRSTUVWXYZ0123456789")
EULER_C = LIME                        # Euler (= DDIM, Episode 9's colour): a path made of straight steps
HEUN_C = INK                          # Heun: look twice
EXACT_C = MUTED                       # the exact path (dashed)
CALL_C = MODEL                        # one network call
LIN_C, COS_C, KAR_C = TIME, mix(BLUE, INK, 0.7), GREEN   # linear (Episode 4's blue), cosine (its lighter blue), Karras
PC, PS, LIM = np.array([-3.2, -0.42, 0.0]), 5.5, 1.45    # the swirl plane (left), its size and data half-width
RX = 3.65                                                 # the right column's centre


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


def plane(center=PC, size=PS, lim=LIM):
    """A bare square plane for the swirl (no axes), data range [−lim, lim]²."""
    return Plot((-lim, lim), (-lim, lim), width=size, height=size, center=tuple(center), axes=False)


def at(plot, xy):
    return plot.coords(float(xy[0]), float(xy[1]))


def zoom_window(pts, center=PC, size=PS, margin=1.25, pad=0.04):
    """A square Plot framing data points `pts` (n × 2), centred on `center`; returns (plot, data centre, half-width)."""
    pts = np.asarray(pts, dtype=float)
    c = (pts.min(0) + pts.max(0)) / 2
    half = margin * float((pts.max(0) - pts.min(0)).max()) / 2 + pad
    plot = Plot((c[0] - half, c[0] + half), (c[1] - half, c[1] + half), width=size, height=size, center=tuple(center),
                axes=False)
    return plot, c, half


def clipped_arms(plot, c, half, size, opacity=0.13):
    """The swirl's arms inside a zoom window, drawn at their true width (two noise stds either side)."""
    arms = m.VGroup()
    width_px = 4 * SWIRL_NOISE * size / (2 * half) * 135 / 2
    for arm in swirl_curve(1600):
        inside = (np.abs(arm[:, 0] - c[0]) < half * 0.97) & (np.abs(arm[:, 1] - c[1]) < half * 0.97)
        runs, cur = [], []
        for q, ok in zip(arm, inside):
            if ok:
                cur.append(q)
            elif cur:
                runs.append(cur)
                cur = []
        if cur:
            runs.append(cur)
        for r in runs:
            if len(r) > 3:
                arms.add(m.VMobject().set_points_smoothly([at(plot, q) for q in r[::4]] + [at(plot, r[-1])])
                         .set_stroke(DATA, width_px, opacity=opacity))
    return arms.set_z_index(-1)


def swirl_band(plot, width=12, opacity=0.18):
    """The swirl's two arms as wide faint TEAL strokes: where the data lives."""
    return m.VGroup(*[m.VMobject().set_points_smoothly([at(plot, p) for p in arm[::3]] + [at(plot, arm[-1])])
                      .set_stroke(DATA, width, opacity=opacity) for arm in swirl_curve(200)]).set_z_index(-1)


def path_line(plot, pts, color, width=3.0, opacity=0.95):
    return m.VMobject().set_points_as_corners([at(plot, p) for p in pts]).set_stroke(color, width, opacity=opacity)


def dashed_path(plot, pts, color=EXACT_C, width=3.0, dashes=36):
    return m.DashedVMobject(path_line(plot, pts, color, width), num_dashes=dashes).set_stroke(color, width)


def stranded(xy):
    return nearest_on_swirl(np.atleast_2d(xy))[0] > 0.1


def end_dot(plot, xy, radius=0.07):
    return m.Dot(at(plot, xy), radius=radius, color=RED if stranded(xy)[0] else DATA).set_z_index(5)


def call_dots(plot, pts, radius=0.05):
    return m.VGroup(*[m.Dot(at(plot, p), radius=radius, color=CALL_C).set_z_index(4) for p in pts])


def grow_steps(plot, paths, lines, heads, calls, seconds, rate=lambda a: float(a) ** 0.6):
    """Walk sampler paths (states × n × 2) step by step: each line follows its head; the AMBER call dot at the start
    of a step lights up when the step begins (calls[n][k] for step k of particle n)."""
    paths = np.asarray(paths, dtype=float)
    steps = len(paths) - 1

    def step(_, a):
        i = a * steps
        j = min(int(np.floor(i)), steps - 1)
        f = i - j
        now = paths[j] + (paths[j + 1] - paths[j]) * f
        for n, (line, head) in enumerate(zip(lines, heads)):
            pts = [at(plot, p) for p in paths[: j + 1, n]] + [at(plot, now[n])]
            line.set_points_as_corners(pts if len(pts) > 1 else pts + pts)
            head.move_to(at(plot, now[n]))
            for k, d in enumerate(calls[n]):
                d.set_opacity(1.0 if k <= j else 0.0)

    anim = m.UpdateFromAlphaFunc(m.VGroup(lines, heads, calls), step, run_time=seconds, rate_func=rate)
    anim.introducer = True          # the lines and call dots appear at this cue, not when the line starts
    return anim


class Swarm(m.VGroup):
    """Swirl samples as dots: VIOLET while still noise, then TEAL on an arm or RED stranded (> 0.1 from the curve).
    Dots that leave the plane's square fade out. `.go(xy)` moves every dot and recolours it on the way."""

    STEPS = 8

    def __init__(self, plot, xy, radius=0.026, noise=False, lim=LIM):
        super().__init__()
        self.plot, self.lim = plot, lim
        self.xy = np.asarray(xy, dtype=float)
        self.cols = [NOISE] * len(self.xy) if noise else self._colours(self.xy)
        for p, c in zip(self.xy, self.cols):
            d = m.Dot(at(plot, self._clip(p)), radius=radius, color=c)
            self.add(d.set_opacity(self._alpha(p)))

    def _clip(self, p):
        return np.clip(p, -self.lim - 0.2, self.lim + 0.2)

    def _alpha(self, p):
        out = max(abs(p[0]), abs(p[1])) - (self.lim - 0.08)
        return 0.9 * float(np.clip(1 - out / 0.14, 0, 1))

    @staticmethod
    def _colours(xy):
        return [RED if s else DATA for s in stranded(xy)]

    def go(self, xy, run_time=1.2):
        start, end = self.xy.copy(), np.asarray(xy, dtype=float)
        c0, c1 = self.cols, self._colours(end)
        ramp = {}
        for a, b in set(zip(c0, c1)):
            ramp[(a, b)] = [mix(a, b, k / self.STEPS) for k in range(self.STEPS + 1)]
        self.xy, self.cols = end, c1

        def step(_, a):
            now = start + (end - start) * a
            k = int(round(self.STEPS * a))
            for d, p, ca, cb in zip(self, now, c0, c1):
                d.move_to(at(self.plot, self._clip(p)))
                d.set_color(ramp[(ca, cb)][k])
                d.set_opacity(self._alpha(p))

        return m.UpdateFromAlphaFunc(self, step, run_time=run_time)


ORIGIN_ = np.zeros(3)


def swatch(color, dashed=False, width=0.5):
    line = m.Line(ORIGIN_, [width, 0, 0]).set_stroke(color, 4)
    return m.DashedVMobject(line, num_dashes=4).set_stroke(color, 3) if dashed else line


def key_row(mark, words, color=INK, size=22):
    lab = text(words, size, color)
    row = m.VGroup(mark, lab).arrange(m.RIGHT, buff=0.2)
    mark.set_y(lab.get_y())
    return row



# ---------------------------------------------------------------------------- the episode


class Ep10SolversAndSchedules(Episode):
    number = 10

    def lesson(self):
        self.hook()
        self.euler()
        self.heun()
        self.placement()
        self.race()
        self.published()
        self.lighthouse()
        self.recap()

    # ------------------------------------------------------------------ 0 · hook

    def hook(self):
        self.next_section("Hook", notes=(
            "Episode 9: without fresh noise, sampling follows the probability-flow ODE (Song et al. 2021). Six of toy.md's "
            "2,000 noise points (z = gaussian_noise(2000, seed=5); indices " + ", ".join(map(str, HOOK)) + "), chosen "
            "one per 60° of the swirl, because DDIM-5 strands them far from their exact ends. Dashed: the exact path (Heun, 1,000 Karras steps, "
            "exact denoiser, every 4th state). LIME: Euler in σ with 5 evenly spaced DDPM steps (= DDIM, η = 0), drawn "
            "in the x_t picture; an AMBER dot at the start of every step is one network call. RED ends: farther than 0.1 "
            "from the swirl's centre line (off_swirl)."))
        pl = plane()
        band = swirl_band(pl)
        ref, hop = RUNS["hook_ref"], RUNS["hook_ddim5"]
        starts = m.VGroup(*[m.Dot(at(pl, p), radius=0.07, color=NOISE).set_z_index(5) for p in ref[0]])
        exact = m.VGroup(*[dashed_path(pl, ref[:, j]) for j in range(len(HOOK))])
        exact_ends = m.VGroup(*[m.Dot(at(pl, p), radius=0.06, color=DATA).set_z_index(4) for p in ref[-1]])
        lines = m.VGroup(*[path_line(pl, hop[:1, j].repeat(2, 0), EULER_C, 3.2) for j in range(len(HOOK))]).set_z_index(3)
        heads = m.VGroup(*[m.Dot(at(pl, hop[0, j]), radius=0.06, color=EULER_C).set_z_index(5) for j in range(len(HOOK))])
        calls = m.VGroup(*[call_dots(pl, hop[:-1, j], 0.055).set_opacity(0) for j in range(len(HOOK))])
        ends = m.VGroup(*[end_dot(pl, hop[-1, j], 0.08) for j in range(len(HOOK))])
        n_strand = int(stranded(hop[-1]).sum())
        assert n_strand == len(HOOK)
        rings = m.VGroup(*[m.Circle(radius=0.2).set_stroke(RED, 3).move_to(e) for e in ends])

        y0 = 2.0
        k_exact = key_row(swatch(EXACT_C, dashed=True), "the exact path", MUTED)
        k_euler = key_row(swatch(EULER_C), "5 straight steps", EULER_C)
        k_call = key_row(m.Dot(radius=0.07, color=CALL_C), "one network call", CALL_C)
        keys = m.VGroup(k_exact, k_euler, k_call).arrange(m.DOWN, buff=0.22, aligned_edge=m.LEFT)
        keys.move_to([RX, y0 - 0.45, 0]).align_to([1.2, 0, 0], m.LEFT)
        k_euler[1].align_to(k_exact[1], m.LEFT)
        k_call[0].set_x(k_exact[0].get_x())
        k_call[1].align_to(k_exact[1], m.LEFT)
        strand = text("land between the arms", 22, RED, weight=m.BOLD).next_to(keys, m.DOWN, buff=0.3).align_to(keys, m.LEFT)

        big = title_text("numerical integration", 44, INK).move_to([RX, -0.45, 0])
        if big.get_right()[0] > 6.6:
            big.shift((6.6 - big.get_right()[0]) * m.RIGHT)

        def task(title, sub, color, y):
            words = m.VGroup(text(title, 26, color, weight=m.BOLD), text(sub, 22, MUTED)).arrange(m.DOWN, buff=0.1)
            box = card(5.0, words.height + 0.42, color, FILLS[color], radius=0.16, stroke=2).move_to([RX, y, 0])
            return m.VGroup(box, words.move_to(box))

        t1 = task("better steps", "the solver: how to step", INK, -1.55)
        t2 = task("better places", "the schedule: where to step", TIME, -2.75)

        self.say(
            "[Episode nine showed that sampling without fresh noise follows a smooth path], [from noise to data].",
            m.AnimationGroup(self.new_header("00 · hook", "Sampling is integration"), m.FadeIn(band),
                             m.FadeIn(starts, scale=0.6), m.FadeIn(k_exact)),
            m.AnimationGroup(m.LaggedStart(*[m.Create(e) for e in exact], lag_ratio=0.12, run_time=1.6),
                             m.FadeIn(exact_ends)),
        )
        self.say(
            "[But the sampler never sees the whole path]. [At each step it asks the network for a direction], "
            "[and walks straight].",
            exact.animate.set_opacity(0.55),
            m.AnimationGroup(m.FadeIn(k_call), m.FadeIn(heads, scale=0.6)),
            m.AnimationGroup(m.FadeIn(k_euler), grow_steps(pl, hop, lines, heads, calls, 2.6)),
        )
        self.say(
            "[With big steps, the straight lines miss the curve], [and points land between the arms].",
            m.AnimationGroup(exact.animate.set_opacity(1.0), m.FadeOut(heads)),
            m.AnimationGroup(m.FadeIn(ends, scale=0.5), m.Create(rings), m.FadeIn(strand, shift=0.1 * m.UP)),
        )
        self.say(
            "[Following a path in steps is numerical integration]. [Today: better steps], [and better places to put them].",
            m.FadeIn(big, shift=0.15 * m.UP),
            m.FadeIn(t1, shift=0.15 * m.UP),
            m.FadeIn(t2, shift=0.15 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 1 · Euler

    def euler(self):
        self.next_section("1 · Euler", notes=(
            "Euler's method on the probability-flow ODE in the σ view (Episode 9: y = x_t/√ᾱ = x₀ + σ·ε, dy/dσ = ε̂): one "
            "network call per step. With evenly spaced DDPM steps it is DDIM (η = 0) exactly (asserted on 300 points: "
            "max difference < 1e-9). The miss: mean distance, over toy.md's 2,000 points, between the sampler's output and "
            "the exact ODE solution from the same noise (Heun, 1,000 Karras steps) — toy.json samplers/exact/runs/ddim_N/"
            f"ode_error: {', '.join(f'{s} steps {EULER_MISS[s]:.4g}' for s in (5, 10, 20, 100))}. Halving per doubling = "
            f"first order (error ∝ step size). The path shown: particle {HERO}, exact denoiser."))
        ref = RUNS["hook_ref"][:, HOOK.index(HERO)]
        zc, zs = np.array([-3.2, -0.42, 0.0]), 5.3
        pl, c, half = zoom_window(np.vstack([ref] + [RUNS[f"hero_ddim{s_}"] for s_ in (5, 10, 20, 100)]), zc, zs, 1.3)
        band = m.VGroup(m.Square(zs).set_stroke(EDGE, 2).move_to(zc), clipped_arms(pl, c, half, zs))
        zoom_lab = cite(f"one point's path, zoomed in · exact denoiser").next_to(band[0], m.DOWN, buff=0.16)
        exact = dashed_path(pl, ref)
        start = m.Dot(at(pl, ref[0]), radius=0.08, color=NOISE).set_z_index(5)
        goal = m.Dot(at(pl, ref[-1]), radius=0.07, color=DATA).set_z_index(5)

        def walk(s):
            xs = RUNS[f"hero_ddim{s}"]
            return m.VGroup(path_line(pl, xs, EULER_C, 3.4).set_z_index(3),
                            call_dots(pl, xs[:-1], 0.06) if s <= 20 else m.VGroup(),
                            end_dot(pl, xs[-1], 0.08))

        walks = {s: walk(s) for s in (5, 10, 20, 100)}
        corner = zc + np.array([-zs / 2 + 0.3, zs / 2 - 0.35, 0])
        steps_now = code("5 steps", 26, EULER_C, weight=m.BOLD).move_to(corner, aligned_edge=m.LEFT)
        call_key = key_row(m.Dot(radius=0.07, color=CALL_C), "one network call", CALL_C)
        call_key.move_to(zc + np.array([zs / 2 - 0.3, zs / 2 - 0.35, 0]), aligned_edge=m.RIGHT)

        f = formula(r"y_s", "=", r"y_t", "+", r"(\sigma_s - \sigma_t)", r"\,\hat\varepsilon", size=50).move_to([RX, 2.0, 0])
        f[4].set_color(TIME)
        f[5].set_color(MODEL)
        lab_eps = text("one network call", 22, MODEL).next_to(f[5], m.DOWN, buff=0.3)
        lab_step = text("one straight step", 22, TIME).next_to(f[4], m.DOWN, buff=0.3)
        lab_step.set_y(lab_eps.get_y())
        if lab_step.get_right()[0] > lab_eps.get_left()[0] - 0.25:
            lab_step.shift((lab_eps.get_left()[0] - 0.25 - lab_step.get_right()[0]) * m.RIGHT)
        same = pill("= DDIM (η = 0) · Episode 9", EULER_C, LIME_FILL, 22).move_to([RX, 0.62, 0])
        view = cite("in the σ view: y = x₀ + σ·ε").next_to(same, m.DOWN, buff=0.18)

        box = card(5.2, 2.15, EDGE, CARD, radius=0.16, stroke=2).move_to([RX, -1.3, 0])
        head = text("average miss, 2,000 points", 22, MUTED).move_to(box.get_top() + [0, -0.36, 0])
        rows = m.VGroup()
        for k, s in enumerate((5, 10, 20)):
            y = box.get_top()[1] - 0.9 - 0.42 * k
            lab = text(f"{s} steps", 24, INK).move_to([box.get_left()[0] + 0.45, y, 0], aligned_edge=m.LEFT)
            num = code(MISS_TXT[s], 26, EULER_C, weight=m.BOLD).move_to([box.get_right()[0] - 0.45, y, 0], aligned_edge=m.RIGHT)
            sit(num, baseline(lab))
            rows.add(m.VGroup(lab, num))
        halves = m.VGroup(*[code("÷2", 22, MUTED).move_to([box.get_x() + 0.55, (a[1].get_y() + b[1].get_y()) / 2, 0])
                            for a, b in zip(rows, rows[1:])])
        first = pill("first order: error ∝ step size", EULER_C, LIME_FILL, 22).next_to(box, m.DOWN, buff=0.22)
        assert first.get_bottom()[1] > -3.55, first.get_bottom()

        self.say(
            "[The simplest solver is Euler's method]: [one network call gives the slope], "
            "[then one straight step to the next noise level].",
            m.AnimationGroup(self.new_header("01 · Euler", "One look per step"), m.FadeIn(band), m.FadeIn(zoom_lab), m.FadeIn(exact),
                             m.FadeIn(start, scale=0.6), m.FadeIn(goal), m.Write(f[:4])),
            m.AnimationGroup(m.Write(f[5]), m.FadeIn(lab_eps, shift=0.1 * m.UP)),
            m.AnimationGroup(m.Write(f[4]), m.FadeIn(lab_step, shift=0.1 * m.UP)),
        )
        self.say(
            "[In the sigma view, that is exactly the deterministic sampler from episode nine].",
            m.AnimationGroup(m.FadeIn(same, shift=0.1 * m.UP), m.FadeIn(view)),
        )
        self.say(
            "[Five steps miss the exact path by zero point three two on average]. [Ten steps, zero point one five]. "
            "[Twenty, about zero point zero eight].",
            m.AnimationGroup(m.FadeIn(walks[5]), m.FadeIn(steps_now), m.FadeIn(call_key), m.FadeIn(box), m.FadeIn(head),
                             m.FadeIn(rows[0], shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeOut(walks[5]), m.FadeIn(walks[10]), m.Transform(steps_now, code("10 steps", 26, EULER_C, weight=m.BOLD).move_to(steps_now, aligned_edge=m.LEFT)),
                             m.FadeIn(rows[1], shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeOut(walks[10]), m.FadeIn(walks[20]), m.Transform(steps_now, code("20 steps", 26, EULER_C, weight=m.BOLD).move_to(steps_now, aligned_edge=m.LEFT)),
                             m.FadeIn(rows[2], shift=0.1 * m.UP)),
        )
        self.say(
            "[Each doubling halves the miss]: [Euler is a first-order method].",
            m.AnimationGroup(m.FadeIn(halves), m.FadeOut(walks[20]), m.FadeIn(walks[100]),
                             m.Transform(steps_now, code("100 steps", 26, EULER_C, weight=m.BOLD).move_to(steps_now, aligned_edge=m.LEFT))),
            m.FadeIn(first, shift=0.1 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 2 · Heun

    def heun(self):
        sa, sb = STEP_SIG
        self.next_section("2 · Heun", notes=(
            "One big step taken apart (precompute.py): particle 711 of toy.md's noise, on its exact path at σ = "
            f"{sa:.3f}, stepping to σ = {sb:.3f} (the 6th of 7 steps of an 8-level Karras grid), exact denoiser. In the σ "
            "view the Euler step is y + h·ε̂(y); Heun (Karras et al. 2022, Algorithm 1) looks again where it lands, "
            "ε̂(ỹ), and redoes the step with the average slope, so its landing is the midpoint of the Euler landing and "
            "the step taken with the second slope (asserted; also equal to heun_ode / euler_ode for this step). Drawn in "
            f"the x_t picture (midpoints are kept). Miss vs the exact path's point at σ = {sb:.2f}: Euler {MISS_E:.3f}, "
            f"Heun {MISS_H:.3f} (this particle, chosen for clarity: ratio {MISS_E / MISS_H:.1f}; the median over all "
            "2,000 points at this step is 1.9). Second order: error ∝ step² (notes §10). Cost: 2 calls per step, 1 on the "
            "last step to σ = 0 (no second look there): N steps = 2N − 1 calls."))
        zc, zs = np.array([-3.2, -0.42, 0.0]), 5.3
        pl, c, half = zoom_window(np.vstack([SEG, STEP["trial"], STEP["other"], STEP["heun"]]), zc, zs)
        frame = m.Square(zs).set_stroke(EDGE, 2).move_to(zc)
        arms = clipped_arms(pl, c, half, zs)
        zoom_lab = cite(f"one big step, zoomed in: σ {sa:.2f} → {sb:.2f}").next_to(frame, m.DOWN, buff=0.16)
        exact = dashed_path(pl, SEG, EXACT_C, 3.5, 16)
        p0 = at(pl, STEP["start"])
        p_t, p_o, p_h = at(pl, STEP["trial"]), at(pl, STEP["other"]), at(pl, STEP["heun"])
        p_e = at(pl, STEP_END)
        start = m.Dot(p0, radius=0.1, color=INK).set_z_index(6)
        call1 = m.Circle(radius=0.2).set_stroke(CALL_C, 4).move_to(p0).set_z_index(6)
        goal = m.Dot(p_e, radius=0.09, color=DATA).set_z_index(6)
        goal_lab = text("exact", 22, MUTED).next_to(goal, m.LEFT, buff=0.16)

        def arw(a, b, color, width=5):
            return m.Arrow(a, b, buff=0.12, color=color, stroke_width=width, max_tip_length_to_length_ratio=0.12)

        e_arrow = arw(p0, p_t, EULER_C)
        trial = m.Dot(p_t, radius=0.09, color=EULER_C).set_z_index(6)
        trial_lab = text("trial", 22, EULER_C, weight=m.BOLD).next_to(trial, m.LEFT, buff=0.3)
        call2 = m.Circle(radius=0.2).set_stroke(CALL_C, 4).move_to(p_t).set_z_index(6)
        d2 = (p_o - p0) / np.linalg.norm(p_o - p0)
        look = arw(p_t, p_t + 0.95 * d2, CALL_C, 4)
        o_line = m.DashedLine(p0, p_o, dash_length=0.12).set_stroke(CALL_C, 3)
        o_dot = m.Circle(radius=0.08).set_stroke(CALL_C, 3).move_to(p_o).set_z_index(6)
        o_lab = text("second slope", 22, CALL_C).next_to(o_dot, m.DOWN, buff=0.16)
        mid = m.DashedLine(p_t, p_o, dash_length=0.08).set_stroke(MUTED, 2)
        h_arrow = arw(p0, p_h, HEUN_C, 6)
        h_dot = m.Dot(p_h, radius=0.09, color=HEUN_C).set_z_index(7)
        h_lab = text("Heun", 22, HEUN_C, weight=m.BOLD).next_to(h_dot, m.RIGHT, buff=0.22)
        for lab in (trial_lab, o_lab, h_lab, goal_lab):
            assert frame.get_left()[0] + 0.1 < lab.get_left()[0] and lab.get_right()[0] < frame.get_right()[0] - 0.1, lab.text

        rows = m.VGroup()
        for k, (num_c, words) in enumerate(((CALL_C, "look: the slope at the start"), (EULER_C, "take that step as a trial"),
                                            (CALL_C, "look again where it lands"), (HEUN_C, "redo it with the average"))):
            rows.add(m.VGroup(code(f"{k + 1}", 24, num_c, weight=m.BOLD), text(words, 23, INK)).arrange(m.RIGHT, buff=0.28))
        rows.arrange(m.DOWN, buff=0.2, aligned_edge=m.LEFT).move_to([RX, 1.6, 0]).align_to([0.95, 0, 0], m.LEFT)
        for r in rows:
            sit(r[0], baseline(r[1]))
        name = title_text("Heun's method", 34, HEUN_C).next_to(rows, m.DOWN, buff=0.38).align_to(rows, m.LEFT)
        misses = m.VGroup(code(f"Euler misses by {MISS_E:.2f}", 22, EULER_C), code(f"Heun misses by  {MISS_H:.3f}", 22, HEUN_C))
        misses.arrange(m.DOWN, buff=0.14, aligned_edge=m.LEFT).next_to(name, m.DOWN, buff=0.22).align_to(rows, m.LEFT)
        order = pill("second order: error ∝ step²", HEUN_C, CARD, 22).next_to(misses, m.DOWN, buff=0.3).align_to(rows, m.LEFT)
        cost = m.VGroup(text("2 network calls per step, 1 on the last:", 22, MUTED),
                        m.VGroup(code("25 steps", 26, INK, weight=m.BOLD), text("cost", 24, MUTED),
                                 code("49 calls", 26, CALL_C, weight=m.BOLD)).arrange(m.RIGHT, buff=0.22))
        sit(cost[1][1], baseline(cost[1][0]))
        cost.arrange(m.DOWN, buff=0.16, aligned_edge=m.LEFT).next_to(order, m.DOWN, buff=0.3).align_to(rows, m.LEFT)
        assert cost.get_bottom()[1] > -3.55 and rows.get_top()[1] < 2.65, (cost.get_bottom(), rows.get_top())
        assert 2 * 25 - 1 == 49

        self.say(
            "[A better step looks twice]. [First, take the Euler step as a trial].",
            m.AnimationGroup(self.new_header("02 · Heun", "Look twice"), m.FadeIn(frame), m.FadeIn(arms), m.FadeIn(zoom_lab),
                             m.Create(exact), m.FadeIn(goal), m.FadeIn(goal_lab), m.FadeIn(start, scale=0.5),
                             m.Create(call1), m.FadeIn(rows[0], shift=0.1 * m.UP)),
            m.AnimationGroup(m.GrowArrow(e_arrow), m.FadeIn(trial, scale=0.5), m.FadeIn(trial_lab), m.FadeIn(rows[1], shift=0.1 * m.UP)),
        )
        self.say(
            "[Then ask the network again, at the point where the trial landed].",
            m.AnimationGroup(m.Create(call2), m.GrowArrow(look), m.FadeIn(rows[2], shift=0.1 * m.UP)),
        )
        self.say(
            "[Redo the step from the start], [with the average of the two directions].",
            m.AnimationGroup(m.FadeOut(look), m.Create(o_line), m.FadeIn(o_dot), m.FadeIn(o_lab)),
            m.AnimationGroup(m.Create(mid), m.GrowArrow(h_arrow), m.FadeIn(h_dot, scale=0.5), m.FadeIn(h_lab),
                             m.FadeIn(rows[3], shift=0.1 * m.UP)),
        )
        self.say(
            "[That is Heun's method]: [here it lands right beside the exact path].",
            m.FadeIn(name, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(misses, shift=0.1 * m.UP), m.Circumscribe(m.VGroup(h_dot, goal), color=HEUN_C, buff=0.12)),
        )
        self.say(
            "[Its error shrinks with the square of the step size]: [a second-order method].",
            m.FadeIn(order, shift=0.1 * m.UP),
            m.Circumscribe(order, color=HEUN_C, buff=0.08),
        )
        self.say(
            "[The price is two network calls per step], [one on the last], [so twenty-five Heun steps cost forty-nine calls].",
            m.AnimationGroup(m.Circumscribe(call1, color=CALL_C, shape=m.Circle, buff=0.06),
                             m.Circumscribe(call2, color=CALL_C, shape=m.Circle, buff=0.06), m.FadeIn(cost[0])),
            m.Indicate(cost[0], color=INK, scale_factor=1.0),
            m.FadeIn(cost[1], shift=0.1 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 3 · placement

    def placement(self):
        self.next_section("3 · Where the steps go", notes=(
            "The noise level σ = √((1 − ᾱ)/ᾱ) at each of 20 steps (log scale), then the last jump to σ = 0 (RED). Evenly "
            "spaced DDPM steps t = 1000, 950, … 50 under the linear schedule (σ from 157 down to "
            f"{LAST['lin']:.3f}); the same steps under Episode 4's cosine schedule (Nichol & Dhariwal 2021), run on our "
            "linear-trained networks at the noise levels they know (timestep_from_sigma; cosine's own first level, "
            f"σ ≈ 20,000, is clipped to 157, so all three start there; last level {LAST['cos']:.3f}); Karras et al. 2022 "
            "(EDM, eq. 5): σ_i = (σ_max^(1/ρ) + i/(N−1)·(σ_min^(1/ρ) − σ_max^(1/ρ)))^ρ, ρ = 7, with OUR schedule's range "
            f"σ_max = σ(t=1000) = {SMAX:.1f}, σ_min = σ(t=1) = {SMIN:.3f} (the paper's: 80 and 0.002). Steps below σ = 1: "
            f"linear {(SIG_LIN < 1).sum()}, cosine {(SIG_COS < 1).sum()}, Karras {(SIG_KAR < 1).sum()}; Karras's steps "
            "shrink toward low noise. Result: the trained swirl MLP, Euler in σ (the same solver on every grid), 20 calls, "
            f"toy.md's 2,000 noise points: stranded (> 0.1 from the curve) {pct(STRAND['lin'])} (toy.json ddim_20), "
            f"{pct(STRAND['cos'])} (computed here, precompute.py), {pct(STRAND['kar'])} (toy.json euler_karras_20). With "
            f"the exact denoiser: 32.0% / {pct(off_swirl(RUNS['exact_cos_20']))} / 1.5%. Not claimed: that the σ "
            "coordinate matters (Euler written in DDPM time is about as accurate on this toy, toy.md caveat); the "
            "difference here is where the steps fall. Shown: the first 1,000 of the 2,000 samples."))
        plot = Plot((0, 20), (0.005, 300), width=5.3, height=3.9, center=(-3.25, -0.3, 0), log_y=True,
                    x_ticks=[0, 5, 10, 15, 20], y_ticks=[0.01, 0.1, 1, 10, 100], x_label="step (of 20)",
                    y_label="noise level σ")
        floor = text("last jump: to σ = 0", 20, RED).move_to(plot.coords(13.2, 0.0098))

        def curve(sig, color):
            xs = np.arange(len(sig))
            line = plot.line(xs, sig, color, 3.5)
            dots = m.VGroup(*[plot.dot(x, s, color, 0.055) for x, s in zip(xs, sig)])
            drop = m.Arrow(plot.coords(19, sig[-1]), plot.coords(20, 0.005), buff=0.05, color=RED, stroke_width=3,
                           max_tip_length_to_length_ratio=0.18)
            return line, dots, drop

        c_lin, c_cos, c_kar = curve(SIG_LIN, LIN_C), curve(SIG_COS, COS_C), curve(SIG_KAR, KAR_C)

        def row(color, name, last, y):
            mark = swatch(color, width=0.45)
            words = text(name, 23, color, weight=m.BOLD)
            first = m.VGroup(mark, words).arrange(m.RIGHT, buff=0.2)
            mark.set_y(words.get_y())
            jump = code(f"last jump from {last}", 20, MUTED)
            r = m.VGroup(first, jump).arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT)
            jump.shift(0.65 * m.RIGHT)
            return r.move_to([0, y, 0]).align_to([0.75, 0, 0], m.LEFT)

        r_lin = row(LIN_C, "linear · even steps", "0.17", 2.0)
        r_cos = row(COS_C, "cosine · even steps", "0.09", 0.95)
        r_kar = row(KAR_C, "Karras · ρ = 7", "0.01", -0.1)
        f = formula(r"\sigma_i = \Big(\sigma_{\max}^{1/\rho} + \tfrac{i}{N-1}\big(\sigma_{\min}^{1/\rho} - "
                    r"\sigma_{\max}^{1/\rho}\big)\Big)^{\rho}", size=32).move_to([RX + 0.1, -1.55, 0])
        f_src = cite("Karras et al., 2022 · eq. 5 · here σ runs\nfrom 157 to 0.01, our schedule's range").next_to(f, m.DOWN, buff=0.2)
        assert f.get_right()[0] < 6.6 and f_src.get_right()[0] < 6.6, (f.get_right(), f_src.get_right())

        # phase B: the same solver and network on each grid
        pl = plane(np.array([-3.3, -0.05, 0.0]), 4.5, LIM)
        band = swirl_band(pl, 10)
        n_show = 1000
        swarm = Swarm(pl, Z[:n_show], radius=0.024, noise=True)
        setup = cite("trained network · Euler · 20 calls · same noise").next_to(pl.coords(0, -LIM), m.DOWN, buff=0.22)
        legend = m.VGroup(m.VGroup(m.Dot(radius=0.08, color=DATA), cite("on an arm", 24)).arrange(m.RIGHT, buff=0.12),
                          m.VGroup(m.Dot(radius=0.08, color=RED), cite("stranded", 24)).arrange(m.RIGHT, buff=0.12))
        legend.arrange(m.RIGHT, buff=0.45).next_to(setup, m.DOWN, buff=0.16)
        assert legend.get_bottom()[1] > -3.55, legend.get_bottom()

        def result(r, key):
            num = title_text(pct(STRAND[key]), 40, RED if STRAND[key] > 0.1 else GREEN)
            return num.move_to([6.45, r.get_y(), 0], aligned_edge=m.RIGHT)

        res = {k: result(r, k) for r, k in ((r_lin, "lin"), (r_cos, "cos"), (r_kar, "kar"))}
        res_head = cite("stranded").move_to([res["lin"].get_center()[0], 2.62, 0])

        self.say(
            "[The second question is where the steps go]. [Here is the noise level at each of twenty steps].",
            self.new_header("03 · placement", "Where the steps go"),
            m.AnimationGroup(m.FadeIn(plot), m.FadeIn(floor)),
        )
        self.say(
            "[Evenly spaced in time, under the linear schedule], [they stop at sigma zero point one seven], "
            "[then jump straight to a clean picture].",
            m.AnimationGroup(m.Create(c_lin[0]), m.FadeIn(c_lin[1], lag_ratio=0.05), m.FadeIn(r_lin[0])),
            m.AnimationGroup(m.Circumscribe(c_lin[1][-1], color=LIN_C, shape=m.Circle, buff=0.08), m.FadeIn(r_lin[1])),
            m.GrowArrow(c_lin[2]),
        )
        self.say(
            "[Episode four's cosine schedule spends more steps at low noise]: [its last jump starts at zero point zero nine].",
            m.AnimationGroup(m.Create(c_cos[0]), m.FadeIn(c_cos[1], lag_ratio=0.05), m.FadeIn(r_cos[0])),
            m.AnimationGroup(m.GrowArrow(c_cos[2]), m.FadeIn(r_cos[1])),
        )
        self.say(
            "[In twenty twenty-two, Karras and colleagues spaced them on purpose]: [evenly in sigma to the power one over seven].",
            m.AnimationGroup(m.FadeIn(r_kar[0]), m.Write(f), m.FadeIn(f_src)),
            m.AnimationGroup(m.Create(c_kar[0]), m.FadeIn(c_kar[1], lag_ratio=0.05)),
        )
        self.say(
            "[The steps shrink toward low noise], [where the fine detail is decided], [and the last jump starts at zero point zero one].",
            m.Circumscribe(c_kar[1][-6:], color=KAR_C, buff=0.12),
            m.Circumscribe(f, color=KAR_C, buff=0.1),
            m.AnimationGroup(m.GrowArrow(c_kar[2]), m.FadeIn(r_kar[1])),
        )
        self.say(
            "[Now run the same solver and the same trained network], [twenty calls each], [and change only the placement].",
            m.AnimationGroup(m.FadeOut(m.VGroup(plot, floor, *c_lin, *c_cos, *c_kar, f, f_src)), m.FadeIn(band),
                             m.LaggedStart(*[m.FadeIn(d, scale=0.5) for d in swarm], lag_ratio=0.001, run_time=1.2)),
            m.AnimationGroup(m.FadeIn(setup), m.FadeIn(legend)),
            m.FadeIn(res_head),
        )
        self.say(
            "[Stranded points drop from thirty-one percent], [to thirteen], [to four point eight].",
            m.AnimationGroup(swarm.go(RUNS["mlp_ddim_20"][:n_show], 1.3), m.FadeIn(res["lin"], shift=0.1 * m.LEFT)),
            m.AnimationGroup(swarm.go(RUNS["mlp_cos_20"][:n_show], 1.1), m.FadeIn(res["cos"], shift=0.1 * m.LEFT)),
            m.AnimationGroup(swarm.go(RUNS["mlp_euler_karras_20"][:n_show], 1.1), m.FadeIn(res["kar"], shift=0.1 * m.LEFT)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 4 · the race

    def race(self):
        self.next_section("4 · Count calls, not steps", notes=(
            "Mean miss vs the exact ODE solution (2,000 points, exact denoiser, Karras grid; toy.json samplers/exact/runs, "
            "recomputed by precompute.py). Euler: " + ", ".join(f"{n} calls {e:.4g}" for n, e in zip(E_NFE, E_ERR)) +
            ". Heun (N steps = 2N − 1 calls): " + ", ".join(f"{n} calls {e:.4g}" for n, e in zip(H_NFE, H_ERR)) +
            f". Heun is behind at 19 calls, leads from 49 (half Euler's miss at 49 vs 50; {LEAD99:.1f}× smaller at 99 vs "
            f"100). Trained network (toy.json mlp): Heun, 49 calls, strands {pct(H49_OFF)}; DDPM ancestral, 1,000 calls, "
            f"{pct(DDPM_OFF)}. Divergence (toy.md caveat): with the exact denoiser, Heun with 9 calls ends {DIV9:.4f} from "
            f"the swirl on average ({pct(OFF9)} stranded), with 5 calls {DIV5:.2f}; its second look at a very low σ, where "
            "the swirl's score is steep, overshoots (the trained network: 0.688 and 27.8). Data: 0.024. Dots that leave "
            "the square fade out. Shown: the first 1,000 samples."))
        plot = Plot((4, 128), (0.002, 10), width=5.6, height=3.9, center=(-2.95, -0.35, 0), log_x=True, log_y=True,
                    x_ticks=[5, 10, 20, 50, 100], y_ticks=[0.01, 0.1, 1, 10], x_label="network calls",
                    y_label="miss: distance from the exact path")
        grid_note = cite("Karras grid · exact denoiser · 2,000 points").next_to(plot.x_label, m.DOWN, buff=0.12)
        e_line = plot.line(E_NFE, E_ERR, EULER_C, 4)
        e_dots = m.VGroup(*[plot.dot(n, e, EULER_C, 0.07) for n, e in zip(E_NFE, E_ERR)])
        h_line = plot.line(H_NFE[2:], H_ERR[2:], HEUN_C, 4)
        h_dots = m.VGroup(*[plot.dot(n, e, HEUN_C, 0.07) for n, e in zip(H_NFE[2:], H_ERR[2:])])
        h_bad = m.DashedVMobject(plot.line(H_NFE[:3], H_ERR[:3], RED, 3), num_dashes=14).set_stroke(RED, 3)
        h_bad_dots = m.VGroup(*[plot.dot(n, e, RED, 0.08) for n, e in zip(H_NFE[:2], H_ERR[:2])])
        e_lab = text("Euler", 24, EULER_C, weight=m.BOLD).move_to(plot.coords(7, 0.48))
        h_lab = text("Heun", 24, HEUN_C, weight=m.BOLD).move_to(plot.coords(30, 0.0075))
        halve = code("÷2 per doubling", 20, EULER_C).move_to(plot.coords(12, 0.034))
        lead_lab = text("from 49 calls, Heun leads", 20, HEUN_C).move_to(plot.coords(45, 0.55))
        steps_lab = cite("Heun: 25 steps = 49 calls").next_to(lead_lab, m.DOWN, buff=0.1)
        x_r = 118
        span = m.DoubleArrow(plot.coords(x_r, E_ERR[-1]), plot.coords(x_r, H_ERR[-1]), buff=0.04, stroke_width=3,
                             color=HEUN_C, max_tip_length_to_length_ratio=0.12)
        ratio = code(f"{LEAD99:.1f}×", 22, HEUN_C, weight=m.BOLD).next_to(span, m.RIGHT, buff=0.12)

        pc = np.array([4.35, 0.45, 0.0])
        pl = plane(pc, 3.1, LIM)
        sq = m.Square(3.1).set_stroke(EDGE, 2).move_to(pc)
        band = swirl_band(pl, 7)
        n_show = 1000
        swarm = Swarm(pl, RUNS["mlp_heun_karras_49"][:n_show], radius=0.02)
        who = cite("trained network").next_to(sq, m.UP, buff=0.14)
        who2 = cite("exact denoiser").move_to(who)
        r1 = m.VGroup(code(f"Heun · 49 calls: {pct(H49_OFF)}", 20, HEUN_C), code(f"DDPM · 1,000 calls: {pct(DDPM_OFF)}", 20, NOISE))
        r1.arrange(m.DOWN, buff=0.14, aligned_edge=m.LEFT).next_to(sq, m.DOWN, buff=0.22)
        r1_head = cite("stranded").next_to(r1, m.UP, buff=0.0)
        r1_head.next_to(sq, m.DOWN, buff=0.16)
        r1.next_to(r1_head, m.DOWN, buff=0.12)
        r2 = m.VGroup(cite("mean distance from the swirl"),
                      code(f"Heun · 9 calls: {DIV9:.2f}", 22, RED, weight=m.BOLD),
                      code(f"Heun · 5 calls: {DIV5:.1f}", 22, RED, weight=m.BOLD),
                      cite(f"the data itself: {DATA_DIST:.3f}"))
        r2.arrange(m.DOWN, buff=0.14).next_to(sq, m.DOWN, buff=0.16)
        assert r2.get_bottom()[1] > -3.55 and r1.get_bottom()[1] > -3.55, (r1.get_bottom(), r2.get_bottom())
        broke = text("diverges", 22, RED, weight=m.BOLD).next_to(plot.coords(9, H_ERR[1]), m.RIGHT, buff=0.3).shift(0.3 * m.UP)

        self.say(
            "[Now race the two solvers fairly], [by network calls, not steps], [on this grid, with the exact denoiser].",
            m.AnimationGroup(self.new_header("04 · the race", "Count calls, not steps"), m.FadeIn(plot)),
            m.Circumscribe(plot.x_label, color=CALL_C, buff=0.08),
            m.FadeIn(grid_note),
        )
        self.say(
            "[Here is how far each one ends from the exact path]. [Euler's error roughly halves when the calls double].",
            m.AnimationGroup(m.Create(e_line), m.FadeIn(e_dots), m.FadeIn(e_lab)),
            m.FadeIn(halve, shift=0.1 * m.UP),
        )
        self.say(
            "[The Heun curve falls faster]: [from about forty-nine calls on, it leads], [and at ninety-nine calls its error is more than five times smaller].",
            m.AnimationGroup(m.Create(h_line), m.FadeIn(h_dots), m.FadeIn(h_lab)),
            m.AnimationGroup(m.FadeIn(lead_lab), m.FadeIn(steps_lab),
                             m.Circumscribe(m.VGroup(h_dots[1], e_dots[3]), color=HEUN_C, buff=0.1)),
            m.AnimationGroup(m.GrowFromCenter(span), m.FadeIn(ratio, shift=0.1 * m.LEFT)),
        )
        self.say(
            "[With the trained network, forty-nine Heun calls strand only zero point eight percent], "
            "[about as few as a thousand noisy steps].",
            m.AnimationGroup(m.FadeIn(sq), m.FadeIn(band), m.FadeIn(swarm), m.FadeIn(who), m.FadeIn(r1_head),
                             m.FadeIn(r1[0], shift=0.1 * m.UP)),
            m.FadeIn(r1[1], shift=0.1 * m.UP),
        )
        self.say(
            "[But with nine calls or fewer, Heun breaks down]: [its second look, at a very low noise level, overshoots].",
            m.AnimationGroup(m.Create(h_bad), m.FadeIn(h_bad_dots, scale=0.5), m.FadeIn(broke)),
            m.AnimationGroup(m.FadeOut(r1), m.FadeOut(r1_head), m.FadeOut(who), m.FadeIn(who2),
                             swarm.go(RUNS["exact_heun_karras_9"][:n_show], 1.8)),
        )
        self.say(
            "[Even with the exact denoiser, nine calls land zero point seven nine from the swirl on average], "
            "[and five calls, four point eight].",
            m.AnimationGroup(m.FadeIn(r2[0]), m.FadeIn(r2[1], shift=0.1 * m.UP), m.FadeIn(r2[3])),
            m.AnimationGroup(m.FadeIn(r2[2], shift=0.1 * m.UP), swarm.go(RUNS["exact_heun_karras_5"][:n_show], 1.4)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 5 · published solvers

    def published(self):
        self.next_section("5 · Published solvers", notes=(
            "Karras, Aittala, Aila & Laine 2022 (EDM, arXiv:2206.00364, NeurIPS 2022), abstract: 'new state-of-the-art "
            "FID of 1.79 for CIFAR-10 in a class-conditional setting … (35 network evaluations per image)'; 35 = 18 Heun "
            "steps (2·18 − 1, notes §10, Appendix A row 10); σ from 80 to 0.002, ρ = 7 (eq. 5). 1.79 is the whole recipe "
            "(Table 2 config F: their sampler plus retraining with new preconditioning, loss weighting and augmentation; the "
            "baseline network with their sampler scored 2.48), hence 'plus a retrained network' on the card. Lu et al. 2022 "
            "(DPM-Solver, arXiv:2206.00927): solves the linear part of the diffusion ODE exactly and approximates only "
            "the network's term (an exponentially weighted integral, in log-SNR time); 'high-quality samples in only 10 to "
            "20 function evaluations'; CIFAR-10 FID 4.70 at 10 and 2.87 at 20. Lu et al. 2022 (DPM-Solver++, "
            "arXiv:2211.01095): data (x₀) prediction and a multistep form; earlier high-order samplers 'suffer from "
            "instability issues' under large guidance; '15 to 20 steps for guided sampling'. All training-free."))
        W, H = 6.0, 4.75

        def frame_card(x, color):
            return card(W, H, color, CARD, radius=0.2, stroke=2.5).move_to([x, -0.45, 0])

        c1, c2 = frame_card(-3.25, KAR_C), frame_card(3.25, TIME)
        t1 = m.VGroup(title_text("EDM", 40, KAR_C), cite("Karras et al., 2022")).arrange(m.RIGHT, buff=0.3)
        sit(t1[1], baseline(t1[0]))
        t1.move_to(c1.get_top() + [0, -0.55, 0])
        b1 = text("Heun steps spaced with ρ = 7,\nplus a retrained network", 22, INK).next_to(t1, m.DOWN, buff=0.28)
        s1 = m.VGroup(title_text("1.79", 72, KAR_C), text("FID on CIFAR-10\nlower is better", 22, MUTED))
        s1.arrange(m.RIGHT, buff=0.35).next_to(b1, m.DOWN, buff=0.35)
        n1 = m.VGroup(code("35 network calls", 24, CALL_C, weight=m.BOLD), cite("= 18 Heun steps · class-conditional"))
        n1.arrange(m.DOWN, buff=0.14).next_to(s1, m.DOWN, buff=0.3)

        t2 = m.VGroup(title_text("DPM-Solver", 40, TIME), cite("Lu et al., 2022")).arrange(m.RIGHT, buff=0.3)
        sit(t2[1], baseline(t2[0]))
        t2.move_to(c2.get_top() + [0, -0.55, 0])
        b2 = text("solves the simple part of the ODE exactly,\nestimates only the network's part", 22, INK)
        b2.next_to(t2, m.DOWN, buff=0.28)
        s2 = m.VGroup(title_text("10–20", 60, TIME), text("network calls for\nhigh-quality samples", 22, MUTED))
        s2.arrange(m.RIGHT, buff=0.35).next_to(b2, m.DOWN, buff=0.28)
        n2 = cite("CIFAR-10 FID 4.70 at 10 calls, 2.87 at 20").next_to(s2, m.DOWN, buff=0.18)
        rule = m.Line([-W / 2 + 0.4, 0, 0], [W / 2 - 0.4, 0, 0]).set_stroke(EDGE, 2).next_to(n2, m.DOWN, buff=0.22)
        rule.set_x(c2.get_x())
        pp = m.VGroup(m.VGroup(text("DPM-Solver++", 24, TIME, weight=m.BOLD), cite("Lu et al., Nov 2022")).arrange(m.RIGHT, buff=0.25),
                      text("stable under strong guidance: 15–20 steps", 22, INK)).arrange(m.DOWN, buff=0.12)
        sit(pp[0][1], baseline(pp[0][0]))
        pp.next_to(rule, m.DOWN, buff=0.2)
        for g, cc in ((m.VGroup(t1, b1, s1, n1), c1), (m.VGroup(t2, b2, s2, n2, pp), c2)):
            assert g.get_bottom()[1] > cc.get_bottom()[1] + 0.15 and g.width < W - 0.3, (g.get_bottom(), g.width)
        foot = cite("the samplers themselves need no retraining: they plug into an already trained model").move_to([0, -3.25, 0])

        self.say(
            "[With Heun steps on that spacing and a retrained network], [that paper reached an FID of one point seven nine], "
            "[in thirty-five network calls].",
            m.AnimationGroup(self.new_header("05 · published", "Solvers in the wild"), m.FadeIn(c1), m.FadeIn(t1), m.FadeIn(b1)),
            m.FadeIn(s1, shift=0.1 * m.UP),
            m.FadeIn(n1, shift=0.1 * m.UP),
        )
        self.say(
            "[Another family, DPM-Solver, solves the simple part of the equation exactly], "
            "[and needs only ten to twenty calls for high-quality samples].",
            m.AnimationGroup(m.FadeIn(c2), m.FadeIn(t2), m.FadeIn(b2)),
            m.AnimationGroup(m.FadeIn(s2, shift=0.1 * m.UP), m.FadeIn(n2)),
        )
        self.say(
            "[Its successor stays stable under strong guidance], [in fifteen to twenty steps].",
            m.AnimationGroup(m.Create(rule), m.FadeIn(pp[0])),
            m.AnimationGroup(m.FadeIn(pp[1], shift=0.1 * m.UP), m.FadeIn(foot)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 6 · the lighthouse

    def lighthouse(self):
        self.next_section("6 · The lighthouse", notes=(
            "Generated on this Mac (mac_runs.md §5): Stable Diffusion 2.1-base (sd2-community mirror), 512×512, prompt "
            "'a lighthouse on a rocky coast at dusk, oil painting', guidance 7.5, seed 0 shown. Sampler: the mlx-examples "
            "SimpleEulerSampler — Euler in σ on evenly spaced timesteps, no fresh noise (no Heun, no Karras spacing). With "
            "guidance each step runs the U-Net on two inputs (batched). Times: total seconds per image (text encoder + "
            "denoising + VAE decode), median of seeds 0–3: " + ", ".join(f"{s} → {LH_TIME[s]:.2f} s" for s in LH_STEPS) +
            ". Shared Mac: ±5–10% jitter. Seen in the images: 1 step a coloured blur, 2 a blurry scene, a lighthouse by 4, "
            "sharp by 20–50."))
        h = 1.82
        frames = m.Group(*[framed(picture(str(LH_FILES[s]), h, pixelated=False)) for s in LH_STEPS]).arrange(m.RIGHT, buff=0.18)
        frames.move_to([0, 1.12, 0])
        labs = m.VGroup(*[code(f"{s} step" + ("s" if s > 1 else ""), 22, INK).next_to(fr, m.DOWN, buff=0.14)
                          for s, fr in zip(LH_STEPS, frames)])
        cap = cite("generated on this Mac · Stable Diffusion 2.1 base · 512×512 · guidance 7.5 · seed 0").move_to([0, 2.55, 0])
        assert frames.get_top()[1] < cap.get_bottom()[1] - 0.08 and frames.width < 12.9, (frames.get_top(), frames.width)

        p1 = pill("plain Euler · evenly spaced steps", EULER_C, LIME_FILL, 22)
        p2 = pill("guidance: two network passes per step", CALL_C, AMBER_FILL, 22)
        pills = m.VGroup(p1, p2).arrange(m.RIGHT, buff=0.5).move_to([0, -0.85, 0])

        base_y, top_h = -2.9, 1.7
        xs = [fr.get_x() for fr in frames]
        bw = 0.9
        bars = m.VGroup(*[m.Rectangle(width=bw, height=max(top_h * LH_TIME[s] / LH_TIME[50], 0.03)).set_stroke(CALL_C, 2)
                          .set_fill(AMBER_FILL, 1).move_to([x, base_y + max(top_h * LH_TIME[s] / LH_TIME[50], 0.03) / 2, 0])
                          for s, x in zip(LH_STEPS, xs)])
        nums = m.VGroup(*[code(f"{LH_TIME[s]:.2f} s", 20, CALL_C).next_to(b, m.UP, buff=0.1) for s, b in zip(LH_STEPS, bars)])
        axis = m.Line([xs[0] - 0.8, base_y, 0], [xs[-1] + 0.8, base_y, 0]).set_stroke(FAINT, 2)
        t_lab = cite("seconds per picture, on this Mac (median of 4 seeds)").move_to([0, -3.3, 0])
        assert nums[-1].get_top()[1] < labs[0].get_bottom()[1] - 0.1, (nums[-1].get_top(), labs[0].get_bottom())

        self.say(
            "[Now a real model, on this Mac]: [the lighthouse at one, two, four, eight, twenty and fifty steps].",
            m.AnimationGroup(self.new_header("06 · the lighthouse", "A real model, step by step"), m.FadeIn(cap)),
            m.LaggedStart(*[m.FadeIn(m.Group(fr, lb), shift=0.1 * m.UP) for fr, lb in zip(frames, labs)], lag_ratio=0.25,
                          run_time=2.4),
        )
        self.say(
            "[One step is a coloured blur]. [By four, a lighthouse]. [By twenty, a sharp painting].",
            m.Circumscribe(frames[0], color=INK, buff=0.06),
            m.Circumscribe(frames[2], color=INK, buff=0.06),
            m.Circumscribe(frames[4], color=INK, buff=0.06),
        )
        self.say(
            "[The sampler is plain Euler on evenly spaced steps], [and guidance runs the network twice per step].",
            m.FadeIn(p1, shift=0.1 * m.UP),
            m.FadeIn(p2, shift=0.1 * m.UP),
        )
        self.say(
            "[Time grows with every step]: [a third of a second for one], [six point nine seconds for fifty].",
            m.AnimationGroup(m.FadeOut(pills), m.FadeIn(axis), m.FadeIn(t_lab),
                             m.LaggedStart(*[m.GrowFromEdge(b, m.DOWN) for b in bars], lag_ratio=0.15, run_time=1.4)),
            m.FadeIn(nums[0], shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(nums[1:], shift=0.1 * m.UP), m.Circumscribe(m.VGroup(bars[-1], nums[-1]), color=CALL_C, buff=0.08)),
        )
        self.say(
            "[Twenty steps take less than half the time of fifty], [for almost the same picture].",
            m.AnimationGroup(m.Circumscribe(m.VGroup(bars[4], nums[4]), color=CALL_C, buff=0.08)),
            m.AnimationGroup(m.Circumscribe(frames[4], color=DATA, buff=0.06), m.Circumscribe(frames[5], color=DATA, buff=0.06)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes=(
            "Integrate (sampling solves the probability-flow ODE in steps), place (put the steps where the detail is "
            "decided: Karras's ρ = 7), count (compare by network calls; Heun costs 2 per step). The strip: the trained "
            f"swirl network, same noise: evenly spaced Euler, 20 calls ({pct(STRAND['lin'])} stranded) → Karras-spaced "
            f"Euler, 20 calls ({pct(STRAND['kar'])}) → Karras-spaced Heun, 49 calls ({pct(H49_OFF)}). Next: Episode 11, "
            "Flow Matching."))
        halo = glow(TIME, radius=4.0)
        words = m.VGroup(*[title_text(w, 58, c) for w, c in (("Integrate", EULER_C), ("Place", TIME), ("Count", CALL_C))])
        subs = m.VGroup(*[code(s, 20, MUTED) for s in ("solve the ODE in steps", "where detail is decided",
                                                      "calls, not steps")])
        words.arrange(m.RIGHT, buff=1.8)
        on_baseline(words)
        for w, s in zip(words, subs):
            s.set_x(w.get_x())
        line_y = min(w.get_bottom()[1] for w in words) - 0.45
        for s in subs:
            sit(s, line_y)
        mid = baseline(words[0]) + 0.3
        dots = m.VGroup(*[m.Dot([(l_.get_right()[0] + r_.get_left()[0]) / 2, mid, 0], radius=0.07, color=MUTED)
                          for l_, r_ in zip(words, words[1:])])
        row = m.VGroup(words, dots, subs).move_to(1.8 * m.UP)
        assert row.width < 12.6 and all(a.get_right()[0] + 0.3 < b.get_left()[0] for a, b in zip(subs, subs[1:]))

        size = 1.6
        minis, caps = [], []
        for k, (key, lab) in enumerate((("mlp_ddim_20", f"even · Euler · 20 calls\n{pct(STRAND['lin'])} stranded"),
                                        ("mlp_euler_karras_20", f"Karras · Euler · 20 calls\n{pct(STRAND['kar'])} stranded"),
                                        ("mlp_heun_karras_49", f"Karras · Heun · 49 calls\n{pct(H49_OFF)} stranded"))):
            c = np.array([-4.0 + 4.0 * k, -0.2, 0.0])
            p = plane(c, size, LIM)
            g = m.VGroup(m.Square(size).set_stroke(EDGE, 2).move_to(c), swirl_band(p, 4, 0.16),
                         Swarm(p, RUNS[key][:600], radius=0.014))
            minis.append(g)
            caps.append(cite(lab).next_to(g[0], m.DOWN, buff=0.14))
        arrows = m.VGroup(*[arrow(a[0].get_right(), b[0].get_left(), MUTED, buff=0.25) for a, b in zip(minis, minis[1:])])
        tag = text("Smarter steps, better placed: the same network, better pictures per call.", 26, MUTED)
        mascot = pixel(0.42)
        nxt = pill("NEXT · FLOW MATCHING", TIME, BLUE_FILL, 22, mono=True)
        tag.move_to([0, -2.5, 0])
        foot = m.VGroup(tag, m.VGroup(mascot, nxt).arrange(m.RIGHT, buff=0.3).move_to([0, -3.12, 0]))
        assert caps[0].get_bottom()[1] > tag.get_top()[1] + 0.1 and foot.get_bottom()[1] > -3.5, (caps[0].get_bottom(), foot.get_bottom())

        self.say(
            "[To sum up]: [integrate], [place], [count].",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.3 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(dots[0]), m.FadeIn(words[1], shift=0.3 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(dots[1]), m.FadeIn(words[2], shift=0.3 * m.UP), m.FadeIn(subs[2])),
        )
        self.say(
            "[Sampling solves an equation in steps]: [put them where the detail is decided], [take smarter ones], "
            "[and count network calls, not steps].",
            m.AnimationGroup(m.FadeIn(minis[0]), m.FadeIn(caps[0])),
            m.AnimationGroup(m.GrowArrow(arrows[0]), m.FadeIn(minis[1]), m.FadeIn(caps[1])),
            m.AnimationGroup(m.GrowArrow(arrows[1]), m.FadeIn(minis[2]), m.FadeIn(caps[2])),
            m.Write(tag),
        )
        self.say(
            "[Next, a way to train on straight lines from noise to data].",
            m.AnimationGroup(m.FadeIn(mascot, shift=0.2 * m.UP), m.FadeIn(nxt, shift=0.2 * m.UP)),
        )
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, *minis, *caps, arrows, foot)], run_time=0.9)
