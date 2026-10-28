"""The toy (2D) experiments for "Out of the Noise": the swirl, its exact scores, the two trained MLPs, every sampler,
guidance, Langevin dynamics and FID's formula. NumPy only; writes research/experiments/toy.json (then run
toy_report.py for toy.md). Needs the trained weights from toy_train.py. Run from the series folder:
    research/experiments/.venv-toy/bin/python research/experiments/scripts/toy_experiments.py
"""

import json
import platform
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import diffusion_reference as dr  # noqa: E402

N = 2000            # samples per sampler run (all metrics use these)
STORE = 1000        # points stored per cloud in toy.json
NOISE_SEED = 5      # z = dr.gaussian_noise(N, NOISE_SEED): the shared starting noise of every sampler comparison
PARTICLES = 8       # trajectories stored for z[:8]
REF = dr.gaussian_stats(dr.swirl(20000, seed=123))     # FD reference: 20,000 data points
WALL = {}


def sig(v, digits=4):
    """Round to significant digits for JSON (keeps 4.04e-05 readable); inf → 'inf'."""
    v = float(v)
    return "inf" if np.isinf(v) else float(f"{v:.{digits}g}")


def cloud(points, n=STORE):
    return np.round(np.asarray(points)[:n], 3).tolist()


def pick(n, max_states=200):
    """Indices of at most max_states evenly spread states out of n, always including the first and last."""
    return np.unique(np.round(np.linspace(0, n - 1, min(n, max_states))).astype(int))


def paths(xs, particles=PARTICLES, max_states=200):
    """Trajectories of the first particles: list over particles of [[x, y], …], at most max_states states."""
    idx = pick(len(xs), max_states)
    return [np.round(xs[idx, i], 3).tolist() for i in range(particles)]


def quality(points):
    return dict(dist=sig(dr.swirl_distance(points)), off=sig(dr.off_swirl(points)),
                coverage=sig(dr.swirl_coverage(points)), fd=sig(dr.frechet_distance(*dr.gaussian_stats(points), *REF)))


def timed(name):
    def wrap(fn):
        def run(*a, **k):
            start = time.perf_counter()
            out = fn(*a, **k)
            WALL[name] = round(time.perf_counter() - start, 2)
            print(f"{name}: {WALL[name]} s", flush=True)
            return out
        return run
    return wrap


def chunked(fn, x, t, size=2000):
    return np.concatenate([fn(x[i:i + size], t if np.ndim(t) == 0 else t[i:i + size]) for i in range(0, len(x), size)])


# ---------------------------------------------------------------------------------------------- 1. forward process

@timed("forward")
def forward():
    ts = [0, 50, 100, 250, 500, 750, 1000]
    snaps = [0, 50, 100, 250, 500, 1000]
    x0, arm = dr.swirl_labelled(20000, seed=0)
    eps = dr.gaussian_noise(20000, seed=1)
    out = dict(snapshot_recipe="x0 = dr.swirl(1000, seed=0); eps = dr.gaussian_noise(1000, seed=1); "
                               "x_t = dr.q_sample(x0, t, eps, alpha_bar)  (alpha_bar = dr.ALPHA_BAR or dr.ALPHA_BAR_COSINE)",
               t=ts, snapshot_t=snaps, curve_t=list(range(0, 1001, 10)))
    for name, ab in (("linear", dr.ALPHA_BAR), ("cosine", dr.ALPHA_BAR_COSINE)):
        betas = dr.betas_from_alpha_bar(ab)
        table = []
        for t in ts:
            xt = dr.q_sample(x0, t, eps, ab)
            p0 = np.concatenate([dr.arm_probability(xt[i:i + 4000], t, 0, alpha_bar=ab) for i in range(0, 20000, 4000)])
            table.append(dict(t=t, beta=sig(betas[t]), alpha_bar=sig(ab[t]), signal=sig(np.sqrt(ab[t])),
                              noise=sig(np.sqrt(1 - ab[t])), snr=sig(dr.snr(ab[t])), log_snr=sig(dr.log_snr(ab[t])),
                              sigma=sig(dr.sigma_from_alpha_bar(ab[t])),
                              arm_accuracy=sig(np.mean((p0 > 0.5) == (arm == 0)), 3)))
        half = float(np.interp(0.5, ab[::-1], np.arange(1001)[::-1]))          # t where ᾱ = 0.5 (SNR = 1)
        one_pct = int(np.argmax(ab < 0.01))
        out[name] = dict(table=table, t_snr_1=sig(half, 4), t_alpha_bar_below_1pct=one_pct,
                         alpha_bar_T=sig(ab[-1]), curve_alpha_bar=[sig(a) for a in ab[::10]],
                         curve_log_snr=[sig(v) for v in dr.log_snr(ab[10::10])],
                         snapshots={str(t): cloud(dr.q_sample(x0[:STORE], t, eps[:STORE], ab)) for t in snaps})
    return out


# ---------------------------------------------------------------------------------------------- 2. score fields

@timed("score")
def score_fields(eps_model, v_model):
    rng = np.random.default_rng(7)
    n = 200000                                              # many points: the loss floor is a noisy average
    x0 = dr.swirl(n, seed=2)
    eps = rng.standard_normal((n, 2))
    levels = [5, 10, 25, 50, 100, 250, 500, 900]
    on_data = []
    for t in levels:
        xt = dr.q_sample(x0[:20000], t, eps[:20000])
        ex, ml = chunked(dr.exact_eps, xt, t), eps_model(xt, t)
        s_ex, s_ml = dr.score_from_eps(ex, dr.ALPHA_BAR[t]), dr.score_from_eps(ml, dr.ALPHA_BAR[t])
        cos = np.sum(s_ex * s_ml, 1) / (np.linalg.norm(s_ex, axis=1) * np.linalg.norm(s_ml, axis=1) + 1e-12)
        on_data.append(dict(t=t, sigma=sig(dr.sigma_from_alpha_bar(dr.ALPHA_BAR[t])),
                            loss_floor=sig(np.mean((ex - eps[:20000]) ** 2)), loss_mlp=sig(np.mean((ml - eps[:20000]) ** 2)),
                            excess=sig(np.mean((ml - ex) ** 2)),
                            rel_score_error=sig(np.sqrt(np.mean((s_ml - s_ex) ** 2) / np.mean(s_ex ** 2))),
                            median_angle_deg=sig(np.degrees(np.median(np.arccos(np.clip(cos, -1, 1)))), 3)))
    # loss floors averaged over the training distribution of t (the lowest training loss possible)
    t = rng.integers(1, 1001, n)
    xt = dr.q_sample(x0, t, eps)
    floor_eps = np.mean((chunked(dr.exact_eps, xt, t) - eps) ** 2)
    tf = rng.uniform(0, 1, n)
    xf = dr.flow_interpolant(x0, tf, eps)
    floor_flow = np.mean((chunked(dr.exact_velocity, xf, tf) - (eps - x0)) ** 2)
    # arrow fields on a grid, for scenes (the grid includes places the training data never visits)
    g = np.linspace(-1.5, 1.5, 21)
    grid = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)
    fields, grid_err = {}, []
    for t in (10, 100, 500):
        ab = dr.ALPHA_BAR[t]
        s_ex = dr.mixture_score(grid, t)
        s_ml = dr.score_from_eps(eps_model(grid, t), ab)
        dens = dr.mixture_density(grid, t)
        cos = np.sum(s_ex * s_ml, 1) / (np.linalg.norm(s_ex, axis=1) * np.linalg.norm(s_ml, axis=1) + 1e-12)
        ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        low = dens < np.quantile(dens, 0.25)
        grid_err.append(dict(t=t, median_angle_deg=sig(np.median(ang), 3), median_angle_low_density_deg=sig(np.median(ang[low]), 3),
                             median_angle_high_density_deg=sig(np.median(ang[~low]), 3)))
        fields[str(t)] = dict(exact=np.round(s_ex, 3).tolist(), mlp=np.round(s_ml, 3).tolist())
    return dict(on_data=on_data, loss_floor_eps=sig(floor_eps), loss_floor_flow=sig(floor_flow), grid_error=grid_err,
                grid=dict(axis=np.round(g, 3).tolist(), note="points = all (x, y) pairs, x fastest; scores (not ε)"),
                fields=fields)


# ---------------------------------------------------------------------------------------------- 3. samplers

def karras(n):
    smin, smax = dr.sigma_from_alpha_bar(dr.ALPHA_BAR[[1, dr.T]])
    return dr.timestep_from_sigma(dr.karras_sigmas(n, smin, smax))


@timed("samplers")
def samplers(eps_model):
    z = dr.gaussian_noise(N, NOISE_SEED)
    out = dict(noise=f"z = dr.gaussian_noise({N}, seed={NOISE_SEED})", karras_grid="sigma_min = σ(t=1) = 0.01, sigma_max = σ(t=1000) = 157.4, rho = 7")
    for name, model in (("exact", dr.exact_eps), ("mlp", eps_model)):
        res, clouds, trajs = {}, {}, {}
        ref = dr.heun_ode(model, z, karras(1000)).xs[-1]      # the ODE's answer, to 1e-5
        err = lambda x: sig(np.linalg.norm(x - ref, axis=1).mean())

        def run(key, fn, nfe, ode=True, keep=False, traj=False):
            start = time.perf_counter()
            tr = fn()
            x = tr.xs[-1]
            res[key] = dict(nfe=nfe, seconds=sig(time.perf_counter() - start, 3), **quality(x))
            if ode:
                res[key]["ode_error"] = err(x)
            if keep:
                clouds[key] = cloud(x)
            if traj:
                trajs[key] = dict(xs=paths(tr.xs), ts=[sig(t) for t in tr.ts[pick(len(tr.ts))]])
            return tr

        p1 = run("ddpm_1000", lambda: dr.ddpm_sample(model, z, seed=1), 1000, ode=False, keep=True, traj=True)
        p2 = run("ddpm_1000_seed2", lambda: dr.ddpm_sample(model, z, seed=2), 1000, ode=False)
        run("ddpm_1000_beta_tilde", lambda: dr.ddpm_sample(model, z, variance="beta_tilde", seed=1), 1000, ode=False, keep=True)
        ddim = {}
        for s in (1000, 100, 50, 20, 10, 5):
            ddim[s] = run(f"ddim_{s}", lambda: dr.ddim_sample(model, z, s), s, keep=s in (100, 50, 10, 5), traj=s in (100, 50, 10))
            run(f"euler_t_{s}", lambda: dr.euler_ode(model, z, s, space="t"), s)
            run(f"euler_karras_{s}", lambda: dr.euler_ode(model, z, karras(s)), s)
        for steps in (50, 25, 10, 5, 3):
            run(f"heun_{2 * steps - 1}", lambda: dr.heun_ode(model, z, steps), 2 * steps - 1)
            run(f"heun_karras_{2 * steps - 1}", lambda: dr.heun_ode(model, z, karras(steps)), 2 * steps - 1,
                keep=steps in (25, 10, 5), traj=steps == 10)
        a, b = ddim[100].xs[-1], p1.xs[-1]
        same = lambda u, v: float(np.mean(dr.nearest_on_swirl(u)[1] == dr.nearest_on_swirl(v)[1]))
        det = dict(ddim100_vs_ddim50=sig(np.linalg.norm(a - ddim[50].xs[-1], axis=1).mean()),
                   ddim100_vs_ddim20=sig(np.linalg.norm(a - ddim[20].xs[-1], axis=1).mean()),
                   ddim100_vs_ddim10=sig(np.linalg.norm(a - ddim[10].xs[-1], axis=1).mean()),
                   ddim100_repeat=sig(np.abs(a - dr.ddim_sample(model, z, 100).xs[-1]).max()),
                   ddpm_seed1_vs_seed2=sig(np.linalg.norm(b - p2.xs[-1], axis=1).mean()),
                   ddim100_vs_ddim50_same_arm=sig(same(a, ddim[50].xs[-1]), 3),
                   ddim100_vs_ddim10_same_arm=sig(same(a, ddim[10].xs[-1]), 3),
                   ddpm_seed1_vs_seed2_same_arm=sig(same(b, p2.xs[-1]), 3),
                   random_pair_baseline=sig(np.linalg.norm(b - b[np.random.default_rng(0).permutation(N)], axis=1).mean()))
        out[name] = dict(runs=res, determinism=det, clouds=clouds, trajectories=trajs,
                         ddpm_x0_guesses=paths(p1.x0s[::5]) if name == "exact" else None)
    data = dr.swirl(N, seed=1)
    out["data"] = quality(data)
    out["data_cloud"] = cloud(data)
    return out


# ---------------------------------------------------------------------------------------------- 4. flow matching

def straightness(xs):
    """Path length ÷ distance between the ends, per particle (1 = a straight line)."""
    length = np.linalg.norm(np.diff(xs, axis=0), axis=2).sum(0)
    return length / np.linalg.norm(xs[-1] - xs[0], axis=1)


@timed("flow")
def flow(eps_model, v_model):
    z = dr.gaussian_noise(N, NOISE_SEED)
    out = {}
    for name, eps_fn, v_fn in (("exact", dr.exact_eps, dr.exact_velocity), ("trained", eps_model, v_model)):
        fl = dr.flow_euler(v_fn, z, 200)
        dd = dr.ddim_sample(eps_fn, z, 200)
        r_f, r_d = straightness(fl.xs), straightness(dd.xs)
        y = dd.xs / np.sqrt(dr.alpha_bar_at(dd.ts))[:, None, None]       # the same DDIM path in the σ view
        r_s = straightness(y[1:])                                       # from σ ≈ 147 (first step) to 0
        res = dict(straightness_flow=dict(mean=sig(r_f.mean()), median=sig(np.median(r_f)), p90=sig(np.quantile(r_f, 0.9))),
                   straightness_ddpm_ode=dict(mean=sig(r_d.mean()), median=sig(np.median(r_d)), p90=sig(np.quantile(r_d, 0.9))),
                   straightness_ddpm_ode_sigma_view=dict(median=sig(np.median(r_s))),
                   paths_flow=paths(fl.xs), paths_ddpm_ode=paths(dd.xs))
        steps = {}
        clouds = {}
        for s in (1, 2, 4, 8, 16, 32):
            f = dr.flow_euler(v_fn, z, s).xs[-1]
            d = dr.ddim_sample(eps_fn, z, s).xs[-1]
            steps[str(s)] = dict(flow=quality(f), ddim=quality(d))
            if s in (1, 2, 4, 8):
                clouds[str(s)] = dict(flow=cloud(f), ddim=cloud(d))
        res.update(quality_vs_steps=steps, clouds=clouds)
        out[name] = res
    # training paths: one (x0, ε) pair each, as the loss sees them (flow: a straight line; DDPM: an arc)
    x0, e = dr.swirl(5000, seed=3), dr.gaussian_noise(5000, seed=4)
    vp = np.array([dr.q_sample(x0, t, e) for t in range(1000, -1, -5)])
    fp = np.array([dr.flow_interpolant(x0, t, e) for t in np.linspace(1, 0, 201)])
    r_vp, r_fp = straightness(vp), straightness(fp)
    out["training_paths"] = dict(recipe="x0 = dr.swirl(5000, seed=3); eps = dr.gaussian_noise(5000, seed=4); "
                                        "DDPM: dr.q_sample(x0, t, eps) for t = 1000 … 0; flow: dr.flow_interpolant(x0, t, eps)",
                                 straightness_flow=dict(mean=sig(r_fp.mean()), median=sig(np.median(r_fp))),
                                 straightness_ddpm=dict(mean=sig(r_vp.mean()), median=sig(np.median(r_vp)),
                                                        p90=sig(np.quantile(r_vp, 0.9))),
                                 paths_flow=paths(fp, max_states=101), paths_ddpm=paths(vp, max_states=101))
    tf = np.linspace(1, 0, 17)
    with np.errstate(divide="ignore"):
        same = dr.ddim_sample(dr.exact_eps, z, dr.timestep_from_sigma(tf / (1 - tf))).xs[-1]
    out["exact_flow16_vs_ddim_on_sigma_grid_max_diff"] = sig(np.abs(dr.flow_euler(dr.exact_velocity, z, 16).xs[-1] - same).max())
    return out


# ---------------------------------------------------------------------------------------------- 5. guidance

@timed("guidance")
def guidance():
    z = dr.gaussian_noise(N, NOISE_SEED)
    out = dict(convention="ε̃ = ε_u + w·(ε_c − ε_u) (guidance scale; w = 0 unconditional, 1 conditional). "
                          "Ho & Salimans' w′ = w − 1.", label=0, sampler="DDIM, 100 steps, η = 0, same noise z for every w",
               runs={}, clouds={}, profiles={}, blobs=blob_guidance(z))
    for w in (0, 1, 3, 7):
        x = dr.ddim_sample(dr.guided_eps(w, label=0), z, 100).xs[-1]
        d, arm, _ = dr.nearest_on_swirl(x)
        r = np.linalg.norm(x, axis=1)
        prof = dr.swirl_profile(x)[0]
        out["runs"][str(w)] = dict(
            on_arm0=sig(np.mean((arm == 0) & (d < 0.1)), 3), on_arm1=sig(np.mean((arm == 1) & (d < 0.1)), 3),
            off=sig(dr.off_swirl(x), 3), coverage_arm0=sig(dr.swirl_coverage(x, arm=0), 3),
            spread=sig(np.sqrt(np.trace(np.cov(x, rowvar=False)))), dist=sig(d.mean()),
            beyond_disc=sig(np.mean(r > 1.1), 3), max_radius=sig(r.max()),
            ddpm_check_on_arm0=None)
        out["profiles"][str(w)] = [sig(p, 3) for p in prof]
        out["clouds"][str(w)] = cloud(x)
    # the same with stochastic DDPM sampling (fewer points), to show it is not a quirk of the ODE
    zs = z[:500]
    for w in (0, 1, 3, 7):
        x = dr.ddpm_sample(dr.guided_eps(w, label=0), zs, seed=1).xs[-1]
        d, arm, _ = dr.nearest_on_swirl(x)
        out["runs"][str(w)]["ddpm_check_on_arm0"] = sig(np.mean((arm == 0) & (d < 0.1)), 3)
        out["runs"][str(w)]["ddpm_check_off"] = sig(dr.off_swirl(x), 3)
    data0 = dr.SWIRL.given(0).sample(N, seed=3)[0]
    out["arm0_data"] = dict(spread=sig(np.sqrt(np.trace(np.cov(data0, rowvar=False)))), coverage_arm0=sig(dr.swirl_coverage(data0, arm=0)))
    return out


def blob_guidance(z):
    """The three-blob toy: class 0 (top) at w = 0, 1, 3, 7 — where the guided samples go."""
    mix = dr.blobs_mixture()
    m0, s = mix.means[0], mix.stds[0]
    data = mix.given(0).sample(N, seed=3)[0]
    ref_beyond2 = float(np.mean(np.linalg.norm(data - m0, axis=1) / s > 2))
    out = dict(setup="dr.blobs_mixture(): 3 round blobs, std 0.15, centres at radius 0.6 (90°, 210°, 330°); class 0 = top",
               data_class0=dict(centre_y=sig(data[:, 1].mean()), std=sig(data.std(0).mean()), beyond_2sd=sig(ref_beyond2, 3),
                                beyond_3sd=sig(np.mean(np.linalg.norm(data - m0, axis=1) / s > 3), 3)),
               runs={}, clouds={}, data_cloud=cloud(dr.blobs(N, seed=1)[0]))
    for w in (0, 1, 3, 7):
        x = dr.ddim_sample(dr.guided_eps(w, 0, mix), z, 100).xs[-1]
        k = np.argmin(((x[:, None] - mix.means[None]) ** 2).sum(-1), 1)
        mine = x[k == 0]
        r = np.linalg.norm(x - m0, axis=1) / s
        out["runs"][str(w)] = dict(class0=sig(np.mean(k == 0), 3), centre_y=sig(mine[:, 1].mean()), std=sig(mine.std(0).mean()),
                                   beyond_2sd=sig(np.mean(r[k == 0] > 2), 3), beyond_3sd=sig(np.mean(r[k == 0] > 3), 3))
        out["clouds"][str(w)] = cloud(x)
    return out


# ---------------------------------------------------------------------------------------------- 6. Langevin

@timed("langevin")
def langevin():
    rng = np.random.default_rng(11)
    start = rng.uniform(-1.5, 1.5, (N, 2))
    sigmas = np.geomspace(1.0, 0.01, 10)
    runs = dict(annealed=dict(sigmas=sigmas, steps=100), plain_small=dict(sigmas=[0.01], steps=1000),
                plain_large=dict(sigmas=[0.3], steps=1000))
    out = dict(start="uniform noise in the square [−1.5, 1.5]², 2,000 particles, seed 11",
               settings=dict(annealed="10 noise levels σ = 1 → 0.01 (geometric), 100 steps each, ε = 2e-5 (α = ε·σ²/σ_L²)",
                             plain_small="σ = 0.01 only, 1,000 steps, α = 2e-5",
                             plain_large="σ = 0.3 only, 1,000 steps, α = 2e-5·(0.3/0.01)² = 0.018"), runs={}, clouds={}, paths={})
    data_prof = dr.swirl_profile(dr.swirl(N, seed=1)).ravel()
    for name, cfg in runs.items():
        step = 2e-5 if name != "plain_large" else 2e-5 * (0.3 / 0.01) ** 2
        tr = dr.langevin_sample(dr.mixture_score_sigma, start, cfg["sigmas"], cfg["steps"], step, seed=12)
        x = tr.xs[-1]
        prof = dr.swirl_profile(x).ravel()
        uniform = np.full(prof.size, 1 / prof.size)
        out["runs"][name] = dict(**quality(x), arm0_share=sig(np.mean(dr.nearest_on_swirl(x)[1] == 0), 3),
                                 profile_tv=sig(0.5 * np.abs(prof / max(prof.sum(), 1e-12) - uniform).sum(), 3),
                                 outer_half_share=sig(dr.swirl_profile(x)[:, 10:].sum() / max(dr.swirl_profile(x).sum(), 1e-12), 3),
                                 steps=int(len(tr.xs) - 1))
        out["clouds"][name] = cloud(x)
        out["paths"][name] = paths(tr.xs)
    out["data_profile_tv"] = sig(0.5 * np.abs(data_prof / data_prof.sum() - 1 / data_prof.size).sum(), 3)
    out["data_outer_half_share"] = sig(dr.swirl_profile(dr.swirl(N, seed=1))[:, 10:].sum() / dr.swirl_profile(dr.swirl(N, seed=1)).sum(), 3)
    return out


# ---------------------------------------------------------------------------------------------- 7. FID's formula

@timed("fid")
def fid():
    rng = np.random.default_rng(21)
    mu1, mu2 = np.zeros(2), np.array([1.0, 0.5])
    a, b = np.array([1.0, 1.0]), np.array([2.0, 0.5])
    closed = float(np.sum((mu1 - mu2) ** 2) + np.sum((np.sqrt(a) - np.sqrt(b)) ** 2))
    computed = dr.frechet_distance(mu1, np.diag(a), mu2, np.diag(b))
    rot = np.array([[np.cos(0.6), -np.sin(0.6)], [np.sin(0.6), np.cos(0.6)]])
    c2 = rot @ np.diag(b) @ rot.T
    direct = float(np.sum((mu1 - mu2) ** 2) + a.sum() + b.sum() - 2 * np.sqrt(np.linalg.eigvals(np.diag(a) @ c2).real).sum())
    out = dict(sanity=dict(case="N((0,0), I) vs N((1, 0.5), diag(2, 0.5))", closed_form=sig(closed, 8), computed=sig(computed, 8),
                           rotated_case_computed=sig(dr.frechet_distance(mu1, np.diag(a), mu2, c2), 8),
                           rotated_case_via_eig_of_product=sig(direct, 8)))
    bias = {}
    for d, reps in ((2, 200), (64, 40)):
        rows = {}
        for n in (100, 1000, 10000):
            vs_truth, two_sample = [], []
            for _ in range(reps):
                s1 = rng.standard_normal((n, d))
                s2 = rng.standard_normal((n, d))
                m1, c1 = dr.gaussian_stats(s1)
                vs_truth.append(dr.frechet_distance(m1, c1, np.zeros(d), np.eye(d)))
                two_sample.append(dr.frechet_distance(m1, c1, *dr.gaussian_stats(s2)))
            rows[str(n)] = dict(vs_truth_mean=sig(np.mean(vs_truth)), vs_truth_std=sig(np.std(vs_truth)),
                                two_sample_mean=sig(np.mean(two_sample)), two_sample_std=sig(np.std(two_sample)))
        bias[f"d{d}"] = dict(repeats=reps, rows=rows)
    out["bias"] = bias
    # the blind spot: a Gaussian with the swirl's mean and covariance
    data = dr.swirl(10000, seed=31)
    m, c = dr.gaussian_stats(data)
    gauss = rng.multivariate_normal(m, c, 10000)
    other = dr.swirl(10000, seed=32)
    out["blind_spot"] = dict(fd_swirl_vs_matched_gaussian=sig(dr.fd_from_samples(data, gauss)),
                             fd_swirl_vs_swirl=sig(dr.fd_from_samples(data, other)),
                             gaussian_quality=quality(gauss), swirl_quality=quality(other), gaussian_cloud=cloud(gauss))
    return out


# ---------------------------------------------------------------------------------------------- training logs

def training_log():
    out = {}
    for kind, name in (("eps", "swirl_mlp.npz"), ("flow", "swirl_flow.npz")):
        f = dr._load_npz(name)
        out[kind] = dict(file=f"research/experiments/weights/{name}", params=dr.count_params(dr.load_weights(name)),
                         steps=int(f["steps"]), batch=int(f["batch"]), lr=float(f["lr"]), ema_decay=float(f["ema_decay"]),
                         points_seen=int(f["points_seen"]), train_seconds=sig(f["train_seconds"], 3),
                         final_loss=sig(f["final_loss"]), loss_curve_steps=f["loss_curve_steps"].tolist(),
                         loss_curve=[sig(v) for v in f["loss_curve"]],
                         architecture="mlp_input(x, t) (98 numbers) → 3 × (Linear 128 → SiLU) → Linear 2",
                         objective="‖ε̂ − ε‖², t ~ U{1..1000}, linear β" if kind == "eps" else
                                   "‖v̂ − (ε − x0)‖², t ~ U(0, 1), x_t = (1 − t)·x0 + t·ε")
    return out


def machine():
    try:
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        mem = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout) / 2 ** 30
    except Exception:            # not a Mac
        cpu, mem = platform.processor(), 0
    return dict(cpu=cpu, memory_gb=round(mem), os=platform.platform(), python=platform.python_version(),
                numpy=np.__version__, date=str(date.today()))


def main():
    start = time.perf_counter()
    eps_model, v_model = dr.load_swirl_model(), dr.load_swirl_flow()
    res = dict(about=dict(machine=machine(), compute="NumPy on the CPU (Apple Accelerate BLAS); no GPU",
                          commands=["research/experiments/.venv-toy/bin/python research/experiments/scripts/toy_train.py",
                                    "research/experiments/.venv-toy/bin/python research/experiments/scripts/toy_experiments.py",
                                    "research/experiments/.venv-toy/bin/python research/experiments/scripts/toy_report.py"],
                          conventions="see the docstring of diffusion_reference.py; DDPM t = 0..1000 (linear β unless "
                                      "'cosine'); flow t ∈ [0, 1] with t = 1 noise; σ = √((1 − ᾱ)/ᾱ)",
                          metrics=dict(dist="mean distance to the swirl's centre line (data: 0.024; lower than that = too tidy)",
                                       off="fraction of samples > 0.1 from the curve (data: 0.001)",
                                       coverage="fraction of the 40 curve pieces with ≥ ¼ of their fair share (data: 1.0)",
                                       fd="Fréchet distance of raw 2D coordinates vs 20,000 data points (FID's formula)",
                                       ode_error="mean distance to the exact ODE solution from the same noise (Heun, 1,000 Karras steps)")))
    res["training"] = training_log()
    res["forward"] = forward()
    res["score"] = score_fields(eps_model, v_model)
    res["samplers"] = samplers(eps_model)
    res["flow"] = flow(eps_model, v_model)
    res["guidance"] = guidance()
    res["langevin"] = langevin()
    res["fid"] = fid()
    res["about"]["wall_seconds"] = dict(WALL, total=round(time.perf_counter() - start, 1),
                                        train_eps=res["training"]["eps"]["train_seconds"],
                                        train_flow=res["training"]["flow"]["train_seconds"])
    res["headline"] = headline(res)
    path = dr.EXPERIMENTS / "toy.json"
    path.write_text(json.dumps(res, ensure_ascii=False, separators=(",", ":")))
    print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB) in {time.perf_counter() - start:.0f} s")


def headline(r):
    """~30 numbers scenes assert (measured("toy", "headline", key)). No units unless the key names one."""
    lin = {row["t"]: row for row in r["forward"]["linear"]["table"]}
    cos = {row["t"]: row for row in r["forward"]["cosine"]["table"]}
    se, sm = r["samplers"]["exact"]["runs"], r["samplers"]["mlp"]["runs"]
    det = r["samplers"]["mlp"]["determinism"]
    on = {row["t"]: row for row in r["score"]["on_data"]}
    ft, tp = r["flow"]["trained"], r["flow"]["training_paths"]
    g, gb = r["guidance"]["runs"], r["guidance"]["blobs"]
    lv = r["langevin"]["runs"]
    fb = r["fid"]["bias"]
    return dict(
        # 1 forward process (linear β unless named)
        alpha_bar_500_linear=lin[500]["alpha_bar"], alpha_bar_500_cosine=cos[500]["alpha_bar"],
        alpha_bar_1000_linear=lin[1000]["alpha_bar"],
        t_snr_1_linear=r["forward"]["linear"]["t_snr_1"], t_snr_1_cosine=r["forward"]["cosine"]["t_snr_1"],
        arm_accuracy_t500_linear=lin[500]["arm_accuracy"],
        # 2 the trained MLP and its score field
        mlp_params=r["training"]["eps"]["params"], mlp_train_seconds=r["training"]["eps"]["train_seconds"],
        loss_floor_eps=r["score"]["loss_floor_eps"], final_loss_eps=r["training"]["eps"]["final_loss"],
        mlp_rel_score_error_t10=on[10]["rel_score_error"], mlp_rel_score_error_t500=on[500]["rel_score_error"],
        # 3 samplers (trained MLP; the same noise for every sampler; data: dist 0.024, off 0.001)
        data_dist=r["samplers"]["data"]["dist"],
        ddpm1000_exact_dist=se["ddpm_1000"]["dist"], ddpm1000_mlp_dist=sm["ddpm_1000"]["dist"],
        ddim100_mlp_dist=sm["ddim_100"]["dist"],
        ddim50_mlp_off=sm["ddim_50"]["off"], ddim20_mlp_off=sm["ddim_20"]["off"], ddim10_mlp_off=sm["ddim_10"]["off"],
        euler_karras20_mlp_off=sm["euler_karras_20"]["off"], heun_karras49_mlp_off=sm["heun_karras_49"]["off"],
        ddim100_vs_ddim50_same_arm_mlp=det["ddim100_vs_ddim50_same_arm"],
        ddpm_seed1_vs_seed2_same_arm_mlp=det["ddpm_seed1_vs_seed2_same_arm"],
        # 4 flow matching vs the DDPM-trained model
        training_path_ratio_ddpm_mean=tp["straightness_ddpm"]["mean"], training_path_ratio_flow_mean=tp["straightness_flow"]["mean"],
        flow8_trained_off=ft["quality_vs_steps"]["8"]["flow"]["off"], ddim8_trained_off=ft["quality_vs_steps"]["8"]["ddim"]["off"],
        flow16_trained_off=ft["quality_vs_steps"]["16"]["flow"]["off"], ddim16_trained_off=ft["quality_vs_steps"]["16"]["ddim"]["off"],
        # 5 guidance (exact scores; w in the guidance-scale form, w = 1 is plain conditional)
        swirl_w7_coverage_arm0=g["7"]["coverage_arm0"], swirl_w1_spread=g["1"]["spread"], swirl_w7_spread=g["7"]["spread"],
        blobs_w1_centre_y=gb["runs"]["1"]["centre_y"], blobs_w7_centre_y=gb["runs"]["7"]["centre_y"],
        blobs_w7_beyond_2sd=gb["runs"]["7"]["beyond_2sd"], blobs_data_beyond_2sd=gb["data_class0"]["beyond_2sd"],
        # 6 Langevin (exact score): share of particles on the outer half of the arms (data 0.50)
        langevin_annealed_outer_share=lv["annealed"]["outer_half_share"], langevin_plain_outer_share=lv["plain_small"]["outer_half_share"],
        # 7 FID's formula: bias of FD(sample, truth) for N(0, I) in 2-D and 64-D; the blob that fools it
        fd_bias_2d_n100=fb["d2"]["rows"]["100"]["vs_truth_mean"], fd_bias_2d_n10000=fb["d2"]["rows"]["10000"]["vs_truth_mean"],
        fd_bias_64d_n1000=fb["d64"]["rows"]["1000"]["vs_truth_mean"],
        fd_swirl_vs_matched_gaussian=r["fid"]["blind_spot"]["fd_swirl_vs_matched_gaussian"],
    )


if __name__ == "__main__":
    main()
