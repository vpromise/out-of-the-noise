"""Episode 10 · precompute the swirl runs the scene shows (~25 s): writes swirl_runs.npz beside this file.

Everything is toy.md §3's set-up, recomputed with diffusion_reference.py from the same shared noise
z = gaussian_noise(2000, seed=5), and asserted against research/experiments/toy.json where toy.json has the run:
  - the exact ODE solution (Heun, 1,000 Karras steps, exact denoiser) — toy.md's "ode error" reference
  - exact denoiser: DDIM (= Euler in σ, evenly spaced DDPM steps) 5 / 10 / 20 / 100, Euler on the Karras grid
    5 / 10 / 20 / 50 / 100, Heun on the Karras grid 5 / 9 / 19 / 49 / 99 network calls    → toy samplers/exact/runs
  - the trained MLP: DDIM 20, Euler Karras 20, Heun Karras 49, DDPM 1,000                 → toy samplers/mlp/runs
Extras for the scene (computed here, NOT in toy.json):
  - Euler on the cosine schedule's 20 evenly spaced noise levels (σ_i = σ(ᾱ_cosine(t_i)), t_i = 1000, 950, … 0),
    run on the same linear-trained networks via timestep_from_sigma (σ beyond the network's range is clipped to
    t = 1000, so the first level is σ = 157 like the others) — trained MLP and exact denoiser
  - the full paths of six particles (exact ODE and DDIM-5), one hero's DDIM 5 / 10 / 20 / 100 paths, and one
    big Heun step taken apart (σ 0.743 → 0.120, the 6th of 7 Karras steps for an 8-level grid)

Karras grid (toy.md): sigma_min = σ(t = 1) = 0.01, sigma_max = σ(t = 1000) = 157.4, ρ = 7.

    ~/.local/share/uv/tools/manimgx/bin/python precompute.py
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np

EP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EP_DIR.parent))
from diffusion_reference import *  # noqa: E402,F403

warnings.filterwarnings("ignore", message=".*encountered in matmul")

N, NOISE_SEED = 2000, 5
Z = gaussian_noise(N, NOISE_SEED)
MLP = load_swirl_model()
SMIN, SMAX = sigma_from_alpha_bar(ALPHA_BAR[[1, T]])
TOY = measured("toy", "samplers")
HOOK = [652, 587, 1837, 1639, 1485, 223]     # six particles, one per 60° of the swirl, that DDIM-5 strands far off
HERO = 1485                                   # the one followed in section 1
STEP_PARTICLE = 711                           # the one whose big step section 2 takes apart (its path bends there)


def karras(n, rho=7.0):
    return timestep_from_sigma(karras_sigmas(n, SMIN, SMAX, rho))


def cosine_grid(n):
    """The cosine schedule's n evenly spaced DDPM steps, as noise levels the linear-trained networks know."""
    return timestep_from_sigma(sigma_from_alpha_bar(ALPHA_BAR_COSINE[ddim_timesteps(n)]))


def main():
    start = time.perf_counter()
    out = {"z": Z}
    ref = heun_ode(exact_eps, Z, karras(1000))
    out["ref_final"] = ref.xs[-1]
    err = lambda x: float(np.linalg.norm(x - ref.xs[-1], axis=1).mean())  # noqa: E731

    def check(model, key, x, ode=True):
        r = TOY[model]["runs"][key]
        d, off = swirl_distance(x), off_swirl(x)
        assert abs(d - r["dist"]) < 2e-4 * max(1, r["dist"]) and abs(off - r["off"]) < 2e-3, (model, key, d, off, r)
        if ode:
            assert abs(err(x) - r["ode_error"]) < 2e-4 * max(1, r["ode_error"]), (model, key, err(x), r)
        out[f"{model}_{key}"] = x
        return x

    for s in (5, 10, 20, 100):
        check("exact", f"ddim_{s}", ddim_sample(exact_eps, Z, s).xs[-1])
    for s in (5, 10, 20, 50, 100):
        check("exact", f"euler_karras_{s}", euler_ode(exact_eps, Z, karras(s)).xs[-1])
    for steps in (3, 5, 10, 25, 50):
        check("exact", f"heun_karras_{2 * steps - 1}", heun_ode(exact_eps, Z, karras(steps)).xs[-1])
    check("mlp", "ddim_20", ddim_sample(MLP, Z, 20).xs[-1], ode=False)        # (MLP ode errors use its own reference)
    check("mlp", "euler_karras_20", euler_ode(MLP, Z, karras(20)).xs[-1], ode=False)
    check("mlp", "heun_karras_49", heun_ode(MLP, Z, karras(25)).xs[-1], ode=False)
    check("mlp", "ddpm_1000", ddpm_sample(MLP, Z, seed=1).xs[-1], ode=False)

    # extras: the cosine schedule's spacing, on the same networks
    out["cos_grid_20"] = cosine_grid(20)
    out["mlp_cos_20"] = euler_ode(MLP, Z, cosine_grid(20)).xs[-1]
    out["exact_cos_20"] = euler_ode(exact_eps, Z, cosine_grid(20)).xs[-1]
    assert np.allclose(out["mlp_cos_20"], ddim_sample(MLP, Z, cosine_grid(20)).xs[-1])   # Euler in σ = DDIM

    # six particles: exact paths and DDIM-5 hops (the exact denoiser treats every point on its own)
    sel = Z[HOOK]
    out["hook_ref"] = heun_ode(exact_eps, sel, karras(1000)).xs[::4]
    assert np.allclose(out["hook_ref"][-1], ref.xs[-1][HOOK])
    out["hook_ddim5"] = ddim_sample(exact_eps, sel, 5).xs
    for s in (5, 10, 20, 100):
        out[f"hero_ddim{s}"] = ddim_sample(exact_eps, Z[[HERO]], s).xs[:, 0]

    # one big step, taken apart (σ view: y = x_t/√ᾱ, dy/dσ = ε̂)
    sg = karras_sigmas(8, SMIN, SMAX)
    sa, sb = sg[5], sg[6]
    p = Z[[STEP_PARTICLE]]
    xa = heun_ode(exact_eps, p, timestep_from_sigma(karras_sigmas(600, sa, SMAX)[:-1])).xs[-1]
    seg = heun_ode(exact_eps, xa, timestep_from_sigma(np.geomspace(sa, sb, 400))).xs[:, 0]
    ca, cb = 1 / np.sqrt(1 + sa ** 2), 1 / np.sqrt(1 + sb ** 2)
    slope = lambda y, s: (y - mixture_posterior(y, 1.0, s).x0) / s  # noqa: E731
    ya, h = xa / ca, sb - sa
    d1 = slope(ya, sa)
    yt = ya + h * d1
    d2 = slope(yt, sb)
    trial, other, heun = yt * cb, (ya + h * d2) * cb, (ya + h * (d1 + d2) / 2) * cb
    ta, tb = timestep_from_sigma(np.array([sa, sb]))
    assert np.allclose(trial, euler_ode(exact_eps, xa, np.array([ta, tb])).xs[-1], atol=1e-12)
    assert np.allclose(heun, heun_ode(exact_eps, xa, np.array([ta, tb])).xs[-1], atol=1e-12)
    assert np.allclose(heun, (trial + other) / 2, atol=1e-12)
    out.update(step_sigmas=np.array([sa, sb]), step_start=xa[0], step_seg=seg, step_trial=trial[0],
               step_other=other[0], step_heun=heun[0])
    np.savez_compressed(EP_DIR / "swirl_runs.npz", **out)
    print(f"wrote swirl_runs.npz in {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
