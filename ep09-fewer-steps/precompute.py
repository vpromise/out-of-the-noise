"""Episode 9 · precompute the swirl runs the scene replays (~12 s): writes swirl_runs.npz beside this file.

The runs are toy.md §3's, recomputed with diffusion_reference.py and the trained swirl MLP from the same shared noise
z = gaussian_noise(2000, seed=5), and asserted against research/experiments/toy.json:
  - DDPM ancestral, 1,000 steps, fresh-noise seeds 1 and 2 (σ² = β)          → toy samplers/mlp/runs/ddpm_1000(_seed2)
  - DDIM (η = 0) with 100 / 50 / 20 / 10 steps                                → toy samplers/mlp/runs/ddim_*
  - same-arm agreement (DDIM-100 vs DDIM-50; DDPM seed 1 vs seed 2)          → toy samplers/mlp/determinism
Extras for the scene (computed here, not in toy.json): the full paths of a few particles, DDIM-1000 paths for the
particles DDIM-10 strands, and DDIM inversion (data → noise → data) with the same MLP.

    ~/.local/share/uv/tools/manimgx/bin/python precompute.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np

EP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EP_DIR.parent))
from diffusion_reference import *  # noqa: E402,F403

warnings.filterwarnings("ignore", message=".*encountered in matmul")

MLP = load_swirl_model()
N, NOISE_SEED = 2000, 5
Z = gaussian_noise(N, NOISE_SEED)
TOY = measured("toy", "samplers", "mlp")


def invert(eps_model, x, steps=100):
    """DDIM inversion: the same Euler steps, run from t = 0 up to t = T (the scene shows this excerpt)."""
    ts = ddim_timesteps(steps)[::-1]                 # 0 → 1000: the same grid, backwards
    xs = [x]
    for t, s in zip(ts[:-1], ts[1:]):
        ab_t, ab_s = alpha_bar_at(t), alpha_bar_at(s)
        eps = eps_model(x, t)                        # the noise the network sees
        x0 = x0_from_eps(x, eps, ab_t)               # its clean guess
        x = np.sqrt(ab_s) * x0 + np.sqrt(1 - ab_s) * eps   # re-noised one level UP
        xs.append(x)
    return np.array(xs)


def check(name, x):
    ref = TOY["runs"][name]
    d, off = swirl_distance(x), off_swirl(x)
    assert abs(d - ref["dist"]) < 2e-4 and abs(off - ref["off"]) < 2e-3, (name, d, off, ref)
    return d, off


def main():
    out = {"z": Z}
    p1 = ddpm_sample(MLP, Z, seed=1)
    p2 = ddpm_sample(MLP, Z, seed=2)
    check("ddpm_1000", p1.xs[-1])
    check("ddpm_1000_seed2", p2.xs[-1])
    ddim = {s: ddim_sample(MLP, Z, s) for s in (100, 50, 20, 10)}
    for s, tr in ddim.items():
        check(f"ddim_{s}", tr.xs[-1])
        out[f"ddim{s}_final"] = tr.xs[-1]
    out["ddpm1_final"], out["ddpm2_final"] = p1.xs[-1], p2.xs[-1]
    arm = lambda x: nearest_on_swirl(x)[1]  # noqa: E731
    same_ddim = float(np.mean(arm(ddim[100].xs[-1]) == arm(ddim[50].xs[-1])))
    same_ddpm = float(np.mean(arm(p1.xs[-1]) == arm(p2.xs[-1])))
    det = TOY["determinism"]
    assert round(same_ddim, 3) == det["ddim100_vs_ddim50_same_arm"] and round(same_ddpm, 3) == det["ddpm_seed1_vs_seed2_same_arm"]
    assert np.abs(ddim_sample(MLP, Z, 100).xs[-1] - ddim[100].xs[-1]).max() == 0.0          # run twice: identical

    # the particles traced on screen: starts inside the view, DDPM's two runs on different arms, DDIM's two on one arm
    r = np.linalg.norm(Z, axis=1)
    a1, a2, d1, d2 = arm(p1.xs[-1]), arm(p2.xs[-1]), arm(ddim[100].xs[-1]), arm(ddim[50].xs[-1])
    on = lambda x: nearest_on_swirl(x)[0] < 0.1  # noqa: E731
    inside = (np.abs(p1.xs).max(axis=(0, 2)) < 2.05) & (np.abs(p2.xs).max(axis=(0, 2)) < 2.05)   # paths stay in view
    ok = (r > 0.8) & (r < 2.0) & (a1 != a2) & (d1 == d2) & on(p1.xs[-1]) & on(p2.xs[-1]) & on(ddim[100].xs[-1]) & inside
    cand = np.flatnonzero(ok)
    ang = np.arctan2(Z[cand, 1], Z[cand, 0])
    chosen = [0]                                         # spread round the circle: each next pick the farthest in angle
    while len(chosen) < 6:
        gap = np.min(np.abs(np.angle(np.exp(1j * (ang[:, None] - ang[chosen][None, :])))), axis=1)
        chosen.append(int(np.argmax(gap)))
    picks = cand[chosen]
    assert len(picks) == 6, picks
    out["picks"] = picks
    out["ddpm1_paths"], out["ddpm2_paths"] = p1.xs[:, picks], p2.xs[:, picks]
    out["ddim100_paths"], out["ddim50_paths"] = ddim[100].xs[:, picks], ddim[50].xs[:, picks]

    # stranded by DDIM-10 but on the swirl after DDIM-1000 (the ODE solved finely): the last big jump lands in a gap
    far10 = nearest_on_swirl(ddim[10].xs[-1])[0] > 0.15
    stranded = np.flatnonzero(far10 & (r < 1.6))[:40]
    fine = ddim_sample(MLP, Z[stranded], 1000)
    good = stranded[nearest_on_swirl(fine.xs[-1])[0] < 0.05][:3]
    assert len(good) == 3
    out["stranded"] = good
    out["stranded_fine"] = ddim_sample(MLP, Z[good], 1000).xs
    out["stranded_ddim10"] = ddim[10].xs[:, good]
    out["stranded_x0s10"] = ddim[10].x0s[:, good]

    # inversion: a few data points up to noise and back (100 steps each way); errors on 300 points at 100 and 1000
    pts = swirl(12, seed=21)
    up = invert(MLP, pts, 100)
    down = ddim_sample(MLP, up[-1], 100)
    out["inv_points"], out["inv_up"], out["inv_down"] = pts, up, down.xs
    many = swirl(300, seed=11)
    for s in (100, 1000):
        back = ddim_sample(MLP, invert(MLP, many, s)[-1], s).xs[-1]
        out[f"inv_err{s}"] = np.linalg.norm(back - many, axis=1)
    np.savez_compressed(EP_DIR / "swirl_runs.npz", **out)
    print("picks", picks, "stranded", good, "same arm", same_ddim, same_ddpm,
          "inv median", {s: float(np.median(out[f'inv_err{s}'])) for s in (100, 1000)})


if __name__ == "__main__":
    main()
