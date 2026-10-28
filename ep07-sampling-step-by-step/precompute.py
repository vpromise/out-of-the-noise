"""Episode 7's swirl runs, recomputed with diffusion_reference.py (~20 s) and checked against toy.json.

    python3 precompute.py      # writes swirl_runs.npz beside this file

The runs are research/experiments/toy.md §3's: z = gaussian_noise(2000, seed=5); DDPM ancestral sampling with
1,000 steps (σ² = β), per-step noise seeds 1 and 2, with the exact ε (exact_eps) and with the trained swirl MLP
(load_swirl_model). Two more runs for the "why fresh noise" section, computed the same way:
  • nonoise: DDPM's mean step with the fresh noise deleted (exact ε), from the same z;
  • one_start: 500 copies of ONE starting point (z[115]), DDPM with fresh noise (exact ε, seed 3).
Stored: every 5th state of the first 1,000 points (states 0, 5, …, 1000 → x_1000, x_995, …, x_0), the final
clouds, full-resolution paths of the first 80 particles, and every stored point's distance to the swirl.
"""

import sys
import warnings
from pathlib import Path

import numpy as np

EP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EP_DIR.parent))
from diffusion_reference import *  # noqa: E402,F403

warnings.filterwarnings("ignore", message=".*encountered in matmul")

N, SHOW, EVERY = 2000, 1000, 5
START_INDEX, START_COPIES = 115, 500
PATHS = 80                                 # the first 80 particles' full paths are kept (scenes pick a few)


def mean_only(eps_model, x):
    """DDPM's reverse step with the fresh noise deleted (not a derived sampler: notes §7)."""
    xs = [x]
    for t in range(T, 0, -1):
        beta = 1 - ALPHA_BAR[t] / ALPHA_BAR[t - 1]
        x = (x - beta / np.sqrt(1 - ALPHA_BAR[t]) * eps_model(x, t)) / np.sqrt(1 - beta)
        xs.append(x)
    return np.array(xs)


def main():
    z = gaussian_noise(N, seed=5)
    mlp = load_swirl_model()
    out = {"z": z}
    ex1 = ddpm_sample(exact_eps, z, seed=1)
    out["exact_states"] = ex1.xs[::EVERY, :SHOW].astype(np.float32)
    out["exact_paths"] = ex1.xs[:, :PATHS].astype(np.float32)
    out["exact_final"] = ex1.xs[-1]
    out["exact_final_seed2"] = ddpm_sample(exact_eps, z, seed=2).xs[-1]
    out["mlp_final"] = ddpm_sample(mlp, z, seed=1).xs[-1]
    out["mlp_final_seed2"] = ddpm_sample(mlp, z, seed=2).xs[-1]
    nn = mean_only(exact_eps, z)
    out["nonoise_states"] = nn[::EVERY, :SHOW].astype(np.float32)
    out["nonoise_final"] = nn[-1]
    one = np.repeat(z[START_INDEX:START_INDEX + 1], START_COPIES, axis=0)
    os_ = ddpm_sample(exact_eps, one, seed=3)
    out["one_start_states"] = os_.xs[::EVERY].astype(np.float32)
    out["one_start_final"] = os_.xs[-1]
    out["start_index"] = np.array(START_INDEX)

    def same_arm(a, b):
        return float(np.mean(nearest_on_swirl(a)[1] == nearest_on_swirl(b)[1]))

    h = lambda k: measured("toy", "headline", k)  # noqa: E731
    det = measured("toy", "samplers", "mlp", "determinism")
    assert round(swirl_distance(out["exact_final"]), 5) == h("ddpm1000_exact_dist") == 0.02454
    assert round(swirl_distance(out["mlp_final"]), 5) == h("ddpm1000_mlp_dist") == 0.02534
    assert round(same_arm(out["mlp_final"], out["mlp_final_seed2"]), 3) == h("ddpm_seed1_vs_seed2_same_arm_mlp") == 0.499
    gap = np.linalg.norm(out["mlp_final"] - out["mlp_final_seed2"], axis=1).mean()
    assert round(gap, 4) == det["ddpm_seed1_vs_seed2"] == 0.9003
    b = out["mlp_final"]
    rand = np.linalg.norm(b - b[np.random.default_rng(0).permutation(N)], axis=1).mean()
    assert round(rand, 4) == det["random_pair_baseline"] == 0.9053
    for name in ("exact", "nonoise", "one_start"):        # distance to the swirl of every stored state
        st = out[f"{name}_states"]
        out[f"{name}_dist"] = nearest_on_swirl(st.reshape(-1, 2))[0].reshape(st.shape[:2]).astype(np.float32)
    np.savez_compressed(EP_DIR / "swirl_runs.npz", **out)
    print("exact", swirl_distance(out["exact_final"]), "mlp", swirl_distance(out["mlp_final"]),
          "nonoise cov", swirl_coverage(out["nonoise_final"]), "one-start cov", swirl_coverage(out["one_start_final"]),
          "one-start arm0", np.mean(nearest_on_swirl(out["one_start_final"])[1] == 0))


if __name__ == "__main__":
    main()
