"""Episode 6's Langevin runs, recomputed exactly as research/experiments/scripts/toy_experiments.py's langevin()
(uniform start in [-1.5, 1.5]², 2,000 particles, rng 11; langevin_sample seed 12, exact score mixture_score_sigma):
  annealed:    10 levels σ = 1 → 0.01 (geometric), 100 steps each, ε = 2e-5   (Song & Ermon 2019's recipe)
  plain_small: σ = 0.01 only, 1,000 steps, α = 2e-5
Writes langevin_runs.npz (every 5th state of the first 1,000 particles, float32; the final 2,000 in full; each shown
particle's distance to the swirl). scene.py asserts the result against toy.json. ~8 s.

    python3 precompute.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from diffusion_reference import (langevin_sample, mixture_score_sigma, nearest_on_swirl,  # noqa: E402
                                 measured)

warnings.filterwarnings("ignore", category=RuntimeWarning)  # NumPy/Accelerate's spurious matmul warnings

N, SHOWN, EVERY = 2000, 1000, 5
start = np.random.default_rng(11).uniform(-1.5, 1.5, (N, 2))
runs = dict(annealed=(np.geomspace(1.0, 0.01, 10), 100, 2e-5), plain_small=([0.01], 1000, 2e-5))
out = dict(start=start)
for name, (sigmas, steps, eps) in runs.items():
    tr = langevin_sample(mixture_score_sigma, start, sigmas, steps, eps, seed=12)
    xs = tr.xs[::EVERY, :SHOWN]
    out[f"{name}_states"] = xs.astype(np.float32)
    out[f"{name}_sigma"] = tr.ts[::EVERY]
    out[f"{name}_final"] = tr.xs[-1]
    out[f"{name}_dist"] = np.array([nearest_on_swirl(x)[0] for x in xs]).astype(np.float16)
    stored = np.array(measured("toy", "langevin", "clouds", name))
    assert np.abs(np.round(tr.xs[-1][:SHOWN], 3) - stored).max() < 1e-9, name   # the very run in toy.json
np.savez_compressed(HERE / "langevin_runs.npz", **out)
print("wrote langevin_runs.npz", {k: v.shape for k, v in out.items()})
