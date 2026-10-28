"""Train the swirl's two tiny denoisers in NumPy (no torch): the same MLP twice.

  eps  (weights/swirl_mlp.npz):  DDPM's simple loss (Ho et al. 2020): t ~ U{1..1000}, x_t = √ᾱ·x0 + √(1−ᾱ)·ε,
                                 minimise ‖ε̂(x_t, t) − ε‖²  (linear β schedule, as diffusion_reference.ALPHA_BAR)
  flow (weights/swirl_flow.npz): flow matching / rectified flow: t ~ U(0, 1), x_t = (1−t)·x0 + t·ε,
                                 minimise ‖v̂(x_t, t) − (ε − x0)‖²  (t is fed to the embedding as 1000·t)

Fresh swirl points every batch (the data is the exact mixture, so this is "infinite data"); Adam with warm-up and
cosine decay, global-norm gradient clipping, and an EMA of the weights (what is saved), as in DDPM's training.
Run from the series folder:   research/experiments/.venv-toy/bin/python research/experiments/scripts/toy_train.py
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import diffusion_reference as dr  # noqa: E402

SIZES = [2 + 3 * dr.EMB_DIM, 128, 128, 128, 2]     # input: dr.mlp_input (x, features of x, time)


def init_params(rng):
    """Variance-preserving init for SiLU layers; the last layer starts small so the first predictions are ≈ 0."""
    params = {}
    for i, (m, n) in enumerate(zip(SIZES[:-1], SIZES[1:])):
        scale = np.sqrt(2.0 / m) if i < len(SIZES) - 2 else 0.1 / np.sqrt(m)
        params[f"W{i}"] = rng.normal(0, scale, (m, n))
        params[f"b{i}"] = np.zeros(n)
    return params


def loss_and_grads(params, x, t_embed, target):
    """Forward through dr.mlp_forward's network, keeping activations, then backpropagate the mean-squared error."""
    layers = len(params) // 2
    h = dr.mlp_input(x, t_embed)
    inputs, pre = [h], []
    for i in range(layers):
        z = h @ params[f"W{i}"] + params[f"b{i}"]
        if i < layers - 1:
            pre.append(z)
            h = dr.silu(z)
            inputs.append(h)
        else:
            h = z
    diff = h - target
    loss = float(np.mean(diff ** 2))
    g = 2 * diff / diff.size
    grads = {}
    for i in reversed(range(layers)):
        grads[f"W{i}"] = inputs[i].T @ g
        grads[f"b{i}"] = g.sum(0)
        if i > 0:
            z = pre[i - 1]
            sig = 0.5 * (1 + np.tanh(z / 2))
            g = (g @ params[f"W{i}"].T) * (sig * (1 + z * (1 - sig)))   # SiLU'(z) = σ(z)·(1 + z·(1 − σ(z)))
    return loss, grads


def batch(kind, rng, n):
    """One training batch: inputs (x_t, the time fed to the embedding) and the regression target."""
    k = rng.integers(0, len(dr.SWIRL.weights), n)          # equal weights: a uniform component
    x0 = dr.SWIRL.means[k] + dr.SWIRL.stds[k, None] * rng.standard_normal((n, 2))
    eps = rng.standard_normal((n, 2))
    if kind == "eps":
        t = rng.integers(1, dr.T + 1, n)
        return dr.q_sample(x0, t, eps), t.astype(float), eps
    t = rng.uniform(0, 1, n)
    return dr.flow_interpolant(x0, t, eps), 1000 * t, dr.flow_velocity(x0, eps)


def train(kind, steps, batch_size, lr, seed, ema_decay=0.999, clip=1.0, warmup=500):
    rng = np.random.default_rng(seed)
    params = init_params(rng)
    ema = {k: v.copy() for k, v in params.items()}
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(v) for k, v in params.items()}
    b1, b2, adam_eps = 0.9, 0.999, 1e-8
    losses, start = [], time.perf_counter()
    for step in range(1, steps + 1):
        x, t, target = batch(kind, rng, batch_size)
        loss, grads = loss_and_grads(params, x, t, target)
        losses.append(loss)
        norm = np.sqrt(sum(np.sum(g ** 2) for g in grads.values()))
        scale = min(1.0, clip / (norm + 1e-12))
        rate = lr * min(1.0, step / warmup) * 0.5 * (1 + np.cos(np.pi * step / steps))
        for k in params:
            g = grads[k] * scale
            m[k] = b1 * m[k] + (1 - b1) * g
            v[k] = b2 * v[k] + (1 - b2) * g * g
            params[k] -= rate * (m[k] / (1 - b1 ** step)) / (np.sqrt(v[k] / (1 - b2 ** step)) + adam_eps)
            ema[k] = ema_decay * ema[k] + (1 - ema_decay) * params[k]
        if step % 5000 == 0:
            print(f"  {kind} step {step:6d}  loss (last 1000) {np.mean(losses[-1000:]):.4f}  "
                  f"{time.perf_counter() - start:5.1f} s", flush=True)
    seconds = time.perf_counter() - start
    curve = np.array(losses).reshape(-1, steps // 200).mean(1)             # 200 points, each a mean of a stretch
    return ema, dict(loss_curve=curve, loss_curve_steps=np.arange(1, 201) * (steps // 200), train_seconds=seconds,
                     steps=steps, batch=batch_size, lr=lr, seed=seed, ema_decay=ema_decay,
                     final_loss=float(np.mean(losses[-2000:])), points_seen=steps * batch_size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["eps", "flow", "both"], default="both")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(dr.WEIGHTS))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for kind in (["eps", "flow"] if args.kind == "both" else [args.kind]):
        print(f"training {kind}: {args.steps} steps × batch {args.batch}, {dr.count_params(init_params(np.random.default_rng(0)))} parameters")
        params, log = train(kind, args.steps, args.batch, args.lr, args.seed + (kind == "flow"))
        name = "swirl_mlp.npz" if kind == "eps" else "swirl_flow.npz"
        np.savez(out / name, **params, **{k: np.asarray(val) for k, val in log.items()}, kind=kind)
        print(f"  saved {out / name}: final loss {log['final_loss']:.4f}, {log['train_seconds']:.0f} s")


if __name__ == "__main__":
    main()
