"""Samplers for the digit models (MLX).

Guidance convention (same as Stable Diffusion's "guidance scale"):
    pred = pred_uncond + w * (pred_cond - pred_uncond)
    w = 0 -> unconditional (label ignored), w = 1 -> plain conditional (no guidance, 1 network pass),
    w > 1 -> classifier-free guidance (2 passes, batched).  Ho & Salimans (2022) write the same thing
    with w_HS = w - 1.

DDPM ancestral: x_{t-1} = (x_t - beta_t/sqrt(1-abar_t) * eps) / sqrt(alpha_t) + sigma_t z,
    sigma_t^2 = beta_t (1-abar_{t-1})/(1-abar_t)  (posterior variance, "fixed small"); no x0 clipping.
DDIM (eta = 0): S timesteps evenly spaced from t = 999 down to t = 0 ("trailing"), last step jumps to abar = 1.
Flow (rectified flow): x_t = (1-t) x0 + t eps, v = eps - x0; Euler from t = 1 to t = 0 in S equal steps.
All samplers clip only the final image to [-1, 1].
"""
import json
import os

import mlx.core as mx
import numpy as np
from mlx.utils import tree_unflatten

from common import T, WEIGHTS, ddpm_schedule
from models_mlx import NULL_LABEL, build

BETAS, ALPHAS, ABAR = ddpm_schedule()


def load_model(name):
    meta = json.load(open(os.path.join(WEIGHTS, f"digit_{name}_ema.json")))
    m = build(meta["config"])
    m.update(tree_unflatten(list(mx.load(os.path.join(WEIGHTS, f"digit_{name}_ema.safetensors")).items())))
    mx.eval(m.parameters())
    return m, meta


class Denoiser:
    """Wraps a model; __call__(x, t_model, y, w) returns the guided prediction (eps or v)."""

    def __init__(self, model):
        self.model = model
        self.f = mx.compile(lambda x, t, y: model(x, t, y))

    def __call__(self, x, t, y, w):
        B = x.shape[0]
        tt = mx.full((B,), float(t), dtype=mx.float32)
        if w == 1.0:
            return self.f(x, tt, y)
        nul = mx.full((B,), NULL_LABEL, dtype=mx.int32)
        if w == 0.0:
            return self.f(x, tt, nul)
        out = self.f(mx.concatenate([x, x]), mx.concatenate([tt, tt]), mx.concatenate([y, nul]))
        c, u = out[:B], out[B:]
        return u + w * (c - u)


def nfe_per_sample(sampler, steps, w):
    return steps * (1 if w in (0.0, 1.0) else 2)


def ddpm_noise_key(seed, batch_start, t):
    return mx.random.key(seed * 1_000_003 + batch_start * 1009 + t)


def sample(den, sampler, steps, xT, y, w, seed=0, batch_start=0, record=None, share_noise=False):
    """xT: mx [B,28,28,1]; y: mx int32 [B]. record: None or a set of step indices (0 = first step) to keep.
    share_noise: DDPM only - draw one per-step noise image and use it for every sample in the batch.
    Returns (x0 in [-1,1], traj) where traj = dict(step_index, t, x_t (input to that step), x0_hat)."""
    x = xT
    traj = dict(i=[], t=[], x_t=[], x0_hat=[])

    def keep(i, t, x_in, x0h):
        if record is not None and i in record:
            traj["i"].append(i); traj["t"].append(t)
            traj["x_t"].append(np.array(x_in)); traj["x0_hat"].append(np.array(x0h))

    if sampler == "ddpm":
        assert steps == T
        for i, t in enumerate(range(T - 1, -1, -1)):
            eps = den(x, t, y, w)
            a, ab = float(ALPHAS[t]), float(ABAR[t])
            x0h = (x - (1 - ab) ** 0.5 * eps) / ab ** 0.5
            keep(i, t, x, x0h)
            mu = (x - (float(BETAS[t]) / (1 - ab) ** 0.5) * eps) / a ** 0.5
            if t > 0:
                var = float(BETAS[t] * (1 - ABAR[t - 1]) / (1 - ABAR[t]))
                zshape = (1,) + tuple(x.shape[1:]) if share_noise else x.shape
                z = mx.random.normal(zshape, key=ddpm_noise_key(seed, batch_start, t))
                x = mu + var ** 0.5 * z
            else:
                x = mu
            mx.eval(x)
    elif sampler == "ddim":
        ts = np.round(np.linspace(T - 1, 0, steps)).astype(int) if steps > 1 else np.array([T - 1])
        for i, t in enumerate(ts):
            eps = den(x, int(t), y, w)
            ab = float(ABAR[t])
            ab_prev = float(ABAR[ts[i + 1]]) if i + 1 < len(ts) else 1.0
            x0h = (x - (1 - ab) ** 0.5 * eps) / ab ** 0.5
            keep(i, int(t), x, x0h)
            x = ab_prev ** 0.5 * x0h + (1 - ab_prev) ** 0.5 * eps
            mx.eval(x)
    elif sampler == "euler":
        ts = np.linspace(1.0, 0.0, steps + 1)
        for i in range(steps):
            t, t_next = float(ts[i]), float(ts[i + 1])
            v = den(x, t * 1000.0, y, w)
            x0h = x - t * v
            keep(i, t, x, x0h)
            x = x + (t_next - t) * v
            mx.eval(x)
    else:
        raise ValueError(sampler)
    return mx.clip(x, -1.0, 1.0), traj


def initial_noise(n, seed=1234):
    """Shared starting noise: sample i always starts from the same x_T, whatever the sampler/batching."""
    return np.random.default_rng(seed).standard_normal((n, 28, 28, 1), dtype=np.float32)
