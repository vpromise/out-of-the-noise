"""Episodes 21-22 toy: a moving digit. Independent vs joint denoising of 8 frames.

Data (synthetic, from MNIST train): an MNIST digit (28x28) slides left -> right across a 28x52 canvas,
3 px per frame, 8 frames (x offset = x0 + 3k, x0 random in 0..3).
  joint model : the same U-Net as the digit model but with 8 input/output channels = the 8 frames stacked,
                so every denoising step sees the whole clip at once.
  frame model : the same U-Net with 1 channel, trained on single frames drawn from the same clips.
Both: DDPM eps-loss, linear beta, T=1000, class-conditional (10% label dropout), same steps/batch/lr.
Sampling: DDIM 50 steps, w=1. Independent clip = 8 frames from the frame model with 8 independent noises.
Metrics: flicker = mean |frame_{k+1} - shift(frame_k, 3 px)| (pixels in [0,1]); horizontal centre of mass per frame,
fitted slope (px/frame, truth = 3) and the residual std of a straight-line fit.

  python gpu/video_toy.py [--steps 3000 --batch 64]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import mlx.optimizers as optim  # noqa: E402
from mlx.utils import tree_flatten, tree_map, tree_unflatten  # noqa: E402

from common import ARRAYS, IMAGES, WEIGHTS, ddpm_schedule, load_mnist, save_grid, write_result  # noqa: E402
from models_mlx import NULL_LABEL, UNet, count_params  # noqa: E402

F, H, W, DX = 8, 28, 52, 3
p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=3000)
p.add_argument("--batch", type=int, default=64)
p.add_argument("--lr", type=float, default=5e-4)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

_, _, ABAR_np = ddpm_schedule()
ABAR = mx.array(ABAR_np.astype(np.float32))
xtr, ytr = load_mnist("train")


def make_clips(idx, x0s):
    """MNIST indices -> clips [B,H,W,F] in [-1,1]."""
    B = len(idx)
    clips = np.zeros((B, H, W, F), dtype=np.float32)
    for b in range(B):
        d = xtr[idx[b]].astype(np.float32) / 255.0
        for k in range(F):
            o = x0s[b] + DX * k
            clips[b, :, o:o + 28, k] = d
    return clips * 2 - 1


def shift_right(frame, px):
    out = np.full_like(frame, frame.min())
    out[:, px:] = frame[:, :-px]
    return out


def clip_metrics(clips01):
    """clips01: [N,F,H,W] in [0,1] -> dict."""
    fl, slopes, resid = [], [], []
    xs = np.arange(W)
    for c in clips01:
        fl.append(np.mean([np.abs(c[k + 1] - shift_right(c[k], DX)).mean() for k in range(F - 1)]))
        com = [(f.sum(0) * xs).sum() / max(f.sum(), 1e-6) for f in c]
        A = np.vstack([np.arange(F), np.ones(F)]).T
        sol, res, *_ = np.linalg.lstsq(A, np.array(com), rcond=None)
        slopes.append(sol[0])
        resid.append(np.std(np.array(com) - A @ sol))
    return dict(flicker=float(np.mean(fl)), slope_px_per_frame=float(np.mean(slopes)),
                slope_std=float(np.std(slopes)), com_residual_px=float(np.mean(resid)))


def train(kind):
    path = os.path.join(WEIGHTS, f"digit_video_{kind}_ema.safetensors")
    in_ch = F if kind == "joint" else 1
    mx.random.seed(args.seed)
    model = UNet(ch=32, mults=(1, 2, 4), nres=2, attn_res=(7,), heads=4, in_ch=in_ch)
    mx.eval(model.parameters())
    if os.path.exists(path):
        model.update(tree_unflatten(list(mx.load(path).items())))
        return model, json.load(open(path.replace(".safetensors", ".json")))
    sched = optim.join_schedules([optim.linear_schedule(1e-8, args.lr, 300),
                                  optim.cosine_decay(args.lr, args.steps - 300, end=args.lr * 0.1)], [300])
    opt = optim.Adam(learning_rate=sched)

    def loss_fn(m, x0, y, key):
        k1, k2, k3 = mx.random.split(key, 3)
        B = x0.shape[0]
        eps = mx.random.normal(x0.shape, key=k2)
        y = mx.where(mx.random.uniform(shape=(B,), key=k3) < 0.1, NULL_LABEL, y)
        t = mx.random.randint(0, 1000, (B,), key=k1)
        a = ABAR[t][:, None, None, None]
        return mx.mean((m(mx.sqrt(a) * x0 + mx.sqrt(1 - a) * eps, t.astype(mx.float32), y) - eps) ** 2)

    lg = nn.value_and_grad(model, loss_fn)
    state = [model.state, opt.state]

    def _step(x0, y, key):
        loss, g = lg(model, x0, y, key)
        g, _ = optim.clip_grad_norm(g, 1.0)
        opt.update(model, g)
        return loss

    step = mx.compile(_step, inputs=state, outputs=state)
    ema = tree_map(lambda a: mx.array(a), model.parameters())
    rng = np.random.default_rng(args.seed + (0 if kind == "joint" else 1))
    t0 = time.perf_counter()
    losses = []
    for s in range(args.steps):
        idx = rng.integers(0, 60000, args.batch)
        clips = make_clips(idx, rng.integers(0, 4, args.batch))
        if kind == "frame":
            k = rng.integers(0, F, args.batch)
            clips = clips[np.arange(args.batch), :, :, k][..., None]
        loss = step(mx.array(clips), mx.array(ytr[idx].astype(np.int32)), mx.random.key(args.seed * 7919 + s))
        d = min(0.999, (1 + s) / (10 + s))
        ema = tree_map(lambda e, p_: d * e + (1 - d) * p_, ema, model.parameters())
        mx.eval(loss, ema, state)
        losses.append(loss.item())
        if (s + 1) % 500 == 0:
            print(kind, s + 1, np.mean(losses[-500:]), round(time.perf_counter() - t0, 1), flush=True)
    wall = time.perf_counter() - t0
    mx.save_safetensors(path, dict(tree_flatten(ema)))
    meta = dict(kind=kind, in_ch=in_ch, params=count_params(model), steps=args.steps, batch=args.batch, wall_s=wall,
                final_loss=float(np.mean(losses[-200:])))
    json.dump(meta, open(path.replace(".safetensors", ".json"), "w"))
    model.update(ema)
    return model, meta


def ddim(model, xT, y, steps=50):
    ts = np.round(np.linspace(999, 0, steps)).astype(int)
    x = xT
    for i, t in enumerate(ts):
        eps = model(x, mx.full((x.shape[0],), float(t)), y)
        ab = float(ABAR_np[t]); abp = float(ABAR_np[ts[i + 1]]) if i + 1 < steps else 1.0
        x0 = (x - (1 - ab) ** 0.5 * eps) / ab ** 0.5
        x = abp ** 0.5 * x0 + (1 - abp) ** 0.5 * eps
        mx.eval(x)
    return np.clip(np.array(x), -1, 1)


if __name__ == "__main__":
    joint, mj = train("joint")
    frame, mf = train("frame")
    N = 100
    labels = (np.arange(N) % 10).astype(np.int32)
    rng = np.random.default_rng(321)
    xT = rng.standard_normal((N, H, W, F), dtype=np.float32)
    J = ddim(joint, mx.array(xT), mx.array(labels)).transpose(0, 3, 1, 2)  # [N,F,H,W]
    # independent: each of the 8 frames from its own noise (we reuse the joint model's noise channel k for frame k)
    Ind = np.stack([ddim(frame, mx.array(xT[..., k:k + 1]), mx.array(labels))[..., 0] for k in range(F)], 1)
    real = make_clips(rng.integers(0, 60000, N), rng.integers(0, 4, N)).transpose(0, 3, 1, 2)
    to01 = lambda a: (a + 1) / 2  # noqa: E731
    res = dict(joint=clip_metrics(to01(J)), independent=clip_metrics(to01(Ind)), real=clip_metrics(to01(real)),
               joint_model=mj, frame_model=mf, frames=F, canvas=[H, W], px_per_frame=DX, n_clips=N,
               sampler="DDIM 50 steps, w=1", note=__doc__.split("\n\n")[1])
    write_result("video_toy", res)
    u8 = lambda a: np.clip(np.round(to01(a) * 255), 0, 255).astype(np.uint8)  # noqa: E731
    np.savez_compressed(os.path.join(ARRAYS, "digit_video_toy.npz"), joint=u8(J), independent=u8(Ind), real=u8(real),
                        labels=labels, note="[clip, frame, 28, 52] uint8; label = clip % 10; joint = 8 frames denoised "
                                            "together, independent = each frame denoised on its own")
    for name, arr in [("joint", J), ("independent", Ind)]:
        save_grid(os.path.join(IMAGES, f"digit_video_toy_{name}_clips0-5.png"), u8(arr[:6]).reshape(-1, H, W), nrow=F,
                  scale=2)
    print(json.dumps({k: v for k, v in res.items() if k in ("joint", "independent", "real")}, indent=0))
