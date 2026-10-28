"""Benchmark: same class-conditional U-Net in MLX vs PyTorch-MPS (train step + inference throughput).

Usage: python gpu/bench_frameworks.py  -> logs/bench_frameworks.json
"""
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import LOGS  # noqa: E402

CONFIGS = [
    dict(ch=32, mults=(1, 2, 4), attn_res=(7,)),
    dict(ch=48, mults=(1, 2, 4), attn_res=(7,)),
    dict(ch=64, mults=(1, 2, 2), attn_res=(7,)),
    dict(ch=64, mults=(1, 2, 4), attn_res=(7,)),
]
BATCH, INF_BATCH = 256, 1000


def bench_mlx(cfg):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from models_mlx import UNet, count_params

    model = UNet(**cfg)
    mx.eval(model.parameters())
    nparams = count_params(model)
    opt = optim.Adam(learning_rate=2e-4)

    def loss_fn(m, x, t, y, eps):
        return mx.mean((m(x, t, y) - eps) ** 2)

    lg = nn.value_and_grad(model, loss_fn)
    state = [model.state, opt.state]

    @mx.compile
    def step(x, t, y, eps):
        loss, g = lg(model, x, t, y, eps)
        opt.update(model, g)
        return loss

    step = mx.compile(step, inputs=state, outputs=state)
    x = mx.random.normal((BATCH, 28, 28, 1))
    t = mx.random.randint(0, 1000, (BATCH,)).astype(mx.float32)
    y = mx.random.randint(0, 11, (BATCH,))
    for _ in range(5):
        mx.eval(step(x, t, y, x), state)
    n = 30
    t0 = time.perf_counter()
    for _ in range(n):
        mx.eval(step(x, t, y, x), state)
    train = (time.perf_counter() - t0) / n

    fwd = mx.compile(lambda a, b, c: model(a, b, c))
    xi = mx.random.normal((INF_BATCH, 28, 28, 1))
    ti = mx.full((INF_BATCH,), 500.0)
    yi = mx.random.randint(0, 11, (INF_BATCH,))
    for _ in range(3):
        mx.eval(fwd(xi, ti, yi))
    n = 10
    t0 = time.perf_counter()
    for _ in range(n):
        mx.eval(fwd(xi, ti, yi))
    inf = (time.perf_counter() - t0) / n
    return dict(params=nparams, train_step_s=train, train_img_per_s=BATCH / train, infer_img_per_s=INF_BATCH / inf)


def bench_torch(cfg):
    import torch
    import torch.nn as tnn
    import torch.nn.functional as F

    dev = torch.device("mps")

    def temb_fn(t, dim):
        half = dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        a = t[:, None].float() * freqs[None]
        return torch.cat([torch.cos(a), torch.sin(a)], -1)

    class Res(tnn.Module):
        def __init__(s, cin, cout, td):
            super().__init__()
            s.n1, s.c1 = tnn.GroupNorm(8, cin), tnn.Conv2d(cin, cout, 3, padding=1)
            s.t = tnn.Linear(td, cout)
            s.n2, s.c2 = tnn.GroupNorm(8, cout), tnn.Conv2d(cout, cout, 3, padding=1)
            s.sk = tnn.Conv2d(cin, cout, 1) if cin != cout else tnn.Identity()

        def forward(s, x, te):
            h = s.c1(F.silu(s.n1(x))) + s.t(te)[:, :, None, None]
            return s.c2(F.silu(s.n2(h))) + s.sk(x)

    class At(tnn.Module):
        def __init__(s, c, heads=4):
            super().__init__()
            s.h, s.n, s.qkv, s.o = heads, tnn.GroupNorm(8, c), tnn.Linear(c, 3 * c), tnn.Linear(c, c)

        def forward(s, x):
            B, C, H, W = x.shape
            h = s.n(x).flatten(2).transpose(1, 2)
            q, k, v = s.qkv(h).chunk(3, -1)
            q, k, v = (a.reshape(B, H * W, s.h, C // s.h).transpose(1, 2) for a in (q, k, v))
            o = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, H * W, C)
            return x + s.o(o).transpose(1, 2).reshape(B, C, H, W)

    class U(tnn.Module):
        def __init__(s, ch, mults, attn_res, nres=2, img=28):
            super().__init__()
            td = 4 * ch
            s.ch = ch
            s.t1, s.t2, s.le = tnn.Linear(ch, td), tnn.Linear(td, td), tnn.Embedding(11, td)
            s.cin = tnn.Conv2d(1, ch, 3, padding=1)
            skips, cur, res = [ch], ch, img
            s.down = tnn.ModuleList()
            s.kinds = []
            for i, m in enumerate(mults):
                for _ in range(nres):
                    s.down.append(Res(cur, ch * m, td)); cur = ch * m; s.kinds.append("r")
                    if res in attn_res:
                        s.down.append(At(cur)); s.kinds.append("a")
                    skips.append(cur)
                if i != len(mults) - 1:
                    s.down.append(tnn.Conv2d(cur, cur, 3, 2, 1)); s.kinds.append("d"); res //= 2; skips.append(cur)
            s.m1, s.ma, s.m2 = Res(cur, cur, td), At(cur), Res(cur, cur, td)
            s.up = tnn.ModuleList(); s.ukinds = []
            for i, m in reversed(list(enumerate(mults))):
                for _ in range(nres + 1):
                    s.up.append(Res(cur + skips.pop(), ch * m, td)); cur = ch * m; s.ukinds.append("r")
                    if res in attn_res:
                        s.up.append(At(cur)); s.ukinds.append("a")
                if i != 0:
                    s.up.append(tnn.Conv2d(cur, cur, 3, padding=1)); s.ukinds.append("u"); res *= 2
            s.no, s.co = tnn.GroupNorm(8, cur), tnn.Conv2d(cur, 1, 3, padding=1)

        def forward(s, x, t, y):
            te = F.silu(s.t2(F.silu(s.t1(temb_fn(t, s.ch)))) + s.le(y))
            h = s.cin(x); hs = [h]
            for k, mod in zip(s.kinds, s.down):
                if k == "r":
                    h = mod(h, te); hs.append(h)
                elif k == "a":
                    h = mod(h); hs[-1] = h
                else:
                    h = mod(h); hs.append(h)
            h = s.m2(s.ma(s.m1(h, te)), te)
            for k, mod in zip(s.ukinds, s.up):
                if k == "r":
                    h = mod(torch.cat([h, hs.pop()], 1), te)
                elif k == "a":
                    h = mod(h)
                else:
                    h = mod(F.interpolate(h, scale_factor=2, mode="nearest"))
            return s.co(F.silu(s.no(h)))

    model = U(cfg["ch"], cfg["mults"], cfg["attn_res"]).to(dev)
    nparams = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)
    x = torch.randn(BATCH, 1, 28, 28, device=dev)
    t = torch.randint(0, 1000, (BATCH,), device=dev).float()
    y = torch.randint(0, 11, (BATCH,), device=dev)

    def step():
        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(x, t, y), x)
        loss.backward()
        opt.step()
        return loss

    for _ in range(5):
        step()
    torch.mps.synchronize()
    n = 30
    t0 = time.perf_counter()
    for _ in range(n):
        step()
    torch.mps.synchronize()
    train = (time.perf_counter() - t0) / n
    xi = torch.randn(INF_BATCH, 1, 28, 28, device=dev)
    ti = torch.full((INF_BATCH,), 500.0, device=dev)
    yi = torch.randint(0, 11, (INF_BATCH,), device=dev)
    with torch.no_grad():
        for _ in range(3):
            model(xi, ti, yi)
        torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            model(xi, ti, yi)
        torch.mps.synchronize()
    inf = (time.perf_counter() - t0) / 10
    return dict(params=nparams, train_step_s=train, train_img_per_s=BATCH / train, infer_img_per_s=INF_BATCH / inf)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    out = {}
    for cfg in CONFIGS:
        key = f"ch{cfg['ch']}_m{''.join(map(str, cfg['mults']))}"
        if which in ("both", "mlx"):
            out[f"mlx_{key}"] = bench_mlx(cfg)
            print("mlx", key, json.dumps({k: round(v, 4) for k, v in out[f'mlx_{key}'].items()}), flush=True)
        if which in ("both", "torch"):
            out[f"torch_{key}"] = bench_torch(cfg)
            print("torch", key, json.dumps({k: round(v, 4) for k, v in out[f'torch_{key}'].items()}), flush=True)
    os.makedirs(LOGS, exist_ok=True)
    with open(os.path.join(LOGS, f"bench_frameworks_{which}.json"), "w") as f:
        json.dump(dict(batch=BATCH, infer_batch=INF_BATCH, results=out), f, indent=1)
