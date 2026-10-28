"""Train a class-conditional digit diffusion model on MNIST with MLX.

  python gpu/train.py --name ddpm --objective ddpm   # epsilon-prediction, linear beta 1e-4..0.02, T=1000
  python gpu/train.py --name flow --objective flow   # rectified flow / velocity target v = eps - x0
  python gpu/train.py --name dit  --objective ddpm --arch dit

Resumable: a checkpoint (model + EMA + optimizer + step) is written every --ckpt_every steps to
weights/digit_<name>_ckpt.safetensors; re-running the same command resumes. Every random draw is a
function of (seed, step), so a resumed run is identical to an uninterrupted one.
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

from common import LOGS, WEIGHTS, ddpm_schedule, load_mnist, to_model_space  # noqa: E402
from models_mlx import NULL_LABEL, DiT, UNet, count_params  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--objective", choices=["ddpm", "flow"], required=True)
p.add_argument("--arch", choices=["unet", "dit"], default="unet")
p.add_argument("--steps", type=int, default=10000)
p.add_argument("--batch", type=int, default=256)
p.add_argument("--lr", type=float, default=5e-4)
p.add_argument("--warmup", type=int, default=500)
p.add_argument("--ema", type=float, default=0.999)
p.add_argument("--p_uncond", type=float, default=0.1)
p.add_argument("--clip", type=float, default=1.0)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--ckpt_every", type=int, default=1000)
p.add_argument("--log_every", type=int, default=50)
p.add_argument("--dit_patch", type=int, default=2)
p.add_argument("--dit_d", type=int, default=192)
p.add_argument("--dit_depth", type=int, default=6)
args = p.parse_args()

os.makedirs(LOGS, exist_ok=True)
os.makedirs(WEIGHTS, exist_ok=True)
ckpt_path = os.path.join(WEIGHTS, f"digit_{args.name}_ckpt.safetensors")
ckpt_meta = os.path.join(WEIGHTS, f"digit_{args.name}_ckpt.json")
log_path = os.path.join(LOGS, f"train_{args.name}.jsonl")

mx.random.seed(args.seed)  # parameter init
if args.arch == "unet":
    model = UNet(ch=32, mults=(1, 2, 4), nres=2, attn_res=(7,), heads=4)
else:
    model = DiT(patch=args.dit_patch, d=args.dit_d, depth=args.dit_depth, heads=4)
mx.eval(model.parameters())
nparams = count_params(model)

sched = optim.join_schedules(
    [optim.linear_schedule(1e-8, args.lr, args.warmup),
     optim.cosine_decay(args.lr, args.steps - args.warmup, end=args.lr * 0.1)], [args.warmup])
opt = optim.Adam(learning_rate=sched)
opt.init(model.trainable_parameters())
ema = tree_map(lambda a: mx.array(a), model.parameters())

start = 0
wall_before = 0.0
if os.path.exists(ckpt_path):
    w = mx.load(ckpt_path)
    model.update(tree_unflatten([(k[6:], v) for k, v in w.items() if k.startswith("model.")]))
    ema = tree_unflatten([(k[4:], v) for k, v in w.items() if k.startswith("ema.")])
    opt.state = tree_unflatten([(k[4:], v) for k, v in w.items() if k.startswith("opt.")])
    meta = json.load(open(ckpt_meta))
    start, wall_before = meta["step"], meta["wall_s"]
    print(f"resumed from step {start}", flush=True)
elif os.path.exists(log_path):
    os.remove(log_path)

xtr, ytr = load_mnist("train")
X = mx.array(to_model_space(xtr))
Y = mx.array(ytr.astype(np.int32))
N = X.shape[0]
_, _, abar_np = ddpm_schedule()
ABAR = mx.array(abar_np.astype(np.float32))


def loss_fn(m, x0, y, key):
    k1, k2, k3 = mx.random.split(key, 3)
    B = x0.shape[0]
    eps = mx.random.normal(x0.shape, key=k2)
    drop = mx.random.uniform(shape=(B,), key=k3) < args.p_uncond
    y = mx.where(drop, NULL_LABEL, y)
    if args.objective == "ddpm":
        t = mx.random.randint(0, 1000, (B,), key=k1)
        a = ABAR[t][:, None, None, None]
        xt = mx.sqrt(a) * x0 + mx.sqrt(1 - a) * eps
        target, tin = eps, t.astype(mx.float32)
    else:
        t = mx.random.uniform(shape=(B,), key=k1)
        tt = t[:, None, None, None]
        xt = (1 - tt) * x0 + tt * eps
        target, tin = eps - x0, t * 1000.0
    return mx.mean((m(xt, tin, y) - target) ** 2)


lg = nn.value_and_grad(model, loss_fn)
state = [model.state, opt.state]


def _step(x0, y, key):
    loss, g = lg(model, x0, y, key)
    g, gnorm = optim.clip_grad_norm(g, args.clip)
    opt.update(model, g)
    return loss, gnorm


step_fn = mx.compile(_step, inputs=state, outputs=state)


@mx.compile
def ema_update(e, p_, d):
    return tree_map(lambda a, b: d * a + (1 - d) * b, e, p_)


def batch_indices(step):
    per_epoch = N // args.batch
    epoch, i = divmod(step, per_epoch)
    perm = np.random.default_rng(args.seed * 100003 + epoch).permutation(N)
    return perm[i * args.batch:(i + 1) * args.batch]


def save_ckpt(step, wall):
    flat = {f"model.{k}": v for k, v in tree_flatten(model.parameters())}
    flat.update({f"ema.{k}": v for k, v in tree_flatten(ema)})
    flat.update({f"opt.{k}": v for k, v in tree_flatten(opt.state)})
    mx.save_safetensors(ckpt_path + ".tmp.safetensors", flat)
    os.replace(ckpt_path + ".tmp.safetensors", ckpt_path)
    json.dump(dict(step=step, wall_s=wall), open(ckpt_meta, "w"))


reset_peak = getattr(mx, "reset_peak_memory", None) or mx.metal.reset_peak_memory
get_peak = getattr(mx, "get_peak_memory", None) or mx.metal.get_peak_memory
reset_peak()
cfg = dict(vars(args), params=nparams, model=model.config, images_per_epoch=N,
           epochs=args.steps * args.batch / N, mlx=mx.__version__)
print(json.dumps(cfg), flush=True)
json.dump(cfg, open(os.path.join(LOGS, f"train_{args.name}_config.json"), "w"), indent=1)

logf = open(log_path, "a")
t0 = time.perf_counter()
acc, accn, gacc = 0.0, 0, 0.0
for step in range(start, args.steps):
    idx = mx.array(batch_indices(step))
    key = mx.random.key(args.seed * 1_000_003 + step)
    loss, gnorm = step_fn(X[idx], Y[idx], key)
    d = min(args.ema, (1 + step) / (10 + step))
    ema = ema_update(ema, model.parameters(), mx.array(d))
    mx.eval(loss, gnorm, ema, state)
    acc += loss.item(); gacc += gnorm.item(); accn += 1
    if (step + 1) % args.log_every == 0 or step + 1 == args.steps:
        wall = wall_before + time.perf_counter() - t0
        rec = dict(step=step + 1, loss=acc / accn, grad_norm=gacc / accn, lr=float(sched(mx.array(step))),
                   wall_s=round(wall, 2), peak_mem_gb=round(get_peak() / 1e9, 3))
        logf.write(json.dumps(rec) + "\n"); logf.flush()
        if (step + 1) % (args.log_every * 10) == 0:
            print(json.dumps(rec), flush=True)
        acc, accn, gacc = 0.0, 0, 0.0
    if (step + 1) % args.ckpt_every == 0 or step + 1 == args.steps:
        save_ckpt(step + 1, wall_before + time.perf_counter() - t0)

wall = wall_before + time.perf_counter() - t0
mx.save_safetensors(os.path.join(WEIGHTS, f"digit_{args.name}_ema.safetensors"), dict(tree_flatten(ema)))
json.dump(dict(config=model.config, params=nparams, objective=args.objective, steps=args.steps,
               batch=args.batch, lr=args.lr, ema=args.ema, p_uncond=args.p_uncond, seed=args.seed,
               wall_s=wall, peak_mem_gb=get_peak() / 1e9, mlx=mx.__version__),
          open(os.path.join(WEIGHTS, f"digit_{args.name}_ema.json"), "w"), indent=1)
print(f"done {args.name}: {args.steps} steps in {wall:.1f}s, peak {get_peak()/1e9:.2f} GB", flush=True)
