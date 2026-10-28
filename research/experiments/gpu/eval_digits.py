"""Evaluate the digit models.

  python gpu/eval_digits.py gen ddpm_ddpm1000_w1 ddpm_ddim50_w3 flow_euler10_w1 ...
      config = <model>_<sampler><steps>_w<guidance>; 10,000 samples (labels i % 10), batch 1000,
      sample i always starts from the same noise (seed 1234). -> arrays/digit_samples_<cfg>.npz,
      images/digit_grid_<cfg>.png, results/gen_<cfg>.json (timing, peak memory)
  python gpu/eval_digits.py metrics      -> results/metrics.json (+ real-data reference rows)
  python gpu/eval_digits.py extras       -> trajectories, same-noise grids, nearest neighbours, loss vs t,
                                            U-Net activations, forward process, latency at small batches
Each output is skipped if it already exists (safe to re-run after a crash).
"""
import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402

import metrics as M  # noqa: E402
from common import (ARRAYS, IMAGES, RESULTS, T, ddpm_schedule, load_mnist, read_result,  # noqa: E402
                    save_grid, to_model_space, to_uint8, write_result)
from models_mlx import upsample2  # noqa: E402
from sampling import Denoiser, initial_noise, load_model, nfe_per_sample, sample  # noqa: E402

N_SAMPLES, BATCH = 10000, 1000
reset_peak = getattr(mx, "reset_peak_memory", None) or mx.metal.reset_peak_memory
get_peak = getattr(mx, "get_peak_memory", None) or mx.metal.get_peak_memory
_models = {}


def get_den(name):
    if name not in _models:
        m, meta = load_model(name)
        _models[name] = (Denoiser(m), m, meta)
    return _models[name]


def parse(cfg):
    m = re.fullmatch(r"(\w+?)_(ddpm|ddim|euler)(\d+)_w([\d.]+)", cfg)
    return m.group(1), m.group(2), int(m.group(3)), float(m.group(4))


def gen(cfg):
    out_npz = os.path.join(ARRAYS, f"digit_samples_{cfg}.npz")
    if os.path.exists(out_npz) and read_result(f"gen_{cfg}"):
        print("skip", cfg); return
    model, sampler, steps, w = parse(cfg)
    den = get_den(model)[0]
    xT_all = initial_noise(N_SAMPLES)
    y_all = (np.arange(N_SAMPLES) % 10).astype(np.int32)
    outs, times = [], []
    reset_peak()
    for b0 in range(0, N_SAMPLES, BATCH):
        t0 = time.perf_counter()
        x0, _ = sample(den, sampler, steps, mx.array(xT_all[b0:b0 + BATCH]), mx.array(y_all[b0:b0 + BATCH]), w,
                       seed=0, batch_start=b0)
        mx.eval(x0)
        times.append(time.perf_counter() - t0)
        outs.append(to_uint8(np.array(x0)))
        print(f"{cfg} batch {b0 // BATCH + 1}/{N_SAMPLES // BATCH}: {times[-1]:.2f}s", flush=True)
    S = np.concatenate(outs)
    np.savez_compressed(out_npz, samples=S, labels=y_all.astype(np.int8),
                        note=f"{cfg}: uint8 28x28, label = index % 10, initial noise seed 1234")
    save_grid(os.path.join(IMAGES, f"digit_grid_{cfg}.png"), S[:100], nrow=10, scale=2)
    steady = times[1:] if len(times) > 1 else times
    write_result(f"gen_{cfg}", dict(cfg=cfg, model=model, sampler=sampler, steps=steps, w=w,
                                    nfe_per_sample=nfe_per_sample(sampler, steps, w), n=N_SAMPLES, batch=BATCH,
                                    batch_times_s=[round(t, 3) for t in times],
                                    s_per_1000=float(np.mean(steady)) * 1000 / BATCH,
                                    first_batch_s=times[0], total_s=float(sum(times)),
                                    ms_per_sample=float(np.mean(steady)) / BATCH * 1000,
                                    peak_mem_gb=get_peak() / 1e9))


# ----------------------------------------------------------------------------- metrics
def compute_metrics():
    import classifier as C
    clf = C.load()
    xte, yte = load_mnist("test")
    xtr, ytr = load_mnist("train")
    cache = read_result("metrics") or {}
    f_te, l_te = C.feats_and_logits(clf, to_model_space(xte))
    rows = dict(cache)
    if "real_train10k_vs_test" not in rows:
        idx = np.random.default_rng(0).choice(60000, 10000, replace=False)
        f_tr, l_tr = C.feats_and_logits(clf, to_model_space(xtr[idx]))
        p, r = M.precision_recall(f_te, f_tr)
        div, per = M.diversity(f_tr, ytr[idx], f_te, yte)
        rows["real_train10k_vs_test"] = dict(cfid=M.frechet(f_tr, f_te), cond_acc=float((l_tr.argmax(1) == ytr[idx]).mean()),
                                             precision=p, recall=r, diversity=div, diversity_per_class=per,
                                             note="10,000 random training images vs the 10,000 test images: the floor")
        print("real", rows["real_train10k_vs_test"]["cfid"], flush=True)
    for fn in sorted(os.listdir(ARRAYS)):
        m = re.fullmatch(r"digit_samples_(.+)\.npz", fn)
        if not m or m.group(1) in rows:
            continue
        cfg = m.group(1)
        d = np.load(os.path.join(ARRAYS, fn))
        S, y = d["samples"], d["labels"].astype(np.int64)
        f, lo = C.feats_and_logits(clf, to_model_space(S))
        pred = lo.argmax(1)
        prob = np.exp(lo - lo.max(1, keepdims=True)); prob /= prob.sum(1, keepdims=True)
        p, r = M.precision_recall(f_te, f)
        div, per = M.diversity(f, y, f_te, yte)
        hist = np.bincount(pred, minlength=10) / len(pred)
        rows[cfg] = dict(cfid=M.frechet(f, f_te), cond_acc=float((pred == y).mean()), precision=p, recall=r,
                         diversity=div, diversity_per_class=per,
                         cond_acc_per_class=[float((pred[y == c] == c).mean()) for c in range(10)],
                         mean_confidence=float(prob.max(1).mean()),
                         predicted_label_hist=[float(v) for v in hist],
                         predicted_label_entropy_bits=float(-(hist[hist > 0] * np.log2(hist[hist > 0])).sum()),
                         mean_ink=float((S > 127).mean()), n=int(len(S)))
        print(cfg, {k: round(v, 4) for k, v in rows[cfg].items() if isinstance(v, float)}, flush=True)
        write_result("metrics", rows)
    write_result("metrics", rows)


# ----------------------------------------------------------------------------- extras
def same_noise_grids():
    out = os.path.join(ARRAYS, "digit_same_noise_ten_labels.npz")
    if os.path.exists(out):
        return
    seeds = list(range(8))
    xT = np.stack([np.random.default_rng(100 + s).standard_normal((28, 28, 1), dtype=np.float32) for s in seeds])
    res = dict(seeds=np.array(seeds), xT=xT[..., 0].astype(np.float16), labels=np.arange(10),
               note="row = seed (initial noise from np.random.default_rng(100+seed)), column = label 0..9; "
                    "the same x_T (and, for DDPM, the same per-step noise) is used for all ten labels")
    for cfg in ["ddpm_ddim50_w1", "ddpm_ddim50_w3", "ddpm_ddpm1000_w1", "flow_euler50_w1", "flow_euler50_w3"]:
        model, sampler, steps, w = parse(cfg)
        den = get_den(model)[0]
        grid = []
        for s in seeds:
            x = mx.array(np.repeat(xT[s:s + 1], 10, axis=0))
            x0, _ = sample(den, sampler, steps, x, mx.arange(10).astype(mx.int32), w, seed=1000 + s,
                           share_noise=True)
            grid.append(to_uint8(np.array(x0)))
        grid = np.stack(grid)
        res[cfg] = grid
        save_grid(os.path.join(IMAGES, f"digit_same_noise_ten_labels_{cfg}.png"), grid.reshape(-1, 28, 28), nrow=10,
                  scale=3)
        print("same-noise", cfg, flush=True)
    np.savez_compressed(out, **res)


def trajectories():
    out = os.path.join(ARRAYS, "digit_trajectory_7.npz")
    if os.path.exists(out):
        return
    xT = np.random.default_rng(7).standard_normal((1, 28, 28, 1), dtype=np.float32)
    y = mx.array(np.array([7], dtype=np.int32))
    res = dict(xT=xT[0, ..., 0].astype(np.float16), label=7,
               note="one '7' from the same x_T (np.random.default_rng(7)); x_t[i] is the input to step i, "
                    "x0_hat[i] the model's clean-image guess at that step; final = output; float16, [-1,1]")
    summary = {}
    for cfg in ["ddpm_ddpm1000_w1", "ddpm_ddim50_w1", "ddpm_ddim10_w1", "ddpm_ddim50_w3", "flow_euler50_w1",
                "flow_euler10_w1"]:
        model, sampler, steps, w = parse(cfg)
        den = get_den(model)[0]
        x0, tr = sample(den, sampler, steps, mx.array(xT), y, w, seed=7, record=set(range(steps)))
        res[f"{cfg}_t"] = np.array(tr["t"], dtype=np.float32)
        res[f"{cfg}_x_t"] = np.stack(tr["x_t"])[:, 0, ..., 0].astype(np.float16)
        res[f"{cfg}_x0_hat"] = np.clip(np.stack(tr["x0_hat"])[:, 0, ..., 0], -3, 3).astype(np.float16)
        res[f"{cfg}_final"] = np.array(x0)[0, ..., 0].astype(np.float16)
        k = np.round(np.linspace(0, steps - 1, min(steps, 40))).astype(int)
        res[f"{cfg}_frames40_index"] = k
        n_png = min(steps, 20)
        kk = np.round(np.linspace(0, steps - 1, n_png)).astype(int)
        strip = np.concatenate([to_uint8(res[f"{cfg}_x_t"][kk].astype(np.float32)),
                                to_uint8(res[f"{cfg}_x0_hat"][kk].astype(np.float32)),
                                ], axis=0)
        save_grid(os.path.join(IMAGES, f"digit_trajectory_7_{cfg}.png"), strip, nrow=n_png, scale=3)
        summary[cfg] = dict(steps=steps, frames=len(tr["t"]))
        print("trajectory", cfg, flush=True)
    # many digits emerging together (DDPM 1000 steps, 20 samples, 40 evenly spaced frames)
    den = get_den("ddpm")[0]
    xT20 = np.random.default_rng(20).standard_normal((20, 28, 28, 1), dtype=np.float32)
    rec = set(np.round(np.linspace(0, T - 1, 40)).astype(int).tolist())
    x0, tr = sample(den, "ddpm", T, mx.array(xT20), mx.array((np.arange(20) % 10).astype(np.int32)), 1.0, seed=20,
                    record=rec)
    res["many20_ddpm1000_w1_t"] = np.array(tr["t"], dtype=np.float32)
    res["many20_ddpm1000_w1_x_t"] = np.stack(tr["x_t"])[..., 0].astype(np.float16)  # [40,20,28,28]
    res["many20_ddpm1000_w1_x0_hat"] = np.clip(np.stack(tr["x0_hat"])[..., 0], -3, 3).astype(np.float16)
    res["many20_ddpm1000_w1_final"] = np.array(x0)[..., 0].astype(np.float16)
    res["many20_labels"] = np.arange(20) % 10
    np.savez_compressed(out, **res)


def nearest_neighbour_study():
    if read_result("nearest_neighbours"):
        return
    xtr, ytr = load_mnist("train")
    xte, _ = load_mnist("test")
    d_te, _ = M.nearest_neighbours(xte, xtr)
    rng = np.random.default_rng(0)
    rand_pairs = np.sqrt((((xte[:5000].reshape(5000, -1) / 255.0) -
                           (xtr[rng.choice(60000, 5000)].reshape(5000, -1) / 255.0)) ** 2).sum(1))
    out = dict(test_to_train=dict(median=float(np.median(d_te)), p1=float(np.percentile(d_te, 1)),
                                  p5=float(np.percentile(d_te, 5)), min=float(d_te.min())),
               random_pair_median=float(np.median(rand_pairs)),
               units="L2 distance in pixel space, pixels scaled to [0,1], 784 dims")
    arr = dict(test_to_train_dist=d_te.astype(np.float32))
    for cfg in ["ddpm_ddim50_w1", "ddpm_ddpm1000_w1", "ddpm_ddim50_w8", "flow_euler50_w1"]:
        p = os.path.join(ARRAYS, f"digit_samples_{cfg}.npz")
        if not os.path.exists(p):
            continue
        S = np.load(p)["samples"]
        d, ix = M.nearest_neighbours(S, xtr)
        thr = np.percentile(d_te, 1)
        out[cfg] = dict(median=float(np.median(d)), p1=float(np.percentile(d, 1)), p5=float(np.percentile(d, 5)),
                        min=float(d.min()), frac_below_test_p1=float((d < thr).mean()),
                        frac_exact_copies_lt_1=float((d < 1.0).mean()))
        arr[f"{cfg}_dist"] = d.astype(np.float32)
        arr[f"{cfg}_nn_index"] = ix.astype(np.int32)
        if cfg == "ddpm_ddim50_w1":
            pick = list(rng.choice(len(S), 12, replace=False)) + list(np.argsort(d)[:4])
            arr["examples_index"] = np.array(pick)
            arr["examples_generated"] = S[pick]
            arr["examples_nearest_train"] = xtr[ix[pick]]
            arr["examples_dist"] = d[pick].astype(np.float32)
            pairs = np.concatenate([S[pick], xtr[ix[pick]]])
            save_grid(os.path.join(IMAGES, "digit_nearest_train_neighbours_ddim50_w1.png"), pairs, nrow=16, scale=3)
        print("nn", cfg, out[cfg], flush=True)
    out["examples_note"] = ("examples: 12 random generated digits (ddpm_ddim50_w1) + the 4 generated digits closest "
                            "to any training image; PNG row 1 = generated, row 2 = nearest training image")
    np.savez_compressed(os.path.join(ARRAYS, "digit_nearest_neighbours.npz"), **arr)
    write_result("nearest_neighbours", out)


def loss_vs_t():
    if read_result("loss_vs_t"):
        return
    xte, yte = load_mnist("test")
    X = mx.array(to_model_space(xte))
    Y = mx.array(yte.astype(np.int32))
    _, _, abar = ddpm_schedule()
    out = {}
    for name in ["ddpm", "flow", "dit"]:
        try:
            _, m, meta = get_den(name)
        except FileNotFoundError:
            continue
        rows = []
        grid = np.linspace(0, 1, 21)[1:-1] if meta["objective"] == "flow" else np.array([0, 10, 25, 50, 100, 200, 300,
                                                                                       400, 500, 600, 700, 800, 900, 999])
        for tv in grid:
            tv = float(tv)  # a numpy float64 times an mx.array would silently become a numpy array
            key = mx.random.key(int(tv * 1000) if meta["objective"] == "flow" else int(tv))
            eps = mx.random.normal(X.shape, key=key)
            tot = 0.0
            for i in range(0, 10000, 2000):
                x0, e, y = X[i:i + 2000], eps[i:i + 2000], Y[i:i + 2000]
                if meta["objective"] == "flow":
                    xt, target, tin = (1 - tv) * x0 + tv * e, e - x0, tv * 1000
                else:
                    a = float(abar[int(tv)])
                    xt, target, tin = a ** 0.5 * x0 + (1 - a) ** 0.5 * e, e, float(tv)
                pred = m(xt, mx.full((x0.shape[0],), float(tin)), y)
                tot += float(mx.mean((pred - target) ** 2).item()) * x0.shape[0]
            rows.append(dict(t=float(tv), mse=tot / 10000))
        out[name] = dict(objective=meta["objective"], rows=rows,
                         note="test-set MSE of the EMA model's prediction (eps for ddpm/dit, v for flow) at fixed t")
        print("loss_vs_t", name, flush=True)
    write_result("loss_vs_t", out)


def unet_activations():
    out = os.path.join(ARRAYS, "digit_unet_activations.npz")
    if os.path.exists(out):
        return
    _, m, _ = get_den("ddpm")
    xtr, ytr = load_mnist("train")
    i7 = int(np.where(ytr == 7)[0][0])
    x0 = mx.array(to_model_space(xtr[i7:i7 + 1]))
    _, _, abar = ddpm_schedule()
    t = 500
    eps = mx.array(np.random.default_rng(5).standard_normal((1, 28, 28, 1), dtype=np.float32))
    a = float(abar[t])
    x = a ** 0.5 * x0 + (1 - a) ** 0.5 * eps
    tt, y = mx.array([float(t)]), mx.array([7], dtype=mx.int32)
    temb = m.embed(tt, y)
    acts, names = [], []

    def rec(name, h):
        names.append(name); acts.append(np.array(h[0]))

    h = m.conv_in(x); rec("conv_in", h); hs = [h]
    for li, lvl in enumerate(m.down):
        for bi, (blk, at) in enumerate(zip(lvl.blocks, lvl.attns)):
            h = at(blk(h, temb)); hs.append(h); rec(f"down{li}.block{bi}", h)
        if lvl.has_resample:
            h = lvl.resample(h); hs.append(h); rec(f"down{li}.downsample", h)
    h = m.mid2(m.mid_attn(m.mid1(h, temb)), temb); rec("mid", h)
    for li, lvl in enumerate(m.up):
        for bi, (blk, at) in enumerate(zip(lvl.blocks, lvl.attns)):
            h = at(blk(mx.concatenate([h, hs.pop()], axis=-1), temb)); rec(f"up{li}.block{bi}", h)
        if lvl.has_resample:
            h = lvl.resample(upsample2(h)); rec(f"up{li}.upsample", h)
    e_hat = m.conv_out(nn.silu(m.norm_out(h)))
    res = dict(names=np.array(names), x0=np.array(x0)[0, ..., 0], x_t=np.array(x)[0, ..., 0], t=t,
               eps_true=np.array(eps)[0, ..., 0], eps_hat=np.array(e_hat)[0, ..., 0],
               note="a training '7' noised to t=500, passed through the DDPM U-Net; per layer: output shape, "
                    "the first 8 channels and the mean |activation| map")
    for n_, a_ in zip(names, acts):
        res[f"{n_}__shape"] = np.array(a_.shape)
        res[f"{n_}__ch8"] = a_[..., :8].transpose(2, 0, 1).astype(np.float16)
        res[f"{n_}__meanabs"] = np.abs(a_).mean(-1).astype(np.float16)
    np.savez_compressed(out, **res)
    print("activations", [(n_, a_.shape) for n_, a_ in zip(names, acts)], flush=True)


def forward_process():
    out = os.path.join(ARRAYS, "digit_forward_process.npz")
    if os.path.exists(out):
        return
    xtr, ytr = load_mnist("train")
    i7 = int(np.where(ytr == 7)[0][0])
    x0 = to_model_space(xtr[i7:i7 + 1])[0, ..., 0]
    betas, alphas, abar = ddpm_schedule()
    eps = np.random.default_rng(4).standard_normal((28, 28)).astype(np.float32)
    ts = np.array([0, 25, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 999])
    xt = np.stack([np.sqrt(abar[t]) * x0 + np.sqrt(1 - abar[t]) * eps for t in ts]).astype(np.float32)
    np.savez_compressed(out, train_index=i7, x0=x0, eps=eps, t=ts, x_t=xt.astype(np.float16), betas=betas, abar=abar,
                        snr=abar / (1 - abar),
                        note="x_t = sqrt(abar_t) x0 + sqrt(1-abar_t) eps, linear beta 1e-4..0.02, T=1000, t is 0-based")
    save_grid(os.path.join(IMAGES, "digit_forward_process_7.png"), to_uint8(np.clip(xt, -1, 1)), nrow=len(ts), scale=3)


def latency():
    if read_result("latency"):
        return
    rows = []
    for cfg in ["ddpm_ddpm1000_w1", "ddpm_ddim50_w1", "ddpm_ddim10_w1", "ddpm_ddim50_w3", "flow_euler50_w1",
                "flow_euler10_w1", "flow_euler1_w1"]:
        model, sampler, steps, w = parse(cfg)
        den = get_den(model)[0]
        for B in [1, 10, 100]:
            xT = mx.array(initial_noise(B))
            y = mx.array((np.arange(B) % 10).astype(np.int32))
            sample(den, sampler, min(steps, 5) if sampler != "ddpm" else steps, xT, y, w)  # warm-up/compile
            ts = []
            for _ in range(3 if steps < 1000 else 1):
                t0 = time.perf_counter()
                x0, _ = sample(den, sampler, steps, xT, y, w)
                mx.eval(x0)
                ts.append(time.perf_counter() - t0)
            rows.append(dict(cfg=cfg, batch=B, s=float(np.median(ts)), ms_per_step=float(np.median(ts)) / steps * 1000))
            print(rows[-1], flush=True)
    write_result("latency", dict(rows=rows, note="median wall time to generate one batch (excludes compile)"))


def architecture():
    if read_result("architecture"):
        return
    out = {}
    for name in ["ddpm", "flow", "dit"]:
        try:
            _, m, meta = get_den(name)
        except FileNotFoundError:
            continue
        groups = {}
        for k, v in tree_flatten(m.parameters()):
            top = ".".join(k.split(".")[:2]) if k.startswith(("down", "up", "blocks")) else k.split(".")[0]
            groups[top] = groups.get(top, 0) + v.size
        out[name] = dict(config=meta["config"], params=int(sum(groups.values())), params_by_part=groups)
    write_result("architecture", out)


def trajectories_more():
    """Alternative '7' trajectories (other seeds) so scenes can pick the clearest digit."""
    out = os.path.join(ARRAYS, "digit_trajectory_7_more_seeds.npz")
    if os.path.exists(out):
        return
    seeds = [17, 27, 37, 47, 57, 67]
    res = dict(seeds=np.array(seeds), label=7,
               note="like digit_trajectory_7.npz but x_T from np.random.default_rng(seed) for each seed; "
                    "all steps recorded; float16 in [-1,1] (x0_hat clipped to [-3,3])")
    finals = []
    for cfg in ["ddpm_ddpm1000_w1", "ddpm_ddim50_w1", "flow_euler50_w1"]:
        model, sampler, steps, w = parse(cfg)
        den = get_den(model)[0]
        for sd in seeds:
            xT = np.random.default_rng(sd).standard_normal((1, 28, 28, 1), dtype=np.float32)
            x0, tr = sample(den, sampler, steps, mx.array(xT), mx.array(np.array([7], dtype=np.int32)), w, seed=sd,
                            record=set(range(steps)))
            res[f"seed{sd}_{cfg}_x_t"] = np.stack(tr["x_t"])[:, 0, ..., 0].astype(np.float16)
            res[f"seed{sd}_{cfg}_x0_hat"] = np.clip(np.stack(tr["x0_hat"])[:, 0, ..., 0], -3, 3).astype(np.float16)
            res[f"seed{sd}_{cfg}_final"] = np.array(x0)[0, ..., 0].astype(np.float16)
            res[f"{cfg}_t"] = np.array(tr["t"], dtype=np.float32)
            finals.append(to_uint8(np.array(x0)[0]))
    save_grid(os.path.join(IMAGES, "digit_trajectory_7_more_seeds_finals.png"), np.stack(finals), nrow=len(seeds),
              scale=3)
    np.savez_compressed(out, **res)
    print("trajectories_more done", flush=True)


def cfid_uncertainty():
    """Bootstrap spread of cFID (resampling generated features) and spread of the real-data floor."""
    if read_result("cfid_uncertainty"):
        return
    import classifier as C
    clf = C.load()
    xte, _ = load_mnist("test")
    xtr, _ = load_mnist("train")
    f_te, _ = C.feats_and_logits(clf, to_model_space(xte))
    out = {}
    floors = []
    for sd in range(5):
        idx = np.random.default_rng(sd).choice(60000, 10000, replace=False)
        f_tr, _ = C.feats_and_logits(clf, to_model_space(xtr[idx]))
        floors.append(M.frechet(f_tr, f_te))
    out["real_floor_5_subsets"] = dict(values=floors, mean=float(np.mean(floors)), std=float(np.std(floors)))
    rng = np.random.default_rng(0)
    for cfg in ["ddpm_ddpm1000_w1", "ddpm_ddim50_w1", "ddpm_ddim20_w1", "ddpm_ddim100_w1", "flow_euler50_w1"]:
        S = np.load(os.path.join(ARRAYS, f"digit_samples_{cfg}.npz"))["samples"]
        f, _ = C.feats_and_logits(clf, to_model_space(S))
        bs = [M.frechet(f[rng.integers(0, len(f), len(f))], f_te) for _ in range(20)]
        out[cfg] = dict(bootstrap_mean=float(np.mean(bs)), bootstrap_std=float(np.std(bs)))
        print("cfid_uncertainty", cfg, out[cfg], flush=True)
    out["note"] = ("bootstrap: 20 resamples (with replacement) of the 10,000 generated feature vectors vs the fixed test "
                   "set; resampling inflates cFID slightly, so use the std as the error bar, not the mean")
    write_result("cfid_uncertainty", out)


if __name__ == "__main__":
    os.makedirs(ARRAYS, exist_ok=True); os.makedirs(IMAGES, exist_ok=True); os.makedirs(RESULTS, exist_ok=True)
    cmd = sys.argv[1]
    if cmd == "gen":
        for c in sys.argv[2:]:
            gen(c)
    elif cmd == "metrics":
        compute_metrics()
    elif cmd == "extras":
        which = sys.argv[2:] or ["architecture", "forward_process", "unet_activations", "loss_vs_t", "same_noise_grids",
                                 "trajectories", "nearest_neighbour_study", "latency"]
        for wname in which:
            globals()[wname]()
