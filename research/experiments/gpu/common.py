"""Shared helpers for the digit-model experiments (data, schedule, paths, image grids).

MNIST source: https://ossci-datasets.s3.amazonaws.com/mnist/ (the torchvision mirror of
http://yann.lecun.com/exdb/mnist/), MD5-verified. License: CC BY-SA 3.0 (Yann LeCun & Corinna Cortes).
"""
import gzip
import json
import os
import time

import numpy as np

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # research/experiments
DATA = os.path.join(EXP, "data", "mnist")
IMAGES = os.path.join(EXP, "images")
ARRAYS = os.path.join(EXP, "arrays")
LOGS = os.path.join(EXP, "logs")
WEIGHTS = os.path.join(EXP, "weights")
RESULTS = os.path.join(EXP, "results")  # per-stage json fragments, merged into mac_runs.json

MNIST_MD5 = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}

T = 1000
BETA_1, BETA_T = 1e-4, 0.02


def _read_idx(path):
    with gzip.open(path, "rb") as f:
        raw = f.read()
    magic = int.from_bytes(raw[:4], "big")
    ndim = magic & 0xFF
    dims = [int.from_bytes(raw[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


def load_mnist(split="train"):
    """Returns (images uint8 [N,28,28], labels int64 [N])."""
    p = "train" if split == "train" else "t10k"
    x = _read_idx(os.path.join(DATA, f"{p}-images-idx3-ubyte.gz"))
    y = _read_idx(os.path.join(DATA, f"{p}-labels-idx1-ubyte.gz")).astype(np.int64)
    return x, y


def to_model_space(x_uint8):
    """uint8 [N,28,28] -> float32 [N,28,28,1] in [-1, 1] (NHWC, MLX layout)."""
    return (x_uint8.astype(np.float32) / 127.5 - 1.0)[..., None]


def to_uint8(x):
    """float [-1,1] (any shape ending in H,W or H,W,1) -> uint8."""
    x = np.asarray(x, dtype=np.float32)
    if x.shape[-1] == 1:
        x = x[..., 0]
    return np.clip(np.round((x + 1.0) * 127.5), 0, 255).astype(np.uint8)


def ddpm_schedule():
    betas = np.linspace(BETA_1, BETA_T, T, dtype=np.float64)
    alphas = 1.0 - betas
    abar = np.cumprod(alphas)
    return betas, alphas, abar


def save_grid(path, imgs_uint8, nrow, pad=2, scale=1, pad_value=0):
    """imgs_uint8: [N,H,W] -> PNG grid with nrow columns."""
    from PIL import Image
    imgs = np.asarray(imgs_uint8)
    n, h, w = imgs.shape
    ncol = nrow
    nr = int(np.ceil(n / ncol))
    G = np.full((nr * (h + pad) + pad, ncol * (w + pad) + pad), pad_value, dtype=np.uint8)
    for i in range(n):
        r, c = divmod(i, ncol)
        G[pad + r * (h + pad): pad + r * (h + pad) + h, pad + c * (w + pad): pad + c * (w + pad) + w] = imgs[i]
    im = Image.fromarray(G, mode="L")
    if scale != 1:
        im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    im.save(path)


def write_result(name, obj):
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, name + ".json"), "w") as f:
        json.dump(obj, f, indent=1)


def read_result(name):
    p = os.path.join(RESULTS, name + ".json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.dt = time.perf_counter() - self.t0
