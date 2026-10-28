"""Small MNIST CNN classifier used only as a *measuring instrument* for the digit model:
its 128-d penultimate features give a "classifier FID" (NOT Inception FID), k-NN precision/recall
and diversity; its predictions give conditional accuracy.

  python gpu/classifier.py   -> weights/digit_classifier.safetensors, results/classifier.json
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import mlx.optimizers as optim  # noqa: E402
from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402

from common import WEIGHTS, load_mnist, to_model_space, write_result  # noqa: E402

PATH = os.path.join(WEIGHTS, "digit_classifier.safetensors")


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 32, 3, padding=1)
        self.c3 = nn.Conv2d(32, 64, 3, padding=1)
        self.c4 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(7 * 7 * 64, 128)
        self.fc2 = nn.Linear(128, 10)
        self.drop = nn.Dropout(0.3)

    def features(self, x):  # x: [B,28,28,1] in [-1,1]
        h = self.pool(nn.relu(self.c2(nn.relu(self.c1(x)))))
        h = self.pool(nn.relu(self.c4(nn.relu(self.c3(h)))))
        return nn.relu(self.fc1(h.reshape(h.shape[0], -1)))  # 128-d penultimate features

    def __call__(self, x):
        return self.fc2(self.drop(self.features(x)))


def load():
    m = Classifier()
    m.update(tree_unflatten(list(mx.load(PATH).items())))
    m.eval()
    mx.eval(m.parameters())
    return m


def feats_and_logits(m, x_np, bs=2000):
    """x_np: float32 [N,28,28,1] in [-1,1] -> (features [N,128], logits [N,10]) as numpy."""
    F, L = [], []
    for i in range(0, len(x_np), bs):
        x = mx.array(x_np[i:i + bs])
        f = m.features(x)
        lo = m.fc2(f)
        mx.eval(f, lo)
        F.append(np.array(f)); L.append(np.array(lo))
    return np.concatenate(F), np.concatenate(L)


if __name__ == "__main__":
    mx.random.seed(0)
    xtr, ytr = load_mnist("train")
    xte, yte = load_mnist("test")
    Xtr, Ytr = mx.array(to_model_space(xtr)), mx.array(ytr.astype(np.int32))
    Xte = to_model_space(xte)
    m = Classifier()
    mx.eval(m.parameters())
    epochs, bs = 6, 128
    steps = epochs * (60000 // bs)
    opt = optim.Adam(learning_rate=optim.cosine_decay(1e-3, steps, end=1e-5))

    def loss_fn(model, x, y):
        return mx.mean(nn.losses.cross_entropy(model(x), y))

    lg = nn.value_and_grad(m, loss_fn)
    state = [m.state, opt.state, mx.random.state]

    @mx.compile
    def _step(x, y):
        loss, g = lg(m, x, y)
        opt.update(m, g)
        return loss

    step = mx.compile(_step, inputs=state, outputs=state)
    t0 = time.perf_counter()
    rng = np.random.default_rng(0)
    m.train()
    for ep in range(epochs):
        perm = rng.permutation(60000)
        for i in range(60000 // bs):
            idx = mx.array(perm[i * bs:(i + 1) * bs])
            # light augmentation: random shifts of up to 2 px (keeps the instrument robust to small offsets)
            x = Xtr[idx]
            dx, dy = (int(v) for v in rng.integers(-2, 3, 2))
            x = mx.roll(mx.roll(x, dx, axis=2), dy, axis=1)
            loss = step(x, Ytr[idx])
            mx.eval(loss, state)
        m.eval()
        _, lo = feats_and_logits(m, Xte)
        acc = float((lo.argmax(1) == yte).mean())
        m.train()
        print(f"epoch {ep + 1}: loss {loss.item():.4f} test acc {acc:.4f}", flush=True)
    wall = time.perf_counter() - t0
    m.eval()
    mx.save_safetensors(PATH, dict(tree_flatten(m.parameters())))
    _, lo = feats_and_logits(m, Xte)
    acc = float((lo.argmax(1) == yte).mean())
    nparams = sum(v.size for _, v in tree_flatten(m.parameters()))
    write_result("classifier", dict(test_accuracy=acc, test_errors=int((lo.argmax(1) != yte).sum()),
                                    params=nparams, epochs=epochs, batch=bs, wall_s=wall,
                                    arch="conv32-conv32-pool-conv64-conv64-pool-fc128(features)-fc10, dropout 0.3, "
                                         "random shifts <=2px", feature_dim=128))
    print(f"classifier test accuracy {acc:.4f} ({nparams} params, {wall:.0f}s)")
