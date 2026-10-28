"""Quality metrics for generated digits, all in the feature space of our own MNIST classifier.

classifier FID ("cFID"): Frechet distance between Gaussians fitted to the 128-d penultimate features of
    generated vs real test images. Same formula as FID, but NOT Inception features: numbers are only
    comparable with other cFIDs computed by this script.
conditional accuracy: classifier argmax == requested label.
precision / recall: k-NN manifold estimate (Kynkaenniemi et al. 2019), k = 3, in classifier-feature space.
diversity: per class, mean pairwise L2 distance between features of generated samples requested as that
    class, divided by the same quantity for real test images of that class (1.0 = as varied as real data).
"""
import mlx.core as mx
import numpy as np


def frechet(f1, f2):
    mu1, mu2 = f1.mean(0), f2.mean(0)
    s1, s2 = np.cov(f1, rowvar=False), np.cov(f2, rowvar=False)
    w, V = np.linalg.eigh(s1)
    s1h = (V * np.sqrt(np.clip(w, 0, None))) @ V.T
    m = s1h @ s2 @ s1h
    tr_covmean = np.sqrt(np.clip(np.linalg.eigvalsh((m + m.T) / 2), 0, None)).sum()
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * tr_covmean)


def _sqdist(a, b):
    """a [n,d], b [m,d] mx arrays -> squared distances [n,m] (mx)."""
    return mx.maximum((a * a).sum(1, keepdims=True) + (b * b).sum(1)[None] - 2 * a @ b.T, 0.0)


def kth_nn_radius(f, k=3, chunk=2000):
    F = mx.array(f.astype(np.float32))
    out = []
    for i in range(0, F.shape[0], chunk):
        d = _sqdist(F[i:i + chunk], F)
        out.append(np.sqrt(np.array(mx.sort(d, axis=1)[:, k])))  # index 0 is the point itself
    return np.concatenate(out)


def coverage_frac(query, ref, ref_radius, chunk=2000):
    Q, R = mx.array(query.astype(np.float32)), mx.array(ref.astype(np.float32))
    rr = mx.array((ref_radius ** 2).astype(np.float32))[None]
    hits = []
    for i in range(0, Q.shape[0], chunk):
        hits.append(np.array(mx.any(_sqdist(Q[i:i + chunk], R) <= rr, axis=1)))
    return float(np.concatenate(hits).mean())


def precision_recall(real, fake, k=3):
    rr, rf = kth_nn_radius(real, k), kth_nn_radius(fake, k)
    return coverage_frac(fake, real, rr), coverage_frac(real, fake, rf)


def mean_pairwise(f):
    F = mx.array(f.astype(np.float32))
    d = mx.sqrt(_sqdist(F, F))
    n = F.shape[0]
    return float(d.sum().item() / (n * (n - 1)))


def diversity(feat, labels, real_feat, real_labels):
    per = []
    for c in range(10):
        g, r = feat[labels == c], real_feat[real_labels == c]
        per.append(mean_pairwise(g) / mean_pairwise(r))
    return float(np.mean(per)), [float(v) for v in per]


def nearest_neighbours(query_u8, ref_u8, chunk=1000):
    """L2 distance in pixel space (pixels scaled to [0,1], 784 dims) to the closest reference image.
    Returns (dist [n], index [n])."""
    R = mx.array(ref_u8.reshape(len(ref_u8), -1).astype(np.float32) / 255.0)
    Q = mx.array(query_u8.reshape(len(query_u8), -1).astype(np.float32) / 255.0)
    ds, ix = [], []
    for i in range(0, Q.shape[0], chunk):
        d = _sqdist(Q[i:i + chunk], R)
        j = mx.argmin(d, axis=1)
        mx.eval(j)
        ds.append(np.sqrt(np.array(mx.take_along_axis(d, j[:, None], axis=1)[:, 0])))
        ix.append(np.array(j))
    return np.concatenate(ds), np.concatenate(ix)
