"""Denoisers for the digit model, in MLX (NHWC).

UNet: DDPM-style class-conditional U-Net (ResBlocks + GroupNorm + SiLU, sinusoidal time embedding -> MLP,
label embedding added to the time embedding, self-attention at the 7x7 level and in the middle).
DiT: a tiny Diffusion Transformer (patchify, adaLN-Zero blocks, linear unpatchify).
Both take (x [B,28,28,1], t [B] float in [0, 1000), y [B] int with y == 10 meaning "no label").
"""
import math

import mlx.core as mx
import mlx.nn as nn

NULL_LABEL = 10


def timestep_embedding(t, dim, max_period=10000.0):
    half = dim // 2
    freqs = mx.exp(-math.log(max_period) * mx.arange(half, dtype=mx.float32) / half)
    args = t.astype(mx.float32)[:, None] * freqs[None, :]
    return mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)


def upsample2(x):
    B, H, W, C = x.shape
    x = mx.broadcast_to(x[:, :, None, :, None, :], (B, H, 2, W, 2, C))
    return x.reshape(B, 2 * H, 2 * W, C)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, tdim, groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, cin, pytorch_compatible=True)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.temb = nn.Linear(tdim, cout)
        self.norm2 = nn.GroupNorm(groups, cout, pytorch_compatible=True)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.conv2.weight = mx.zeros_like(self.conv2.weight)  # zero-init residual branch
        self.has_skip = cin != cout
        if self.has_skip:
            self.skip = nn.Conv2d(cin, cout, 1)

    def __call__(self, x, temb):
        h = self.conv1(nn.silu(self.norm1(x)))
        h = h + self.temb(temb)[:, None, None, :]
        h = self.conv2(nn.silu(self.norm2(h)))
        return h + (self.skip(x) if self.has_skip else x)


class Attn(nn.Module):
    def __init__(self, c, heads=4, groups=8):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(groups, c, pytorch_compatible=True)
        self.qkv = nn.Linear(c, 3 * c)
        self.out = nn.Linear(c, c)
        self.out.weight = mx.zeros_like(self.out.weight)

    def __call__(self, x):
        B, H, W, C = x.shape
        h = self.norm(x).reshape(B, H * W, C)
        q, k, v = mx.split(self.qkv(h), 3, axis=-1)
        d = C // self.heads
        q, k, v = (a.reshape(B, H * W, self.heads, d).transpose(0, 2, 1, 3) for a in (q, k, v))
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(d))
        o = o.transpose(0, 2, 1, 3).reshape(B, H * W, C)
        return x + self.out(o).reshape(B, H, W, C)


class Level(nn.Module):
    def __init__(self, blocks, attns, resample):
        super().__init__()
        self.blocks = blocks
        self.attns = attns  # list, same length as blocks; entries are Attn or nn.Identity
        self.has_resample = resample is not None
        if self.has_resample:
            self.resample = resample


class UNet(nn.Module):
    def __init__(self, ch=64, mults=(1, 2, 2), nres=2, attn_res=(7,), num_classes=10, in_ch=1, img=28, heads=4):
        super().__init__()
        self.ch = ch
        tdim = 4 * ch
        self.config = dict(arch="unet", ch=ch, mults=list(mults), nres=nres, attn_res=list(attn_res),
                           num_classes=num_classes, img=img, tdim=tdim, heads=heads)
        self.time1 = nn.Linear(ch, tdim)
        self.time2 = nn.Linear(tdim, tdim)
        self.label_emb = nn.Embedding(num_classes + 1, tdim)
        self.conv_in = nn.Conv2d(in_ch, ch, 3, padding=1)
        skips = [ch]
        cur, res = ch, img
        self.down = []
        for i, m in enumerate(mults):
            blocks, attns = [], []
            for _ in range(nres):
                blocks.append(ResBlock(cur, ch * m, tdim))
                cur = ch * m
                attns.append(Attn(cur, heads) if res in attn_res else nn.Identity())
                skips.append(cur)
            down = None
            if i != len(mults) - 1:
                down = nn.Conv2d(cur, cur, 3, stride=2, padding=1)
                res //= 2
                skips.append(cur)
            self.down.append(Level(blocks, attns, down))
        self.mid1 = ResBlock(cur, cur, tdim)
        self.mid_attn = Attn(cur, heads)
        self.mid2 = ResBlock(cur, cur, tdim)
        self.up = []
        for i, m in reversed(list(enumerate(mults))):
            blocks, attns = [], []
            for _ in range(nres + 1):
                blocks.append(ResBlock(cur + skips.pop(), ch * m, tdim))
                cur = ch * m
                attns.append(Attn(cur, heads) if res in attn_res else nn.Identity())
            up = None
            if i != 0:
                up = nn.Conv2d(cur, cur, 3, padding=1)
                res *= 2
            self.up.append(Level(blocks, attns, up))
        self.norm_out = nn.GroupNorm(8, cur, pytorch_compatible=True)
        self.conv_out = nn.Conv2d(cur, in_ch, 3, padding=1)
        self.conv_out.weight = mx.zeros_like(self.conv_out.weight)

    def embed(self, t, y):
        temb = self.time2(nn.silu(self.time1(timestep_embedding(t, self.ch))))
        return nn.silu(temb + self.label_emb(y))

    def __call__(self, x, t, y):
        temb = self.embed(t, y)
        h = self.conv_in(x)
        hs = [h]
        for lvl in self.down:
            for blk, at in zip(lvl.blocks, lvl.attns):
                h = at(blk(h, temb))
                hs.append(h)
            if lvl.has_resample:
                h = lvl.resample(h)
                hs.append(h)
        h = self.mid2(self.mid_attn(self.mid1(h, temb)), temb)
        for lvl in self.up:
            for blk, at in zip(lvl.blocks, lvl.attns):
                h = at(blk(mx.concatenate([h, hs.pop()], axis=-1), temb))
            if lvl.has_resample:
                h = lvl.resample(upsample2(h))
        return self.conv_out(nn.silu(self.norm_out(h)))


# ----------------------------------------------------------------------------- tiny DiT
class DiTBlock(nn.Module):
    def __init__(self, d, heads, cdim, mlp_ratio=4):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(d, affine=False, eps=1e-6)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d, affine=False, eps=1e-6)
        self.fc1 = nn.Linear(d, mlp_ratio * d)
        self.fc2 = nn.Linear(mlp_ratio * d, d)
        self.ada = nn.Linear(cdim, 6 * d)  # adaLN-Zero: shift/scale/gate for attn and mlp
        self.ada.weight = mx.zeros_like(self.ada.weight)
        self.ada.bias = mx.zeros_like(self.ada.bias)

    def __call__(self, x, c):
        B, L, D = x.shape
        s1, sc1, g1, s2, sc2, g2 = mx.split(self.ada(c)[:, None, :], 6, axis=-1)
        h = self.norm1(x) * (1 + sc1) + s1
        q, k, v = mx.split(self.qkv(h), 3, axis=-1)
        d = D // self.heads
        q, k, v = (a.reshape(B, L, self.heads, d).transpose(0, 2, 1, 3) for a in (q, k, v))
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(d))
        x = x + g1 * self.proj(o.transpose(0, 2, 1, 3).reshape(B, L, D))
        h = self.norm2(x) * (1 + sc2) + s2
        return x + g2 * self.fc2(nn.gelu_approx(self.fc1(h)))


class DiT(nn.Module):
    def __init__(self, patch=2, d=256, depth=6, heads=4, num_classes=10, in_ch=1, img=28):
        super().__init__()
        self.patch, self.d, self.img, self.in_ch = patch, d, img, in_ch
        self.config = dict(arch="dit", patch=patch, d=d, depth=depth, heads=heads, num_classes=num_classes,
                           img=img, tokens=(img // patch) ** 2)
        n = img // patch
        self.patch_embed = nn.Linear(patch * patch * in_ch, d)
        # fixed 2D sin-cos position embedding
        pe = self._pos_embed(d, n)
        self._pe = pe
        self.time1 = nn.Linear(d, d)
        self.time2 = nn.Linear(d, d)
        self.label_emb = nn.Embedding(num_classes + 1, d)
        self.blocks = [DiTBlock(d, heads, d) for _ in range(depth)]
        self.norm_out = nn.LayerNorm(d, affine=False, eps=1e-6)
        self.ada_out = nn.Linear(d, 2 * d)
        self.ada_out.weight = mx.zeros_like(self.ada_out.weight)
        self.ada_out.bias = mx.zeros_like(self.ada_out.bias)
        self.out = nn.Linear(d, patch * patch * in_ch)
        self.out.weight = mx.zeros_like(self.out.weight)
        self.out.bias = mx.zeros_like(self.out.bias)

    @staticmethod
    def _pos_embed(d, n):
        import numpy as np
        q = d // 4
        omega = 1.0 / 10000 ** (np.arange(q) / q)
        ys, xs = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        out = []
        for g in (ys.reshape(-1), xs.reshape(-1)):
            a = g[:, None] * omega[None]
            out += [np.sin(a), np.cos(a)]
        return mx.array(np.concatenate(out, axis=1).astype("float32"))

    def __call__(self, x, t, y):
        B, H, W, C = x.shape
        p, n = self.patch, H // self.patch
        tok = x.reshape(B, n, p, n, p, C).transpose(0, 1, 3, 2, 4, 5).reshape(B, n * n, p * p * C)
        h = self.patch_embed(tok) + self._pe[None]
        c = self.time2(nn.silu(self.time1(timestep_embedding(t, self.d)))) + self.label_emb(y)
        c = nn.silu(c)
        for blk in self.blocks:
            h = blk(h, c)
        shift, scale = mx.split(self.ada_out(c)[:, None, :], 2, axis=-1)
        h = self.out(self.norm_out(h) * (1 + scale) + shift)
        return h.reshape(B, n, n, p, p, C).transpose(0, 1, 3, 2, 4, 5).reshape(B, H, W, C)


def build(config):
    cfg = dict(config)
    arch = cfg.pop("arch")
    if arch == "unet":
        cfg.pop("tdim", None)
        return UNet(**cfg)
    cfg.pop("tokens", None)
    return DiT(**cfg)


def count_params(model):
    from mlx.utils import tree_flatten
    return sum(v.size for k, v in tree_flatten(model.parameters()) if not k.startswith("_"))
