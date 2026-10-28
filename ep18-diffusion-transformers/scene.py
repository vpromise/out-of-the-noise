"""Out of the Noise · Episode 18 · Diffusion Transformers (Module 5 · Scaling Up)

Transformers on patches replace the U-Net at scale. A diffusion transformer (DiT) cuts the noisy latent into p × p
patches, flattens each into a token, embeds it linearly, adds a position code, runs a stack of transformer blocks
(attention + MLP, Episode 8's attention) and turns every token back into a patch of the noise guess ε̂. Time and label
enter through adaLN-Zero: they set a per-channel scale γ, shift β and gate α (initialised to zero) in every block,
instead of Episode 8's added per-channel bias. DiT's Gflops-vs-FID results show quality improving steadily with
compute; MM-DiT (Stable Diffusion 3) puts text and image tokens into one sequence with separate weights and joint
attention. Then the honest run on this Mac: a tiny DiT vs our U-Net, same data, steps, batch and optimiser.

Running example: the digit model (our U-Net vs a tiny DiT trained on this Mac); the mechanism is drawn on the
lighthouse's 64 × 64 × 4 latent from Episode 17 (linked, not re-taught); DiT's scaling results are cited.

Every number on screen is computed below (diffusion_reference, the run's json and arrays, the DiT paper's tables copied
with their provenance) and asserted, or cited with its source.

Sources (research/notes.md §18, §1, §21, Appendix A row 18, Appendix C row 18; read October 2026):
- Peebles & Xie, "Scalable Diffusion Models with Transformers" (DiT), arXiv:2212.09748 (19 Dec 2022; ICCV 2023).
  Table 1 (configs), Table 4 (all twelve S/B/L/XL × 8/4/2 models: Gflops, params, FID-50K at 400K steps, no guidance;
  block designs for XL/2), Table 2 (DiT-XL/2-G cfg 1.50: FID 2.27; LDM-4-G 3.60; ADM-G, ADM-U 3.94). Table 4 and the
  block-design rows were re-read on ar5iv (arXiv HTML) for this episode on 2026-10-27: they match notes §18.
  §3.2: "we regress them from the sum of the embedding vectors of t and c"; "We initialize the MLP to output the
  zero-vector for all α; this initializes the full DiT block as the identity function"; token count T = (I/p)²,
  "halving p will quadruple T, and thus at least quadruple total transformer Gflops".
- Esser et al., "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis" (SD3), arXiv:2403.03206
  (5 Mar 2024): MM-DiT, separate weights per modality, joint attention over the concatenated sequence, up to 8B.
- Dated list (October 2026, notes §1, §18, §21): Sora report (15 Feb 2024, "Sora is a diffusion transformer"); SD3
  (Mar 2024); FLUX.1 (1 Aug 2024, 12B); Movie Gen (Oct 2024, 30B transformer); HunyuanVideo (3 Dec 2024, 13B);
  Wan 2.1 (25 Feb 2025, 1.3B / 14B).
- research/experiments/mac_runs.md §3 ("The tiny DiT uses patch size 2 (196 tokens of 2x2 pixels), width 192, depth
  6, 4 heads, adaLN-Zero conditioning on time + label, fixed 2D sin-cos positions, and a linear unpatchify") and §4
  ("U-Net vs tiny DiT"), mac_runs.json architecture.dit/ddpm (params_by_part, recounted below), digit_models.*,
  digit_sampling.configs.{ddpm,dit}_ddim50_w{1,3}, digit_sampling.real_reference, loss_vs_t; arrays
  digit_samples_{ddpm,dit}_ddim50_w1.npz (same starting noise, seed 1234; sample i asks for label i mod 10);
  lighthouse_latents_30steps.npz (SD 2.1-base, mlx Euler sampler, 30 steps, guidance 7.5, seed 0).

Illustrative (labelled on screen or here): the shading of the token vectors (made-up values, seeded); the attention
arcs (which tokens are compared, not learned weights); the bars of the scale-and-shift demo (made-up values, seeded);
the digits circled as "broken" (picked by eye from the grid); the conv-vs-attention reach sketch. The ε̂ shown at the top
of the transformer is the lighthouse model's (a U-Net's) real noise guess at step 22, recovered exactly from the
stored trajectory: any denoiser outputs this shape.

    VOICE=draft manimgx inspect scene.py      # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403
from diffusion_reference import *  # noqa: E402,F403

EP_DIR = Path(__file__).resolve().parent          # set AFTER the import: diffusion_reference exports HERE (the series)

PREFIX = "ep18"
ARR = EXPERIMENTS / "arrays"
MAC_MD = (EXPERIMENTS / "mac_runs.md").read_text()
NOTES = (EP_DIR.parent / "research" / "notes.md").read_text()
H = lambda *keys: measured("mac_runs", "headline", *keys)  # noqa: E731

# ---------------------------------------------------------------------------- the tiny DiT and our U-Net (mac_runs §3)

DARCH = measured("mac_runs", "architecture", "dit")
UARCH = measured("mac_runs", "architecture", "ddpm")
DCFG, DPARTS, UPARTS = DARCH["config"], DARCH["params_by_part"], UARCH["params_by_part"]
PATCH, WIDTH, DEPTH, HEADS, IMG, N_TOK = (DCFG[k] for k in ("patch", "d", "depth", "heads", "img", "tokens"))
NUM_CLASSES = DCFG["num_classes"]
assert (PATCH, WIDTH, DEPTH, HEADS, IMG, N_TOK) == (2, 192, 6, 4, 28, 196) and (IMG // PATCH) ** 2 == N_TOK
assert "patch size 2 (196 tokens of 2x2 pixels), width 192, depth 6, 4 heads, adaLN-Zero" in MAC_MD


def _lin(i, o):
    return i * o + o


# one block: qkv + attention out + MLP (4× wide) + the adaLN-Zero modulation (6 numbers per channel); LayerNorms have
# no weights of their own (DiT's design) — recounted from the config and asserted equal to the json, part by part
DIALS = _lin(WIDTH, 6 * WIDTH)
BLOCK = _lin(WIDTH, 3 * WIDTH) + _lin(WIDTH, WIDTH) + _lin(WIDTH, 4 * WIDTH) + _lin(4 * WIDTH, WIDTH) + DIALS
RECOUNT = {"patch_embed": _lin(PATCH * PATCH, WIDTH), "time1": _lin(WIDTH, WIDTH), "time2": _lin(WIDTH, WIDTH),
           "label_emb": (NUM_CLASSES + 1) * WIDTH, **{f"blocks.{i}": BLOCK for i in range(DEPTH)},
           "ada_out": _lin(WIDTH, 2 * WIDTH), "out": _lin(WIDTH, PATCH * PATCH)}
assert RECOUNT == DPARTS, (RECOUNT, DPARTS)
DIT_PARAMS, UNET_PARAMS = H("dit_params"), H("digit_unet_params")
assert DIT_PARAMS == DARCH["params"] == sum(DPARTS.values()) == 4_150_660
assert UNET_PARAMS == UARCH["params"] == sum(UPARTS.values()) == 3_940_801
DIAL_SHARE = DIALS / BLOCK
BLOCKS_SHARE = DEPTH * BLOCK / DIT_PARAMS
assert BLOCK == 666_432 and DIALS == 222_336 and round(DIAL_SHARE, 3) == 0.334   # "a third of each block: the dials"
assert BLOCKS_SHARE > 0.96                                                         # "nearly all in six identical blocks"
assert f"{DIT_PARAMS / 1e6:.2f}" == "4.15"                                         # "four point one five million"

# the parts as bars (millions), in the order data flows
D_BARS = [("in", DPARTS["patch_embed"]), ("t+y", DPARTS["time1"] + DPARTS["time2"] + DPARTS["label_emb"])]
D_BARS += [(str(i + 1), DPARTS[f"blocks.{i}"]) for i in range(DEPTH)] + [("out", DPARTS["ada_out"] + DPARTS["out"])]
U_MID = UPARTS["mid1"] + UPARTS["mid_attn"] + UPARTS["mid2"]
U_EMB = UPARTS["time1"] + UPARTS["time2"] + UPARTS["label_emb"]
U_BARS = [("28↓", UPARTS["down.0"]), ("14↓", UPARTS["down.1"]), ("7↓", UPARTS["down.2"]), ("mid", U_MID),
          ("7↑", UPARTS["up.0"]), ("14↑", UPARTS["up.1"]), ("28↑", UPARTS["up.2"]), ("emb", U_EMB)]
assert sum(v for _, v in D_BARS) == DIT_PARAMS
assert sum(v for _, v in U_BARS) + UPARTS["conv_in"] + UPARTS["norm_out"] + UPARTS["conv_out"] == UNET_PARAMS
assert [f"{v / 1e6:.2f}" for _, v in U_BARS] == ["0.05", "0.19", "0.69", "0.69", "1.74", "0.44", "0.11", "0.02"]

# training: the same recipe for both (mac_runs §3)
DM = measured("mac_runs", "digit_models")
for _k in ("objective", "steps", "batch", "lr_peak", "lr_schedule", "optimizer", "ema", "label_dropout", "seed"):
    assert DM["dit"][_k] == DM["ddpm"][_k], _k
STEPS, BATCH = DM["dit"]["steps"], DM["dit"]["batch"]
assert (STEPS, BATCH) == (10_000, 256)                                             # "ten thousand steps"
MIN_D, MIN_U = f"{DM['dit']['wall_min']:.1f}", f"{DM['ddpm']['wall_min']:.1f}"
MEM_D, MEM_U = f"{DM['dit']['peak_mem_gb']:.1f}", f"{DM['ddpm']['peak_mem_gb']:.1f}"
assert (MIN_D, MIN_U, MEM_D, MEM_U) == ("34.7", "31.8", "10.3", "6.3")             # "thirty-five … thirty-two"
assert MIN_D == f"{H('dit_train_minutes'):.1f}"

# sampling: DDIM 50, same starting noise; our classifier-feature FID (not Inception), floor = real vs real
CONF = measured("mac_runs", "digit_sampling", "configs")
CU, CD = CONF["ddpm_ddim50_w1"], CONF["dit_ddim50_w1"]
CU3, CD3 = CONF["ddpm_ddim50_w3"], CONF["dit_ddim50_w3"]
for _c in (CU, CD, CU3, CD3):
    assert _c["sampler"] == "ddim" and _c["steps"] == 50
FLOOR = measured("mac_runs", "digit_sampling", "real_reference", "cfid")
CF_U, CF_D, FLOOR_S = f"{CU['cfid']:.2f}", f"{CD['cfid']:.1f}", f"{FLOOR:.2f}"
ACC_U, ACC_D = f"{100 * CU['cond_acc']:.1f}%", f"{100 * CD['cond_acc']:.1f}%"
REC_U, REC_D = f"{CU['recall']:.2f}", f"{CD['recall']:.2f}"
W3_U, W3_D = f"{CU3['cfid']:.0f}", f"{CD3['cfid']:.0f}"
SP_U, SP_D = f"{CU['s_per_1000']:.1f}", f"{CD['s_per_1000']:.1f}"
assert (CF_U, CF_D, FLOOR_S) == ("3.35", "15.9", "2.06")                           # "fifteen point nine … three point three five"
assert (ACC_U, ACC_D, REC_U, REC_D) == ("98.1%", "94.2%", "0.89", "0.87")         # "ninety-four … ninety-eight"
assert (W3_U, W3_D, SP_U, SP_D) == ("42", "101", "8.9", "10.4")                   # "a hundred and one against forty-two"
assert abs(CD["cfid"] - H("cfid_dit_ddim50")) < 0.01 and abs(CD["cond_acc"] - H("dit_ddim50_cond_acc")) < 1e-4
assert "| DiT, patch 2 | 4.15 M | 34.7 min | 10.3 GB | 10.4 | 15.9 | 94.2% | 0.87 |" in MAC_MD


def _loss_at(model, t):
    rows = measured("mac_runs", "loss_vs_t", model, "rows")
    return next(r["mse"] for r in rows if r["t"] == t)


LOSS_U, LOSS_D = f"{_loss_at('ddpm', 200.0):.3f}", f"{_loss_at('dit', 200.0):.3f}"
assert (LOSS_U, LOSS_D) == ("0.033", "0.035")                                      # "zero point zero three five …"

SAMPLES = {}
for _name in ("ddpm", "dit"):
    _f = np.load(ARR / f"digit_samples_{_name}_ddim50_w1.npz")
    assert "initial noise seed 1234" in str(_f["note"]) and np.all(_f["labels"][:50] == np.arange(50) % 10)
    SAMPLES[_name] = _f["samples"][:50].reshape(5, 10, 28, 28)                    # rows of labels 0 … 9
GRID_ROWS = 5
BROKEN = [(0, 5), (2, 0), (2, 8), (4, 5)]          # picked by eye in the DiT grid (illustrative)
X0_DIGIT = load_mnist("train")[0][15]              # the first 7 in MNIST (Episodes 5 and 8), for the patch grid

# ---------------------------------------------------------------------------- the lighthouse latent (Episode 17)

LZ = np.load(ARR / "lighthouse_latents_30steps.npz")
LX, LX0, LSIG = (LZ[k].astype(np.float64) for k in ("x_t", "x0_hat", "sigma"))
assert LX.shape == (31, 64, 64, 4) and LX0.shape == (30, 64, 64, 4) and "30 steps" in str(LZ["note"])


def lighthouse_eps(i):
    """The lighthouse model's guided noise guess at step i: the mlx Euler sampler keeps x_t = (x0 + σε)/√(σ² + 1), so
    ε̂ = (√(σ² + 1)·x_t − x̂0)/σ."""
    return (np.sqrt(LSIG[i] ** 2 + 1) * LX[i] - LX0[i]) / LSIG[i]


for _i in range(29):                                # the stored steps obey the sampler's update with this ε̂
    _e = lighthouse_eps(_i)
    _nxt = (np.sqrt(LSIG[_i] ** 2 + 1) * LX[_i] + _e * (LSIG[_i + 1] - LSIG[_i])) / np.sqrt(LSIG[_i + 1] ** 2 + 1)
    assert np.abs(_nxt - LX[_i + 1]).max() < 0.01, _i
STEP = 24
XT_L, EPS_L = LX[STEP], lighthouse_eps(STEP)
assert 1 / LSIG[STEP] ** 2 > 1 and 0.8 < EPS_L.std() < 1.1                         # more picture than noise; ε̂ ~ unit
LAT_SIDE, LAT_CH = XT_L.shape[0], XT_L.shape[2]
assert (LAT_SIDE, LAT_CH) == (64, 4)                                               # "sixty-four by sixty-four, four channels"
PS = (8, 4, 2)
TOKENS_L = {p: (LAT_SIDE // p) ** 2 for p in PS}
VALUES_L = {p: p * p * LAT_CH for p in PS}
DIT_IMG, VAE_DOWN = 256, 8                                                         # ImageNet 256 through SD's VAE
DIT_SIDE = DIT_IMG // VAE_DOWN
TOKENS_D = {p: (DIT_SIDE // p) ** 2 for p in PS}
assert TOKENS_L == {8: 64, 4: 256, 2: 1024} and VALUES_L[8] == 256                 # "two hundred fifty-six … one thousand and twenty-four"
assert DIT_SIDE == 32 and TOKENS_D == {8: 16, 4: 64, 2: 256}                       # "sixteen, sixty-four or two hundred fifty-six"
assert all(TOKENS_L[p // 2] == 4 * TOKENS_L[p] for p in (8, 4))                    # "halve the patch: four times the tokens"
assert "p = 8 → (32/8)² = **16** tokens" in NOTES and "halving p will quadruple T" in NOTES
SHOWN = 10                                                                         # tokens drawn in the tower
PICK = (2, 5)                                                                      # the patch lifted out (the tower)


def chan_lohi(a):
    return np.percentile(a, [1, 99])


def chan_rgb(a, color, lohi=None):
    lo, hi = lohi if lohi is not None else chan_lohi(a)
    return tint(np.clip((a - lo) / (hi - lo), 0, 1), color, 0, 1)


def patch_of(arr, k, p=8, ch=0):
    r, c = divmod(k, LAT_SIDE // p)
    return arr[r * p:(r + 1) * p, c * p:(c + 1) * p, ch]


# ---------------------------------------------------------------------------- DiT's results (cited)

# Peebles & Xie 2022, Table 4: ImageNet 256×256, 400K steps, no guidance → (Gflops, params in M, FID-50K)
T4 = {"S/8": (0.36, 33, 153.60), "S/4": (1.41, 33, 100.41), "S/2": (6.06, 33, 68.40),
      "B/8": (1.42, 131, 122.74), "B/4": (5.56, 130, 68.38), "B/2": (23.01, 130, 43.47),
      "L/8": (5.01, 459, 118.87), "L/4": (19.70, 458, 45.64), "L/2": (80.71, 458, 23.33),
      "XL/8": (7.39, 676, 106.41), "XL/4": (29.05, 675, 43.01), "XL/2": (118.64, 675, 19.47)}
SIZES = ("S", "B", "L", "XL")
for _s in ("S/8 0.36 / 33M / 153.60", "XL/8 7.39 / 676M / 106.41", "XL/4 29.05 / 675M / 43.01",
           "**XL/2 118.64 / 675M / 19.47**", "L/2 80.71 / 458M / 23.33", "B/2 23.01 / 130M / 43.47"):
    assert _s in NOTES, _s                                                         # the rows notes §18 quotes
for _z in SIZES:                                                                   # smaller patches: more compute, lower FID
    _g = [T4[f"{_z}/{p}"] for p in PS]
    assert _g[0][0] < _g[1][0] < _g[2][0] and _g[0][2] > _g[1][2] > _g[2][2]
for _p in PS:                                                                      # bigger models: more compute, lower FID
    _g = [T4[f"{z}/{_p}"] for z in SIZES]
    assert all(a[0] < b[0] and a[2] > b[2] for a, b in zip(_g, _g[1:]))
_lg = np.log10([v[0] for v in T4.values()])
_lf = np.log10([v[2] for v in T4.values()])
CORR = float(np.corrcoef(_lg, _lf)[0, 1])
FIT = np.polyfit(_lg, _lf, 1)
assert CORR < -0.9                                                                 # "the points fall steadily"
XL8, XL2 = T4["XL/8"], T4["XL/2"]
assert round(XL2[0] / XL8[0], 1) == 16.1 and TOKENS_D[2] / TOKENS_D[8] == 16       # 16× tokens → 16.1× Gflops
assert XL8[2] > 106 and XL2[2] < 20                                                # "about a hundred and six to under twenty"
XL_LAYERS, XL_WIDTH, XL_HEADS = 28, 1152, 16                                       # Table 1
assert "DiT-XL 28 / 1152 / 16 / 29.1" in NOTES                                     # "twenty-eight times"
BEST, UNGUIDED_LONG, LDM_G, ADM_G = 2.27, 9.62, 3.60, 3.94                         # Table 2 / Table 4 (7,000K steps)
assert '"DiT-XL/2-G (cfg=1.50) 2.27"' in NOTES and "7,000K steps **9.62**" in NOTES
assert "LDM-4-G 3.60" in NOTES and "3.94" in NOTES
DESIGNS = [("in-context", 35.24), ("cross-attn", 26.14), ("adaLN", 25.21), ("adaLN-Zero", 19.47)]
assert "in-context 35.24, cross-attention 26.14 (137.62 Gflops), adaLN 25.21, **adaLN-Zero 19.47**" in NOTES

# who uses transformers on patches (dated October 2026; notes §1, §18, §21)
USERS = [("Sora (video)", "Feb 2024", "not public"), ("Stable Diffusion 3", "Mar 2024", "up to 8B"),
         ("FLUX.1", "Aug 2024", "12B"), ("Movie Gen (video)", "Oct 2024", "30B"),
         ("HunyuanVideo", "Dec 2024", "13B"), ("Wan 2.1 (video)", "Feb 2025", "1.3B / 14B")]
for _s in ("Sora is a diffusion transformer", "MM-DiT up to 8B", "2024-08-01 | **FLUX.1**", "30B parameter transformer",
           "| HunyuanVideo (Tencent) | 3 Dec 2024 |", "| Wan 2.1 (Alibaba) | 25 Feb 2025 | 1.3B / 14B |",
           "Video models (Sora, Movie Gen, Wan, HunyuanVideo) are all DiT-style"):
    assert _s in NOTES, _s
LIGHT_M = H("lighthouse_unet_params_M")
assert LIGHT_M == 865.9                                                            # "the lighthouse's: 866 M"

# ---------------------------------------------------------------------------- look

FLAT = set("abcdefhiklmnorstuvwxzABCDEFGHIJKLMNOPRSTUVWXYZ0123456789")
SIZE_COL = {"S": mix(AMBER_FILL, AMBER, 0.45), "B": mix(AMBER_FILL, AMBER, 0.63), "L": mix(AMBER_FILL, AMBER, 0.82),
            "XL": AMBER}
RNG = np.random.default_rng(18)
VEC_IN = RNG.uniform(0.15, 0.95, (SHOWN, 6))       # illustrative shading of the token vectors
VEC_OUT = RNG.uniform(0.15, 0.95, (SHOWN, 6))


def cite(s, size=20):
    return text(s, size, MUTED)


def baseline(words):
    chars = [c for c in words.text if not c.isspace()]
    if len(words) == len(chars):
        bottoms = sorted(g.get_bottom()[1] for g, c in zip(words, chars) if c in FLAT)
        if bottoms:
            return bottoms[len(bottoms) // 2]
    return float(np.median([g.get_bottom()[1] for g in words]))


def sit(words, y):
    return words.shift((y - baseline(words)) * m.UP)


def flash(mob, color=MODEL, width=7, time_width=0.8, run_time=0.9):
    return m.ShowPassingFlash(mob.copy().set_stroke(color, width), time_width=time_width, run_time=run_time)


def rounded(width, height, color, fill, radius=0.12, stroke=2):
    return card(width, height, color, fill, radius=radius, stroke=stroke)


def labelled_chip(label, color, fill, size=20, height=0.46, pad=0.4, mono=False, ink=INK):
    words = code(label, size, ink) if mono else text(label, size, ink)
    box = rounded(max(words.width + pad, height), height, color, fill)
    return m.VGroup(box, words.move_to(box))


def tex_chip(s, color, fill, size=30, side=0.56):
    box = rounded(side, side, color, fill, radius=0.1)
    return m.VGroup(box, tex(s, size, INK).move_to(box))


def plus_node(color, radius=0.2):
    ring = m.Circle(radius=radius).set_stroke(color, 3).set_fill(BG, 1)
    bar_h = m.Line([-radius * 0.55, 0, 0], [radius * 0.55, 0, 0]).set_stroke(color, 3)
    bar_v = m.Line([0, -radius * 0.55, 0], [0, radius * 0.55, 0]).set_stroke(color, 3)
    return m.VGroup(ring, bar_h, bar_v)


def grid_lines(pic, n, color=INK, width=1.2, opacity=1.0):
    s = pic.height / n
    c = pic.get_center()
    h = pic.height / 2
    lines = m.VGroup()
    for k in range(1, n):
        off = -h + k * s
        lines.add(m.Line(c + [off, -h, 0], c + [off, h, 0]), m.Line(c + [-h, off, 0], c + [h, off, 0]))
    return lines.set_stroke(color, width, opacity=opacity)


def cell_box(pic, r, c, n, color=INK, width=4):
    s = pic.height / n
    centre = pic.get_center() + np.array([(c - (n - 1) / 2) * s, ((n - 1) / 2 - r) * s, 0])
    return m.Square(s).set_stroke(color, width).set_fill(opacity=0).move_to(centre)


def deck(arr, color, h, center, step=0.13, label_color=None):
    """A latent (side × side × channels) as stacked channel tiles, channel 0 in front; back tiles a little darker."""
    tiles = []
    back = mix(BG, color, 0.62)
    for ch in range(arr.shape[2] - 1, -1, -1):
        img = picture(chan_rgb(arr[..., ch], color if ch == 0 else back), h)
        fr = framed(img, color if ch == 0 else mix(BG, color, 0.5), buff=0.035, stroke=2)
        fr.move_to(np.array([*center, 0.0][:3], dtype=float) + ch * np.array([step, step, 0]))
        tiles.append(fr)
    g = m.Group(*tiles)
    g.front = tiles[-1]
    g.img = tiles[-1][1]
    return g


def vec_col(vals, color, center, w=0.22, cell=0.095):
    cells = m.VGroup(*[m.Rectangle(width=w, height=cell * 0.86).set_stroke(width=0).set_fill(mix(CARD_HI, color, v), 1)
                       for v in vals]).arrange(m.DOWN, buff=cell * 0.14)
    rim = m.Rectangle(width=w + 0.07, height=cells.height + 0.07).set_stroke(color, 1.6).set_fill(opacity=0)
    return m.VGroup(rim, cells).move_to(center)


def arc_up(a, b, color=MODEL, width=2.2, opacity=0.8, bend=0.95):
    angle = -bend if b[0] > a[0] else bend
    return m.ArcBetweenPoints(a, b, angle=angle).set_stroke(color, width, opacity=opacity)


def digit_grid_rgb(block, pad=2, k=3, color=DATA):
    """rows × cols digits (uint8) as one crisp picture."""
    rows, cols = block.shape[:2]
    bg = np.array([int(BG[i:i + 2], 16) for i in (1, 3, 5)], np.uint8)
    out = np.zeros((rows * 28 + (rows - 1) * pad, cols * 28 + (cols - 1) * pad, 3), np.uint8)
    out[:] = bg
    for r in range(rows):
        for c in range(cols):
            out[r * (28 + pad): r * (28 + pad) + 28, c * (28 + pad): c * (28 + pad) + 28] = tint(block[r, c], color, 0, 255)
    return np.repeat(np.repeat(out, k, axis=0), k, axis=1)


def grid_cell(image, rows, cols, r, c, pad=2):
    w_px = cols * 28 + (cols - 1) * pad
    s = image.width / w_px
    x = image.get_left()[0] + (c * (28 + pad) + 14) * s
    y = image.get_top()[1] - (r * (28 + pad) + 14) * s
    return np.array([x, y, 0.0]), 28 * s


def ellipsis(color=MUTED, center=(0, 0, 0), r=0.04, gap=0.13):
    """Three dots (a Text "…" is too small for the layout check)."""
    return m.VGroup(*[m.Dot(radius=r, color=color) for _ in range(3)]).arrange(m.RIGHT, buff=gap - 2 * r).move_to(
        np.array([*center, 0.0][:3], dtype=float))


def later(anim):
    """Keep play() from showing this animation's object before its cue (gotcha 21)."""
    anim.introducer = True
    return anim


def counter(label, value, total, centre, color=MODEL):
    return code(f"{label} {value} of {total}", 22, color, weight=m.BOLD).move_to(centre)


# ---------------------------------------------------------------------------- the episode


class Ep18DiffusionTransformers(Episode):
    number = 18

    def lesson(self):
        self.hook()
        self.patches()
        self.blocks()
        self.dials()
        self.scaling()
        self.joint_text()
        self.our_run()
        self.verdict()
        self.recap()

    # ------------------------------------------------------------------ 0 · hook

    def hook(self):
        self.next_section("Hook", notes=(
            "Every denoiser so far is a U-shaped convolutional network: the digit model's (Episode 8, "
            f"{UNET_PARAMS:,} parameters) and the lighthouse model's (SD 2.1-base U-Net, {LIGHT_M} M, mac_runs §5). Since "
            "DiT (Peebles & Xie, Dec 2022), the largest image and video models use transformers on patches: SD3 "
            "(Mar 2024), FLUX.1 (Aug 2024), Sora (Feb 2024) — dated October 2026, notes §1, §18, §21. The tower is a "
            "diagram, not a real model."))
        u = UNet(sizes=(28, 14, 7), channels=(32, 64, 128), center=(-3.45, 0.75), scale=0.62, labels=False)
        u_l = m.VGroup(text("so far: convolutions, shaped like a U", 24, INK, weight=m.BOLD),
                       cite(f"the digit model 3.9 M · the lighthouse's {LIGHT_M:.0f} M")).arrange(m.DOWN, buff=0.14)
        u_l.move_to([-3.45, -1.55, 0])
        tx = 3.45
        tiles = m.Group(*[picture(chan_rgb(patch_of(XT_L, k), DATA, chan_lohi(XT_L[..., 0])), 0.36) for k in range(7)])
        tiles.arrange(m.RIGHT, buff=0.08).move_to([tx, -0.55, 0])
        bands = m.VGroup(*[rounded(3.2, 0.36, MODEL, AMBER_FILL, radius=0.08, stroke=2) for _ in range(4)])
        bands.arrange(m.UP, buff=0.14).next_to(tiles, m.UP, buff=0.3)
        t_l = m.VGroup(text("at scale: a transformer on patches", 24, INK, weight=m.BOLD),
                       cite("DiT 2022 · SD3 2024 · FLUX 2024 · Sora 2024")).arrange(m.DOWN, buff=0.14)
        t_l.move_to([tx, -1.55, 0])
        sit(t_l[0], baseline(u_l[0]))
        sit(t_l[1], baseline(u_l[1]))
        go = arrow([u.get_right()[0] + 0.2, 0.75, 0], [bands.get_left()[0] - 0.2, 0.75, 0], MUTED)
        mascot = pixel(0.42)
        tease = pill("trained on this Mac: a tiny transformer vs our U-Net, same budget", DATA, TEAL_FILL, 20)
        foot = m.VGroup(mascot, tease).arrange(m.RIGHT, buff=0.3).move_to([0, -2.85, 0])
        assert u_l.get_left()[0] > -6.6 and t_l.get_right()[0] < 6.6 and foot.width < 12.8

        self.say(
            "[Every denoiser in this course so far has been built from convolutions, shaped like the letter U]: "
            "[our digit model's, and the lighthouse's].",
            m.AnimationGroup(self.new_header("00 · hook", "A new denoiser"), m.FadeIn(u)),
            m.AnimationGroup(m.FadeIn(u_l, shift=0.1 * m.UP), u.flow(MODEL, run_time=1.6)),
        )
        self.say(
            "[In the last few years, the biggest image and video models have swapped it] "
            "[for a transformer that reads the picture as patches].",
            m.GrowArrow(go),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(t, shift=0.15 * m.UP) for t in tiles], lag_ratio=0.12),
                             m.LaggedStart(*[m.FadeIn(b, shift=0.1 * m.UP) for b in bands], lag_ratio=0.2),
                             m.FadeIn(t_l, shift=0.1 * m.UP)),
        )
        self.say(
            "[Today: how that works and why it scales], [and what happened when we trained a tiny one on this Mac].",
            m.LaggedStart(*[flash(b, MODEL, 6, run_time=0.8) for b in bands], lag_ratio=0.25),
            m.AnimationGroup(m.FadeIn(mascot, shift=0.2 * m.UP), m.FadeIn(tease, shift=0.2 * m.UP)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 1 · patches

    def patches(self):
        lohi0 = chan_lohi(XT_L[..., 0])
        self.next_section("1 · Patches as tokens", notes=(
            "The lighthouse's latent from Episode 17: SD 2.1-base, 512×512×3 → 64×64×4, here the noisy x_t at step "
            f"{STEP} of 30 (σ = {LSIG[STEP]:.3f}, more picture than noise: SNR {1 / LSIG[STEP] ** 2:.2f}), from "
            "arrays/lighthouse_latents_30steps.npz; each channel stretched between its 1st and 99th percentile, "
            "channel 0 in front. Tokens = (side/p)²: 64×64 → "
            + ", ".join(f"p = {p}: {TOKENS_L[p]:,}" for p in PS)
            + f"; DiT's ImageNet 256×256 through SD's VAE → 32×32×4 → "
            + ", ".join(f"p = {p}: {TOKENS_D[p]}" for p in PS)
            + ' (DiT §3.2: T = (I/p)²; "halving p will quadruple T, and thus at least quadruple total transformer '
            'Gflops"). The token\'s 16 cells are the first 16 of the patch\'s 256 real values, shaded.'))
        dk = deck(XT_L, DATA, 3.0, (-3.35, -0.05))
        img = dk.img
        lab = m.VGroup(m.VGroup(tex(r"x_t", 32, DATA), code("64×64×4 latent", 22, INK)).arrange(m.RIGHT, buff=0.2),
                       cite(f"the lighthouse, step {STEP} of 30 · Episode 17")).arrange(m.DOWN, buff=0.12)
        lab.next_to(dk, m.DOWN, buff=0.22).set_x(img.get_x() + 0.2)
        g8 = grid_lines(img, 8, INK, 1.8, 0.75)
        g4 = grid_lines(img, 16, INK, 1.2, 0.55)
        g2 = grid_lines(img, 32, INK, 0.8, 0.42)
        box = cell_box(img, PICK[0], PICK[1], 8, INK, 4)

        k = PICK[0] * 8 + PICK[1]
        pd = m.Group()
        for ch in range(LAT_CH - 1, -1, -1):
            pic = picture(chan_rgb(patch_of(XT_L, k, 8, ch), DATA if ch == 0 else mix(BG, DATA, 0.62),
                                   chan_lohi(XT_L[..., ch])), 1.15)
            pd.add(framed(pic, DATA if ch == 0 else mix(BG, DATA, 0.5), buff=0.03, stroke=2)
                   .move_to(np.array([0.95, 1.25, 0]) + ch * np.array([0.1, 0.1, 0])))
        pd_l = code("8×8×4", 20, INK).next_to(pd, m.DOWN, buff=0.16)
        flat = patch_of(XT_L, k, 8, 0).ravel()
        vals = np.clip((flat[:16] - lohi0[0]) / (lohi0[1] - lohi0[0]), 0, 1)
        cells = m.VGroup(*[m.Square(0.17).set_stroke(DATA, 1.2).set_fill(mix(CARD_HI, DATA, float(v)), 1) for v in vals])
        cells.arrange(m.RIGHT, buff=0.03)
        dots = ellipsis()
        token = m.VGroup(cells, dots).arrange(m.RIGHT, buff=0.1).move_to([4.45, 1.3, 0])
        tok_box = m.SurroundingRectangle(token, buff=0.1, corner_radius=0.08).set_stroke(DATA, 2.5)
        tok_l = m.VGroup(code(f"{VALUES_L[8]} numbers", 20, INK), text("= one token, like a word", 20, DATA))
        tok_l.arrange(m.DOWN, buff=0.1).next_to(tok_box, m.DOWN, buff=0.18)
        flat_a = arrow([pd.get_right()[0] + 0.1, 1.3, 0], [tok_box.get_left()[0] - 0.05, 1.3, 0], DATA)
        assert tok_box.get_right()[0] < 6.55 and tok_l.get_right()[0] < 6.55, (tok_box.get_right(), tok_l.get_right())

        # the table: tokens per patch size, this latent and DiT's
        cx = [1.05, 3.3, 5.45]
        heads = m.VGroup(text("patch", 20, MUTED), text("this latent", 20, MUTED), text("DiT, 256×256", 20, MUTED))
        subs = m.VGroup(m.VGroup(), code("64×64×4", 20, MUTED), code("32×32×4", 20, MUTED))
        y_h, y_s, y0, dy = -0.45, -0.8, -1.35, 0.55
        for x, h_, s_ in zip(cx, heads, subs):
            h_.move_to([x, y_h, 0])
            if len(s_):
                s_.move_to([x, y_s, 0])
        for h_ in heads[1:]:
            sit(h_, baseline(heads[0]))
        rule = m.Line([cx[0] - 0.7, y_s - 0.25, 0], [cx[2] + 1.05, y_s - 0.25, 0]).set_stroke(EDGE, 2)
        rows = []
        for i, p in enumerate(PS):
            y = y0 - i * dy
            r = m.VGroup(code(f"{p}×{p}", 22, INK).move_to([cx[0], y, 0]),
                         code(f"{TOKENS_L[p]:,}", 24, DATA, weight=m.BOLD).move_to([cx[1], y, 0]),
                         code(f"{TOKENS_D[p]}", 24, DATA, weight=m.BOLD).move_to([cx[2], y, 0]))
            for w in r[1:]:
                sit(w, baseline(r[0]))
            rows.append(r)
        quad = m.VGroup()
        for i in range(2):
            ya, yb = y0 - i * dy, y0 - (i + 1) * dy
            q = code("×4", 20, INK).move_to([cx[1] + 0.95, (ya + yb) / 2, 0])
            quad.add(q)
        cite_d = cite("Peebles & Xie, 2022").move_to([cx[2], y0 - 3 * dy - 0.3, 0]).align_to([6.45, 0, 0], m.RIGHT)
        compute = text("tokens ×4 → compute at least ×4", 20, INK).move_to([cx[1] + 0.2, y0 - 3 * dy + 0.05, 0])
        compute.align_to([cx[0] - 0.7, 0, 0], m.LEFT)
        assert rows[-1].get_bottom()[1] > -3.2 and cite_d.get_bottom()[1] > -3.5
        assert compute.get_bottom()[1] > cite_d.get_top()[1] + 0.05, (compute.get_bottom(), cite_d.get_top())
        hi_rows = [m.SurroundingRectangle(r, buff=0.08, corner_radius=0.06).set_stroke(DATA, 2) for r in rows]

        self.say(
            "[Start from the latent of episode seventeen]: [sixty-four by sixty-four numbers, in four channels], "
            "[partly denoised].",
            m.AnimationGroup(self.new_header("01 · patches", "A picture as a sentence"),
                             m.LaggedStart(*[m.FadeIn(t, shift=0.1 * m.DOWN) for t in dk], lag_ratio=0.15)),
            m.FadeIn(lab[0], shift=0.1 * m.UP),
            m.FadeIn(lab[1], shift=0.1 * m.UP),
        )
        self.say(
            "[Cut it into a grid of small squares, called patches]: [eight by eight of them].",
            m.Create(g8, run_time=1.4),
            m.AnimationGroup(m.FadeIn(heads), m.FadeIn(subs), m.Create(rule), m.FadeIn(rows[0][:2], shift=0.1 * m.UP)),
        )
        self.say(
            "[Each patch, eight by eight by four numbers], [is flattened into one list]: "
            "[a token, like a word in a sentence].",
            m.AnimationGroup(m.Create(box), m.FadeIn(pd, target_position=box), m.FadeIn(pd_l)),
            m.AnimationGroup(m.GrowArrow(flat_a), m.LaggedStart(*[m.FadeIn(c_, shift=0.1 * m.RIGHT) for c_ in cells],
                                                              lag_ratio=0.06), m.FadeIn(dots)),
            m.AnimationGroup(m.Create(tok_box), m.FadeIn(tok_l, shift=0.1 * m.UP)),
        )
        self.say(
            "[Smaller patches make a longer sentence]: [patches of four give two hundred fifty-six tokens], "
            "[patches of two, one thousand and twenty-four].",
            m.Create(hi_rows[0]),
            m.AnimationGroup(m.FadeOut(box), m.ReplacementTransform(g8, g4), m.FadeIn(rows[1][:2], shift=0.1 * m.UP),
                             m.ReplacementTransform(hi_rows[0], hi_rows[1])),
            m.AnimationGroup(later(m.ReplacementTransform(g4, g2)), m.FadeIn(rows[2][:2], shift=0.1 * m.UP),
                             later(m.ReplacementTransform(hi_rows[1], hi_rows[2]))),
        )
        self.say(
            "[The DiT paper's smaller images have a thirty-two by thirty-two latent]: "
            "[sixteen, sixty-four or two hundred fifty-six tokens].",
            m.AnimationGroup(m.FadeOut(hi_rows[2]), m.FadeIn(cite_d)),
            m.LaggedStart(*[m.FadeIn(r[2], shift=0.1 * m.UP) for r in rows], lag_ratio=0.3),
        )
        self.say(
            "[Cut the patch side in half, and the tokens quadruple]: [so does the compute].",
            m.LaggedStart(*[m.FadeIn(q, shift=0.1 * m.DOWN) for q in quad], lag_ratio=0.3),
            m.FadeIn(compute, shift=0.1 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 2 · through the blocks

    def blocks(self):
        self.next_section("2 · Through the blocks", notes=(
            "DiT's pipeline (Peebles & Xie §3, Fig. 3): patchify → linear embedding to width d (DiT-XL: d = 1,152) "
            "+ fixed 2D sin-cos position embeddings → N transformer blocks (self-attention over ALL tokens + a "
            "per-token MLP; DiT-XL: 28 blocks, 16 heads, Table 1) → final layer norm, linear decoder to p×p×C per token "
            "→ rearranged into the noise prediction (DiT also predicts a variance). Drawn with p = 8 on the "
            "lighthouse's 64×64×4 latent: 64 tokens, 10 shown (the first 10 in reading order); the vectors' shading "
            "and the attention arcs are illustrative. The ε̂ at the top is the lighthouse model's (SD 2.1-base, a "
            f"U-Net) real guided noise guess at step {STEP}, recovered as (√(σ²+1)·x_t − x̂0)/σ from the stored "
            "trajectory (checked against every Euler step): it shows the shape every denoiser returns."))
        lohi_x, lohi_e = chan_lohi(XT_L[..., 0]), chan_lohi(EPS_L[..., 0])
        dx, dh = -5.4, 1.5
        din = deck(XT_L, DATA, dh, (dx, -1.95))
        dout = deck(EPS_L, MODEL, dh, (dx, 1.25))
        gin = grid_lines(din.img, 8, INK, 1.4, 0.7)
        in_l = m.VGroup(tex(r"x_t", 30, DATA), code("64×64×4", 20, INK)).arrange(m.RIGHT, buff=0.18)
        in_l.next_to(din, m.DOWN, buff=0.16).set_x(din.img.get_x() + 0.2)
        out_l = m.VGroup(tex(r"\hat\varepsilon", 32, MODEL), code("64×64×4", 20, INK)).arrange(m.RIGHT, buff=0.18)
        out_l.next_to(dout, m.DOWN, buff=0.14).set_x(dout.img.get_x() + 0.2)
        assert in_l.get_bottom()[1] > -3.5 and out_l.get_bottom()[1] > din.get_top()[1] + 0.12, (in_l.get_bottom(), out_l.get_bottom(), din.get_top())

        tx, tw = -0.1, 6.4
        xs = [tx - tw / 2 + 0.5 + i * 0.52 for i in range(SHOWN)]
        y_row, y_emb, y_blk, y_out = -3.0, -2.1, -0.15, 1.75
        b_emb = rounded(tw, 0.86, MODEL, CARD, radius=0.14, stroke=2.5).move_to([tx, y_emb, 0])
        b_blk = rounded(tw, 2.05, MODEL, CARD, radius=0.14, stroke=2.5).move_to([tx, y_blk, 0])
        b_out = rounded(tw, 0.86, MODEL, CARD, radius=0.14, stroke=2.5).move_to([tx, y_out, 0])
        tower = m.VGroup(b_emb, b_blk, b_out).set_z_index(-2)
        gaps = m.VGroup(*[m.Arrow([tx + tw / 2 - 0.35, a.get_top()[1] + 0.02, 0], [tx + tw / 2 - 0.35, b.get_bottom()[1] - 0.02, 0],
                                  buff=0, color=MUTED, stroke_width=3, max_tip_length_to_length_ratio=0.35)
                          for a, b in ((b_emb, b_blk), (b_blk, b_out))])

        tiles = m.Group(*[picture(chan_rgb(patch_of(XT_L, k), DATA, lohi_x), 0.44).move_to([xs[k], y_row, 0])
                          for k in range(SHOWN)])
        more = ellipsis(center=[xs[-1] + 0.5, y_row, 0])
        v_in = [vec_col(VEC_IN[k], DATA, [xs[k], y_emb, 0]) for k in range(SHOWN)]
        v_blk = [vec_col(VEC_IN[k], DATA, [xs[k], y_blk - 0.5, 0]) for k in range(SHOWN)]
        v_mid = [vec_col(VEC_OUT[k], MODEL, [xs[k], y_blk - 0.5, 0]) for k in range(SHOWN)]
        e_tiles = m.Group(*[picture(chan_rgb(patch_of(EPS_L, k), MODEL, lohi_e), 0.44).move_to([xs[k], y_out, 0])
                            for k in range(SHOWN)])
        more_v = [ellipsis(center=[xs[-1] + 0.5, y, 0]) for y in (y_emb, y_blk - 0.5, y_out)]
        src = 3
        tops = [v.get_top() + np.array([0, 0.04, 0]) for v in v_blk]
        arcs = m.VGroup(*[arc_up(tops[src], tops[j], bend=1.5) for j in range(SHOWN) if j != src])
        assert max(a.get_top()[1] for a in arcs) < b_blk.get_top()[1] - 0.1, max(a.get_top()[1] for a in arcs)

        lx = tx + tw / 2 + 0.35

        def side(lines, y, first_color=INK):
            rows_ = [text(lines[0], 22, first_color, weight=m.BOLD)] + [text(s, 20, MUTED) for s in lines[1:]]
            g = m.VGroup(*rows_).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
            return g.move_to([0, y, 0]).align_to([lx, 0, 0], m.LEFT)

        s_row = side([f"{TOKENS_L[8]} tokens", f"{SHOWN} shown"], y_row + 0.05, DATA)
        s_emb = side(["linear → d numbers", "+ where its patch sits"], y_emb)
        s_blk = side(["transformer block", "attention: all with all", "MLP: each on its own"], y_blk + 0.35)
        cnt = counter("block", 1, XL_LAYERS, [0, y_blk - 0.62, 0])
        cnt.align_to([lx, 0, 0], m.LEFT)
        cnt_l = cite(f"largest DiT: d = {XL_WIDTH:,}").next_to(cnt, m.DOWN, buff=0.1).align_to(cnt, m.LEFT)
        s_out = side(["linear → patch", "8×8×4 numbers again"], y_out)
        for g_ in (s_row, s_emb, s_blk, s_out, cnt, cnt_l):
            assert g_.get_right()[0] < 6.6, g_.get_right()
        src_l = m.VGroup(cite("shown: the lighthouse"), cite(f"model's ε̂, step {STEP}"))
        src_l.arrange(m.DOWN, buff=0.06, aligned_edge=m.LEFT).next_to(out_l, m.DOWN, buff=0.08)
        src_l.align_to(dout.front, m.LEFT)
        assert src_l.get_bottom()[1] > din.get_top()[1] + 0.1, (src_l.get_bottom(), din.get_top())
        assert src_l.get_right()[0] < tx - tw / 2 - 0.15, src_l.get_right()
        assert src_l.get_bottom()[1] > din.get_top()[1] + 0.05 or src_l.get_left()[0] > din.get_right()[0], src_l.get_bottom()

        go_in = arrow([din.get_right()[0] + 0.05, -2.2, 0], [xs[0] - 0.35, y_row + 0.05, 0], DATA, buff=0.05)
        go_out = arrow([tx - tw / 2 - 0.05, y_out, 0], [dout.get_right()[0] + 0.08, y_out - 0.3, 0], MODEL, buff=0.05)
        every = pill(f"every block: all {TOKENS_L[8]} tokens, same size", MODEL, AMBER_FILL, 20).move_to([tx, y_blk - 0.5, 0])
        state = {"i": 0}

        def tick(_, a):
            i = 1 + int(round(a * (XL_LAYERS - 1)))
            if i != state["i"]:
                state["i"] = i
                cnt.become(counter("block", i, XL_LAYERS, cnt.get_center()).align_to([lx, 0, 0], m.LEFT))

        rounds = m.Succession(*[m.AnimationGroup(*[flash(a_, MODEL, 4, 0.7, 0.55) for a_ in arcs]) for _ in range(4)])

        self.say(
            "[Now push the tokens through a transformer]: [ten of the sixty-four are drawn here].",
            m.AnimationGroup(self.new_header("02 · blocks", "Tokens through a transformer"), m.FadeIn(din), m.FadeIn(gin),
                             m.FadeIn(in_l), m.FadeIn(tower), m.FadeIn(gaps)),
            m.AnimationGroup(m.GrowArrow(go_in),
                             m.LaggedStart(*[m.FadeIn(t, target_position=din.img) for t in tiles], lag_ratio=0.08),
                             m.FadeIn(more), m.FadeIn(s_row)),
        )
        self.say(
            "[A linear layer turns each token into a longer vector], [and adds a code for where its patch sits].",
            m.AnimationGroup(*[m.FadeTransform(t, v) for t, v in zip(tiles, v_in)], m.FadeTransform(more, more_v[0]),
                             m.FadeIn(s_emb[0]), flash(b_emb, MODEL, 6), run_time=1.3),
            m.AnimationGroup(m.FadeIn(s_emb[1]), m.LaggedStart(*[m.Indicate(v, color=INK, scale_factor=1.12) for v in v_in],
                                                               lag_ratio=0.08)),
        )
        self.say(
            "[Then comes a stack of transformer blocks]. [In each, attention lets every patch look at every other], "
            "[as in episode eight],",
            m.AnimationGroup(*[m.ReplacementTransform(a, b) for a, b in zip(v_in, v_blk)],
                             m.ReplacementTransform(more_v[0], more_v[1]), m.FadeIn(s_blk[0]), m.FadeIn(cnt),
                             m.FadeIn(cnt_l), run_time=1.0),
            m.AnimationGroup(m.LaggedStart(*[m.Create(a_) for a_ in arcs], lag_ratio=0.08, run_time=1.4),
                             m.FadeIn(s_blk[1])),
            m.AnimationGroup(*[flash(a_, MODEL, 5, 0.7, 1.0) for a_ in arcs]),
        )
        self.say(
            "[and a small network refines each token on its own]. "
            "[The paper's largest model repeats this twenty-eight times].",
            m.AnimationGroup(m.FadeIn(s_blk[2]),
                             *[m.ReplacementTransform(a, b) for a, b in zip(v_blk, v_mid)], run_time=1.2),
            m.AnimationGroup(m.UpdateFromAlphaFunc(cnt, tick, run_time=2.4, rate_func=m.linear), rounds),
        )
        self.say(
            "[At the top, a last linear layer turns each token back into a patch]: "
            "[the noise guess, epsilon hat, as big as the latent].",
            m.AnimationGroup(m.FadeOut(arcs), *[m.FadeTransform(v, t) for v, t in zip(v_mid, e_tiles)],
                             m.ReplacementTransform(more_v[1], more_v[2]), m.FadeIn(s_out), run_time=1.2),
            m.AnimationGroup(m.GrowArrow(go_out), m.FadeIn(dout, target_position=e_tiles[0]),
                             m.FadeIn(out_l), m.FadeIn(src_l), run_time=1.2),
        )
        self.say(
            "[Nothing shrinks or grows on the way]: [every block sees all the patches, at full detail].",
            m.Circumscribe(b_blk, color=MODEL, buff=0.06),
            m.AnimationGroup(*[flash(a_, MODEL, 4, 0.8, 1.0) for a_ in gaps], m.FadeIn(every, scale=0.9)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 3 · adaLN-Zero

    def dials(self):
        self.next_section("3 · Time sets the dials", notes=(
            "Episode 8 (our U-Net): t and the label become one 128-number vector; each residual block adds one number "
            "per channel (Linear 128 → C) between its two convolutions. DiT's adaLN-Zero (§3.2): (γ₁, β₁, α₁, γ₂, β₂, α₂) "
            "= MLP(t_emb + c_emb) per block (\"we regress them from the sum of the embedding vectors of t and c\"); "
            "h ← h + α₁·Attn(γ₁·LN(h) + β₁); h ← h + α₂·MLP(γ₂·LN(h) + β₂); the MLP is initialised so every α = 0 (\"this "
            "initializes the full DiT block as the identity function\"). LayerNorm here has no learned scale of its own. "
            "Design comparison, DiT-XL/2 at 400K steps, FID-50K (Table 4): in-context 35.24, cross-attention 26.14 "
            "(137.62 Gflops, ~15% more), adaLN 25.21, adaLN-Zero 19.47. The bars in the scale-and-shift demo are "
            "made-up numbers (seeded), the dial values too."))
        # Episode 8's block, small (top left)
        y1 = 1.75
        e8 = rounded(6.15, 1.6, EDGE, CARD, radius=0.16).move_to([-3.45, y1, 0])
        e8_t = m.VGroup(text("Episode 8", 20, INK, weight=m.BOLD), cite("our U-Net")).arrange(m.RIGHT, buff=0.18)
        e8_t.align_to(e8, m.UL).shift(0.25 * m.RIGHT + 0.15 * m.DOWN)
        yy = y1 + 0.02
        h_ = code("h", 22, INK).move_to([-6.15, yy, 0])
        c1 = labelled_chip("conv 3×3", MODEL, AMBER_FILL, 20, 0.44, 0.3, mono=True).move_to([-4.95, yy, 0])
        pn = plus_node(TIME, 0.18).move_to([-3.5, yy, 0])
        c2 = labelled_chip("conv 3×3", MODEL, AMBER_FILL, 20, 0.44, 0.3, mono=True).move_to([-2.05, yy, 0])
        e8_r = [arrow(a.get_right(), b.get_left(), MUTED, buff=0.08) for a, b in ((h_, c1), (c1, pn), (pn, c2))]
        bias = text("+ one number per channel, from t and label", 20, TIME).move_to([-3.5, y1 - 0.5, 0])
        drop = m.Arrow([pn.get_x(), bias.get_top()[1] + 0.02, 0], pn.get_bottom(), buff=0.03, color=TIME,
                       stroke_width=3, max_tip_length_to_length_ratio=0.3)
        ep8 = m.VGroup(e8, e8_t, h_, c1, pn, c2, *e8_r, bias, drop)
        assert bias.get_right()[0] < e8.get_right()[0] - 0.12 and bias.get_bottom()[1] > e8.get_bottom()[1] + 0.06, bias.get_corner(m.DR)
        assert c2.get_right()[0] < e8.get_right()[0] - 0.12 and e8_t.get_bottom()[1] > c1.get_top()[1] + 0.06

        # DiT: time + label → a small network → six dials (top right)
        t_c = labelled_chip("t", TIME, BLUE_FILL, 22, 0.46, 0.36, mono=True).move_to([0.75, y1 + 0.33, 0])
        y_c = labelled_chip("label", PROMPT, PINK_FILL, 20, 0.46, 0.3, mono=True).move_to([0.75, y1 - 0.33, 0])
        net = m.VGroup(rounded(1.55, 0.86, MODEL, AMBER_FILL, radius=0.14, stroke=2.5),
                       text("small", 20, INK, weight=m.BOLD), text("network", 20, INK, weight=m.BOLD))
        m.VGroup(net[1], net[2]).arrange(m.DOWN, buff=0.04).move_to(net[0])
        net.move_to([2.55, y1, 0])
        ins = m.VGroup(arrow(t_c.get_right(), net[0].get_left() + [0, 0.15, 0], TIME, buff=0.06),
                       arrow(y_c.get_right(), net[0].get_left() + [0, -0.15, 0], PROMPT, buff=0.06))
        names = [r"\gamma_1", r"\beta_1", r"\alpha_1", r"\gamma_2", r"\beta_2", r"\alpha_2"]
        dl = m.VGroup(*[tex_chip(s, TIME, BLUE_FILL, 28, 0.56) for s in names])
        dl.arrange_in_grid(rows=2, cols=3, buff=(0.1, 0.12)).move_to([4.75, y1, 0])
        outs = arrow(net[0].get_right(), dl.get_left(), TIME, buff=0.1)
        per = cite("every block, per channel").next_to(dl, m.DOWN, buff=0.12)
        assert dl.get_right()[0] < 6.55 and per.get_bottom()[1] > y1 - 1.3, (dl.get_right(), per.get_bottom())

        # the formula (middle left) and the scale-and-shift demo (middle right)
        f = m.MathTex(r"h \;\leftarrow\; h \;+\;", r"\alpha_1", r"\cdot", r"\mathrm{Attn}(", r"\gamma_1",
                      r"\cdot \mathrm{LN}(h) +", r"\beta_1", r")", font_size=40, color=INK)
        f.move_to([-2.65, -0.05, 0])
        for i in (1, 4, 6):
            f[i].set_color(FAINT)
        wire = {i: m.Line(dl[j].get_bottom() + [0, -0.04, 0], f[i].get_top() + [0, 0.06, 0]).set_stroke(TIME, 4)
                for i, j in ((4, 0), (6, 1), (1, 2))}                          # a dial wired into the formula
        f_l = cite("the same again for the MLP, with γ₂, β₂, α₂").next_to(f, m.DOWN, buff=0.22)
        zero = m.VGroup(code("α = 0 at the start", 20, TIME, weight=m.BOLD), text("→ the block passes h straight through", 20, INK))
        zero.arrange(m.RIGHT, buff=0.2).next_to(f_l, m.DOWN, buff=0.18)
        assert f.get_left()[0] > -6.55 and zero.get_left()[0] > -6.55, (f.get_left(), zero.get_left())

        dcx, dcy, dw, dh_ = 4.05, -0.3, 3.5, 1.8
        demo_rng = np.random.default_rng(7)
        raw = demo_rng.normal(0.9, 0.9, 8)
        normed = (raw - raw.mean()) / raw.std()
        gam = demo_rng.uniform(0.35, 1.5, 8)
        bet = demo_rng.uniform(-0.6, 0.6, 8)
        stages = [raw, normed, gam * normed, gam * normed + bet]
        lim = max(np.abs(s).max() for s in stages) * 1.08
        bw = dw / 8 * 0.62
        zero_y = dcy

        def bars_at(vals, color):
            g = m.VGroup()
            for i, v in enumerate(vals):
                hgt = max(abs(v) / lim * dh_ / 2, 0.02)
                x = dcx - dw / 2 + dw / 8 * (i + 0.5)
                g.add(m.Rectangle(width=bw, height=hgt).set_stroke(color, 1.8).set_fill(FILLS[color], 1)
                      .move_to([x, zero_y + np.sign(v) * hgt / 2, 0]))
            return g

        axis = m.Line([dcx - dw / 2 - 0.1, zero_y, 0], [dcx + dw / 2 + 0.1, zero_y, 0]).set_stroke(FAINT, 2)
        b0, b1, b2, b3 = bars_at(stages[0], DATA), bars_at(stages[1], DATA), bars_at(stages[2], TIME), bars_at(stages[3], TIME)
        cap_y = zero_y - dh_ / 2 - 0.3
        caps = [text(s, 20, c).move_to([dcx, cap_y, 0]) for s, c in (("one token's numbers", MUTED),
                                                                    ("normalised", MUTED),
                                                                    ("× γ: each channel scaled", TIME),
                                                                    ("+ β: each channel shifted", TIME))]
        illus = cite("illustrative numbers").next_to(caps[0], m.DOWN, buff=0.1)
        for c_ in caps[1:]:
            sit(c_, baseline(caps[0]))
        assert max(b.get_top()[1] for st in (b0, b1, b2, b3) for b in st) < per.get_bottom()[1] - 0.15
        assert caps[2].get_right()[0] < 6.55 and min(b.get_bottom()[1] for st in (b0, b1, b2, b3) for b in st) > cap_y + 0.18

        # the four designs (bottom)
        bars = Bars([v for _, v in DESIGNS], top=40, bottom=0, width=6.0, height=1.15, gap=0.6,
                    colors=[MUTED, MUTED, MUTED, GOOD], show_values=False, center=(-2.75, -2.45, 0))
        b_nums = m.VGroup(*[code(f"{v:.2f}", 20, GOOD if i == 3 else INK).move_to([bars.xs[i], bars.level(v) + 0.2, 0])
                            for i, (_, v) in enumerate(DESIGNS)])
        b_labs = m.VGroup(*[text(n, 20, GOOD if i == 3 else MUTED).move_to([bars.xs[i], bars.y0 - 0.27, 0])
                            for i, (n, _) in enumerate(DESIGNS)])
        assert all(a.get_right()[0] < b.get_left()[0] - 0.15 for a, b in zip(b_labs, b_labs[1:]))
        for w in b_labs[1:]:
            sit(w, baseline(b_labs[0]))
        b_cap = m.VGroup(text("four ways to feed in t and label", 22, INK, weight=m.BOLD),
                         text("FID at 400K steps · lower is better", 20, MUTED),
                         cite("DiT-XL/2 · Peebles & Xie, 2022 · Table 4")).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        b_cap.move_to([0, -2.6, 0]).align_to([1.15, 0, 0], m.LEFT)
        assert b_cap.get_top()[1] < illus.get_bottom()[1] - 0.12 and b_cap.get_left()[0] > bars.get_right()[0] + 0.2, (b_cap.get_top(), illus.get_bottom())
        assert b_labs.get_bottom()[1] > -3.5 and b_cap.get_right()[0] < 6.55 and b_cap.get_bottom()[1] > -3.5, (b_labs.get_bottom(), b_cap.get_right(), b_cap.get_bottom())
        assert b_nums.get_top()[1] < zero.get_bottom()[1] - 0.12, (b_nums.get_top(), zero.get_bottom())
        gate = m.SurroundingRectangle(m.VGroup(f[3], f[4], f[5], f[6], f[7]), buff=0.08, corner_radius=0.06).set_stroke(MUTED, 2)

        self.say(
            "[In episode eight, our network got the time and label as one added number per channel].",
            m.AnimationGroup(self.new_header("03 · conditioning", "Time sets the dials"),
                             m.FadeIn(ep8, shift=0.1 * m.UP), flash(e8_r[1], TIME, 6)),
        )
        self.say(
            "[The DiT paper turns them into dials instead]: "
            "[a small network makes six sets of dials, for every block].",
            m.AnimationGroup(m.FadeIn(t_c, shift=0.1 * m.RIGHT), m.FadeIn(y_c, shift=0.1 * m.RIGHT),
                             m.GrowArrow(ins[0]), m.GrowArrow(ins[1]), m.FadeIn(net)),
            m.AnimationGroup(m.GrowArrow(outs), m.LaggedStart(*[m.FadeIn(d, shift=0.1 * m.RIGHT) for d in dl], lag_ratio=0.12),
                             m.FadeIn(per)),
        )
        self.say(
            "[Each block normalises its tokens], [then scales them by gamma], [and shifts them by beta].",
            m.AnimationGroup(m.FadeIn(f), m.Create(axis), m.FadeIn(illus),
                             m.Succession(m.AnimationGroup(m.FadeIn(b0), m.FadeIn(caps[0]), run_time=0.5),
                                          m.AnimationGroup(m.Transform(b0, b1), m.Transform(caps[0], caps[1]), run_time=0.7))),
            m.AnimationGroup(f[4].animate.set_color(TIME), m.ShowPassingFlash(wire[4], time_width=0.6, run_time=0.9),
                             m.Transform(b0, b2), m.Transform(caps[0], caps[2])),
            m.AnimationGroup(f[6].animate.set_color(TIME), m.ShowPassingFlash(wire[6], time_width=0.6, run_time=0.9),
                             m.Transform(b0, b3), m.Transform(caps[0], caps[3])),
        )
        self.say(
            "[A third dial, alpha, gates what the block adds back]. [It starts at zero], "
            "[so each new block begins by passing its input straight through].",
            m.AnimationGroup(f[1].animate.set_color(TIME), m.ShowPassingFlash(wire[1], time_width=0.6, run_time=0.9),
                             m.FadeIn(f_l)),
            m.AnimationGroup(m.Create(gate), m.FadeIn(zero[0], shift=0.1 * m.UP)),
            m.AnimationGroup(m.VGroup(f[3], f[4], f[5], f[6], f[7]).animate.set_opacity(0.3),
                             m.FadeIn(zero[1], shift=0.1 * m.UP)),
        )
        self.say(
            "[Of four ways the paper tried to feed in time and label], [this one, adaptive layer norm zero, scored best].",
            m.AnimationGroup(m.FadeIn(bars.zero), m.FadeIn(b_labs), m.FadeIn(b_cap),
                             m.LaggedStart(*[m.GrowFromEdge(b, m.DOWN) for b in bars.bars], lag_ratio=0.2), m.FadeIn(b_nums)),
            m.Circumscribe(m.VGroup(bars.bars[3], b_labs[3], b_nums[3]), color=GOOD, buff=0.08),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 4 · scaling

    def scaling(self):
        self.next_section("4 · Compute buys quality", notes=(
            "Peebles & Xie 2022, Table 4 (ImageNet 256×256, 400K steps, batch 256, no guidance, FID-50K with 250 DDPM "
            "steps): " + "; ".join(f"{k} {v[0]} Gflops / {v[1]}M / {v[2]}" for k, v in T4.items())
            + f". Log–log correlation of Gflops and FID over the 12 models: {CORR:.2f} (computed here); the dashed line "
            f"is the least-squares fit (slope {FIT[0]:.2f}). DiT-XL/8 → XL/2: same ~675M parameters, 16× the tokens, "
            f"{XL2[0] / XL8[0]:.1f}× the Gflops, FID {XL8[2]} → {XL2[2]}. DiT-XL/2 trained to 7,000K steps: FID 9.62 "
            "unguided; with classifier-free guidance 1.5 (on 3 of 4 latent channels): 2.27, the best on ImageNet "
            "256×256 at the time (Table 2: LDM-4-G 3.60, ADM-G, ADM-U 3.94; U-ViT's 2.29 was close, notes §18). FID is "
            "Episode 19's subject; lower is better."))
        plot = Plot((0.2, 200), (15, 200), width=6.3, height=3.4, center=(-2.35, -0.3, 0), x_ticks=[1, 10, 100],
                    y_ticks=[20, 50, 100], x_label="Gflops per forward pass (log)", y_label="FID · lower is better (log)",
                    log_x=True, log_y=True)
        dots, lines = m.VGroup(), m.VGroup()
        for z in SIZES:
            pts = [T4[f"{z}/{p}"] for p in PS]
            lines.add(m.VMobject().set_points_as_corners([plot.coords(g, f_) for g, _, f_ in pts])
                      .set_stroke(SIZE_COL[z], 3, opacity=0.8))
            for g, _, f_ in pts:
                dots.add(m.Dot(plot.coords(g, f_), radius=0.085, color=SIZE_COL[z]).set_z_index(3))
        xg = np.array([0.25, 140.0])
        fit = m.DashedLine(plot.coords(xg[0], 10 ** np.polyval(FIT, np.log10(xg[0]))),
                           plot.coords(xg[1], 10 ** np.polyval(FIT, np.log10(xg[1]))), dash_length=0.12).set_stroke(GOOD, 3)
        legend = m.VGroup()
        for z in SIZES:
            legend.add(m.VGroup(m.Dot(radius=0.08, color=SIZE_COL[z]), code(z, 20, INK)).arrange(m.RIGHT, buff=0.1))
        legend.arrange(m.RIGHT, buff=0.3)
        leg_cap = cite("model size")
        leg = m.VGroup(leg_cap, legend).arrange(m.RIGHT, buff=0.25)
        leg.move_to(plot.coords(1, 19)).align_to([plot.coords(0.26, 15)[0], 0, 0], m.LEFT)
        tags = m.VGroup()
        for p, off in zip(PS, ([0.0, 0.3, 0], [0.42, 0.12, 0], [0.22, 0.42, 0])):
            g, _, f_ = T4[f"XL/{p}"]
            tags.add(code(f"/{p}", 20, AMBER).move_to(plot.coords(g, f_) + np.array(off)))
        twelve = cite(f"{len(T4)} models: 4 sizes × 3 patch sizes · 400K steps · no guidance")
        twelve.next_to(plot, m.DOWN, buff=0.12)
        src = cite("Peebles & Xie, 2022 · Table 4").next_to(twelve, m.DOWN, buff=0.06)
        assert src.get_bottom()[1] > -3.55 and leg.get_top()[1] < 2.75, (src.get_bottom(), leg.get_top())

        # right column
        rx = 1.6
        cA = m.VGroup(text(f"DiT-XL, same size (~{XL2[1]} M)", 22, INK, weight=m.BOLD),
                      code(f"{TOKENS_D[8]:>3} tokens: {XL8[0]:>6} Gflops", 20, INK),
                      code(f"{TOKENS_D[2]:>3} tokens: {XL2[0]:>6} Gflops", 20, INK),
                      m.VGroup(code("FID", 20, MUTED), code(f"{XL8[2]}", 22, BAD, weight=m.BOLD), code("→", 20, MUTED),
                               code(f"{XL2[2]}", 22, GOOD, weight=m.BOLD)).arrange(m.RIGHT, buff=0.18))
        cA.arrange(m.DOWN, buff=0.14, aligned_edge=m.LEFT)
        boxA = rounded(cA.width + 0.5, cA.height + 0.4, MODEL, CARD, radius=0.14).move_to(cA)
        cardA = m.VGroup(boxA, cA).move_to([0, 1.5, 0]).align_to([rx, 0, 0], m.LEFT)
        big = title_text(f"{BEST}", 56, GOOD)
        big_r = m.VGroup(text("FID on ImageNet", 22, INK, weight=m.BOLD), text("256×256 · Dec 2022", 20, MUTED))
        big_r.arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT)
        top_row = m.VGroup(big, big_r).arrange(m.RIGHT, buff=0.3)
        big_l = m.VGroup(text("DiT-XL/2 · 7M steps · guidance 1.5", 20, MUTED),
                         code(f"before: LDM {LDM_G:.2f} · ADM {ADM_G}", 20, MUTED))
        big_l.arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        cB = m.VGroup(top_row, big_l).arrange(m.DOWN, buff=0.18, aligned_edge=m.LEFT)
        boxB = rounded(cB.width + 0.44, cB.height + 0.42, GOOD, CARD, radius=0.14).move_to(cB)
        cardB = m.VGroup(boxB, cB).move_to([0, -0.85, 0]).align_to([rx, 0, 0], m.LEFT)
        srcB = cite("Peebles & Xie, 2022 · Table 2").next_to(cardB, m.DOWN, buff=0.1).align_to(cardB, m.RIGHT)
        moral = text("buy quality with compute", 24, GOOD, weight=m.BOLD).move_to([0, -2.75, 0])
        moral.align_to(cardB, m.LEFT)
        for g_ in (cardA, cardB, srcB, moral):
            assert g_.get_right()[0] < 6.6, (g_.get_right(), g_)
        assert cardA.get_top()[1] < 2.75 and cardA.get_bottom()[1] > cardB.get_top()[1] + 0.15, (cardA.get_bottom(), cardB.get_top())
        assert moral.get_top()[1] < srcB.get_bottom()[1] - 0.1 and moral.get_bottom()[1] > -3.5, (moral.get_top(), srcB.get_bottom())
        assert cardA.get_left()[0] > plot.get_right()[0] + 0.1, (cardA.get_left(), plot.get_right())
        xl_hi = m.VGroup(lines[3].copy().set_stroke(AMBER, 6), *[d.copy() for d in dots[9:12]])

        self.say(
            "[Why switch]? [The DiT paper trained twelve of these]: [four model sizes, each with three patch sizes].",
            m.AnimationGroup(self.new_header("04 · scaling", "Compute buys quality"), m.FadeIn(plot)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(d, scale=0.5) for d in dots], lag_ratio=0.08), m.FadeIn(src)),
            m.AnimationGroup(m.FadeIn(leg), m.FadeIn(twelve), m.FadeIn(tags)),
        )
        self.say(
            "[Plot their FID, a picture-quality score, against the compute of one forward pass]: "
            "[the points fall steadily], [and lower is better].",
            m.LaggedStart(*[m.Create(ln) for ln in lines], lag_ratio=0.2, run_time=1.6),
            m.Create(fit, run_time=1.2),
            m.Indicate(plot.y_label, color=GOOD, scale_factor=1.08),
        )
        self.say(
            "[Bigger models help], [and so do smaller patches]: [the same model size with sixteen times the tokens] "
            "[went from about a hundred and six to under twenty].",
            m.LaggedStart(*[m.Indicate(dots[3 * i + 2], color=INK, scale_factor=1.6) for i in range(4)], lag_ratio=0.2),
            m.Circumscribe(lines[3], color=AMBER, buff=0.12),
            m.AnimationGroup(m.FadeIn(cardA[0]), m.FadeIn(cA[0]), m.FadeIn(cA[1]), m.FadeIn(cA[2]), flash(xl_hi[0], AMBER, 7)),
            m.FadeIn(cA[3], shift=0.1 * m.UP),
        )
        self.say(
            "[Trained much longer, with guidance], [the largest reached an FID of two point two seven on ImageNet]: "
            "[the best diffusion result at the time].",
            m.FadeIn(boxB),
            m.AnimationGroup(m.FadeIn(big, scale=0.8), m.FadeIn(big_r), m.FadeIn(big_l[0])),
            m.AnimationGroup(m.FadeIn(big_l[1]), m.FadeIn(srcB)),
        )
        self.say(
            "[Quality you can buy, predictably, with compute]: [that is why transformers took over at scale].",
            m.FadeIn(moral, shift=0.1 * m.UP),
            m.ShowPassingFlash(fit.copy().set_stroke(GOOD, 7), time_width=0.6, run_time=1.4),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 5 · MM-DiT

    def joint_text(self):
        self.next_section("5 · Words and patches, one sequence", notes=(
            "MM-DiT (Esser et al. 2024, SD3, arXiv:2403.03206 §4): text tokens and image tokens keep separate weights "
            "(their own projections, MLPs and modulation) but are concatenated for ONE attention operation, so "
            "information flows both ways; unlike Episode 15's cross-attention, where image queries read fixed text "
            "keys/values. SD3: up to 8B (depth 38), 2×2 patches of a 16-channel latent. FLUX.1: 19 double-stream "
            "(MM-DiT) blocks then 38 single-stream blocks, 12B (11.9B counted). Dated list (October 2026; notes §1, "
            "§18, §21): Sora report 15 Feb 2024 (\"Sora is a diffusion transformer\", size not disclosed); SD3 paper "
            "5 Mar 2024; FLUX.1 1 Aug 2024; Movie Gen Oct 2024 (30B); HunyuanVideo 3 Dec 2024 (13B); Wan 2.1 25 Feb 2025 "
            "(1.3B / 14B). All still diffuse VAE latents: only the denoiser changed. Arcs are illustrative."))
        lohi_x = chan_lohi(XT_L[..., 0])
        words = token_row(["lighthouse", "on", "rocky", "coast"], PROMPT, 24)
        w_more = ellipsis()
        tiles = m.Group(*[picture(chan_rgb(patch_of(XT_L, k), DATA, lohi_x), 0.62) for k in (21, 29, 37, 45)])
        tiles.arrange(m.RIGHT, buff=0.12)
        t_more = ellipsis()
        y_row = 0.35
        seq = m.Group(words, w_more, tiles, t_more).arrange(m.RIGHT, buff=0.2).move_to([0, y_row, 0])
        assert seq.width < 12.6, seq.width
        wa = m.VGroup(words, w_more)
        ta = m.Group(tiles, t_more)
        start_w = wa.copy().move_to([-3.0, 2.05, 0])
        start_t = ta.copy().move_to([3.0, -1.35, 0])
        w_tag = text("the prompt's tokens", 20, PROMPT).next_to(start_w, m.DOWN, buff=0.18)
        t_tag = text("the latent's patches", 20, DATA).next_to(start_t, m.DOWN, buff=0.18)
        assert w_tag.get_left()[0] > -6.55 and t_tag.get_right()[0] < 6.55, (w_tag.get_left(), t_tag.get_right())
        # own weights per kind
        y_w = y_row - 0.95
        ww = labelled_chip("text weights", MODEL, AMBER_FILL, 20, 0.46, 0.4).move_to([wa.get_x(), y_w, 0])
        tw_ = labelled_chip("image weights", MODEL, AMBER_FILL, 20, 0.46, 0.4).move_to([ta.get_x(), y_w, 0])
        ww[1].set_color(PROMPT)
        tw_[1].set_color(DATA)
        uw = m.VGroup(*[m.Line([g.get_x(), y_row - 0.4, 0], [g.get_x(), y_w + 0.25, 0]).set_stroke(EDGE, 2) for g in (wa, ta)])
        # one joint attention: arcs between words and patches, both ways
        anchors = [c.get_top() + np.array([0, 0.06, 0]) for c in words] + [t.get_top() + np.array([0, 0.06, 0]) for t in tiles]
        pairs = [(1, 4), (0, 6), (3, 5), (2, 7), (4, 0), (5, 2), (6, 1), (7, 3), (5, 6), (0, 2)]
        arcs = m.VGroup(*[arc_up(anchors[a], anchors[b], PROMPT if a < 4 else DATA, 2.4, 0.85, 1.0) for a, b in pairs])
        top_arc = max(a_.get_top()[1] for a_ in arcs)
        joint = text("one joint attention", 22, INK, weight=m.BOLD).move_to([0, top_arc + 0.3, 0])
        assert joint.get_top()[1] < 2.75 and top_arc < 2.3, (joint.get_top(), top_arc)
        back = m.VGroup(*[arc_up(anchors[b], anchors[a], DATA, 5, 1.0, 1.0) for a, b in ((0, 4), (1, 6), (3, 5))])
        e15 = m.VGroup(text("Episode 15, cross-attention: the words only answer", 20, MUTED),
                       text("here: the words are updated by the picture too", 22, PROMPT, weight=m.BOLD)).arrange(m.DOWN, buff=0.14)
        e15.move_to([0, y_w - 1.05, 0])
        sd3 = cite("MM-DiT · Stable Diffusion 3 · Esser et al., 2024").next_to(e15, m.DOWN, buff=0.22)
        assert sd3.get_bottom()[1] > -3.5, sd3.get_bottom()
        diagram = m.Group(seq, ww, tw_, uw, arcs, joint, e15, sd3)

        # who uses transformers on patches (its own card, after the diagram)
        rows = m.VGroup()
        col = [0.0, 2.8, 4.45]
        for i, (name, when, size) in enumerate(USERS):
            y = -0.46 * i
            r = m.VGroup(text(name, 22, INK), code(when, 22, MUTED), code(size, 22, DATA))
            for x, w in zip(col, r):
                w.move_to([0, y, 0]).align_to([x, 0, 0], m.LEFT)
            sit(r[0], y)
            for w in r[1:]:
                sit(w, baseline(r[0]))
            rows.add(r)
        assert all(r[0].get_right()[0] < col[1] - 0.15 and r[1].get_right()[0] < col[2] - 0.15 for r in rows)
        head = text("transformers on patches", 26, INK, weight=m.BOLD)
        body = m.VGroup(head, rows).arrange(m.DOWN, buff=0.32, aligned_edge=m.LEFT)
        lbox = rounded(body.width + 0.7, body.height + 0.6, DATA, CARD, radius=0.16).move_to(body)
        lcard = m.VGroup(lbox, body).move_to([0, 0.3, 0])
        note = m.VGroup(cite("as of October 2026 · Sora's size is not public"),
                        cite("all of them still denoise latents: only the denoiser changed"))
        note.arrange(m.DOWN, buff=0.08).next_to(lcard, m.DOWN, buff=0.2)
        assert lcard.get_right()[0] < 6.6 and lcard.get_top()[1] < 2.75 and note.get_bottom()[1] > -3.5, (lcard.get_right(), lcard.get_top(), note.get_bottom())

        self.say(
            "[Text-to-image models must also read the prompt]. "
            "[Stable Diffusion three puts its words into the same sequence as the patches].",
            m.AnimationGroup(self.new_header("05 · text", "Words and patches, one sequence"),
                             m.FadeIn(start_w, shift=0.1 * m.RIGHT), m.FadeIn(w_tag), m.FadeIn(start_t, shift=0.1 * m.LEFT),
                             m.FadeIn(t_tag)),
            m.AnimationGroup(m.ReplacementTransform(start_w, wa), m.FadeOut(w_tag), m.FadeOut(t_tag),
                             m.FadeOut(start_t, target_position=ta), m.FadeIn(ta, target_position=start_t), run_time=1.3),
        )
        self.say(
            "[Each kind keeps its own weights], [but one joint attention lets words and patches read each other],",
            m.AnimationGroup(m.Create(uw), m.FadeIn(ww, shift=0.1 * m.UP), m.FadeIn(tw_, shift=0.1 * m.UP)),
            m.AnimationGroup(m.LaggedStart(*[m.Create(a_) for a_ in arcs], lag_ratio=0.1, run_time=1.6), m.FadeIn(joint)),
        )
        self.say(
            "[so, unlike the cross-attention of episode fifteen], [the words are updated by the picture too].",
            m.FadeIn(e15[0], shift=0.1 * m.UP),
            m.AnimationGroup(m.LaggedStart(*[m.ShowPassingFlash(b_.copy(), time_width=0.7) for b_ in back], lag_ratio=0.2,
                                           run_time=1.4),
                             m.LaggedStart(*[m.Indicate(c, color=PROMPT, scale_factor=1.1) for c in words], lag_ratio=0.15),
                             m.FadeIn(e15[1], shift=0.1 * m.UP), m.FadeIn(sd3)),
        )
        self.say(
            "[Today, FLUX and most big video models, Sora among them], [are transformers on patches too].",
            m.AnimationGroup(m.FadeOut(diagram), m.FadeIn(lbox), m.FadeIn(head),
                             m.LaggedStart(*[m.FadeIn(r, shift=0.1 * m.LEFT) for r in rows], lag_ratio=0.15)),
            m.FadeIn(note, shift=0.1 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 6 · our run: the setup

    def our_run(self):
        self.next_section("6 · A tiny DiT on this Mac", notes=(
            f"Trained on this Mac (mac_runs §3): tiny DiT, patch {PATCH} → {N_TOK} tokens of 2×2 pixels, width {WIDTH}, "
            f"depth {DEPTH}, {HEADS} heads, adaLN-Zero on time + label, fixed 2D sin-cos positions, linear unpatchify; "
            f"{DIT_PARAMS:,} parameters, recounted here part by part (block = {BLOCK:,}: attention qkv + out, MLP 192 → "
            f"768 → 192, and {DIALS:,} for the six dials = {100 * DIAL_SHARE:.1f}%; six blocks = "
            f"{100 * BLOCKS_SHARE:.1f}% of all). Our U-Net: {UNET_PARAMS:,} (Episode 8's bars). Same recipe: "
            f"{STEPS:,} steps × batch {BATCH}, Adam 5e-4 with warm-up and cosine decay, EMA 0.999, 10% label dropout, "
            f"seed 0. Wall time {MIN_D} vs {MIN_U} min; peak training memory {MEM_D} vs {MEM_U} GB. The digit in the "
            "card is the first 7 in MNIST (training image 15) with the 14×14 patch grid."))
        # the two cards
        y_c, ch = 1.45, 2.15
        cu = rounded(6.1, ch, MODEL, CARD, radius=0.16).move_to([-3.35, y_c, 0])
        cd = rounded(6.1, ch, DATA, CARD, radius=0.16).move_to([3.35, y_c, 0])
        u_t = m.VGroup(text("our U-Net", 24, INK, weight=m.BOLD), code(f"{UNET_PARAMS:,}", 20, MODEL)).arrange(m.RIGHT, buff=0.25)
        d_t = m.VGroup(text("tiny DiT", 24, INK, weight=m.BOLD), code(f"{DIT_PARAMS:,}", 20, DATA)).arrange(m.RIGHT, buff=0.25)
        u_t.align_to(cu, m.UL).shift(0.3 * m.RIGHT + 0.2 * m.DOWN)
        d_t.align_to(cd, m.UL).shift(0.3 * m.RIGHT + 0.2 * m.DOWN)
        sit(d_t[0], baseline(u_t[0]))
        sit(d_t[1], baseline(u_t[1]))
        u_mini = UNet(sizes=(28, 14, 7), channels=(32, 64, 128), scale=0.31, labels=False)
        u_mini.move_to([cu.get_left()[0] + 1.5, y_c - 0.3, 0])
        u_txt = m.VGroup(text("convolutions", 20, INK), text("28 → 14 → 7 pixels", 20, MUTED),
                         text("attention only at 7×7", 20, MUTED)).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        u_txt.move_to([0, y_c - 0.3, 0]).align_to([cu.get_left()[0] + 3.0, 0, 0], m.LEFT)
        dig = picture(tint(X0_DIGIT, DATA, 0, 255), 1.3)
        dig.move_to([cd.get_left()[0] + 1.0, y_c - 0.28, 0])
        dig_f = framed(dig, DATA, buff=0.03, stroke=2)
        dgrid = grid_lines(dig, IMG // PATCH, INK, 0.9, 0.55)
        d_txt = m.VGroup(code(f"{N_TOK} tokens of {PATCH}×{PATCH} px", 20, INK),
                         text(f"{DEPTH} blocks · {HEADS} heads · width {WIDTH}", 20, MUTED),
                         text("adaLN-Zero on time + label", 20, MUTED)).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        d_txt.move_to([0, y_c - 0.3, 0]).align_to([dig_f.get_right()[0] + 0.3, 0, 0], m.LEFT)
        for g_ in (u_txt, d_txt):
            assert g_.get_bottom()[1] > y_c - ch / 2 + 0.08, g_.get_bottom()
        assert u_txt.get_right()[0] < cu.get_right()[0] - 0.15 and d_txt.get_right()[0] < cd.get_right()[0] - 0.15, (u_txt.get_right(), d_txt.get_right())
        assert u_mini.get_left()[0] > cu.get_left()[0] + 0.1 and u_mini.get_right()[0] < u_txt.get_left()[0] - 0.1, (u_mini.get_left(), u_mini.get_right(), u_txt.get_left())

        # parameters by part, same scale
        top_v = 1.9
        bu = Bars([v / 1e6 for _, v in U_BARS], labels=[n for n, _ in U_BARS], top=top_v, width=5.6, height=1.95, gap=0.16,
                  colors=[MODEL if n in ("7↓", "mid", "7↑") else MUTED for n, _ in U_BARS], show_values=False,
                  center=(-3.35, -1.65, 0))
        bd = Bars([v / 1e6 for _, v in D_BARS], labels=[n for n, _ in D_BARS], top=top_v, width=5.6, height=1.95, gap=0.16,
                  colors=[MUTED, MUTED] + [MODEL] * DEPTH + [MUTED], show_values=False, center=(3.35, -1.65, 0))
        for b in (bu, bd):
            for w in b.labels[1:]:
                sit(w, baseline(b.labels[0]))
        n_u = code(f"{U_BARS[4][1] / 1e6:.2f}", 20, INK).move_to([bu.xs[4], bu.level(U_BARS[4][1] / 1e6) + 0.2, 0])
        dial_caps = m.VGroup()
        for i in range(DEPTH):
            bar = bd.bars[2 + i]
            hgt = bar.height * DIAL_SHARE
            dial_caps.add(m.Rectangle(width=bar.width, height=hgt).set_stroke(TIME, 2).set_fill(BLUE_FILL, 1)
                          .move_to([bar.get_x(), bar.get_top()[1] - hgt / 2, 0]))
        blk_l = code(f"{BLOCK / 1e6:.2f} M each", 20, INK).move_to([(bd.xs[2] + bd.xs[7]) / 2, bd.bars[2].get_top()[1] + 0.24, 0])
        dial_l = text("a third of each: the dials", 20, TIME).next_to(blk_l, m.UP, buff=0.1)
        unit = cite("millions of weights by part, in the order data flows").move_to([0, -3.3, 0])
        assert unit.get_bottom()[1] > -3.5 and dial_l.get_top()[1] < y_c - ch / 2 - 0.08, dial_l.get_top()
        assert n_u.get_top()[1] < y_c - ch / 2 - 0.08
        same_t = m.VGroup(text("same data, same optimiser", 20, INK, weight=m.BOLD),
                          code(f"{STEPS:,} steps × {BATCH} digits", 20, INK)).arrange(m.DOWN, buff=0.08)
        same = m.VGroup(rounded(same_t.width + 0.5, same_t.height + 0.3, INK, CARD, radius=0.14, stroke=2).move_to(same_t), same_t)
        same.move_to([0, y_c - ch / 2 - 0.68, 0])
        assert same.get_bottom()[1] > -1.0 and same.get_left()[0] > n_u.get_right()[0] + 0.3, same.get_bottom()
        time_u = code(f"{MIN_U} min · {MEM_U} GB", 20, MODEL)
        time_d = code(f"{MIN_D} min · {MEM_D} GB", 20, DATA)
        time_u.move_to([-4.65, same.get_y(), 0])
        time_d.move_to([4.65, same.get_y(), 0])
        assert time_u.get_right()[0] < same.get_left()[0] - 0.15 and time_d.get_left()[0] > same.get_right()[0] + 0.15, (time_u.get_right(), same.get_left())

        self.say(
            "[So we trained a tiny one on this Mac], [next to our convolutional digit model].",
            m.AnimationGroup(self.new_header("06 · on this Mac", "A tiny DiT on digits"), m.FadeIn(cd), m.FadeIn(d_t),
                             m.FadeIn(dig_f)),
            m.AnimationGroup(m.FadeIn(cu), m.FadeIn(u_t), m.FadeIn(u_mini), m.FadeIn(u_txt)),
        )
        self.say(
            "[Its patches are two by two pixels]: [a hundred ninety-six tokens], [through six blocks with four attention heads].",
            m.Create(dgrid, run_time=1.2),
            m.FadeIn(d_txt[0], shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(d_txt[1], shift=0.1 * m.UP), m.FadeIn(d_txt[2], shift=0.1 * m.UP)),
        )
        self.say(
            "[About four point one five million weights, nearly all in six identical blocks], "
            "[a third of each just making the dials].",
            m.AnimationGroup(m.FadeIn(bu.zero), m.FadeIn(bu.labels), m.FadeIn(bd.zero), m.FadeIn(bd.labels), m.FadeIn(unit),
                             m.LaggedStart(*[m.GrowFromEdge(b, m.DOWN) for b in bu.bars], lag_ratio=0.08),
                             m.LaggedStart(*[m.GrowFromEdge(b, m.DOWN) for b in bd.bars], lag_ratio=0.08),
                             m.FadeIn(n_u), m.FadeIn(blk_l)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(c, shift=0.05 * m.DOWN) for c in dial_caps], lag_ratio=0.1),
                             m.FadeIn(dial_l)),
        )
        self.say(
            "[Same data, same ten thousand steps, same batch and optimiser]: "
            "[thirty-five minutes of training against thirty-two].",
            m.FadeIn(same, shift=0.1 * m.DOWN),
            m.AnimationGroup(m.FadeIn(time_d, shift=0.1 * m.LEFT), m.FadeIn(time_u, shift=0.1 * m.LEFT)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 7 · our run: the result

    def verdict(self):
        self.next_section("7 · Same noise, two denoisers", notes=(
            "Generated on this Mac (mac_runs §4): DDIM 50 steps, w = 1 (no guidance), 10,000 digits per model, sample i "
            "asks for label i mod 10 and starts from the same noise (seed 1234) in both models; shown: samples 0–49. "
            f"Our classifier-feature FID (cFID: the FID formula on our MNIST classifier's 128-d features, NOT Inception; "
            f"real-vs-real floor {FLOOR_S}, bootstrap std ≈ 0.3): U-Net {CF_U}, DiT {CF_D}. Label accuracy "
            f"{ACC_U} vs {ACC_D}; recall {REC_U} vs {REC_D}; precision {CU['precision']:.2f} vs {CD['precision']:.2f}; "
            f"s per 1,000 digits {SP_U} vs {SP_D}. With guidance w = 3: cFID {W3_U} vs {W3_D}. Test-set ε-MSE at "
            f"t = 200: {LOSS_U} vs {LOSS_D} (loss_vs_t). One training run each, one budget: the cause is not established; "
            "convolutions' built-in locality helping small models is a reasonable explanation, labelled as such. The "
            "circled DiT digits were picked by eye (illustrative)."))
        gw = 5.75
        _rgb_u, _rgb_d = digit_grid_rgb(SAMPLES["ddpm"]), digit_grid_rgb(SAMPLES["dit"])
        gh = gw * _rgb_u.shape[0] / _rgb_u.shape[1]
        gu_img = picture(_rgb_u, gh, pixelated=False)
        gd_img = picture(_rgb_d, gh, pixelated=False)
        gu_img.move_to([-3.35, 1.05, 0])
        gd_img.move_to([3.35, 1.05, 0])
        gu = framed(gu_img, MODEL, buff=0.08, stroke=2.5)
        gd = framed(gd_img, DATA, buff=0.08, stroke=2.5)
        assert gu.get_top()[1] < 2.72 and gd.get_right()[0] < 6.6, (gu.get_top(), gd.get_right())
        y_t = gu.get_bottom()[1] - 0.3
        lu = text("our U-Net", 24, INK, weight=m.BOLD).move_to([-3.35, y_t, 0]).align_to(gu, m.LEFT)
        ld = text("tiny DiT", 24, INK, weight=m.BOLD).move_to([3.35, y_t, 0]).align_to(gd, m.LEFT)
        sit(ld, baseline(lu))
        gen = cite("generated on this Mac · DDIM 50 · same starting noise · labels 0–9 left to right").move_to([0, y_t - 0.4, 0])
        rings = m.VGroup()
        for r, c in BROKEN:
            centre, side_ = grid_cell(gd_img, GRID_ROWS, 10, r, c)
            rings.add(m.Circle(radius=side_ * 0.68).set_stroke(BAD, 3.5).move_to(centre))
        by_eye = cite("circled: picked by eye").move_to([0, y_t, 0]).align_to(gd, m.RIGHT)
        sit(by_eye, baseline(ld))
        assert by_eye.get_left()[0] > ld.get_right()[0] + 0.2

        # metric rows: U-Net value · what · DiT value
        rows_spec = [(CF_U, f"our classifier score ↓ · real digits {FLOOR_S}", CF_D),
                     (ACC_U, "right label", ACC_D),
                     (W3_U, "score with guidance w = 3", W3_D),
                     (LOSS_U, "test loss at t = 200", LOSS_D)]
        y0, dy = y_t - 0.95, 0.47
        mrows = []
        for i, (a, what, b) in enumerate(rows_spec):
            y = y0 - i * dy
            good = i < 3
            r = m.VGroup(code(a, 24, GOOD if good else INK, weight=m.BOLD).move_to([-3.35, y, 0]),
                         text(what, 20, MUTED).move_to([0, y, 0]),
                         code(b, 24, BAD if good else INK, weight=m.BOLD).move_to([3.35, y, 0]))
            sit(r[0], baseline(r[1]))
            sit(r[2], baseline(r[1]))
            mrows.append(r)
        assert mrows[-1].get_bottom()[1] > -3.5 and mrows[0].get_top()[1] < gen.get_bottom()[1] - 0.08, (mrows[-1].get_bottom(), mrows[0].get_top(), gen.get_bottom())
        assert all(r[1].get_left()[0] > r[0].get_right()[0] + 0.3 and r[1].get_right()[0] < r[2].get_left()[0] - 0.3 for r in mrows)
        slight = text("only slightly higher", 20, INK).next_to(mrows[3][2], m.RIGHT, buff=0.25)
        assert slight.get_right()[0] < 6.6, slight.get_right()

        # why (a reasonable explanation, not proven)
        why_box = rounded(12.9, 2.25, EDGE, CARD, radius=0.16).move_to([0, -2.25, 0])
        dg_u = picture(tint(SAMPLES["ddpm"][3, 7], DATA, 0, 255), 1.2).move_to([-5.2, -1.95, 0])
        dg = picture(tint(SAMPLES["dit"][3, 7], DATA, 0, 255), 1.2).move_to([-2.65, -1.95, 0])
        win = cell_box(dg_u, 4, 5, 14, INK, 3).scale(3)              # a 3×3 neighbourhood of 2-px cells
        src_pt = dg.get_center() + np.array([0.15, 0.2, 0])
        reach = m.VGroup(*[m.Line(src_pt, dg.get_center() + np.array([x, y, 0])).set_stroke(INK, 1.2, opacity=0.55)
                           for x in np.linspace(-0.5, 0.5, 5) for y in np.linspace(-0.5, 0.5, 5)])
        c_lab = m.VGroup(text("convolution", 20, MODEL, weight=m.BOLD), text("neighbours, built in", 20, MUTED))
        a_lab = m.VGroup(text("attention", 20, DATA, weight=m.BOLD), text("anywhere, learned", 20, MUTED))
        c_lab.arrange(m.DOWN, buff=0.06).next_to(dg_u, m.DOWN, buff=0.1)
        a_lab.arrange(m.DOWN, buff=0.06).next_to(dg, m.DOWN, buff=0.1)
        sit(a_lab[0], baseline(c_lab[0]))
        sit(a_lab[1], baseline(c_lab[1]))
        assert c_lab.get_right()[0] < a_lab.get_left()[0] - 0.12 and a_lab.get_bottom()[1] > -3.3, (c_lab.get_right(), a_lab.get_left())
        why = m.VGroup(text("Why? One run, one budget: not proven.", 22, INK, weight=m.BOLD),
                       text("A reasonable explanation: convolutions build in that", 20, MUTED),
                       text("nearby pixels go together, which helps a small model;", 20, MUTED),
                       text("a small transformer has to learn it from data.", 20, MUTED)).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        why.move_to([0, -2.25, 0]).align_to([-1.15, 0, 0], m.LEFT)
        assert why.get_left()[0] > a_lab.get_right()[0] + 0.2, (why.get_left(), a_lab.get_right())
        assert why.get_right()[0] < 6.3 and why_box.get_bottom()[1] > -3.45, why.get_right()
        verdict_u = pill("here, small and short: convolutions win", GOOD, GOOD_FILL, 20)
        verdict_d = pill("at scale: transformers win · cited", DATA, TEAL_FILL, 20)
        verdict_u.move_to(lu).align_to(gu, m.LEFT)
        verdict_d.move_to(ld).align_to(gd, m.LEFT)

        self.say(
            "[Here are their digits, from the same starting noise], [fifty steps each].",
            m.AnimationGroup(self.new_header("07 · the result", "Same noise, two denoisers"),
                             m.FadeIn(gu, shift=0.1 * m.UP), m.FadeIn(lu)),
            m.AnimationGroup(m.FadeIn(gd, shift=0.1 * m.UP), m.FadeIn(ld), m.FadeIn(gen)),
        )
        self.say(
            "[The transformer's are visibly worse]: [broken strokes], [and some drift toward the wrong digit].",
            m.Circumscribe(gd, color=BAD, buff=0.04),
            m.AnimationGroup(m.Create(rings[0]), m.Create(rings[3]), m.FadeIn(by_eye)),
            m.AnimationGroup(m.Create(rings[1]), m.Create(rings[2])),
        )
        self.say(
            "[Our classifier-based score agrees]: [fifteen point nine against three point three five], "
            "[where real digits score about two].",
            m.FadeIn(mrows[0][1]),
            m.AnimationGroup(m.FadeIn(mrows[0][2], shift=0.1 * m.UP), m.FadeIn(mrows[0][0], shift=0.1 * m.UP)),
            m.Circumscribe(mrows[0][1], color=MUTED, buff=0.06),
        )
        self.say(
            "[It gets the label right ninety-four percent of the time, against ninety-eight]; "
            "[with guidance three, the gap grows: a hundred and one against forty-two].",
            m.FadeIn(mrows[1], shift=0.1 * m.UP),
            m.FadeIn(mrows[2], shift=0.1 * m.UP),
        )
        self.say(
            "[Yet its test loss is only slightly higher]: [zero point zero three five against zero point zero three three].",
            m.FadeIn(mrows[3][1], shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(mrows[3][2], shift=0.1 * m.UP), m.FadeIn(mrows[3][0], shift=0.1 * m.UP),
                             m.FadeIn(slight)),
        )
        mgroup = m.VGroup(*mrows, slight, gen)
        self.say(
            "[One run and one budget can't tell us why]. [A reasonable guess: convolutions build in that nearby pixels "
            "go together], [while a small transformer has to learn it].",
            m.AnimationGroup(m.FadeOut(mgroup), m.FadeIn(why_box), m.FadeIn(why[0], shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(why[1:]), m.FadeIn(dg_u), m.Create(win), m.FadeIn(c_lab)),
            m.AnimationGroup(m.FadeIn(dg), m.LaggedStart(*[m.Create(l_) for l_ in reach], lag_ratio=0.03, run_time=1.2),
                             m.FadeIn(a_lab)),
        )
        self.say(
            "[So at this size, our convolutional denoiser wins clearly]; [the transformer's advantage shows up at scale], "
            "[as the paper's curve showed].",
            m.AnimationGroup(m.FadeOut(lu), m.FadeIn(verdict_u, scale=0.9)),
            m.AnimationGroup(m.FadeOut(ld), m.FadeOut(by_eye), m.FadeIn(verdict_d, scale=0.9)),
            m.Circumscribe(verdict_d, color=DATA, buff=0.06),
        )
        self.wait(0.5)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes=(
            "Patch (the latent becomes a sequence of tokens: 64×64×4 at p = 2 → 1,024), attend (every block, all tokens, "
            "time and label setting the adaLN-Zero dials), scale (Gflops predict FID: DiT-XL/2 2.27 with guidance). On "
            f"our digits at ~4 M weights and ~30 minutes the U-Net wins (cFID {CF_U} vs {CF_D}). Next: Episode 19, "
            "Measuring Pictures (FID and our classifier-based score)."))
        halo = glow(DATA, radius=4.0)
        words = m.VGroup(*[title_text(w, 58, c) for w, c in (("Patch", DATA), ("Attend", MODEL), ("Scale", GOOD))])
        subs = m.VGroup(*[code(s, 20, MUTED) for s in (f"64×64×4 → {TOKENS_L[2]:,} tokens", "time sets the dials",
                                                       "more compute, lower FID")])
        cols_ = [m.VGroup(w, s_).arrange(m.DOWN, buff=0.3) for w, s_ in zip(words, subs)]
        m.VGroup(*cols_).arrange(m.RIGHT, buff=1.0)
        for w in words[1:]:
            sit(w, baseline(words[0]))
        line_y = min(w.get_bottom()[1] for w in words) - 0.45
        for s_ in subs:
            sit(s_, line_y)
        mid = baseline(words[0]) + 0.3
        dots = m.VGroup(*[m.Dot([(l_.get_right()[0] + r_.get_left()[0]) / 2, mid, 0], radius=0.07, color=MUTED)
                          for l_, r_ in zip(cols_, cols_[1:])])
        row = m.VGroup(words, dots, subs).move_to(1.0 * m.UP)
        assert row.width < 12.6
        tag = text("Same job as the U: one noise guess per number. A different way to look.", 26, INK)
        tag.next_to(row, m.DOWN, buff=0.65)
        mascot = pixel(0.42)
        ours = pill(f"on our digits: convolutions {CF_U} · tiny transformer {CF_D}", GOOD, GOOD_FILL, 20)
        nxt = pill("NEXT · EPISODE 19 · MEASURING PICTURES", DATA, TEAL_FILL, 20, mono=True)
        foot = m.VGroup(m.VGroup(mascot, ours).arrange(m.RIGHT, buff=0.3), nxt).arrange(m.DOWN, buff=0.3)
        foot.next_to(tag, m.DOWN, buff=0.6)
        assert foot.width < 12.8 and foot.get_bottom()[1] > -3.5, (foot.width, foot.get_bottom())

        self.say(
            "[To sum up]: [patch], [attend], [scale].",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.3 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(dots[0]), m.FadeIn(words[1], shift=0.3 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(dots[1]), m.FadeIn(words[2], shift=0.3 * m.UP), m.FadeIn(subs[2])),
        )
        self.say(
            "[Cut the latent into tokens, let every block attend to all of them while time sets the dials], "
            "[and buy quality with compute].",
            m.AnimationGroup(m.Write(tag), m.Circumscribe(words[0], color=DATA, buff=0.1),
                             m.Circumscribe(words[1], color=MODEL, buff=0.1)),
            m.Circumscribe(words[2], color=GOOD, buff=0.1),
        )
        self.say(
            "[On our tiny digits, convolutions still win]. [Next, a closer look at how we score pictures at all].",
            m.AnimationGroup(m.FadeIn(mascot, shift=0.2 * m.UP), m.FadeIn(ours, shift=0.2 * m.UP)),
            m.FadeIn(nxt, shift=0.2 * m.UP),
        )
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, tag, foot)], run_time=0.9)
