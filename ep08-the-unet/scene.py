"""Out of the Noise · Episode 8 · The U-Net (Module 2 · Learning to Denoise — the module's last episode)

The workhorse denoiser sees the whole picture and the fine detail. Running example: the digit model's REAL U-Net
(research/experiments/mac_runs.md §1 and §3, mac_runs.json "architecture"), drawn with kit.UNet(sizes=(28, 14, 7),
channels=(32, 64, 128)) and walked block by block with the run's own activations (arrays/digit_unet_activations.npz:
the first 7 in MNIST noised to t = 500 — the same training step as Episode 5 — passed through the trained U-Net; the
first 8 channels of every layer were saved). The picture shrinks 28 → 14 → 7 through the encoder (2 residual blocks
per level), passes the middle (residual block, 4-head attention, residual block), grows back through the decoder
(3 residual blocks per level, each concatenating one skip: 9 skips) and leaves as ε̂, the same 28 × 28 as the input.
The time embedding (diffusion_reference.timestep_embedding, 32 sines and cosines → MLP → 128) plus an 11 × 128 label
table enters all 17 residual blocks as a per-channel bias. Parameter counts are recounted from the config (the same
formulas gpu/models_mlx.py builds) and asserted equal to mac_runs.json part by part (3,940,801).

Every number on screen is computed below from diffusion_reference.py, the run's arrays or measured(...) and asserted,
or cited with its source.

Sources (research/notes.md §8, Appendix A, Appendix C row 8; read October 2026):
- Ronneberger, Fischer & Brox 2015 (U-Net, arXiv:1505.04597): "a contracting path to capture context and a symmetric
  expanding path that enables precise localization"; biomedical segmentation (one label per pixel).
- Ho, Jain & Abbeel 2020 (DDPM), App. B: PixelCNN++-style U-Net, GroupNorm, 2 residual blocks per resolution,
  self-attention at 16 × 16, the Transformer sinusoidal embedding added into each residual block; CIFAR-10 model
  35.7 M parameters.
- Stable Diffusion 1.x U-Net: 860 M (CompVis README, 2022; 859,520,964 counted in the v1-5 safetensors, notes §8).
- research/experiments/mac_runs.md §3 (architecture, parameters by part), §5 (the lighthouse: SD 2.1-base, U-Net
  865.9 M, measured on this Mac); mac_runs.json headline digit_unet_params, lighthouse_unet_params_M.

Illustrative (a diagram, not data): the attention lines (which cells CAN be compared, not learned weights); the
convolution's 3 × 3 window. Display choices: every feature map is shown with its sign flipped, if needed, so that it
correlates positively with what it resembles most (the noise ε or the digit, both average-pooled to the layer's size),
then stretched between its 2nd and 98th percentile; the skip panel shows the 3 most digit-like / most noise-like of
the 8 saved channels.

    VOICE=draft manimgx inspect scene.py      # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import sys
from pathlib import Path

import numpy as np

EP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EP_DIR.parent))
from kit import *  # noqa: E402,F403
from diffusion_reference import *  # noqa: E402,F403

PREFIX = "ep08"
ARR = EXPERIMENTS / "arrays"
H = lambda *keys: measured("mac_runs", "headline", *keys)  # noqa: E731

# ---------------------------------------------------------------------------- the architecture (mac_runs §3)

ARCH = measured("mac_runs", "architecture", "ddpm")
CFG, PARTS = ARCH["config"], ARCH["params_by_part"]
PARAMS = H("digit_unet_params")
CH, MULTS, NRES, TDIM, HEADS, IMG = CFG["ch"], CFG["mults"], CFG["nres"], CFG["tdim"], CFG["heads"], CFG["img"]
ATTN_RES, NUM_CLASSES = CFG["attn_res"], CFG["num_classes"]
CHANNELS = tuple(CH * k for k in MULTS)
SIZES = tuple(IMG // 2 ** i for i in range(len(MULTS)))
assert CHANNELS == (32, 64, 128) and SIZES == (28, 14, 7)                         # "32 / 64 / 128 at 28 / 14 / 7"
assert NRES == 2 and HEADS == 4 and ATTN_RES == [7] and TDIM == 128 == 4 * CH     # 2 down, 3 up; 4 heads at 7×7
assert CH == EMB_DIM == 32                                                        # the sinusoidal part: 32 numbers
assert PARAMS == ARCH["params"] == sum(PARTS.values()) == 3_940_801


def _lin(i, o):
    return i * o + o


def _conv(i, o, k=3):
    return i * o * k * k + o


def _norm(c):
    return 2 * c


def _res(i, o):
    """A residual block: GroupNorm, SiLU, conv 3×3, + bias from the embedding (Linear 128 → o), GroupNorm, SiLU,
    conv 3×3, and a 1×1 conv on the shortcut when the channel count changes."""
    return _norm(i) + _conv(i, o) + _lin(TDIM, o) + _norm(o) + _conv(o, o) + (_conv(i, o, 1) if i != o else 0)


def _attn(c):
    return _norm(c) + _lin(c, 3 * c) + _lin(c, c)


def recount():
    """The U-Net's parameters part by part, from its config, built the way gpu/models_mlx.py builds it."""
    parts = {"time1": _lin(CH, TDIM), "time2": _lin(TDIM, TDIM), "label_emb": (NUM_CLASSES + 1) * TDIM,
             "conv_in": _conv(1, CH)}
    skips, cur, res = [CH], CH, IMG
    for i, k in enumerate(MULTS):
        n = 0
        for _ in range(NRES):
            n += _res(cur, CH * k)
            cur = CH * k
            n += _attn(cur) if res in ATTN_RES else 0
            skips.append(cur)
        if i != len(MULTS) - 1:
            n += _conv(cur, cur)
            res //= 2
            skips.append(cur)
        parts[f"down.{i}"] = n
    parts.update(mid1=_res(cur, cur), mid_attn=_attn(cur), mid2=_res(cur, cur))
    for j, (i, k) in enumerate(reversed(list(enumerate(MULTS)))):
        n = 0
        for _ in range(NRES + 1):
            n += _res(cur + skips.pop(), CH * k)
            cur = CH * k
            n += _attn(cur) if res in ATTN_RES else 0
        if i != 0:
            n += _conv(cur, cur)
            res *= 2
        parts[f"up.{j}"] = n
    parts.update(norm_out=_norm(cur), conv_out=_conv(cur, 1))
    assert not skips
    return parts


assert recount() == PARTS                                                         # every label below is the real net
N_RES = NRES * len(MULTS) + 2 + (NRES + 1) * len(MULTS)
N_SKIPS = (NRES + 1) * len(MULTS)
N_ATTN = NRES + 1 + (NRES + 1)                                                    # 7×7 only: 2 down, middle, 3 up
assert (N_RES, N_SKIPS, N_ATTN) == (17, 9, 6)                                     # "seventeen", "nine in all"
assert PARTS["label_emb"] == 11 * TDIM                                            # 11 rows: ten digits + no label
assert PARTS["time1"] == _lin(32, 128) and PARTS["time2"] == _lin(128, 128)       # "32 → 128 → 128"

MID = PARTS["mid1"] + PARTS["mid_attn"] + PARTS["mid2"]
EMB = PARTS["time1"] + PARTS["time2"] + PARTS["label_emb"]
EDGES = PARTS["conv_in"] + PARTS["norm_out"] + PARTS["conv_out"]                  # the in/out convs (not drawn)
BY_PART = [("28×28", "down", PARTS["down.0"]), ("14×14", "down", PARTS["down.1"]), ("7×7", "down", PARTS["down.2"]),
           ("7×7", "middle", MID), ("7×7", "up", PARTS["up.0"]), ("14×14", "up", PARTS["up.1"]),
           ("28×28", "up", PARTS["up.2"]), ("emb.", "t + label", EMB)]
assert sum(v for *_, v in BY_PART) + EDGES == PARAMS and EDGES == 673
assert [f"{v / 1e6:.2f}" for *_, v in BY_PART] == ["0.05", "0.19", "0.69", "0.69", "1.74", "0.44", "0.11", "0.02"]
SHARE_7 = (PARTS["down.2"] + MID + PARTS["up.0"]) / PARAMS
SHARE_7_STRICT = (PARTS["down.2"] + MID + PARTS["up.0"] - _conv(128, 128)) / PARAMS   # up.0's upsample runs at 14×14
assert round(100 * SHARE_7) == 79 and SHARE_7_STRICT > 0.75                       # "more than three quarters"
POS_7, POS_28 = SIZES[2] ** 2, SIZES[0] ** 2
assert (POS_7, POS_28) == (49, 784)                                               # "forty-nine … seven hundred eighty-four"
PAIRS_7, PAIRS_28 = POS_7 ** 2, POS_28 ** 2
assert (PAIRS_7, PAIRS_28) == (2401, 614_656)                                     # "2,401 scores … over 600,000"

# ---------------------------------------------------------------------------- sizes for scale (cited / measured)

DDPM_M, SD1_M = 35.7, 860.0                    # Ho et al. 2020 App. B; CompVis README (notes §8)
LIGHT_M = H("lighthouse_unet_params_M")       # SD 2.1-base on this Mac (mac_runs §5)
assert LIGHT_M == 865.9 and round(DDPM_M * 1e6 / PARAMS) == 9                     # "about nine times ours"
assert 200 < SD1_M * 1e6 / PARAMS < LIGHT_M * 1e6 / PARAMS < 225 and round(LIGHT_M * 1e6 / PARAMS) == 220  # "about 220×"
LIGHT_PNG = EXPERIMENTS / "images" / "lighthouse" / "lighthouse_sd21_cfg7.5_steps50_seed0.png"
assert LIGHT_PNG.exists()

# ---------------------------------------------------------------------------- the run's activations (t = 500)

ACT = np.load(ARR / "digit_unet_activations.npz")
IMAGES, LABELS = load_mnist("train")
DIGIT_INDEX = int(np.flatnonzero(LABELS == 7)[0])
X0 = IMAGES[DIGIT_INDEX] / 127.5 - 1.0
T_STEP = int(ACT["t"])
XT, EPS, EPS_HAT = (ACT[k].astype(np.float64) for k in ("x_t", "eps_true", "eps_hat"))
assert DIGIT_INDEX == 15 and T_STEP == 500 and np.abs(ACT["x0"] - X0).max() < 1e-6  # Episode 5's step
_ab = ALPHA_BAR[T_STEP + 1]                                                       # code t = paper t + 1
assert np.abs(XT - (np.sqrt(_ab) * X0 + np.sqrt(1 - _ab) * EPS)).max() < 1e-4
EPS_CORR = float(np.corrcoef(EPS.ravel(), EPS_HAT.ravel())[0, 1])
assert EPS_CORR > 0.99 and XT.shape == EPS_HAT.shape == (28, 28)                 # "the same size as the input"


def level_shape(name):
    """The (side, channels) a layer should have in the drawn architecture."""
    if name == "conv_in":
        return SIZES[0], CHANNELS[0]
    if name == "mid":
        return SIZES[2], CHANNELS[2]
    part, kind = name.split(".")
    lv = int(part[-1]) if part.startswith("down") else len(MULTS) - 1 - int(part[-1])
    if kind == "downsample":
        return SIZES[lv + 1], CHANNELS[lv]
    if kind == "upsample":
        return SIZES[lv - 1], CHANNELS[lv]
    return SIZES[lv], CHANNELS[lv]


NAMES = [str(n) for n in ACT["names"]]
for _n in NAMES:
    _s, _c = level_shape(_n)
    assert tuple(ACT[_n + "__shape"]) == (_s, _s, _c), _n                          # the saved shapes match the U


def pool(a, side):
    k = a.shape[0] // side
    return a.reshape(side, k, side, k).mean(axis=(1, 3))


def corrs(name):
    """|correlation| of each saved channel with the digit and with the noise (both pooled to the layer's size)."""
    ch = ACT[name + "__ch8"].astype(np.float64)
    side = ch.shape[1]
    d, e = pool(X0, side).ravel(), pool(EPS, side).ravel()
    cd = np.array([np.corrcoef(c.ravel(), d)[0, 1] for c in ch])
    ce = np.array([np.corrcoef(c.ravel(), e)[0, 1] for c in ch])
    return cd, ce


MATCH = {}
for _n in NAMES:
    _cd, _ce = corrs(_n)
    MATCH[_n] = (float(np.abs(_ce).max()), float(np.abs(_cd).max()))              # (noise, seven)
R2 = lambda name, i: round(MATCH[name][i], 2)  # noqa: E731
assert (R2("down0.block1", 0), R2("down0.block1", 1)) == (0.80, 0.20)           # "at full size: mostly the noise"
assert R2("mid", 1) > 2.5 * R2("mid", 0)                                          # "deep in the U: the seven"
assert R2("up0.block2", 1) > 2 * R2("up0.block2", 0) and R2("up1.block1", 1) == 0.73
assert R2("up2.block2", 0) == 0.83 and R2("up2.block2", 0) > 5 * R2("up2.block2", 1)  # "the noise pattern again"
assert all(max(v) < 0.9 for k, v in MATCH.items() if k in ("conv_in", "down0.block1", "down1.block1", "down2.block1",
                                                          "mid", "up0.block2", "up1.block1", "up2.block2"))

# the skip at 28 × 28: what comes up from below (the seven) and what comes across (the noise)
_cd_up, _ = corrs("up1.upsample")
_, _ce_skip = corrs("down0.block1")
UP_PICK = list(np.argsort(-np.abs(_cd_up))[:3])
SKIP_PICK = list(np.argsort(-np.abs(_ce_skip))[:3])
assert np.abs(_cd_up[UP_PICK]).min() > 0.5 and np.abs(_ce_skip[SKIP_PICK]).min() > 0.6
assert CHANNELS[1] + CHANNELS[0] == 96                                            # "64 + 32 = 96 channels"

# one 7×7 cell is a 4×4 patch; the query (top of the 7) and its foot, on the pooled digit
D7 = pool(X0, 7)
_ink = np.argwhere(D7 > -0.2)
Q_CELL = tuple(int(v) for v in min(_ink, key=lambda rc: (rc[0], -rc[1])))       # topmost, rightmost stroke cell
FOOT = tuple(int(v) for v in max(_ink, key=lambda rc: (rc[0], -rc[1])))         # lowest stroke cell
assert FOOT[0] - Q_CELL[0] >= 4 and IMG // SIZES[2] == 4                          # far apart; "four by four"

# ---------------------------------------------------------------------------- the time embedding

T_ROWS = [0, 500, 501, 999]
EMB_ROWS = timestep_embedding(T_ROWS)
assert EMB_ROWS.shape == (4, 32)
_fast, _slow = slice(0, 3), slice(12, 16)                                          # within the sin half
assert np.abs(EMB_ROWS[1, _fast] - EMB_ROWS[2, _fast]).min() > 0.05              # neighbours differ in fast waves
assert np.abs(EMB_ROWS[1, _slow] - EMB_ROWS[2, _slow]).max() < 0.01              # … not in slow ones
assert np.abs(EMB_ROWS[0, _slow] - EMB_ROWS[3, _slow]).max() > 0.15              # slow waves: where in the schedule

# ---------------------------------------------------------------------------- Episode 7's glimpse (seed 37)

TRAJ = np.load(ARR / "digit_trajectory_7_more_seeds.npz")
TRAJ_SEED = 37
TRAJ_TS = TRAJ["ddpm_ddpm1000_w1_t"].astype(int)
TRAJ_XT = TRAJ[f"seed{TRAJ_SEED}_ddpm_ddpm1000_w1_x_t"].astype(np.float64)
TRAJ_FINAL = TRAJ[f"seed{TRAJ_SEED}_ddpm_ddpm1000_w1_final"].astype(np.float64)
assert int(TRAJ["label"]) == 7 and len(TRAJ_TS) == 1000 and TRAJ_TS[0] == 999 and TRAJ_TS[-1] == 0

# ---------------------------------------------------------------------------- look

FLAT = set("abcdefhiklmnorstuvwxzABCDEFGHIJKLMNOPRSTUVWXYZ0123456789")
U_CENTER, U_SCALE = (0.0, 0.85, 0.0), 0.9
STRIP_Y, STRIP_H = -2.52, 1.9
IN_X, OUT_X, IO_Y, IO_H = -5.3, 5.3, 1.7, 1.45
MAP = 0.66
CAP_LEFT, READ_LEFT = -6.3, 2.62
BUS_Y = -1.36


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


def side_colour(t):
    return TEAL if float(snr(ALPHA_BAR[int(t) + 1])) > 1 else VIOLET


def texture(rgb):
    k = max(1, int(np.ceil(540 / max(rgb.shape[:2]))))
    arr = np.repeat(np.repeat(rgb, k, axis=0), k, axis=1)
    return np.concatenate([arr, np.full(arr.shape[:2] + (1,), 255, np.uint8)], axis=-1)


def swap(image, rgb):
    image.paint = image.paint.but(texture=texture(rgb))


def map_rgb(a, target=None):
    """A feature map in the model's colour: sign flipped to match `target` if given, stretched 2nd–98th percentile."""
    a = np.asarray(a, dtype=np.float64)
    if target is not None and np.corrcoef(a.ravel(), target.ravel())[0, 1] < 0:
        a = -a
    lo, hi = np.percentile(a, [2, 98])
    return tint(np.clip((a - lo) / (hi - lo + 1e-9), 0, 1), MODEL, 0, 1)


def channel_map(name, i):
    ch = ACT[name + "__ch8"].astype(np.float64)[i]
    side = ch.shape[0]
    d, e = pool(X0, side), pool(EPS, side)
    cd, ce = abs(np.corrcoef(ch.ravel(), d.ravel())[0, 1]), abs(np.corrcoef(ch.ravel(), e.ravel())[0, 1])
    return map_rgb(ch, d if cd > ce else e)


def mini(rgb, size=MAP, stroke=1.5):
    return framed(picture(rgb, size), EDGE, buff=0.03, stroke=stroke)


def caption_block(lines, left=CAP_LEFT, y=STRIP_Y):
    rows = [code(lines[0], 22, INK, weight=m.BOLD)] + [text(s, 20, MUTED) for s in lines[1:]]
    group = m.VGroup(*rows).arrange(m.DOWN, buff=0.13, aligned_edge=m.LEFT)
    return group.move_to([0, y, 0]).align_to([left, 0, 0], m.LEFT)


def readout(name, left=READ_LEFT, y=STRIP_Y):
    """What the layer looks like most: best |correlation| of its 8 saved channels with the noise and the seven."""
    noise, seven = MATCH[name]
    head = text("looks most like", 20, MUTED)
    rows = m.VGroup()
    for label, v, col in (("the noise", noise, NOISE), ("the seven", seven, DATA)):
        words = text(label, 20, INK)
        bar = Meter(1.0, v, col, height=0.15)
        num = code(f"{v:.2f}", 20, INK)
        rows.add(m.VGroup(words, bar, num))
    foot = text("best correlation, 8 channels", 20, MUTED)
    block = m.VGroup(head, *rows, foot).arrange(m.DOWN, buff=0.16, aligned_edge=m.LEFT)
    block.move_to([0, y, 0]).align_to([left, 0, 0], m.LEFT)
    for r in rows:
        r[0].align_to([left, 0, 0], m.LEFT)
        r[1].move_to([left + 1.85, r[0].get_y(), 0])
        r[2].next_to(r[1], m.RIGHT, buff=0.18)
        sit(r[2], baseline(r[0]))
    return block


def stop(name, lines):
    """One stop of the walk: a caption, the layer's 8 saved channels (real) and the readout."""
    ch = ACT[name + "__ch8"]
    pics = m.Group(*[mini(channel_map(name, i)) for i in range(len(ch))]).arrange(m.RIGHT, buff=0.08)
    cap = caption_block(lines + [f"8 of {int(ACT[name + '__shape'][2])} channels"])
    out = readout(name)
    pics.move_to([(cap.get_right()[0] + out.get_left()[0]) / 2, STRIP_Y, 0])
    assert pics.get_left()[0] > cap.get_right()[0] + 0.15 and pics.get_right()[0] < out.get_left()[0] - 0.15, name
    assert out.get_right()[0] < 6.45 and out.get_bottom()[1] > STRIP_Y - STRIP_H / 2 + 0.08, name
    group = m.Group(cap, pics, out)
    group.pics = pics
    return group


def cell_centre(pic, r, c, n=7):
    s = pic.height / n
    return pic.get_center() + np.array([(c - (n - 1) / 2) * s, ((n - 1) / 2 - r) * s, 0])


def grid_lines(pic, n=7, color=EDGE, width=1.2, opacity=1.0):
    s = pic.height / n
    c = pic.get_center()
    h = pic.height / 2
    lines = m.VGroup()
    for k in range(1, n):
        off = -h + k * s
        lines.add(m.Line(c + [off, -h, 0], c + [off, h, 0]), m.Line(c + [-h, off, 0], c + [h, off, 0]))
    return lines.set_stroke(color, width, opacity=opacity)


def cell_box(pic, r0, c0, span=1, n=7, color=INK, width=3):
    s = pic.height / n
    centre = (cell_centre(pic, r0, c0, n) + cell_centre(pic, r0 + span - 1, c0 + span - 1, n)) / 2
    return m.Square(span * s).set_stroke(color, width).set_fill(opacity=0).move_to(centre)


def heat_row(values, left, y, cell_w, cell_h=0.26, gap_mid=0.12):
    """A sinusoidal embedding as a row of BLUE cells: −1 dark … +1 bright (the sin half, a gap, the cos half)."""
    cells = m.VGroup()
    half = len(values) // 2
    for k, v in enumerate(values):
        x = left + cell_w * (k + 0.5) + (gap_mid if k >= half else 0)
        cells.add(m.Rectangle(width=cell_w * 0.86, height=cell_h).set_stroke(width=0)
                  .set_fill(mix(CARD_HI, TIME, (float(v) + 1) / 2), 1).move_to([x, y, 0]))
    return cells


def rounded(width, height, color, fill, radius=0.12, stroke=2):
    return card(width, height, color, fill, radius=radius, stroke=stroke)


def labelled_chip(label, color, fill, size=20, height=0.46, pad=0.4, mono=False):
    words = code(label, size, INK) if mono else text(label, size, INK)
    box = rounded(words.width + pad, height, color, fill)
    return m.VGroup(box, words.move_to(box))


def plus_node(color, radius=0.2):
    ring = m.Circle(radius=radius).set_stroke(color, 3).set_fill(BG, 1)
    bar_h = m.Line([-radius * 0.55, 0, 0], [radius * 0.55, 0, 0]).set_stroke(color, 3)
    bar_v = m.Line([0, -radius * 0.55, 0], [0, radius * 0.55, 0]).set_stroke(color, 3)
    return m.VGroup(ring, bar_h, bar_v)


def flash(mob, color=MODEL, width=7, time_width=0.8, run_time=0.9):
    return m.ShowPassingFlash(mob.copy().set_stroke(color, width), time_width=time_width, run_time=run_time)


def skip_line(u, k):
    a = u.down[k].get_right() + np.array([0.12, 0, 0])
    b = u.up[k].get_left() - np.array([0.12, 0, 0])
    return m.Line(a, b)


def counter_text(i, centre):
    return code(f"call {i:,} of 1,000", 22, MODEL, weight=m.BOLD).move_to(centre)


# ---------------------------------------------------------------------------- the episode


class Ep08TheUNet(Episode):
    number = 8

    def lesson(self):
        self.hook()
        self.the_shape()
        self.encoder()
        self.middle()
        self.decoder()
        self.time_in()
        self.weights()
        self.recap()

    # ------------------------------------------------------------------ the U, built once

    def build_u(self):
        u = UNet(sizes=SIZES, channels=CHANNELS, center=U_CENTER, scale=U_SCALE, labels=False)
        self.u = u
        ys = [blk.get_y() for blk in u.down]
        self.levels = m.VGroup()
        for k, (s, c) in enumerate(zip(SIZES, CHANNELS)):
            words = code(f"{s}×{s} · {c} channels", 20, MUTED)
            if k == 2:
                words = code(f"{s}×{s} · {c}", 20, MUTED)
            self.levels.add(words.move_to([0, ys[k] + 0.27, 0]))
        assert self.levels[2].width < u.up[2].get_left()[0] - u.down[2].get_right()[0] - 0.25
        self.enc = text("encoder", 22, MUTED).move_to([-2.55, -0.5, 0])
        self.dec = text("decoder", 22, MUTED).move_to([2.55, -0.5, 0])
        self.mid_lab = text("middle", 22, MUTED).move_to([0, -1.36, 0])
        sit(self.dec, baseline(self.enc))
        x_in = picture(tint(np.clip(XT, -1, 1), NOISE, -1, 1), IO_H)
        e_out = picture(tint(EPS_HAT, MODEL, -2.5, 2.5), IO_H)
        self.x_in, self.e_out = x_in, e_out
        self.inp = framed(x_in, NOISE)
        self.out = framed(e_out, MODEL)
        self.in_lab = m.VGroup(tex(r"x_{500}", 32, NOISE), code("t = 500", 20, TIME)).arrange(m.RIGHT, buff=0.25)
        self.out_lab = m.VGroup(tex(r"\hat\varepsilon", 34, MODEL), text("noise guess", 20, MUTED)).arrange(m.RIGHT, buff=0.22)
        return u

    def cursor_on(self, block):
        return m.SurroundingRectangle(block, buff=0.09, corner_radius=0.05).set_stroke(INK, 3)

    # ------------------------------------------------------------------ 0 · hook

    def hook(self):
        self.next_section("Hook", notes=(
            "Callback to Episode 5: the first 7 in MNIST (training image 15) noised to t = 500 (the run's 0-based step), "
            "and the trained U-Net's own guess ε̂ for it — arrays/digit_unet_activations.npz, the same step Episode 5 "
            "showed (x_t shown clipped to −1 … 1 in VIOLET; ε̂ over −2.5 … 2.5 in AMBER). The box is Episode 5's network "
            "card. The inset is Episode 7's job: one 7 generated on this Mac by the digit model, DDPM ancestral sampling, "
            f"1,000 network calls, seed {TRAJ_SEED} (arrays/digit_trajectory_7_more_seeds.npz), every x_t played back "
            "(TEAL once the picture outweighs the noise)."))
        u = self.build_u()
        y0 = 0.95
        self.inp.move_to([-3.7, y0, 0])
        self.out.move_to([3.7, y0, 0])
        box = card(2.6, 1.7, MODEL, CARD, radius=0.18, stroke=3).move_to([0, y0, 0])
        small = UNet(sizes=SIZES, channels=CHANNELS, scale=0.27, labels=False).move_to(box)
        mark = text("what is inside?", 22, MODEL).next_to(box, m.DOWN, buff=0.22)
        go_in = arrow(self.inp.get_right(), box.get_left(), NOISE, buff=0.18)
        go_out = arrow(box.get_right(), self.out.get_left(), MODEL, buff=0.18)
        self.in_lab.next_to(self.inp, m.DOWN, buff=0.2)
        self.out_lab.next_to(self.out, m.DOWN, buff=0.2)
        ep5 = pill("Episode 5 · the network from last time", MODEL, AMBER_FILL, 20).move_to([0, 2.35, 0])

        traj = picture(tint(np.clip(TRAJ_XT[0], -1, 1), NOISE, -1, 1), 1.15)
        traj_f = framed(traj, MODEL).move_to([-1.2, -2.35, 0])
        calls = counter_text(1, [1.9, -2.05, 0])
        calls_cap = cite(f"Episode 7 · generated on this Mac\nDDPM, 1,000 steps · seed {TRAJ_SEED}")
        calls_cap.next_to(calls, m.DOWN, buff=0.18).align_to(calls, m.LEFT)
        loop = m.CurvedArrow(box.get_bottom() + np.array([0.6, -0.1, 0]), traj_f.get_top() + np.array([0.35, 0.08, 0]),
                             angle=-0.5).set_stroke(MUTED, 3)
        state = {"i": -1}

        def play_back(_, a):
            i = min(int(a * 1000), 1000)
            if i == state["i"]:
                return
            state["i"] = i
            if i < 1000:
                swap(traj, tint(np.clip(TRAJ_XT[i], -1, 1), side_colour(TRAJ_TS[i]), -1, 1))
            else:
                swap(traj, tint(TRAJ_FINAL, DATA, -1, 1))
            calls.become(counter_text(min(i + 1, 1000), calls.get_center()).align_to(calls, m.LEFT))

        self.say(
            "[In episode five, a network looked at a noisy seven], [and guessed the noise in it].",
            m.AnimationGroup(self.new_header("00 · hook", "Inside the denoiser"),
                             m.FadeIn(m.Group(self.inp, self.in_lab), shift=0.2 * m.RIGHT),
                             m.GrowArrow(go_in), m.FadeIn(m.VGroup(box, small, mark)), m.FadeIn(ep5, shift=0.1 * m.DOWN)),
            m.AnimationGroup(m.GrowArrow(go_out), m.FadeIn(m.Group(self.out, self.out_lab), shift=0.2 * m.RIGHT)),
        )
        self.say(
            "[In episode seven, that network runs a thousand times], [to turn static into a digit].",
            m.AnimationGroup(m.FadeOut(ep5), m.FadeOut(mark), m.Create(loop), m.FadeIn(traj_f), m.FadeIn(calls), m.FadeIn(calls_cap)),
            m.UpdateFromAlphaFunc(traj, play_back, run_time=3.0, rate_func=m.linear),
        )
        self.say(
            "[Today, we open the box.] [Inside is a U-Net]: [it sees the whole picture, and the fine detail].",
            m.AnimationGroup(m.FadeOut(m.Group(loop, traj_f, calls, calls_cap)),
                             m.Circumscribe(box, color=MODEL, buff=0.08)),
            m.AnimationGroup(m.FadeOut(box), m.FadeOut(go_in), m.FadeOut(go_out),
                             m.ReplacementTransform(small, u, run_time=1.6),
                             self.inp.animate.move_to([IN_X, IO_Y, 0]),
                             self.out.animate.move_to([OUT_X, IO_Y, 0]),
                             self.in_lab.animate.move_to([IN_X, IO_Y - IO_H / 2 - 0.4, 0]),
                             self.out_lab.animate.move_to([OUT_X, IO_Y - IO_H / 2 - 0.4, 0])),
            u.flow(MODEL, run_time=2.0),
        )
        self.wait(0.3)

    # ------------------------------------------------------------------ 1 · the shape

    def the_shape(self):
        self.next_section("1 · Same size in, same size out", notes=(
            "The denoiser maps 28 × 28 numbers to 28 × 28 numbers: one noise guess per pixel. The U-Net comes from "
            "Ronneberger, Fischer & Brox 2015 (arXiv:1505.04597), built for biomedical segmentation (a label for every "
            "pixel of microscope images): 'a contracting path to capture context and a symmetric expanding path that "
            "enables precise localization'. DDPM inherited it via PixelCNN++ (Ho et al. 2020, App. B). The diagram is "
            "kit.UNet(sizes=(28, 14, 7), channels=(32, 64, 128)) — the digit model's real levels."))
        u = self.u
        in_arrow = arrow(self.inp.get_right(), u.down[0].get_left(), NOISE, buff=0.12)
        out_arrow = arrow(u.up[0].get_right(), self.out.get_left(), MODEL, buff=0.12)
        self.in_arrow, self.out_arrow = in_arrow, out_arrow
        sz_in = code("28 × 28", 22, INK).next_to(self.in_lab, m.DOWN, buff=0.2)
        sz_out = code("28 × 28", 22, INK).next_to(self.out_lab, m.DOWN, buff=0.2)
        sit(sz_out, baseline(sz_in))
        per = pill("one noise guess for every pixel", MODEL, AMBER_FILL, 22).move_to([0, STRIP_Y + 0.35, 0])
        origin = m.VGroup(text("first built in 2015 to label every pixel of microscope images", 22, INK),
                          cite("Ronneberger, Fischer & Brox, 2015")).arrange(m.DOWN, buff=0.14).move_to([0, STRIP_Y - 0.45, 0])
        down = m.AnimationGroup(*[flash(a) for a in u.path[:2]], lag_ratio=0.6)
        bottom = m.AnimationGroup(*[flash(a) for a in u.path[2:4]],
                                  u.mid.animate(rate_func=m.there_and_back, run_time=1.0).set_fill(mix(AMBER_FILL, AMBER, 0.5)),
                                  lag_ratio=0.3)
        up = m.AnimationGroup(*[flash(a) for a in u.path[4:]], lag_ratio=0.6)
        skips = m.AnimationGroup(*[flash(skip_line(u, k), INK, 6, run_time=1.2) for k in range(3)], lag_ratio=0.3)

        self.say(
            "[Its job has a fixed shape]: [twenty-eight by twenty-eight numbers in], [twenty-eight by twenty-eight out], "
            "[one noise guess for every pixel].",
            m.AnimationGroup(self.new_header("01 · the shape", "Same size in, same size out"), m.GrowArrow(in_arrow),
                             m.GrowArrow(out_arrow)),
            m.FadeIn(sz_in, shift=0.1 * m.DOWN),
            m.FadeIn(sz_out, shift=0.1 * m.DOWN),
            m.FadeIn(per, shift=0.1 * m.UP),
        )
        self.say(
            "[That shape was first built in twenty fifteen], [to label every pixel of microscope images].",
            m.FadeIn(origin[1], shift=0.1 * m.UP),
            m.FadeIn(origin[0], shift=0.1 * m.UP),
        )
        self.say(
            "[Down the left side, the picture shrinks]; [at the bottom, the network takes in the whole digit]; "
            "[up the right side, it grows back].",
            m.AnimationGroup(down, m.FadeIn(self.enc)),
            m.AnimationGroup(bottom, m.FadeIn(self.mid_lab)),
            m.AnimationGroup(up, m.FadeIn(self.dec)),
        )
        self.say(
            "[And across the middle, skip connections carry the fine detail over].",
            m.AnimationGroup(skips, m.LaggedStart(*[m.FadeIn(w, shift=0.1 * m.UP) for w in self.levels], lag_ratio=0.3)),
        )
        self.wait(0.3)
        self.play(m.FadeOut(m.VGroup(per, origin, sz_in, sz_out)), run_time=0.6)

    # ------------------------------------------------------------------ 2 · the encoder

    def encoder(self):
        self.next_section("2 · Down the U", notes=(
            "Real activations of the trained digit U-Net (arrays/digit_unet_activations.npz: the first 8 channels of each "
            "layer, for the 7 at t = 500). conv_in: conv 3×3, 1 → 32 channels. Each level: 2 residual blocks (GroupNorm, "
            "SiLU, conv 3×3, + embedding bias, GroupNorm, SiLU, conv 3×3, shortcut), then a stride-2 conv 3×3 "
            "(28 → 14 → 7). 'Looks most like' = the best |correlation| over the 8 saved channels with the noise ε and with "
            "the digit, both average-pooled to the layer's size: "
            + "; ".join(f"{n}: noise {MATCH[n][0]:.2f}, seven {MATCH[n][1]:.2f}" for n in
                        ("conv_in", "down0.block1", "down1.block1", "down2.block1"))
            + ". A 7×7 cell covers a 4×4 block of input pixels (the grid on x_t); its receptive field is larger."))
        u = self.u
        s_in = stop("conv_in", ["28×28 · 32", "first convolution"])
        s_d0 = stop("down0.block1", ["28×28 · 32", "2 residual blocks"])
        s_d1 = stop("down1.block1", ["14×14 · 64", "down, second level"])
        s_d2 = stop("down2.block1", ["7×7 · 128", "down, third level"])
        frame = card(13.2, STRIP_H, EDGE, CARD, radius=0.18, stroke=2).move_to([0, STRIP_Y, 0]).set_z_index(-5)
        self.strip_frame = frame
        cur = self.cursor_on(u.down[0])
        self.cur = cur
        grid = grid_lines(self.x_in, 7, INK, 1.2, 0.55)
        q_in = cell_box(self.x_in, Q_CELL[0], Q_CELL[1], color=INK, width=3)
        q_map = cell_box(s_d2.pics[0][1], Q_CELL[0], Q_CELL[1], color=INK, width=3)
        patch = code("1 cell = 4×4 px", 20, INK).next_to(self.in_lab, m.DOWN, buff=0.16)
        patch.shift(max(0.0, -6.4 - patch.get_left()[0]) * m.RIGHT)

        self.say(
            "[A first convolution turns the one grey channel into thirty-two feature maps]: [real ones, from our model].",
            m.AnimationGroup(self.new_header("02 · encoder", "Down the U"), m.FadeIn(frame), m.Create(cur)),
            m.FadeIn(s_in, shift=0.15 * m.UP),
        )
        self.say(
            "[Two residual blocks refine them]. [At full size, the maps mostly echo the noise], [not the seven].",
            m.AnimationGroup(m.FadeOut(s_in), m.FadeIn(s_d0)),
            m.Circumscribe(s_d0[2][1], color=NOISE, buff=0.06),
            m.Circumscribe(s_d0[2][2], color=DATA, buff=0.06),
        )
        self.say(
            "[Then a stride-two convolution halves the size]: [fourteen by fourteen, with sixty-four channels], "
            "[then seven by seven, with a hundred twenty-eight].",
            flash(u.path[0], MODEL, run_time=1.0),
            m.AnimationGroup(m.Transform(cur, self.cursor_on(u.down[1])), m.FadeOut(s_d0), m.FadeIn(s_d1)),
            m.AnimationGroup(flash(u.path[1], MODEL, run_time=0.8), m.Transform(cur, self.cursor_on(u.down[2])),
                             m.FadeOut(s_d1), m.FadeIn(s_d2)),
        )
        self.say(
            "[Each step down trades detail for reach]: [one cell at seven by seven stands for a four by four patch "
            "of the input].",
            m.AnimationGroup(m.Create(grid), m.FadeIn(patch, shift=0.1 * m.UP)),
            m.AnimationGroup(m.Create(q_in), m.Create(q_map)),
        )
        self.wait(0.3)
        self.play(m.FadeOut(m.Group(grid, q_in, q_map, patch)), run_time=0.6)
        self.stop_now = s_d2

    # ------------------------------------------------------------------ 3 · the middle

    def middle(self):
        self.next_section("3 · The middle and attention", notes=(
            "The middle: residual block, self-attention (4 heads, GroupNorm, qkv Linear 128 → 384, out Linear), residual "
            "block (mac_runs §3; gpu/models_mlx.py). Attention also runs after every 7×7 residual block (2 down, 3 up): "
            "6 attention layers, all at 7×7 (attn_res = [7]). The grids show the middle's real mean |activation| map "
            "(7×7). Illustrative: the 3×3 window and the lines from one cell (the top of the 7) to all 49 cells show which "
            "cells are compared, not learned weights. Scores per head: 49² = 2,401 at 7×7; 784² = 614,656 at 28×28. "
            "DDPM used attention at 16×16 (Ho et al. 2020 App. B)."))
        u, cur = self.u, self.cur
        stack = m.VGroup(labelled_chip("residual block", MODEL, AMBER_FILL),
                         labelled_chip(f"attention · {HEADS} heads", MODEL, AMBER_FILL),
                         labelled_chip("residual block", MODEL, AMBER_FILL)).arrange(m.DOWN, buff=0.12)
        stack.move_to([0, STRIP_Y, 0])
        mean7 = ACT["mid__meanabs"].astype(np.float64)
        g_conv = picture(map_rgb(mean7), 1.3).move_to([-2.6, STRIP_Y, 0])
        g_attn = picture(map_rgb(mean7), 1.3).move_to([2.6, STRIP_Y, 0])
        lines_c, lines_a = grid_lines(g_conv), grid_lines(g_attn)
        window = cell_box(g_conv, Q_CELL[0] - 1, Q_CELL[1] - 1, span=3, color=INK, width=3)
        qa = cell_box(g_attn, Q_CELL[0], Q_CELL[1], color=INK, width=3)
        foot = cell_box(g_attn, FOOT[0], FOOT[1], color=INK, width=3)
        src = cell_centre(g_attn, *Q_CELL)
        rays = m.VGroup(*[m.Line(src, cell_centre(g_attn, r, c)).set_stroke(INK, 1.3, opacity=0.5)
                          for r in range(7) for c in range(7) if (r, c) != Q_CELL])
        long_ray = m.Line(src, cell_centre(g_attn, *FOOT)).set_stroke(INK, 4)
        c_cap = m.VGroup(text("a convolution", 20, INK, weight=m.BOLD), text("mixes each cell with", 20, MUTED),
                         text("its 8 neighbours", 20, MUTED)).arrange(m.DOWN, buff=0.1, aligned_edge=m.RIGHT)
        c_cap.next_to(g_conv, m.LEFT, buff=0.3)
        a_cap = m.VGroup(text("attention", 20, INK, weight=m.BOLD), text("every cell with all 49", 20, MUTED))
        a_cap.arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT).next_to(g_attn, m.RIGHT, buff=0.3).align_to(g_attn, m.UP)
        cost = m.VGroup(code(f"7×7: {PAIRS_7:,} scores", 20, INK), code(f"28×28: {PAIRS_28:,}", 20, BAD))
        cost.arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT).next_to(a_cap, m.DOWN, buff=0.22).align_to(a_cap, m.LEFT)
        assert cost.get_right()[0] < 6.55 and c_cap.get_left()[0] > -6.55 and cost.get_bottom()[1] > -3.45, (cost.get_right(), c_cap.get_left(), cost.get_bottom(), a_cap.get_top())
        assert stack.get_left()[0] > g_conv.get_right()[0] + 0.3 and stack.get_right()[0] < g_attn.get_left()[0] - 0.3
        at7 = m.VGroup(u.down[2], u.mid, u.up[2])

        self.say(
            "[At the very bottom sits the middle]: [a residual block], [an attention layer], [and another residual block].",
            m.AnimationGroup(self.new_header("03 · middle", "Distant pixels talk"), m.Transform(cur, self.cursor_on(u.mid)),
                             flash(u.path[2], MODEL), m.FadeOut(self.stop_now)),
            m.FadeIn(stack[0], shift=0.1 * m.UP),
            m.FadeIn(stack[1], shift=0.1 * m.UP),
            m.FadeIn(stack[2], shift=0.1 * m.UP),
        )
        self.say(
            "[A convolution only mixes each cell with its neighbours].",
            m.AnimationGroup(m.FadeIn(m.Group(g_conv, lines_c)), m.FadeIn(c_cap), m.Create(window)),
        )
        self.say(
            "Attention, in one line: [it lets distant pixels talk], [so the top of the seven can check on its foot].",
            m.AnimationGroup(m.FadeIn(m.Group(g_attn, lines_a)), m.Create(qa), m.FadeIn(a_cap),
                             m.LaggedStart(*[m.Create(r) for r in rays], lag_ratio=0.02, run_time=1.4)),
            m.AnimationGroup(m.Create(long_ray), m.Create(foot)),
        )
        self.say(
            "[It runs with four heads, and only at seven by seven].",
            m.AnimationGroup(*[m.Circumscribe(b, color=MODEL, buff=0.12) for b in at7]),
        )
        self.say(
            "[There, forty-nine cells make two thousand four hundred one scores]; "
            "[full size would make over six hundred thousand].",
            m.FadeIn(cost[0], shift=0.1 * m.UP),
            m.FadeIn(cost[1], shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.play(m.FadeOut(m.Group(stack, g_conv, g_attn, lines_c, lines_a, window, qa, foot, rays, long_ray,
                                    c_cap, a_cap, cost)), run_time=0.6)

    # ------------------------------------------------------------------ 4 · the decoder

    def decoder(self):
        self.next_section("4 · Back up, with skips", notes=(
            "Up path: 3 residual blocks per level, each first concatenating one stored map from the way down (9 skips: "
            "3 per level), with attention after the 7×7 ones; upsampling = nearest ×2 then conv 3×3; finally GroupNorm, "
            "SiLU, conv 3×3 to 1 channel = ε̂ (mac_runs §3). Real maps: "
            + "; ".join(f"{n}: noise {MATCH[n][0]:.2f}, seven {MATCH[n][1]:.2f}" for n in ("up0.block2", "up1.block1",
                                                                                         "up2.block2"))
            + ". The skip panel: 3 of the 8 saved channels of up1.upsample (the 28×28 maps grown back from 14×14, chosen "
            "as the most digit-like) and of down0.block1 (the skip, chosen as the most noise-like); the first 28×28 up "
            f"block stacks them: 64 + 32 = 96 channels → 32. ε̂ vs the true ε: correlation {EPS_CORR:.3f} "
            "(Episode 5: squared error 0.0043)."))
        u, cur = self.u, self.cur
        s_u0 = stop("up0.block2", ["7×7 · 128", "up, third level"])
        s_u1 = stop("up1.block1", ["14×14 · 64", "up, second level"])
        s_u2 = stop("up2.block2", ["28×28 · 32", "last up block"])

        ups = m.Group(*[mini(channel_map("up1.upsample", i)) for i in UP_PICK]).arrange(m.RIGHT, buff=0.1)
        sks = m.Group(*[mini(channel_map("down0.block1", i)) for i in SKIP_PICK]).arrange(m.RIGHT, buff=0.1)
        y_maps = STRIP_Y + 0.17
        ups.move_to([-3.1, y_maps, 0])
        plus = plus_node(INK).move_to([-1.15, y_maps, 0])
        sks.move_to([0.8, y_maps, 0])
        ups_cap = cite("grown back from 14×14").next_to(ups, m.DOWN, buff=0.15)
        sks_cap = cite("skip from the way down").next_to(sks, m.DOWN, buff=0.15)
        sit(sks_cap, baseline(ups_cap))
        lead = caption_block(["28×28", "first up block:", "two inputs"])
        result = m.VGroup(code("64 + 32 = 96", 22, INK, weight=m.BOLD), text("channels, stacked", 20, MUTED),
                          text("then a residual block", 20, MUTED)).arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT)
        result.move_to([0, STRIP_Y, 0]).align_to([3.6, 0, 0], m.LEFT)
        eq = arrow([sks.get_right()[0] + 0.05, y_maps, 0], [result.get_left()[0] - 0.05, y_maps, 0], MUTED)
        concat = m.Group(lead, ups, plus, sks, ups_cap, sks_cap, eq, result)
        assert ups_cap.get_bottom()[1] > -3.45 and result.get_right()[0] < 6.55

        tags = m.VGroup()
        tag_at = [(u.up[0].get_right()[0] + 0.32, u.up[0].get_y() - 0.6), (u.up[1].get_right()[0] + 0.32, u.up[1].get_y() - 0.25),
                  (u.up[2].get_right()[0] + 0.32, u.up[2].get_y() - 0.2)]
        for x, y in tag_at:
            tags.add(code("×3", 20, INK).move_to([x, y, 0]))
        total = pill(f"{N_SKIPS} skip connections", INK, CARD, 20)
        total.move_to([5.15, -0.45, 0])
        e_hi = m.SurroundingRectangle(self.out, buff=0.08, corner_radius=0.05).set_stroke(MODEL, 4)
        same = code("28×28 · 1 channel", 20, INK).next_to(self.out_lab, m.DOWN, buff=0.14)

        self.say(
            "[Then the decoder grows the picture back]. [Deep down, the maps follow the seven], [not the noise].",
            m.AnimationGroup(self.new_header("04 · decoder", "Back up, with skips"), flash(u.path[3], MODEL),
                             m.Transform(cur, self.cursor_on(u.up[2]))),
            m.FadeIn(s_u0, shift=0.15 * m.UP),
            m.AnimationGroup(m.Circumscribe(s_u0[2][2], color=DATA, buff=0.06)),
        )
        self.say(
            "[At fourteen by fourteen, the seven is clearer still].",
            m.AnimationGroup(flash(u.path[4], MODEL), m.Transform(cur, self.cursor_on(u.up[1])), m.FadeOut(s_u0),
                             m.FadeIn(s_u1)),
        )
        self.say(
            "[But maps grown back from a coarse picture are blocky].",
            m.AnimationGroup(m.FadeOut(s_u1), m.FadeIn(lead), m.FadeIn(ups, shift=0.1 * m.UP), m.FadeIn(ups_cap)),
        )
        self.say(
            "[Skip connections bring the fine detail straight across], "
            "[from each level on the way down to the same level on the way up].",
            m.AnimationGroup(*[flash(skip_line(u, k), INK, 7, run_time=1.3) for k in range(3)],
                             m.FadeIn(plus), m.FadeIn(sks, shift=0.1 * m.LEFT), m.FadeIn(sks_cap)),
            m.AnimationGroup(*[u.skips[k].animate.set_color(INK) for k in range(3)]),
        )
        self.say(
            "[Each up block stacks one of them onto its input]: [three per level], [nine in all].",
            m.AnimationGroup(m.FadeIn(eq), m.FadeIn(result, shift=0.1 * m.LEFT)),
            m.LaggedStart(*[m.FadeIn(t, shift=0.1 * m.LEFT) for t in tags], lag_ratio=0.3),
            m.FadeIn(total, shift=0.1 * m.DOWN),
        )
        self.say(
            "[Back at full size, the last block picks out the noise pattern again],",
            m.AnimationGroup(flash(u.path[5], MODEL), m.Transform(cur, self.cursor_on(u.up[0])), m.FadeOut(concat),
                             m.FadeIn(s_u2)),
        )
        self.say(
            "[and a last convolution turns thirty-two channels into one]: [the noise guess, epsilon hat], "
            "[the same size as the input].",
            flash(self.out_arrow, MODEL, 8),
            m.AnimationGroup(m.FadeOut(cur), m.Create(e_hi)),
            m.FadeIn(same, shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.play(m.FadeOut(m.Group(s_u2, e_hi, same, tags, total)), run_time=0.6)

    # ------------------------------------------------------------------ 5 · the time

    def time_in(self):
        self.next_section("5 · Telling the network the time", notes=(
            "One network serves every step ('Parameters are shared across time', Ho et al. 2020 §4), so t is an input. "
            "diffusion_reference.timestep_embedding(t, 32): 16 frequencies f_k = 10000^(−k/16), [sin(t·f), cos(t·f)] "
            "(DDPM divides by half − 1; the digit model's code divides by half and puts the cosines first: same numbers, "
            "other order). Rows: t = 0, 500, 501, 999 (cells: −1 dark … +1 bright BLUE). Then Linear 32 → 128, SiLU, "
            "Linear 128 → 128 (time1, time2), plus an 11 × 128 label table (index 10 = 'no label', used for "
            "classifier-free guidance in Episode 14), summed, SiLU, and fed to all 17 residual blocks, where Linear "
            "128 → C adds one number per channel between the two convolutions (mac_runs §3; gpu/models_mlx.py). "
            "ADM later used scale-and-shift (AdaGN) instead (notes §8)."))
        u = self.u
        lefts = -5.15
        cell_w, gap_h = 4.8 / 32, 0.36
        ys = [-1.9, -2.24, -2.58, -2.92]
        rows = m.VGroup(*[heat_row(EMB_ROWS[i], lefts, y, cell_w, gap_mid=gap_h) for i, y in enumerate(ys)])
        labs = m.VGroup(*[code(f"t = {t}", 20, TIME).move_to([0, y, 0]).align_to([-6.45, 0, 0], m.LEFT)
                          for t, y in zip(T_ROWS, ys)])
        half_w = 16 * cell_w
        sin_x, cos_x = lefts + half_w / 2, lefts + half_w + gap_h + half_w / 2
        y_u = -3.3
        names = m.VGroup(code("sin", 20, INK).move_to([sin_x, y_u, 0]), code("cos", 20, INK).move_to([cos_x, y_u, 0]))
        fasts, slows = m.VGroup(), m.VGroup()
        for x0 in (lefts, lefts + half_w + gap_h):
            fasts.add(code("fast", 20, TIME).move_to([0, y_u, 0]).align_to([x0 + 0.02, 0, 0], m.LEFT))
            slows.add(code("slow", 20, MUTED).move_to([0, y_u, 0]).align_to([x0 + half_w - 0.02, 0, 0], m.RIGHT))
        for w in (*names[1:], *fasts, *slows):
            sit(w, baseline(names[0]))
        under = m.VGroup(names, fasts, slows)
        assert all(a.get_right()[0] + 0.15 < b.get_left()[0] for a, b in ((fasts[0], names[0]), (names[0], slows[0]),
                                                                          (fasts[1], names[1]), (names[1], slows[1])))
        f1 = tex(r"\sin(t f_k),\ \cos(t f_k)", 36, TIME).move_to([3.4, -2.0, 0])
        f2 = tex(r"f_k = 10000^{-k/16},\quad k = 0 \dots 15", 30, INK).move_to([3.4, -2.68, 0])
        f3 = code("timestep_embedding(t, 32)", 20, MUTED).move_to([3.4, -3.25, 0])
        fast_box = m.SurroundingRectangle(m.VGroup(*rows[1][:3], *rows[2][:3]), buff=0.05).set_stroke(INK, 3)
        slow_box = m.SurroundingRectangle(m.VGroup(*rows[0][11:16], *rows[3][11:16]), buff=0.05).set_stroke(INK, 3)
        assert under[1].get_right()[0] < f2.get_left()[0] - 0.2 and f2.get_right()[0] < 6.55

        # phase B: the pipeline into the bus
        row500 = heat_row(EMB_ROWS[1], -6.3, STRIP_Y + 0.15, 2.0 / 32, cell_h=0.3, gap_mid=0.06)
        lab500 = code("t = 500", 20, TIME).next_to(row500, m.UP, buff=0.14).align_to(row500, m.LEFT)
        cap500 = cite("32 sines and cosines").next_to(row500, m.DOWN, buff=0.16).align_to(row500, m.LEFT)
        mlp = m.VGroup(rounded(2.7, 0.86, MODEL, AMBER_FILL, radius=0.14, stroke=2.5),
                       m.VGroup(text("small network", 20, INK, weight=m.BOLD), code("32 → 128 → 128", 20, INK))
                       .arrange(m.DOWN, buff=0.08))
        mlp[1].move_to(mlp[0])
        mlp.move_to([-2.3, STRIP_Y + 0.15, 0])
        assert mlp[1].width < mlp[0].width - 0.3 and mlp.get_left()[0] > row500.get_right()[0] + 0.45
        add = plus_node(TIME, 0.22).move_to([0, STRIP_Y + 0.15, 0])
        a1 = arrow(row500.get_right(), mlp.get_left(), TIME, buff=0.12)
        a2 = arrow(mlp.get_right(), add.get_left(), TIME, buff=0.1)
        chips = m.VGroup()
        for k in range(NUM_CLASSES + 1):
            name = str(k) if k < NUM_CLASSES else "none"
            hot = k == 7
            chips.add(labelled_chip(name, PROMPT, PINK_FILL if hot else CARD, 20, 0.44, 0.22, mono=True))
        chips.arrange(m.RIGHT, buff=0.06).move_to([0, STRIP_Y + 0.15, 0]).align_to([1.0, 0, 0], m.LEFT)
        for k, c in enumerate(chips):
            if k != 7:
                c[0].set_stroke(PROMPT, 1.5, opacity=0.55)
                c[1].set_color(MUTED)
        a3 = arrow(chips.get_left(), add.get_right(), PROMPT, buff=0.1)
        tab_cap = cite("label table: 11 rows × 128").next_to(chips, m.DOWN, buff=0.16).align_to(chips, m.LEFT)
        none_cap = text("Episode 14", 20, PROMPT).next_to(chips[-1], m.DOWN, buff=0.16)
        sit(none_cap, baseline(tab_cap))
        if none_cap.get_left()[0] < tab_cap.get_right()[0] + 0.25:
            none_cap.align_to(chips[-1], m.RIGHT)
        assert chips.get_right()[0] < 6.55 and none_cap.get_left()[0] > tab_cap.get_right()[0] + 0.2
        rise = m.Line(add.get_top(), [0, BUS_Y, 0]).set_stroke(TIME, 4)
        blocks = list(u.down) + [u.mid] + list(u.up)
        xs = [b.get_x() for b in blocks]
        bus = m.Line([min(xs), BUS_Y, 0], [max(xs), BUS_Y, 0]).set_stroke(TIME, 4)
        ticks = m.VGroup(*[m.Line([b.get_x(), BUS_Y, 0], [b.get_x(), b.get_bottom()[1] - 0.04, 0]).set_stroke(TIME, 3)
                           for b in blocks])
        bus_lab = m.VGroup(text(f"into all {N_RES}", 20, INK), text("residual blocks", 20, INK))
        bus_lab.arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT).next_to(bus, m.RIGHT, buff=0.25)
        bus_lab.shift((BUS_Y - 0.1 - bus_lab.get_bottom()[1]) * m.UP)
        assert bus_lab.get_right()[0] < 6.55
        pulses = m.LaggedStart(*[flash(t, TIME, 8, 0.9, 0.8) for t in ticks], lag_ratio=0.12)
        glow_blocks = m.AnimationGroup(*[b.animate(rate_func=m.there_and_back, run_time=1.4).set_stroke(TIME, 4)
                                         for b in blocks])
        pipe = m.Group(row500, lab500, cap500, mlp, add, a1, a2, chips, a3, tab_cap, none_cap, rise)

        # phase C: inside one residual block
        y_r = STRIP_Y + 0.12
        blk_a = labelled_chip("norm · SiLU · conv 3×3", MODEL, AMBER_FILL, 20, 0.5, 0.4)
        blk_b = labelled_chip("norm · SiLU · conv 3×3", MODEL, AMBER_FILL, 20, 0.5, 0.4)
        blk_a.move_to([-4.15, y_r, 0])
        p_t = plus_node(TIME, 0.2).move_to([-2.0, y_r, 0])
        blk_b.move_to([0.15, y_r, 0])
        p_s = plus_node(INK, 0.2).move_to([2.25, y_r, 0])
        h_in = code("h", 22, INK).move_to([-6.25, y_r, 0])
        r1 = arrow(h_in.get_right(), blk_a.get_left(), MUTED, buff=0.1)
        r2 = arrow(blk_a.get_right(), p_t.get_left(), MUTED, buff=0.08)
        r3 = arrow(p_t.get_right(), blk_b.get_left(), MUTED, buff=0.08)
        r4 = arrow(blk_b.get_right(), p_s.get_left(), MUTED, buff=0.08)
        drop = m.Arrow([p_t.get_x(), BUS_Y, 0], p_t.get_top(), buff=0.04, color=TIME, stroke_width=4,
                       max_tip_length_to_length_ratio=0.2)
        drop_lab = code("one number per channel", 20, TIME).next_to(drop, m.RIGHT, buff=0.15)
        drop_lab.set_y((STRIP_Y + STRIP_H / 2 + blk_b.get_top()[1]) / 2)
        y_s = y_r - 0.62
        short = m.VMobject().set_points_as_corners([[-5.95, y_r, 0], [-5.95, y_s, 0], [p_s.get_x(), y_s, 0],
                                                    [p_s.get_x(), p_s.get_bottom()[1] - 0.05, 0]]).set_stroke(MUTED, 3)
        short_lab = cite("shortcut").move_to([-1.9, y_s - 0.25, 0])
        anat_cap = m.VGroup(text("inside each", 22, INK, weight=m.BOLD), text("residual block", 22, INK, weight=m.BOLD))
        anat_cap.arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT).move_to([0, y_r, 0]).align_to([3.1, 0, 0], m.LEFT)
        anat = m.VGroup(h_in, r1, blk_a, r2, p_t, r3, blk_b, r4, p_s, short, short_lab, anat_cap)
        assert drop_lab.get_bottom()[1] > blk_b.get_top()[1] and short_lab.get_bottom()[1] > -3.45

        # one network, every step: the same 7 at four steps (the run's schedule, Episode 5's noise)
        noisy = m.Group()
        for k, t in enumerate((50, 300, 600, 999)):
            ab = ALPHA_BAR[t + 1]
            img = picture(tint(np.clip(np.sqrt(ab) * X0 + np.sqrt(1 - ab) * EPS, -1, 1), side_colour(t), -1, 1), 0.92)
            fr = framed(img, EDGE).move_to([-5.2 + 1.15 * k, STRIP_Y + 0.17, 0])
            lab = code(f"t={t}", 20, TIME).next_to(fr, m.DOWN, buff=0.12)
            noisy.add(m.Group(fr, lab))
        for g in noisy[1:]:
            sit(g[1], baseline(noisy[0][1]))
        one = labelled_chip("one set of weights", MODEL, AMBER_FILL, 22, 0.56, 0.45).move_to([1.45, STRIP_Y + 0.17, 0])
        to_one = arrow([noisy.get_right()[0] + 0.1, one.get_y(), 0], one.get_left(), MUTED, buff=0.1)
        sig = code("eps_model(x_t, t)", 22, INK, t2c={r"\bt\b": TIME, r"eps_model": TEAL}).move_to([4.75, STRIP_Y + 0.17, 0])
        sig_cap = text("t is an input", 20, MUTED).next_to(sig, m.DOWN, buff=0.16)
        one_cap = text("every block must know t", 20, MUTED).next_to(one, m.DOWN, buff=0.16)
        sit(sig_cap, baseline(one_cap))
        shared = m.Group(noisy, one, to_one, sig, sig_cap, one_cap)
        assert sig.get_left()[0] > one.get_right()[0] + 0.3 and sig.get_right()[0] < 6.4
        assert noisy.get_bottom()[1] > STRIP_Y - STRIP_H / 2 + 0.08

        self.say(
            "[One input is still missing]: [the step, t].",
            m.AnimationGroup(self.new_header("05 · time", "Telling the network the time"),
                             m.FadeOut(m.VGroup(self.enc, self.dec, self.mid_lab))),
            m.AnimationGroup(m.Circumscribe(self.in_lab[1], color=TIME, buff=0.08),
                             m.LaggedStart(*[m.FadeIn(g, shift=0.1 * m.UP) for g in noisy], lag_ratio=0.25)),
        )
        self.say(
            "[The same weights serve all thousand steps], [so every block must be told how noisy its input is].",
            m.AnimationGroup(u.animate(rate_func=m.there_and_back, run_time=1.2).set_stroke(TIME),
                             m.GrowArrow(to_one), m.FadeIn(one)),
            m.AnimationGroup(m.FadeIn(one_cap), m.FadeIn(sig, shift=0.1 * m.LEFT), m.FadeIn(sig_cap)),
        )
        self.say(
            "[First, t becomes thirty-two sines and cosines]: [fast waves on the left], [slow waves on the right].",
            m.AnimationGroup(m.FadeOut(shared),
                             m.LaggedStart(*[m.FadeIn(m.VGroup(lb, r), shift=0.1 * m.RIGHT) for lb, r in zip(labs, rows)],
                                           lag_ratio=0.25), m.FadeIn(names), m.Write(f1), m.FadeIn(f2), m.FadeIn(f3)),
            m.FadeIn(fasts, shift=0.1 * m.UP),
            m.FadeIn(slows, shift=0.1 * m.UP),
        )
        self.say(
            "[Neighbouring steps differ only in the fast waves]; [the slow ones tell roughly where we are, "
            "from zero to a thousand].",
            m.Create(fast_box),
            m.AnimationGroup(m.FadeOut(fast_box), m.Create(slow_box)),
        )
        self.say(
            "[A small network turns them into a hundred twenty-eight numbers].",
            m.AnimationGroup(m.FadeOut(m.VGroup(slow_box, labs[0], labs[2], labs[3], rows[0], rows[2], rows[3], under,
                                                f1, f2, f3, labs[1])),
                             m.ReplacementTransform(rows[1], row500), m.FadeIn(lab500), m.FadeIn(cap500),
                             m.GrowArrow(a1), m.FadeIn(mlp, shift=0.1 * m.RIGHT)),
        )
        self.say(
            "[The label gets its own hundred twenty-eight], [from a table of eleven rows]: "
            "[ten digits, plus one meaning no label, for episode fourteen].",
            m.AnimationGroup(m.GrowArrow(a2), m.FadeIn(add)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(c, shift=0.1 * m.UP) for c in chips], lag_ratio=0.08),
                             m.FadeIn(tab_cap)),
            m.AnimationGroup(m.Circumscribe(chips[-1], color=PROMPT, buff=0.06), m.FadeIn(none_cap), m.GrowArrow(a3)),
        )
        self.say(
            "[The two are added], [and the sum is wired into all seventeen residual blocks].",
            m.AnimationGroup(m.Circumscribe(add, color=TIME, buff=0.08), m.Create(rise)),
            m.Succession(m.AnimationGroup(m.Create(bus), m.Create(ticks), m.FadeIn(bus_lab), run_time=1.0),
                         m.AnimationGroup(pulses, glow_blocks)),
        )
        self.say(
            "[Inside each block, it becomes one number per channel], [added between the two convolutions].",
            m.AnimationGroup(m.FadeOut(pipe), m.FadeIn(anat, shift=0.1 * m.UP)),
            m.AnimationGroup(m.GrowArrow(drop), m.FadeIn(drop_lab), m.Circumscribe(p_t, color=TIME, buff=0.08)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 6 · the weights

    def weights(self):
        self.next_section("6 · Where the weights are", notes=(
            f"Parameters recounted from the config and asserted equal to mac_runs.json architecture.ddpm.params_by_part: "
            f"{PARAMS:,} in all. Bars (millions, in the order of the U): "
            + ", ".join(f"{a} {b} {v / 1e6:.2f}" for a, b, v in BY_PART)
            + f"; the in/out convolutions ({EDGES} parameters) are not drawn. The 7×7 down level, the middle and the 7×7 up "
            f"level hold {100 * SHARE_7:.1f}% ({100 * SHARE_7_STRICT:.1f}% without the 7×7 up level's upsampling conv, "
            "which runs at 14×14): 'more than three quarters'. A 3×3 conv weight is applied at every position: 49 at "
            "7×7, 784 at 28×28. Sizes for scale: DDPM's CIFAR-10 U-Net 35.7 M (Ho et al. 2020 App. B); Stable Diffusion "
            "1.x U-Net 860 M (CompVis README 2022; 859,520,964 counted, notes §8); the lighthouse model, SD 2.1-base, "
            f"U-Net {LIGHT_M} M (mac_runs §5, measured on this Mac). Squares: area ∝ parameters. Larger models moved to "
            "transformers (Episode 18; SDXL's U-Net is 2.6 B, notes §8)."))
        stat_n = title_text(f"{PARAMS:,}", 64, INK)
        stat_l = text("parameters in the digit model's U-Net", 24, MUTED)
        stat_g = m.VGroup(stat_n, stat_l).arrange(m.DOWN, buff=0.16).move_to([0, 2.0, 0])
        vals = [v / 1e6 for *_, v in BY_PART]
        hot = [2, 3, 4]
        bars = Bars(vals, top=1.9, width=8.6, height=2.3, colors=[MODEL if i in hot else MUTED for i in range(8)],
                    gap=0.32, show_values=False, center=(-1.25, -0.95, 0))
        nums = m.VGroup(*[code(f"{v:.2f}", 20, INK).move_to([x, bars.level(v) + 0.22, 0]) for x, v in zip(bars.xs, vals)])
        labs = m.VGroup()
        for x, (a, b, _) in zip(bars.xs, BY_PART):
            top = code(a, 20, INK).move_to([x, bars.y0 - 0.3, 0])
            bot = text(b, 20, MUTED).move_to([x, bars.y0 - 0.64, 0])
            labs.add(m.VGroup(top, bot))
        for lab in labs[1:]:
            sit(lab[0], baseline(labs[0][0]))
            sit(lab[1], baseline(labs[0][1]))
        unit = cite("millions of parameters, in the order of the U").move_to([-1.25, -3.3, 0])
        left_x = bars.xs[2] - bars.bar_w / 2
        right_x = bars.xs[4] + bars.bar_w / 2
        brace = m.Brace(m.Line([left_x, 0, 0], [right_x, 0, 0]), m.UP, buff=0).set_color(MODEL)
        brace.move_to([(left_x + right_x) / 2, bars.level(vals[4]) + 0.62, 0])
        share = code(f"{100 * SHARE_7:.0f}% in the 7×7 levels", 22, INK, weight=m.BOLD).next_to(brace, m.UP, buff=0.12)
        assert share.get_top()[1] < stat_l.get_bottom()[1] - 0.1

        g7, g28 = m.Square(1.15), m.Square(1.15)
        gx = 5.35
        g7.move_to([gx, 0.2, 0]).set_stroke(EDGE, 2)
        g28.move_to([gx, -1.95, 0]).set_stroke(EDGE, 2)
        gl7 = grid_lines(g7, 7, MODEL, 1.6, 0.9)
        gl28 = grid_lines(g28, 28, MUTED, 0.8, 0.6)
        n7 = code(f"7×7 → {POS_7}", 20, INK).next_to(g7, m.DOWN, buff=0.14)
        n28 = code(f"28×28 → {POS_28}", 20, INK).next_to(g28, m.DOWN, buff=0.14)
        per = text("positions per weight", 20, MUTED).next_to(g7, m.UP, buff=0.14)
        assert per.get_top()[1] < stat_l.get_bottom()[1] - 0.05 and n7.get_right()[0] < 6.55
        part_one = m.VGroup(bars, nums, labs, unit, brace, share, g7, g28, gl7, gl28, n7, n28, per)

        # sizes for scale: area ∝ parameters
        base_y = -2.75
        sd_side = 3.3
        side = lambda p: sd_side * np.sqrt(p / (SD1_M * 1e6))  # noqa: E731
        sq = []
        for p, x, col in ((PARAMS, -5.7, MODEL), (DDPM_M * 1e6, -3.45, MUTED), (SD1_M * 1e6, 0.0, MUTED)):
            s = side(p)
            sq.append(m.Square(s).set_stroke(col, 2.5).set_fill(FILLS[col], 1).move_to([x, base_y + s / 2, 0]))
        l_ours = m.VGroup(text("ours", 22, INK, weight=m.BOLD), code("3.94 M", 20, INK)).arrange(m.DOWN, buff=0.1)
        l_ddpm = m.VGroup(text("DDPM, small photos", 22, INK, weight=m.BOLD), code(f"{DDPM_M} M", 20, INK),
                          cite("Ho et al., 2020")).arrange(m.DOWN, buff=0.1)
        l_sd = m.VGroup(text("Stable Diffusion 1.x", 22, INK, weight=m.BOLD), code(f"{SD1_M:.0f} M", 20, INK),
                        cite("CompVis, 2022")).arrange(m.DOWN, buff=0.1)
        l_ours.next_to(sq[0], m.UP, buff=0.2)
        l_ddpm.next_to(sq[1], m.UP, buff=0.2)
        l_sd.next_to(sq[2], m.UP, buff=0.2)
        assert l_ours.get_right()[0] < l_ddpm.get_left()[0] - 0.15 and l_ddpm.get_right()[0] < sq[2].get_left()[0] - 0.15, (l_ours.get_right(), l_ddpm.get_left(), l_ddpm.get_right(), sq[2].get_left())
        base = m.Line([-6.3, base_y, 0], [1.95, base_y, 0]).set_stroke(FAINT, 2)
        area = cite("area ∝ parameters").move_to([-2.3, base_y - 0.35, 0])
        light = picture(LIGHT_PNG, 1.4)
        light_f = framed(light, EDGE).move_to([3.1, -1.5, 0])
        l_light = m.VGroup(text("the lighthouse", 22, INK, weight=m.BOLD), code(f"{LIGHT_M} M", 20, INK),
                           cite("SD 2.1-base U-Net"), cite("counted on this Mac")).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        l_light.next_to(light_f, m.RIGHT, buff=0.25).align_to(light_f, m.UP)
        gen = cite("generated on this Mac\n50 steps · seed 0").next_to(light_f, m.DOWN, buff=0.14).align_to(light_f, m.LEFT)
        times = title_text(f"×{LIGHT_M * 1e6 / PARAMS:.0f}", 56, INK).move_to([4.4, 1.3, 0])
        times_l = text("the lighthouse U-Net vs ours", 20, MUTED).next_to(times, m.DOWN, buff=0.12)
        dit = pill("bigger still: transformers · Episode 18", TEAL, TEAL_FILL, 20).move_to([3.7, 2.35, 0])
        assert dit.get_right()[0] < 6.5
        assert l_light.get_right()[0] < 6.6 and gen.get_bottom()[1] > -3.5 and gen.get_right()[0] < 6.6, (l_light.get_right(), gen.get_corner(m.DR))

        self.say(
            "[Now count the weights]: [three million, nine hundred forty thousand, eight hundred and one].",
            self.new_header("06 · size", "Where the weights are"),
            m.FadeIn(stat_g, shift=0.15 * m.UP),
        )
        self.say(
            "[Laid out block by block], [most of them sit at the bottom]: [more than three quarters work at seven by seven].",
            m.AnimationGroup(m.FadeIn(bars.zero), m.FadeIn(labs), m.FadeIn(unit)),
            m.AnimationGroup(*[m.GrowFromEdge(b, m.DOWN) for b in bars.bars], m.FadeIn(nums)),
            m.AnimationGroup(m.GrowFromCenter(brace), m.FadeIn(share, shift=0.1 * m.UP)),
        )
        self.say(
            "[That is where they are cheap]: [each one is used at forty-nine positions], [not seven hundred eighty-four].",
            m.FadeIn(per),
            m.AnimationGroup(m.FadeIn(g7), m.Create(gl7), m.FadeIn(n7)),
            m.AnimationGroup(m.FadeIn(g28), m.Create(gl28), m.FadeIn(n28)),
        )
        self.say(
            "[For scale]: [the DDPM paper's network for small photos had about thirty-six million], "
            "[and Stable Diffusion's has eight hundred sixty million].",
            m.AnimationGroup(m.FadeOut(part_one), m.FadeOut(stat_g), m.Create(base), m.FadeIn(area),
                             m.GrowFromEdge(sq[0], m.DOWN), m.FadeIn(l_ours)),
            m.AnimationGroup(m.GrowFromEdge(sq[1], m.DOWN), m.FadeIn(l_ddpm)),
            m.AnimationGroup(m.GrowFromEdge(sq[2], m.DOWN, run_time=1.4), m.FadeIn(l_sd)),
        )
        self.say(
            "[The lighthouse model's denoiser, run on this Mac, has eight hundred sixty-six million]: [about two hundred twenty times the digit model]. "
            "[Bigger still, transformers took over].",
            m.AnimationGroup(m.FadeIn(light_f, shift=0.1 * m.UP), m.FadeIn(l_light), m.FadeIn(gen)),
            m.AnimationGroup(m.FadeIn(times, shift=0.1 * m.UP), m.FadeIn(times_l)),
            m.FadeIn(dit, shift=0.1 * m.DOWN),
        )
        self.wait(0.5)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes=(
            "Context (the encoder and the middle see the whole digit), detail (skip connections carry the full-size maps "
            "across), time (the step and label enter all 17 residual blocks). Module 2: Episode 5 (predict the noise), 6 "
            "(the score), 7 (DDPM sampling, 1,000 calls), 8 (the U-Net). Next: Module 3, Faster Sampling, Episode 9."))
        halo = glow(MODEL, radius=4.0)
        words = m.VGroup(*[title_text(w, 58, c) for w, c in (("Context", MODEL), ("Detail", DATA), ("Time", TIME))])
        subs = m.VGroup(*[code(s, 20, MUTED) for s in ("shrink to see it all", "skips carry the pixels",
                                                       "t enters every block")])
        words.arrange(m.RIGHT, buff=1.7)
        on_baseline(words)
        for w, s in zip(words, subs):
            s.set_x(w.get_x())
        line_y = min(w.get_bottom()[1] for w in words) - 0.45
        for s in subs:
            sit(s, line_y)
        mid = baseline(words[0]) + 0.3
        dots = m.VGroup(*[m.Dot([(a.get_right()[0] + b.get_left()[0]) / 2, mid, 0], radius=0.07, color=MUTED)
                          for a, b in zip(words, words[1:])])
        row = m.VGroup(words, dots, subs).move_to(1.35 * m.UP)
        assert row.width < 12.6 and all(a.get_right()[0] + 0.3 < b.get_left()[0] for a, b in zip(subs, subs[1:]))
        mod = m.VGroup(*[pill(s, MODEL, AMBER_FILL, 20) for s in ("5 · predict the noise", "6 · the score",
                                                                   "7 · a thousand steps", "8 · the U-Net")])
        mod.arrange(m.RIGHT, buff=0.25).move_to([0, -0.85, 0])
        mod_t = text("Module 2 · Learning to denoise", 24, INK, weight=m.BOLD).next_to(mod, m.UP, buff=0.3)
        assert mod.width < 12.8
        mascot = pixel(0.42)
        nxt = pill("NEXT · MODULE 3 · FASTER SAMPLING", TIME, BLUE_FILL, 20, mono=True)
        foot = m.VGroup(mascot, nxt).arrange(m.RIGHT, buff=0.3).move_to([0, -2.4, 0])

        self.say(
            "[To sum up]: [context], [detail], [and time].",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.3 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(dots[0]), m.FadeIn(words[1], shift=0.3 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(dots[1]), m.FadeIn(words[2], shift=0.3 * m.UP), m.FadeIn(subs[2])),
        )
        self.say(
            "[The network sees the whole digit through its bottleneck], [the fine detail through its skips], "
            "[and the step in every block].",
            m.Circumscribe(words[0], color=MODEL, buff=0.1),
            m.Circumscribe(words[1], color=DATA, buff=0.1),
            m.Circumscribe(words[2], color=TIME, buff=0.1),
        )
        self.say(
            "[That closes module two]: [what the network predicts], [why that points toward the data], "
            "[how sampling uses it], [and what it looks like inside].",
            m.FadeIn(mod_t, shift=0.1 * m.UP),
            m.FadeIn(mod[0], shift=0.1 * m.UP),
            m.FadeIn(mod[1], shift=0.1 * m.UP),
            m.FadeIn(mod[2], shift=0.1 * m.UP),
            m.FadeIn(mod[3], shift=0.1 * m.UP),
        )
        self.say(
            "[From here, module three keeps this network], [and makes sampling much faster].",
            m.FadeIn(mascot, shift=0.2 * m.UP),
            m.FadeIn(nxt, shift=0.2 * m.UP),
        )
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, mod, mod_t, foot)], run_time=0.9)
