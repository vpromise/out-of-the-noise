"""Out of the Noise · Episode 17 · Latent Diffusion (Module 5 · Scaling Up — the module's first episode)

Diffuse in a small compressed space, then decode. An autoencoder (a VAE with a tiny KL penalty, trained with
reconstruction, perceptual and adversarial losses) squeezes a 512×512×3 picture into a 64×64×4 latent — 8× per side,
48× fewer numbers — the denoiser does all its steps there, and the decoder paints the pixels once at the end.
Running example: the lighthouse, generated on this Mac (Stable Diffusion 2.1-base in MLX, research/experiments/mac_runs.md
§5): the 30-step run's latents (arrays/lighthouse_latents_30steps.npz: x_t [31, 64, 64, 4], x0_hat [30, 64, 64, 4], t, σ)
and its decoded frames (images/lighthouse_frames30/xt_*.png, x0hat_*.png); the 50-step seed-0 picture and its VAE
encoding (arrays/lighthouse_vae_latent.npz). Episodes 15 (text encoder, cross-attention) and 16 (SDEdit encodes the
source first) are linked by number, not re-taught; Episode 23 times the whole pipeline.

Every number on screen is computed below from mac_runs.json (measured(...)), the run's arrays, or cited with its
source and asserted. Nothing is run or downloaded.

Sources (research/notes.md §17, §8, §23, Appendix A rows 17, Appendix C row 17; read October 2026):
- Rombach, Blattmann, Lorenz, Esser & Ommer, "High-Resolution Image Synthesis with Latent Diffusion Models",
  arXiv:2112.10752 (20 Dec 2021; CVPR 2022). f = 8, c = 4: 512×512×3 → 64×64×4 (48×). App. G: KL weight ~10⁻⁶; latents
  rescaled to unit standard deviation (SD config scale_factor 0.18215 ⇒ σ̂ ≈ 5.49). Table 8: KL f = 8, c = 4 autoencoder
  PSNR 24.19 (R-FID 0.90). Table 18 (ImageNet 256, class-conditional): ADM-G FID 4.59 at 962 V100-days; LDM-4-G
  (cfg 1.5) FID 3.60 at 271 V100-days (3.5× less training compute).
- Stable Diffusion releases (notes §17 table, [DATED]): v1.4 public release 22 Aug 2022 (CompVis / Stability AI);
  v1.5 Oct 2022 (RunwayML); 2.0 24 Nov 2022; 2.1 7 Dec 2022; SDXL 1.0 26 Jul 2023 (U-Net 2.6 B); SD3 Medium 12 Jun 2024
  (MM-DiT, 16-channel VAE, §11). SD 1.5 weights counted from the safetensors (notes §8, §23): U-Net 859,520,964; CLIP
  text encoder 123,060,480; VAE 83,653,863 (decoder 49,490,199); text + U-Net + decoder 1,032,071,643 × 2 B = 2.06 GB.
- Training data: subsets of LAION-5B (laion2B-en, laion-high-resolution, laion-aesthetics v2 5+; CompVis README);
  LAION-5B = 5.85 billion CLIP-filtered image–text pairs (Schuhmann et al., 2022). Detail in Episode 20.
- Measured on this Mac (mac_runs §5): SD 2.1-base U-Net 865,910,724, OpenCLIP-H text encoder 340,387,840, VAE 83,653,863
  (encoder 34,163,664, decoder 49,490,199); latent 64×64×4 confirmed; VAE round trip 38.1 dB PSNR on the seed-0 picture
  (flattering: the picture is itself a VAE output); timing (median of 3): text 0.017 s, 50 × 2 U-Net passes 7.04 s,
  VAE decode 0.265 s, total 7.32 s.

Illustrative (labelled or noted): the tower boxes (placed from the picture: the lamp is its reddest pixel); the funnel
shapes; the colour of the latent tiles while sampling (VIOLET → TEAL by the signal's share of the variance, 1/(1 + σ²)).

    VOICE=draft manimgx inspect scene.py      # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403
from diffusion_reference import *  # noqa: E402,F403

EP_DIR = Path(__file__).resolve().parent          # set AFTER the import: diffusion_reference exports HERE (the series)
PREFIX = "ep17"
ARR = EXPERIMENTS / "arrays"
IMG = EXPERIMENTS / "images"
PX = 135                                          # pixels per scene unit at 1080p
HL = lambda *keys: measured("mac_runs", "headline", *keys)  # noqa: E731
LH = measured("mac_runs", "lighthouse")

# ---------------------------------------------------------------------------- shapes: 512×512×3 → 64×64×4

SIDE, LAT_SIDE, LAT_C = HL("lighthouse_image_px"), HL("lighthouse_latent_h"), HL("lighthouse_latent_c")
assert (SIDE, LAT_SIDE, HL("lighthouse_latent_w"), LAT_C) == (512, 64, 64, 4)
FACTOR = SIDE // LAT_SIDE
N_PIX, N_LAT = SIDE * SIDE * 3, LAT_SIDE * LAT_SIDE * LAT_C
assert FACTOR == 8 and (N_PIX, N_LAT) == (786_432, 16_384)                       # "eight times", the two counts
assert N_PIX / N_LAT == 48 == HL("lighthouse_vae_compression_ratio")             # "forty-eight times fewer"
VAE_RUN = LH["vae"]
assert VAE_RUN["image_shape"] == [512, 512, 3] and VAE_RUN["latent_shape"] == [64, 64, 4]

PSNR = HL("lighthouse_vae_roundtrip_psnr_db")
assert f"{PSNR:.1f}" == "38.1" and abs(VAE_RUN["roundtrip_psnr_db"] - PSNR) < 0.01  # "thirty-eight decibels"
RMS = 10 ** (-PSNR / 20)                          # root-mean-square error as a share of the 0–255 range
assert f"{100 * RMS:.1f}" == "1.2"                                                  # "about one percent"
PSNR_PAPER = 24.19                                # Rombach et al. 2022, Table 8: KL, f = 8, c = 4 (notes §17)
assert round(PSNR_PAPER) == 24                                                      # "about twenty-four decibels"

# ---------------------------------------------------------------------------- the networks' sizes

PARTS = LH["models"]["sd2-community/stable-diffusion-2-1-base"]["parts"]
UNET_P, TEXT_P = PARTS["unet"]["params"], PARTS["text_encoder"]["params"]
VAE_P, ENC_P, DEC_P = PARTS["vae"]["params"], PARTS["vae"]["encoder_params"], PARTS["vae"]["decoder_params"]
M = lambda p: f"{p / 1e6:.1f}"  # noqa: E731
assert M(UNET_P) == f"{HL('lighthouse_unet_params_M'):.1f}" == "865.9"              # "eight hundred sixty-six million"
assert M(TEXT_P) == f"{HL('lighthouse_text_encoder_params_M'):.1f}" == "340.4"      # "three hundred forty million"
assert M(VAE_P) == f"{HL('lighthouse_vae_params_M'):.1f}" == "83.7"                 # "eighty-four million"
assert (M(ENC_P), M(DEC_P)) == ("34.2", "49.5") and ENC_P + DEC_P == VAE_P
SD15_UNET, SD15_TEXT, SD15_VAE, SD15_DEC = 859_520_964, 123_060_480, 83_653_863, 49_490_199   # notes §8, §23
assert SD15_VAE == VAE_P and SD15_DEC == DEC_P                                      # the same autoencoder
SD15_GEN = SD15_TEXT + SD15_UNET + SD15_DEC
assert SD15_GEN == 1_032_071_643 and f"{SD15_GEN * 2 / 1e9:.2f}" == "2.06"          # "about two gigabytes"
assert round(SD15_UNET / 1e6) == 860 and round(SD15_TEXT / 1e6) == 123 and round(SD15_VAE / 1e6) == 84
SDXL_UNET = 2_567_463_684                          # SDXL safetensors (notes §17): "2.6B parameters in the UNet"
assert f"{SDXL_UNET / 1e9:.1f}" == "2.6"

# ---------------------------------------------------------------------------- the latent scale factor

SCALE = 0.18215                                   # SD's scale_factor (config); the run's encode() applies it
VAE_NPZ = np.load(ARR / "lighthouse_vae_latent.npz")
assert "0.18215" in str(VAE_NPZ["note"]) and "lighthouse_sd21_cfg7.5_steps50_seed0.png" in str(VAE_NPZ["note"])
VAE_LAT = VAE_NPZ["latent"].astype(np.float64)
assert VAE_LAT.shape == (64, 64, 4)
LAT_STD = float(VAE_LAT.std())
assert abs(LAT_STD - VAE_RUN["latent_std"]) < 1e-4
RAW_STD = LAT_STD / SCALE
assert (f"{RAW_STD:.1f}", f"{LAT_STD:.1f}") == ("3.7", "0.7")                       # "three point seven … zero point seven"
assert f"{1 / SCALE:.2f}" == "5.49"                                                 # the paper's σ̂ (notes §17, derived)

# ---------------------------------------------------------------------------- the 30-step run (Episodes 1 and 17)

RUN = LH["frames"]
STEPS, CFG_W, SEED = RUN["steps"], RUN["cfg"], RUN["seed"]
assert (STEPS, CFG_W, SEED) == (30, 7.5, 0) and RUN["latent_shape"] == [64, 64, 4]
LAT = np.load(ARR / "lighthouse_latents_30steps.npz")
assert "30 steps, cfg 7.5, seed 0" in str(LAT["note"])
XT_LAT = LAT["x_t"].astype(np.float64)
X0_LAT = LAT["x0_hat"].astype(np.float64)
T_RUN, SIG_RUN = LAT["t"].astype(np.float64), LAT["sigma"].astype(np.float64)
assert XT_LAT.shape == (STEPS + 1, 64, 64, 4) and X0_LAT.shape == (STEPS, 64, 64, 4) and len(T_RUN) == STEPS
assert T_RUN[0] == 1000 and abs(XT_LAT[0].std() - 1) < 0.02                         # starts as unit noise at t = 1000
T_SHOWN = [int(round(t)) for t in T_RUN] + [0]
SIG_SHOWN = list(SIG_RUN) + [0.0]
SIGNAL_SHARE = [1 / (1 + s * s) for s in SIG_SHOWN]                                # ᾱ = 1/(1 + σ²)
assert SIGNAL_SHARE[0] < 0.01 and SIGNAL_SHARE[-2] > 0.95 and SIGNAL_SHARE[-1] == 1.0
_corr = [np.corrcoef(VAE_LAT[..., c].ravel(), XT_LAT[STEPS][..., c].ravel())[0, 1] for c in range(4)]
assert min(_corr) > 0.98                          # the 30-step and 50-step latents are nearly the same picture


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))


FRAMES = IMG / "lighthouse_frames30"
XT_FR = [load_rgb(FRAMES / f"xt_{k:02d}.png") for k in range(STEPS + 1)]
X0_FR = [load_rgb(FRAMES / f"x0hat_{k:02d}.png") for k in range(STEPS)]
assert all(f.shape == (512, 512, 3) for f in XT_FR + X0_FR)
assert np.array_equal(XT_FR[STEPS], X0_FR[STEPS - 1])                               # the last guess is the picture
PIC50 = load_rgb(IMG / "lighthouse" / "lighthouse_sd21_cfg7.5_steps50_seed0.png")
assert PIC50.shape == (512, 512, 3)

# where the lamp is (the reddest pixel near the top) → the tower's column in the latent
_top = PIC50[:200].astype(int)
LAMP_R, LAMP_C = np.unravel_index((_top[..., 0] - (_top[..., 1] + _top[..., 2]) / 2).argmax(), _top.shape[:2])
assert (LAMP_R, LAMP_C) == (78, 333)
TOWER = (8, 38, 30, 8)                            # latent (row, col, rows, cols): the tower, read off the picture
assert TOWER[0] <= LAMP_R // 8 < TOWER[0] + TOWER[2] and TOWER[1] <= LAMP_C // 8 < TOWER[1] + TOWER[3]
CROP_R, CROP_C = (LAMP_R // 8 - 4) * 8, (LAMP_C // 8 - 4) * 8                      # a 64 × 64 pixel window, 8 × 8 cells
assert (CROP_R, CROP_C) == (40, 296) and CROP_R % 8 == 0 and CROP_C % 8 == 0
CROP_PIX = PIC50[CROP_R:CROP_R + 64, CROP_C:CROP_C + 64]
CROP_LAT = VAE_LAT[CROP_R // 8:CROP_R // 8 + 8, CROP_C // 8:CROP_C // 8 + 8]
assert CROP_PIX.shape == (64, 64, 3) and CROP_LAT.shape == (8, 8, 4)

# ---------------------------------------------------------------------------- why it is cheaper

POS_PIX, POS_LAT = SIDE * SIDE, LAT_SIDE * LAT_SIDE
assert (POS_PIX, POS_LAT, POS_PIX // POS_LAT) == (262_144, 4_096, 64)              # "sixty-four times fewer"
PAIRS_PIX, PAIRS_LAT = POS_PIX ** 2, POS_LAT ** 2
assert (f"{PAIRS_PIX / 1e9:.1f}", f"{PAIRS_LAT / 1e6:.1f}", PAIRS_PIX // PAIRS_LAT) == ("68.7", "16.8", 4096)
assert (49 ** 2, 784 ** 2) == (2_401, 614_656)                                      # Episode 8's attention count
ADM_DAYS, LDM_DAYS, ADM_FID, LDM_FID = 962, 271, 4.59, 3.60                        # Rombach et al. 2022, Table 18
assert f"{ADM_DAYS / LDM_DAYS:.1f}" == "3.5" and LDM_FID < ADM_FID                 # "three and a half times less"

TM = LH["timing"]["cfg7.5_50steps"]
TXT_S, DEN_S, DEC_S, TOT_S = TM["text_encoder_s"], TM["denoise_total_s"], TM["vae_decode_s"], TM["total_s"]
PASSES = TM["denoise_steps"] * TM["unet_passes_per_step"]
assert (TM["denoise_steps"], PASSES) == (50, 100)                                   # "fifty guided steps, a hundred passes"
assert (f"{TXT_S:.3f}", f"{DEN_S:.2f}", f"{DEC_S:.3f}", f"{TOT_S:.2f}") == ("0.017", "7.04", "0.265", "7.32")
assert abs(TXT_S + DEN_S + DEC_S - TOT_S) < 1e-3
SHARE = {k: 100 * v / TOT_S for k, v in (("text", TXT_S), ("den", DEN_S), ("dec", DEC_S))}
assert (f"{SHARE['den']:.1f}", f"{SHARE['dec']:.1f}", f"{SHARE['text']:.1f}") == ("96.2", "3.6", "0.2")
assert round(TOT_S, 2) == HL("lighthouse_total_s_50steps_cfg7.5") and round(TOT_S) == 7      # "about seven seconds"

# ---------------------------------------------------------------------------- Stable Diffusion's timeline (dated)

RELEASES = [  # (months after Dec 2021, date, name, above the axis?)   notes §17 releases table (1.5: Oct 2022, notes)
    (0, "Dec 2021", "the paper", False),
    (8, "22 Aug 2022", "open release", True),
    (12, "7 Dec 2022", "2.1 · our run", False),
    (19, "Jul 2023", "SDXL · 2.6 B", True),
    (30, "Jun 2024", "SD3 · transformer", False),
]

# ---------------------------------------------------------------------------- look


def cite(s, size=20):
    return text(s, size, MUTED)


FLAT = set("abcdefhiklmnorstuvwxzABCDEFGHIJKLMNOPRSTUVWXYZ0123456789")


def baseline(words):
    chars = [c for c in words.text if not c.isspace()]
    if len(words) == len(chars):
        bottoms = sorted(g.get_bottom()[1] for g, c in zip(words, chars) if c in FLAT)
        if bottoms:
            return bottoms[len(bottoms) // 2]
    return float(np.median([g.get_bottom()[1] for g in words]))


def sit(words, y):
    return words.shift((y - baseline(words)) * m.UP)


_TEX = {}


def texture_of(rgb, px, nearest=False):
    """An RGB array → an RGBA texture about px pixels wide: small arrays blown up with crisp square pixels, large
    ones resized smoothly (the engine samples, not averages, when it draws a texture smaller than it is)."""
    key = (id(rgb), px, nearest)
    if key in _TEX:
        return _TEX[key][1]
    h, w = rgb.shape[:2]
    if nearest:
        k = max(1, int(np.ceil(2 * px / w)))
        arr = np.repeat(np.repeat(rgb, k, axis=0), k, axis=1)
    elif px != w:
        arr = np.asarray(Image.fromarray(rgb).resize((px, max(1, round(px * h / w))),
                                                     Image.LANCZOS if px < w else Image.BICUBIC))
    else:
        arr = rgb
    out = np.concatenate([arr, np.full(arr.shape[:2] + (1,), 255, np.uint8)], axis=-1)
    out.flags.writeable = False
    _TEX[key] = (rgb, out)
    return out


def photo(rgb, height, nearest=None):
    """A picture `height` units tall from an RGB array (crisp pixels for small arrays)."""
    nearest = max(rgb.shape[:2]) <= 64 if nearest is None else nearest
    image = m.ImageMobject(texture_of(rgb, int(np.ceil(height * PX)), nearest))
    image.height = height
    return image


class Reel:
    """A picture that steps through real frames of a run: `.image`, `.show(k)`."""

    def __init__(self, frames, height, k=0, nearest=None):
        nearest = max(frames[0].shape[:2]) <= 64 if nearest is None else nearest
        px = int(np.ceil(height * PX))
        self.tex = [texture_of(f, px, nearest) for f in frames]
        self.image = m.ImageMobject(self.tex[k])
        self.image.height = height
        self.k = k

    def show(self, k):
        k = int(np.clip(round(k), 0, len(self.tex) - 1))
        if k != self.k:
            self.image.paint = self.image.paint.but(texture=self.tex[k])
            self.k = k


def drive(mob, a0, a1, hooks, run_time=1.4, rate=m.linear):
    """Animate a value from a0 to a1, calling every hook(value) each frame."""

    def step(_, a):
        v = a0 + (a1 - a0) * a
        for hook in hooks:
            hook(v)

    return m.UpdateFromAlphaFunc(mob, step, run_time=run_time, rate_func=rate)


def chan_rgb(lat, c, color, lo, hi):
    """One latent channel as a picture in one colour, stretched between lo and hi."""
    return tint(np.clip((lat[..., c] - lo) / (hi - lo), 0, 1), color, 0, 1)


VAE_LOHI = [tuple(np.percentile(VAE_LAT[..., c], [1, 99])) for c in range(4)]
XT_LOHI = [tuple(np.percentile(XT_LAT[STEPS][..., c], [1, 99])) for c in range(4)]
VAE_TILES = [chan_rgb(VAE_LAT, c, DATA, *VAE_LOHI[c]) for c in range(4)]
XT_TILES = [[chan_rgb(XT_LAT[k], c, mix(NOISE, DATA, SIGNAL_SHARE[k]), *XT_LOHI[c]) for k in range(STEPS + 1)]
            for c in range(4)]
CROP_LAT_RGB = chan_rgb(CROP_LAT, 0, DATA, *VAE_LOHI[0])


def funnel(center, squeeze=True, wide=2.2, narrow=0.95, length=1.25, label="", color=DATA):
    """An encoder (squeeze=True: wide → narrow) or decoder (narrow → wide), as in Episode 3 (TEAL here: the
    autoencoder is the picture side of the pipeline; AMBER stays the denoiser's)."""
    x, y = center[0], center[1]
    hl, hr = (wide, narrow) if squeeze else (narrow, wide)
    l, r = x - length / 2, x + length / 2
    shape = m.Polygon([l, y + hl / 2, 0], [r, y + hr / 2, 0], [r, y - hr / 2, 0], [l, y - hl / 2, 0],
                      stroke_color=color, stroke_width=2.5, fill_color=FILLS[color], fill_opacity=1)
    if not label:
        return m.VGroup(shape)
    words = text(label, 20, INK, weight=m.BOLD).move_to([x, y, 0])
    return m.VGroup(shape, words)


def tile_grid(images, center, side, gap=0.1, stroke=1.5):
    """Four channel pictures in a 2 × 2 grid of thin frames. Returns Group of framed tiles (channel order)."""
    cx, cy = center
    off = (side + gap) / 2
    spots = [(-off, off), (off, off), (-off, -off), (off, -off)]
    tiles = m.Group()
    for img, (dx, dy) in zip(images, spots):
        tiles.add(framed(img, EDGE, buff=0.03, stroke=stroke).move_to([cx + dx, cy + dy, 0]))
    return tiles


def cell_rect(pic, r0, c0, rows, cols, n=64, color=INK, width=3):
    """A rectangle over cells [r0, r0+rows) × [c0, c0+cols) of an n × n picture."""
    s = pic.height / n
    ul = pic.get_corner(m.UL)
    rect = m.Rectangle(width=cols * s, height=rows * s).set_stroke(color, width).set_fill(opacity=0)
    return rect.move_to(ul + np.array([(c0 + cols / 2) * s, -(r0 + rows / 2) * s, 0]))


def grid_lines(pic, n, color=INK, width=1.0, opacity=0.5):
    s = pic.height / n
    c = pic.get_center()
    h = pic.height / 2
    lines = m.VGroup()
    for k in range(1, n):
        off = -h + k * s
        lines.add(m.Line(c + [off, -h, 0], c + [off, h, 0]), m.Line(c + [-h, off, 0], c + [h, off, 0]))
    return lines.set_stroke(color, width, opacity=opacity)


def step_text(k, anchor, size=24, centre=False):
    words = code(f"step {k:>2}/{STEPS} · t = {T_SHOWN[k]}", size, TIME, weight=m.BOLD)
    return words.move_to(anchor) if centre else words.move_to(anchor, aligned_edge=m.LEFT)


class StepCounter(m.VGroup):
    """'step k/30 · t = …' in BLUE, left-aligned at a fixed point: `.set(k)`."""

    def __init__(self, left, y, k=0, size=24, centre=False):
        super().__init__()
        self.anchor = np.array([left, y, 0.0])
        self.size, self.centre = size, centre
        self.words = step_text(k, self.anchor, size, centre)
        self.add(self.words)
        self.k = k

    def set(self, k):
        k = int(np.clip(round(k), 0, STEPS))
        if k != self.k:
            self.words.become(step_text(k, self.anchor, self.size, self.centre))
            self.k = k


def flash(mob, color=DATA, width=7, time_width=0.8, run_time=0.9):
    return m.ShowPassingFlash(mob.copy().set_stroke(color, width), time_width=time_width, run_time=run_time)


def row_layout(widths, left=-6.3, right=6.3):
    """Centres for items of the given widths spread over [left, right] with equal gaps."""
    gap = (right - left - sum(widths)) / (len(widths) - 1)
    xs, x = [], left
    for w in widths:
        xs.append(x + w / 2)
        x += w + gap
    return xs, gap


GEN_CITE = f"generated on this Mac · Stable Diffusion 2.1 base · {STEPS} steps · guidance {CFG_W} · seed {SEED}"

# ---------------------------------------------------------------------------- the episode


class Ep17LatentDiffusion(Episode):
    number = 17

    def lesson(self):
        self.hook()
        self.squeeze()
        self.channels()
        self.diffuse()
        self.cheaper()
        self.stable()
        self.recap()

    # ------------------------------------------------------------------ 0 · hook

    def hook(self):
        self.next_section("Hook", notes=(
            "Callback to Episode 1: the lighthouse appearing out of static, generated on this Mac (SD 2.1-base, MLX Euler "
            f"sampler, {STEPS} steps, guidance {CFG_W}, seed {SEED}; images/lighthouse_frames30/xt_00..30.png = the noisy "
            "latent decoded after each step). Right: the latent the model actually denoised, x_t (arrays/"
            "lighthouse_latents_30steps.npz), its 4 channels as 64×64 pictures with crisp pixels, each stretched between the "
            "1st and 99th percentile of the finished latent's channel; colour from VIOLET to TEAL by the signal's share of "
            "the variance, 1/(1 + σ²) (illustrative colouring). t is the run's own timestep (1000 → 33, then the picture)."))
        y0 = 0.45
        pic = Reel(XT_FR, 3.0)
        pic_f = framed(pic.image, EDGE).move_to([-3.7, y0, 0])
        tiles = [Reel(XT_TILES[c], 1.3, k=STEPS) for c in range(4)]
        grid = tile_grid([t.image for t in tiles], (3.7, y0), 1.3, gap=0.12)
        mid_x = (pic_f.get_right()[0] + grid.get_left()[0]) / 2
        counter_w = step_text(0, np.zeros(3)).width
        counter = StepCounter(mid_x, y0 - 0.5, 0, 24, centre=True)
        gen = cite(GEN_CITE).move_to([0, -3.3, 0])
        pic_cap = m.VGroup(text("what you saw: the picture", 22, INK), code("512 × 512 × 3", 22, INK))
        pic_cap.arrange(m.DOWN, buff=0.12).next_to(pic_f, m.DOWN, buff=0.22)
        grid_title = text("what the model denoised", 22, INK, weight=m.BOLD).next_to(grid, m.UP, buff=0.22)
        grid_cap = code("64 × 64 × 4", 22, DATA).next_to(grid, m.DOWN, buff=0.22)
        sit(grid_cap, baseline(pic_cap[1]))
        a_dec = arrow([grid.get_left()[0] - 0.15, y0, 0], [pic_f.get_right()[0] + 0.15, y0, 0], DATA)
        dec_lab = text("decoded, to show you", 20, MUTED).next_to(a_dec, m.UP, buff=0.14)
        steps = m.VGroup(pill("1 · squeeze", DATA, TEAL_FILL, 22), pill("2 · diffuse", MODEL, AMBER_FILL, 22),
                         pill("3 · decode", DATA, TEAL_FILL, 22)).arrange(m.RIGHT, buff=0.9).move_to([0, -2.6, 0])
        links = m.VGroup(*[arrow(a.get_right(), b.get_left(), MUTED, buff=0.12) for a, b in zip(steps, steps[1:])])
        assert grid_title.get_top()[1] < 2.6 and pic_cap.get_bottom()[1] > steps.get_top()[1] + 0.1
        assert counter.get_left()[0] > pic_f.get_right()[0] + 0.15 and counter.get_right()[0] < grid.get_left()[0] - 0.15
        assert dec_lab.width < grid.get_left()[0] - pic_f.get_right()[0] - 0.4
        assert counter.get_bottom()[1] > steps.get_top()[1] + 0.15 and steps.get_bottom()[1] > gen.get_top()[1] + 0.1

        def both(v):
            pic.show(v)
            for t in tiles:
                t.show(v)
            counter.set(v)

        self.say(
            "[In episode one, our lighthouse came out of static], [in thirty steps].",
            m.AnimationGroup(self.new_header("00 · hook", "Where the static lives"), m.FadeIn(pic_f, shift=0.15 * m.UP),
                             m.FadeIn(counter), m.FadeIn(gen)),
            drive(pic.image, 0, STEPS, [pic.show, counter.set], run_time=3.0),
        )
        self.say(
            "[But that picture is not what the model worked on].",
            m.FadeIn(pic_cap, shift=0.1 * m.UP),
        )
        self.say(
            "[Every step changed this instead]: [a small grid, sixty-four by sixty-four, with four channels].",
            m.AnimationGroup(m.FadeIn(grid, shift=0.15 * m.LEFT), m.FadeIn(grid_title)),
            m.AnimationGroup(m.FadeIn(grid_cap, shift=0.1 * m.UP), m.GrowArrow(a_dec), m.FadeIn(dec_lab)),
        )
        self.say(
            "[Watch them side by side]: [each step changes the small grid], [and the picture is only its decoding].",
            drive(pic.image, 0, 0, [both], run_time=0.4),
            drive(pic.image, 0, STEPS, [both], run_time=4.2),
            m.Circumscribe(dec_lab, color=DATA, buff=0.08),
        )
        self.say(
            "[This is latent diffusion]: [squeeze the picture], [diffuse in the small space], [and decode at the end].",
            m.Circumscribe(grid, color=DATA, buff=0.1),
            m.FadeIn(steps[0], shift=0.1 * m.UP),
            m.AnimationGroup(m.GrowArrow(links[0]), m.FadeIn(steps[1], shift=0.1 * m.UP)),
            m.AnimationGroup(m.GrowArrow(links[1]), m.FadeIn(steps[2], shift=0.1 * m.UP)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 1 · squeeze and paint back

    def squeeze(self):
        self.next_section("1 · The autoencoder", notes=(
            "Rombach et al., 'High-Resolution Image Synthesis with Latent Diffusion Models', arXiv:2112.10752 (20 Dec 2021; "
            "CVPR 2022). The picture: the 50-step seed-0 lighthouse (images/lighthouse/lighthouse_sd21_cfg7.5_steps50_seed0.png). "
            "Its latent: the run's VAE encoder mean × 0.18215 (arrays/lighthouse_vae_latent.npz), 4 channels as 64×64 TEAL "
            f"pictures (each stretched 1st–99th percentile). Encoder {M(ENC_P)} M, decoder {M(DEC_P)} M parameters (mac_runs "
            f"§5). 512/64 = {FACTOR} per side; {N_PIX:,} / {N_LAT:,} = 48. The inset: a 64×64-pixel window around the lamp "
            f"(rows {CROP_R}–{CROP_R + 63}, cols {CROP_C}–{CROP_C + 63}) with an 8-pixel grid, and the matching 8×8 cells of "
            f"channel 1. Round trip (encode, decode) of the seed-0 picture: {PSNR:.2f} dB PSNR, measured on this Mac; RMS "
            f"error 10^(−PSNR/20) = {100 * RMS:.2f}% of the range. Caveat (mac_runs §8): the picture is itself a VAE output, "
            f"so this flatters the VAE; the paper's KL f = 8, c = 4 autoencoder scores {PSNR_PAPER} dB PSNR (Table 8). The "
            "autoencoder is trained with reconstruction + perceptual (LPIPS) + patch-GAN losses and a KL weight ~10⁻⁶ "
            "(App. G), so it is sharp, unlike Episode 3's blurry VAE. The decoded picture is not shown separately: the card "
            "states the measurement."))
        Y = 0.85
        widths = [2.62, 1.25, 1.84, 1.25, 2.8]
        xs, gap = row_layout(widths)
        assert gap > 0.6
        P = photo(PIC50, 2.5)
        p_f = framed(P, DATA).move_to([xs[0], Y, 0])
        enc = funnel((xs[1], Y), True, label="encoder")
        lat = tile_grid([photo(t, 0.87) for t in VAE_TILES], (xs[2], Y), 0.87, gap=0.1)
        dec = funnel((xs[3], Y), False, label="decoder")
        box = card(widths[4], 2.72, DATA, CARD, radius=0.18, stroke=2).move_to([xs[4], Y, 0])
        a1 = arrow(p_f.get_right(), enc[0].get_left(), DATA, buff=0.1)
        a2 = arrow(enc[0].get_right(), lat.get_left(), DATA, buff=0.1)
        a3 = arrow(lat.get_right(), dec[0].get_left(), DATA, buff=0.1)
        a4 = arrow(dec[0].get_right(), box.get_left(), DATA, buff=0.1)
        ly = Y - 1.25 - 0.45
        l_p = code("512×512×3", 22, INK).move_to([xs[0], ly, 0])
        l_e = code(f"{M(ENC_P)} M weights", 20, MUTED).move_to([xs[1], ly, 0])
        l_l = code("64×64×4", 22, DATA).move_to([xs[2], ly, 0])
        l_d = code(f"{M(DEC_P)} M weights", 20, MUTED).move_to([xs[3], ly, 0])
        for w in (l_e, l_l, l_d):
            sit(w, baseline(l_p))
        n_p = cite(f"{N_PIX:,} numbers").move_to([xs[0], ly - 0.4, 0])
        n_l = cite(f"{N_LAT:,} numbers").move_to([xs[2], ly - 0.4, 0])
        sit(n_l, baseline(n_p))
        src = cite("Rombach et al., 2022 · arXiv, December 2021").move_to([0, 2.5, 0]).align_to([6.4, 0, 0], m.RIGHT)
        assert src.get_bottom()[1] > box.get_top()[1] + 0.1

        # the card: back to pixels, then the round trip
        c1 = text("back to pixels", 22, INK, weight=m.BOLD)
        c2 = code("512×512×3", 22, INK)
        c_top = m.VGroup(c1, c2).arrange(m.DOWN, buff=0.14).move_to([xs[4], Y + 0.83, 0])
        rule = m.Line([xs[4] - 1.1, Y + 0.3, 0], [xs[4] + 1.1, Y + 0.3, 0]).set_stroke(EDGE, 2)
        c3 = text("round trip", 20, MUTED)
        big = m.VGroup(title_text(f"{PSNR:.1f}", 42, INK), code("dB PSNR", 20, INK)).arrange(m.RIGHT, buff=0.15)
        sit(big[1], baseline(big[0]))
        c4 = code(f"error ≈ {100 * RMS:.1f}%", 20, MUTED)
        c_low = m.VGroup(c3, big, c4).arrange(m.DOWN, buff=0.12).move_to([xs[4], Y - 0.48, 0])
        here = cite("measured on this Mac").move_to([xs[4], ly, 0])
        sit(here, baseline(l_d))
        assert here.get_left()[0] > l_d.get_right()[0] + 0.2 and here.get_right()[0] < 6.5
        assert c_low.width < widths[4] - 0.3 and c_low.get_bottom()[1] > Y - 1.36 + 0.2 and c_top.get_top()[1] < Y + 1.36 - 0.15

        # the inset: 64 × 64 pixels = 8 × 8 cells
        iy = -2.35
        crop1 = photo(CROP_PIX, 1.25, nearest=True)
        crop1_f = framed(crop1, DATA, buff=0.03, stroke=2).move_to([-5.5, iy, 0])
        g1 = grid_lines(crop1, 8, INK, 1.2, 0.6)
        crop2 = photo(CROP_LAT_RGB, 1.25, nearest=True)
        crop2_f = framed(crop2, DATA, buff=0.03, stroke=2).move_to([-3.05, iy, 0])
        g2 = grid_lines(crop2, 8, BG, 1.2, 0.5)
        a_in = arrow(crop1_f.get_right(), crop2_f.get_left(), DATA, buff=0.12)
        k1 = cite("64 × 64 pixels").next_to(crop1_f, m.DOWN, buff=0.12)
        k2 = cite("8 × 8 cells").next_to(crop2_f, m.DOWN, buff=0.12)
        sit(k2, baseline(k1))
        rule1 = m.VGroup(code("1 cell", 22, INK, weight=m.BOLD), code("= 8×8 pixels", 22, INK))
        rule1.arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT).next_to(crop2_f, m.RIGHT, buff=0.35)
        m_p = cell_rect(P, CROP_R, CROP_C, 64, 64, 512, INK, 2.5)
        m_l = cell_rect(lat[0][1], CROP_R // 8, CROP_C // 8, 8, 8, 64, INK, 2)
        inset = m.Group(crop1_f, g1, crop2_f, g2, a_in, k1, k2, rule1)
        assert k1.get_bottom()[1] > -3.5 and rule1.get_right()[0] < 0.6, (k1.get_bottom(), rule1.get_right())

        # 48× fewer, then the caveat
        fewer = m.VGroup(title_text("48×", 72, DATA), text("fewer numbers", 26, INK)).arrange(m.RIGHT, buff=0.3)
        fewer.move_to([3.55, -2.45, 0])
        sit(fewer[1], baseline(fewer[0]))
        cav1 = m.VGroup(text("optimistic:", 20, RED, weight=m.BOLD), text("made by this same decoder", 20, INK))
        cav1.arrange(m.RIGHT, buff=0.14)
        sit(cav1[1], baseline(cav1[0]))
        cav2 = code(f"real photos, same shape: {PSNR_PAPER:.1f} dB", 20, INK)
        cav3 = cite("Rombach et al., 2022 · Table 8 · f = 8, 4 channels")
        cav = m.VGroup(cav1, cav2, cav3).arrange(m.DOWN, buff=0.16, aligned_edge=m.LEFT)
        cav.move_to([3.6, -2.45, 0]).align_to([rule1.get_right()[0] + 0.45, 0, 0], m.LEFT)
        assert fewer.get_left()[0] > rule1.get_right()[0] + 0.3 and cav.get_left()[0] > rule1.get_right()[0] + 0.25
        assert cav.get_right()[0] < 6.5 and cav.get_bottom()[1] > -3.5, (cav.get_right(), cav.get_bottom())
        sharp = cite("trained with a critic").move_to([xs[3], ly - 0.4, 0])
        sit(sharp, baseline(n_l))
        assert sharp.get_left()[0] > n_l.get_right()[0] + 0.15 and sharp.get_bottom()[1] > fewer.get_top()[1] + 0.1

        self.say(
            "[The trick comes from a paper by Rombach and colleagues, in late twenty twenty-one]: "
            "[run the diffusion model inside an autoencoder, a VAE].",
            m.AnimationGroup(self.new_header("01 · autoencoder", "Squeeze, then paint back"),
                             m.FadeIn(p_f, shift=0.15 * m.UP), m.FadeIn(l_p), m.FadeIn(src)),
            m.AnimationGroup(m.FadeIn(enc, shift=0.1 * m.RIGHT), m.FadeIn(dec, shift=0.1 * m.RIGHT)),
        )
        self.say(
            "[Its encoder squeezes our five hundred twelve by five hundred twelve picture, with three colour channels], "
            "[into sixty-four by sixty-four, with four].",
            m.AnimationGroup(m.GrowArrow(a1), flash(enc[0], DATA), m.FadeIn(l_e)),
            m.AnimationGroup(m.GrowArrow(a2), m.FadeIn(lat, target_position=enc), m.FadeIn(l_l)),
        )
        self.say(
            "[Each side shrinks eight times]: [one cell of the latent covers an eight by eight patch of pixels].",
            m.AnimationGroup(m.Create(m_p), m.FadeIn(crop1_f, shift=0.1 * m.UP), m.Create(g1), m.FadeIn(k1)),
            m.AnimationGroup(m.Create(m_l), m.GrowArrow(a_in), m.FadeIn(crop2_f, shift=0.1 * m.UP), m.Create(g2),
                             m.FadeIn(k2), m.FadeIn(rule1)),
        )
        self.say(
            "[That's seven hundred eighty-six thousand numbers, down to sixteen thousand]: [forty-eight times fewer].",
            m.AnimationGroup(m.FadeIn(n_p, shift=0.1 * m.UP), m.FadeIn(n_l, shift=0.1 * m.UP)),
            m.FadeIn(fewer, shift=0.15 * m.UP),
        )
        self.say(
            "[Its decoder paints the pixels back]: [sharp, because it was also trained against a critic, "
            "like the GANs of episode three].",
            m.AnimationGroup(m.GrowArrow(a3), flash(dec[0], DATA), m.FadeIn(l_d), m.GrowArrow(a4), m.FadeIn(box),
                             m.FadeIn(c_top)),
            m.FadeIn(sharp, shift=0.1 * m.UP),
        )
        self.say(
            "[On our picture, the round trip measures thirty-eight decibels]: [a typical pixel is off by about one percent].",
            m.AnimationGroup(m.Create(rule), m.FadeIn(c3), m.FadeIn(big, shift=0.1 * m.UP), m.FadeIn(here)),
            m.FadeIn(c4, shift=0.1 * m.UP),
        )
        self.say(
            "[That flatters it, though]: [this picture was painted by the same decoder in the first place].",
            m.FadeOut(fewer),
            m.FadeIn(cav1, shift=0.1 * m.UP),
        )
        self.say(
            "[On real photos, the paper's autoencoder of this shape scored about twenty-four decibels].",
            m.AnimationGroup(m.FadeIn(cav2, shift=0.1 * m.UP), m.FadeIn(cav3)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 2 · the four channels

    def channels(self):
        self.next_section("2 · Four channels", notes=(
            "The seed-0 lighthouse (50 steps) and its latent, the run's VAE encoding × 0.18215 (arrays/"
            "lighthouse_vae_latent.npz): each channel stretched between its 1st and 99th percentile, TEAL, crisp 64×64 "
            "pixels. The white boxes mark the tower (latent rows 8–37, columns 38–45, read off the picture; the lamp, the "
            f"reddest pixel near the top, is at pixel ({LAMP_R}, {LAMP_C})). Notes §17: the channels are learned features, "
            "not RGB + alpha. Scale factor: LDM App. G rescales latents to unit standard deviation, σ̂ estimated from the "
            f"first batch; SD's config scale_factor 0.18215 ⇒ σ̂ ≈ {1 / SCALE:.2f}. Our latent: std {LAT_STD:.3f} after the "
            f"factor, {RAW_STD:.2f} before it (one picture, not a batch); the noise ε has std 1. Bars are drawn to scale "
            "(5 = full)."))
        Y = 0.8
        P = photo(PIC50, 2.05)
        p_f = framed(P, DATA)
        tiles = m.Group(*[framed(photo(t, 2.05), EDGE, buff=0.03, stroke=2) for t in VAE_TILES]).arrange(m.RIGHT, buff=0.22)
        row = m.Group(p_f, tiles).arrange(m.RIGHT, buff=0.6).move_to([0, Y, 0])
        assert row.width < 12.6
        t_p = text("the picture", 22, INK, weight=m.BOLD).next_to(p_f, m.UP, buff=0.22)
        t_t = text("its latent, channel by channel", 22, INK, weight=m.BOLD).next_to(tiles, m.UP, buff=0.22)
        sit(t_t, baseline(t_p))
        assert t_t.get_top()[1] < 2.62
        labs = m.VGroup(*[code(f"channel {c + 1}", 20, MUTED).next_to(tiles[c], m.DOWN, buff=0.16) for c in range(4)])
        l_p = code("512×512×3", 20, MUTED).next_to(p_f, m.DOWN, buff=0.16)
        sit(l_p, baseline(labs[0]))
        boxes = m.VGroup(*[cell_rect(tiles[c][1], *TOWER, 64, INK, 2.5) for c in range(4)])
        not_rgb = pill("learned features: not red, green, blue, alpha", DATA, TEAL_FILL, 22).move_to([1.45, -1.2, 0])
        assert not_rgb.get_top()[1] < labs.get_bottom()[1] - 0.12

        # the scale factor
        fy = -2.15
        formula = code(f"z = {SCALE} × encoder(x)", 24, INK, t2c={r"0\.18215": DATA, r"\bz\b": DATA})
        formula.move_to([-3.45, fy, 0])
        f_cite = cite("so the latent matches the noise's scale").next_to(formula, m.DOWN, buff=0.2)
        f_src = cite("Rombach et al., 2022, App. G").next_to(f_cite, m.DOWN, buff=0.1)
        scale_glyphs = glyphs(formula, str(SCALE))
        head = text("spread (standard deviation)", 20, MUTED)
        rows = m.VGroup()
        for label, v, col in (("as encoded", RAW_STD, MUTED), (f"× {SCALE}", LAT_STD, DATA), ("the noise", 1.0, NOISE)):
            words = code(label, 20, INK)
            bar = Meter(2.2, v / 5.0, col, height=0.18)
            num = code(f"{v:.2f}", 20, INK)
            rows.add(m.VGroup(words, bar, num))
        left = 1.25
        block = m.VGroup(head, *rows).arrange(m.DOWN, buff=0.17, aligned_edge=m.LEFT)
        block.move_to([0, -2.5, 0]).align_to([left, 0, 0], m.LEFT)
        for r in rows:
            r[0].align_to([left, 0, 0], m.LEFT)
            r[1].move_to([left + 1.85 + 1.1, r[0].get_y(), 0])
            r[2].next_to(r[1], m.RIGHT, buff=0.2)
            sit(r[2], baseline(r[0]))
        assert rows.get_right()[0] < 6.5 and block.get_bottom()[1] > -3.5 and block.get_top()[1] < not_rgb.get_bottom()[1] - 0.1
        assert f_src.get_bottom()[1] > -3.5 and formula.get_right()[0] < left - 0.3

        self.say(
            "[Here is that latent, channel by channel], [next to the picture it encodes].",
            m.AnimationGroup(self.new_header("02 · latent", "Four channels, not colours"), m.FadeIn(t_t),
                             m.LaggedStart(*[m.FadeIn(t, shift=0.15 * m.UP) for t in tiles], lag_ratio=0.2),
                             m.FadeIn(labs)),
            m.AnimationGroup(m.FadeIn(p_f, shift=0.15 * m.RIGHT), m.FadeIn(t_p), m.FadeIn(l_p)),
        )
        self.say(
            "[Each channel is a rough sketch of the same scene]: [the tower stands out in all four].",
            m.AnimationGroup(*[m.Circumscribe(t[0], color=DATA, buff=0.05) for t in tiles]),
            m.LaggedStart(*[m.Create(b) for b in boxes], lag_ratio=0.2),
        )
        self.say(
            "[But they are not red, green and blue]: [the autoencoder learned these four for itself].",
            m.FadeIn(not_rgb, shift=0.1 * m.UP),
            m.Circumscribe(not_rgb, color=DATA, buff=0.08),
        )
        self.say(
            "[One more detail]: [the latent is multiplied by zero point one eight two one five].",
            m.FadeIn(formula, shift=0.1 * m.UP),
            m.AnimationGroup(m.Circumscribe(scale_glyphs, color=DATA, buff=0.06), m.FadeIn(f_src)),
        )
        self.say(
            "[That shrinks its spread from about three point seven to about zero point seven], "
            "[close to the noise it will be mixed with].",
            m.AnimationGroup(m.FadeIn(head), m.FadeIn(rows[0]), m.FadeIn(rows[1]), m.FadeIn(f_cite)),
            m.FadeIn(rows[2], shift=0.1 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 3 · diffuse in the latent

    def diffuse(self):
        self.next_section("3 · Diffuse in the latent", notes=(
            f"The {STEPS}-step run (arrays/lighthouse_latents_30steps.npz): x_t after each step, its 4 channels (each "
            "stretched 1st–99th percentile of the finished latent), coloured VIOLET → TEAL by the signal's share of the "
            "variance 1/(1 + σ²) (illustrative). The run stores x_t in the variance-preserving scale (std "
            f"{XT_LAT[0].std():.2f} at t = 1000). Previews: images/lighthouse_frames30/xt_k.png (x_t decoded) and "
            "x0hat_k.png (the model's clean-latent guess x̂₀ decoded); the last guess is the finished picture. The sampler "
            "is mlx-examples' deterministic Euler sampler, guidance 7.5 (Episodes 10, 14); Episode 7's loop is DDPM's, the "
            "idea is the same. The previews are for watching: a normal run decodes once (Episode 23 times it)."))
        Y = 0.75
        den = m.VGroup(card(1.95, 1.3, MODEL, AMBER_FILL, radius=0.16, stroke=2.5),
                       m.VGroup(text("denoiser", 24, INK, weight=m.BOLD), code(f"{M(UNET_P)} M", 20, MUTED))
                       .arrange(m.DOWN, buff=0.1))
        den[1].move_to(den[0])
        den.move_to([-5.15, Y, 0])
        tiles = [Reel(XT_TILES[c], 1.05, k=0) for c in range(4)]
        grid = tile_grid([t.image for t in tiles], (-2.05, Y), 1.05, gap=0.1)
        g_lab = m.VGroup(tex(r"x_t", 34, NOISE), code("64×64×4", 20, MUTED)).arrange(m.RIGHT, buff=0.22)
        g_lab.next_to(grid, m.DOWN, buff=0.2)
        a_in = m.Arrow([grid.get_left()[0] - 0.08, Y + 0.38, 0], [den.get_right()[0] + 0.08, Y + 0.38, 0], buff=0,
                       color=NOISE, stroke_width=5, max_tip_length_to_length_ratio=0.3)
        a_out = m.Arrow([den.get_right()[0] + 0.08, Y - 0.38, 0], [grid.get_left()[0] - 0.08, Y - 0.38, 0], buff=0,
                        color=DATA, stroke_width=5, max_tip_length_to_length_ratio=0.3)
        loop_lab = cite("Episode 7's loop").next_to(den, m.DOWN, buff=0.2)
        sit(loop_lab, baseline(g_lab[1]))
        dec = funnel((0.4, Y), False, wide=1.5, narrow=0.7, length=0.85, label="")
        a_dec = arrow(grid.get_right(), dec[0].get_left(), MUTED, buff=0.1)
        xt_r = Reel(XT_FR, 2.15)
        x0_r = Reel(X0_FR + [XT_FR[STEPS]], 2.15)
        pv1 = framed(xt_r.image, EDGE).move_to([2.55, Y, 0])
        pv2 = framed(x0_r.image, EDGE).move_to([5.0, Y, 0])
        a_pv = arrow(dec[0].get_right(), pv1.get_left(), MUTED, buff=0.1)
        assert pv2.get_right()[0] < 6.5 and pv1.get_left()[0] > dec[0].get_right()[0] + 0.25
        l1 = m.VGroup(tex(r"x_t", 34, NOISE), text("decoded", 20, MUTED)).arrange(m.RIGHT, buff=0.18)
        l2 = m.VGroup(tex(r"\hat{x}_0", 34, DATA), text("decoded", 20, MUTED)).arrange(m.RIGHT, buff=0.18)
        l1.next_to(pv1, m.DOWN, buff=0.2)
        l2.next_to(pv2, m.DOWN, buff=0.2)
        sit(l2[1], baseline(l1[1]))
        dec_lab = cite("decoder").move_to([0.4, 0, 0])
        sit(dec_lab, baseline(l1[1]))
        watch = text("decoded just for watching", 20, MUTED).next_to(m.Group(pv1, pv2), m.UP, buff=0.2)
        assert watch.get_top()[1] < 2.62 and l2.get_right()[0] < 6.55 and l1.get_right()[0] < l2.get_left()[0] - 0.2
        counter = StepCounter(-6.2, -1.5, 0, 24)
        track = m.Rectangle(width=12.4, height=0.1).set_stroke(EDGE, 1.5).set_fill(CARD, 1).move_to([0, -2.0, 0])
        bar = m.Rectangle(width=0.004, height=0.1).set_stroke(width=0).set_fill(TIME, 1).move_to([-6.2, -2.0, 0])

        def put_bar(v):
            w = max(12.4 * v / STEPS, 0.004)
            bar.become(m.Rectangle(width=w, height=0.1).set_stroke(width=0).set_fill(TIME, 1).move_to([-6.2 + w / 2, -2.0, 0]))

        once = pill("a real run decodes once, at the very end", DATA, TEAL_FILL, 22).move_to([0, -2.65, 0])
        gen = cite(GEN_CITE).move_to([0, -3.3, 0])
        assert counter.get_bottom()[1] > track.get_top()[1] + 0.1 and g_lab.get_bottom()[1] > counter.get_top()[1] + 0.1

        def run(v):
            for t in tiles:
                t.show(v)
            xt_r.show(v)
            x0_r.show(v)
            counter.set(v)
            put_bar(v)

        self.say(
            "[Now generate in that space]: [start from pure noise, shaped sixty-four by sixty-four by four].",
            m.AnimationGroup(self.new_header("03 · diffuse", "Denoising the small grid"), m.FadeIn(gen)),
            m.AnimationGroup(m.FadeIn(grid, shift=0.1 * m.UP), m.FadeIn(g_lab), m.FadeIn(counter), m.FadeIn(track),
                             m.FadeIn(bar)),
        )
        self.say(
            "[The denoiser works on this small grid], [step after step], [the same loop as in episode seven].",
            m.FadeIn(den, shift=0.1 * m.RIGHT),
            m.AnimationGroup(m.GrowArrow(a_in), m.GrowArrow(a_out)),
            m.FadeIn(loop_lab, shift=0.1 * m.UP),
        )
        self.say(
            "[To watch it, we decode every step]: [the noisy latent decodes to coloured static], "
            "[the clean guess to a blur].",
            m.AnimationGroup(m.GrowArrow(a_dec), m.FadeIn(dec), m.FadeIn(dec_lab), m.GrowArrow(a_pv), m.FadeIn(watch)),
            m.AnimationGroup(m.FadeIn(pv1, shift=0.1 * m.UP), m.FadeIn(l1)),
            m.AnimationGroup(m.FadeIn(pv2, shift=0.1 * m.UP), m.FadeIn(l2)),
        )
        self.say(
            "Now let all thirty steps run, and watch the lighthouse settle out of the noise.",
            drive(xt_r.image, 0, STEPS, [run], run_time=6.0),
        )
        self.say(
            "[Those previews are only for us]: [a real run decodes just once, at the very end].",
            m.AnimationGroup(m.Circumscribe(watch, color=MUTED, buff=0.08)),
            m.AnimationGroup(m.FadeIn(once, shift=0.1 * m.UP), m.Circumscribe(pv2, color=DATA, buff=0.06)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 4 · why it is cheaper

    def cheaper(self):
        self.next_section("4 · Why it is cheaper", notes=(
            f"Positions per channel: 512² = {POS_PIX:,} vs 64² = {POS_LAT:,} (64×); a convolution's work per layer grows "
            f"with positions. Self-attention compares every pair: {POS_PIX:,}² = {PAIRS_PIX:,} vs {POS_LAT:,}² = "
            f"{PAIRS_LAT:,} pairs (4,096×) for one attention map over the whole grid (Episode 8: 7×7 → 2,401, 28×28 → "
            "614,656). SD's U-Net has attention at the latent's full 64×64 (notes §8). The grid on the latent is 64 × 64. "
            "Rombach et al. 2022, Table 18 (ImageNet 256, class-conditional): ADM-G (pixel space) FID 4.59 with 962 "
            "V100-days; LDM-4-G (latent, f = 4, cfg 1.5) FID 3.60 with 271 V100-days: 3.5× less training compute (and "
            "~5.7× higher sampling throughput). Timing measured on this Mac (mac_runs §5, median of 3 runs: 7.35 / 7.32 / "
            f"7.31 s): text encoder {TXT_S:.3f} s ({SHARE['text']:.1f}%), 50 steps × 2 batched U-Net passes "
            f"{DEN_S:.2f} s ({SHARE['den']:.1f}%), VAE decode {DEC_S:.3f} s ({SHARE['dec']:.1f}%), total {TOT_S:.2f} s. "
            "The VAE runs in float32, the U-Net in float16 (pipeline defaults). Bar lengths to scale; 50 ticks = 50 steps."))
        Y = 0.9
        P = photo(PIC50, 2.4)
        p_f = framed(P, DATA).move_to([-3.7, Y, 0])
        L = photo(VAE_TILES[0], 2.4)
        l_f = framed(L, DATA).move_to([3.7, Y, 0])
        mesh = grid_lines(L, 64, BG, 0.8, 0.55)
        c_p = code(f"512×512 = {POS_PIX:,} positions", 20, INK).next_to(p_f, m.DOWN, buff=0.2)
        c_l = code(f"64×64 = {POS_LAT:,} positions", 20, DATA).next_to(l_f, m.DOWN, buff=0.2)
        sit(c_l, baseline(c_p))
        div = m.VGroup(title_text("64×", 64, DATA), text("fewer", 24, INK)).arrange(m.DOWN, buff=0.1).move_to([0, Y, 0])
        assert c_p.get_left()[0] > -6.5 and c_l.get_right()[0] < 6.5

        att_t = text("attention compares every pair of positions", 22, INK, weight=m.BOLD)
        att1 = code(f"pixels: {POS_PIX:,}² = {PAIRS_PIX / 1e9:.1f} billion pairs", 20, INK)
        att2 = code(f"latent:   {POS_LAT:,}² = {PAIRS_LAT / 1e6:.1f} million pairs", 20, DATA)
        att = m.VGroup(att_t, att1, att2).arrange(m.DOWN, buff=0.16, aligned_edge=m.LEFT).move_to([-2.6, -2.15, 0])
        att.align_to([-6.2, 0, 0], m.LEFT)
        att_big = m.VGroup(title_text(f"{PAIRS_PIX // PAIRS_LAT:,}×", 56, DATA), text("fewer pairs", 22, INK))
        att_big.arrange(m.DOWN, buff=0.08).move_to([4.6, -1.95, 0])
        ep8 = pill("Episode 8: 2,401 vs 614,656", MODEL, AMBER_FILL, 20).next_to(att_big, m.DOWN, buff=0.22)
        assert att.get_right()[0] < att_big.get_left()[0] - 0.3 and ep8.get_bottom()[1] > -3.5
        assert att.get_top()[1] < c_p.get_bottom()[1] - 0.2 and att.get_bottom()[1] > -3.5
        part_a = m.Group(p_f, l_f, mesh, c_p, c_l, div, att, att_big, ep8)

        # the paper's ImageNet result
        top_y = 2.25
        hd = text("ImageNet, 256 × 256, one class at a time", 22, INK, weight=m.BOLD).move_to([0, top_y, 0])
        hd.align_to([-6.2, 0, 0], m.LEFT)
        hs = cite("Rombach et al., 2022, Table 18 · FID: lower is better").next_to(hd, m.RIGHT, buff=0.35)
        sit(hs, baseline(hd))
        assert hs.get_right()[0] < 6.5
        x0b, span = -3.0, 4.4
        bars = []
        for k, (name, sub, days, fid, col, y) in enumerate((("in pixels", "ADM-G", ADM_DAYS, ADM_FID, MUTED, 1.45),
                                                            ("in a latent", "LDM-4-G", LDM_DAYS, LDM_FID, DATA, 0.65))):
            w = span * days / ADM_DAYS
            rect = m.Rectangle(width=w, height=0.42).set_stroke(col, 2).set_fill(mix(CARD, col, 0.28), 1)
            rect.move_to([x0b + w / 2, y, 0])
            lab = m.VGroup(text(name, 22, INK), cite(sub)).arrange(m.DOWN, buff=0.06, aligned_edge=m.LEFT)
            lab.move_to([0, y, 0]).align_to([-6.2, 0, 0], m.LEFT)
            val = code(f"{days} V100-days · FID {fid:.2f}", 20, GOOD if k else INK).next_to(rect, m.RIGHT, buff=0.2)
            bars.append(m.VGroup(rect, lab, val))
        less = pill(f"{ADM_DAYS / LDM_DAYS:.1f}× less training", GOOD, GOOD_FILL, 22)
        less.move_to([4.65, 0.65, 0])
        assert bars[0][2].get_right()[0] < 6.5 and less.get_left()[0] > bars[1][2].get_right()[0] + 0.2
        assert less.get_right()[0] < 6.5 and bars[0][1].get_right()[0] < x0b - 0.15

        # where the time goes in our run
        by, bh, bl, br = -1.45, 0.5, -6.0, 6.0
        span_t = br - bl
        w_txt, w_den = span_t * TXT_S / TOT_S, span_t * DEN_S / TOT_S
        w_dec = span_t - w_txt - w_den
        seg_txt = m.Rectangle(width=max(w_txt, 0.05), height=bh).set_stroke(width=0).set_fill(PROMPT, 1)
        seg_txt.move_to([bl + max(w_txt, 0.05) / 2, by, 0])
        seg_den = m.Rectangle(width=w_den, height=bh).set_stroke(MODEL, 2).set_fill(AMBER_FILL, 1)
        seg_den.move_to([bl + w_txt + w_den / 2, by, 0])
        seg_dec = m.Rectangle(width=w_dec, height=bh).set_stroke(DATA, 2).set_fill(DATA, 1)
        seg_dec.move_to([br - w_dec / 2, by, 0])
        ticks = m.VGroup(*[m.Line([bl + w_txt + w_den * i / 50, by - bh / 2 + 0.06, 0],
                                  [bl + w_txt + w_den * i / 50, by + bh / 2 - 0.06, 0]).set_stroke(MODEL, 1.2, opacity=0.6)
                           for i in range(1, 50)])
        lab_den = code(f"denoiser · {TM['denoise_steps']} steps × 2 passes · {DEN_S:.2f} s · {SHARE['den']:.1f}%", 20,
                       MODEL).move_to([0, by + bh / 2 + 0.3, 0])
        lab_dec = code(f"decoder · once · {DEC_S:.3f} s · {SHARE['dec']:.1f}%", 20, DATA)
        lab_dec.next_to(seg_dec, m.DOWN, buff=0.22).align_to([br, 0, 0], m.RIGHT)
        lab_txt = code(f"text encoder · {TXT_S:.3f} s", 20, PROMPT)
        lab_txt.next_to(seg_txt, m.DOWN, buff=0.22).align_to([bl, 0, 0], m.LEFT)
        sit(lab_txt, baseline(lab_dec))
        total = text(f"one 512 × 512 picture: {TOT_S:.2f} s in all", 22, INK).move_to([0, -2.7, 0])
        t_cite = cite("measured on this Mac · SD 2.1 base · 50 steps · guidance 7.5 · median of 3 runs · Episode 23")
        t_cite.move_to([0, -3.3, 0])
        assert lab_den.get_top()[1] < bars[1][0].get_bottom()[1] - 0.2 and lab_txt.get_right()[0] < lab_dec.get_left()[0] - 0.3
        assert total.get_top()[1] < lab_dec.get_bottom()[1] - 0.1 and t_cite.width < 12.6
        timing = m.VGroup(seg_txt, seg_den, ticks, seg_dec)

        self.say(
            "[Why is this so much cheaper]? [Every layer of the denoiser does its work at every position of the grid].",
            m.AnimationGroup(self.new_header("04 · cost", "Fewer positions, less work"), m.FadeIn(p_f, shift=0.1 * m.UP),
                             m.FadeIn(l_f, shift=0.1 * m.UP)),
            m.Create(mesh, run_time=1.6),
        )
        self.say(
            "[Five hundred twelve by five hundred twelve is two hundred sixty-two thousand positions]; "
            "[sixty-four by sixty-four, just four thousand ninety-six].",
            m.FadeIn(c_p, shift=0.1 * m.UP),
            m.FadeIn(c_l, shift=0.1 * m.UP),
        )
        self.say(
            "[That's sixty-four times fewer, in every layer, at every step].",
            m.FadeIn(div, shift=0.15 * m.UP),
        )
        self.say(
            "[Attention grows faster still]: [it compares every position with every other], "
            "[as we counted in episode eight].",
            m.FadeIn(att_t, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(att1), m.FadeIn(att2), m.FadeIn(att_big, shift=0.1 * m.UP)),
            m.FadeIn(ep8, shift=0.1 * m.UP),
        )
        self.say(
            "[In the paper, a latent model beat a pixel model at making ImageNet pictures], "
            "[with three and a half times less training compute].",
            m.AnimationGroup(m.FadeOut(part_a), m.FadeIn(hd), m.FadeIn(hs), *[m.FadeIn(b[1]) for b in bars],
                             m.GrowFromEdge(bars[0][0], m.LEFT), m.GrowFromEdge(bars[1][0], m.LEFT),
                             m.FadeIn(bars[0][2]), m.FadeIn(bars[1][2])),
            m.FadeIn(less, shift=0.1 * m.LEFT),
        )
        self.say(
            "[On this Mac, the denoiser takes ninety-six percent of the time]: [fifty guided steps, a hundred passes].",
            m.AnimationGroup(m.FadeIn(seg_txt), m.GrowFromEdge(seg_den, m.LEFT, run_time=1.4), m.FadeIn(seg_dec),
                             m.FadeIn(lab_den), m.FadeIn(t_cite)),
            m.Create(ticks, run_time=1.2),
        )
        self.say(
            "[The decoder runs only once, at the end]: [a quarter of a second, under four percent].",
            m.AnimationGroup(m.FadeIn(lab_dec, shift=0.1 * m.UP), m.FadeIn(lab_txt, shift=0.1 * m.UP)),
            m.AnimationGroup(m.Circumscribe(seg_dec, color=DATA, buff=0.06), m.FadeIn(total)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 5 · Stable Diffusion

    def stable(self):
        self.next_section("5 · Stable Diffusion", notes=(
            "Stable Diffusion = a latent diffusion model: a frozen text encoder (Episode 15) + a U-Net denoiser in the "
            "latent + the f = 8, 4-channel autoencoder. Our run, SD 2.1-base (sd2-community mirror; counted on this Mac, "
            f"mac_runs §5): U-Net {UNET_P:,}, OpenCLIP-H text encoder {TEXT_P:,}, VAE {VAE_P:,} ({ENC_P:,} encoder + "
            f"{DEC_P:,} decoder). SD 1.5 (safetensors counts, notes §8/§23): U-Net {SD15_UNET:,}, CLIP ViT-L text encoder "
            f"{SD15_TEXT:,}, VAE {SD15_VAE:,}; a generation touches text + U-Net + decoder = {SD15_GEN:,} weights × 2 bytes "
            f"= {SD15_GEN * 2 / 1e9:.2f} GB (fp16). Bars: our run, length ∝ parameters. Dates [DATED, notes §17 releases, "
            "checked Oct 2026]: arXiv:2112.10752 on 20 Dec 2021; v1.4 public release 22 Aug 2022 (CreativeML OpenRAIL-M); "
            "v1.5 Oct 2022 (RunwayML; earliest archive 20 Oct 2022); 2.1 on 7 Dec 2022 (2.0: 24 Nov 2022); SDXL 1.0 on 26 Jul "
            "2023 (U-Net 2,567,463,684); SD3 Medium on 12 Jun 2024 (MM-DiT transformer, 16-channel autoencoder; SD3 "
            "announced 22 Feb 2024). Timeline: months to scale. Data (CompVis README): subsets of LAION-5B — laion2B-en, "
            "laion-high-resolution, laion-aesthetics v2 5+; LAION-5B has 5.85 billion CLIP-filtered image–text pairs "
            "(Schuhmann et al., 2022). Detail in Episode 20; transformers in Episode 18."))
        cy, ch, cw = 1.85, 1.12, 3.4
        specs = (("text encoder", "prompt → vectors", PROMPT, -4.25, TEXT_P, round(SD15_TEXT / 1e6)),
                 ("denoiser", "noise → clean latent", MODEL, 0.0, UNET_P, round(SD15_UNET / 1e6)),
                 ("autoencoder", "latent → pixels", DATA, 4.25, VAE_P, round(SD15_VAE / 1e6)))
        cards, bars, vals, olds = [], [], [], []
        unit = 3.0 / UNET_P
        for name, role, col, x, p, old in specs:
            box = card(cw, ch, col, FILLS[col], radius=0.16, stroke=2.5).move_to([x, cy, 0])
            words = m.VGroup(text(name, 24, INK, weight=m.BOLD), text(role, 20, MUTED)).arrange(m.DOWN, buff=0.08)
            cards.append(m.VGroup(box, words.move_to(box)))
            w = unit * p
            left = x - cw / 2 + 0.2
            bar = m.Rectangle(width=w, height=0.26).set_stroke(col, 2).set_fill(FILLS[col], 1).move_to([left + w / 2, 0.98, 0])
            bars.append(bar)
            vals.append(code(f"2.1 base: {M(p)} M", 20, INK).move_to([0, 0.55, 0]).align_to([left, 0, 0], m.LEFT))
            olds.append(code(f"1.5: {old} M", 20, MUTED).move_to([0, 0.15, 0]).align_to([left, 0, 0], m.LEFT))
        for v in vals[1:]:
            sit(v, baseline(vals[0]))
        for v in olds[1:]:
            sit(v, baseline(olds[0]))
        assert all(v.get_right()[0] < c[0].get_right()[0] + 0.05 for v, c in zip(vals, cards))
        gb = code(f"1.5: text + denoiser + decoder = {SD15_GEN / 1e9:.2f} B weights ≈ "
                  f"{SD15_GEN * 2 / 1e9:.2f} GB in fp16", 20, INK, t2c={r"2\.06 GB": DATA}).move_to([0, -0.35, 0])
        data = text("trained on subsets of LAION-5B: 5.85 billion web images with captions · Episode 20", 20, INK)
        data.move_to([0, -0.8, 0])
        assert gb.width < 12.8 and data.width < 12.8

        # the timeline (months since Dec 2021)
        ay, ax0, ax1 = -2.3, -5.6, 5.6
        axis = m.Line([ax0 - 0.3, ay, 0], [ax1 + 0.3, ay, 0]).set_stroke(TIME, 3)
        x_of = lambda mo: ax0 + (ax1 - ax0) * mo / 30  # noqa: E731
        marks = []
        for mo, date, name, above in RELEASES:
            x = x_of(mo)
            dot = m.Dot([x, ay, 0], radius=0.08, color=TIME)
            lab = m.VGroup(code(date, 20, TIME), text(name, 20, INK)).arrange(m.DOWN, buff=0.06).set_x(x)
            sit(lab[0], ay + (0.62 if above else -0.42))
            sit(lab[1], ay + (0.3 if above else -0.74))
            if lab.get_right()[0] > 6.3:
                lab.align_to([6.3, 0, 0], m.RIGHT)
            if lab.get_left()[0] < -6.3:
                lab.align_to([-6.3, 0, 0], m.LEFT)
            marks.append(m.VGroup(dot, lab))
        for side in (True, False):
            same = [mk for mk, r in zip(marks, RELEASES) if r[3] == side]
            assert all(a[1].get_right()[0] + 0.15 < b[1].get_left()[0] for a, b in zip(same, same[1:])), side
        assert max(mk[1].get_top()[1] for mk in marks) < data.get_bottom()[1] - 0.1
        assert min(mk[1].get_bottom()[1] for mk in marks) > -3.5

        self.say(
            "[Put the three networks together, and you have Stable Diffusion]: [a text encoder], [a denoiser], "
            "[and the autoencoder].",
            self.new_header("05 · stable diffusion", "Three networks, open weights"),
            m.FadeIn(cards[0], shift=0.1 * m.UP),
            m.FadeIn(cards[1], shift=0.1 * m.UP),
            m.FadeIn(cards[2], shift=0.1 * m.UP),
        )
        self.say(
            "[In our run, the denoiser has eight hundred sixty-six million weights], [the text encoder three hundred "
            "forty million], [the autoencoder eighty-four million].",
            m.AnimationGroup(m.GrowFromEdge(bars[1], m.LEFT), m.FadeIn(vals[1])),
            m.AnimationGroup(m.GrowFromEdge(bars[0], m.LEFT), m.FadeIn(vals[0])),
            m.AnimationGroup(m.GrowFromEdge(bars[2], m.LEFT), m.FadeIn(vals[2])),
        )
        self.say(
            "[Version one point five had a smaller text encoder], [and needed about two gigabytes of weights to make "
            "a picture].",
            m.LaggedStart(*[m.FadeIn(o, shift=0.1 * m.UP) for o in olds], lag_ratio=0.2),
            m.FadeIn(gb, shift=0.1 * m.UP),
        )
        self.say(
            "[The paper appeared in December twenty twenty-one]; "
            "[the open release followed in August twenty twenty-two].",
            m.AnimationGroup(m.Create(axis), m.FadeIn(marks[0], shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(marks[1], shift=0.1 * m.DOWN), m.Flash(marks[1][0], color=TIME)),
        )
        self.say(
            "[It learned from subsets of a public collection of billions of captioned web images]: "
            "[more on that in episode twenty].",
            m.FadeIn(data, shift=0.1 * m.UP),
            m.Circumscribe(data, color=PROMPT, buff=0.08),
        )
        self.say(
            "[Our version, two point one, came out in December twenty twenty-two].",
            m.AnimationGroup(m.FadeIn(marks[2], shift=0.1 * m.UP), m.Flash(marks[2][0], color=TIME)),
        )
        self.say(
            "[Later came a two point six billion denoiser in SDXL], "
            "[and a transformer in version three, with sixteen latent channels].",
            m.FadeIn(marks[3], shift=0.1 * m.DOWN),
            m.FadeIn(marks[4], shift=0.1 * m.UP),
        )
        self.wait(0.5)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes=(
            "Squeeze (the encoder: 512×512×3 → 64×64×4, 48× fewer numbers), diffuse (every denoising step in the latent: "
            "64× fewer positions per layer), decode (the decoder paints the pixels once). Our lighthouse: 50 guided steps in "
            f"{TOT_S:.2f} s on this Mac (mac_runs §5). Module 5: 17 latent diffusion, 18 diffusion transformers, 19 measuring "
            "pictures, 20 data and alignment. Next: Episode 18."))
        halo = glow(DATA, radius=4.0)
        words = m.VGroup(*[title_text(w, 58, c) for w, c in (("Squeeze", DATA), ("Diffuse", MODEL), ("Decode", DATA))])
        subs = m.VGroup(*[code(s, 20, MUTED) for s in ("512×512×3 → 64×64×4", "every step in the latent",
                                                       "pixels once, at the end")])
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
        mod = m.VGroup(pill("17 · latent diffusion", DATA, TEAL_FILL, 20), pill("18 · transformers", DATA, CARD, 20),
                       pill("19 · measuring pictures", DATA, CARD, 20), pill("20 · data", DATA, CARD, 20))
        for p in mod[1:]:
            p[0].set_stroke(DATA, 1.5, opacity=0.55)
            p[1].set_color(MUTED)
        mod.arrange(m.RIGHT, buff=0.25).move_to([0, -0.85, 0])
        mod_t = text("Module 5 · Scaling up", 24, INK, weight=m.BOLD).next_to(mod, m.UP, buff=0.3)
        assert mod.width < 12.8
        mascot = pixel(0.42)
        nxt = pill("NEXT · EPISODE 18 · DIFFUSION TRANSFORMERS", DATA, TEAL_FILL, 20, mono=True)
        foot = m.VGroup(mascot, nxt).arrange(m.RIGHT, buff=0.3).move_to([0, -2.4, 0])

        self.say(
            "[To sum up]: [squeeze], [diffuse], [decode].",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.3 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(dots[0]), m.FadeIn(words[1], shift=0.3 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(dots[1]), m.FadeIn(words[2], shift=0.3 * m.UP), m.FadeIn(subs[2])),
        )
        self.say(
            "[Squeeze the picture forty-eight times], [do every denoising step in that small space], "
            "[and paint the pixels once, at the end].",
            m.Circumscribe(words[0], color=DATA, buff=0.1),
            m.Circumscribe(words[1], color=MODEL, buff=0.1),
            m.Circumscribe(words[2], color=DATA, buff=0.1),
        )
        self.say(
            "[That is how our lighthouse took fifty steps in about seven seconds, on this Mac].",
            m.AnimationGroup(m.FadeIn(mod_t, shift=0.1 * m.UP), m.FadeIn(mod, shift=0.1 * m.UP)),
        )
        self.say(
            "[Next in module five], [the denoiser itself changes: transformers, working on patches of the latent].",
            m.Circumscribe(mod[1], color=DATA, buff=0.08),
            m.AnimationGroup(m.FadeIn(mascot, shift=0.2 * m.UP), m.FadeIn(nxt, shift=0.2 * m.UP)),
        )
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, mod, mod_t, foot)], run_time=0.9)
