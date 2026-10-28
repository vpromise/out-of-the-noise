"""Out of the Noise · Episode 21 · From Pictures to Video (Module 6 · Video and Your Mac — the module's first episode)

A video is a stack of frames, diffused together: one block of frames × height × width × channels. Episode 4's forward
process noises the whole block at once (q_sample works on any shape) and Episode 7's loop denoises the whole block at
once: the same loop, one more axis. Then the architectures, dated: Video Diffusion Models (Ho et al., Apr 2022: a
three-dimensional U-Net, 1×3×3 convolutions, factorised attention — spatial within a frame, temporal across frames at
each position), the 2022 cascades (Imagen Video, Make-A-Video), latent video (2023–: Episode 17's autoencoder squeezing
time as well), spacetime patches (Episode 18's patches with a time axis; the Sora report of 15 Feb 2024, quoted
exactly, with its own limitations), Movie Gen and the open-weight models (dated October 2026). Then the cost: tokens grow
with frames and attention pairs with their square (Episode 8's argument), computed for clip sizes and for named models
with their own compression factors. Last, our own tiny clip, honestly.

Running example: the digit model (research/experiments/mac_runs.md §6, arrays/digit_video_noise_correlation.npz): clip 7
of 20 (label 7), 8 frames, DDIM 50, w = 1, every frame from one shared starting noise (ρ = 1), each frame pasted 3 px
further right on a 28 × 52 canvas — the MOTION IS PASTED ON, the model never moves anything, and each frame is generated
on its own as a picture (this is not a video model). One glimpse of the same clip with a fresh noise per frame (ρ = 0),
flickering, as the pointer to Episode 22 (the correlation trick itself is Episode 22's and is not taught here). The
jointly-denoised video toys of the GPU run FAILED (undertrained) and are quarantined: never shown, never quoted.

Every number on screen is computed below (diffusion_reference, the run's json and arrays, the notes' figures with the
token formula recomputed) and asserted, or cited with its source.

Sources (research/notes.md §21, §1, Appendix A rows 21, Appendix C row 21; read October 2026):
- Ho, Salimans, Gritsenko, Chan, Norouzi & Fleet, "Video Diffusion Models", arXiv:2204.03458 (Apr 2022): "we change each
  3x3 convolution into a 1x3x3 convolution"; temporal attention "over the first axis" with the spatial axes as batch;
  base 16 × 64 × 64. Factorised cost ∝ T·(HW)² + HW·T² instead of (T·HW)² (notes §21 key formulas).
- Imagen Video, Ho et al. (Google), arXiv:2210.02303 (5 Oct 2022): 1280×768 at 24 fps, 128 frames (≈ 5.3 s); 1 base +
  3 SSR + 3 TSR = 7 video diffusion models, 11.6B diffusion parameters; base 16×40×24 at 3 fps (Appendix A: "7
  diffusion models + a text encoder").
- Make-A-Video, Singer et al. (Meta), arXiv:2209.14792 (29 Sep 2022): "it does not require paired text-video data";
  16 frames at 64×64 → 76 frames → 256² → 768².
- Video LDM (Apr 2023) and Stable Video Diffusion (Nov 2023): temporal layers in Stable Diffusion's U-Net (2023 latents).
- Sora report, OpenAI, "Video generation models as world simulators", 15 Feb 2024 (read by the notes via a Wayback
  capture; openai.com returns 403 — re-tried for this episode on 2026-10-27, still 403): the four passages quoted
  exactly. No capability claims beyond the report.
- Movie Gen, Meta, arXiv:2410.13720 (submitted 17 Oct 2024 — re-checked on arXiv for this episode): 30B transformer,
  73K video tokens = 16 s at 16 fps; TAE 8× in t, h, w, 16 channels; 2×2 patches; 768² × 256 frames → 73,728 tokens;
  flow matching; 7B upsampler to 1080p; up to 6,144 H100 GPUs.
- Open-weight table (notes §21, [DATED], October 2026): CogVideoX (Aug 2024), HunyuanVideo (3 Dec 2024), Wan 2.1
  (25 Feb 2025), LTX-2 (announced 23 Oct 2025); Veo 3 (20 May 2025, audio); Wan 3.0 (API, GA 24 Aug 2026, up to 30 s).
- Token arithmetic (notes §21 worked examples, recomputed here): tokens = T_lat × (H/(s·p)) × (W/(s·p)), T_lat = 1 +
  (T − 1)/c_t for causal VAEs.

Illustrative (labelled on screen or here): the clean clip "emerging" in section 1 is our clip's own forward noise taken
away again (no video model was run); the 4 × 4-cell frames of the attention sketch (cells shaded by the 7's 7×7 means);
the slab decks of the cascade and latent sections (counts not to scale); the shaded cells of the token chip; the clip
sizes of the cost section (512 × 512 frames through Episode 17's 8× autoencoder, 2 × 2 patches, 24 fps — assumptions,
labelled).

    VOICE=draft manimgx inspect scene.py      # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403
from diffusion_reference import *  # noqa: E402,F403

EP_DIR = Path(__file__).resolve().parent          # set AFTER the import: diffusion_reference exports HERE (the series)
PREFIX = "ep21"
ARR = EXPERIMENTS / "arrays"
PX = 135                                          # pixels per scene unit at 1080p
NOTES = (EP_DIR.parent / "research" / "notes.md").read_text()
MAC_MD = (EXPERIMENTS / "mac_runs.md").read_text()


def need(*snippets):
    """Every cited figure must be in the research notes as written."""
    for s in snippets:
        assert s in NOTES, s


# ---------------------------------------------------------------------------- our clip (mac_runs §6)

VID = measured("mac_runs", "video_toy_noise_correlation")
assert VID["frames"] == 8 and VID["canvas"] == [28, 52] and VID["px_per_frame"] == 3 and VID["n_clips"] == 20
assert VID["sampler"] == "DDIM 50, w=1" and "digit DDPM U-Net" in VID["model"]
assert "Motion is IMPOSED by pasting frame k 3 px further right" in VID["note"]
assert "**Motion is imposed, not generated:** frame k is pasted 3 px further right on a 28x52 canvas." in MAC_MD
assert "Do not claim this toy is a video model." in MAC_MD
_Z = np.load(ARR / "digit_video_noise_correlation.npz")
assert "label = clip % 10" in str(_Z["note"]) and "frame k pasted at x = 3k" in str(_Z["note"])
CLIP = 7
assert int(_Z["labels"][CLIP]) == 7 and list(_Z["rhos"]) == [0.0, 0.5, 0.9, 0.99, 1.0]
SHARED, OWN, DIG = _Z["rho1_canvas"][CLIP], _Z["rho0_canvas"][CLIP], _Z["rho1_frames"][CLIP]
OWN_DIG = _Z["rho0_frames"][CLIP]
N_FR, CH, CW = SHARED.shape
SHIFT = VID["px_per_frame"]
assert (N_FR, CH, CW, SHIFT) == (8, 28, 52, 3)
for _k in range(N_FR):                            # the motion is pasted: frame k is the 28×28 digit at x = 3k, nothing else
    assert np.array_equal(SHARED[_k][:, SHIFT * _k:SHIFT * _k + 28], DIG[_k]) and int(SHARED[_k].sum()) == int(DIG[_k].sum())
    assert np.array_equal(DIG[_k], DIG[0])        # one shared noise: the same 7 in every frame
    assert np.array_equal(OWN[_k][:, SHIFT * _k:SHIFT * _k + 28], OWN_DIG[_k])
assert all(np.abs(OWN_DIG[_k + 1].astype(int) - OWN_DIG[_k]).mean() > 20 for _k in range(N_FR - 1))   # ρ = 0: a new 7 each frame
CLIP_NUMBERS = N_FR * CH * CW
assert CLIP_NUMBERS == 11_648 and 28 * 28 == 784                                   # "eleven thousand six hundred forty-eight"
GEN_CITE = f"generated on this Mac · the digit model · DDIM 50 · w = 1 · label {CLIP} · {N_FR} frames"

# Episode 4's forward process on the whole block (the model's range is [−1, 1])
X0 = SHARED.astype(np.float64) / 127.5 - 1.0
EPS = np.random.default_rng(21).standard_normal(X0.shape)
LEVELS = np.arange(0, T + 1, 50)
AB = alpha_bar_at(LEVELS)
assert LEVELS[-1] == T == 1000 and abs(AB[0] - 1) < 1e-12 and AB[-1] < 1e-4      # "turn t up to a thousand"
assert np.allclose(q_sample(X0, 500, EPS)[3], q_sample(X0[3], 500, EPS[3]))      # same jump, any shape


def shade(ab):
    """The signal's share of the variance: TEAL when clean, VIOLET when it is all noise (as in Episode 17)."""
    return mix(NOISE, DATA, float(ab))


NOISY = [[tint(np.clip((q_sample(X0[k], t, EPS[k]) + 1) / 2, 0, 1), shade(a), 0, 1) for t, a in zip(LEVELS, AB)]
         for k in range(N_FR)]
CLEAN_RGB = [tint(SHARED[k], DATA, 0, 255) for k in range(N_FR)]
OWN_RGB = [tint(OWN[k], DATA, 0, 255) for k in range(N_FR)]

# ---------------------------------------------------------------------------- 2022: factorised attention (Ho et al.)

need("Video Diffusion Models", "arXiv:2204.03458, Apr 2022", "we change each 3x3 convolution into a 1x3x3 convolution",
     "Base 16×64×64", "after each spatial attention block, we insert a temporal attention block",
     "cost ∝ T·(HW)² + HW·T² instead of (T·HW)² for full attention")
G_FR, G_SIDE = 4, 4
G_POS = G_SIDE * G_SIDE
SPATIAL, TEMPORAL, FULL = G_FR * G_POS ** 2, G_POS * G_FR ** 2, (G_FR * G_POS) ** 2
assert (SPATIAL, TEMPORAL, SPATIAL + TEMPORAL, FULL) == (1024, 256, 1280, 4096)  # "one thousand two hundred eighty … four thousand ninety-six"
POOL = DIG[0].reshape(G_SIDE, 7, G_SIDE, 7).mean(axis=(1, 3)) / 255.0            # the 7, 7×7 means (illustrative shading)
PICK = (1, 2)

# cascades (2022)
need("arXiv:2210.02303, 5 Oct 2022", "generates high definition 1280×768 (width × height) videos at 24 frames per second, "
     "for 128 frames (≈ 5.3 seconds)", "for a total of 7 video diffusion models, with a total of 11.6B diffusion model "
     "parameters", "Base 5.6B at 16×40×24 @ 3 fps", "arXiv:2209.14792, 29 Sep 2022",
     "it does not require paired text-video data", "16 frames at 64×64 → frame interpolation to 76 frames → 256² → 768²")
IV_FRAMES, IV_FPS = 128, 24
assert round(IV_FRAMES / IV_FPS, 1) == 5.3                                          # "about five seconds"

# ---------------------------------------------------------------------------- latents and spacetime patches


def latent_tokens(frames, h, w, ct, s, p, causal=True):
    """tokens = T_lat × (H/(s·p)) × (W/(s·p)); causal video VAEs keep the first frame on its own (notes §21)."""
    t_lat = 1 + (frames - 1) // ct if causal else frames // ct
    return t_lat * (h // (s * p)) * (w // (s * p))


need("| Wan 2.1 (Alibaba) | 25 Feb 2025 | 1.3B / 14B | 480p / 720p, 81 f @ 16 fps | 4×8×8, C 16, 1×2×2 | Apache 2.0 |",
     "81 × 720 × 1280 × 3 = 223,948,800 values → 21 × 90 × 160 × 16 = 4,838,400 latent values (46×)",
     "Video LDM", "Stable Video Diffusion")
WAN_F, WAN_H, WAN_W, WAN_CT, WAN_S, WAN_C, WAN_P = 81, 720, 1280, 4, 8, 16, 2
WAN_LF = 1 + (WAN_F - 1) // WAN_CT
PIX_VALUES = WAN_F * WAN_H * WAN_W * 3
LAT_VALUES = WAN_LF * (WAN_H // WAN_S) * (WAN_W // WAN_S) * WAN_C
SQUEEZE = PIX_VALUES / LAT_VALUES
assert (WAN_LF, PIX_VALUES, LAT_VALUES, round(SQUEEZE)) == (21, 223_948_800, 4_838_400, 46)   # "forty-six times fewer"
PATCH_PX = WAN_S * WAN_P                                                            # one latent patch covers 16 × 16 px
PATCH_PIX = WAN_CT * PATCH_PX ** 2 * 3
PATCH_LAT = 1 * WAN_P * WAN_P * WAN_C
assert (PATCH_PX, PATCH_PIX, PATCH_LAT) == (16, 3072, 64)                          # "four frames by sixteen by sixteen"

# the Sora report (15 Feb 2024), quoted exactly
Q_PATCH = ("We leverage a transformer architecture that operates on spacetime patches of video",
           "and image latent codes.")
Q_NONE = ("Model and implementation details are not included in this report.",)
Q_PATH = ("Our results suggest that scaling video generation models is a promising path towards",
          "building general purpose simulators of the physical world.")
Q_LIMIT = ("Sora currently exhibits numerous limitations as a simulator. For example, it does not",
           "accurately model the physics of many basic interactions, like glass shattering. Other",
           "interactions, like eating food, do not always yield correct changes in object state.")
for _q in (Q_PATCH, Q_NONE, Q_PATH, Q_LIMIT):
    need(" ".join(_q))
need("(OpenAI technical report \"Video generation models as world simulators\", 15 Feb 2024",
     "Sora report 15 Feb 2024")

# Movie Gen (Meta, Oct 2024) and the open-weight models (dated October 2026)
need("Our largest video generation model is a 30B parameter transformer trained with a maximum context length of 73K "
     "video tokens, corresponding to a generated video of 16 seconds at 16 frames-per-second.",
     "TAE compresses \"8× across each of the spatio-temporal dimensions\", C = 16; 2×2 spatial patches; \"768 × 768 px "
     "video with 256 frames\" → 73,728 tokens", "trained with **flow matching**", "\"up to 6,144 H100 GPUs\"",
     "7B spatial upsampler to 1080p")
MG_TOKENS = latent_tokens(256, 768, 768, 8, 8, 2, causal=False)
assert MG_TOKENS == 73_728 and 256 / 16 == 16                                       # "seventy-three thousand … sixteen seconds"
OPEN = [("CogVideoX", "Aug 2024", "2B / 5B", "720×480 · 6 s · 8 fps", "4 × 8 × 8"),
        ("HunyuanVideo", "Dec 2024", "13B", "1280×720 · 129 frames", "4 × 8 × 8"),
        ("Wan 2.1", "Feb 2025", "1.3B / 14B", "480p, 720p · 81 frames", "4 × 8 × 8"),
        ("LTX-2", "Oct 2025", "14B + 5B audio", "up to 4K · 50 fps", "not given")]
need("| CogVideoX 2B/5B | Aug 2024 | 2B / 5B | 720×480, 6 s @ 8 fps (v1.5: 1360×768, 16 fps) | 4×8×8 |",
     "| HunyuanVideo (Tencent) | 3 Dec 2024 | \"over 13 billion\" | 1280×720, 129 f @ 24 fps | 4×8×8, C 16, 1×2×2 |",
     "| LTX-2 | announced 23 Oct 2025; paper 6 Jan 2026 | 14B video + 5B audio streams | \"native 4K and up to 50 fps\"",
     "| 21 | HunyuanVideo size | own paper \"over 13 billion\"",
     "Veo 3 (20 May 2025: \"for the first time, can also generate videos with audio\"",
     "Wan 3.0 (API, GA 24 Aug 2026, up to 30 s)")
assert min(1.3, 2) >= 1 and max(5, 13, 14, 14) == 14                                # "from one to fourteen billion"

# ---------------------------------------------------------------------------- the cost (tokens and pairs)

LAT_SIDE, PATCH, FPS, SECS, SQ_T = 64, 2, 24, 5, 4          # assumed: Episode 17's 64×64 latent, 2×2 patches, 24 fps
TOK_FRAME = (LAT_SIDE // PATCH) ** 2
FRAMES = FPS * SECS
TOK_CLIP = FRAMES * TOK_FRAME
PAIRS = TOK_CLIP ** 2
TOK_SQ = (FRAMES // SQ_T) * TOK_FRAME
PAIRS_SQ = TOK_SQ ** 2
assert (TOK_FRAME, FRAMES, TOK_CLIP, TOK_SQ) == (1024, 120, 122_880, 30_720)       # "a hundred twenty-two thousand …"
assert (TOK_CLIP // TOK_FRAME) ** 2 == 14_400 and PAIRS // PAIRS_SQ == 16          # "fourteen thousand four hundred … sixteen"
assert f"{PAIRS / 1e9:.1f}" == "15.1" and f"{PAIRS_SQ / 1e9:.2f}" == "0.94"         # "about fifteen billion"
assert latent_tokens(1 + FRAMES - 1, 512, 512, SQ_T, 8, 2) == TOK_SQ               # a causal VAE gives the same 30 frames


def pairs_b(seconds, squeeze=1):
    return (FPS * seconds / squeeze * TOK_FRAME) ** 2 / 1e9


MODELS = [("image, 1024²", "SD3 · FLUX", latent_tokens(1, 1024, 1024, 1, 8, 2)),
          ("Wan 2.1", "480p · 5 s · 2025", latent_tokens(81, 480, 832, 4, 8, 2)),
          ("Movie Gen", "768² · 16 s · 2024", MG_TOKENS),
          ("Wan 2.1", "720p · 5 s · 2025", latent_tokens(81, 720, 1280, 4, 8, 2)),
          ("HunyuanVideo", "720p · 5.4 s · 2024", latent_tokens(129, 720, 1280, 4, 8, 2))]
assert [v for _, _, v in MODELS] == [4096, 32_760, 73_728, 75_600, 118_800]
need("| SD3/FLUX image 1024² (for comparison) | 64 × 64 | 4,096 |", "| Same at 480p (832×480) | 21 × 30 × 52 | 32,760",
     "= 21 × 45 × 80 | **75,600** |", "| HunyuanVideo 1280×720 × 129 f | 33 × 45 × 80 | **118,800** |",
     "| Movie Gen 768² × 256 f (16 s @ 16 fps) | 32 × 48 × 48 | **73,728**")
assert round(MODELS[1][2] / 1000) == 33 and round(MODELS[-1][2] / 1000) == 119      # "thirty-three … a hundred nineteen thousand"
assert round(MODELS[0][2] / 1000) == 4                                              # "four thousand for a large image"

# ---------------------------------------------------------------------------- look

FLAT = set("abcdefhiklmnorstuvwxzABCDEFGHIJKLMNOPRSTUVWXYZ0123456789")


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


_TEX = {}


def texture_of(rgb, px, nearest=False):
    """An RGB array → an RGBA texture about px pixels wide (small arrays blown up with crisp square pixels)."""
    key = (id(rgb), px, nearest)
    if key in _TEX:
        return _TEX[key][1]
    h, w = rgb.shape[:2]
    if nearest:
        k = max(1, int(np.ceil(2 * px / w)))
        arr = np.repeat(np.repeat(rgb, k, axis=0), k, axis=1)
    elif px != w:
        arr = np.asarray(Image.fromarray(rgb).resize((px, max(1, round(px * h / w))), Image.LANCZOS))
    else:
        arr = rgb
    out = np.concatenate([arr, np.full(arr.shape[:2] + (1,), 255, np.uint8)], axis=-1)
    out.flags.writeable = False
    _TEX[key] = (rgb, out)
    return out


def photo(rgb, height):
    """A picture `height` units tall from an RGB array, crisp pixels."""
    image = m.ImageMobject(texture_of(rgb, int(np.ceil(height * PX * rgb.shape[1] / rgb.shape[0])), True))
    image.height = height
    return image


class Reel:
    """A picture that steps through frames: `.image`, `.show(k)` (as in Episode 17)."""

    def __init__(self, frames, height, k=0):
        px = int(np.ceil(height * PX * frames[0].shape[1] / frames[0].shape[0]))
        self.tex = [texture_of(f, px, True) for f in frames]
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


def play_clip(reel, loops=2, fps=4.0):
    """Step a Reel through its frames at `fps`, `loops` times round."""
    n = len(reel.tex)

    def step(_, a):
        reel.show(min(int(a * n * loops), n * loops - 1) % n)

    return m.UpdateFromAlphaFunc(reel.image, step, run_time=n * loops / fps, rate_func=m.linear)


def flash(mob, color=DATA, width=7, time_width=0.8, run_time=0.9):
    return m.ShowPassingFlash(mob.copy().set_stroke(color, width), time_width=time_width, run_time=run_time)


def later(anim):
    """Keep play() from showing this animation's object before its cue (gotcha 21)."""
    anim.introducer = True
    return anim


def arc_up(a, b, color, width=3, opacity=0.9, bend=0.8):
    angle = -bend if b[0] > a[0] else bend
    return m.ArcBetweenPoints(a, b, angle=angle).set_stroke(color, width, opacity=opacity)


def ellipsis(color=MUTED, center=(0, 0, 0), r=0.04, gap=0.13):
    return m.VGroup(*[m.Dot(radius=r, color=color) for _ in range(3)]).arrange(m.RIGHT, buff=gap - 2 * r).move_to(
        np.array([*center, 0.0][:3], dtype=float))


def v3(p):
    return np.array([*p, 0.0][:3], dtype=float)


def clip_deck(images, front, step):
    """A clip as one block: frame 1 in front (lower left), later frames behind it (up and right). `images` are
    ImageMobjects in frame order. Returns a Group (back to front) with .tiles[k] = framed frame k."""
    tiles = [None] * len(images)
    for k in range(len(images) - 1, -1, -1):
        fr = framed(images[k], DATA if k == 0 else mix(BG, DATA, 0.55), buff=0.03, stroke=2)
        tiles[k] = fr.move_to(v3(front) + k * v3(step))
    deck = m.Group(*tiles[::-1])
    deck.tiles = tiles
    return deck


def slab_deck(n, w, h, front, step, color=DATA, stroke=2):
    """Plain slabs (frames of a video we don't draw), back to front; slabs[k] is k-th from the front."""
    slabs = [None] * n
    for k in range(n - 1, -1, -1):
        r = m.Rectangle(width=w, height=h).set_stroke(color if k == 0 else mix(BG, color, 0.6), stroke)
        slabs[k] = r.set_fill(FILLS[color], 1).move_to(v3(front) + k * v3(step))
    g = m.VGroup(*slabs[::-1])
    g.slabs = slabs
    return g


def quote_card(lines, width, color=EDGE, size=22, t2c=None):
    words = text("\n".join(lines), size, INK, t2c=t2c or {})
    box = card(width, words.height + 0.5, color, CARD, radius=0.16, stroke=2)
    words.move_to(box).align_to(box, m.LEFT).shift(0.35 * m.RIGHT)
    return m.VGroup(box, words)


def code_row(label, expr, result, color, y, sign="=", x0=-6.3, size=20):
    """One line of arithmetic: a MUTED label, the expression in INK, the result in its colour (mono, aligned)."""
    row = code(f"{label:<11}{expr:<14}{sign} {result}", size, INK,
               t2c={rf"^{re.escape(label)}": MUTED, rf"{re.escape(result)}$": color})
    return row.move_to([0, y, 0]).align_to([x0, 0, 0], m.LEFT)


def hbar(value, vmax, length, color, y, x0, height=0.3):
    w = max(length * value / vmax, 0.03)
    return m.Rectangle(width=w, height=height).set_stroke(color, 2).set_fill(FILLS[color], 1).move_to([x0 + w / 2, y, 0])


# ---------------------------------------------------------------------------- the episode


class Ep21FromPicturesToVideo(Episode):
    number = 21

    def lesson(self):
        self.hook()
        self.cube()
        self.factorised()
        self.cascades()
        self.latents()
        self.sora()
        self.today()
        self.cost()
        self.on_this_mac()
        self.recap()

    # ------------------------------------------------------------------ 0 · hook

    def hook(self):
        self.next_section("Hook", notes=(
            "Episodes 1–20 made one picture: height × width × channels (our digit 28 × 28 × 1). A video adds a time "
            f"axis: frames. Shown: the digit model's clip {CLIP} (mac_runs §6; {GEN_CITE}); frame k is pasted "
            f"{SHIFT} px further right on a 28 × 52 canvas — said honestly in section 8."))
        one = framed(photo(tint(DIG[0], DATA, 0, 255), 2.3), DATA, buff=0.04, stroke=2.5).move_to([0, 0.45, 0])
        one_l = m.VGroup(code("28 × 28 × 1", 24, INK), cite("height × width × channels")).arrange(m.DOWN, buff=0.12)
        one_l.next_to(one, m.DOWN, buff=0.22)
        strip_ = m.Group(*[framed(photo(CLEAN_RGB[k], 0.74), DATA, buff=0.03, stroke=2) for k in range(N_FR)])
        strip_.arrange(m.RIGHT, buff=0.14).move_to([0, 0.55, 0])
        nums = m.VGroup(*[code(str(k + 1), 20, MUTED).next_to(strip_[k], m.UP, buff=0.14) for k in range(N_FR)])
        t_ax = m.Arrow([strip_.get_left()[0], -0.2, 0], [strip_.get_right()[0], -0.2, 0], buff=0, color=TIME,
                       stroke_width=4, max_tip_length_to_length_ratio=0.03)
        t_l = text("time", 24, TIME, weight=m.BOLD).next_to(t_ax, m.DOWN, buff=0.14)
        assert strip_.get_left()[0] > -6.5 and strip_.get_right()[0] < 6.5, strip_.get_right()
        mascot = pixel(0.42)
        chips = m.VGroup(pill("one block, denoised together", DATA, TEAL_FILL, 20),
                         pill("the published video models", MODEL, AMBER_FILL, 20),
                         pill("what time costs", TIME, BLUE_FILL, 20)).arrange(m.RIGHT, buff=0.3)
        foot = m.Group(mascot, chips).arrange(m.RIGHT, buff=0.35).move_to([0, -2.2, 0])
        assert foot.width < 12.8

        self.say(
            "[Every episode so far has made one picture]: "
            "[a grid of numbers, like our twenty-eight by twenty-eight digit].",
            m.AnimationGroup(self.new_header("00 · hook", "One more axis"), m.FadeIn(one, scale=0.9)),
            m.FadeIn(one_l, shift=0.1 * m.UP),
        )
        self.say(
            "[A video is a stack of pictures]: [frames, one after another], [along a new axis, time].",
            m.AnimationGroup(m.FadeOut(one, target_position=strip_[0]), m.FadeOut(one_l),
                             m.LaggedStart(*[m.FadeIn(f, shift=0.15 * m.RIGHT) for f in strip_], lag_ratio=0.12)),
            m.LaggedStart(*[m.FadeIn(n_, shift=0.1 * m.DOWN) for n_ in nums], lag_ratio=0.1),
            m.AnimationGroup(m.GrowArrow(t_ax), m.FadeIn(t_l, shift=0.1 * m.RIGHT)),
        )
        self.say(
            "[Today: the whole stack at once], [how the big video models are built], [and why video costs so much].",
            m.AnimationGroup(m.FadeIn(mascot, shift=0.2 * m.UP), m.FadeIn(chips[0], shift=0.2 * m.UP)),
            m.FadeIn(chips[1], shift=0.2 * m.UP),
            m.FadeIn(chips[2], shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 1 · the cube

    def cube(self):
        self.next_section("1 · The cube", notes=(
            f"Our clip as one block: {N_FR} frames × {CH} × {CW} × 1 channel = {CLIP_NUMBERS:,} numbers (one digit: 784). "
            "Episode 4's forward process x_t = √ᾱ_t·x₀ + √(1−ᾱ_t)·ε applied to the whole block at once "
            "(diffusion_reference.q_sample, the course's linear schedule, T = 1000; ε seeded 21, one standard normal per "
            "number); colour from TEAL to VIOLET by ᾱ_t, the signal's share. The clean clip 'emerging' afterwards is "
            "ILLUSTRATIVE: the same noise taken away again — no video model was run; a real video model runs Episode "
            "7's loop on the whole block (notes §21: denoising all frames jointly is what makes them agree)."))
        w_f, h_f = 3.0, 3.0 * CH / CW
        front, step = (-4.45, -1.05), (0.19, 0.15)
        reels = [Reel(NOISY[k], h_f) for k in range(N_FR)]
        deck = clip_deck([r.image for r in reels], front, step)
        f0, f7 = deck.tiles[0], deck.tiles[-1]
        dim_w = m.VGroup(m.Line(f0.get_corner(m.DL) + [0, -0.18, 0], f0.get_corner(m.DR) + [0, -0.18, 0]).set_stroke(MUTED, 2),
                         code(f"{CW}", 20, MUTED))
        dim_w[1].next_to(dim_w[0], m.DOWN, buff=0.1)
        dim_h = m.VGroup(m.Line(f0.get_corner(m.UL) + [-0.18, 0, 0], f0.get_corner(m.DL) + [-0.18, 0, 0]).set_stroke(MUTED, 2),
                         code(f"{CH}", 20, MUTED))
        dim_h[1].next_to(dim_h[0], m.LEFT, buff=0.1)
        t_ax = m.Arrow(f0.get_corner(m.DR) + [0.2, -0.05, 0], f7.get_corner(m.DR) + [0.2, -0.05, 0], buff=0,
                       color=TIME, stroke_width=4, max_tip_length_to_length_ratio=0.15)
        t_l = text(f"{N_FR} frames", 22, TIME, weight=m.BOLD).next_to(t_ax.get_end(), m.RIGHT, buff=0.12)
        assert dim_h[1].get_left()[0] > -6.6 and t_l.get_right()[0] < 0.9, (dim_h[1].get_left(), t_l.get_right())
        shape = code(f"{N_FR} × {CH} × {CW} × 1", 30, INK, weight=m.BOLD, t2c={rf"^{N_FR}": TIME})
        shape_s = text("frames × height × width × channels", 20, MUTED, t2c={"frames": TIME})
        shape_g = m.VGroup(shape, shape_s).arrange(m.DOWN, buff=0.14).move_to([-3.75, 1.75, 0])
        assert shape_g.get_bottom()[1] > f7.get_top()[1] + 0.15, (shape_g.get_bottom(), f7.get_top())

        cx = 3.45
        big = title_text(f"{CLIP_NUMBERS:,}", 62, DATA)
        big_l = text("numbers in our clip", 22, INK)
        big_c = cite("one digit: 28 × 28 = 784")
        stat_ = m.VGroup(big, big_l, big_c).arrange(m.DOWN, buff=0.12).move_to([cx, 1.6, 0])
        formula = m.MathTex(r"x_t", r"=", r"\sqrt{\bar\alpha_t}", r"\,x_0", r"+", r"\sqrt{1-\bar\alpha_t}",
                            r"\,\varepsilon", font_size=40, color=INK).move_to([cx, 0.05, 0])
        formula[0].set_color(mix(NOISE, DATA, 0.5))
        formula[3].set_color(DATA)
        formula[6].set_color(NOISE)
        for i in (2, 5):
            formula[i].set_color(TIME)
        f_l = text("clean clip, noise, noisy clip: one shape", 20, MUTED).next_to(formula, m.DOWN, buff=0.2)
        f_c = cite("Episode 4: jump to any step in one go").next_to(f_l, m.DOWN, buff=0.1)
        left_x = 1.85
        cnt_state = {"i": 0}

        def t_words(i):
            return code(f"t = {LEVELS[i]}", 28, TIME, weight=m.BOLD).move_to([0, -1.55, 0]).align_to([left_x, 0, 0], m.LEFT)

        def s_words(i):
            return code(f"signal kept {100 * AB[i]:.1f}%", 22, DATA).move_to([0, -2.1, 0]).align_to([left_x, 0, 0], m.LEFT)

        cnt = t_words(0)
        sig = s_words(0)

        def set_level(v):
            i = int(np.clip(round(v), 0, len(LEVELS) - 1))
            for r in reels:
                r.show(i)
            if i != cnt_state["i"]:
                cnt_state["i"] = i
                cnt.become(t_words(i))
                sig.become(s_words(i))

        last = len(LEVELS) - 1
        # the loop (illustrative): noisy block → denoiser → a noise guess for every number → back
        den = m.VGroup(card(3.1, 1.25, MODEL, AMBER_FILL), text("denoiser", 28, INK, weight=m.BOLD),
                       text("sees all 8 frames at once", 20, MUTED))
        m.VGroup(den[1], den[2]).arrange(m.DOWN, buff=0.1).move_to(den[0])
        den.move_to([2.9, 1.25, 0])
        go = arrow([f7.get_right()[0] + 0.1, 0.55, 0], [den[0].get_left()[0] - 0.05, 1.1, 0], NOISE)
        guess = m.VGroup(text("noise guess", 20, MODEL), tex(r"\hat\varepsilon", 30, MODEL),
                         code(f"{N_FR} × {CH} × {CW}", 20, MODEL)).arrange(m.RIGHT, buff=0.15)
        guess.next_to(den, m.UP, buff=0.16).align_to(den, m.RIGHT)
        back = m.CurvedArrow(den[0].get_left() + [-0.05, -0.35, 0], [f7.get_right()[0] + 0.12, 0.05, 0],
                             angle=-0.35 * PI_).set_stroke(MUTED, 3)
        illus = cite("illustration: no video model was run")
        illus.move_to([0, -2.75, 0]).align_to([-6.3, 0, 0], m.LEFT)
        agree = pill("denoised together, the frames can agree", GOOD, GOOD_FILL, 22).move_to([cx - 0.3, -2.75, 0])
        assert guess.get_top()[1] < 2.6 and illus.get_right()[0] < agree.get_left()[0] - 0.2, (guess.get_top(), illus.get_right())
        assert agree.get_top()[1] < sig.get_bottom()[1] - 0.1 and agree.get_right()[0] < 6.5, agree.get_corner(m.UR)

        self.say(
            "[Stack the frames], [and a clip becomes one block of numbers]: [frames by height by width by channels].",
            m.AnimationGroup(self.new_header("01 · the cube", "Noise the whole block"),
                             m.LaggedStart(*[m.FadeIn(deck.tiles[k], shift=0.25 * m.DL) for k in range(N_FR - 1, -1, -1)],
                                           lag_ratio=0.12)),
            m.AnimationGroup(m.Create(dim_w[0]), m.FadeIn(dim_w[1]), m.Create(dim_h[0]), m.FadeIn(dim_h[1]),
                             m.GrowArrow(t_ax), m.FadeIn(t_l)),
            m.FadeIn(shape_g, shift=0.1 * m.UP),
        )
        self.say(
            f"[Our clip is eight frames of twenty-eight by fifty-two]: [eleven thousand six hundred forty-eight numbers], "
            "[against seven hundred eighty-four for one digit].",
            m.Circumscribe(shape, color=TIME, buff=0.1),
            m.AnimationGroup(m.FadeIn(big, scale=0.9), m.FadeIn(big_l)),
            m.FadeIn(big_c, shift=0.1 * m.UP),
        )
        self.say(
            "[The forward process of episode four doesn't care about shape]: "
            "[mix the block with noise of the same shape], [every number at once].",
            m.FadeIn(formula, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(f_l), m.FadeIn(f_c), m.Circumscribe(formula[6], color=NOISE, buff=0.06)),
            m.AnimationGroup(m.FadeIn(cnt), m.FadeIn(sig)),
        )
        self.say(
            "[Run the clock to step one thousand], [and all eight frames dissolve together].",
            m.Circumscribe(cnt, color=TIME, buff=0.08),
            drive(reels[0].image, 0, last, [set_level], run_time=3.0, rate=m.smooth),
        )
        self.say(
            "[To make a clip, run episode seven's loop on the whole block]: [the denoiser sees all eight frames], "
            "[and guesses the noise in every number].",
            m.AnimationGroup(m.FadeOut(m.VGroup(stat_, formula, f_l, f_c)), m.FadeIn(den, shift=0.1 * m.UP),
                             m.GrowArrow(go), m.FadeIn(illus)),
            m.AnimationGroup(flash(go, NOISE, 8), m.Create(back), m.FadeIn(guess, shift=0.1 * m.DOWN)),
            drive(reels[1].image, last, 0, [set_level], run_time=3.2, rate=m.smooth),
        )
        self.say(
            "[The same loop as before, with one more axis]. [Denoised together, the frames can agree with each other].",
            m.AnimationGroup(flash(go, NOISE, 8), flash(back, MUTED, 6), m.Circumscribe(t_ax, color=TIME, buff=0.08)),
            m.AnimationGroup(m.FadeIn(agree, shift=0.15 * m.UP), m.Circumscribe(deck, color=GOOD, buff=0.1)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 2 · factorised attention (2022)

    def factorised(self):
        self.next_section("2 · Space, then time", notes=(
            "Video Diffusion Models (Ho, Salimans, Gritsenko, Chan, Norouzi & Fleet, Google, arXiv:2204.03458, Apr 2022): "
            "a 3D U-Net, base 16 frames × 64 × 64; \"we change each 3x3 convolution into a 1x3x3 convolution\"; \"after "
            "each spatial attention block, we insert a temporal attention block that performs attention over the first "
            "axis and treats the spatial axes as batch axes\". Drawn on 4 frames × 4 × 4 positions (illustrative; cells "
            f"shaded by the 7's 7×7 means): spatial {G_FR} × {G_POS}² = {SPATIAL:,} pairs, temporal {G_POS} × {G_FR}² = "
            f"{TEMPORAL} pairs, together {SPATIAL + TEMPORAL:,}; all with all ({G_FR * G_POS})² = {FULL:,} (notes §21: "
            "cost ∝ T·(HW)² + HW·T² instead of (T·HW)²)."))
        size, gap = 0.42, 0.06
        xs = [-4.8, -1.6, 1.6, 4.8]
        gy = 0.15
        grids = m.VGroup()
        for x in xs:
            g = cell_grid(G_SIDE, G_SIDE, size, gap, EDGE, CARD).move_to([x, gy, 0])
            for r in range(G_SIDE):
                for c in range(G_SIDE):
                    g[r][c].set_fill(mix(CARD, DATA, 0.12 + 0.6 * float(POOL[r, c])), 1).set_stroke(mix(BG, DATA, 0.45), 1.5)
            grids.add(g)
        labels = m.VGroup(*[code(f"frame {i + 1}", 20, MUTED).next_to(grids[i], m.UP, buff=0.16) for i in range(G_FR)])
        src = 1
        pr, pc = PICK
        hub = grids[src][pr][pc].get_center()
        sel = m.Square(size + 0.04).set_stroke(INK, 4).set_fill(opacity=0).move_to(hub)
        rays = m.VGroup(*[m.Line(hub, grids[src][r][c].get_center()).set_stroke(MODEL, 2.5, opacity=0.9)
                          for r in range(G_SIDE) for c in range(G_SIDE) if (r, c) != PICK])
        same = m.VGroup(*[m.Square(size + 0.04).set_stroke(TIME, 4).set_fill(opacity=0).move_to(grids[i][pr][pc])
                          for i in range(G_FR) if i != src])
        tops = [grids[i][pr][pc].get_top() + [0, 0.02, 0] for i in range(G_FR)]
        arcs = m.VGroup(*[arc_up(tops[src], tops[i], TIME, 3.5, 0.95, bend=0.9 if abs(i - src) == 1 else 0.55)
                          for i in range(G_FR) if i != src])
        assert max(a.get_top()[1] for a in arcs) < labels.get_bottom()[1] - 0.05, (max(a.get_top()[1] for a in arcs), labels.get_bottom())
        wins = m.VGroup(*[m.Square(3 * size + 2 * gap + 0.06).set_stroke(MODEL, 3).set_fill(opacity=0).move_to(grids[i][pr][pc])
                          for i in range(G_FR)])
        t_ax = m.Arrow([-5.9, -1.0, 0], [6.3, -1.0, 0], buff=0, color=TIME, stroke_width=4,
                       max_tip_length_to_length_ratio=0.025)
        t_l = text("time", 22, TIME, weight=m.BOLD).next_to(t_ax.get_end(), m.DOWN, buff=0.24)
        t_l.align_to([6.3, 0, 0], m.RIGHT)
        sp = m.VGroup(text("spatial attention", 24, MODEL, weight=m.BOLD),
                      text("inside one frame: every position with every other", 20, MUTED)).arrange(m.DOWN, buff=0.1)
        tp = m.VGroup(text("temporal attention", 24, TIME, weight=m.BOLD),
                      text("across frames: one position, all its moments", 20, MUTED)).arrange(m.DOWN, buff=0.1)
        sp.move_to([-3.2, -1.65, 0])
        tp.move_to([3.2, -1.65, 0])
        sit(tp[0], baseline(sp[0]))
        sit(tp[1], baseline(sp[1]))
        top = m.VGroup(text("Video Diffusion Models", 22, INK, weight=m.BOLD),
                       cite("Ho et al. (Google), 2022 · 16 frames × 64 × 64")).arrange(m.RIGHT, buff=0.25)
        top.move_to([0, 2.3, 0]).align_to([-6.3, 0, 0], m.LEFT)
        conv = code("convolutions 1 × 3 × 3: inside one frame, never across", 20, MODEL).move_to([0, 1.88, 0])
        conv.align_to([-6.3, 0, 0], m.LEFT)
        assert conv.get_bottom()[1] > labels.get_top()[1] + 0.1, (conv.get_bottom(), labels.get_top())
        tally = m.VGroup(
            code(f"spatial {G_FR} × {G_POS}² = {SPATIAL:,}  +  temporal {G_POS} × {G_FR}² = {TEMPORAL}  "
                 f"=  {SPATIAL + TEMPORAL:,} pairs", 20, INK,
                 t2c={r"^spatial": MODEL, r"temporal": TIME, rf"{SPATIAL + TEMPORAL:,} pairs": GOOD}),
            code(f"everything with everything: {G_FR * G_POS}² = {FULL:,} pairs", 20, MUTED,
                 t2c={rf"{FULL:,} pairs": BAD}),
            cite("this little grid, for example · the paper's cost: T·(HW)² + HW·T² instead of (T·HW)²"),
        ).arrange(m.DOWN, buff=0.14).move_to([0, -2.85, 0])
        assert tally.get_top()[1] < sp.get_bottom()[1] - 0.12 and tally.get_bottom()[1] > -3.55, (tally.get_top(), tally.get_bottom())
        assert tally.width < 12.8

        self.say(
            "[The first video diffusion models came in twenty twenty-two], [from a team at Google]: "
            "[a three-dimensional denoiser for sixteen frames].",
            m.AnimationGroup(self.new_header("02 · 2022", "Attention in space, then time"),
                             m.LaggedStart(*[m.FadeIn(g, shift=0.15 * m.RIGHT) for g in grids], lag_ratio=0.15)),
            m.AnimationGroup(m.FadeIn(top, shift=0.1 * m.UP), m.FadeIn(labels)),
            m.AnimationGroup(m.GrowArrow(t_ax), m.FadeIn(t_l)),
        )
        self.say(
            "[Its convolutions stay inside each frame], [so the frames only meet through attention], [which comes in two kinds].",
            m.AnimationGroup(m.FadeIn(conv), m.LaggedStart(*[m.Create(w_) for w_ in wins], lag_ratio=0.15)),
            m.Create(sel),
            m.LaggedStart(*[m.Indicate(g, color=INK, scale_factor=1.04) for g in grids], lag_ratio=0.15),
        )
        self.say(
            "[Spatial attention]: [inside one frame, every position looks at every other].",
            m.AnimationGroup(m.FadeIn(sp, shift=0.1 * m.UP), m.FadeOut(wins)),
            m.LaggedStart(*[m.Create(r_) for r_ in rays], lag_ratio=0.05, run_time=1.6),
        )
        self.say(
            "[Temporal attention]: [one position looks along time], [at the same spot in every frame].",
            m.AnimationGroup(m.FadeIn(tp, shift=0.1 * m.UP), m.FadeOut(rays)),
            m.LaggedStart(*[m.Create(a_) for a_ in arcs], lag_ratio=0.25, run_time=1.4),
            m.AnimationGroup(m.LaggedStart(*[m.Create(s_) for s_ in same], lag_ratio=0.2),
                             m.Circumscribe(t_ax, color=TIME, buff=0.06)),
        )
        self.say(
            "[On this little grid, that makes one thousand two hundred eighty pairs], "
            "[where everything with everything would make four thousand ninety-six].",
            m.AnimationGroup(m.FadeIn(tally[0], shift=0.1 * m.UP), m.FadeIn(rays)),
            m.AnimationGroup(m.FadeIn(tally[1], shift=0.1 * m.UP), m.FadeIn(tally[2])),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 3 · cascades (2022)

    def cascades(self):
        self.next_section("3 · Cascades", notes=(
            "Imagen Video (Ho et al., Google, arXiv:2210.02303, 5 Oct 2022): 1 frozen text encoder + 1 base + 3 SSR + 3 "
            "TSR = 7 video diffusion models, 11.6B diffusion parameters in total; base 5.6B at 16 × 40 × 24 at 3 fps; "
            "output 1280 × 768 at 24 fps, 128 frames (≈ 5.3 s). The order of the stages is not drawn (the paper "
            "interleaves them). Make-A-Video (Singer et al., Meta, arXiv:2209.14792, 29 Sep 2022): 16 frames at 64 × 64 "
            "→ interpolation to 76 frames → 256² → 768²; \"it does not require paired text-video data\". Slab counts "
            "are not to scale (illustrative)."))
        base = slab_deck(6, 0.8, 0.48, (-5.15, 0.55), (0.08, 0.06), DATA)
        base_l = m.VGroup(code("16 frames", 20, INK), code("40 × 24 · 3 fps", 20, MUTED)).arrange(m.DOWN, buff=0.1)
        base_l.next_to(base, m.DOWN, buff=0.25)
        c_t = pill("3 models add frames", TIME, BLUE_FILL, 22).move_to([-1.05, 1.15, 0])
        c_s = pill("3 models add pixels", DATA, TEAL_FILL, 22).move_to([-1.05, 0.05, 0])
        final = slab_deck(10, 2.3, 1.38, (3.25, 0.1), (0.09, 0.065), DATA)
        final_l = m.VGroup(code("128 frames · ≈ 5.3 s", 20, INK), code("1280 × 768 · 24 fps", 20, MUTED)).arrange(m.DOWN, buff=0.1)
        final_l.next_to(final, m.DOWN, buff=0.25).set_x(final.slabs[0].get_x() + 0.4)
        a1 = arrow(base.get_right() + [0.1, 0.2, 0], c_t.get_left() + [-0.05, 0, 0], TIME)
        a2 = arrow(base.get_right() + [0.1, -0.15, 0], c_s.get_left() + [-0.05, 0, 0], DATA)
        a3 = arrow(c_t.get_right() + [0.05, 0, 0], final.slabs[0].get_left() + [-0.1, 0.35, 0], TIME)
        a4 = arrow(c_s.get_right() + [0.05, 0, 0], final.slabs[0].get_left() + [-0.1, -0.2, 0], DATA)
        head = text("Imagen Video · Google, Oct 2022", 22, INK, weight=m.BOLD).move_to([0, 2.3, 0]).align_to([-6.3, 0, 0], m.LEFT)
        head_r = cite("1 base + 3 + 3 = 7 video diffusion models · 11.6B parameters")
        head_r.move_to([0, 2.3, 0]).align_to([6.35, 0, 0], m.RIGHT)
        sit(head_r, baseline(head))
        assert head.get_right()[0] < head_r.get_left()[0] - 0.3
        mav_box = card(12.7, 1.5, EDGE, CARD, radius=0.16, stroke=2).move_to([0, -2.55, 0])
        mav = m.VGroup(text("Make-A-Video · Meta, Sep 2022", 22, INK, weight=m.BOLD),
                       code("16 frames at 64 × 64  →  76 frames  →  768 × 768", 20, INK),
                       text("learned motion from video without captions: “it does not require paired text-video data”",
                            20, MUTED)).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        mav.move_to(mav_box).align_to(mav_box, m.LEFT).shift(0.35 * m.RIGHT)
        mav_c = cite("Singer et al., 2022").move_to(mav_box.get_corner(m.UR) + [-1.25, -0.3, 0])
        sit(mav_c, baseline(mav[0]))
        mav_g = m.VGroup(mav_box, mav, mav_c)
        assert mav.width < 12.0 and final_l.get_bottom()[1] > mav_box.get_top()[1] + 0.12, (mav.width, final_l.get_bottom())
        assert base_l.get_bottom()[1] > mav_box.get_top()[1] + 0.12

        self.say(
            "[Later that year, Google and Meta built cascades]: [a small, short video first], "
            "[then models that add frames], [and models that add pixels].",
            m.AnimationGroup(self.new_header("03 · cascades", "Small first, then bigger"), m.FadeIn(head),
                             m.FadeIn(mav_g, shift=0.1 * m.UP)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(s_) for s_ in base.slabs[::-1]], lag_ratio=0.1), m.FadeIn(base_l)),
            m.AnimationGroup(m.GrowArrow(a1), m.FadeIn(c_t, shift=0.1 * m.RIGHT)),
            m.AnimationGroup(m.GrowArrow(a2), m.FadeIn(c_s, shift=0.1 * m.RIGHT)),
        )
        self.say(
            "[The chain from Google used seven diffusion models, eleven point six billion parameters], "
            "[for about five seconds of video].",
            m.AnimationGroup(m.FadeIn(head_r), m.Circumscribe(m.VGroup(c_t, c_s), color=MODEL, buff=0.1)),
            m.AnimationGroup(m.GrowArrow(a3), m.GrowArrow(a4),
                             m.LaggedStart(*[m.FadeIn(s_) for s_ in final.slabs[::-1]], lag_ratio=0.08),
                             m.FadeIn(final_l, shift=0.1 * m.UP)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 4 · latents and spacetime patches

    def latents(self):
        self.next_section("4 · Latent video, spacetime patches", notes=(
            "From 2023, video models work in latents (Video LDM, Apr 2023; Stable Video Diffusion, Nov 2023: temporal "
            "layers in SD's U-Net); from 2024, video VAEs squeeze time too and a transformer reads spacetime patches. "
            f"Numbers: Wan 2.1 (Alibaba, 25 Feb 2025; notes §21): 4 × 8 × 8 VAE, 16 channels, 1 × 2 × 2 patches; 81 "
            f"frames × 720 × 1280 × 3 = {PIX_VALUES:,} values → {WAN_LF} × 90 × 160 × 16 = {LAT_VALUES:,} "
            f"({SQUEEZE:.1f}×, causal: the first frame is kept on its own). One latent patch = 1 × 2 × 2 × 16 = "
            f"{PATCH_LAT} numbers covering up to 4 frames × 16 × 16 px × 3 = {PATCH_PIX:,} pixel values. Slab decks "
            "are not to scale; the token's shading is illustrative. Episode 18 = patches as tokens."))
        px_deck = slab_deck(12, 2.2, 1.24, (-5.1, 0.35), (0.1, 0.075), DATA)
        px_l = m.VGroup(code("81 × 720 × 1280 × 3", 20, INK), cite("frames × height × width × colours"))
        px_l.arrange(m.DOWN, buff=0.1).next_to(px_deck, m.DOWN, buff=0.22).set_x(px_deck.slabs[0].get_x() + 0.4)
        fx, fy = -1.3, 0.85
        funnel = m.VGroup(m.Polygon([fx - 0.7, fy + 0.85, 0], [fx + 0.7, fy + 0.42, 0], [fx + 0.7, fy - 0.42, 0],
                                    [fx - 0.7, fy - 0.85, 0], stroke_color=DATA, stroke_width=2.5, fill_color=TEAL_FILL,
                                    fill_opacity=1),
                          text("encoder", 20, INK, weight=m.BOLD).move_to([fx, fy, 0]))
        f_cap = cite("Episode 17's autoencoder, squeezing time too").move_to([fx + 0.5, 2.05, 0])
        lat = slab_deck(4, 1.3, 0.73, (1.55, 0.6), (0.17, 0.13), DATA, stroke=2.5)
        lat_grid = m.VGroup()
        s0 = lat.slabs[0]
        for i in range(1, 8):
            x = s0.get_left()[0] + i * s0.width / 8
            lat_grid.add(m.Line([x, s0.get_bottom()[1], 0], [x, s0.get_top()[1], 0]))
        for j in range(1, 4):
            y = s0.get_bottom()[1] + j * s0.height / 4
            lat_grid.add(m.Line([s0.get_left()[0], y, 0], [s0.get_right()[0], y, 0]))
        lat_grid.set_stroke(mix(BG, DATA, 0.85), 1.6).set_z_index(2)
        lat_l = m.VGroup(code(f"{WAN_LF} × 90 × 160 × {WAN_C}", 20, INK), cite("latent frames × h × w × channels"))
        lat_l.arrange(m.DOWN, buff=0.1).next_to(lat, m.DOWN, buff=0.22)
        sit(lat_l[0], baseline(px_l[0]))
        sit(lat_l[1], baseline(px_l[1]))
        go = arrow([px_deck.get_right()[0] + 0.05, fy, 0], [fx - 0.75, fy, 0], DATA)
        go2 = arrow([fx + 0.75, fy, 0], [s0.get_left()[0] - 0.08, fy, 0], DATA)
        sq = m.VGroup(code("time ÷ 4", 22, TIME, weight=m.BOLD), code("space ÷ 8 × 8", 22, DATA, weight=m.BOLD))
        sq.arrange(m.DOWN, buff=0.08).move_to([fx, -1.5, 0])
        stat_ = m.VGroup(title_text(f"{round(SQUEEZE)}×", 60, DATA), text("fewer numbers", 22, INK),
                         cite("Wan 2.1 · Feb 2025")).arrange(m.DOWN, buff=0.1).move_to([5.25, 0.75, 0])
        assert lat.get_right()[0] < stat_.get_left()[0] - 0.25, (lat.get_right(), stat_.get_left())
        assert sq.get_top()[1] < min(px_l.get_bottom()[1], lat_l.get_bottom()[1]) - 0.1 or (
            sq.get_left()[0] > px_l.get_right()[0] + 0.1 and sq.get_right()[0] < lat_l.get_left()[0] - 0.1), (sq.get_top(), lat_l.get_bottom())

        # the spacetime patch: a little box through four frames of the pixel deck, and its latent cells
        sp_ = 0.34
        p0 = px_deck.slabs[0].get_center() + np.array([0.45, 0.15, 0])
        d = np.array([0.1, 0.075, 0])
        sq_f = m.Square(sp_).set_stroke(INK, 3).set_fill(DATA, 0.45).move_to(p0)
        sq_b = m.Square(sp_).set_stroke(INK, 2).set_fill(opacity=0).move_to(p0 + 3 * d)
        edges = m.VGroup(*[m.Line(sq_f.get_corner(c), sq_b.get_corner(c)).set_stroke(INK, 2) for c in (m.UL, m.UR, m.DR)])
        cuboid = m.VGroup(sq_b, edges, sq_f).set_z_index(3)
        cell = m.Rectangle(width=2 * s0.width / 8, height=2 * s0.height / 4).set_stroke(INK, 3.5).set_fill(DATA, 0.4)
        cell.move_to(s0.get_corner(m.UL) + np.array([3 * s0.width / 8, -1 * s0.height / 4, 0])).set_z_index(3)
        rng = np.random.default_rng(21)
        cells = m.VGroup(*[m.Square(0.17).set_stroke(DATA, 1.2).set_fill(mix(CARD_HI, DATA, float(v)), 1)
                           for v in rng.uniform(0.15, 0.95, 14)]).arrange(m.RIGHT, buff=0.03)
        dots = ellipsis()
        token = m.VGroup(cells, dots).arrange(m.RIGHT, buff=0.1).move_to([4.5, -2.55, 0])
        tok_box = m.SurroundingRectangle(token, buff=0.1, corner_radius=0.08).set_stroke(DATA, 2.5)
        tok_l = text("one token", 22, DATA, weight=m.BOLD).next_to(tok_box, m.UP, buff=0.14)
        rows = m.VGroup(code(f"4 frames × {PATCH_PX} × {PATCH_PX} px × 3 = {PATCH_PIX:,} values", 20, INK),
                        code(f"→ 1 × 2 × 2 latent cells × {WAN_C} = {PATCH_LAT} numbers", 20, INK),
                        cite("Episode 18's patches, with a time axis")).arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT)
        rows.move_to([0, -2.62, 0]).align_to([-6.3, 0, 0], m.LEFT)
        assert rows.get_right()[0] < tok_box.get_left()[0] - 0.2 and rows.get_top()[1] < px_l.get_bottom()[1] - 0.15, (rows.get_right(), tok_box.get_left(), rows.get_top(), px_l.get_bottom())
        assert rows.get_top()[1] < sq.get_bottom()[1] - 0.1 or rows.get_right()[0] < sq.get_left()[0], (rows.get_top(), sq.get_bottom())
        assert tok_box.get_right()[0] < 6.5 and tok_l.get_top()[1] < stat_.get_bottom()[1] - 0.1
        to_tok = arrow([rows[1].get_right()[0] + 0.1, tok_box.get_y(), 0], tok_box.get_left() + [-0.05, 0, 0], DATA)

        self.say(
            "[From twenty twenty-three, video moved into latents]: "
            "[episode seventeen's autoencoder, now squeezing time as well as space].",
            m.AnimationGroup(self.new_header("04 · latents", "Squeeze time, then patch it"),
                             m.LaggedStart(*[m.FadeIn(s_) for s_ in px_deck.slabs[::-1]], lag_ratio=0.06)),
            m.AnimationGroup(m.GrowArrow(go), m.FadeIn(funnel), m.FadeIn(f_cap), m.GrowArrow(go2),
                             m.LaggedStart(*[m.FadeIn(s_) for s_ in lat.slabs[::-1]], lag_ratio=0.15), m.Create(lat_grid)),
        )
        self.say(
            "[One open model turns eighty-one frames into twenty-one], [and each eight by eight square into one cell]: "
            "[forty-six times fewer numbers].",
            m.AnimationGroup(m.FadeIn(px_l, shift=0.1 * m.UP), m.FadeIn(lat_l, shift=0.1 * m.UP), m.FadeIn(sq[0])),
            m.FadeIn(sq[1]),
            m.FadeIn(stat_, scale=0.9),
        )
        self.say(
            "[Then episode eighteen's patches gain a time axis]: "
            "[four frames of sixteen by sixteen pixels], [as one token].",
            m.AnimationGroup(m.FadeIn(cuboid, scale=1.3), m.FadeIn(rows[2])),
            m.AnimationGroup(m.FadeIn(rows[0], shift=0.1 * m.UP), m.FadeIn(cell, scale=1.3), m.FadeIn(rows[1])),
            m.AnimationGroup(m.GrowArrow(to_tok), m.LaggedStart(*[m.FadeIn(c_, shift=0.1 * m.RIGHT) for c_ in cells],
                                                                lag_ratio=0.05), m.FadeIn(dots), m.Create(tok_box),
                             m.FadeIn(tok_l)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 5 · the Sora report (Feb 2024)

    def sora(self):
        self.next_section("5 · The Sora report", notes=(
            "OpenAI, \"Video generation models as world simulators\" (the Sora technical report), 15 Feb 2024, quoted "
            "exactly (notes §21, read via a Wayback capture; openai.com returns 403). The report gives no model or "
            "implementation details. Its simulator sentence says \"a promising path\"; the same report lists its "
            "limitations (glass shattering, eating food). No capability claims beyond the report."))
        w = 12.7
        src = cite("OpenAI, “Video generation models as world simulators”, the Sora report, 15 Feb 2024 · quoted exactly")
        src.move_to([0, 2.3, 0]).align_to([-6.35, 0, 0], m.LEFT)
        q1 = quote_card([f"“{Q_PATCH[0]}", f"{Q_PATCH[1]}”"], w, EDGE, 22, {"spacetime patches": DATA})
        q2 = quote_card([f"“{Q_NONE[0]}”"], w, EDGE, 22)
        q3 = quote_card([f"“{Q_PATH[0]}", f"{Q_PATH[1]}”"], w, EDGE, 22, {"a promising path": LIME})
        q4 = quote_card([f"“{Q_LIMIT[0]}", Q_LIMIT[1], f"{Q_LIMIT[2]}”"], w, RED, 22,
                        {"glass shattering": RED, "eating food": RED})
        stack = m.VGroup(q1, q2, q3, q4).arrange(m.DOWN, buff=0.16)
        stack.next_to(src, m.DOWN, buff=0.2).set_x(0)
        assert stack.get_bottom()[1] > -3.55 and all(q[1].width < w - 0.6 for q in stack), (stack.get_bottom(), [q[1].width for q in stack])
        none_tag = pill("no sizes, no code", MUTED, CARD, 20).move_to(q2[0].get_right() + [-1.45, 0, 0])
        assert none_tag.get_left()[0] > q2[1].get_right()[0] + 0.2, (none_tag.get_left(), q2[1].get_right())

        self.say(
            "[In February twenty twenty-four, OpenAI's Sora report described this]: "
            "[a transformer on spacetime patches of latents].",
            m.AnimationGroup(self.new_header("05 · Feb 2024", "Spacetime patches, in its own words"), m.FadeIn(src)),
            m.FadeIn(q1, shift=0.1 * m.UP),
        )
        self.say(
            "[The report gives no implementation details], [and says so].",
            m.FadeIn(q2, shift=0.1 * m.UP),
            m.FadeIn(none_tag, scale=0.9),
        )
        self.say(
            "[It calls scaling a promising path toward simulators of the physical world],",
            m.FadeIn(q3, shift=0.1 * m.UP),
        )
        self.say(
            "[and in the same report lists failures]: [glass shattering isn't modelled accurately], "
            "[and eating food doesn't always change the food].",
            m.FadeIn(q4, shift=0.1 * m.UP),
            m.Circumscribe(q4[1], color=RED, buff=0.08, run_time=1.2),
            flash(q4[0], RED, 5, run_time=1.2),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 6 · dated: Movie Gen and the open models

    def today(self):
        self.next_section("6 · Movie Gen and open models", notes=(
            "Movie Gen (Meta, arXiv:2410.13720, submitted 17 Oct 2024): \"Our largest video generation model is a 30B "
            "parameter transformer trained with a maximum context length of 73K video tokens, corresponding to a "
            "generated video of 16 seconds at 16 frames-per-second\"; TAE 8× in t, h, w, 16 channels; 2 × 2 patches; "
            f"768 × 768 × 256 frames → {MG_TOKENS:,} tokens; flow matching; 7B upsampler to 1080p; up to 6,144 H100 "
            "GPUs. Open-weight models, dated October 2026 (notes §21 table): CogVideoX Aug 2024 (2B / 5B); HunyuanVideo "
            "3 Dec 2024 (\"over 13 billion\"); Wan 2.1 25 Feb 2025 (1.3B / 14B); LTX-2 announced 23 Oct 2025 (paper 6 Jan "
            "2026; 14B video + 5B audio streams, \"native 4K and up to 50 fps\"). Sound: Veo 3, 20 May 2025. 30 s per clip: "
            "Wan 3.0 (API, GA 24 Aug 2026), as announced."))
        box = card(12.7, 1.95, MODEL, CARD, radius=0.16, stroke=2).move_to([0, 1.35, 0])
        title_ = m.VGroup(title_text("Movie Gen", 32, INK), cite("Meta · arXiv, 17 Oct 2024")).arrange(m.RIGHT, buff=0.3)
        title_.move_to(box.get_top() + [0, -0.38, 0]).align_to(box, m.LEFT).shift(0.35 * m.RIGHT)
        sit(title_[1], baseline(title_[0]))
        facts = [("30B-parameter transformer", 0, 0), ("trained with flow matching", 1, 0),
                 (f"{MG_TOKENS:,} tokens = 16 s at 16 fps", 0, 1), ("768 × 768 px, then 1080p", 1, 1),
                 ("8× autoencoder · 2 × 2 patches", 0, 2), ("up to 6,144 H100 GPUs", 1, 2)]
        cols = [box.get_left()[0] + 0.35, 0.45]
        f_m = [code(s, 20, INK, t2c={r"30B": MODEL, r"flow matching": FLOW, rf"{MG_TOKENS:,} tokens": DATA,
                                    r"6,144": MODEL}).move_to([0, box.get_top()[1] - 0.85 - 0.38 * r, 0]).align_to([cols[c], 0, 0], m.LEFT)
               for s, c, r in facts]
        for f_ in f_m:
            assert f_.get_right()[0] < box.get_right()[0] - 0.3, f_.get_right()
        assert f_m[-1].get_bottom()[1] > box.get_bottom()[1] + 0.12

        cx = [-6.3, -3.0, -0.75, 2.55, 5.55]
        heads = ["open weights", "released", "parameters", "native clip", "latent t × h × w"]
        y_h = -0.08
        hrow = m.VGroup(*[text(h_, 20, MUTED) for h_ in heads])
        hrow[0].move_to([0, y_h, 0]).align_to([cx[0], 0, 0], m.LEFT)
        for i in range(1, 5):
            hrow[i].move_to([cx[i], y_h, 0])
            sit(hrow[i], baseline(hrow[0]))
        hrow[4].align_to([6.35, 0, 0], m.RIGHT)
        rule = m.Line([-6.3, y_h - 0.26, 0], [6.35, y_h - 0.26, 0]).set_stroke(EDGE, 2)
        trs = m.VGroup()
        for r, row in enumerate(OPEN):
            y = y_h - 0.62 - 0.42 * r
            cells_ = [text(row[0], 22, INK, weight=m.BOLD)] + [code(v, 20, INK if i != 2 else MODEL) for i, v in enumerate(row[1:], 1)]
            cells_[0].move_to([0, y, 0]).align_to([cx[0], 0, 0], m.LEFT)
            for i in range(1, 5):
                cells_[i].move_to([cx[i], y, 0])
                sit(cells_[i], baseline(cells_[0]))
            cells_[4].set_x(hrow[4].get_x())
            if row[4] == "not given":
                cells_[4].set_color(MUTED)
            trs.add(m.VGroup(*cells_))
        for tr in trs:
            for a, b in zip(tr[:-1], tr[1:]):
                assert a.get_right()[0] < b.get_left()[0] - 0.15, (a.text, b.text)
        clip_col = m.VGroup(*[tr[3] for tr in trs])
        foot = m.VGroup(code("2025: sound joins the picture (Veo 3, May 2025)", 20, INK, t2c={r"sound": PROMPT}),
                        code("2026: up to 30 s per clip (Wan 3.0, Aug 2026, as announced)", 20, INK, t2c={r"30 s": TIME}))
        foot.arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT).move_to([0, -2.85, 0]).align_to([-6.3, 0, 0], m.LEFT)
        dated = cite("dated October 2026").move_to([0, -3.08, 0]).align_to([6.35, 0, 0], m.RIGHT)
        assert foot.get_top()[1] < trs[-1].get_bottom()[1] - 0.12 and foot.get_bottom()[1] > -3.55, (foot.get_top(), trs[-1].get_bottom())
        assert foot.get_right()[0] < dated.get_left()[0] - 0.2, (foot.get_right(), dated.get_left())

        self.say(
            "[In October twenty twenty-four, Meta's Movie Gen gave real numbers]: "
            "[a thirty-billion-parameter transformer], [trained with episode eleven's flow matching].",
            m.AnimationGroup(self.new_header("06 · dated Oct 2026", "Bigger, longer, with sound"), m.FadeIn(box),
                             m.FadeIn(title_, shift=0.1 * m.UP)),
            m.FadeIn(f_m[0], shift=0.1 * m.UP),
            m.FadeIn(f_m[1], shift=0.1 * m.UP),
        )
        self.say(
            "[Its context holds seventy-three thousand tokens, sixteen seconds of video], "
            "[and it trained on up to six thousand one hundred forty-four GPUs].",
            m.LaggedStart(*[m.FadeIn(f_, shift=0.1 * m.UP) for f_ in (f_m[2], f_m[3], f_m[4])], lag_ratio=0.3),
            m.FadeIn(f_m[5], shift=0.1 * m.UP),
        )
        self.say(
            "[Open-weight models followed], [from one to fourteen billion parameters], "
            "[most making clips of about five seconds].",
            m.AnimationGroup(m.FadeIn(hrow), m.Create(rule)),
            m.LaggedStart(*[m.FadeIn(tr, shift=0.1 * m.UP) for tr in trs], lag_ratio=0.25),
            later(m.Circumscribe(clip_col, color=TIME, buff=0.1)),
        )
        self.say(
            "[By twenty twenty-six, sound has joined the picture], [and new models announce clips of thirty seconds].",
            m.AnimationGroup(m.FadeIn(foot[0], shift=0.1 * m.UP), m.FadeIn(dated)),
            m.FadeIn(foot[1], shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 7 · the cost

    def cost(self):
        self.next_section("7 · Tokens grow with frames", notes=(
            f"Assumed clip (labelled): 512 × 512 frames through Episode 17's 8× autoencoder → a 64 × 64 latent per frame, "
            f"2 × 2 patches → {TOK_FRAME:,} tokens per frame; 5 s at 24 fps = {FRAMES} frames → {TOK_CLIP:,} tokens; "
            f"full attention compares every pair: {TOK_CLIP:,}² = {PAIRS:,} ≈ {PAIRS / 1e9:.1f} billion per attention "
            f"map (per head, per layer) — {FRAMES}× the tokens of one frame, {FRAMES ** 2:,}× the pairs (Episode 8: 49 "
            f"cells → 2,401 scores). Squeezing time 4× → {TOK_SQ:,} tokens, {PAIRS // PAIRS_SQ}× fewer pairs. Named "
            "models with their own factors (notes §21 worked examples, recomputed with T_lat = 1 + (T − 1)/c_t): "
            + "; ".join(f"{n} {v:,}" for n, _, v in MODELS) + "."))
        rows = [code_row("1 frame", "32 × 32", f"{TOK_FRAME:,} tokens", DATA, 2.05),
                code_row(f"{FRAMES} frames", f"{FRAMES} × {TOK_FRAME:,}", f"{TOK_CLIP:,} tokens", DATA, 1.63),
                code_row("pairs", f"{TOK_CLIP:,}²", f"{PAIRS / 1e9:.1f} billion", RED, 1.21, "≈"),
                code_row("time ÷ 4", f"{FRAMES // SQ_T} × {TOK_FRAME:,}", f"{TOK_SQ:,} tokens", DATA, 0.66),
                code_row("pairs", f"{TOK_SQ:,}²", f"{PAIRS_SQ / 1e9:.2f} billion", GREEN, 0.24, "≈")]
        assume = m.VGroup(cite("assumed: 512 × 512 frames → 64 × 64 latent (Episode 17),"),
                          cite("2 × 2 patches, 24 frames a second, full attention")).arrange(m.DOWN, buff=0.06, aligned_edge=m.LEFT)
        assume.move_to([0, -0.55, 0]).align_to([-6.3, 0, 0], m.LEFT)
        for r_ in rows:
            assert r_.get_right()[0] < 0.6, (r_.text, r_.get_right())

        # 120 frames as small squares (one square = one frame's 1,024 tokens), 4 rows of 30
        sq_s, sq_g = 0.15, 0.035
        frames_sq = m.VGroup(*[m.Square(sq_s).set_stroke(width=0).set_fill(DATA, 0.85) for _ in range(FRAMES)])
        frames_sq.arrange_in_grid(rows=4, cols=30, buff=sq_g).move_to([0, -2.05, 0]).align_to([-6.3, 0, 0], m.LEFT)
        fr_l = cite("one square = one frame = 1,024 tokens").next_to(frames_sq, m.DOWN, buff=0.16).align_to(frames_sq, m.LEFT)
        fr_l2 = cite("green: four frames squeezed into one").move_to(fr_l).align_to(fr_l, m.LEFT)
        merged = m.VGroup(*[m.Square(sq_s).set_stroke(width=0).set_fill(GREEN, 0.95).move_to([frames_sq[j].get_x(),
                                                                                             frames_sq.get_y(), 0])
                            for j in range(FRAMES // SQ_T)])
        squeeze_anims = [m.AnimationGroup(*[m.FadeOut(frames_sq[SQ_T * j + i], target_position=merged[j]) for i in range(SQ_T)],
                                          m.FadeIn(merged[j], scale=1.4)) for j in range(FRAMES // SQ_T)]
        assert fr_l.get_bottom()[1] > -3.55 and frames_sq.get_right()[0] < 0.6, (fr_l.get_bottom(), frames_sq.get_right())

        plot = Plot((0, 10), (0, 64), width=4.3, height=2.6, center=(3.75, 0.15, 0), x_ticks=[0, 5, 10],
                    y_ticks=[0, 20, 40, 60], x_label="seconds of video", y_label="attention pairs, billions")
        xs = np.linspace(0, 10, 81)
        raw = plot.line(xs, [pairs_b(x) for x in xs], RED, 4)
        squeezed = plot.line(xs, [pairs_b(x, SQ_T) for x in xs], GREEN, 4)
        d5 = plot.dot(SECS, pairs_b(SECS), RED, 0.09)
        d5_l = code(f"{pairs_b(SECS):.1f} B", 20, RED).next_to(d5, m.RIGHT, buff=0.15)
        e10 = plot.coords(10, pairs_b(10, SQ_T))
        sq_l = code("time ÷ 4", 20, GREEN).next_to(e10, m.UP, buff=0.15).shift(0.35 * m.LEFT)
        assert abs(pairs_b(SECS) * 1e9 - PAIRS) < 1 and plot.get_right()[0] < 6.5 and plot.get_left()[0] > 0.8, (plot.get_left(), plot.get_right())
        assert pairs_b(10) < 64

        # named models (their own compression), horizontal bars
        vmax, blen, bx = 118_800, 1.9, 3.25
        bys = [1.6 - 0.7 * i for i in range(len(MODELS))]
        b_names = m.VGroup(*[m.VGroup(text(n_, 20, INK), cite(d_)).arrange(m.DOWN, buff=0.04, aligned_edge=m.RIGHT)
                             .move_to([0, y, 0]).align_to([bx - 0.15, 0, 0], m.RIGHT) for (n_, d_, _), y in zip(MODELS, bys)])
        bars = m.VGroup(*[hbar(v, vmax, blen, TIME if i else DATA, y, bx) for i, ((_, _, v), y) in enumerate(zip(MODELS, bys))])
        b_vals = m.VGroup(*[code(f"{v:,}", 20, INK).next_to(b, m.RIGHT, buff=0.12) for (_, _, v), b in zip(MODELS, bars)])
        b_cap = cite("tokens per clip, with each model's own squeeze").move_to([3.6, bys[-1] - 0.62, 0])
        b_cap.align_to([6.4, 0, 0], m.RIGHT)
        b_grp = m.VGroup(b_names, bars, b_vals, b_cap)
        assert b_names.get_left()[0] > 0.75 and b_vals.get_right()[0] < 6.5, (b_names.get_left(), b_vals.get_right())
        assert b_cap.get_bottom()[1] > -3.55 and b_cap.get_left()[0] > 0.62, b_cap.get_left()

        self.say(
            "[Why so many GPUs]? [Count tokens]. [One frame of episode seventeen's latent, in patches of two], "
            "[is one thousand and twenty-four].",
            self.new_header("07 · cost", "Tokens grow with frames"),
            m.FadeIn(assume),
            m.FadeIn(frames_sq[0], scale=2),
            m.AnimationGroup(m.FadeIn(rows[0], shift=0.1 * m.UP), m.FadeIn(fr_l)),
        )
        self.say(
            "[Five seconds at twenty-four frames a second is a hundred twenty frames]: "
            "[a hundred twenty-two thousand, eight hundred eighty tokens].",
            m.LaggedStart(*[m.FadeIn(s_, scale=1.5) for s_ in frames_sq[1:]], lag_ratio=0.02, run_time=2.0),
            m.FadeIn(rows[1], shift=0.1 * m.UP),
        )
        self.say(
            "[Attention compares every token with every other], "
            "[so, as in episode eight, the pairs grow with the square].",
            m.FadeIn(plot),
            m.Create(raw, run_time=1.8),
        )
        self.say(
            "[A hundred twenty times the tokens], [fourteen thousand four hundred times the pairs]: "
            "[about fifteen billion, in one attention map].",
            m.Indicate(rows[1], color=DATA, scale_factor=1.04),
            m.AnimationGroup(m.FadeIn(rows[2], shift=0.1 * m.UP), m.FadeIn(d5, scale=1.5)),
            m.FadeIn(d5_l, shift=0.1 * m.RIGHT),
        )
        self.say(
            "[Squeeze time four times], [and the clip drops to thirty thousand seven hundred twenty tokens]: "
            "[sixteen times fewer pairs].",
            m.AnimationGroup(m.LaggedStart(*squeeze_anims, lag_ratio=0.04, run_time=1.6), m.FadeOut(fr_l), m.FadeIn(fr_l2)),
            m.FadeIn(rows[3], shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(rows[4], shift=0.1 * m.UP), m.Create(squeezed), m.FadeIn(sq_l)),
        )
        self.say(
            "[Published models land in the same range]: "
            "[thirty-three thousand to a hundred nineteen thousand tokens per clip], [against four thousand for a large image].",
            m.AnimationGroup(m.FadeOut(m.VGroup(plot, raw, squeezed, d5, d5_l, sq_l)), m.FadeIn(b_cap)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(m.VGroup(n_, b_, v_), shift=0.1 * m.RIGHT)
                                             for n_, b_, v_ in list(zip(b_names, bars, b_vals))[1:]], lag_ratio=0.2)),
            m.FadeIn(m.VGroup(b_names[0], bars[0], b_vals[0]), shift=0.1 * m.RIGHT),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 8 · on this Mac

    def on_this_mac(self):
        self.next_section("8 · On this Mac", notes=(
            f"The digit model's clip {CLIP} (mac_runs §6; {GEN_CITE}): every frame generated on its own as a picture, all "
            f"from one shared starting noise (ρ = 1), so it is the same 7 in every frame; frame k is pasted {SHIFT}·k px "
            f"to the right on a {CH} × {CW} canvas — the motion is pasted on, the model never moves anything, and this "
            "is not a video model (mac_runs: \"Do not claim this toy is a video model\"). The flickering glimpse is the "
            "same clip with a fresh noise per frame (ρ = 0). How to keep frames together: Episode 22. The GPU run's "
            "jointly-denoised toys failed (undertrained) and are not shown."))
        fh = 0.7
        strip_ = m.Group(*[framed(photo(CLEAN_RGB[k], fh), DATA, buff=0.03, stroke=2) for k in range(N_FR)])
        strip_.arrange(m.RIGHT, buff=0.13).move_to([0, 1.62, 0])
        src = cite(GEN_CITE).next_to(strip_, m.UP, buff=0.2)
        t_ax = m.Arrow([strip_.get_left()[0], 1.03, 0], [strip_.get_right()[0], 1.03, 0], buff=0, color=TIME,
                       stroke_width=4, max_tip_length_to_length_ratio=0.03)
        wins = m.VGroup()
        for k in range(N_FR):
            img = strip_[k][1]
            s_ = img.height / CH
            wins.add(m.Rectangle(width=28 * s_, height=img.height).set_stroke(LIME, 2.5)
                     .set_fill(opacity=0).move_to(img.get_left() + [(SHIFT * k + 14) * s_, 0, 0]))
        paste = pill(f"the motion is pasted on: each frame {SHIFT} px further right", LIME, LIME_FILL, 20)
        paste.next_to(t_ax, m.DOWN, buff=0.22)
        ph = 1.55
        reel = Reel(CLEAN_RGB, ph)
        player = framed(reel.image, DATA, buff=0.05, stroke=2.5).move_to([-3.55, -1.3, 0])
        p_l = cite("playing · the same starting noise for every frame").next_to(player, m.DOWN, buff=0.16)
        deck = clip_deck([photo(CLEAN_RGB[k], 0.85) for k in range(N_FR)], (1.95, -1.85), (0.13, 0.1))
        d_l = code(f"{N_FR} × {CH} × {CW} = {CLIP_NUMBERS:,} numbers", 20, INK)
        d_l.next_to(deck, m.DOWN, buff=0.16).set_x(deck.get_x())
        honest = m.VGroup(text("not a video model:", 20, INK, weight=m.BOLD), text("each frame made", 20, MUTED),
                          text("on its own", 20, MUTED)).arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT)
        honest.move_to([0, deck.get_y(), 0]).align_to([deck.get_right()[0] + 0.25, 0, 0], m.LEFT)
        assert honest.get_right()[0] < 6.5, honest.get_right()
        reel0 = Reel(OWN_RGB, ph)
        player0 = framed(reel0.image, RED, buff=0.05, stroke=2.5).move_to([3.2, -1.3, 0])
        p0_l = text("a fresh noise for every frame", 20, RED).next_to(player0, m.DOWN, buff=0.16)
        sit(p0_l, baseline(p_l))
        nxt = pill("Episode 22: keeping frames together", LIME, LIME_FILL, 22).move_to([0, -3.2, 0])
        mascot = pixel(0.4).next_to(nxt, m.LEFT, buff=0.3)
        assert deck.get_top()[1] < paste.get_bottom()[1] - 0.15, (deck.get_top(), paste.get_bottom())
        assert player.get_top()[1] < paste.get_bottom()[1] - 0.1 and p_l.get_bottom()[1] > nxt.get_top()[1] + 0.1, (player.get_top(), p_l.get_bottom())
        assert strip_.get_right()[0] < 6.5 and d_l.get_right()[0] < 6.5 and p0_l.get_right()[0] < 6.5
        assert p_l.get_right()[0] < min(d_l.get_left()[0], p0_l.get_left()[0]) - 0.2, (p_l.get_right(), d_l.get_left())

        self.say(
            "[Here is that shape on this Mac]: [our digit model made eight frames of the digit seven], [with fifty steps for each].",
            self.new_header("08 · on this Mac", "The shape of a video"),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(f, shift=0.15 * m.RIGHT) for f in strip_], lag_ratio=0.1),
                             m.GrowArrow(t_ax)),
            m.FadeIn(src, shift=0.1 * m.UP),
        )
        self.say(
            "[One honest note]: [we pasted each frame three pixels further right]. [The model never moved anything].",
            m.LaggedStart(*[m.Create(w_) for w_ in wins], lag_ratio=0.12),
            m.FadeIn(paste, shift=0.1 * m.UP),
            m.Circumscribe(paste, color=LIME, buff=0.08),
        )
        self.say(
            "[Every frame started from the same noise], [so it is the same seven, sliding along].",
            m.AnimationGroup(m.FadeIn(player, shift=0.1 * m.UP), m.FadeIn(p_l)),
            play_clip(reel, loops=2, fps=4.0),
        )
        self.say(
            "[Stacked, it is our cube again], [but not a video model]: [each frame was made on its own].",
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(deck.tiles[k], target_position=strip_[k]) for k in range(N_FR - 1, -1, -1)],
                                           lag_ratio=0.08), m.FadeIn(d_l)),
            m.FadeIn(honest, shift=0.1 * m.UP),
            play_clip(reel, loops=1, fps=4.0),
        )
        self.say(
            "[Give each frame its own noise instead], [and the seven flickers, a different seven every frame]. "
            "[Keeping frames together is episode twenty-two].",
            m.AnimationGroup(m.FadeOut(m.Group(deck, d_l, honest)), m.FadeIn(player0, shift=0.1 * m.UP), m.FadeIn(p0_l)),
            m.AnimationGroup(play_clip(reel0, loops=2, fps=4.0), play_clip(reel, loops=2, fps=4.0)),
            m.AnimationGroup(m.FadeIn(mascot, shift=0.2 * m.UP), m.FadeIn(nxt, shift=0.2 * m.UP),
                             m.Circumscribe(player0, color=RED, buff=0.08)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes=(
            "Stack (a clip is one block, frames × height × width × channels, noised and denoised together: Episodes 4 "
            "and 7 with one more axis), squeeze (a video autoencoder: time ÷ 4, space ÷ 8 × 8 in most open models), "
            "patch (spacetime patches as tokens for a transformer: Episode 18; Sora report Feb 2024, Movie Gen Oct "
            f"2024). Cost: tokens grow with frames, pairs with their square ({TOK_CLIP:,} tokens → {PAIRS / 1e9:.1f} "
            "billion pairs for an assumed 5 s clip). Next: Episode 22, Keeping Frames Together."))
        halo = glow(LIME, radius=4.0)
        words = m.VGroup(*[title_text(w_, 58, c_) for w_, c_ in (("Stack", TIME), ("Squeeze", DATA), ("Patch", MODEL))])
        subs = m.VGroup(*[code(s_, 20, MUTED) for s_ in ("frames × height × width", "time ÷ 4 · space ÷ 8",
                                                          "spacetime tokens")])
        cols_ = [m.VGroup(w_, s_).arrange(m.DOWN, buff=0.3) for w_, s_ in zip(words, subs)]
        m.VGroup(*cols_).arrange(m.RIGHT, buff=1.0)
        for w_ in words[1:]:
            sit(w_, baseline(words[0]))
        line_y = min(w_.get_bottom()[1] for w_ in words) - 0.45
        for s_ in subs:
            sit(s_, line_y)
        mid = baseline(words[0]) + 0.3
        dots = m.VGroup(*[m.Dot([(l_.get_right()[0] + r_.get_left()[0]) / 2, mid, 0], radius=0.07, color=MUTED)
                          for l_, r_ in zip(cols_, cols_[1:])])
        row = m.VGroup(words, dots, subs).move_to(1.0 * m.UP)
        assert row.width < 12.6
        tag = text("The same loop as before, one more axis, and a cost that grows with the square.", 26, INK)
        tag.next_to(row, m.DOWN, buff=0.65)
        assert tag.width < 12.8
        mascot = pixel(0.42)
        ours = pill(f"5 s at 24 fps: {TOK_CLIP:,} tokens · {PAIRS / 1e9:.1f} billion pairs", RED, RED_FILL, 20)
        nxt = pill("NEXT · EPISODE 22 · KEEPING FRAMES TOGETHER", LIME, LIME_FILL, 20, mono=True)
        foot = m.VGroup(m.VGroup(mascot, ours).arrange(m.RIGHT, buff=0.3), nxt).arrange(m.DOWN, buff=0.3)
        foot.next_to(tag, m.DOWN, buff=0.6)
        assert foot.width < 12.8 and foot.get_bottom()[1] > -3.5, (foot.width, foot.get_bottom())

        self.say(
            "[To sum up]: [stack], [squeeze], [patch].",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.3 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(dots[0]), m.FadeIn(words[1], shift=0.3 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(dots[1]), m.FadeIn(words[2], shift=0.3 * m.UP), m.FadeIn(subs[2])),
        )
        self.say(
            "[Stack the frames and denoise them together], [squeeze time and space with an autoencoder], "
            "[and cut spacetime patches for a transformer].",
            m.AnimationGroup(m.Write(tag), m.Circumscribe(words[0], color=TIME, buff=0.1)),
            m.Circumscribe(words[1], color=DATA, buff=0.1),
            m.Circumscribe(words[2], color=MODEL, buff=0.1),
        )
        self.say(
            "[The loop never changes]. [Only the block grows], [and the cost grows with its square].",
            m.FadeIn(mascot, shift=0.2 * m.UP),
            m.FadeIn(ours, shift=0.2 * m.UP),
            m.FadeIn(nxt, shift=0.2 * m.UP),
        )
        self.wait(0.4)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, tag, foot)], run_time=0.9)
