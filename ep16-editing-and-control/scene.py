"""Out of the Noise · Episode 16 · Editing and Control (Module 4 · Steering the Picture — the module's last episode)

The same machinery edits. Running example: the lighthouse (research/experiments/mac_runs.md §5, "Image-to-image /
SDEdit"): the seed-0 picture (SD 2.1-base, 50 Euler steps, guidance 7.5) re-noised to t = 1000·s and denoised with
int(50·s) Euler steps at guidance 7.5 with a new prompt. Honest result: with the prompt "… at dusk in winter, snow,
oil painting" no snow appears at any strength 0.3–0.9 (mean pixel change 6 / 9 / 10 / 9 %); a more explicit prompt
gives the classic progression (6 / 9 / 10 / 13 / 15 % at 0.3 / 0.5 / 0.7 / 0.9 / 1.0; snowy at 0.9, nearly a new
picture at 1.0, the lighthouse still on the right because SD's scaled-linear schedule keeps √ᾱ = 0.068 of the signal at
its last step — computed below, Lin et al. 2023 Table 1: 0.068265).

The mechanism on the swirl, computed here with diffusion_reference (no digit edit was run): SDEdit from a rough sketch
of one arm (300 points; q_sample to t, then DDIM 50 steps from t to 0 with the exact denoiser) — the faithfulness vs
realism trade-off measured at 27 noise levels (mean distance moved from the sketch vs mean distance to the swirl's
centre line); inpainting by replacement (keep x = 0.45, regenerate y; after every step x is overwritten with its
correctly noised value) — plain (Song et al. 2021's imputation trick) and with RePaint's resampling (Lugmayr et al.
2022: step down, re-noise one step, repeat; here 5 times per step).

ControlNet (Zhang, Rao & Agrawala 2023), LoRA (Hu et al. 2021: arithmetic on SD 2.1-base's own attention shapes, read
from its safetensors header by precompute.py) and DreamBooth (Ruiz et al. 2023) as cards, facts from research/notes.md
§16 and Appendix C row 16.

Every number on screen is computed below (diffusion_reference, the run's files, measured(...)) and asserted, or cited
with its source. Illustrative (said "for example" or drawn as a schematic): the inpainting mask on the lighthouse (no
inpainting run exists), the ControlNet and LoRA diagrams, DreamBooth's four pictures of Pixel.

    ~/.local/share/uv/tools/manimgx/bin/python precompute.py   # once: writes sd21_attention_shapes.json (header only)
    VOICE=draft manimgx inspect scene.py                       # silent draft
    manimgx render scene.py                                    # with the narrator serving from the series folder
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403
from diffusion_reference import *  # noqa: E402,F403

EP_DIR = Path(__file__).resolve().parent  # set AFTER the import: diffusion_reference exports HERE (the series folder)
warnings.filterwarnings("ignore", message=".*encountered in matmul")  # NumPy/Accelerate's spurious warnings

PREFIX = "ep16"
IMG = EXPERIMENTS / "images"
LH = IMG / "lighthouse"
F30 = IMG / "lighthouse_frames30"
assert T == 1000


def load_rgb(path, size=None):
    im = Image.open(path).convert("RGB")
    if size:
        im = im.resize((size, size), Image.LANCZOS)
    return np.asarray(im).copy()


# ---------------------------------------------------------------------------- the lighthouse runs (mac_runs §5)

_LHM = measured("mac_runs", "lighthouse")
WINTER, SNOW = _LHM["sdedit"], _LHM["sdedit_snow"]
assert WINTER["source"] == SNOW["source"] == "lighthouse_sd21_cfg7.5_steps50_seed0.png"
assert WINTER["cfg"] == SNOW["cfg"] == 7.5 and WINTER["seed"] == SNOW["seed"] == 0 and WINTER["base_steps"] == 50
WINTER_PROMPT, SNOW_PROMPT = WINTER["prompt"], SNOW["prompt"]
assert WINTER_PROMPT == "a lighthouse on a rocky coast at dusk in winter, snow, oil painting"
assert SNOW_PROMPT == ("a lighthouse on a snowy rocky coast in winter, snow-covered rocks, falling snow, at dusk, "
                       "oil painting")
W_S = [r["strength"] for r in WINTER["rows"]]
S_S = [r["strength"] for r in SNOW["rows"]]
assert W_S == [0.3, 0.5, 0.7, 0.9] and S_S == [0.3, 0.5, 0.7, 0.9, 1.0]
for _r in WINTER["rows"] + SNOW["rows"]:
    assert _r["start_t"] == 1000 * _r["strength"] and _r["steps_run"] == int(50 * _r["strength"])
W_CHANGE = [round(100 * r["mean_abs_pixel_change"]) for r in WINTER["rows"]]
S_CHANGE = [round(100 * r["mean_abs_pixel_change"]) for r in SNOW["rows"]]
assert W_CHANGE == [6, 9, 10, 9] and S_CHANGE == [6, 9, 10, 13, 15]       # "six to ten percent"
SD_PARAMS = _LHM["models"]["sd2-community/stable-diffusion-2-1-base"]["parts"]["unet"]["params"]

SOURCE = load_rgb(LH / "lighthouse_sd21_cfg7.5_steps50_seed0.png")
W_IMG = {s: load_rgb(LH / r["file"]) for s, r in zip(W_S, WINTER["rows"])}
S_IMG = {s: load_rgb(LH / r["file"]) for s, r in zip(S_S, SNOW["rows"])}
NOISED = {s: load_rgb(LH / f"lighthouse_sdedit_winter_strength{s}_noised_start.png") for s in W_S}
assert SOURCE.shape == (512, 512, 3)
# the 30-step run's decoded x_t frames (Episode 1), ending on the 50-step picture we edit (the two look alike: same seed)
HOOK_FRAMES = [load_rgb(F30 / f"xt_{i:02d}.png", 384) for i in range(31)]
assert float(np.abs(HOOK_FRAMES[-1].astype(float) - load_rgb(LH / "lighthouse_sd21_cfg7.5_steps50_seed0.png", 384)).mean()) < 12
HOOK_FRAMES.append(load_rgb(LH / "lighthouse_sd21_cfg7.5_steps50_seed0.png", 384))

# Stable Diffusion's "scaled linear" schedule (√β linear from √0.00085 to √0.012, 1,000 steps), in this course's
# indexing (alpha_bar[0] = 1, alpha_bar[1000] = the product of all 1,000 factors — the sampler's t = 1000)
SD_INFO = json.loads((EP_DIR / "sd21_attention_shapes.json").read_text())
_sch = SD_INFO["scheduler"]
assert (_sch["beta_schedule"], _sch["beta_start"], _sch["beta_end"], _sch["num_train_timesteps"]) == \
    ("scaled_linear", 0.00085, 0.012, 1000)
SD_AB = alphas_cumprod(np.concatenate([[0.0], np.linspace(np.sqrt(0.00085), np.sqrt(0.012), 1000) ** 2]))
SD_ROOT = {s: float(np.sqrt(SD_AB[int(round(1000 * s))])) for s in (0.3, 0.5, 0.7, 0.9, 1.0)}
assert abs(SD_ROOT[1.0] - 0.068265) < 5e-6 and f"{SD_ROOT[1.0]:.3f}" == "0.068"      # Lin et al. 2023, Table 1
assert abs(SD_AB[1000] / (1 - SD_AB[1000]) - 0.004682) < 1e-6                          # their terminal SNR
assert [f"{SD_ROOT[s]:.2f}" for s in (0.3, 0.5, 0.7, 0.9)] == ["0.77", "0.53", "0.29", "0.12"]
assert SD_ROOT[0.3] > 0.75 and 1 / 9 < SD_ROOT[0.9] < 1 / 8 + 0.005                    # "three quarters", "an eighth"

# ---------------------------------------------------------------------------- the swirl: SDEdit from a sketch

N_SK = 300
_arm = swirl_curve(N_SK)[0]
_u = np.linspace(0, 1, N_SK)
_out = _arm / np.linalg.norm(_arm, axis=1, keepdims=True)
_hand = 0.012 * np.random.default_rng(3).standard_normal(N_SK).cumsum() / np.sqrt(N_SK) * 3   # a wobbly hand
SKETCH = _arm + (0.13 * np.sin(2 * np.pi * 1.7 * _u + 0.9) + _hand)[:, None] * _out  # a rough sketch of arm 0
EPS_SK = gaussian_noise(N_SK, seed=161)
_d0, _a0, _ = nearest_on_swirl(SKETCH)
assert np.all(_a0 == 0) and f"{_d0.mean():.3f}" == "0.087"                 # every point is nearest to arm 0


def sdedit_swirl(t0, steps=50):
    """SDEdit on the swirl: jump the sketch to step t0 (q_sample), then DDIM (no fresh noise) back to 0."""
    xt = q_sample(SKETCH, t0, EPS_SK)
    return ddim_sample(exact_eps, xt, np.linspace(t0, 0, steps + 1))


def trade(x):
    moved = float(np.linalg.norm(x - SKETCH, axis=1).mean())
    d, arm, _ = nearest_on_swirl(x)
    return moved, float(d.mean()), float(np.mean(arm == 0))


T_GRID = [0, 1, 2, 3, 5, 7, 10, 14, 20, 25, 30, 40, 50, 60, 75, 100, 130, 160, 200, 250, 300, 400, 500, 600, 700,
          850, 1000]
FRONT = {0: trade(SKETCH)}
for _t in T_GRID[1:]:
    FRONT[_t] = trade(sdedit_swirl(_t).xs[-1])
T_RUNS = (5, 20, 300)
RUNS = {t: sdedit_swirl(t) for t in T_RUNS}
FRONT_ON = {t: float(np.mean(nearest_on_swirl(RUNS[t].xs[-1])[0] < 0.06)) for t in T_RUNS}
SIG = {t: float(sigma_from_alpha_bar(ALPHA_BAR[t])) for t in T_RUNS}
DATA_DIST = measured("toy", "headline", "data_dist")
assert f"{DATA_DIST:.3f}" == "0.024"
assert [f"{FRONT[t][1]:.3f}" for t in T_RUNS] == ["0.064", "0.033", "0.018"]    # distance to the swirl
assert [f"{FRONT[t][0]:.2f}" for t in T_RUNS] == ["0.04", "0.11", "0.71"]       # distance moved from the sketch
assert FRONT[5][2] > 0.98 and FRONT[20][2] > 0.85 and 0.4 < 1 - FRONT[300][2] < 0.5   # "almost half on the other arm"
assert 0.85 * _d0.mean() < SIG[20] < _d0.mean()                                  # σ ≈ the sketch's mistakes
assert FRONT[1000][0] > 0.85 and all(FRONT[a][1] >= FRONT[b][1] - 0.002 for a, b in zip(T_GRID[:12], T_GRID[1:12]))

# ---------------------------------------------------------------------------- the swirl: inpainting by replacement

XK = 0.45                                    # the kept coordinate; y is masked and regenerated
N_IN = 400


def inpaint(resample=1, steps=100, seed=0):
    """Ancestral DDIM (η = 1) on the swirl's exact denoiser; after every step the known x is overwritten with
    √ᾱ_s·XK + √(1 − ᾱ_s)·z (fresh z). resample > 1: RePaint's back-and-forth — re-noise s → t and redo the step."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((N_IN, 2))
    ts = ddim_timesteps(steps)
    path = [x.copy()]
    for t, s in zip(ts[:-1], ts[1:]):
        ab_t, ab_s = ALPHA_BAR[t], ALPHA_BAR[s]
        for u in range(resample):
            eps = exact_eps(x, t)
            x0 = x0_from_eps(x, eps, ab_t)
            sig = np.sqrt((1 - ab_s) / (1 - ab_t) * (1 - ab_t / ab_s))
            x_s = np.sqrt(ab_s) * x0 + np.sqrt(1 - ab_s - sig ** 2) * eps + sig * rng.standard_normal(x.shape)
            x_s[:, 0] = np.sqrt(ab_s) * XK + np.sqrt(1 - ab_s) * rng.standard_normal(N_IN)   # overwrite the kept part
            if u < resample - 1 and s > 0:
                x = np.sqrt(ab_t / ab_s) * x_s + np.sqrt(1 - ab_t / ab_s) * rng.standard_normal(x.shape)  # back up
            else:
                x = x_s
        path.append(x.copy())
    return np.array(path)


IN_PLAIN, IN_REPAINT = inpaint(1), inpaint(5)
ON = {k: float(np.mean(nearest_on_swirl(p[-1])[0] < 0.06)) for k, p in (("plain", IN_PLAIN), ("repaint", IN_REPAINT))}
assert np.allclose(IN_PLAIN[-1][:, 0], XK) and np.allclose(IN_REPAINT[-1][:, 0], XK)
assert round(100 * ON["plain"]) == 80 and round(100 * ON["repaint"]) == 100     # "eighty percent", "every point"
# for the notes: the exact conditional p(y | x = 0.45) of the swirl mixture puts 97.5 % within 0.06 of the curve
_w = np.exp(-(XK - SWIRL.means[:, 0]) ** 2 / (2 * SWIRL.stds ** 2))
_k = np.random.default_rng(1).choice(len(_w), size=4000, p=_w / _w.sum())
_ex = np.stack([np.full(4000, XK), SWIRL.means[_k, 1] + SWIRL.stds[_k] * np.random.default_rng(2).standard_normal(4000)], 1)
EXACT_ON = float(np.mean(nearest_on_swirl(_ex)[0] < 0.06))
_dense = swirl_curve(4000).reshape(-1, 2)
CROSS_Y = np.sort(_dense[np.abs(_dense[:, 0] - XK) < 0.003][:, 1])

# ---------------------------------------------------------------------------- LoRA arithmetic on SD 2.1-base's shapes

PROJ = SD_INFO["attention_projections"]
assert len(PROJ) == 128 and SD_INFO["unet_params"] == SD_PARAMS == 865910724
ONE = "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight"
D_ONE, K_ONE = PROJ[ONE]
RANK = 4
FULL_ONE, LORA_ONE = D_ONE * K_ONE, RANK * (D_ONE + K_ONE)
assert (D_ONE, K_ONE, FULL_ONE, LORA_ONE) == (320, 320, 102400, 2560) and f"{100 * LORA_ONE / FULL_ONE:.1f}" == "2.5"
LORA_ALL = sum(RANK * (d + k) for d, k in PROJ.values())
LORA_MB = 2 * LORA_ALL / 1e6                                                     # fp16: 2 bytes per number
UNET_GB = SD_INFO["unet_file_bytes"] / 1e9
assert LORA_ALL == 829952 and f"{LORA_MB:.1f}" == "1.7" and LORA_MB < 2 and f"{UNET_GB:.2f}" == "1.73"


# ---------------------------------------------------------------------------- an edge map of the lighthouse (ControlNet)

def edge_map(rgb, size=256, keep=0.11):
    g = np.asarray(Image.fromarray(rgb).convert("L").resize((size, size), Image.LANCZOS), float) / 255
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[1:-1, 1:-1] = (g[:-2, 2:] + 2 * g[1:-1, 2:] + g[2:, 2:]) - (g[:-2, :-2] + 2 * g[1:-1, :-2] + g[2:, :-2])
    gy[1:-1, 1:-1] = (g[2:, :-2] + 2 * g[2:, 1:-1] + g[2:, 2:]) - (g[:-2, :-2] + 2 * g[:-2, 1:-1] + g[:-2, 2:])
    mag = np.hypot(gx, gy)
    return (mag > np.quantile(mag, 1 - keep)).astype(float)


EDGES = edge_map(SOURCE)

# ---------------------------------------------------------------------------- look

FLAT = set("abcdefhiklmnorstuvwxzABCDEFGHIJKLMNOPRSTUVWXYZ0123456789")
SKETCH_C = PROMPT          # the user's guide (sketch, kept pixels, control image): conditioning, this module's colour
FROZEN_C, TRAIN_C = MODEL, PROMPT


def cite(s, size=20):
    return text(s, size, MUTED)


def baseline(words):
    """The baseline of a Text: the median bottom of its flat-bottomed letters."""
    chars = [c for c in words.text if not c.isspace()]
    if len(words) == len(chars):
        bottoms = sorted(g.get_bottom()[1] for g, c in zip(words, chars) if c in FLAT)
        if bottoms:
            return bottoms[len(bottoms) // 2]
    return float(np.median([g.get_bottom()[1] for g in words]))


def sit(words, y):
    return words.shift((y - baseline(words)) * m.UP)


def rgba(rgb):
    """The texture picture() makes from a large image (no pixel repeat), for swapping in place."""
    arr = np.concatenate([rgb, np.full(rgb.shape[:2] + (1,), 255, np.uint8)], axis=-1)
    arr.flags.writeable = False
    return arr


class Flip:
    """A picture whose pixels are swapped for frame k of a list (only when k changes)."""

    def __init__(self, frames, k, height):
        self.frames, self.k = frames, int(k)
        self.image = picture(frames[self.k], height)

    def show(self, k):
        k = int(k)
        if k != self.k:
            self.k = k
            self.image.paint = self.image.paint.but(texture=rgba(self.frames[k]))


def drive(mob, hooks, run_time=1.4, rate=m.linear):
    def step(_, a):
        for hook in hooks:
            hook(a)

    return m.UpdateFromAlphaFunc(mob, step, run_time=run_time, rate_func=rate)


def photo(rgb, h, color=EDGE, stroke=2.5):
    return framed(picture(rgb, h), color, buff=0.05, stroke=stroke)


def plane(center, size, lim):
    return Plot((-lim, lim), (-lim, lim), width=size, height=size, center=tuple(center), axes=False)


def at(plot, xy):
    return plot.coords(float(xy[0]), float(xy[1]))


def swirl_band(plot, width=9, opacity=0.2):
    return m.VGroup(*[m.VMobject().set_points_smoothly([at(plot, p) for p in arm[::3]] + [at(plot, arm[-1])])
                      .set_stroke(DATA, width, opacity=opacity) for arm in swirl_curve(200)]).set_z_index(-1)


def arrow_to(a, b, color=MUTED, width=4, buff=0.12):
    return m.Arrow(a, b, buff=buff, color=color, stroke_width=width, max_tip_length_to_length_ratio=0.25)


class Dots(m.VGroup):
    """Swirl points as dots inside a square plane of half-width lim: dots outside the view are hidden.
    `.walk(path, colours)` replays a path (states × n × 2) while blending colour from start to end."""

    def __init__(self, plot, xy, lim, color, radius=0.03):
        super().__init__()
        self.plot, self.lim, self.xy = plot, lim, np.asarray(xy, dtype=float)
        for p in self.xy:
            self.add(m.Dot(at(plot, p), radius=radius, color=color).set_z_index(4))
        self._show(self.xy, [color] * len(self.xy))

    def _show(self, xy, colours):
        inside = np.all(np.abs(xy) < self.lim * 0.98, axis=1)
        for d, p, c, ok in zip(self, xy, colours, inside):
            d.move_to(at(self.plot, np.clip(p, -self.lim, self.lim)))
            d.set_color(c)
            d.set_opacity(0.9 if ok else 0.0)

    def walk(self, path, start_c, end_c, seconds=2.0, final=None, rate=m.smooth):
        """final: per-dot colours for the last state (e.g. GOOD/BAD); else end_c."""
        path = np.asarray(path, dtype=float)
        n_dots = len(self)
        last = final or [end_c] * n_dots
        shades = [mix(start_c, end_c, s / 12) for s in range(13)]

        def step(_, a):
            i = a * (len(path) - 1)
            j = int(np.floor(i))
            k = min(j + 1, len(path) - 1)
            now = path[j] + (path[k] - path[j]) * (i - j)
            if a >= 0.999:
                self._show(path[-1], last)
            else:
                self._show(now, [shades[int(round(12 * a))]] * n_dots)
            self.xy = now

        return m.UpdateFromAlphaFunc(self, step, run_time=seconds, rate_func=rate)


def zero_conv(color=TRAIN_C):
    box = m.Square(0.46).set_stroke(color, 2.5).set_fill(BG, 1)
    return m.VGroup(box, code("0", 22, color, weight=m.BOLD).move_to(box))


def block(label, w, h, color, fill, size=22):
    box = card(w, h, color, fill, radius=0.14, stroke=2.5)
    return m.VGroup(box, text(label, size, INK, weight=m.BOLD).move_to(box))


def tower_ring(ph, h, color=GOOD):
    """A ring around the lighthouse tower in a 512-px picture of height h (tower at 66% across, 34% down)."""
    c = ph.get_center() + np.array([(0.665 - 0.5) * h, (0.5 - 0.34) * h, 0])
    return m.Ellipse(width=0.3 * h, height=0.62 * h).set_stroke(color, 3).move_to(c)


def masked_source():
    """The source with the lighthouse tower replaced by VIOLET noise: an illustration of an inpainting mask."""
    img = SOURCE.copy()
    rng = np.random.default_rng(16)
    noise = tint(rng.standard_normal((32, 32)), NOISE, -2.5, 2.5)
    noise = np.repeat(np.repeat(noise, 16, axis=0), 16, axis=1)
    y0, y1, x0, x1 = 20, 300, 290, 400
    img[y0:y1, x0:x1] = noise[y0:y1, x0:x1]
    return img, (x0 / 512, x1 / 512, y0 / 512, y1 / 512)


# ---------------------------------------------------------------------------- the episode


class Ep16EditingAndControl(Episode):
    number = 16

    def lesson(self):
        self.hook()
        self.sdedit()
        self.strength()
        self.swirl_sketch()
        self.inpainting()
        self.control()
        self.finetunes()
        self.recap()

    # ------------------------------------------------------------------ 0 · hook

    def hook(self):
        self.next_section("Hook", notes=(
            "Left: the 30-step run's decoded x_t frames (research/experiments/images/lighthouse_frames30, Episode 1), "
            "landing on the 50-step picture we edit (lighthouse_sd21_cfg7.5_steps50_seed0.png; same seed and prompt, the "
            "two look alike). Right: the same picture after SDEdit at strength 0.9 with the explicit snow prompt "
            f"('{SNOW_PROMPT}') — mac_runs.md §5. SD 2.1-base (sd2-community mirror), Euler sampler of mlx-examples, "
            "guidance 7.5, seed 0. This episode closes Module 4 (Steering the Picture)."))
        h = 3.1
        flip = Flip(HOOK_FRAMES, 0, h)
        left = m.Group(m.Rectangle(width=h + 0.1, height=h + 0.1).set_stroke(EDGE, 2.5), flip.image).move_to([-3.0, 0.55, 0])
        flip.image.move_to(left[0])
        cap_l = cite("generated on this Mac · SD 2.1-base · seed 0").next_to(left, m.DOWN, buff=0.18)
        right = photo(S_IMG[0.9], h).move_to([3.0, 0.55, 0])
        cap_r = cite("edited on this Mac · strength 0.9").next_to(right, m.DOWN, buff=0.18)
        sit(cap_r, baseline(cap_l))
        go = arrow_to(left.get_right(), right.get_left(), PROMPT, 5, 0.18)
        how = text("noise part way,\nthen denoise", 20, MUTED).next_to(go, m.UP, buff=0.2)
        snow = pill("+ snow", PROMPT, PINK_FILL, 20).next_to(go, m.DOWN, buff=0.22)
        words = ("repaint", "fill a hole", "follow a sketch", "learn a subject")
        chips_ = m.VGroup(*[chip(w, PROMPT, PINK_FILL, width=text(w, 24, weight=m.BOLD).width + 0.55, height=0.62, size=24)
                            for w in words]).arrange(m.RIGHT, buff=0.3)
        for c_ in chips_[1:]:
            sit(c_[1], baseline(chips_[0][1]))
        mascot = pixel(0.5)
        row = m.VGroup(chips_, mascot).arrange(m.RIGHT, buff=0.45).move_to([0, -2.75, 0])
        assert row.width < 12.6 and cap_l.get_bottom()[1] > row.get_top()[1] + 0.3, (row.width, cap_l.get_bottom())

        def run(a):
            flip.show(min(len(HOOK_FRAMES) - 1, int(np.floor(a * len(HOOK_FRAMES)))))

        self.say(
            "[So far, every picture began as pure noise], [and the denoiser carved it out, step by step].",
            m.AnimationGroup(self.new_header("00 · hook", "Start from a picture"), m.FadeIn(left), m.FadeIn(cap_l)),
            drive(left, [run], run_time=3.0, rate=lambda a: float(a) ** 0.8),
        )
        self.say(
            "[Today we start from a picture we already have], [and change it with the very same model].",
            m.AnimationGroup(m.GrowArrow(go), m.FadeIn(how), m.FadeIn(snow, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(right, target_position=left), m.FadeIn(cap_r)),
        )
        self.say(
            "[You can repaint it], [fill a hole], [follow a sketch], [or learn a new subject]: [this episode closes module four].",
            m.FadeIn(chips_[0], shift=0.15 * m.UP),
            m.FadeIn(chips_[1], shift=0.15 * m.UP),
            m.FadeIn(chips_[2], shift=0.15 * m.UP),
            m.FadeIn(chips_[3], shift=0.15 * m.UP),
            m.FadeIn(mascot, shift=0.15 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 1 · SDEdit: noise part way

    def sdedit(self):
        self.next_section("1 · Noise part way, then denoise", notes=(
            "SDEdit (Meng, He, Song, Song, Wu, Zhu & Ermon, arXiv:2108.01073, Aug 2021; ICLR 2022): x_t0 = √ᾱ·x_guide + "
            "√(1 − ᾱ)·ε, then the normal reverse process from t0. Our run (mac_runs.md §5): strength s → encode the source, "
            "add noise to t = 1000·s, run int(50·s) Euler steps at guidance 7.5 (Episode 14's scale) with the new prompt. "
            "The four noised starts are the run's own files (lighthouse_sdedit_winter_strength*_noised_start.png: the "
            "latent at step t, decoded by the VAE — latent noise decodes as coloured blotches). √ᾱ is computed from SD's "
            "scaled-linear schedule (scheduler_config: β 0.00085 → 0.012, √β linear, 1,000 steps), in the sampler's "
            "indexing where t = 1000 is the last step: "
            + ", ".join(f"s = {s}: √ᾱ = {SD_ROOT[s]:.3f}" for s in (0.3, 0.5, 0.7, 0.9, 1.0))
            + ". Notes §16's worked example gives the same ᾱ (0.592 / 0.278 / 0.083 / 0.014 at 300 / 500 / 700 / 900)."))
        y_ax, xl, xr = -0.95, -5.3, 5.3
        xt = lambda t: xl + (xr - xl) * t / 1000  # noqa: E731
        axis = m.Arrow([xl - 0.15, y_ax, 0], [xr + 0.45, y_ax, 0], buff=0, color=TIME, stroke_width=4,
                       max_tip_length_to_length_ratio=0.02)
        t_lab = code("t", 24, TIME, weight=m.BOLD).next_to(axis.get_end(), m.UP, buff=0.12)
        ticks = m.VGroup()
        for t in (0, 300, 500, 700, 900, 1000):
            ticks.add(m.Line([xt(t), y_ax - 0.12, 0], [xt(t), y_ax + 0.12, 0]).set_stroke(TIME, 3),
                      code(f"{t}", 20, TIME).move_to([xt(t), y_ax - 0.38, 0]))
        for lab in ticks[3::2]:
            sit(lab, baseline(ticks[1]))
        h, y_pic = 1.7, 0.45
        src = photo(SOURCE, h, DATA).move_to([xt(0), y_pic, 0])
        src_l = text("our picture", 20, DATA, weight=m.SEMIBOLD).next_to(src, m.UP, buff=0.14)
        starts, roots, stems = {}, {}, m.VGroup()
        for s in W_S:
            starts[s] = photo(NOISED[s], h, NOISE).move_to([xt(1000 * s), y_pic, 0])
            roots[s] = code(f"√ᾱ = {SD_ROOT[s]:.2f}", 22, TIME).next_to(starts[s], m.UP, buff=0.14)
        for s in W_S[1:]:
            sit(roots[s], baseline(roots[W_S[0]]))
        sit(src_l, baseline(roots[W_S[0]]))
        for x in [0] + [1000 * s for s in W_S]:
            stems.add(m.Line([xt(x), y_ax + 0.14, 0], [xt(x), y_pic - h / 2 - 0.08, 0]).set_stroke(FAINT, 2))
        knob = m.Dot([xt(0), y_ax, 0], radius=0.12, color=INK).set_z_index(5)
        res = photo(W_IMG[0.5], 1.45, DATA).move_to([-4.55, -2.55, 0])
        back = m.CurvedArrow([xt(500) - 0.15, y_ax - 0.62, 0], res.get_right() + [0.12, 0, 0], angle=-0.55)
        back.set_stroke(MODEL, 4)
        back.get_tip().set_fill(MODEL).set_stroke(MODEL)
        back_l = m.VGroup(text("25 Euler steps, new prompt", 20, MODEL, weight=m.SEMIBOLD),
                          cite("guidance 7.5 · Episode 14")).arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT)
        back_l.move_to([-1.55, -3.0, 0])
        f = tex(r"x_t = \sqrt{\bar\alpha_t}\,x + \sqrt{1-\bar\alpha_t}\,\varepsilon", 34)
        rule = code("strength s:  noise to t = 1000·s\nthen int(50·s) of the 50 steps", 20, INK,
                    t2c={r"strength s:": TIME})
        body = m.VGroup(f, rule).arrange(m.DOWN, buff=0.25, aligned_edge=m.LEFT)
        box = card(body.width + 0.6, body.height + 0.5, EDGE, CARD, radius=0.16, stroke=2).move_to(body)
        recipe = m.VGroup(box, body).move_to([3.65, -2.55, 0])
        src_c = cite("Meng et al., 2021 · SDEdit").move_to([0, 2.3, 0]).align_to([6.5, 0, 0], m.RIGHT)
        cap = cite("generated on this Mac · SD 2.1-base · the noised starting points, decoded").move_to([0, 2.3, 0])
        cap.align_to([-6.5, 0, 0], m.LEFT)
        assert recipe.get_right()[0] < 6.6 and back_l.get_right()[0] < recipe.get_left()[0] - 0.1, (recipe.get_right(), back_l.get_right())
        assert cap.get_right()[0] < src_c.get_left()[0] - 0.3 and roots[0.9].get_top()[1] < 2.05

        self.say(
            "[The trick is called SDEdit, from Meng and colleagues in twenty twenty-one]: "
            "[push a picture part way up the noise schedule], [then denoise it with a new prompt].",
            m.AnimationGroup(self.new_header("01 · sdedit", "Noise part way, then denoise"), m.FadeIn(axis), m.FadeIn(t_lab),
                             m.FadeIn(ticks), m.FadeIn(src), m.FadeIn(src_l), m.FadeIn(stems[0]), m.FadeIn(knob),
                             m.FadeIn(src_c)),
            m.AnimationGroup(knob.animate(run_time=1.4).move_to([xt(500), y_ax, 0]), m.FadeIn(stems[2]),
                             m.FadeIn(starts[0.5], target_position=src, run_time=1.4), m.FadeIn(roots[0.5])),
            m.AnimationGroup(m.Create(back), m.FadeIn(res, target_position=starts[0.5]), m.FadeIn(back_l)),
        )
        self.say(
            "[The strength says how far up]: [zero point five means noise to step five hundred], "
            "[then twenty-five of the fifty steps, with episode fourteen's guidance at seven point five].",
            m.FadeIn(recipe, shift=0.1 * m.UP),
            m.Circumscribe(ticks[5], color=TIME, buff=0.1),
            m.Circumscribe(back_l, color=MODEL, buff=0.08),
        )
        self.say(
            "[These are the real starting points from this Mac], [at strengths from zero point three to zero point nine].",
            m.FadeIn(cap),
            m.LaggedStart(*[m.AnimationGroup(m.FadeIn(stems[i + 1]), m.FadeIn(starts[s], target_position=src),
                                             m.FadeIn(roots[s])) for i, s in enumerate(W_S) if s != 0.5],
                          lag_ratio=0.4, run_time=2.2),
        )
        self.say(
            "[At zero point three, more than three quarters of the picture's signal is left]; [at zero point nine, only about an eighth].",
            m.AnimationGroup(roots[0.3].animate.set_color(GOOD), m.Circumscribe(roots[0.3], color=GOOD, buff=0.08)),
            m.AnimationGroup(roots[0.9].animate.set_color(RED), m.Circumscribe(roots[0.9], color=RED, buff=0.08)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 2 · strength: the real runs

    def strength(self):
        self.next_section("2 · Keep the layout, or repaint it", notes=(
            "mac_runs.md §5 and mac_runs.json lighthouse.sdedit / sdedit_snow (generated on this Mac, SD 2.1-base, "
            "Euler, guidance 7.5, seed 0). Prompt as specified: "
            f"'{WINTER_PROMPT}' — strengths 0.3 / 0.5 / 0.7 / 0.9 (15 / 25 / 35 / 45 steps): the layout is kept and no snow "
            f"appears; mean absolute pixel change {W_CHANGE} %. The pipeline matches the library exactly (mac_runs §5: "
            "identical img2img output), so this is genuine behaviour, not a bug; our reading (not tested further): the "
            "rest of the prompt already describes the picture. Explicit prompt: "
            f"'{SNOW_PROMPT}' — strengths 0.3–1.0, mean pixel change {S_CHANGE} %; snowy at 0.9, nearly new at 1.0 with "
            f"the lighthouse still on the right: SD's scaled-linear schedule keeps √ᾱ = {SD_ROOT[1.0]:.4f} at its last "
            "step (computed here; Lin, Liu, Li & Yang, 'Common Diffusion Noise Schedules and Sample Steps are Flawed', "
            "arXiv:2305.08891, 2023, Table 1: √ᾱ_T = 0.068265, SNR(T) = 0.004682). The prompt is read by the text "
            "encoder and cross-attention of Episode 15."))

        def prompt_card(s, rules):
            words = code(s, 22, INK, t2c=rules)
            box = card(words.width + 0.6, words.height + 0.36, PROMPT, PINK_FILL, radius=0.14, stroke=2).move_to(words)
            return m.VGroup(box, words)

        p1 = prompt_card(WINTER_PROMPT, {"in winter, snow": PROMPT}).move_to([0, 2.12, 0])
        h1, gap1 = 2.0, 2.45
        xs1 = [(-2 + i) * gap1 for i in range(5)]
        row1 = m.Group(*[photo(img, h1, DATA if i == 0 else EDGE).move_to([x, 0.28, 0])
                         for i, (x, img) in enumerate(zip(xs1, [SOURCE] + [W_IMG[s] for s in W_S]))])
        labs1 = m.VGroup(code("source", 22, DATA), *[code(f"s = {s}", 22, TIME) for s in W_S])
        pct1 = m.VGroup(text("pixels changed", 20, MUTED), *[code(f"{v}%", 24, INK, weight=m.BOLD) for v in W_CHANGE])
        for i, x in enumerate(xs1):
            labs1[i].move_to([x, -0.98, 0])
            pct1[i].move_to([x, -1.45, 0])
        for g in (*labs1[1:],):
            sit(g, baseline(labs1[1]))
        sit(labs1[0], baseline(labs1[1]))
        for g in pct1[2:]:
            sit(g, baseline(pct1[1]))
        sit(pct1[0], baseline(pct1[1]))
        nosnow = pill("no snow at any strength", RED, RED_FILL, 22).move_to([np.mean(xs1[1:]), -2.15, 0])
        cap1 = cite("generated on this Mac · SD 2.1-base · Euler, guidance 7.5 · seed 0").move_to([0, -3.05, 0])
        cap2 = cite("strength s: noise to t = 1000·s, then int(50·s) steps · prompts are read as in Episode 15")
        cap2.move_to([0, -3.4, 0])
        assert row1.get_right()[0] < 6.6 and cap2.width < 12.8 and p1.width < 12.8, (row1.get_right(), cap1.width, p1.width)

        self.say(
            "[First, the prompt as asked, read as in episode fifteen]: [the same lighthouse, at dusk in winter, snow].",
            m.AnimationGroup(self.new_header("02 · strength", "Keep the layout, or repaint it"), m.FadeIn(row1[0]),
                             m.FadeIn(labs1[0]), m.FadeIn(cap1), m.FadeIn(cap2)),
            m.AnimationGroup(m.FadeIn(p1, shift=0.1 * m.DOWN), m.Circumscribe(glyphs(p1[1], "in winter, snow"), color=PROMPT, buff=0.06)),
        )
        self.say(
            "[At every strength, the layout stays], [and no snow appears at all].",
            m.LaggedStart(*[m.AnimationGroup(m.FadeIn(row1[i], target_position=row1[0]), m.FadeIn(labs1[i]))
                            for i in range(1, 5)], lag_ratio=0.35, run_time=2.2),
            m.FadeIn(nosnow, scale=0.85),
        )
        self.say(
            "[The pixels change by only six to ten percent]: "
            "[two words of snow, inside a prompt that already fits the picture, were not enough].",
            m.LaggedStart(*[m.FadeIn(p, shift=0.1 * m.UP) for p in pct1], lag_ratio=0.2, run_time=1.4),
            m.Circumscribe(p1, color=PROMPT, buff=0.06),
        )
        self.wait(0.3)
        screen1 = m.Group(p1, row1, labs1, pct1, nosnow)

        p2 = prompt_card("a lighthouse on a snowy rocky coast in winter, snow-covered rocks,\nfalling snow, at dusk, oil painting",
                         {r"snowy|snow-covered rocks|falling snow": PROMPT}).move_to([0, 2.12, 0])
        assert " ".join(p2[1].text.split()) == SNOW_PROMPT
        h2, gap2 = 1.8, 2.12
        xs2 = [(-2.5 + i) * gap2 for i in range(6)]
        imgs2 = [SOURCE] + [S_IMG[s] for s in S_S]
        row2 = m.Group(*[photo(img, h2, DATA if i == 0 else EDGE).move_to([x, 0.45, 0]) for i, (x, img) in enumerate(zip(xs2, imgs2))])
        labs2 = m.VGroup(code("source", 22, DATA), *[code(f"s = {s:g}", 22, TIME) for s in S_S])
        pct2 = m.VGroup(text("pixels changed", 20, MUTED), *[code(f"{v}%", 24, INK, weight=m.BOLD) for v in S_CHANGE])
        for i, x in enumerate(xs2):
            labs2[i].move_to([x, -0.75, 0])
            pct2[i].move_to([x, -1.18, 0])
        for g in labs2:
            sit(g, baseline(labs2[1]))
        for g in pct2:
            sit(g, baseline(pct2[1]))
        kept = m.VGroup(m.Line([xs2[1] - h2 / 2, -1.46, 0], [xs2[3] + h2 / 2, -1.46, 0]).set_stroke(GOOD, 3))
        kept.add(text("the picture is kept", 20, GOOD, weight=m.BOLD).next_to(kept[0], m.DOWN, buff=0.1))
        rings = m.VGroup(tower_ring(row2[0], h2, GOOD), tower_ring(row2[5], h2, GOOD))
        leak = m.VGroup(code(f"top of SD's schedule, t = 1000:  √ᾱ = {SD_ROOT[1.0]:.3f}  → about 7% survives", 22, TIME),
                        cite("Lin et al., 2023: the schedule never reaches pure noise, so the layout leaks through")
                        ).arrange(m.DOWN, buff=0.12)
        leak_box = card(leak.width + 0.6, leak.height + 0.4, TIME, CARD, radius=0.14, stroke=2).move_to(leak)
        leak_g = m.VGroup(leak_box, leak).move_to([0, -2.52, 0])
        cap3 = cite("generated on this Mac · SD 2.1-base · Euler, guidance 7.5 · seed 0 · the explicit snow prompt").move_to([0, -3.38, 0])
        assert row2.get_left()[0] > -6.6 and leak_g.get_bottom()[1] > cap3.get_top()[1] + 0.05, (row2.get_left(), leak_g.get_bottom())
        assert leak_g.get_top()[1] < kept[1].get_bottom()[1] - 0.05 and row2.get_top()[1] < p2.get_bottom()[1] - 0.1, (leak_g.get_top(), kept[1].get_bottom(), row2.get_top(), p2.get_bottom())
        assert round(100 * SD_ROOT[1.0]) == 7

        self.say(
            "[So we made the prompt explicit]: [a snowy rocky coast, snow-covered rocks, falling snow].",
            m.AnimationGroup(m.FadeOut(screen1), m.FadeOut(cap1), m.FadeOut(cap2), m.FadeIn(row2[0]), m.FadeIn(labs2[0]),
                             m.FadeIn(pct2[0]), m.FadeIn(cap3)),
            m.FadeIn(p2, shift=0.1 * m.DOWN),
        )
        self.say(
            "[From zero point three to zero point seven, the picture is kept]; "
            "[at zero point nine, the same layout turns into a snowy scene].",
            m.AnimationGroup(m.LaggedStart(*[m.AnimationGroup(m.FadeIn(row2[i], target_position=row2[0]), m.FadeIn(labs2[i]),
                                                              m.FadeIn(pct2[i])) for i in (1, 2, 3)], lag_ratio=0.35, run_time=1.8),
                             m.FadeIn(kept)),
            m.AnimationGroup(m.FadeIn(row2[4], target_position=row2[3]), m.FadeIn(labs2[4]), m.FadeIn(pct2[4])),
        )
        self.say(
            "[At strength one, it is nearly a new picture], [yet the lighthouse still stands on the right].",
            m.AnimationGroup(m.FadeIn(row2[5], target_position=row2[4]), m.FadeIn(labs2[5]), m.FadeIn(pct2[5])),
            m.AnimationGroup(m.Create(rings[0]), m.Create(rings[1])),
        )
        self.say(
            "[That is because Stable Diffusion's schedule never reaches pure noise]: "
            "[even at the top, about seven percent of the signal survives].",
            m.FadeIn(leak_g, shift=0.1 * m.UP),
            m.Circumscribe(leak[0], color=TIME, buff=0.08),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 3 · the swirl: SDEdit from a sketch

    def swirl_sketch(self):
        self.next_section("3 · Noise as big as the mistakes", notes=(
            "Our computation with diffusion_reference: the sketch is arm 0's centre line (300 points) pushed sideways by "
            "0.13·sin(…) plus a small random wobble (seed 3), mean distance "
            f"{FRONT[0][1]:.3f} from the swirl, every point nearest to arm 0. SDEdit: q_sample to step t with "
            "gaussian_noise(300, seed=161), then ddim_sample with the exact denoiser (exact_eps), 50 steps from t to 0 "
            "(η = 0: no fresh noise). Measured at 27 values of t: mean distance moved from the sketch (faithfulness; "
            "lower = more faithful) and mean distance to the swirl's centre line (realism; the data's own is "
            f"{DATA_DIST:.3f}). Runs shown: "
            + "; ".join(f"t = {t}: σ = {SIG[t]:.3f}, moved {FRONT[t][0]:.3f}, to swirl {FRONT[t][1]:.3f}, "
                        f"on arm 0 {FRONT[t][2]:.1%}" for t in T_RUNS)
            + f". σ at t = 20 ({SIG[20]:.3f}) is about the sketch's own error ({FRONT[0][1]:.3f}). The swirl's arms are "
            "thin and close (a 2D toy), so its sweet spot is at small t; for pictures Meng et al. 2021 (Fig. 3) found good "
            "results for t0 ∈ [0.3, 0.6] in their example. Inversion: Episode 9 (DDIM run backwards; approximate)."))
        lim, size = 1.2, 4.7
        pl = plane((-3.55, -0.4, 0), size, lim)
        frame = m.Square(size).set_stroke(EDGE, 2).move_to(pl.coords(0, 0))
        band = swirl_band(pl, 10, 0.2)
        stroke = m.VMobject().set_points_smoothly([at(pl, p) for p in SKETCH[::5]] + [at(pl, SKETCH[-1])])
        stroke.set_stroke(SKETCH_C, 3, opacity=0.55).set_z_index(1)
        dots = Dots(pl, SKETCH, lim, SKETCH_C, radius=0.03)
        sk_l = text("a rough sketch of one arm", 22, SKETCH_C, weight=m.BOLD).next_to(frame, m.UP, buff=0.15)
        sk_l.align_to(frame, m.LEFT)
        readout = code("noise to t = 5", 24, TIME, weight=m.BOLD).next_to(frame, m.UP, buff=0.15).align_to(frame, m.LEFT)
        sit(readout, baseline(sk_l))
        cap = m.VGroup(cite("our computation · the swirl's exact denoiser"), cite("DDIM, 50 steps from t back to 0"))
        cap.arrange(m.DOWN, buff=0.06).next_to(frame, m.DOWN, buff=0.18)

        pt = Plot((0, 0.9), (0, 0.1), width=4.5, height=3.0, center=(3.45, 0.35, 0), x_ticks=[0, 0.3, 0.6, 0.9],
                  y_ticks=[0, 0.05, 0.1], x_label="distance moved from the sketch", y_label="distance to the swirl")
        xs_ = [FRONT[t][0] for t in T_GRID]
        ys_ = [FRONT[t][1] for t in T_GRID]
        curve = pt.line(xs_, ys_, TIME, 4)
        data_l = pt.hline(DATA_DIST, FAINT, 2)
        data_t = cite("the data itself").next_to(pt.coords(0.9, DATA_DIST), m.UP, buff=0.1).align_to(pt.coords(0.9, 0), m.RIGHT)
        marks, mlabs = m.VGroup(), m.VGroup()
        for t, side in zip(T_RUNS, (m.RIGHT, m.UR, m.DOWN)):
            d_ = pt.dot(FRONT[t][0], FRONT[t][1], TIME, 0.09)
            marks.add(d_)
            mlabs.add(code(f"t = {t}", 20, TIME).next_to(d_, side, buff=0.1))
        corner_a = text("faithful,\nnot realistic", 20, SKETCH_C, weight=m.SEMIBOLD).move_to(pt.coords(0.1, 0.09), aligned_edge=m.LEFT)
        corner_b = text("realistic,\nnot faithful", 20, DATA, weight=m.SEMIBOLD).move_to(pt.coords(0.6, 0.06), aligned_edge=m.LEFT)
        sweet = m.Circle(radius=0.22).set_stroke(GOOD, 3).move_to(marks[1])
        sig_note = m.VGroup(text(f"at t = 20, the noise (σ = {SIG[20]:.3f})", 20, GOOD),
                            text(f"is about the sketch's error ({FRONT[0][1]:.3f})", 20, GOOD)).arrange(m.DOWN, buff=0.06)
        sig_note.move_to([pt.get_center()[0], -2.32, 0])
        inv = pill("Episode 9 · inversion: photo → noise → edit", PROMPT, PINK_FILL, 20).move_to([3.3, -3.05, 0])
        assert pt.get_left()[0] > frame.get_right()[0] + 0.4 and inv.get_right()[0] < 6.6, (pt.get_left(), inv.get_right())
        assert pt.get_bottom()[1] > sig_note.get_top()[1] + 0.1 and sig_note.get_bottom()[1] > inv.get_top()[1] + 0.1, (pt.get_bottom(), sig_note.get_top(), sig_note.get_bottom(), inv.get_top())
        assert 1 - FRONT_ON[5] > 0.5 and FRONT_ON[20] > 0.85                         # "most stay off the arm", "land on the arm"

        def run(t, prev=None):
            path = RUNS[t].xs
            jump = np.stack([SKETCH, path[0]])
            out = [DATA if d < 0.06 else RED for d in nearest_on_swirl(path[-1])[0]]
            seq = []
            if prev is not None:
                seq.append(dots.walk(np.stack([RUNS[prev].xs[-1], SKETCH]), DATA, SKETCH_C, 0.6))
            seq += [dots.walk(jump, SKETCH_C, NOISE, 0.9), dots.walk(path, NOISE, DATA, 2.0, final=out)]
            return m.Succession(*seq)

        self.say(
            "[Why does this work? Back on the swirl, here is a rough sketch of one arm]: [close to the data, but not on it].",
            m.AnimationGroup(self.new_header("03 · the swirl", "Noise as big as the mistakes"), m.FadeIn(frame), m.FadeIn(band),
                             m.Create(stroke), m.FadeIn(sk_l), m.FadeIn(cap)),
            m.AnimationGroup(m.FadeIn(dots, scale=0.5), m.FadeIn(pt.axes), m.FadeIn(pt.ticks), m.FadeIn(pt.x_label),
                             m.FadeIn(pt.y_label), m.FadeIn(data_l), m.FadeIn(data_t)),
        )
        self.say(
            "[Noise it to step five, and let the exact denoiser bring it back]: [the points barely move, and most stay off the arm].",
            m.AnimationGroup(m.FadeOut(sk_l, shift=0.2 * m.UP), m.FadeIn(readout, shift=0.2 * m.UP), run(5)),
            m.AnimationGroup(m.FadeIn(marks[0], scale=0.5), m.FadeIn(mlabs[0]), m.FadeIn(corner_a)),
        )
        self.say(
            "[Noise to step twenty, about the size of the sketch's mistakes]: [the points land on the arm, near where they were drawn].",
            m.AnimationGroup(m.Transform(readout, code("noise to t = 20", 24, TIME, weight=m.BOLD).move_to(readout, aligned_edge=m.RIGHT)),
                             run(20, 5)),
            m.AnimationGroup(m.FadeIn(marks[1], scale=0.5), m.FadeIn(mlabs[1]), m.Create(sweet), m.FadeIn(sig_note)),
        )
        self.say(
            "[Noise to step three hundred]: [they land on the swirl, but scattered, almost half on the other arm].",
            m.AnimationGroup(m.Transform(readout, code("noise to t = 300", 24, TIME, weight=m.BOLD).move_to(readout, aligned_edge=m.RIGHT)),
                             run(300, 20)),
            m.AnimationGroup(m.FadeIn(marks[2], scale=0.5), m.FadeIn(mlabs[2]), m.FadeIn(corner_b)),
        )
        self.say(
            "[Our own measurement, at every step]: [the higher you start, the more realistic], [and the less faithful to the sketch].",
            m.Create(curve, run_time=2.0),
            m.Circumscribe(corner_b, color=DATA, buff=0.08),
            m.Circumscribe(corner_a, color=SKETCH_C, buff=0.08),
        )
        self.say(
            "[For a real photo, episode nine's inversion gives a gentler start]: "
            "[run the photo backwards to its own noise, then edit from there].",
            m.FadeIn(inv, shift=0.1 * m.UP),
            m.Circumscribe(inv, color=PROMPT, buff=0.08),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 4 · inpainting

    def inpainting(self):
        self.next_section("4 · Repaint only the hole", notes=(
            "Left: an illustration on the source picture (no inpainting run exists): the masked tower drawn as noise. "
            "Mechanism (notes §16): sample as usual while overwriting the unmasked part with the original, noised to the "
            "current step (Song et al. 2021's imputation; RePaint, Lugmayr, Danelljan, Romero, Yu, Timofte & Van Gool, "
            "arXiv:2201.09865, Jan 2022, CVPR 2022: 'we only alter the reverse diffusion iterations by sampling the "
            "unmasked regions using the given image information', plus resampling). Inpainting models (GLIDE, SD "
            "inpainting) instead take the mask and the masked image as extra input channels. Our computation on the swirl: "
            f"keep x = {XK}, regenerate y; {N_IN} points, 100 ancestral DDIM steps (η = 1) with the exact denoiser; after "
            "every step x ← √ᾱ_s·0.45 + √(1 − ᾱ_s)·z. Within 0.06 of the swirl's centre line at the end: plain "
            f"{ON['plain']:.1%}, with RePaint-style resampling (re-noise one step and redo it, 5 times per step; jump "
            f"length 1) {ON['repaint']:.1%}. The exact conditional p(y | x = 0.45) of the swirl mixture: {EXACT_ON:.1%}. "
            "The line x = 0.45 crosses the arms near y = " + ", ".join(f"{y:.2f}" for y in np.unique(np.round(CROSS_Y, 1))) + "."))
        hl = 2.7
        src = photo(SOURCE, hl, EDGE).move_to([-4.75, 0.65, 0])
        mimg, (fx0, fx1, fy0, fy1) = masked_source()
        msk = photo(mimg, hl, EDGE).move_to(src)
        corner = src.get_corner(m.UL) + np.array([0.05, -0.05, 0])
        mw, mh = (fx1 - fx0) * hl, (fy1 - fy0) * hl
        mask = m.Rectangle(width=mw, height=mh).set_stroke(SKETCH_C, 3).move_to(corner + np.array([fx0 * hl + mw / 2, -(fy0 * hl + mh / 2), 0]))
        sq = lambda c: m.Square(0.24).set_stroke(c, 2).set_fill(FILLS[c], 1)  # noqa: E731
        leg = m.VGroup(m.VGroup(sq(DATA), text("kept: the original, re-noised", 20, INK)).arrange(m.RIGHT, buff=0.18),
                       m.VGroup(sq(NOISE), text("masked: the model's work", 20, INK)).arrange(m.RIGHT, buff=0.18))
        leg.arrange(m.DOWN, buff=0.2, aligned_edge=m.LEFT).next_to(src, m.DOWN, buff=0.35).align_to(src, m.LEFT)
        ill = cite("illustration, not a run").next_to(leg, m.DOWN, buff=0.25).align_to(leg, m.LEFT)

        lim, size = 1.25, 4.3
        pl = plane((1.15, -0.25, 0), size, lim)
        frame = m.Square(size).set_stroke(EDGE, 2).move_to(pl.coords(0, 0))
        band = swirl_band(pl, 10, 0.2)
        line = m.Line(at(pl, (XK, -lim)), at(pl, (XK, lim))).set_stroke(SKETCH_C, 2, opacity=0.6).set_z_index(2)
        top = code(f"keep x = {XK} · mask y", 22, SKETCH_C, weight=m.BOLD).next_to(frame, m.UP, buff=0.15)
        dots = Dots(pl, IN_PLAIN[0], lim, NOISE, radius=0.034)
        cap = cite("our computation · exact denoiser · 100 steps").next_to(frame, m.DOWN, buff=0.2)
        rx = 5.2
        s1 = title_text(f"{100 * ON['plain']:.0f}%", 60, INK).move_to([rx, 1.2, 0])
        s1_l = text("plain overwrite:\nland on the swirl", 20, MUTED).next_to(s1, m.DOWN, buff=0.12)
        s2 = title_text(f"{100 * ON['repaint']:.0f}%", 60, GOOD).move_to([rx, -1.0, 0])
        s2_l = text("with RePaint's\nback-and-forth", 20, MUTED).next_to(s2, m.DOWN, buff=0.12)
        s2_c = cite("Lugmayr et al., 2022").next_to(s2_l, m.DOWN, buff=0.12)
        assert frame.get_left()[0] > max(src.get_right()[0], leg.get_right()[0], ill.get_right()[0]) + 0.25 and s1_l.get_left()[0] > frame.get_right()[0] + 0.2, \
            (frame.get_left(), src.get_right(), s1_l.get_left(), frame.get_right())
        assert s2_c.get_bottom()[1] > -3.5 and max(g.get_right()[0] for g in (s1_l, s2_l, s2_c)) < 6.6
        fin_plain = [DATA if d < 0.06 else RED for d in nearest_on_swirl(IN_PLAIN[-1])[0]]
        fin_rep = [DATA if d < 0.06 else RED for d in nearest_on_swirl(IN_REPAINT[-1])[0]]

        self.say(
            "[In inpainting, only a masked part of a picture is regenerated], [and the rest is kept].",
            m.AnimationGroup(self.new_header("04 · inpainting", "Repaint only the hole"), m.FadeIn(src)),
            m.AnimationGroup(m.FadeIn(msk), m.Create(mask), m.FadeIn(ill)),
        )
        self.say(
            "[The simplest way]: [denoise as usual, but after every step, overwrite the kept part with the original, noised to match].",
            m.FadeIn(leg[1], shift=0.1 * m.UP),
            m.FadeIn(leg[0], shift=0.1 * m.UP),
        )
        self.say(
            "[On the swirl, keep x at zero point four five, and mask y]: [every step, x is overwritten], "
            "[and only y follows the denoiser].",
            m.AnimationGroup(m.FadeIn(frame), m.FadeIn(band), m.Create(line), m.FadeIn(top), m.FadeIn(cap)),
            m.FadeIn(dots, scale=0.5),
            dots.walk(IN_PLAIN, NOISE, DATA, 3.0, final=fin_plain),
        )
        self.say(
            "[Eighty percent land on the swirl]: [the free part never fully agrees with the kept part].",
            m.AnimationGroup(m.FadeIn(s1, shift=0.1 * m.UP), m.FadeIn(s1_l)),
            m.Circumscribe(s1, color=INK, buff=0.1),
        )
        self.say(
            "[A twenty twenty-two method called RePaint adds a back-and-forth]: [after each step, add noise back and step again]. "
            "[Now every point lands on the swirl].",
            m.AnimationGroup(m.FadeIn(s2_c), dots.walk(np.stack([IN_PLAIN[-1], IN_REPAINT[0]]), DATA, NOISE, 0.8)),
            dots.walk(IN_REPAINT, NOISE, DATA, 3.0, final=fin_rep),
            m.AnimationGroup(m.FadeIn(s2, shift=0.1 * m.UP), m.FadeIn(s2_l)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 5 · ControlNet

    def control(self):
        self.next_section("5 · Follow a sketch", notes=(
            "ControlNet (Zhang, Rao & Agrawala, 'Adding Conditional Control to Text-to-Image Diffusion Models', "
            "arXiv:2302.05543, Feb 2023; ICCV 2023), notes §16: the production model is locked; a trainable copy of its "
            "encoder (and middle block) reads the control image; the copy is connected through 'zero convolutions' (1×1 "
            "convolutions with weights and bias initialised to 0), so at the start of training it adds exactly zero. "
            "Conditions tested: Canny edges, Hough lines, user scribbles, human key points, segmentation maps, normals, "
            "depth, cartoon line drawings. Training cost on SD: about 23% more GPU memory and 34% more time per iteration "
            "(A100 40 GB). The edge map shown is our own: a Sobel filter on the source lighthouse (top 11% of gradient "
            "magnitudes, 256×256), computed in this scene — not a ControlNet run. Diagram simplified (the copy feeds every "
            "decoder level and the middle block; drawn as one path)."))
        y_top, y_bot = 0.95, -1.25
        enc = block("encoder", 2.6, 1.0, FROZEN_C, AMBER_FILL).move_to([-2.1, y_top, 0])
        mid = block("middle", 1.4, 1.0, FROZEN_C, AMBER_FILL).move_to([0.3, y_top, 0])
        dec = block("decoder", 2.1, 1.0, FROZEN_C, AMBER_FILL).move_to([2.45, y_top, 0])
        links = m.VGroup(arrow_to(enc.get_right(), mid.get_left(), FROZEN_C, 4, 0.04),
                         arrow_to(mid.get_right(), dec.get_left(), FROZEN_C, 4, 0.04))
        model = m.VGroup(enc, mid, dec, links)
        inp = m.VGroup(tex(r"x_t", 34, NOISE), text("+ prompt", 22, PROMPT)).arrange(m.RIGHT, buff=0.15).move_to([-5.35, y_top, 0])
        out = m.VGroup(tex(r"\hat\varepsilon", 38, MODEL), text("noise guess", 20, MUTED)).arrange(m.DOWN, buff=0.08).move_to([4.75, y_top, 0])
        a_in = arrow_to(inp.get_right(), enc.get_left(), MUTED)
        a_out = arrow_to(dec.get_right(), out.get_left(), MODEL)
        lock = icon(recolored("lock", FROZEN_C, AMBER_FILL, PREFIX), 0.55)
        lock_l = text("locked: not trained", 22, FROZEN_C, weight=m.BOLD)
        lock_g = m.VGroup(lock, lock_l).arrange(m.RIGHT, buff=0.2).next_to(model, m.UP, buff=0.3)
        edges = framed(picture(tint(EDGES, SKETCH_C, 0, 1), 1.55), SKETCH_C, buff=0.05).move_to([-5.5, y_bot, 0])
        edges_l = cite("edges of our lighthouse").next_to(edges, m.DOWN, buff=0.12)
        z_in = zero_conv().move_to([-4.05, y_bot, 0])
        copy = block("trainable copy\nof the encoder", 2.6, 1.05, TRAIN_C, PINK_FILL, 22).move_to([-2.1, y_bot, 0])
        z_out = zero_conv().move_to([2.45, -0.2, 0])
        a_e = arrow_to(edges.get_right(), z_in.get_left(), SKETCH_C, 4, 0.08)
        a_z = arrow_to(z_in.get_right(), copy.get_left(), SKETCH_C, 4, 0.08)
        a_c = arrow_to(copy.get_right(), z_out.get_left() + [0, -0.08, 0], TRAIN_C, 4, 0.1)
        a_up = arrow_to(z_out.get_top(), dec.get_bottom(), TRAIN_C, 4, 0.06)
        a_x = arrow_to(enc.get_bottom() + [-0.85, 0, 0], copy.get_top() + [-0.85, 0, 0], NOISE, 3, 0.08)
        zero_l = text("zero convolutions:\nweights start at 0", 20, TRAIN_C, weight=m.SEMIBOLD).next_to(z_out, m.RIGHT, buff=0.3)
        plus0 = code("+ 0", 26, TRAIN_C, weight=m.BOLD).next_to(a_up, m.LEFT, buff=0.12)
        conds = m.VGroup(*[chip(w, SKETCH_C, PINK_FILL, width=text(w, 22, weight=m.BOLD).width + 0.5, height=0.56, size=22)
                           for w in ("edges", "depth", "pose", "scribbles", "segmentation")]).arrange(m.RIGHT, buff=0.22)
        for c_ in conds[1:]:
            sit(c_[1], baseline(conds[0][1]))
        conds.move_to([1.2, -2.6, 0])
        src = cite("Zhang, Rao & Agrawala, 2023 · training costs about 23% more memory and 34% more time per step")
        src.move_to([0, -3.35, 0])
        assert out.get_right()[0] < 6.6 and zero_l.get_right()[0] < 6.6 and conds.get_right()[0] < 6.6, (out.get_right(), zero_l.get_right())
        assert edges_l.get_bottom()[1] > src.get_top()[1] and conds.get_left()[0] > edges_l.get_right()[0] + 0.2, (edges_l.get_bottom(), conds.get_left())

        self.say(
            "[Words are blunt about layout]. "
            "[To follow edges, depth or a pose, Zhang and colleagues built ControlNet in twenty twenty-three].",
            m.AnimationGroup(self.new_header("05 · control", "Follow a sketch"), m.FadeIn(inp), m.GrowArrow(a_in), m.FadeIn(model),
                             m.GrowArrow(a_out), m.FadeIn(out)),
            m.AnimationGroup(m.FadeIn(edges), m.FadeIn(edges_l), m.LaggedStart(*[m.FadeIn(c_, shift=0.1 * m.UP) for c_ in conds],
                                                                                lag_ratio=0.2, run_time=1.4), m.FadeIn(src)),
        )
        self.say(
            "[The big model is locked]. [A trainable copy of its encoder reads the control image], "
            "[and feeds back in through convolutions that start at zero].",
            m.FadeIn(lock_g, shift=0.1 * m.DOWN),
            m.AnimationGroup(m.GrowArrow(a_e), m.FadeIn(z_in), m.GrowArrow(a_z), m.FadeIn(copy, target_position=enc),
                             m.GrowArrow(a_x)),
            m.AnimationGroup(m.GrowArrow(a_c), m.FadeIn(z_out), m.GrowArrow(a_up), m.FadeIn(zero_l)),
        )
        self.say(
            "[So at the start of training, the copy adds exactly nothing], [and cannot harm the model it steers].",
            m.AnimationGroup(m.FadeIn(plus0, shift=0.1 * m.RIGHT), m.Circumscribe(z_out, color=TRAIN_C, buff=0.08),
                             m.Circumscribe(z_in, color=TRAIN_C, buff=0.08)),
            m.Circumscribe(model, color=FROZEN_C, buff=0.1),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 6 · LoRA and DreamBooth

    def finetunes(self):
        self.next_section("6 · Teach it something new", notes=(
            "LoRA (Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang & Chen, arXiv:2106.09685, Jun 2021; ICLR 2022): h = W₀x + BAx, "
            "B ∈ ℝ^{d×r}, A ∈ ℝ^{r×k}; B starts at zero; trainable numbers per matrix r(d + k). Shapes read from SD "
            "2.1-base's own unet file header (precompute.py → sd21_attention_shapes.json; the lighthouse model): "
            f"{ONE} is {D_ONE}×{K_ONE} = {FULL_ONE:,}; rank {RANK}: {LORA_ONE:,} ({LORA_ONE / FULL_ONE:.1%}). All "
            f"{len(PROJ)} attention projections (to_q, to_k, to_v, to_out of 16 transformer blocks; cross-attention keys "
            f"and values are {PROJ[ONE.replace('attn1.to_q', 'attn2.to_k')][0]}×1024): rank {RANK} → {LORA_ALL:,} numbers, "
            f"{LORA_MB:.2f} MB in fp16 (our arithmetic; real files add a little metadata, and many use higher ranks), "
            f"against the denoiser's {SD_PARAMS:,} parameters / {UNET_GB:.2f} GB fp16 (mac_runs.md §5). Notes §16: "
            "'files are megabytes instead of gigabytes'. DreamBooth (Ruiz, Li, Jampani, Pritch, Rubinstein & Aberman, "
            "arXiv:2208.12242, Aug 2022; CVPR 2023): 3–5 images of a subject bound to a rare identifier ('a [V] dog'), "
            "fine-tuning the whole model with a class-specific prior-preservation loss on ~1,000 images the frozen model "
            "generated for the class prompt (λ = 1); ~1,000 iterations, about 5 minutes on one A100 for Stable Diffusion. "
            "The four pictures of Pixel are an illustration (our mascot standing in for the subject); often implemented "
            "with LoRA to save memory."))
        cw, ch = 6.25, 5.65
        left = card(cw, ch, EDGE, CARD, radius=0.2, stroke=2).move_to([-3.35, -0.5, 0])
        right = card(cw, ch, EDGE, CARD, radius=0.2, stroke=2).move_to([3.35, -0.5, 0])

        def head(name, sub, box):
            t_ = title_text(name, 34, INK)
            s_ = cite(sub)
            g = m.VGroup(t_, s_).arrange(m.RIGHT, buff=0.3)
            sit(s_, baseline(t_))
            return g.align_to(box, m.UL).shift(0.35 * m.RIGHT + 0.3 * m.DOWN)

        l_head = head("LoRA", "low-rank adapters · Hu et al., 2021", left)
        r_head = head("DreamBooth", "Ruiz et al., 2023", right)
        s = 1.7
        cx, cy = left.get_center()[0], 0.35
        W = m.Square(s).set_stroke(FROZEN_C, 3).set_fill(AMBER_FILL, 1)
        W_l = tex("W", 40, FROZEN_C).move_to(W)
        plus = tex("+", 40, INK)
        Bm = m.Rectangle(width=0.2, height=s).set_stroke(TRAIN_C, 3).set_fill(PINK_FILL, 1)
        times = tex(r"\times", 36, INK)
        Am = m.Rectangle(width=s, height=0.2).set_stroke(TRAIN_C, 3).set_fill(PINK_FILL, 1)
        eq = m.VGroup(m.VGroup(W, W_l), plus, Bm, times, Am).arrange(m.RIGHT, buff=0.3).move_to([cx, cy, 0])
        W_d = code(f"{D_ONE} × {K_ONE}", 20, FROZEN_C).next_to(W, m.DOWN, buff=0.12)
        B_d = tex("B", 32, TRAIN_C).next_to(Bm, m.LEFT, buff=0.08).shift(0.5 * m.UP)
        A_d = tex("A", 32, TRAIN_C).next_to(Am, m.UP, buff=0.1)
        r_d = code(f"rank {RANK}", 20, TRAIN_C).next_to(Am, m.DOWN, buff=0.12)
        sit(r_d, baseline(W_d))
        frozen_t = text("frozen", 20, FROZEN_C, weight=m.SEMIBOLD).next_to(W, m.UP, buff=0.1)
        learned_t = text("learned", 20, TRAIN_C, weight=m.SEMIBOLD).next_to(m.VGroup(Bm, Am), m.UP, buff=0.1).shift(0.2 * m.UP)
        sit(learned_t, baseline(frozen_t))
        n1 = code(f"one attention matrix: {FULL_ONE:,}", 20, INK)
        n2 = code(f"rank {RANK}: {RANK} × {D_ONE + K_ONE} = {LORA_ONE:,} · {100 * LORA_ONE / FULL_ONE:.1f}%", 20, TRAIN_C)
        n3 = code(f"all {len(PROJ)} matrices: {LORA_ALL:,} ≈ {LORA_MB:.1f} MB", 20, TRAIN_C)
        n4 = code(f"the denoiser itself: {UNET_GB:.2f} GB", 20, FROZEN_C)
        nums = m.VGroup(n1, n2, n3, n4).arrange(m.DOWN, buff=0.16, aligned_edge=m.LEFT).move_to([cx, -1.85, 0])
        l_src = cite("our arithmetic · our lighthouse model's shapes").next_to(nums, m.DOWN, buff=0.22)
        assert nums.width < cw - 0.5 and l_src.width < cw - 0.3 and l_src.get_bottom()[1] > left.get_bottom()[1] + 0.1, (nums.width, l_src.width, l_src.get_bottom())

        rcx = right.get_center()[0]
        subs = m.Group()
        for i, (sc, ang) in enumerate(((0.62, 0.0), (0.5, 0.25), (0.7, -0.15), (0.55, 0.1))):
            fr = m.Square(1.05).set_stroke(EDGE, 2).set_fill(BG, 1)
            p_ = pixel(sc, seed=7).rotate(ang).move_to(fr)
            subs.add(m.Group(fr, p_))
        subs.arrange(m.RIGHT, buff=0.22).move_to([rcx, 0.95, 0])
        subs_l = cite("for example: four pictures of one subject").next_to(subs, m.DOWN, buff=0.14)
        token = code("a [V] mascot", 26, SKETCH_C, weight=m.BOLD)
        tok_box = card(token.width + 0.5, token.height + 0.36, SKETCH_C, PINK_FILL, radius=0.14, stroke=2).move_to(token)
        tok = m.VGroup(tok_box, token).move_to([rcx - 1.2, -0.55, 0])
        tok_l = text("a rare word\nfor the subject", 20, MUTED).next_to(tok, m.RIGHT, buff=0.3)
        prior = m.VGroup(text("+ prior preservation:", 20, INK, weight=m.BOLD),
                         text("also train on about 1,000 pictures of", 20, INK),
                         text("the plain class, made by the frozen model", 20, INK)).arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT)
        prior.move_to([rcx, -1.75, 0])
        facts = m.VGroup(code("3–5 images · ~1,000 steps", 20, MUTED),
                         code("~5 min on one A100 (for SD)", 20, MUTED)).arrange(m.DOWN, buff=0.12).move_to([rcx, -2.78, 0])
        assert facts.width < cw - 0.4 and prior.width < cw - 0.4 and subs.width < cw - 0.4, (facts.width, prior.width, subs.width)
        lora_g = m.VGroup(left, l_head, eq, W_d, B_d, A_d, r_d, frozen_t, learned_t, nums, l_src)
        home = lora_g.get_center().copy()
        lora_g.move_to([0, home[1], 0])                 # alone, the LoRA card sits in the middle; it moves aside later

        self.say(
            "[To teach a new style, people train a small add-on called a LoRA]: [a low-rank adapter, from twenty twenty-one].",
            m.AnimationGroup(self.new_header("06 · fine-tunes", "Teach it something new"), m.FadeIn(left), m.FadeIn(l_head[0])),
            m.FadeIn(l_head[1]),
        )
        self.say(
            "[Instead of changing a big weight matrix, it learns two thin ones], [whose product is the change].",
            m.AnimationGroup(m.FadeIn(eq[0]), m.FadeIn(W_d), m.FadeIn(frozen_t)),
            m.AnimationGroup(m.FadeIn(eq[1:], shift=0.1 * m.LEFT), m.FadeIn(B_d), m.FadeIn(A_d), m.FadeIn(r_d), m.FadeIn(learned_t)),
        )
        self.say(
            "[On one attention matrix of our lighthouse model], [rank four trains just two and a half percent of the numbers].",
            m.FadeIn(n1),
            m.FadeIn(n2),
        )
        self.say(
            "[On all of them, that is about one point seven megabytes], [against one point seven gigabytes for the denoiser itself].",
            m.AnimationGroup(m.FadeIn(n3), m.FadeIn(l_src)),
            m.FadeIn(n4),
        )
        self.say(
            "[To teach one particular subject, DreamBooth, from Ruiz and colleagues, fine-tunes on three to five photos], "
            "[tied to a rare word].",
            m.Succession(lora_g.animate(run_time=0.9).move_to(home),
                         m.AnimationGroup(m.FadeIn(right), m.FadeIn(r_head),
                                          m.LaggedStart(*[m.FadeIn(g, shift=0.1 * m.UP) for g in subs], lag_ratio=0.25,
                                                        run_time=1.4), m.FadeIn(subs_l))),
            m.AnimationGroup(m.FadeIn(tok, shift=0.1 * m.UP), m.FadeIn(tok_l)),
        )
        self.say(
            "[A prior preservation loss keeps the plain class word working]: [about a thousand steps, around five minutes].",
            m.FadeIn(prior, shift=0.1 * m.UP),
            m.FadeIn(facts, shift=0.1 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ recap and the end of Module 4

    def recap(self):
        self.next_section("Recap · end of Module 4", notes=(
            "Repaint (SDEdit / image-to-image: noise part way, denoise with a new prompt), inpaint (regenerate a masked "
            "part; overwrite the kept part each step), adapt (ControlNet, LoRA, DreamBooth). Module 4 (Steering the "
            "Picture): Episode 13 class labels and classifier guidance, Episode 14 classifier-free guidance, Episode 15 "
            "text encoders and cross-attention, Episode 16 editing and control. Next: Module 5, Scaling Up (Episode 17, "
            "Latent Diffusion: 512×512×3 → 64×64×4)."))
        halo = glow(PROMPT, radius=4.0)
        words = m.VGroup(*[title_text(w, 58, c) for w, c in (("Repaint", TIME), ("Inpaint", DATA), ("Adapt", PROMPT))])
        subs = m.VGroup(*[code(s_, 20, MUTED) for s_ in ("noise part way, denoise", "regenerate a hole",
                                                          "ControlNet, LoRA, DreamBooth")])
        cols_ = [m.VGroup(w, s_).arrange(m.DOWN, buff=0.3) for w, s_ in zip(words, subs)]
        m.VGroup(*cols_).arrange(m.RIGHT, buff=0.6)
        for w in words[1:]:
            sit(w, baseline(words[0]))
        line_y = min(w.get_bottom()[1] for w in words) - 0.45
        for s_ in subs:
            sit(s_, line_y)
        mid_y = baseline(words[0]) + 0.3
        dots = m.VGroup(*[m.Dot([(l_.get_right()[0] + r_.get_left()[0]) / 2, mid_y, 0], radius=0.07, color=MUTED)
                          for l_, r_ in zip(words, words[1:])])
        row = m.VGroup(words, dots, subs).move_to(1.75 * m.UP)
        assert row.width < 12.6 and all(a.get_right()[0] + 0.25 < b.get_left()[0] for a, b in zip(subs, subs[1:])), row.width
        tag = text("One denoiser: started from a picture, or given a little extra to read.", 28, INK).next_to(row, m.DOWN, buff=0.5)

        mod = text("Module 4 · Steering the Picture · Episodes 13–16", 22, PROMPT, weight=m.BOLD)
        steps = [("13", "ask for a label"), ("14", "guidance"), ("15", "read the words"), ("16", "edit and control")]
        chips_ = m.VGroup()
        for n, s_ in steps:
            w_ = m.VGroup(code(n, 20, PROMPT, weight=m.BOLD), text(s_, 20, INK)).arrange(m.RIGHT, buff=0.16)
            box = card(w_.width + 0.45, 0.66, PROMPT if n == "16" else EDGE, PINK_FILL if n == "16" else CARD, radius=0.14, stroke=2)
            w_.move_to(box)
            sit(w_[0], baseline(w_[1]))
            chips_.add(m.VGroup(box, w_))
        chips_.arrange(m.RIGHT, buff=0.38)
        arrows = m.VGroup(*[arrow_to(a.get_right(), b.get_left(), MUTED, 3, 0.06) for a, b in zip(chips_, chips_[1:])])
        module = m.VGroup(mod, m.VGroup(chips_, arrows)).arrange(m.DOWN, buff=0.3).next_to(tag, m.DOWN, buff=0.7)
        assert module.width < 12.8, module.width
        mascot = pixel(0.42)
        nxt = pill("NEXT · MODULE 5 · SCALING UP", TEAL, TEAL_FILL, 20, mono=True)
        foot = m.VGroup(mascot, nxt).arrange(m.RIGHT, buff=0.3).next_to(module, m.DOWN, buff=0.55)
        assert foot.get_bottom()[1] > -3.55, foot.get_bottom()

        self.say(
            "[To sum up]: [repaint], [inpaint], [adapt].",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.3 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(dots[0]), m.FadeIn(words[1], shift=0.3 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(dots[1]), m.FadeIn(words[2], shift=0.3 * m.UP), m.FadeIn(subs[2])),
        )
        self.say(
            "[One denoiser]: [started from a picture instead of noise], [or given a little extra to read].",
            m.Circumscribe(row, color=PROMPT, buff=0.12),
            m.Write(tag),
            m.Circumscribe(words[2], color=PROMPT, buff=0.1),
        )
        self.say(
            "[That closes module four]: [ask for a label], [push with guidance], [read the words], [and edit what you made].",
            m.FadeIn(mod, shift=0.1 * m.UP),
            m.FadeIn(chips_[0], shift=0.1 * m.UP),
            m.AnimationGroup(m.GrowArrow(arrows[0]), m.FadeIn(chips_[1], shift=0.1 * m.UP)),
            m.AnimationGroup(m.GrowArrow(arrows[1]), m.FadeIn(chips_[2], shift=0.1 * m.UP)),
            m.AnimationGroup(m.GrowArrow(arrows[2]), m.FadeIn(chips_[3], shift=0.1 * m.UP)),
        )
        self.say(
            "[Next, we make all of this cheaper], [by diffusing in a small, compressed space].",
            m.FadeIn(mascot, shift=0.2 * m.UP),
            m.FadeIn(nxt, shift=0.2 * m.UP),
        )
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, tag, module, foot)], run_time=0.9)


ORIGIN_ = np.zeros(3)
