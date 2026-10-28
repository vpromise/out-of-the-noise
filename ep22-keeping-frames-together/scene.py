"""Out of the Noise · Episode 22 · Keeping Frames Together (Module 6 · Video and Your Mac)

Consistency across frames is the hard part of video. Running example: the digit model (research/experiments/mac_runs.md
§6), a real run on this Mac, told honestly; then the swirl as a computed, idealised joint model; then four dated cards.

The story:
- THE REAL RUN (mac_runs.md §6, arrays/digit_video_noise_correlation.npz, recomputed and asserted here): our trained
  digit model generated 20 clips of 8 frames (DDIM 50, w = 1, one label per clip). Frame k starts from
  ε_k = √ρ·ε_shared + √(1−ρ)·ε_k,own (the mixed-noise idea of PYoCo, Ge et al. 2023). Flicker = mean |frame_{k+1} −
  frame_k moved 3 px| on the 28×52 canvas, pixels in [0, 1]: 0.078 (ρ = 0) → 0.054 (0.5) → 0.024 (0.9) → 0.007 (0.99)
  → 0 (1); frame-to-frame change (L2, 28×28 digits) 9.2 / 7.2 / 3.9 / 1.5 / 0. The MOTION IS PASTED ON (frame k is
  pasted at x = 3k by the script; the model is never asked to move anything) — said on screen. Not a video model.
  Honest limit: at ρ = 1 every frame is the same digit (flicker 0 bought by giving up any change).
- COMPUTED · AN IDEALISED JOINT MODEL (the swirl, exact maths, verified below with diffusion_reference.exact_denoiser):
  a toy video whose frame k is one swirl point turned by k·45°. Rotations are orthogonal, so turning every noisy frame
  back leaves standard noise; the exact joint denoiser E[x0 | all 8 frames] is the one-frame denoiser applied to the
  average of the turned-back frames, whose noise is √8 smaller. Brute-force Gaussian conditioning on all 16 numbers
  agrees to 1e-15. DDIM 50 from the same per-frame noise, 200 clips: each frame on its own lands on a different swirl
  point (mean jump between frames, turned back: 0.88); together, every frame is the same point (jump exactly 0).
- Temporal attention (Episode 21 drew it), frame interpolation, long videos (sliding windows, drift), world models:
  one card each, dated facts from research/notes.md §22 (read October 2026).

Every number on screen is computed below and asserted, or cited with its source. Illustrative ("for example"): the
starting-noise tiles (the run's recipe, our own random numbers: the run did not save its noise), the noise-to-digit
blend in section 1 (the real DDIM path was not saved: only start and end are shown), the ball in the interpolation
card, the timeline in the long-video card, the game frame in the world-model card.

    VOICE=draft manimgx inspect scene.py      # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403
from diffusion_reference import *  # noqa: E402,F403

EP_DIR = Path(__file__).resolve().parent  # set AFTER the import: diffusion_reference exports HERE (the series folder)
warnings.filterwarnings("ignore", message=".*encountered in matmul")  # NumPy/Accelerate's spurious warnings

ARR = EXPERIMENTS / "arrays"
MD = (EXPERIMENTS / "mac_runs.md").read_text()
NOTES = (SERIES / "research" / "notes.md").read_text()

# ---------------------------------------------------------------------------- 1–4 · the real run (mac_runs.md §6)

VID = np.load(ARR / "digit_video_noise_correlation.npz")
RHOS = ("0", "0.5", "0.9", "0.99", "1")
NF, PX, CLIP = 8, 3, 7                     # frames per clip, pixels pasted per frame, the clip on screen (asks for a 7)
RUN = measured("mac_runs", "video_toy_noise_correlation")
assert [float(r) for r in RHOS] == list(VID["rhos"])
assert RUN["frames"] == NF and RUN["canvas"] == [28, 52] and RUN["px_per_frame"] == PX and RUN["n_clips"] == 20
assert RUN["sampler"] == "DDIM 50, w=1" and "PYoCo, Ge et al. 2023" in RUN["note"] and "EMA" in RUN["model"]
assert int(VID["labels"][CLIP]) == 7
FRAMES = {r: VID[f"rho{r}_frames"] for r in RHOS}   # [clip, frame, 28, 28] uint8
CANVAS = {r: VID[f"rho{r}_canvas"] for r in RHOS}   # [clip, frame, 28, 52] uint8
for _r in RHOS:
    assert FRAMES[_r].shape == (20, NF, 28, 28) and CANVAS[_r].shape == (20, NF, 28, 52)
    for _k in range(NF):                            # the motion is pasted on: frame k sits at x = 3k, nothing else
        assert np.array_equal(CANVAS[_r][:, _k, :, PX * _k: PX * _k + 28], FRAMES[_r][:, _k])
        _rest = CANVAS[_r][:, _k].copy()
        _rest[:, :, PX * _k: PX * _k + 28] = 0
        assert _rest.max() == 0


def flicker_of(canvas):
    """mean |frame_{k+1} − frame_k moved 3 px| on the canvas, pixels in [0, 1] (mac_runs.md §6)."""
    c = canvas.astype(float) / 255
    moved = np.zeros_like(c)
    moved[..., PX:] = c[..., :-PX]
    return float(np.abs(c[:, 1:] - moved[:, :-1]).mean())


def change_of(frames):
    """mean L2 distance between consecutive 28×28 digits, pixels in [0, 1]."""
    f = frames.astype(float) / 255
    return float(np.sqrt(((f[:, 1:] - f[:, :-1]) ** 2).sum((-1, -2))).mean())


FLICK = {r: flicker_of(CANVAS[r]) for r in RHOS}
CHANGE = {r: change_of(FRAMES[r]) for r in RHOS}
for _row, _r in zip(RUN["rows"], RHOS):
    assert _row["rho"] == float(_r) and abs(_row["flicker"] - FLICK[_r]) < 1e-4 and abs(_row["change_px"] - CHANGE[_r]) < 0.01
    assert abs(measured("mac_runs", "headline", f"video_flicker_rho{_r}") - FLICK[_r]) < 1e-3
F_TXT = {r: ("0" if FLICK[r] == 0 else f"{FLICK[r]:.3f}") for r in RHOS}
C_TXT = {r: ("0" if CHANGE[r] == 0 else f"{CHANGE[r]:.1f}") for r in RHOS}
assert list(F_TXT.values()) == ["0.078", "0.054", "0.024", "0.007", "0"], F_TXT
assert list(C_TXT.values()) == ["9.2", "7.2", "3.9", "1.5", "0"], C_TXT
assert "| flicker | **0.078** | 0.054 | 0.024 | 0.007 | **0** |" in MD
assert "| frame-to-frame change (L2) | 9.2 | 7.2 | 3.9 | 1.5 | 0 |" in MD
assert "**Motion is imposed, not generated:**" in MD and "At ρ = 0.9 there is an occasional jump; at 0.99 the\ndigit is steady" in MD
assert ('On screen, say: "the cheapest consistency trick, correlated starting noise, removes the flicker; a real video model\n'
        'learns consistency by denoising all frames together."') in MD
assert np.all(FRAMES["1"][:, 1:] == FRAMES["1"][:, :1])                  # ρ = 1: every frame is the very same digit
_f9 = FRAMES["0.9"][CLIP].astype(float) / 255
_steps9 = np.sqrt(((_f9[1:] - _f9[:-1]) ** 2).sum((-1, -2)))
JUMP_K = int(np.argmax(_steps9[:-1] + _steps9[1:])) + 1                    # the frame that jumps at ρ = 0.9 (both its edges)
assert JUMP_K == 6 and min(_steps9[5], _steps9[6]) > 2 * max(_steps9[:4]), _steps9
CLIP_FLICK = {r: flicker_of(CANVAS[r][CLIP:CLIP + 1]) for r in RHOS}      # the clip on screen follows the same order
assert all(CLIP_FLICK[a] > CLIP_FLICK[b] for a, b in zip(RHOS, RHOS[1:]) if b != "1") and CLIP_FLICK["1"] == 0

# the starting-noise recipe, drawn with our own random numbers for the pictures (the run did not save its noise)
_nr = np.random.default_rng(2022)
EPS_SHARED = _nr.standard_normal((28, 28))
EPS_OWN = _nr.standard_normal((NF, 28, 28))


def start_noise(rho):
    """ε_k = √ρ·ε_shared + √(1 − ρ)·ε_k,own: unit variance for any ρ, correlation ρ between any two frames."""
    return np.sqrt(rho) * EPS_SHARED[None] + np.sqrt(1 - rho) * EPS_OWN


_mix = start_noise(0.5).reshape(NF, -1)
assert abs(np.corrcoef(_mix)[np.triu_indices(NF, 1)].mean() - 0.5) < 0.06 and abs(_mix.std() - 1) < 0.03

# ---------------------------------------------------------------------------- 5 · an idealised joint model (the swirl)
# A toy video: frame k is ONE swirl point x0 turned by k·45°. Noisy frame k: y_k = a·R_k·x0 + b·ε_k. Rotations are
# orthogonal, so R_kᵀ·y_k = a·x0 + b·(R_kᵀ ε_k) with R_kᵀ ε_k still standard noise: the likelihood of all 8 frames depends
# on x0 only through their turned-back average z̄ = a·x0 + (b/√8)·ε̄. So E[x0 | 8 frames] = the one-frame denoiser at the
# smaller noise b/√8, applied to z̄. Checked here three ways: brute force on all 16 numbers, mixture_posterior(z̄, a, b/√8),
# and diffusion_reference.exact_denoiser on z̄ rescaled to the step with that noise level.

SW_F, SW_THETA = 8, np.pi / 4


def rot(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


ROTS = np.array([rot(k * SW_THETA) for k in range(SW_F)])


def _exactness_check(t=400, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.sqrt(alpha_bar_at(t)), np.sqrt(1 - alpha_bar_at(t))
    x0 = swirl(1, seed=5)[0]
    ys = np.array([a * R @ x0 + b * rng.standard_normal(2) for R in ROTS])
    stacked = np.concatenate(ROTS, 0)                                       # (16, 2): all frames as one observation
    s2 = SWIRL.stds[0] ** 2
    cov = a * a * s2 * stacked @ stacked.T + b * b * np.eye(2 * SW_F)
    inv = np.linalg.inv(cov)
    y = ys.reshape(-1)
    logits, means = [], []
    for mu, w in zip(SWIRL.means, SWIRL.weights):
        r = y - a * stacked @ mu
        logits.append(np.log(w) - 0.5 * r @ inv @ r)
        means.append(mu + a * s2 * stacked.T @ inv @ r)
    p = np.exp(np.array(logits) - max(logits))
    brute = (p / p.sum()) @ np.array(means)
    zbar = np.mean([R.T @ yk for R, yk in zip(ROTS, ys)], 0)
    pooled = mixture_posterior(zbar[None], a, b / np.sqrt(SW_F)).x0[0]
    scale = np.sqrt(a * a + b * b / SW_F)
    via_exact = exact_denoiser((zbar / scale)[None], timestep_from_sigma(b / np.sqrt(SW_F) / a))[0]
    return max(np.abs(brute - pooled).max(), np.abs(brute - via_exact).max())


assert _exactness_check(400) < 1e-9 and _exactness_check(800, seed=1) < 1e-9 and _exactness_check(150, seed=2) < 1e-9
SQRT_F = f"{np.sqrt(SW_F):.2f}"
assert SQRT_F == "2.83"

SW_N, SW_STEPS = 200, 50
SW_EPS = np.random.default_rng(22).standard_normal((SW_N, SW_F, 2))      # frame k's starting noise, in its own turned frame
SW_Z = np.einsum("fji,nfj->nfi", ROTS, SW_EPS)                            # turned back: still standard noise
_ind = ddim_sample(exact_eps, SW_Z.reshape(-1, 2), steps=SW_STEPS)        # each frame on its own (exact one-frame model)
IND_PATH = np.concatenate([_ind.x0s, _ind.xs[-1:]]).reshape(SW_STEPS + 1, SW_N, SW_F, 2)


def joint_ddim(z, steps=SW_STEPS):
    """DDIM with the exact JOINT denoiser: every frame's guess is E[x0 | all frames] (pooled at noise √((1−ᾱ)/F))."""
    ts = ddim_timesteps(steps)
    guesses = []
    for t, s in zip(ts[:-1], ts[1:]):
        ab_t, ab_s = alpha_bar_at(t), alpha_bar_at(s)
        x0 = mixture_posterior(z.mean(1), np.sqrt(ab_t), np.sqrt((1 - ab_t) / z.shape[1])).x0
        eps = (z - np.sqrt(ab_t) * x0[:, None]) / np.sqrt(1 - ab_t)
        z = np.sqrt(ab_s) * x0[:, None] + np.sqrt(1 - ab_s) * eps
        guesses.append(np.repeat(x0[:, None], z.shape[1], 1))
    return np.array(guesses + [z])


JNT_PATH = joint_ddim(SW_Z)                                               # (51, N, F, 2), turned-back coordinates
IND_END, JNT_END = IND_PATH[-1], JNT_PATH[-1]
_jump = lambda q: float(np.linalg.norm(q[:, 1:] - q[:, :-1], axis=-1).mean())  # noqa: E731
JUMP_IND, JUMP_JNT = _jump(IND_END), _jump(JNT_END)
assert f"{JUMP_IND:.2f}" == "0.88" and JUMP_JNT < 1e-12, (JUMP_IND, JUMP_JNT)
assert swirl_distance(IND_END.reshape(-1, 2)) < 0.05 and swirl_distance(JNT_END[:, 0]) < 0.05   # all land on the swirl
_r_jnt = np.linalg.norm(JNT_END[:, 0], axis=1)
_spread = np.linalg.norm(IND_END[:, 1:] - IND_END[:, :-1], axis=-1).mean(1)
_cand = np.flatnonzero(_r_jnt > 0.6)
SW_CLIP = int(_cand[np.argmax(_spread[_cand])])                           # the clip on screen: a far-out point, a wild row


def framed_path(path, n=SW_CLIP):
    """One clip's path in each frame's OWN (turned) coordinates: frame k shows R_k · (turned-back point)."""
    return np.einsum("fij,sfj->sfi", ROTS, path[:, n])


IND_SHOW, JNT_SHOW = framed_path(IND_PATH), framed_path(JNT_PATH)

# ---------------------------------------------------------------------------- 6–9 · cited cards (notes §22, Appendix C)

for _q in ("Temporal layers are often initialised as identity so a pretrained image model is unchanged at the start",
           "Video Diffusion Models (Ho et al.) turned 3×3 convolutions into 1×3×3 (space-only) and added temporal attention",
           "HunyuanVideo: Full Attention \"has demonstrated superior performance compared to divided spatiotemporal attention\"",
           "Imagen Video TSR models \"increase temporal resolution by filling in intermediate frames between input frames\"",
           "Make-A-Video's ↑F turns 16 frames into 76 (\"(16-1)×5+1\")",
           "\"allows us to forego frame-interpolation models commonly used in prior work\"",
           "\"a diffusion model is trained to denoise a set of tokens with independent per-token noise levels\"",
           "frame k gets a local time t_k ≈ clip((k + t_global)/W) so later frames are noisier",
           "\"real-time video generation at 17 FPS with sub-second latency on a single H100 GPU\"",
           "Veo 3.1 Extend conditions on \"the final second of the previous clip\"",
           "\"runs at 20 frames per second on a single TPU\"", "conditions on the last 64 frames and actions; 4 DDIM steps",
           "\"navigate in real time at 24 frames per second, retaining consistency for a few minutes at a resolution of 720p\"",
           "Genie 3 (DeepMind blog, 5 Aug 2025)", "**Not diffusion**: video tokenizer + MaskGIT dynamics",
           "Errors accumulate (\"drift\")", "without noise augmentation quality \"degrades quickly after 10-20 frames\""):
    assert _q in NOTES, _q
MAV_FRAMES = (16 - 1) * 5 + 1
assert MAV_FRAMES == 76

# ---------------------------------------------------------------------------- look

FLAT = set("abcdefhiklmnorstuvwxzABCDEFGHIJKLMNOPRSTUVWXYZ0123456789")


def cite(s, size=20, color=MUTED):
    return text(s, size, color)


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


def digit_rgb(img, color=DATA):
    return tint(img, color, 0, 255)


def noise_rgb(eps):
    return tint(eps, NOISE, -2.5, 2.5)


def texture(rgb):
    """The RGBA texture picture() makes from a small image (crisp pixels), for swapping in place."""
    k = max(1, int(np.ceil(540 / max(rgb.shape[:2]))))
    arr = np.repeat(np.repeat(rgb, k, axis=0), k, axis=1)
    rgba = np.concatenate([arr, np.full(arr.shape[:2] + (1,), 255, np.uint8)], axis=-1)
    rgba.flags.writeable = False
    return rgba


class Flip:
    """A picture whose pixels are swapped for frame k of a sequence (only when k changes)."""

    def __init__(self, rgb_of, k, height):
        self.rgb_of, self.k = rgb_of, int(k)
        self.image = picture(rgb_of(self.k), height)

    def show(self, k):
        k = int(k)
        if k != self.k:
            self.k = k
            self.image.paint = self.image.paint.but(texture=texture(self.rgb_of(k)))


class Readout(m.VGroup):
    """A mono text display, left-aligned at a fixed point; .show(v) updates it."""

    def __init__(self, left, y, fmt, value, size=40, color=INK, bold=True):
        super().__init__()
        self.anchor, self.fmt, self.size, self.color, self.bold = np.array([left, y, 0.0]), fmt, size, color, bold
        self.value = value
        self.words = self._make(value)
        self.add(self.words)

    def _make(self, value):
        words = code(self.fmt.format(value), self.size, self.color, weight=m.BOLD if self.bold else m.NORMAL)
        return words.move_to(self.anchor, aligned_edge=m.LEFT)

    def show(self, value):
        if value != self.value:
            self.value = value
            self.words.become(self._make(value))
        return self


def play(flip, counter=None, loops=2, seconds=2.4, frames=NF):
    """Play a clip: swap the picture through its frames `loops` times (and update a 'frame k / 8' counter)."""
    total = frames * loops
    mob = m.Group(flip.image, counter) if counter is not None else flip.image

    def step(_, a):
        i = min(int(a * total), total - 1) % frames
        flip.show(i)
        if counter is not None:
            counter.show(f"{i + 1} / {frames}")

    return m.UpdateFromAlphaFunc(mob, step, run_time=seconds, rate_func=m.linear)


def digit_row(images, color, height, gap_px=4, k=8):
    """Several 28×28 digits (uint8) as ONE picture with dark gaps, `height` units tall."""
    n = len(images)
    canvas = np.zeros((28, n * 28 + (n - 1) * gap_px))
    for i, img in enumerate(images):
        canvas[:, i * (28 + gap_px): i * (28 + gap_px) + 28] = np.asarray(img, dtype=float) / 255
    rgb = np.repeat(np.repeat(tint(canvas, color, 0, 1), k, axis=0), k, axis=1)
    return picture(rgb, height, pixelated=False)


def row_x(pic, i, gap_px=4):
    """Scene x of digit i's centre inside a digit_row picture."""
    scale = pic.height / 28
    return pic.get_left()[0] + (i * (28 + gap_px) + 14) * scale


def fact_rows(rows, width, size=20, stacked=True):
    """Dated facts: (name, source, line) → name (bold) and source (MUTED) on one baseline, the line under it
    (stacked) or beside it. Returns a VGroup of rows, left-aligned."""
    out = m.VGroup()
    for name, source, line in rows:
        head = m.VGroup(text(name, size + 2, INK, weight=m.BOLD), cite(source, size)).arrange(m.RIGHT, buff=0.2)
        sit(head[1], baseline(head[0]))
        body = text(line, size, MUTED)
        if stacked:
            out.add(m.VGroup(head, body).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT))
        else:
            out.add(m.VGroup(head, body))
    if stacked:
        out.arrange(m.DOWN, buff=0.26, aligned_edge=m.LEFT)
    else:                                   # two columns: names and sources, then the facts, one baseline per row
        heads = m.VGroup(*[r_[0] for r_ in out]).arrange(m.DOWN, buff=0.24, aligned_edge=m.LEFT)
        col = heads.get_left()[0] + max(h_.width for h_ in heads) + 0.45
        for h_, b_ in out:
            b_.move_to([col + b_.width / 2, h_.get_center()[1], 0])
            sit(b_, baseline(h_[0]))
    assert out.width < width - 0.5, (out.width, width)
    return out


def boxed(group, width, color=EDGE, pad=0.3):
    box = card(width, group.height + 2 * pad, color, CARD, radius=0.16, stroke=2).move_to(group)
    group.align_to(box, m.LEFT).shift(0.35 * m.RIGHT)
    return box


def mini_plane(center, side):
    return Plot((-1.15, 1.15), (-1.15, 1.15), width=side, height=side, center=(center[0], center[1], 0), axes=False)


def swirl_band(plot, color=DATA, width=4, opacity=0.3):
    band = m.VGroup()
    for arm in swirl_curve(120):
        band.add(m.VMobject().set_points_smoothly([plot.coords(*p) for p in arm[::3]] + [plot.coords(*arm[-1])])
                 .set_stroke(color, width, opacity=opacity))
    return band


def replay_dots(dots, plots, path, seconds=2.4):
    """Walk one dot per frame along `path` (states × F × 2, each frame's own coordinates), VIOLET → TEAL."""

    def step(_, a):
        i = a * (len(path) - 1)
        j = int(np.floor(i))
        k = min(j + 1, len(path) - 1)
        p = path[j] + (path[k] - path[j]) * (i - j)
        col = mix(NOISE, DATA, a)
        for d, pl, q in zip(dots, plots, p):
            d.move_to(pl.coords(*q))
            d.set_color(col)

    return m.UpdateFromAlphaFunc(dots, step, run_time=seconds, rate_func=m.smooth)


# ---------------------------------------------------------------------------- the episode


class Ep22KeepingFramesTogether(Episode):
    number = 22

    def lesson(self):
        self.hook()
        self.alone()
        self.share()
        self.measured_flicker()
        self.frozen()
        self.together()
        self.attention()
        self.interpolation()
        self.long_videos()
        self.world_models()
        self.recap()

    # ------------------------------------------------------------------ 0 · hook

    def hook(self):
        self.next_section("Hook", notes=(
            "Generated on this Mac (mac_runs.md §6): the digit model (the DDPM-trained U-Net, EMA weights), DDIM 50 steps, "
            f"w = 1, clip {CLIP} of 20 (it asks for a 7). Clip A: every frame from its own noise (ρ = 0); clip B: the frames "
            "share 99% of their starting noise (ρ = 0.99). The slide is pasted on by the script: frame k sits 3 px further "
            "right on a 28×52 canvas. Episode 21 (From Pictures to Video) built the video cube and drew temporal attention."))
        players, labels, counters = [], m.VGroup(), m.VGroup()
        for r, x, name in (("0", -3.3, "clip A"), ("0.99", 3.3, "clip B")):
            flip = Flip(lambda k, r=r: digit_rgb(CANVAS[r][CLIP, k]), 0, 2.3)
            box = framed(flip.image, EDGE).move_to([x, 0.08, 0])
            lab = code(name, 24, INK, weight=m.BOLD).next_to(box, m.UP, buff=0.18).align_to(box, m.LEFT)
            ctr = Readout(box.get_right()[0] - code("8 / 8", 22).width, 0, "{}", "1 / 8", 22, TIME, bold=False)
            sit(ctr.words, baseline(lab))
            ctr.anchor[1] = ctr.words.get_center()[1]
            players.append((flip, box))
            labels.add(lab)
            counters.add(ctr)
        fr_l = code("frame", 22, TIME).next_to(counters[0], m.LEFT, buff=0.15)
        fr_r = code("frame", 22, TIME).next_to(counters[1], m.LEFT, buff=0.15)
        sit(fr_l, baseline(labels[0]))
        sit(fr_r, baseline(labels[1]))
        bad = pill("flickers", RED, RED_FILL, 22).next_to(players[0][1], m.DOWN, buff=0.25)
        good = pill("holds steady", GREEN, GREEN_FILL, 22).next_to(players[1][1], m.DOWN, buff=0.25)
        cap = cite("generated on this Mac · the digit model, 50 DDIM steps · the slide is pasted on: 3 px per frame")
        cap.move_to([0, -2.3, 0])
        claim = title_text("Consistency across frames is the hard part of video.", 31)
        ep21 = pill("Episode 21 · a video is a cube of frames", BLUE, BLUE_FILL, 20).move_to([0, 2.25, 0])
        mascot = pixel(0.5)
        foot = m.VGroup(claim, mascot).arrange(m.RIGHT, buff=0.35).move_to([0, -3.12, 0])
        assert foot.width < 12.6 and ep21.get_bottom()[1] > labels.get_top()[1] + 0.12, (foot.width, ep21.get_bottom())
        assert cap.width < 13.2 and bad.get_bottom()[1] > cap.get_top()[1] + 0.08, (cap.width, bad.get_bottom())
        assert players[0][1].get_left()[0] > -6.6 and players[1][1].get_right()[0] < 6.6

        (fa, ba), (fb, bb) = players
        self.say(
            "[Two tiny clips of a sliding seven], [both generated on this Mac by our digit model].",
            m.AnimationGroup(self.new_header("00 · hook", "Two clips, one model"), m.FadeIn(ba), m.FadeIn(bb), m.FadeIn(labels)),
            m.AnimationGroup(m.FadeIn(cap), m.FadeIn(counters), m.FadeIn(fr_l), m.FadeIn(fr_r)),
        )
        self.say(
            "[On the left, the seven changes shape every frame]: [it flickers]. [On the right, it holds steady].",
            play(fa, counters[0], loops=2, seconds=2.6),
            m.FadeIn(bad, shift=0.1 * m.UP),
            m.AnimationGroup(play(fb, counters[1], loops=2, seconds=2.6), m.FadeIn(good, shift=0.1 * m.UP)),
        )
        self.say(
            "[Episode twenty-one stacked frames into one cube]. [Today, the hard part]: [keeping those frames together].",
            m.FadeIn(ep21, shift=0.1 * m.DOWN),
            m.AnimationGroup(m.Write(claim), m.FadeIn(mascot, shift=0.2 * m.UP)),
            m.AnimationGroup(play(fa, counters[0], loops=2, seconds=2.6), play(fb, counters[1], loops=2, seconds=2.6)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 1 · frames drawn alone

    def alone(self):
        self.next_section("1 · Frames drawn alone", notes=(
            "How clip A was made (mac_runs.md §6, ρ = 0): eight frames, each from its own starting noise, each denoised by "
            "the digit model on its own (DDIM 50, w = 1, label 7). The VIOLET noise tiles are drawn for illustration with "
            "our own random numbers (the run did not save its noise), and the noise-to-digit blend stands in for the 50 "
            "real steps (only the end is the run's). The digits are the run's frames, clip 7. Motion is imposed: the "
            "script pastes frame k at x = 3k on a 28×52 canvas (verified in the arrays); the model only ever makes 28×28 "
            "digits. Not a video model."))
        h, gap = 0.95, 0.2
        xs = [-(NF - 1) / 2 * (h + gap) + i * (h + gap) for i in range(NF)]
        y_noise, y_model, y_digit, y_num = 1.7, 0.55, -0.6, -1.4
        noise = start_noise(0.0)
        tiles = m.Group(*[picture(noise_rgb(noise[k]), h).move_to([x, y_noise, 0]) for k, x in enumerate(xs)])
        flips = [Flip(lambda s, k=k: (((1 - s / 24) * noise_rgb(noise[k]).astype(float)
                                       + s / 24 * digit_rgb(FRAMES["0"][CLIP, k]).astype(float)).round().astype(np.uint8)),
                      0, h) for k in range(NF)]
        digits = m.Group(*[f.image.move_to([x, y_digit, 0]) for f, x in zip(flips, xs)])
        band = card(xs[-1] - xs[0] + h + 0.3, 0.56, MODEL, AMBER_FILL, radius=0.14, stroke=2.5).move_to([0, y_model, 0])
        band_t = text("the digit model · 50 steps · asked for a 7 · one frame at a time", 22, INK).move_to(band)
        arr_in = m.VGroup(*[arrow(tiles[k].get_bottom(), [x, band.get_top()[1], 0], NOISE, buff=0.06) for k, x in enumerate(xs)])
        arr_out = m.VGroup(*[arrow([x, band.get_bottom()[1], 0], digits[k].get_top(), DATA, buff=0.06) for k, x in enumerate(xs)])
        row_l = m.VGroup(code("noise", 22, NOISE), code("model", 22, MODEL), code("frames", 22, DATA))
        for lab, y in zip(row_l, (y_noise, y_model, y_digit)):
            lab.move_to([xs[0] - h / 2 - 0.3 - lab.width / 2, y, 0])
        nums = m.VGroup(*[code(str(k + 1), 22, TIME, weight=m.BOLD).move_to([x, y_num, 0]) for k, x in enumerate(xs)])
        neq = m.VGroup(*[code("≠", 26, RED, weight=m.BOLD).move_to([(a + b) / 2, y_num, 0]) for a, b in zip(xs, xs[1:])])
        t_arrow = m.Arrow([xs[0] - h / 2, y_num - 0.38, 0], [xs[-1] + h / 2 + 0.2, y_num - 0.38, 0], buff=0, color=TIME,
                          stroke_width=3, max_tip_length_to_length_ratio=0.03)
        t_lab = code("time", 20, TIME).next_to(t_arrow, m.RIGHT, buff=0.15)
        flick = pill("flicker", RED, RED_FILL, 22).next_to(t_lab, m.UP, buff=0.3)
        noise_note = cite("noise drawn\nfor illustration").next_to(tiles, m.RIGHT, buff=0.2)
        assert row_l.get_left()[0] > -6.6 and t_lab.get_right()[0] < 6.6 and noise_note.get_right()[0] < 6.6, (row_l.get_left(), t_lab.get_right(), noise_note.get_right())
        assert band_t.width < band.width - 0.3 and tiles.get_top()[1] < 2.6

        # the honest note: the slide is pasted on
        honest = card(12.6, 1.2, MUTED, CARD, radius=0.16, stroke=2).move_to([0, -2.85, 0])
        pflip = Flip(lambda k: digit_rgb(CANVAS["0"][CLIP, k]), 0, 0.8)
        pbox = framed(pflip.image, EDGE).move_to([honest.get_left()[0] + 0.35 + 0.8 * 52 / 28 / 2 + 0.06, -2.85, 0])
        h1 = text("The motion is pasted on, not generated.", 26, INK, weight=m.BOLD)
        h2 = text("We put frame k at x = 3k on a 28×52 canvas; the model only makes 28×28 digits.", 20, MUTED)
        hs = m.VGroup(h1, h2).arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT).next_to(pbox, m.RIGHT, buff=0.4)
        assert hs.get_right()[0] < honest.get_right()[0] - 0.2 and honest.get_top()[1] < t_arrow.get_bottom()[1] - 0.12, (hs.get_right(), honest.get_top())

        self.say(
            "[Here is how clip A was made]: [eight frames, each starting from its own random noise].",
            m.AnimationGroup(self.new_header("01 · one at a time", "Frames drawn alone"), m.FadeIn(row_l[0])),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(t_, shift=0.1 * m.DOWN) for t_ in tiles], lag_ratio=0.15, run_time=1.4),
                             m.FadeIn(noise_note)),
        )

        def emerge(_, a):
            for f in flips:
                f.show(round(24 * a))

        self.say(
            "[Our digit model denoises every frame on its own], [fifty steps each, always asking for a seven].",
            m.AnimationGroup(m.FadeIn(band), m.FadeIn(band_t), m.FadeIn(row_l[1]), m.LaggedStart(*[m.GrowArrow(a_) for a_ in arr_in], lag_ratio=0.08)),
            m.Succession(m.AnimationGroup(m.FadeIn(digits, target_position=tiles), m.LaggedStart(*[m.GrowArrow(a_) for a_ in arr_out], lag_ratio=0.08),
                                          m.FadeIn(row_l[2]), run_time=0.9),
                         m.UpdateFromAlphaFunc(digits, emerge, run_time=2.0, rate_func=m.smooth)),
        )
        self.say(
            "[Every frame is a fine seven]. [But each one is a different seven], [so the clip flickers].",
            m.AnimationGroup(m.Circumscribe(digits, color=DATA, buff=0.1), m.FadeIn(nums), m.GrowArrow(t_arrow), m.FadeIn(t_lab)),
            m.LaggedStart(*[m.FadeIn(n_, scale=0.6) for n_ in neq], lag_ratio=0.15, run_time=1.3),
            m.FadeIn(flick, shift=0.1 * m.UP),
        )
        self.say(
            "[One honest note]: [the motion is pasted on]. [We slide each frame three pixels to the right ourselves]; "
            "[the model never moves anything].",
            m.FadeIn(honest),
            m.AnimationGroup(m.FadeIn(pbox), m.FadeIn(h1, shift=0.1 * m.UP)),
            m.AnimationGroup(play(pflip, None, loops=1, seconds=2.0), m.FadeIn(h2)),
            m.Circumscribe(band, color=MODEL, buff=0.06),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 2 · share the noise

    def share(self):
        self.next_section("2 · Correlated starting noise", notes=(
            "The run's recipe (mac_runs.md §6): ε_k = √ρ·ε_shared + √(1−ρ)·ε_k,own — every frame's noise still has unit "
            "variance, and any two frames' noises have correlation ρ. This is the 'mixed noise' idea of PYoCo (Ge et al., "
            "'Preserve Your Own Correlation', 2023), which proposed correlated noise priors for video diffusion. The tiles "
            "are drawn here with our own random numbers (seed 2022), with the same recipe; the run's own noise was not "
            "saved. The slider sweeps 0 → 0.5 → 0.99 → 1."))
        tex_parts = [r"\varepsilon_k", "=", r"\sqrt{\rho}", r"\,\varepsilon_{\mathrm{shared}}", "+", r"\sqrt{1-\rho}",
                     r"\,\varepsilon_{k,\mathrm{own}}"]
        f = m.MathTex(*tex_parts, font_size=44, color=INK)
        for i in (0, 3, 6):
            f[i].set_color(NOISE)
        f.move_to([0, 1.95, 0])
        sh_h, h, gap = 1.35, 1.0, 0.17
        shared = picture(noise_rgb(EPS_SHARED), sh_h).move_to([-5.3, -0.05, 0])
        shared_f = m.SurroundingRectangle(shared, buff=0.06).set_stroke(NOISE, 2.5)
        shared_l = code("shared", 22, NOISE, weight=m.BOLD).next_to(shared_f, m.UP, buff=0.15)
        xs = [1.55 - (NF - 1) / 2 * (h + gap) + i * (h + gap) for i in range(NF)]
        state = {"rho": 0.0}
        flips = [Flip(lambda s, k=k: noise_rgb(start_noise(s / 1000)[k]), 0, h) for k in range(NF)]
        tiles = m.Group(*[fl.image.move_to([x, -0.05, 0]) for fl, x in zip(flips, xs)])
        row_l = code("each frame's starting noise", 22, MUTED).next_to(tiles, m.UP, buff=0.18).align_to(tiles, m.LEFT)
        nums = m.VGroup(*[code(str(k + 1), 20, TIME).next_to(t_, m.DOWN, buff=0.12) for k, t_ in enumerate(tiles)])
        link = arrow(shared_f.get_right(), [tiles.get_left()[0], -0.05, 0], NOISE, buff=0.12)
        link_l = cite("+ its own").next_to(link, m.DOWN, buff=0.12)
        y_s = -1.75
        r_lab = code("correlation ρ =", 24, MUTED)
        r_val = Readout(0, 0, "{:.2f}", 0.0, 30, INK)
        r_lab.move_to([0, y_s, 0]).align_to([-6.3, 0, 0], m.LEFT)
        r_val.anchor = np.array([r_lab.get_right()[0] + 0.2, y_s, 0.0])
        r_val.words.move_to(r_val.anchor, aligned_edge=m.LEFT)
        sit(r_lab, baseline(r_val.words))
        x0 = r_val.anchor[0] + code("0.00", 30).width + 0.7
        x1 = x0 + 6.0
        track = m.Line([x0, y_s, 0], [x1, y_s, 0]).set_stroke(EDGE, 8)
        knob = m.Circle(radius=0.14).set_stroke(INK, 3).set_fill(BG, 1).move_to([x0, y_s, 0])
        fill = m.Line([x0, y_s, 0], [x0 + 0.001, y_s, 0]).set_stroke(NOISE, 8)
        ends = m.VGroup(m.VGroup(code("0", 20, INK, weight=m.BOLD), cite("independent")).arrange(m.RIGHT, buff=0.15),
                        m.VGroup(code("1", 20, INK, weight=m.BOLD), cite("one shared noise")).arrange(m.RIGHT, buff=0.15))
        for e, x in zip(ends, (x0, x1)):
            sit(e[1], baseline(e[0]))
            e.next_to([x, y_s - 0.25, 0], m.DOWN, buff=0.1)
        ends[0].align_to([x0 - 0.1, 0, 0], m.LEFT)
        ends[1].align_to([x1 + 0.1, 0, 0], m.RIGHT)
        src = cite("the mixed-noise idea of PYoCo (Ge et al., 2023) · noise drawn here for illustration, with the same recipe")
        src.move_to([0, -3.3, 0])
        assert tiles.get_right()[0] < 6.6 and shared_f.get_left()[0] > -6.6 and r_lab.get_left()[0] > -6.6, (tiles.get_right(), shared_f.get_left(), r_lab.get_left())
        assert link.get_length() > 0.5 and ends.get_bottom()[1] > src.get_top()[1] + 0.1 and row_l.get_top()[1] < f.get_bottom()[1] - 0.1, (link.get_length(), ends.get_bottom(), row_l.get_top())
        assert x1 < 5.2, x1

        def sweep(r0, r1, seconds=1.6):
            def step(_, a):
                rho = r0 + (r1 - r0) * a
                state["rho"] = rho
                for fl in flips:
                    fl.show(round(1000 * rho))
                x = x0 + (x1 - x0) * rho
                knob.move_to([x, y_s, 0])
                fill.become(m.Line([x0, y_s, 0], [max(x, x0 + 0.001), y_s, 0]).set_stroke(NOISE, 8))
                r_val.show(round(rho, 2))
            return m.UpdateFromAlphaFunc(m.Group(tiles, knob, fill, r_val), step, run_time=seconds, rate_func=m.smooth)

        self.say(
            "[The cheapest fix: let the frames share some of their starting noise].",
            m.AnimationGroup(self.new_header("02 · share the noise", "Correlated starting noise"), m.FadeIn(shared), m.Create(shared_f),
                             m.FadeIn(shared_l), m.Write(f, run_time=2.0)),
        )
        self.say(
            "[Each frame mixes one shared noise with its own], [and the correlation sets the mix].",
            m.AnimationGroup(m.GrowArrow(link), m.FadeIn(link_l), m.FadeIn(row_l),
                             m.LaggedStart(*[m.FadeIn(t_, shift=0.1 * m.RIGHT) for t_ in tiles], lag_ratio=0.12, run_time=1.4), m.FadeIn(nums)),
            m.AnimationGroup(m.FadeIn(track), m.FadeIn(fill), m.FadeIn(knob), m.FadeIn(ends), m.FadeIn(r_lab), m.FadeIn(r_val),
                             m.Circumscribe(m.VGroup(f[2], f[5]), color=INK, buff=0.08), m.FadeIn(src)),
        )
        self.say(
            "[At zero, the frames start from unrelated noise]. [Turn the correlation up], [and the starting noises grow alike], "
            "[until, at one, they are all the same].",
            m.Circumscribe(tiles, color=RED, buff=0.08),
            sweep(0.0, 0.5, 1.4),
            sweep(0.5, 0.99, 1.8),
            m.AnimationGroup(sweep(0.99, 1.0, 0.8), m.Circumscribe(tiles, color=GREEN, buff=0.08, run_time=1.4)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 3 · flicker, measured

    def measured_flicker(self):
        self.next_section("3 · Flicker, measured", notes=(
            "Measured on this Mac (mac_runs.md §6; recomputed here from arrays/digit_video_noise_correlation.npz and "
            "asserted against mac_runs.json): flicker = mean |frame_{k+1} − frame_k moved 3 px| on the 28×52 canvas, "
            "pixels in [0, 1], averaged over all 20 clips × 7 frame pairs: "
            + ", ".join(f"ρ = {r}: {F_TXT[r]}" for r in RHOS) + ". Frame-to-frame change (L2 distance between consecutive "
            "28×28 digits): " + " / ".join(C_TXT[r] for r in RHOS) + f". The strips are clip {CLIP} (a 7); its own flicker "
            f"follows the same order ({', '.join(f'{CLIP_FLICK[r]:.3f}' for r in RHOS)}). At ρ = 0.9 the 7th frame jumps "
            "(the run's notes: 'an occasional jump'). The x axis is evenly spaced by setting, not to scale."))
        sh, gap_y = 0.46, 0.64
        ys = [1.85 - i * gap_y for i in range(5)]
        strips = m.Group()
        labs = m.VGroup()
        for r, y in zip(RHOS, ys):
            pic = digit_row(FRAMES[r][CLIP], DATA, sh).move_to([-2.75, y, 0])
            strips.add(pic)
            lab = code(f"ρ = {r}", 20, INK).move_to([0, y, 0])
            lab.align_to(pic, m.LEFT).shift((lab.width + 0.3) * m.LEFT)
            labs.add(lab)
        jump_x = row_x(strips[2], JUMP_K)
        jump_box = m.Rectangle(width=sh * 1.12, height=sh * 1.15).set_stroke(RED, 3).move_to([jump_x, ys[2], 0])
        defn = m.VGroup(code("flicker = mean |frame k+1 − frame k|", 20, INK),
                        cite("after moving frame k back by our 3 px slide")).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        defn.move_to([0, -1.75, 0]).align_to(labs, m.LEFT)
        assert labs.get_left()[0] > -6.6 and strips.get_right()[0] < -0.5 and strips.get_top()[1] < 2.6, (labs.get_left(), strips.get_right())
        assert defn.get_top()[1] < strips.get_bottom()[1] - 0.15, (defn.get_top(), strips.get_bottom())

        pl = Plot((-0.4, 4.4), (0, 0.1), width=4.6, height=3.0, center=(3.5, 0.4, 0), y_ticks=[0, 0.04, 0.08],
                  y_fmt="{:g}", y_label="flicker")
        xt = m.VGroup(*[code(r, 20, MUTED).next_to(pl.coords(i, 0), m.DOWN, buff=0.18) for i, r in enumerate(RHOS)])
        x_lab = text("noise correlation ρ (evenly spaced)", 22, MUTED).next_to(xt, m.DOWN, buff=0.14).set_x(pl.coords(2, 0)[0])
        cols = [mix(RED, GREEN, i / 4) for i in range(5)]
        dts = m.VGroup(*[pl.dot(i, FLICK[r], cols[i], 0.085) for i, r in enumerate(RHOS)])
        segs = m.VGroup(*[m.Line(pl.coords(i, FLICK[a]), pl.coords(i + 1, FLICK[b])).set_stroke(INK, 3)
                          for i, (a, b) in enumerate(zip(RHOS, RHOS[1:]))])
        vals = m.VGroup()
        for i, r in enumerate(RHOS):
            vals.add(code(F_TXT[r], 22, cols[i], weight=m.BOLD).next_to(dts[i], m.UR, buff=0.04))
        ch_lab = code("change", 20, MUTED)
        ch_vals = m.VGroup(*[code(C_TXT[r], 20, cols[i]).move_to([pl.coords(i, 0)[0], 0, 0]) for i, r in enumerate(RHOS)])
        ch_row = m.VGroup(ch_lab, ch_vals)
        y_ch = x_lab.get_bottom()[1] - 0.35
        for c_ in ch_vals:
            c_.set_y(y_ch)
        ch_lab.next_to(ch_vals[0], m.LEFT, buff=0.3)
        sit(ch_lab, baseline(ch_vals[0]))
        ch_note = cite("distance between frames (L2)").next_to(ch_vals, m.DOWN, buff=0.1).set_x(pl.coords(2, 0)[0])
        src = cite(f"generated on this Mac · the digit model · DDIM 50 · w = 1 · numbers: 20 clips × 8 frames · strips: clip {CLIP}")
        src.move_to([0, -3.35, 0])
        assert pl.get_right()[0] < 6.6 and ch_lab.get_left()[0] > strips.get_right()[0] + 0.2 and pl.get_top()[1] < 2.6, (pl.get_right(), ch_lab.get_left(), pl.get_top())
        assert ch_note.get_bottom()[1] > src.get_top()[1] + 0.06 and defn.get_bottom()[1] > src.get_top()[1] + 0.06, (ch_note.get_bottom(), defn.get_bottom())
        assert vals[0].get_right()[0] < pl.get_right()[0] + 0.3

        self.say(
            "[To measure flicker], [we compare each frame with the one before], [after undoing our three pixel slide].",
            m.AnimationGroup(self.new_header("03 · measured", "Flicker, measured"), m.FadeIn(pl), m.FadeIn(xt), m.FadeIn(x_lab)),
            m.FadeIn(defn[0], shift=0.1 * m.UP),
            m.FadeIn(defn[1], shift=0.1 * m.UP),
        )
        self.say(
            "[With independent noise], [the flicker is zero point zero seven eight].",
            m.AnimationGroup(m.FadeIn(strips[0], shift=0.1 * m.RIGHT), m.FadeIn(labs[0]), m.FadeIn(src)),
            m.AnimationGroup(m.FadeIn(dts[0], scale=0.5), m.FadeIn(vals[0])),
        )
        self.say(
            "[Share half the noise]: [zero point zero five four]. [Ninety percent]: [zero point zero two four], [with an occasional jump].",
            m.AnimationGroup(m.FadeIn(strips[1], shift=0.1 * m.RIGHT), m.FadeIn(labs[1])),
            m.AnimationGroup(m.Create(segs[0]), m.FadeIn(dts[1], scale=0.5), m.FadeIn(vals[1])),
            m.AnimationGroup(m.FadeIn(strips[2], shift=0.1 * m.RIGHT), m.FadeIn(labs[2])),
            m.AnimationGroup(m.Create(segs[1]), m.FadeIn(dts[2], scale=0.5), m.FadeIn(vals[2])),
            m.Create(jump_box),
        )
        self.say(
            "[At zero point nine nine, the seven holds steady]: [zero point zero zero seven].",
            m.AnimationGroup(m.FadeIn(strips[3], shift=0.1 * m.RIGHT), m.FadeIn(labs[3])),
            m.AnimationGroup(m.Create(segs[2]), m.FadeIn(dts[3], scale=0.5), m.FadeIn(vals[3])),
        )
        self.say(
            "[Share all of it], [and the flicker is exactly zero].",
            m.AnimationGroup(m.FadeIn(strips[4], shift=0.1 * m.RIGHT), m.FadeIn(labs[4])),
            m.AnimationGroup(m.Create(segs[3]), m.FadeIn(dts[4], scale=0.5), m.FadeIn(vals[4])),
        )
        self.say(
            "[The change between frames falls the same way], [from nine point two down to zero].",
            m.AnimationGroup(m.FadeIn(ch_lab), m.FadeIn(ch_note)),
            m.LaggedStart(*[m.FadeIn(c_, shift=0.1 * m.UP) for c_ in ch_vals], lag_ratio=0.25, run_time=1.6),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 4 · steady, but frozen

    def frozen(self):
        self.next_section("4 · Steady, but frozen", notes=(
            "At ρ = 1 all eight frames start from one noise and the sampler is deterministic (DDIM, η = 0), so every frame is "
            "the very same digit (asserted for all 20 clips): flicker 0, change 0. The only motion is the 3 px slide the "
            "script pastes on. The on-screen sentence is the run's own (mac_runs.md §6, 'On screen, say'). Real video "
            "models learn consistency by denoising all frames together (temporal attention, 3D convolutions or full "
            "spacetime attention: notes §22; Episode 21)."))
        h, buff = 0.92, 0.42
        frames = m.Group(*[picture(digit_rgb(FRAMES["1"][CLIP, k]), h) for k in range(NF)]).arrange(m.RIGHT, buff=buff)
        frames.move_to([0, 1.62, 0])
        eqs = m.VGroup(*[code("=", 40, MUTED, weight=m.BOLD).move_to([(a.get_right()[0] + b.get_left()[0]) / 2, 1.62, 0])
                         for a, b in zip(frames[:-1], frames[1:])])
        top_l = code("ρ = 1 · eight frames, one starting noise", 20, MUTED).next_to(frames, m.UP, buff=0.18).align_to(frames, m.LEFT)
        flip = Flip(lambda k: digit_rgb(CANVAS["1"][CLIP, k]), 0, 1.15)
        pbox = framed(flip.image, EDGE).move_to([-3.75, -0.12, 0])
        p_l = code("played as a clip", 20, MUTED).next_to(pbox, m.DOWN, buff=0.12)
        stats = m.VGroup(
            m.VGroup(code("flicker 0", 26, GREEN, weight=m.BOLD), text("steady", 24, INK)).arrange(m.RIGHT, buff=0.3),
            m.VGroup(code("change  0", 26, RED, weight=m.BOLD), text("nothing of its own moves", 24, INK)).arrange(m.RIGHT, buff=0.3),
        ).arrange(m.DOWN, buff=0.22, aligned_edge=m.LEFT)
        for s_ in stats:
            sit(s_[1], baseline(s_[0]))
        stats.next_to(pbox, m.RIGHT, buff=0.7).shift(0.15 * m.UP)
        _mid = m.Group(pbox, p_l, stats)
        _mid.shift(-_mid.get_center()[0] * m.RIGHT)
        need = m.VGroup(title_text("Real videos need frames that agree,", 34), title_text("yet move.", 34, GREEN)).arrange(m.RIGHT, buff=0.25)
        sit(need[1], baseline(need[0]))
        need.move_to([0, -1.75, 0])
        say_box = card(12.8, 1.05, MODEL, AMBER_FILL, radius=0.16, stroke=2).move_to([0, -3.0, 0])
        s1 = text("the cheapest consistency trick, correlated starting noise, removes the flicker;", 22, INK)
        s2 = text("a real video model learns consistency by denoising all frames together", 22, INK, weight=m.BOLD)
        ss = m.VGroup(s1, s2).arrange(m.DOWN, buff=0.1).move_to(say_box)
        assert frames.width < 12.8 and top_l.get_top()[1] < 2.6 and stats.get_right()[0] < 6.6, (frames.width, top_l.get_top(), stats.get_right())
        assert need.width < 12.8 and need.get_top()[1] < p_l.get_bottom()[1] - 0.1 and ss.width < say_box.width - 0.4, (need.get_top(), p_l.get_bottom(), ss.width)
        assert need.get_bottom()[1] > say_box.get_top()[1] + 0.1 and say_box.get_bottom()[1] > -3.6

        self.say(
            "[But look closer]: [every frame is now the very same seven].",
            m.AnimationGroup(self.new_header("04 · the catch", "Steady, but frozen"),
                             m.LaggedStart(*[m.FadeIn(f_, shift=0.1 * m.UP) for f_ in frames], lag_ratio=0.1, run_time=1.2), m.FadeIn(top_l)),
            m.LaggedStart(*[m.FadeIn(e_, scale=0.6) for e_ in eqs], lag_ratio=0.15, run_time=1.2),
        )
        self.say(
            "[The only motion left is the slide we pasted on]: [consistency, bought by giving up any change].",
            m.AnimationGroup(m.FadeIn(pbox), m.FadeIn(p_l), play(flip, None, loops=1, seconds=2.4)),
            m.LaggedStart(m.FadeIn(stats[0], shift=0.1 * m.LEFT), m.FadeIn(stats[1], shift=0.1 * m.LEFT), lag_ratio=0.5, run_time=1.4),
        )
        self.say(
            "[Real videos need frames that agree], [yet move].",
            m.FadeIn(need[0], shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(need[1], shift=0.1 * m.UP), play(flip, None, loops=1, seconds=1.6)),
        )
        self.say(
            "[So: correlated starting noise is the cheapest consistency trick]. "
            "[A real video model learns consistency by denoising all frames together].",
            m.AnimationGroup(m.FadeIn(say_box), m.FadeIn(s1)),
            m.FadeIn(s2, shift=0.1 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 5 · an idealised joint model (the swirl)

    def together(self):
        self.next_section("5 · An idealised joint model", notes=(
            "COMPUTED, not trained: the swirl (diffusion_reference). A toy 'video' of 8 frames: frame k is one swirl point "
            "turned by k·45°, so the whole swirl turns 45° per frame. Both rows start from the same per-frame noise "
            "(seed 22) and run DDIM 50 with exact denoisers; the dots show each step's guess of the clean point "
            "(x̂0) in each frame. Top: each frame alone (the exact one-frame denoiser): every frame lands on the swirl, "
            "but on a different point. Bottom: the exact JOINT denoiser. Because rotations are orthogonal, turning every "
            "noisy frame back leaves standard noise, and the best guess from all 8 frames is the one-frame denoiser applied "
            f"to their average, whose noise is √8 = {SQRT_F}× smaller (checked against brute-force conditioning on all 16 "
            "numbers and against exact_denoiser, to 1e-15). Every frame then shows the same point, turning with the "
            f"swirl. Over {SW_N} clips, the mean jump between consecutive frames (turned back) is {JUMP_IND:.3f} alone and "
            f"{JUMP_JNT:.0f} together. Idealised: this model is TOLD how the frames move; a real model must learn it. Clip "
            f"{SW_CLIP} of {SW_N} is shown (a point far from the centre)."))
        side, gap = 1.05, 0.15
        xs = [-6.4 + side / 2 + i * (side + gap) for i in range(SW_F)]
        y1, y2 = 1.2, -1.0
        rows = []
        for y in (y1, y2):
            plots = [mini_plane((x, y), side) for x in xs]
            boxes = m.VGroup(*[m.Square(side).set_stroke(EDGE, 2).move_to(p.coords(0, 0)) for p in plots])
            bands = [swirl_band(p) for p in plots]
            rows.append((plots, boxes, bands))
        (p1, b1, s1), (p2, b2, s2) = rows
        dots1 = m.VGroup(*[m.Dot(p.coords(*IND_SHOW[0, k]), radius=0.055, color=NOISE).set_z_index(3) for k, p in enumerate(p1)])
        dots2 = m.VGroup(*[m.Dot(p.coords(*JNT_SHOW[0, k]), radius=0.055, color=NOISE).set_z_index(3) for k, p in enumerate(p2)])
        l1 = text("each frame on its own", 24, INK, weight=m.BOLD).next_to(b1, m.UP, buff=0.18).align_to(b1, m.LEFT)
        l2 = text("all frames together", 24, INK, weight=m.BOLD).next_to(b2, m.UP, buff=0.26).align_to(b2, m.LEFT)
        tag = pill("computed · an idealised joint model", INK, CARD_HI, 20)
        tag.move_to([0, l1.get_center()[1], 0]).align_to([6.45, 0, 0], m.RIGHT)
        joint_box = m.RoundedRectangle(corner_radius=0.12, width=xs[-1] - xs[0] + side + 0.2, height=side + 0.2)
        joint_box.set_stroke(MODEL, 3).set_fill(opacity=0).move_to([(xs[0] + xs[-1]) / 2, y2, 0])
        nums = m.VGroup(*[code(str(k + 1), 20, TIME, weight=m.BOLD).move_to([x, y2 - side / 2 - 0.36, 0]) for k, x in enumerate(xs)])
        t_arrow = m.Arrow([xs[0] - side / 2, y2 - side / 2 - 0.66, 0], [xs[-1] + side / 2, y2 - side / 2 - 0.66, 0], buff=0,
                          color=TIME, stroke_width=3, max_tip_length_to_length_ratio=0.03)
        t_lab = code("frames · the swirl turns 45° each", 20, TIME).next_to(t_arrow, m.DOWN, buff=0.1).align_to(t_arrow, m.LEFT)
        pool = m.VGroup(code("turn every noisy frame back, then average:", 20, INK),
                        code(f"the noise shrinks by √8 = {SQRT_F}", 20, MODEL, weight=m.BOLD)).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        pool.move_to([0, -3.2, 0]).align_to([xs[0] - side / 2, 0, 0], m.LEFT)
        rx = 4.85
        j1 = m.VGroup(cite("jump between frames"), code(f"{JUMP_IND:.2f}", 40, RED, weight=m.BOLD), cite("a new point every frame")).arrange(m.DOWN, buff=0.08)
        j2 = m.VGroup(cite("jump between frames"), code("0", 40, GREEN, weight=m.BOLD), cite("one point, turning")).arrange(m.DOWN, buff=0.08)
        j1.move_to([rx, y1 - 0.16, 0])
        j2.move_to([rx, y2, 0])
        j_src = cite(f"mean over {SW_N} clips,").next_to(j2, m.DOWN, buff=0.35)
        j_src2 = cite("frames turned back").next_to(j_src, m.DOWN, buff=0.06)
        agree = pill("agree, yet move", GREEN, GREEN_FILL, 22).move_to([rx, -3.32, 0])
        assert tag.get_left()[0] > l1.get_right()[0] + 0.3 and j1.get_left()[0] > xs[-1] + side / 2 + 0.3, (tag.get_left(), l1.get_right(), j1.get_left())
        assert l2.get_top()[1] < b1.get_bottom()[1] - 0.1 and joint_box.get_top()[1] < l2.get_bottom()[1] - 0.08, (l2.get_top(), joint_box.get_top())
        assert pool.get_right()[0] < agree.get_left()[0] - 0.2 and t_lab.get_bottom()[1] > pool.get_top()[1] + 0.08, (pool.get_right(), agree.get_left(), t_lab.get_bottom(), pool.get_top())
        assert j_src2.get_bottom()[1] > agree.get_top()[1] + 0.08 and tag.get_top()[1] < 2.6 and l1.get_top()[1] < 2.6, (j_src2.get_bottom(), agree.get_top())

        def turn(bands, plots, sign=1, seconds=1.4):
            return m.AnimationGroup(*[m.Rotate(b, angle=sign * k * SW_THETA, about_point=p.coords(0, 0))
                                      for k, (b, p) in enumerate(zip(bands, plots)) if k], run_time=seconds)

        def turn_all(sign, seconds):
            """Turn both rows (swirls and their points) back to frame 1's angle, or forward again."""
            parts = []
            for bands, plots, dots in ((s1, p1, dots1), (s2, p2, dots2)):
                for k in range(1, SW_F):
                    c_ = plots[k].coords(0, 0)
                    parts += [m.Rotate(bands[k], angle=sign * k * SW_THETA, about_point=c_),
                              m.Rotate(dots[k], angle=sign * k * SW_THETA, about_point=c_)]
            return m.AnimationGroup(*parts, run_time=seconds)

        self.say(
            "[So what does denoising frames together buy]? [Here is an idealised joint model], [computed exactly on the swirl].",
            m.AnimationGroup(self.new_header("05 · together", "An idealised joint model"), m.FadeIn(b1), m.FadeIn(b2)),
            m.AnimationGroup(m.FadeIn(tag, shift=0.1 * m.DOWN), m.FadeIn(m.VGroup(*s1)), m.FadeIn(m.VGroup(*s2))),
            m.AnimationGroup(m.FadeIn(nums), m.GrowArrow(t_arrow)),
        )
        self.say(
            "[In this toy video, the whole swirl turns forty-five degrees every frame].",
            m.AnimationGroup(turn(s1, p1, 1, 2.0), turn(s2, p2, 1, 2.0), m.FadeIn(t_lab)),
        )
        self.say(
            "[Denoise each frame on its own], [and every frame lands on a different point]: [flicker].",
            m.AnimationGroup(m.FadeIn(l1), b1.animate.set_stroke(MODEL, 3), m.FadeIn(dots1, scale=0.5)),
            replay_dots(dots1, p1, IND_SHOW, 2.6),
            m.FadeIn(j1, shift=0.1 * m.LEFT),
        )
        self.say(
            "[Denoise them together]: [turn every noisy frame back, and average]. [The noise shrinks by the square root of eight].",
            m.AnimationGroup(m.FadeIn(l2), m.Create(joint_box)),
            m.Succession(turn(s2, p2, -1, 1.1), m.Wait(0.3), turn(s2, p2, 1, 1.1)),
            m.FadeIn(pool, shift=0.1 * m.UP),
        )
        self.say(
            "[Now every frame shows the same point], [riding along with the swirl].",
            m.AnimationGroup(m.FadeIn(dots2, scale=0.5), replay_dots(dots2, p2, JNT_SHOW, 2.6)),
            m.FadeIn(j2, shift=0.1 * m.LEFT),
        )
        self.say(
            "[Turn the frames back to compare]: [the top row jumps around], [the bottom row stands still].",
            m.AnimationGroup(turn_all(-1, 1.6), m.FadeIn(j_src), m.FadeIn(j_src2)),
            m.Circumscribe(j1, color=RED, buff=0.08),
            m.Circumscribe(j2, color=GREEN, buff=0.08),
        )
        self.say(
            "[Turn them forward again]: [frames that agree, yet move].",
            turn_all(1, 1.6),
            m.FadeIn(agree, shift=0.1 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 6 · temporal attention

    def attention(self):
        self.next_section("6 · Temporal attention", notes=(
            "Real models are not told how frames move; they learn it by denoising all frames together. Notes §21–22 "
            "(read October 2026): Video Diffusion Models (Ho et al., arXiv:2204.03458, Apr 2022) made 3×3 convolutions "
            "1×3×3 (space only) and added temporal attention across frames; Make-A-Video (Singer et al., arXiv:2209.14792, "
            "Sep 2022) and Video LDM initialise their temporal layers as the identity, so the pretrained image model is "
            "unchanged at the start; CogVideoX (arXiv:2408.06072) and HunyuanVideo (arXiv:2412.03603, 2024) use full 3D "
            "attention over all spacetime patches. Episode 21 drew factorised space–time attention. The frames drawn are "
            f"our run's ρ = 0.99 clip {CLIP}; the highlighted square is one pixel position, for example."))
        h, gap = 1.22, 0.28
        xs = [-6.35 + h / 2 + i * (h + gap) for i in range(4)]
        y0 = -0.25
        frames = m.Group(*[picture(digit_rgb(FRAMES["0.99"][CLIP, 2 * k]), h).move_to([x, y0, 0]) for k, x in enumerate(xs)])
        borders = m.VGroup(*[m.SurroundingRectangle(f_, buff=0.04).set_stroke(EDGE, 2) for f_ in frames])
        px = h / 28
        pos = (4, 15)                                                          # row, column of the highlighted position
        sq_c = lambda f_: f_.get_corner(m.UL) + np.array([(pos[1] + 0.5) * px, -(pos[0] + 0.5) * px, 0])  # noqa: E731
        squares = m.VGroup(*[m.Square(px * 3).set_stroke(MODEL, 3).set_fill(MODEL, 0.25).move_to(sq_c(f_)).set_z_index(3)
                             for f_ in frames])
        q = 1
        arcs = m.VGroup(*[m.ArcBetweenPoints(squares[q].get_top() + [0, 0.02, 0], squares[j].get_top() + [0, 0.02, 0],
                                             angle=-1.6 if j > q else 1.6).set_stroke(MODEL, 4)
                          for j in range(4) if j != q])
        rng = np.random.default_rng(6)
        far = m.VGroup()
        for j in range(4):
            for _ in range(3):
                p = frames[j].get_corner(m.UL) + np.array([rng.uniform(0.15, 0.85) * h, -rng.uniform(0.15, 0.85) * h, 0])
                far.add(m.Line(squares[q].get_center(), p).set_stroke(MODEL, 1.5, opacity=0.55))
        far.set_z_index(2)
        nums = m.VGroup(*[code(f"frame {2 * k + 1}", 20, TIME).next_to(b_, m.DOWN, buff=0.14) for k, b_ in enumerate(borders)])
        t_arrow = m.Arrow([xs[0] - h / 2, y0 - h / 2 - 0.62, 0], [xs[-1] + h / 2 + 0.1, y0 - h / 2 - 0.62, 0], buff=0, color=TIME,
                          stroke_width=3, max_tip_length_to_length_ratio=0.05)
        t_lab = code("time", 20, TIME).next_to(t_arrow, m.DOWN, buff=0.1).align_to(t_arrow, m.RIGHT)
        cap = cite("one position, in every frame").next_to(arcs, m.UP, buff=0.22)
        frames_src = cite(f"frames: our ρ = 0.99 clip {CLIP}").next_to(t_lab, m.DOWN, buff=0.25).align_to(t_arrow, m.LEFT)
        ep21 = pill("drawn in Episode 21", BLUE, BLUE_FILL, 20).next_to(frames_src, m.DOWN, buff=0.4).set_x((xs[0] + xs[-1]) / 2)
        facts = fact_rows((
            ("Video Diffusion Models", "Ho et al., 2022", "space-only layers, plus attention across time"),
            ("Make-A-Video", "Singer et al., 2022", "temporal layers start as the identity: the image\nmodel is unchanged at first"),
            ("CogVideoX · HunyuanVideo", "2024", "full 3D attention: every patch sees every patch"),
        ), 6.4)
        f_box = boxed(facts, 6.4).move_to([3.15, -0.1, 0])
        facts.move_to(f_box).align_to(f_box, m.LEFT).shift(0.35 * m.RIGHT)
        f_t = code("how real models share across frames", 20, MODEL, weight=m.BOLD).next_to(f_box, m.UP, buff=0.15).align_to(f_box, m.LEFT)
        assert f_box.get_right()[0] < 6.6 and f_box.get_left()[0] > xs[-1] + h / 2 + 0.25 and f_t.get_top()[1] < 2.6, (f_box.get_right(), f_box.get_left(), f_t.get_top())
        assert arcs.get_top()[1] < cap.get_bottom()[1] - 0.08 and ep21.get_top()[1] < frames_src.get_bottom()[1] - 0.06, (arcs.get_top(), cap.get_bottom(), ep21.get_top())
        assert f_box.get_bottom()[1] > -3.6 and frames_src.get_left()[0] > -6.6

        self.say(
            "[Real models are not told how the frames move]: [they learn it, with temporal attention].",
            m.AnimationGroup(self.new_header("06 · temporal attention", "How real models share"),
                             m.LaggedStart(*[m.FadeIn(m.Group(f_, b_), shift=0.1 * m.RIGHT) for f_, b_ in zip(frames, borders)], lag_ratio=0.2, run_time=1.4),
                             m.FadeIn(nums), m.GrowArrow(t_arrow), m.FadeIn(t_lab), m.FadeIn(frames_src)),
            m.AnimationGroup(m.FadeIn(squares, scale=0.5), m.LaggedStart(*[m.Create(a_) for a_ in arcs], lag_ratio=0.3, run_time=1.4), m.FadeIn(cap)),
        )
        self.say(
            "[Each position attends to itself in every other frame], [as Episode twenty-one drew it].",
            m.LaggedStart(*[m.ShowPassingFlash(a_.copy().set_stroke(INK, 7), time_width=0.6) for a_ in arcs], lag_ratio=0.25, run_time=1.6),
            m.FadeIn(ep21, shift=0.1 * m.UP),
        )
        self.say(
            "[These layers often start out doing nothing], [so a trained image model keeps working], [then learns to share across frames].",
            m.AnimationGroup(m.FadeIn(f_box), m.FadeIn(f_t), m.FadeIn(facts[0], shift=0.1 * m.UP)),
            m.FadeIn(facts[1], shift=0.1 * m.UP),
            m.LaggedStart(*[m.ShowPassingFlash(a_.copy().set_stroke(MODEL, 7), time_width=0.6) for a_ in arcs], lag_ratio=0.25, run_time=1.4),
        )
        self.say(
            "[Newer models go further]: [full attention, where every patch sees every patch in the clip].",
            m.FadeIn(facts[2], shift=0.1 * m.UP),
            m.LaggedStart(*[m.Create(l_) for l_ in far], lag_ratio=0.06, run_time=1.6),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 7 · frame interpolation

    def interpolation(self):
        self.next_section("7 · Keyframes, then fill in", notes=(
            "Notes §21–22 (read October 2026): Imagen Video (Ho et al., arXiv:2210.02303, Oct 2022) cascades temporal "
            "super-resolution models that 'increase temporal resolution by filling in intermediate frames between input "
            f"frames'; Make-A-Video (Singer et al., 2022) interpolates 16 frames to 76 = (16 − 1)·5 + 1 = {MAV_FRAMES}; "
            "Movie Gen (Meta, arXiv:2410.13720, Oct 2024): its temporal autoencoder compresses time 8×, which 'allows us to "
            "forego frame-interpolation models commonly used in prior work'. NUWA-XL (Yin et al., 2023) applies the same "
            "idea to long videos: keyframes across the whole range, then fill in. The thrown ball is illustrative."))
        n, h, gap = 9, 1.05, 0.2
        xs = [-(n - 1) / 2 * (h + gap) + i * (h + gap) for i in range(n)]
        y0 = 1.15
        keys = (0, 4, 8)
        slots = m.VGroup(*[m.Square(h).set_stroke(FAINT, 2).move_to([x, y0, 0]) for x in xs])
        balls, grounds = m.VGroup(), m.VGroup()
        for i, x in enumerate(xs):
            u = i / (n - 1)
            c = np.array([x - 0.32 + 0.64 * u, y0 - 0.3 + 0.62 * (1 - (2 * u - 1) ** 2), 0])
            balls.add(m.Circle(radius=0.1).set_stroke(DATA, 2.5).set_fill(DATA, 0.85 if i in keys else 0.45).move_to(c))
            grounds.add(m.Line([x - 0.38, y0 - 0.42, 0], [x + 0.38, y0 - 0.42, 0]).set_stroke(MUTED, 2))
        key_frames = m.VGroup(*[m.Square(h).set_stroke(DATA, 4).move_to(slots[i]) for i in keys])
        mid_frames = m.VGroup(*[m.Square(h).set_stroke(mix(DATA, BG, 0.45), 2.5).move_to(slots[i]) for i in range(n) if i not in keys])
        key_l = m.VGroup(*[code("key", 20, DATA, weight=m.BOLD).next_to(slots[i], m.UP, buff=0.12) for i in keys])
        fill_l = m.VGroup(*[cite("filled in").move_to([(xs[a + 1] + xs[b - 1]) / 2, key_l[0].get_center()[1], 0]) for a, b in ((0, 4), (4, 8))])
        for fl in fill_l:
            sit(fl, baseline(key_l[0]))
        t_arrow = m.Arrow([xs[0] - h / 2, y0 - h / 2 - 0.3, 0], [xs[-1] + h / 2 + 0.15, y0 - h / 2 - 0.3, 0], buff=0, color=TIME,
                          stroke_width=3, max_tip_length_to_length_ratio=0.025)
        t_lab = code("time", 20, TIME).next_to(t_arrow, m.DOWN, buff=0.1).align_to(t_arrow, m.RIGHT)
        ex = cite("for example: a thrown ball").next_to(t_arrow, m.DOWN, buff=0.1).align_to(t_arrow, m.LEFT)
        facts = fact_rows((
            ("Imagen Video", "Ho et al., 2022", "fills in frames between the frames it is given"),
            ("Make-A-Video", "Singer et al., 2022", f"16 frames → {MAV_FRAMES}:  (16 − 1) × 5 + 1"),
            ("Movie Gen", "Meta, 2024", "its autoencoder compresses time 8×: no interpolation model"),
        ), 12.6, stacked=False)
        f_box = boxed(facts, 12.6).move_to([0, -1.95, 0])
        facts.move_to(f_box).align_to(f_box, m.LEFT).shift(0.35 * m.RIGHT)
        assert slots.width < 12.6 and key_l.get_top()[1] < 2.6 and f_box.get_top()[1] < ex.get_bottom()[1] - 0.12, (slots.width, key_l.get_top(), f_box.get_top(), ex.get_bottom())
        assert f_box.get_bottom()[1] > -3.6

        self.say(
            "[Another trick]: [generate a few keyframes first], [then fill in the frames between them].",
            m.AnimationGroup(self.new_header("07 · interpolation", "Keyframes, then fill in"), m.FadeIn(slots), m.GrowArrow(t_arrow),
                             m.FadeIn(t_lab), m.FadeIn(ex)),
            m.AnimationGroup(*[m.AnimationGroup(m.FadeIn(key_frames[j]), m.FadeIn(balls[i], scale=0.5), m.FadeIn(grounds[i]), m.FadeIn(key_l[j]))
                               for j, i in enumerate(keys)]),
            m.AnimationGroup(m.LaggedStart(*[m.AnimationGroup(m.FadeIn(mid_frames[j]), m.FadeIn(balls[i], scale=0.5), m.FadeIn(grounds[i]))
                                             for j, i in enumerate(i_ for i_ in range(n) if i_ not in keys)], lag_ratio=0.2, run_time=1.8),
                             m.FadeIn(fill_l)),
        )
        self.say(
            "[Early cascades did this to raise the frame rate]: [Make-A-Video turned sixteen frames into seventy-six].",
            m.AnimationGroup(m.FadeIn(f_box), m.FadeIn(facts[0], shift=0.1 * m.UP)),
            m.FadeIn(facts[1], shift=0.1 * m.UP),
        )
        self.say(
            "[Newer models compress time inside the autoencoder], [and skip the separate interpolation model].",
            m.FadeIn(facts[2], shift=0.1 * m.UP),
            m.Circumscribe(facts[2], color=DATA, buff=0.08),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 8 · long videos

    def long_videos(self):
        self.next_section("8 · Long videos", notes=(
            "Notes §22 (read October 2026). Long videos exceed one pass: extend autoregressively, conditioning on the last "
            "frames; or a sliding window whose later frames are noisier (Rolling Diffusion: Ruhe, Heek, Salimans & "
            "Hoogeboom, arXiv:2402.09470, Feb 2024); Diffusion Forcing (Chen et al., arXiv:2407.01392, Jul 2024; NeurIPS "
            "2024) trains with an independent noise level per frame; Self Forcing (Huang et al., arXiv:2506.08009, Jun "
            "2025) trains on the model's own rollouts: 'real-time video generation at 17 FPS with sub-second latency on a "
            "single H100 GPU'. Errors accumulate ('drift'); noise augmentation of the context and training on the model's "
            "own rollouts fight it. Product (dated): Veo 3.1 (preview 15 Oct 2025) Extend conditions on 'the final second "
            "of the previous clip'. The timeline (16 frames, a window of 8 sliding by 4) and the drift colours are "
            "illustrative."))
        n, w, gap = 16, 0.68, 0.1
        xs = [-(n - 1) / 2 * (w + gap) + i * (w + gap) for i in range(n)]
        y0 = 1.58
        slots = m.VGroup(*[m.Square(w).set_stroke(FAINT, 2).set_fill(CARD, 1).move_to([x, y0, 0]) for x in xs])
        drift = [mix(DATA, RED, max(0.0, (i - 3) / 12) ** 1.3) for i in range(n)]
        fills = m.VGroup(*[m.Square(w).set_stroke(DATA, 2).set_fill(TEAL_FILL, 1).move_to([x, y0, 0]) for x in xs])
        marks = m.VGroup(*[m.Circle(radius=0.16).set_stroke(DATA, 3).move_to([x, y0, 0]) for x in xs]).set_z_index(3)
        win_w = 8 * w + 7 * gap + 0.24

        def window_at(i0):
            return [xs[i0] - w / 2 - 0.12 + win_w / 2, y0, 0]

        window = m.RoundedRectangle(corner_radius=0.12, width=win_w, height=w + 0.28).set_stroke(MODEL, 4).set_fill(opacity=0)
        window.move_to(window_at(0))
        win_l = code("one pass: 8 frames", 20, MODEL, weight=m.BOLD).next_to(window, m.UP, buff=0.12)
        win_g = m.VGroup(window, win_l)
        win_off = win_g.get_center() - window.get_center()
        ctx = m.VGroup(*[m.Square(w + 0.08).set_stroke(TIME, 3).move_to([x, y0, 0]) for x in xs])
        ctx4, ctx8 = ctx[4:8], ctx[8:12]
        ctx_l = code("context: the last frames", 20, TIME)
        t_arrow = m.Arrow([xs[0] - w / 2, y0 - w / 2 - 0.3, 0], [xs[-1] + w / 2 + 0.1, y0 - w / 2 - 0.3, 0], buff=0, color=TIME,
                          stroke_width=3, max_tip_length_to_length_ratio=0.02)
        t_lab = code("time", 20, TIME).next_to(t_arrow, m.DOWN, buff=0.1).align_to(t_arrow, m.RIGHT)
        ctx_l.next_to(t_arrow, m.DOWN, buff=0.1).set_x(np.mean(xs[4:8]))
        d_bar = m.VGroup(*[m.Rectangle(width=w + gap, height=0.14).set_stroke(width=0).set_fill(mix(GREEN, RED, i / (n - 1)), 1)
                           .move_to([x, y0 - w / 2 - 1.05, 0]) for i, x in enumerate(xs)])
        d_lab = m.VGroup(code("drift", 22, RED, weight=m.BOLD), cite("small errors pile up, chunk after chunk")).arrange(m.RIGHT, buff=0.25)
        sit(d_lab[1], baseline(d_lab[0]))
        d_lab.next_to(d_bar, m.DOWN, buff=0.14).align_to(d_bar, m.LEFT)
        ex = cite("for example: 16 frames, a window of 8, sliding by 4").next_to(d_bar, m.DOWN, buff=0.14).align_to(d_bar, m.RIGHT)
        facts = fact_rows((
            ("Rolling Diffusion", "Ruhe et al., 2024", "a sliding window: later frames are noisier"),
            ("Diffusion Forcing", "Chen et al., 2024", "every frame gets its own noise level"),
            ("Self Forcing", "Huang et al., 2025", "trains on its own rollouts · 17 frames/s on one H100"),
            ("Veo 3.1 Extend", "Google, 2025", "continues from “the final second of the previous clip”"),
        ), 12.6, stacked=False)
        f_box = boxed(facts, 12.6)
        f_box.next_to(d_lab, m.DOWN, buff=0.22).set_x(0)
        facts.move_to(f_box).align_to(f_box, m.LEFT).shift(0.35 * m.RIGHT)
        assert slots.width < 12.6 and win_l.get_top()[1] < 2.6 and f_box.get_top()[1] < d_lab.get_bottom()[1] - 0.12, (slots.width, win_l.get_top(), f_box.get_top(), d_lab.get_bottom())
        assert f_box.get_bottom()[1] > -3.6 and ex.get_left()[0] > d_lab.get_right()[0] + 0.3, (f_box.get_bottom(), ex.get_left(), d_lab.get_right())
        assert ctx_l.get_bottom()[1] > d_bar.get_top()[1] + 0.05, (ctx_l.get_bottom(), d_bar.get_top())

        def make(idx, seconds=0.9):
            return m.LaggedStart(*[m.AnimationGroup(m.FadeIn(fills[i]), m.FadeIn(marks[i], scale=0.5)) for i in idx], lag_ratio=0.15, run_time=seconds)

        def recolour(seconds=1.6):
            def step(_, a):
                for i in range(n):
                    fills[i].set_stroke(mix(DATA, drift[i], a), 2).set_fill(mix(TEAL_FILL, RED_FILL, a * max(0.0, (i - 3) / 12)), 1)
            return m.UpdateFromAlphaFunc(fills, step, run_time=seconds)

        wobble = [marks[i].animate(run_time=1.8).stretch(1 + 0.6 * max(0.0, (i - 3) / 12), 0).set_stroke(drift[i], 3) for i in range(4, n)]
        self.say(
            "[Long videos don't fit in one pass]. [So the model slides a window along], [making each new chunk from the last frames it made].",
            m.AnimationGroup(self.new_header("08 · long videos", "Sliding along, and drifting"), m.FadeIn(slots), m.GrowArrow(t_arrow), m.FadeIn(t_lab)),
            m.AnimationGroup(m.FadeIn(win_g), make(range(8), 1.2)),
            m.Succession(
                m.AnimationGroup(win_g.animate.move_to(np.array(window_at(4)) + win_off), m.FadeIn(ctx4), m.FadeIn(ctx_l), run_time=0.8),
                make(range(8, 12), 0.9),
                m.AnimationGroup(win_g.animate.move_to(np.array(window_at(8)) + win_off), m.FadeOut(ctx4), m.FadeIn(ctx8),
                                 ctx_l.animate.set_x(np.mean(xs[8:12])), run_time=0.8),
                make(range(12, 16), 0.9)),
        )
        self.say(
            "[The catch is drift]: [small errors pile up], [chunk after chunk].",
            m.AnimationGroup(m.FadeIn(d_lab[0], shift=0.1 * m.UP), m.FadeOut(ctx8), m.FadeOut(ctx_l)),
            m.AnimationGroup(recolour(1.8), *wobble),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(s_) for s_ in d_bar], lag_ratio=0.08, run_time=1.4), m.FadeIn(d_lab[1]), m.FadeIn(ex)),
        )
        self.say(
            "[The fixes: give every frame its own noise level], [and train on the model's own rollouts].",
            m.AnimationGroup(m.FadeIn(f_box), m.FadeIn(facts[0], shift=0.1 * m.UP), m.FadeIn(facts[1], shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(facts[2], shift=0.1 * m.UP), m.FadeIn(facts[3], shift=0.1 * m.UP)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 9 · world models

    def world_models(self):
        self.next_section("9 · World models", notes=(
            "Notes §22 and Appendix C row 22 (dated, read October 2026). GameNGen (Valevski, Leviathan, Arar & Fruchter, "
            "arXiv:2408.14837, Aug 2024; ICLR 2025): 'runs at 20 frames per second on a single TPU'; built on Stable "
            "Diffusion v1.4, conditioned on the last 64 frames and the actions, 4 DDIM steps; without noise augmentation "
            "of the context its quality 'degrades quickly after 10-20 frames'. Genie 3 (Google DeepMind blog, 5 Aug 2025): "
            "'navigate in real time at 24 frames per second, retaining consistency for a few minutes at a resolution of "
            "720p'; architecture not disclosed. Genie (Bruce et al., arXiv:2402.15391, Feb 2024; 11B parameters) is NOT a "
            "diffusion model (video tokenizer + MaskGIT dynamics). No 'Genie 4' as of 2026-10-25. The game frame drawn "
            "is illustrative."))
        cx = -3.15
        action = pill("your action: move left", PROMPT, PINK_FILL, 22).move_to([cx, 1.65, 0])
        model = m.VGroup(card(3.0, 0.95, MODEL, AMBER_FILL), text("video model", 26, INK, weight=m.BOLD))
        model[1].move_to(model[0])
        model.move_to([cx, 0.2, 0])
        scr = m.Rectangle(width=2.4, height=1.5).set_stroke(DATA, 3).set_fill(CARD, 1).move_to([cx, -1.7, 0])
        horizon = m.Line(scr.get_left() + [0.15, -0.15, 0], scr.get_right() + [-0.15, -0.15, 0]).set_stroke(MUTED, 2)
        hero = m.Square(0.24).set_stroke(DATA, 3).set_fill(DATA, 0.8).move_to(scr.get_center() + [0.35, 0.0, 0])
        scr_l = code("next frame", 22, DATA, weight=m.BOLD).next_to(scr, m.DOWN, buff=0.14)
        a1 = arrow(action.get_bottom(), model[0].get_top(), PROMPT)
        a2 = arrow(model[0].get_bottom(), scr.get_top(), DATA)
        back = m.CurvedArrow(scr.get_left() + [-0.05, 0, 0], model[0].get_left() + [-0.05, 0, 0], angle=-0.55 * np.pi)
        back.set_stroke(TIME, 3)
        back_l = code("context", 20, TIME).next_to(back, m.LEFT, buff=0.12)
        ex = cite("for example").next_to(scr_l, m.DOWN, buff=0.12)
        facts = fact_rows((
            ("GameNGen", "Valevski et al., Aug 2024", "plays DOOM at 20 frames/s on one TPU,\nfrom the last 64 frames and the actions"),
            ("Genie 3", "Google DeepMind, Aug 2025", "explorable worlds at 720p, 24 frames/s,\nconsistent “for a few minutes”"),
            ("Genie", "Bruce et al., 2024", "a world model that is not diffusion (MaskGIT)"),
        ), 7.2)
        f_box = boxed(facts, 7.2).move_to([2.95, -0.25, 0])
        facts.move_to(f_box).align_to(f_box, m.LEFT).shift(0.35 * m.RIGHT)
        dated = cite("as of October 2026").next_to(f_box, m.DOWN, buff=0.15).align_to(f_box, m.RIGHT)
        assert f_box.get_right()[0] < 6.6 and f_box.get_left()[0] > max(model.get_right()[0], action.get_right()[0]) + 0.2, (f_box.get_right(), f_box.get_left())
        assert back_l.get_left()[0] > -6.6 and ex.get_bottom()[1] > -3.6 and action.get_top()[1] < 2.6, (back_l.get_left(), ex.get_bottom())

        loop = m.VGroup(a1, a2, back)
        self.say(
            "[Push further, and a video model becomes a world model]: [it predicts the next frames from your actions], [in real time].",
            m.AnimationGroup(self.new_header("09 · world models", "Video models that respond"), m.FadeIn(model)),
            m.AnimationGroup(m.FadeIn(action, shift=0.1 * m.DOWN), m.GrowArrow(a1), m.GrowArrow(a2), m.FadeIn(scr), m.FadeIn(horizon),
                             m.FadeIn(hero), m.FadeIn(scr_l), m.FadeIn(ex)),
            m.AnimationGroup(m.Create(back), m.FadeIn(back_l),
                             m.Succession(pulse(a1, PROMPT), pulse(a2, DATA), hero.animate(run_time=0.6).shift(0.6 * m.LEFT))),
        )
        self.say(
            "[In twenty twenty-four], [a diffusion model ran the game Doom at twenty frames per second], [on a single chip].",
            m.AnimationGroup(m.FadeIn(f_box), m.FadeIn(dated)),
            m.FadeIn(facts[0], shift=0.1 * m.UP),
            m.Succession(pulse(a1, PROMPT), pulse(a2, DATA), m.AnimationGroup(hero.animate(run_time=0.5).shift(0.5 * m.RIGHT))),
        )
        self.say(
            "[And in twenty twenty-five], [DeepMind's Genie three kept explorable worlds consistent for a few minutes].",
            m.FadeIn(facts[1], shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(facts[2], shift=0.1 * m.UP), m.Circumscribe(loop, color=TIME, buff=0.15)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes=(
            f"Share: correlated starting noise cut our digit model's flicker from {F_TXT['0']} to {F_TXT['0.99']} (ρ = 0.99), "
            "and to 0 at ρ = 1 — where every frame is the same digit (measured on this Mac; the motion was pasted on). "
            "Attend: denoising frames together lets them agree, yet move (the swirl's idealised joint model: jump "
            f"{JUMP_IND:.2f} → 0; real models: temporal or full spacetime attention). Remember: long videos condition on "
            "the last frames, and fight drift. Next: Episode 23, Diffusion on Your Mac."))
        halo = glow(TIME, radius=4.0)
        words = m.VGroup(*[title_text(w, 58, c) for w, c in (("Share", NOISE), ("Attend", MODEL), ("Remember", TIME))])
        subs = m.VGroup(*[code(s, 20, MUTED) for s in ("the starting noise:", "across all frames:", "the last frames:")])
        subs2 = m.VGroup(*[code(s, 20, MUTED) for s in ("steady, but frozen", "agree, yet move", "longer, but drifting")])
        words.arrange(m.RIGHT, buff=1.6)
        on_baseline(words)
        line_y = min(w_.get_bottom()[1] for w_ in words) - 0.45
        for w_, s, s2 in zip(words, subs, subs2):
            s.set_x(w_.get_x())
            s2.set_x(w_.get_x())
            sit(s, line_y)
            sit(s2, line_y - 0.36)
        mid = baseline(words[0]) + 0.3
        dts = m.VGroup(*[m.Dot([(l_.get_right()[0] + r_.get_left()[0]) / 2, mid, 0], radius=0.07, color=MUTED)
                         for l_, r_ in zip(words, words[1:])])
        row = m.VGroup(words, dts, subs, subs2).move_to(0.9 * m.UP)
        assert row.width < 12.8 and all(a.get_right()[0] + 0.3 < b.get_left()[0] for a, b in zip(subs2, subs2[1:])), [(s_.get_left()[0], s_.get_right()[0]) for s_ in subs2]
        tag = text("Frames must agree, yet move: that is the hard part of video.", 28, MUTED)
        mascot = pixel(0.42)
        nxt = pill("NEXT · DIFFUSION ON YOUR MAC", LIME, LIME_FILL, 20, mono=True)
        foot = m.VGroup(tag, m.VGroup(mascot, nxt).arrange(m.RIGHT, buff=0.3)).arrange(m.DOWN, buff=0.4)
        foot.next_to(row, m.DOWN, buff=0.7)
        assert foot.get_bottom()[1] > -3.6 and tag.width < 13.0, (foot.get_bottom(), tag.width)

        self.say(
            "[To sum up]: [share], [attend], [remember].",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.3 * m.UP), m.FadeIn(subs[0]), m.FadeIn(subs2[0])),
            m.AnimationGroup(m.FadeIn(dts[0]), m.FadeIn(words[1], shift=0.3 * m.UP), m.FadeIn(subs[1]), m.FadeIn(subs2[1])),
            m.AnimationGroup(m.FadeIn(dts[1]), m.FadeIn(words[2], shift=0.3 * m.UP), m.FadeIn(subs[2]), m.FadeIn(subs2[2])),
        )
        self.say(
            "[Frames must agree], [yet move]: [that is the hard part of video].",
            m.FadeIn(tag),
            m.Wait(0.1),
            m.AnimationGroup(m.FadeIn(mascot, shift=0.2 * m.UP), m.FadeIn(nxt, shift=0.2 * m.UP)),
        )
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, foot)], run_time=0.9)
