"""Out of the Noise · Episode 4 · Adding Noise (Module 1 · Pictures from Noise — the module's last episode)

The forward process. Running examples: the swirl (2D points, diffusion_reference) and a real handwritten digit (the
first 7 in the MNIST training set, load_mnist("train")[15], the digit model's data), both pushed through DDPM's linear
schedule. The episode watches the swirl melt into a Gaussian blob (q_sample on the toy.json points, zooming out as the
cloud spreads) and the digit dissolve at t = 0, 50, 100, 250, 500, 1000; says what Gaussian noise is (the digit's own
noise, 784 draws); takes one small step x_t = √(1 − β_t)·x_(t−1) + √β_t·ε with DDPM's β (1e-4 → 0.02, T = 1000);
shows why one jump is allowed (Gaussians add; 500 small steps vs one jump give the same cloud; the closed form
x_t = √ᾱ_t·x₀ + √(1 − ᾱ_t)·ε, ᾱ_t = ∏(1 − β_s)); compares the linear and cosine schedules (ᾱ at 100 … 1000) and the
signal-to-noise ratio on a log axis (SNR = 1 at t ≈ 259 vs 496). It closes Module 1: making noise needs no learning;
undoing it is Module 2 (Episode 5).

Every number on screen is computed below from diffusion_reference.py and asserted (against measured('toy', …) where
the swirl's run has it), or cited with its source.

Sources (research/notes.md §4 and §0 rows 1–3, Appendix A row "4 · Cosine ᾱ_T", Appendix C row 4; facts dated October 2026):
- Ho, Jain & Abbeel 2020 (DDPM), eq. 2 (one step), eq. 4 (the closed form), §4 (T = 1000, β linear 1e-4 → 0.02).
- Nichol & Dhariwal 2021 (iDDPM), eq. 17 (cosine schedule, s = 0.008, β ≤ 0.999), §3.2 (the end of the linear schedule
  is "too noisy, and so doesn't contribute very much to sample quality" for 64×64 and 32×32 images; Fig. 3: "the last
  quarter of the linear schedule are almost purely noise").
- Song et al. 2021 (score SDE): the "variance preserving" name. Lin et al. 2023, Table 1: no schedule reaches SNR = 0.
- research/experiments/toy.md §1 / toy.json "forward" and "headline": the swirl snapshots, ᾱ_500, t(SNR = 1), ᾱ_1000,
  the perfect classifier's arm accuracy at t = 500 (20,000 points).

Illustrative (said "for example" or plainly a picture): the two narrow bells (std 0.6 and 0.8) that add up to one of
std 1 in "Jump to any step".

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403
from diffusion_reference import *  # noqa: E402,F403

PREFIX = "ep04"
AB, ABC = ALPHA_BAR, ALPHA_BAR_COSINE
BETAS = linear_betas()
PALE = mix(TEAL, INK, 0.7)       # the swirl's second arm: still data, a lighter teal
LIGHT = mix(BLUE, INK, 0.5)      # the cosine schedule: still the schedule's blue, lighter
H = lambda *keys: measured("toy", "headline", *keys)  # noqa: E731

# ---------------------------------------------------------------------------- the schedule (DDPM, linear)

assert T == 1000 and len(AB) == T + 1 and AB[0] == 1.0 and len(ABC) == T + 1       # arrays indexed by t
assert np.isclose(BETAS[1], 1e-4) and np.isclose(BETAS[T], 0.02)                 # "one in ten thousand … two in a hundred"
assert np.allclose(np.cumprod(1 - BETAS), AB)                                    # ᾱ_t = ∏ (1 − β_s)
assert abs(AB[500] - H("alpha_bar_500_linear")) < 5e-5 and round(100 * AB[500]) == 8   # "only eight percent"
assert abs(ABC[500] - H("alpha_bar_500_cosine")) < 5e-5 and f"{ABC[500]:.3f}" == "0.494"   # "about half"
assert 0.5 < AB[250] < 0.55                                                       # "about half the signal is gone"
assert AB[750] < 0.004 and round(100 * ABC[750]) == 14                            # "almost nothing" / "fourteen percent"
assert abs(AB[T] - H("alpha_bar_1000_linear")) < 5e-8 and f"{AB[T]:.1e}" == "4.0e-05"  # "four in a hundred thousand"


AB_T_M, AB_T_E = f"{AB[T]:.1e}".split("e")
AB_T_E = int(AB_T_E)
assert (AB_T_M, AB_T_E) == ("4.0", -5)


def t_even(ab):
    """The (fractional) step where ᾱ = ½, i.e. SNR = 1: signal and noise equally strong."""
    return float(np.interp(0.5, ab[::-1], np.arange(T + 1)[::-1]))


T_EVEN, T_EVEN_COS = t_even(AB), t_even(ABC)
assert abs(T_EVEN - H("t_snr_1_linear")) < 0.05 and round(T_EVEN) == 259          # "two hundred fifty-nine"
assert abs(T_EVEN_COS - H("t_snr_1_cosine")) < 0.05 and round(T_EVEN_COS) == 496  # "four hundred ninety-six"
assert round(snr(AB[1])) == 9999                                                  # "about ten thousand"
ARM_500 = H("arm_accuracy_t500_linear")
assert ARM_500 == 0.528                                                           # "fifty-two point eight percent"

TABLE_T = [100, 250, 500, 750, 1000]
for _sched, _ab in (("linear", AB), ("cosine", ABC)):
    _rows = {r["t"]: r for r in measured("toy", "forward", _sched, "table")}
    for _t in TABLE_T:
        assert abs(_ab[_t] - _rows[_t]["alpha_bar"]) <= 1e-3 * _ab[_t]            # the table agrees with toy.json


def fmt_ab(v):
    """ᾱ for the table: three significant digits, tiny values as 4.0e-5."""
    return f"{v:.1e}".replace("e-0", "e-") if v < 1e-3 else f"{v:.3g}"


assert [fmt_ab(AB[t]) for t in TABLE_T] == ["0.897", "0.524", "0.0786", "0.00335", "4.0e-5"]
assert [fmt_ab(ABC[t]) for t in TABLE_T] == ["0.972", "0.847", "0.494", "0.144", "2.4e-9"]   # notes §4, Appendix A

B250 = BETAS[250]
SHRINK_250, ADD_250 = np.sqrt(1 - B250), np.sqrt(B250)
assert f"{SHRINK_250:.4f}" == "0.9975" and f"{ADD_250:.3f}" == "0.071"
SIG_500, NOI_500 = np.sqrt(AB[500]), np.sqrt(1 - AB[500])
assert f"{SIG_500:.2f}" == "0.28" and f"{NOI_500:.2f}" == "0.96"                  # "zero point two eight … nine six"
SNR_250, SNR_500 = float(snr(AB[250])), float(snr(AB[500]))
assert f"{SNR_250:.2f}" == "1.10" and f"{SNR_500:.3f}" == "0.085"

# ---------------------------------------------------------------------------- the digit (MNIST training set)

IMAGES, LABELS = load_mnist("train")
DIGIT_INDEX = int(np.flatnonzero(LABELS == 7)[0])
assert DIGIT_INDEX == 15 and IMAGES.shape == (60000, 28, 28)                      # "the first seven"
X0_DIGIT = IMAGES[DIGIT_INDEX] / 127.5 - 1.0                                      # the model's range, −1 … 1
NOISE_SEED = 4
EPS_DIGIT = np.random.default_rng(NOISE_SEED).standard_normal((28, 28))
STRIP_T = [0, 50, 100, 250, 500, 1000]
assert STRIP_T == measured("toy", "forward", "snapshot_t")
WITHIN_1 = float(np.mean(np.abs(EPS_DIGIT) < 1))
BEYOND_3 = int(np.sum(np.abs(EPS_DIGIT) > 3))
assert round(100 * WITHIN_1) == 68 and BEYOND_3 == 2                              # "two in three", "only two"
FRESH = np.random.default_rng(5).standard_normal((28, 28))                        # one step's fresh noise
X249 = q_sample(X0_DIGIT, 249, EPS_DIGIT)
X250_STEP = SHRINK_250 * X249 + ADD_250 * FRESH
assert np.abs(X250_STEP - X249).mean() < 0.06                                    # "you can't see the change"

# ---------------------------------------------------------------------------- the swirl (toy.json's points)

_x, _arm = swirl_labelled(20000, seed=0)            # toy.json's forward-process points are the first 1,000 of these
SW_X0, SW_ARM = _x[:1000], _arm[:1000]
SW_EPS = gaussian_noise(20000, seed=1)[:1000]
for _sched, _ab in (("linear", AB), ("cosine", ABC)):
    for _t in STRIP_T:
        _snap = np.array(measured("toy", "forward", _sched, "snapshots", str(_t)))
        assert np.abs(q_sample(SW_X0, _t, SW_EPS, _ab) - _snap).max() < 6e-4       # = the stored snapshots
SW_EXT = np.maximum.accumulate([np.abs(q_sample(SW_X0, t, SW_EPS)).max() for t in range(T + 1)])

# 500 small steps (fresh noise each time) against one jump to t = 500
N_CHECK = 100_000
CH_X0, CH_ARM = swirl_labelled(N_CHECK, seed=3)


def chain_walk(x0, t_end=500, seed=2, every=5, keep=1000):
    """The forward process one step at a time: x_t = √(1 − β_t)·x_(t−1) + √β_t·z, fresh z every step."""
    rng = np.random.default_rng(seed)
    x, path = x0.copy(), [x0[:keep].copy()]
    for t in range(1, t_end + 1):
        x = np.sqrt(1 - BETAS[t]) * x + np.sqrt(BETAS[t]) * rng.standard_normal(x.shape)
        if t % every == 0:
            path.append(x[:keep].copy())
    return x, np.array(path)


CH_END, CH_PATH = chain_walk(CH_X0)
JUMP_END = q_sample(CH_X0, 500, gaussian_noise(N_CHECK, seed=7))
spread = lambda p: float(np.sqrt(p.var(0).mean()))                                # noqa: E731  (RMS std of x and y)
SPREAD_THEORY = float(np.sqrt((AB[500] * CH_X0.var(0) + 1 - AB[500]).mean()))
assert f"{spread(CH_END):.2f}" == f"{spread(JUMP_END):.2f}" == f"{SPREAD_THEORY:.2f}" == "0.97"
assert np.abs(CH_END.mean(0)).max() < 0.01 and np.abs(JUMP_END.mean(0)).max() < 0.01   # the same centre

# ---------------------------------------------------------------------------- the illustrative bells

BELL_A, BELL_B = 0.6, 0.8
assert np.isclose(BELL_A ** 2 + BELL_B ** 2, 1.0)                                 # "for example: 0.36 + 0.64 = 1"

# ---------------------------------------------------------------------------- look

FLAT = set("abcdefhiklmnorstuvwxzABCDEFGHIJKLMNOPRSTUVWXYZ0123456789")
RX = 3.55                     # the right column's centre


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
    """Put the baseline of Text `words` at height y."""
    return words.shift((y - baseline(words)) * m.UP)


def side(t):
    """Teal while the picture outweighs the noise (SNR > 1), violet once the noise wins."""
    return TEAL if float(snr(alpha_bar_at(t))) > 1 else VIOLET


def frame_rgb(t, x0=X0_DIGIT, eps=EPS_DIGIT):
    """x_t as the eye sees it: clipped to the model's range −1 … 1, tinted by which side is winning."""
    return tint(q_sample(x0, t, eps), side(t), -1, 1)


def texture(rgb):
    """The RGBA texture picture() makes from a small image (crisp pixels), for swapping in place."""
    k = max(1, int(np.ceil(540 / max(rgb.shape[:2]))))
    arr = np.repeat(np.repeat(rgb, k, axis=0), k, axis=1)
    rgba = np.concatenate([arr, np.full(arr.shape[:2] + (1,), 255, np.uint8)], axis=-1)
    rgba.flags.writeable = False
    return rgba


class Dissolve:
    """A digit whose pixels follow x_t = √ᾱ_t·x₀ + √(1 − ᾱ_t)·ε as t changes (its texture is swapped each frame)."""

    def __init__(self, height, t=0, x0=X0_DIGIT, eps=EPS_DIGIT):
        self.x0, self.eps = x0, eps
        self.image = picture(frame_rgb(t, x0, eps), height)

    def set_t(self, t):
        self.image.paint = self.image.paint.but(texture=texture(frame_rgb(t, self.x0, self.eps)))


class Counter(m.VGroup):
    """A step counter "t = 250" (mono, BLUE: time), left-aligned at a fixed point."""

    def __init__(self, left, y, t=0, size=44, color=TIME, prefix="t = "):
        super().__init__()
        self.anchor = np.array([left, y, 0.0])
        self.size, self.color, self.prefix = size, color, prefix
        self.words = self._make(t)
        self.add(self.words)

    def _make(self, t):
        return code(f"{self.prefix}{int(round(t))}", self.size, self.color, weight=m.BOLD).move_to(
            self.anchor, aligned_edge=m.LEFT)

    def set_t(self, t):
        self.words.become(self._make(t))


class PowerBar(m.VGroup):
    """How a noisy picture's power splits at step t: signal ᾱ_t (TEAL, from the left), noise 1 − ᾱ_t (VIOLET)."""

    def __init__(self, center, span=4.4, bar_h=0.36, labels=True, size=22):
        super().__init__()
        self.c, self.span, self.bar_h = np.array(center, dtype=float), span, bar_h
        self.sig, self.noi = self._parts(1.0)
        self.outline = m.Rectangle(width=span, height=bar_h).set_stroke(EDGE, 2).set_fill(opacity=0).move_to(self.c)
        self.add(self.sig, self.noi, self.outline)
        if labels:
            self.sig_label = text("signal", size, TEAL).next_to(self.outline, m.UP, buff=0.16).align_to(self.outline, m.LEFT)
            self.noise_label = text("noise", size, VIOLET).next_to(self.outline, m.UP, buff=0.16).align_to(self.outline, m.RIGHT)
            sit(self.noise_label, baseline(self.sig_label))
            self.add(self.sig_label, self.noise_label)

    def _parts(self, ab):
        ab = float(np.clip(ab, 0.0, 1.0))
        left = self.c[0] - self.span / 2
        ws, wn = max(self.span * ab, 0.004), max(self.span * (1 - ab), 0.004)
        sig = m.Rectangle(width=ws, height=self.bar_h).set_stroke(width=0).set_fill(TEAL, 0.8)
        noi = m.Rectangle(width=wn, height=self.bar_h).set_stroke(width=0).set_fill(VIOLET, 0.8)
        return sig.move_to([left + ws / 2, self.c[1], 0]), noi.move_to([left + self.span - wn / 2, self.c[1], 0])

    def set_t(self, t):
        s, n = self._parts(float(alpha_bar_at(t)))
        self.sig.become(s)
        self.noi.become(n)


class SwirlView(m.VGroup):
    """Swirl points drawn at center + scale·xy. The scale shrinks as the cloud spreads (the view zooms out so every
    point stays on screen); a FAINT ring of radius 1, the disc the clean swirl fits in, shows the current scale."""

    def __init__(self, xy, arms, center, scale, radius=0.028, ring=True):
        super().__init__()
        self.c, self.s = np.array(center, dtype=float), scale
        self.dots = m.VGroup(*[m.Dot(radius=radius, color=(TEAL, PALE)[int(k)]) for k in arms])
        self.ring = m.Circle(radius=scale).set_stroke(FAINT, 2.5).move_to(self.c) if ring else None
        if ring:
            self.add(self.ring)
        self.add(self.dots)
        self.put(xy, scale)

    def put(self, xy, scale=None):
        if scale is not None and abs(scale - self.s) > 1e-9:
            self.s = scale
            if self.ring is not None:
                self.ring.become(m.Circle(radius=scale).set_stroke(FAINT, 2.5).move_to(self.c))
        for d, (x, y) in zip(self.dots, np.asarray(xy, dtype=float)):
            d.move_to(self.c + self.s * np.array([x, y, 0.0]))
        return self


def drive(mob, t0, t1, hooks, run_time=1.4, rate=m.smooth):
    """Animate the step t from t0 to t1, calling every hook(t) each frame (pictures, counters, bars, clouds)."""

    def step(_, a):
        t = t0 + (t1 - t0) * a
        for hook in hooks:
            hook(t)

    return m.UpdateFromAlphaFunc(mob, step, run_time=run_time, rate_func=rate)


def bell(plot, sigma, color, width=4, opacity=1.0):
    xs = np.linspace(plot.x0, plot.x1, 241)
    ys = np.exp(-0.5 * (xs / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    return plot.line(xs, ys, color, width).set_stroke(opacity=opacity)


def formula(*parts, size=48, color=INK):
    """A formula made of separately coloured parts (kit's tex() takes one string)."""
    return m.MathTex(*parts, font_size=size, color=color)


def tagged(image, label, buff=0.16):
    """A framed picture with a label (a formula or words) under it: Group(frame, image, label)."""
    pic = framed(image)
    label.next_to(pic, m.DOWN, buff=buff)
    pic.add(label)
    return pic


def img_eq(parts, buff=0.32):
    """A row of pictures and operators (× 0.28, +, =), vertically centred."""
    return m.Group(*parts).arrange(m.RIGHT, buff=buff)


# ---------------------------------------------------------------------------- the episode


class Ep04AddingNoise(Episode):
    number = 4

    def lesson(self):
        self.hook()
        self.swirl_melts()
        self.digit_dissolves()
        self.gaussian()
        self.one_step()
        self.shortcut()
        self.schedule()
        self.signal_to_noise()
        self.recap()

    # ------------------------------------------------------------------ 0 · hook

    def hook(self):
        self.next_section("Hook", notes=(
            "Callback to Episode 2 (a picture is numbers: 28×28 = 784). The digit is the first 7 in the MNIST training set "
            f"(load_mnist('train')[{DIGIT_INDEX}], scaled to −1…1), dissolved with q_sample and DDPM's linear schedule, noise "
            f"seed {NOISE_SEED}; shown clipped to −1…1, TEAL while SNR > 1, VIOLET after. The bar is ᾱ_t (signal power) vs "
            "1 − ᾱ_t (noise power). The forward process needs no learning (Ho et al. 2020, §2)."))
        big = Dissolve(3.0)
        pic = framed(big.image).move_to([-3.1, 0.25, 0])
        cap = code("28 × 28 = 784 numbers", 22, INK).next_to(pic, m.DOWN, buff=0.3)
        ep2 = cite("Episode 2 · Images Are Numbers").next_to(cap, m.DOWN, buff=0.16)
        step_lab = text("step", 24, MUTED).move_to([RX, 1.75, 0]).align_to([RX - 2.2, 0, 0], m.LEFT)
        counter = Counter(RX - 2.2, 1.15, 0, 48)
        bar = PowerBar((RX, -0.3, 0), span=4.4)
        readout = m.VGroup(step_lab, counter, bar)

        title = title_text("the forward process", 36).move_to([RX, 1.6, 0])
        chain = tex(r"x_0 \to x_1 \to x_2 \to \cdots \to x_{1000}", 44).move_to([RX, 0.45, 0])
        chain[0][0:2].set_color(DATA)
        chain[0][-5:].set_color(NOISE)
        tags = m.VGroup(pill("no learning needed", GREEN, GREEN_FILL, 22),
                        pill("jump to any step at once", TIME, BLUE_FILL, 22)).arrange(m.DOWN, buff=0.28)
        tags.move_to([RX, -1.1, 0])

        def at(t):
            big.set_t(t)
            counter.set_t(t)
            bar.set_t(t)

        self.say(
            "[In episode two, a picture became numbers]: [seven hundred eighty-four of them, for one small digit].",
            m.AnimationGroup(self.new_header("00 · hook", "Destroying a picture on purpose"), m.FadeIn(pic, shift=0.2 * m.UP)),
            m.AnimationGroup(m.FadeIn(cap, shift=0.1 * m.UP), m.FadeIn(ep2)),
        )
        self.say(
            "Today we [destroy it on purpose]: [a little noise at a time], [until nothing but noise is left].",
            m.AnimationGroup(m.FadeOut(ep2), m.FadeIn(readout, shift=0.15 * m.LEFT)),
            drive(big.image, 0, 250, [at], run_time=1.8),
            drive(big.image, 250, 1000, [at], run_time=2.0),
        )
        self.say(
            "[This is the forward process.] [It needs no learning at all], [and we can jump to any step in one go].",
            m.AnimationGroup(m.FadeOut(readout, shift=0.15 * m.RIGHT), m.FadeIn(title, shift=0.1 * m.UP), m.Write(chain)),
            m.FadeIn(tags[0], shift=0.1 * m.UP),
            m.FadeIn(tags[1], shift=0.1 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 1 · the swirl melts

    def swirl_melts(self):
        self.next_section("1 · The swirl melts", notes=(
            "The swirl's 1,000 points are toy.json's forward-process points (the first 1,000 of swirl_labelled(20000, seed=0), "
            "noise gaussian_noise(20000, seed=1)[:1000]); x_t = q_sample(x0, t, eps) with the linear schedule, asserted equal "
            "to toy.json's snapshots at t = 0, 50, 100, 250, 500, 1000 (note: toy.md's printed recipe says swirl(1000, seed=0), "
            "but the stored snapshots are these points). The view zooms out as the cloud spreads (scale = 2.95 / max |x_t| so "
            "far, capped at 2.2 units per data unit); the faint ring is radius 1 at the current scale. Arm accuracy: the exact "
            f"posterior p(arm | x_t) on 20,000 points, toy.json: {ARM_500:.1%} at t = 500 (78% at 50, 64.3% at 100, 58.7% at "
            "250, 50.2% at 1000). The rings at the end: radius 1 and 2 of N(0, I)."))
        center = np.array([-3.2, -0.45, 0.0])
        scale = lambda t: min(2.2, 2.95 / float(np.interp(t, np.arange(T + 1), SW_EXT)))  # noqa: E731
        view = SwirlView(SW_X0, SW_ARM, center, scale(0))
        step_lab = text("step", 24, MUTED).move_to([RX, 2.0, 0]).align_to([RX - 2.2, 0, 0], m.LEFT)
        counter = Counter(RX - 2.2, 1.45, 0, 48)
        bar = PowerBar((RX, 0.35, 0), span=4.4)
        legend = m.VGroup(*[m.VGroup(m.Dot(radius=0.08, color=c), text(f"arm {k + 1}", 22, MUTED)).arrange(m.RIGHT, buff=0.15)
                            for k, c in enumerate((TEAL, PALE))]).arrange(m.RIGHT, buff=0.6)
        legend.move_to([RX, -0.6, 0])
        zoom = cite("the view zooms out as the cloud spreads;\nthe faint ring has radius 1").move_to([RX, -3.15, 0])
        readout = m.VGroup(step_lab, counter, bar, legend)

        card_box = card(5.0, 1.8, EDGE, CARD, radius=0.18, stroke=2).move_to([RX, -1.85, 0])
        who = text("a perfect classifier names the arm", 22, INK).move_to(card_box.get_top() + [0, -0.38, 0])
        pct = title_text(f"{100 * ARM_500:.1f}%", 50, INK)
        coin = code("coin flip: 50%", 20, MUTED)
        m.VGroup(pct, coin).arrange(m.RIGHT, buff=0.45).move_to(card_box.get_center() + [0, -0.28, 0])
        sit(coin, baseline(pct))
        arm_card = m.VGroup(card_box, who, pct, coin)
        noise_box = card(5.0, 1.8, NOISE, VIOLET_FILL, radius=0.18, stroke=2).move_to([RX, -1.85, 0])
        noise_tex = tex(r"\varepsilon \sim \mathcal{N}(0,\, I)", 44, INK).move_to(noise_box.get_center() + [0, 0.25, 0])
        noise_tex[0][0].set_color(NOISE)
        noise_sub = text("pure noise: a round Gaussian blob", 22, INK).next_to(noise_tex, m.DOWN, buff=0.22)
        noise_card = m.VGroup(noise_box, noise_tex, noise_sub)

        def at(t):
            view.put(q_sample(SW_X0, t, SW_EPS), scale(t))
            counter.set_t(t)
            bar.set_t(t)

        self.say(
            "[First, the swirl]: [a thousand points on two spiral arms], [inside a faint ring of radius one].",
            self.new_header("01 · the swirl", "The swirl melts"),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(d, scale=0.5) for d in view.dots], lag_ratio=0.002, run_time=1.4),
                             m.FadeIn(readout, shift=0.1 * m.LEFT)),
            m.AnimationGroup(m.Create(view.ring), m.FadeIn(zoom)),
        )
        self.say(
            "[After fifty steps, the arms are blurred]; [after a hundred, they start to touch]; "
            "[by two hundred fifty, about half the signal is gone].",
            drive(view.ring, 0, 50, [at], run_time=1.6),
            drive(view.ring, 50, 100, [at], run_time=1.4),
            drive(view.ring, 100, 250, [at], run_time=1.8),
        )
        self.say(
            "[By step five hundred, it is a smudge]: "
            "[even a perfect classifier names a point's arm just fifty-two point eight percent of the time].",
            drive(view.ring, 250, 500, [at], run_time=1.8),
            m.FadeIn(arm_card, shift=0.15 * m.UP),
        )
        rings = m.VGroup(*[m.DashedVMobject(m.Circle(radius=r * scale(1000)), num_dashes=int(28 * r)).set_stroke(VIOLET, 3)
                           .move_to(center) for r in (1, 2)])
        self.say(
            "[At step one thousand, the swirl is gone]: [a round Gaussian blob], [the shape of pure noise].",
            drive(view.ring, 500, 1000, [at], run_time=2.0),
            m.Create(rings, run_time=1.2),
            m.AnimationGroup(m.FadeOut(arm_card, shift=0.1 * m.DOWN), m.FadeIn(noise_card, shift=0.1 * m.UP)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 2 · a digit dissolves

    def digit_dissolves(self):
        self.next_section("2 · A digit dissolves", notes=(
            f"A real digit: MNIST training image {DIGIT_INDEX} (the first 7; load_mnist('train')), the digit model's data, "
            f"scaled to −1…1. x_t = q_sample(x0, t, eps), eps from default_rng({NOISE_SEED}); shown clipped to −1…1 (as a "
            "saved image would be). TEAL frames: SNR > 1 (more signal than noise power); VIOLET: SNR < 1. The crossing is at "
            f"t = {T_EVEN:.1f} (Section 7)."))
        big = Dissolve(2.8)
        pic = framed(big.image).move_to([-2.9, 0.75, 0])
        left = -0.75
        src = code(f"MNIST training image {DIGIT_INDEX} · label 7", 20, MUTED).move_to([0, 1.95, 0]).align_to([left, 0, 0], m.LEFT)
        counter = Counter(left, 1.2, 0, 48)
        px = code("28 × 28 pixels, each −1 … 1", 22, INK).move_to([0, 0.45, 0]).align_to([left, 0, 0], m.LEFT)
        frames = [framed(picture(frame_rgb(t), 1.25), label=f"t = {t}", size=20) for t in STRIP_T]
        strip_row = m.Group(*frames).arrange(m.RIGHT, buff=0.32).move_to([0, -2.45, 0])
        teal = m.VGroup(*[f[0] for f in frames[:4]])
        violet = m.VGroup(*[f[0] for f in frames[4:]])
        b1 = m.Brace(teal, direction=m.UP, color=TEAL, buff=0.1)
        b2 = m.Brace(violet, direction=m.UP, color=VIOLET, buff=0.1)
        l1 = text("more digit than noise", 22, TEAL).next_to(b1, m.UP, buff=0.12)
        l2 = text("the noise wins", 22, VIOLET).next_to(b2, m.UP, buff=0.12)
        sit(l2, baseline(l1))

        def at(t):
            big.set_t(t)
            counter.set_t(t)

        def go(k, run_time=1.2):
            return m.Succession(drive(big.image, STRIP_T[k - 1], STRIP_T[k], [at], run_time=run_time),
                                m.FadeIn(frames[k], target_position=pic, run_time=0.6))

        self.say(
            "[Now a real digit]: [the first seven in the MNIST training set], [twenty-eight by twenty-eight pixels].",
            m.AnimationGroup(self.new_header("02 · a digit", "A digit dissolves"), m.FadeIn(pic, shift=0.2 * m.UP)),
            m.AnimationGroup(m.FadeIn(src), m.FadeIn(counter)),
            m.AnimationGroup(m.FadeIn(px, shift=0.1 * m.UP), m.FadeIn(frames[0], target_position=pic)),
        )
        self.say(
            "[At step fifty, it is lightly speckled], [at a hundred, grainy], [at two hundred fifty, noisy but still readable].",
            go(1, 0.9), go(2, 0.9), go(3, 1.2),
        )
        self.say(
            "[At five hundred, it is barely there], [and at a thousand], [it is pure static].",
            go(4, 1.2), go(5, 1.2), m.Circumscribe(frames[5][0], color=VIOLET, buff=0.08),
        )
        self.say(
            "[In teal, the frames with more digit than noise]; [in violet, the frames where the noise wins].",
            m.AnimationGroup(m.GrowFromCenter(b1), m.FadeIn(l1, shift=0.1 * m.UP)),
            m.AnimationGroup(m.GrowFromCenter(b2), m.FadeIn(l2, shift=0.1 * m.UP)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 3 · Gaussian noise

    def gaussian(self):
        self.next_section("3 · Gaussian noise", notes=(
            f"The digit's own noise ε (default_rng({NOISE_SEED}).standard_normal((28, 28))): 784 independent draws from "
            f"N(0, 1). Within ±1: {WITHIN_1:.1%} (the bell's 68.3%); beyond ±3: {BEYOND_3} of 784 (expected 2.1). The "
            "curve is the standard normal density; bars are the histogram as a density (bin 0.5). On the swirl ε has two "
            "coordinates per point (SW_EPS, the swirl's own noise)."))
        noise = framed(picture(tint(EPS_DIGIT, NOISE, -1, 1), 2.8)).move_to([-4.3, 0.35, 0])
        n_cap = code("ε: 784 random numbers", 22, INK).next_to(noise, m.DOWN, buff=0.3)
        plot = Plot((-4, 4), (0, 0.55), width=6.6, height=3.4, center=(2.45, 0.1, 0), x_ticks=[-3, -2, -1, 0, 1, 2, 3],
                    x_label="pixel value")
        edges = np.arange(-4, 4.01, 0.5)
        counts, _ = np.histogram(EPS_DIGIT.ravel(), bins=edges)
        dens = counts / (EPS_DIGIT.size * 0.5)
        bars = m.VGroup()
        for lo, d in zip(edges[:-1], dens):
            a, b = plot.coords(lo + 0.03, 0), plot.coords(lo + 0.47, max(d, 0.004))
            bars.add(m.Rectangle(width=b[0] - a[0], height=b[1] - a[1]).set_stroke(NOISE, 2).set_fill(VIOLET_FILL, 1)
                     .move_to((a + b) / 2))
        curve = bell(plot, 1.0, INK, 4)
        inner = m.VGroup(*[b for lo, b in zip(edges[:-1], bars) if -1 <= lo < 1])
        in_lab = code(f"{100 * WITHIN_1:.0f}% within ±1", 22, NOISE, weight=m.BOLD).move_to(plot.coords(2.4, 0.5))
        out_lab = code(f"beyond ±3: {BEYOND_3} of 784", 20, MUTED).move_to(plot.coords(-1.75, 0.5))
        hist = m.VGroup(plot, bars, curve, in_lab, out_lab)

        cloud_c = np.array([2.45, -0.25, 0.0])
        s = 2.75 / np.abs(SW_EPS).max()
        blob = m.VGroup(*[m.Dot(cloud_c + s * np.array([x, y, 0]), radius=0.026, color=NOISE) for x, y in SW_EPS])
        b_cap = code("ε on the swirl: 2 numbers per point", 22, INK).next_to(blob, m.DOWN, buff=0.25)
        b_cap.set_x(cloud_c[0])

        self.say(
            "[That static is Gaussian noise]: [every pixel gets its own random number], [drawn from a bell curve].",
            m.AnimationGroup(self.new_header("03 · noise", "Gaussian noise"), m.FadeIn(noise, shift=0.2 * m.UP),
                             m.FadeIn(n_cap)),
            m.AnimationGroup(m.FadeIn(plot), m.LaggedStart(*[m.GrowFromEdge(b, m.DOWN) for b in bars], lag_ratio=0.08)),
            m.Create(curve, run_time=1.2),
        )
        self.say(
            "Most are small: [two thirds of them lie between minus one and one]; "
            "[only two of the seven hundred eighty-four go past plus or minus three].",
            m.AnimationGroup(inner.animate.set_fill(VIOLET, 0.55), m.FadeIn(in_lab, shift=0.1 * m.UP)),
            m.FadeIn(out_lab, shift=0.1 * m.UP),
        )
        self.say(
            "We call this noise epsilon. [On the swirl, each point gets two such numbers], [so pure noise is a round, fuzzy blob].",
            m.AnimationGroup(m.FadeOut(hist, shift=0.1 * m.DOWN),
                             m.LaggedStart(*[m.FadeIn(d, scale=0.5) for d in blob], lag_ratio=0.002, run_time=1.4)),
            m.FadeIn(b_cap, shift=0.1 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 4 · one small step

    def one_step(self):
        self.next_section("4 · One small step", notes=(
            "DDPM eq. 2: q(x_t | x_(t−1)) = N(√(1 − β_t)·x_(t−1), β_t·I), sampled as x_t = √(1 − β_t)·x_(t−1) + √β_t·ε. "
            f"At t = 250, β = {B250:.5f}: × {SHRINK_250:.4f} and + {ADD_250:.4f} × fresh noise (default_rng(5)); x_249 = "
            "q_sample(x0, 249, ε). DDPM §4: T = 1000, β linear from 1e-4 to 0.02 (linear_betas()). Shrinking keeps a "
            "unit-variance picture at unit variance: (1 − β)·1 + β = 1 ('variance preserving', Song et al. 2021)."))
        f = formula(r"x_t", "=", r"\sqrt{1-\beta_t}", r"\,x_{t-1}", "+", r"\sqrt{\beta_t}", r"\,\varepsilon", size=50)
        f.move_to([0, 1.9, 0])
        f[2].set_color(TIME)
        f[5].set_color(TIME)
        f[6].set_color(NOISE)

        prev = tagged(picture(frame_rgb(249), 1.9), tex(r"x_{249}", 34))
        fresh = tagged(picture(tint(FRESH, NOISE, -1, 1), 1.9), text("fresh noise", 22, MUTED))
        after = tagged(picture(tint(X250_STEP, side(250), -1, 1), 1.9), tex(r"x_{250}", 34))
        times1 = tex(rf"\times\,{SHRINK_250:.4f}", 40, TIME)
        times2 = tex(rf"\times\,{ADD_250:.3f}", 40, TIME)
        plus, eq = tex("+", 48), tex("=", 66)
        row = img_eq([prev, times1, plus, fresh, times2, eq, after], buff=0.3).move_to([0, -0.75, 0])
        for op in (times1, plus, times2, eq):
            op.set_y(prev[0].get_y())
        same = cite("step 250: the change is too small to see").next_to(row, m.DOWN, buff=0.3)

        sched = pill("the noise schedule: one β for every step", TIME, BLUE_FILL, 22).move_to([0, 0.75, 0])
        plot = Plot((0, 1000), (0, 0.022), width=6.8, height=2.4, center=(0, -1.75, 0), x_ticks=[0, 250, 500, 750, 1000],
                    y_ticks=[0, 0.01, 0.02], y_fmt="{:g}", x_label="step t")
        line = plot.line(np.arange(1, T + 1), BETAS[1:], TIME, 5)
        d1, d2 = plot.dot(1, BETAS[1], TIME, 0.09), plot.dot(T, BETAS[T], TIME, 0.09)
        l1 = tex(r"\beta_1 = 0.0001", 34, TIME).move_to(plot.coords(170, 0.0125))
        l1_arrow = arrow(l1.get_bottom() + [-0.3, -0.05, 0], d1.get_center() + [0.08, 0.1, 0], TIME, buff=0.05)
        l2 = tex(r"\beta_{1000} = 0.02", 34, TIME).next_to(d2, m.UP, buff=0.2).shift(0.6 * m.LEFT)
        src = cite("Ho et al., 2020 · T = 1000").move_to([4.9, -3.4, 0])

        keep = formula(r"(1-\beta_t)", "+", r"\beta_t", "=", "1", size=46).move_to([0, 0.3, 0])
        keep[0].set_color(TIME)
        keep[2].set_color(TIME)
        keep_sub = text("shrink by just enough to make room for the new noise", 24, INK).next_to(keep, m.DOWN, buff=0.3)
        bar = PowerBar((0, -1.75, 0), span=7.0)
        counter = Counter(4.0, -1.75, 0, 30)
        vp = cite("“variance preserving” · Song et al., 2021").move_to([0, -2.85, 0])

        def at(t):
            bar.set_t(t)
            counter.set_t(t)

        self.say(
            "[The forward process adds that noise in many small steps.] [Each step shrinks the picture a tiny bit], "
            "[and adds a little fresh noise].",
            m.AnimationGroup(self.new_header("04 · one step", "A little at a time"), m.Write(f)),
            m.AnimationGroup(m.Circumscribe(m.VGroup(f[2], f[3]), color=TIME, buff=0.08),
                             m.FadeIn(m.Group(prev, times1), shift=0.1 * m.UP)),
            m.AnimationGroup(m.Circumscribe(m.VGroup(f[5], f[6]), color=NOISE, buff=0.08),
                             m.FadeIn(m.Group(plus, fresh, times2, eq, after), shift=0.1 * m.UP), m.FadeIn(same)),
        )
        self.say(
            "[The size of each step is called beta t], [and the list of all the betas is the noise schedule].",
            m.AnimationGroup(m.Circumscribe(f[2], color=TIME, buff=0.06), m.Circumscribe(f[5], color=TIME, buff=0.06)),
            m.AnimationGroup(m.FadeOut(m.Group(row, same), shift=0.1 * m.DOWN), m.FadeIn(sched, shift=0.1 * m.UP)),
        )
        self.say(
            "The DDPM paper uses [a thousand steps], [with beta rising in a straight line], "
            "[from one in ten thousand to two in a hundred].",
            m.AnimationGroup(m.FadeIn(plot), m.FadeIn(src)),
            m.Create(line, run_time=1.4),
            m.AnimationGroup(m.FadeIn(d1, scale=0.5), m.FadeIn(l1), m.GrowArrow(l1_arrow), m.FadeIn(d2, scale=0.5), m.FadeIn(l2)),
        )
        self.say(
            "[Why shrink at all?] [Shrinking makes room for the new noise], [so the total stays the same as the picture fades].",
            m.AnimationGroup(m.FadeOut(m.VGroup(plot, line, d1, d2, l1, l1_arrow, l2, sched, src)), m.Circumscribe(f[2], color=TIME, buff=0.06)),
            m.AnimationGroup(m.FadeIn(keep, shift=0.1 * m.UP), m.FadeIn(keep_sub)),
            m.Succession(m.AnimationGroup(m.FadeIn(bar), m.FadeIn(counter), m.FadeIn(vp), run_time=0.5),
                         drive(bar, 0, 1000, [at], run_time=2.4)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 5 · the shortcut

    def shortcut(self):
        self.next_section("5 · Jump to any step", notes=(
            "Why one jump is allowed: each step adds independent Gaussian noise, and a sum of independent Gaussians is a "
            "Gaussian whose variance is the sum (the bells: an illustrative 0.6² + 0.8² = 1²). So q(x_t | x0) = "
            "N(√ᾱ_t·x0, (1 − ᾱ_t)·I) (DDPM eq. 4), ᾱ_t = ∏(1 − β_s) (asserted against cumprod). Check on the swirl: "
            f"{N_CHECK:,} points (swirl_labelled(…, seed=3)), 500 steps with fresh noise (seed 2) vs q_sample to t = 500 "
            f"(noise seed 7): RMS spread {spread(CH_END):.4f} vs {spread(JUMP_END):.4f} (theory {SPREAD_THEORY:.4f}); "
            "the first 1,000 points are drawn. The digit at t = 500: √ᾱ = "
            f"{SIG_500:.4f}, √(1 − ᾱ) = {NOI_500:.4f}. The code is q_sample from diffusion_reference.py, simplified "
            "(the real one interpolates fractional t and takes one t per row). Episode 5 trains with a random t per image."))
        ts = [0, 1, 2, 500]
        minis = [framed(picture(frame_rgb(t), 1.1)) for t in ts]
        labels = [tex(s, 34) for s in (r"x_0", r"x_1", r"x_2", r"x_t")]
        xs = [-4.6, -2.6, -0.6, 3.6]
        for mini, lab, x in zip(minis, labels, xs):
            mini.move_to([x, 0.55, 0])
            lab.next_to(mini, m.DOWN, buff=0.18)
        labels[0].set_color(DATA)
        dots = m.VGroup(*[m.Dot(radius=0.06, color=MUTED) for _ in range(3)]).arrange(m.RIGHT, buff=0.16).move_to([1.5, 0.55, 0])
        steps = m.VGroup(*[arrow(a[0].get_right(), b.get_left(), TIME, buff=0.12)
                           for a, b in ((minis[0], minis[1][0]), (minis[1], minis[2][0]), (minis[2], dots))],
                         arrow(dots.get_right(), minis[3][0].get_left(), TIME, buff=0.12))
        chain = m.Group(*minis, *labels, dots, steps)
        jump = m.CurvedArrow(minis[0][0].get_top() + [0, 0.12, 0], minis[3][0].get_top() + [0, 0.12, 0], angle=-0.55,
                             color=INK, stroke_width=4)
        jump_lab = text("one jump", 26, INK, weight=m.BOLD).move_to([-0.5, 2.25, 0])

        bp = Plot((-3.6, 3.6), (0, 0.72), width=5.4, height=1.8, center=(-1.6, -2.4, 0), x_ticks=[-2, 0, 2])
        bell_a, bell_b = bell(bp, BELL_A, NOISE, 3, 0.55), bell(bp, BELL_B, NOISE, 3, 0.55)
        bell_c = bell(bp, 1.0, NOISE, 6)
        add_lab = m.VGroup(code("for example:", 20, MUTED),
                           code(f"{BELL_A}² + {BELL_B}² = 1²", 24, NOISE, weight=m.BOLD),
                           text("variances add", 22, INK)).arrange(m.DOWN, buff=0.14, aligned_edge=m.LEFT)
        add_lab.move_to([3.6, -2.3, 0])

        f = formula(r"x_t", "=", r"\sqrt{\bar\alpha_t}", r"\,x_0", "+", r"\sqrt{1-\bar\alpha_t}", r"\,\varepsilon", size=66)
        f.move_to([0, 1.1, 0])
        u1 = text("some of the clean picture", 24, DATA).next_to(m.VGroup(f[2], f[3]), m.DOWN, buff=0.35)
        u2 = text("plus some noise", 24, NOISE).next_to(m.VGroup(f[5], f[6]), m.DOWN, buff=0.35)
        sit(u2, baseline(u1))
        eq4 = cite("Ho et al., 2020, eq. 4").move_to([5.0, -3.4, 0])
        prod = formula(r"\bar\alpha_t", "=", r"(1-\beta_1)(1-\beta_2)\cdots(1-\beta_t)", size=42).move_to([0, -1.15, 0])
        prod[0].set_color(TIME)
        prod[2].set_color(TIME)
        bar = PowerBar((-0.8, -2.55, 0), span=6.0)
        counter = Counter(2.6, -2.55, 0, 30)

        def at(t):
            bar.set_t(t)
            counter.set_t(t)

        cap_s, R = 1.6, 2.05
        lc, rc = np.array([-3.4, -0.3, 0.0]), np.array([3.4, -0.3, 0.0])
        alpha = np.linspace(0, 1, len(CH_PATH))
        ext_l = np.array([np.abs(p).max() for p in CH_PATH])
        ext_r = np.array([np.abs((1 - a) * CH_X0[:1000] + a * JUMP_END[:1000]).max() for a in alpha])
        ext = np.maximum.accumulate(np.maximum(ext_l, ext_r))
        zoom = lambda a: min(cap_s, R / float(np.interp(a, alpha, ext)))  # noqa: E731
        lv = SwirlView(CH_X0[:1000], CH_ARM[:1000], lc, zoom(0))
        rv = SwirlView(CH_X0[:1000], CH_ARM[:1000], rc, zoom(0))
        lt = text("500 small steps, fresh noise each", 24, INK).move_to([lc[0], 2.3, 0])
        rt = text("one jump to t = 500", 24, INK).move_to([rc[0], 2.3, 0])
        sit(rt, baseline(lt))
        lcount = Counter(lc[0] - 0.9, 1.75, 0, 26, prefix="step ")
        approx = title_text("≈", 64, INK).move_to([0, -0.3, 0])
        st_l = code(f"spread {spread(CH_END):.2f}", 24, INK, weight=m.BOLD).move_to([lc[0], -2.75, 0])
        st_r = code(f"spread {spread(JUMP_END):.2f}", 24, INK, weight=m.BOLD).move_to([rc[0], -2.75, 0])
        st_note = cite(f"{N_CHECK:,} points each · theory {SPREAD_THEORY:.2f}").move_to([0, -3.4, 0])

        def walk(_, a):
            i = a * (len(CH_PATH) - 1)
            j = int(np.floor(i))
            k = min(j + 1, len(CH_PATH) - 1)
            lv.put(CH_PATH[j] + (CH_PATH[k] - CH_PATH[j]) * (i - j), zoom(a))
            lcount.set_t(5 * i)

        def leap(_, a):
            rv.put((1 - a) * CH_X0[:1000] + a * JUMP_END[:1000], zoom(a))

        d0 = tagged(picture(tint(X0_DIGIT, DATA, -1, 1), 2.1), tex(r"x_0", 36, DATA))
        de = tagged(picture(tint(EPS_DIGIT, NOISE, -1, 1), 2.1), tex(r"\varepsilon", 36, NOISE))
        d5 = tagged(picture(frame_rgb(500), 2.1), tex(r"x_{500}", 36))
        c1 = tex(rf"\times\,{SIG_500:.2f}", 42, TIME)
        c2 = tex(rf"\times\,{NOI_500:.2f}", 42, TIME)
        p1, e1 = tex("+", 50), tex("=", 66)
        digit_row = img_eq([d0, c1, p1, de, c2, e1, d5], buff=0.3).move_to([0, -0.85, 0])
        for op in (c1, p1, c2, e1):
            op.set_y(d0[0].get_y())
        f500 = formula(r"x_{500}", "=", f"{SIG_500:.2f}", r"\,x_0", "+", f"{NOI_500:.2f}", r"\,\varepsilon", size=50)
        f500.move_to([0, 1.65, 0])
        for i, c in ((2, TIME), (3, DATA), (5, TIME), (6, NOISE)):
            f500[i].set_color(c)
        part1 = m.Group(d0, c1)
        part2 = m.Group(p1, de, c2, e1, d5)

        src = "\n".join([
            "def q_sample(x0, t, eps, alpha_bar=ALPHA_BAR):",
            '    """Jump straight to step t, no loop needed."""',
            "    ab = alpha_bar[t]",
            "    return np.sqrt(ab) * x0 + np.sqrt(1 - ab) * eps",
        ])
        block = code_block(src, 22, title="diffusion_reference.py · simplified").move_to([0, 0.75, 0])
        train = text("training picks a random step for every picture, then jumps there", 24, INK)
        train.next_to(block, m.DOWN, buff=0.45)
        ep5 = pill("Episode 5 · Learning to Undo Noise", MODEL, AMBER_FILL, 22).next_to(train, m.DOWN, buff=0.4)

        self.say(
            "[Walking a thousand steps would be slow.] [But there's a shortcut]: [one jump, straight from x zero to x t].",
            m.AnimationGroup(self.new_header("05 · the shortcut", "Jump to any step"),
                             m.LaggedStart(*[m.FadeIn(x, shift=0.1 * m.RIGHT) for x in
                                             (m.Group(minis[0], labels[0]), steps[0], m.Group(minis[1], labels[1]), steps[1],
                                              m.Group(minis[2], labels[2]), steps[2], dots, steps[3],
                                              m.Group(minis[3], labels[3]))], lag_ratio=0.25)),
            m.Wait(0.1),
            m.AnimationGroup(m.Create(jump), m.FadeIn(jump_lab, shift=0.1 * m.DOWN)),
        )
        self.say(
            "[Each step adds a little Gaussian noise], [and Gaussians add up to one Gaussian], [with their variances simply added].",
            m.AnimationGroup(m.FadeIn(bp), m.Create(bell_a), m.Create(bell_b)),
            m.Create(bell_c, run_time=1.2),
            m.FadeIn(add_lab, shift=0.1 * m.UP),
        )
        self.say(
            "So [the whole chain collapses into one line]: [some of the clean picture], [plus some noise].",
            m.AnimationGroup(m.FadeOut(m.Group(chain, jump, jump_lab, bp, bell_a, bell_b, bell_c, add_lab)),
                             m.Write(f), m.FadeIn(eq4)),
            m.AnimationGroup(f[2].animate.set_color(TIME), f[3].animate.set_color(DATA), m.FadeIn(u1, shift=0.1 * m.UP)),
            m.AnimationGroup(f[5].animate.set_color(TIME), f[6].animate.set_color(NOISE), m.FadeIn(u2, shift=0.1 * m.UP)),
        )
        self.say(
            "Here, alpha bar is [all the shrink factors multiplied together]: [the share of the signal that survives] to step t.",
            m.FadeIn(prod, shift=0.1 * m.UP),
            m.Succession(m.AnimationGroup(m.FadeIn(bar), m.FadeIn(counter), run_time=0.5),
                         drive(bar, 0, 500, [at], run_time=1.8)),
        )
        self.say(
            "[Let's check on the swirl]: [five hundred small steps with fresh noise each time], [against one jump to step five hundred].",
            m.AnimationGroup(m.FadeOut(m.VGroup(f, u1, u2, eq4, prod, bar, counter)),
                             m.FadeIn(lv), m.FadeIn(rv), m.FadeIn(lt), m.FadeIn(rt), m.FadeIn(lcount)),
            m.UpdateFromAlphaFunc(lv, walk, run_time=3.2, rate_func=m.linear),
            m.UpdateFromAlphaFunc(rv, leap, run_time=1.4, rate_func=m.smooth),
        )
        self.say(
            "[The points differ, but the clouds match]: [the same blob, with the same spread].",
            m.FadeIn(approx, scale=0.7),
            m.AnimationGroup(m.FadeIn(st_l, shift=0.1 * m.UP), m.FadeIn(st_r, shift=0.1 * m.UP), m.FadeIn(st_note)),
        )
        self.say(
            "For our digit at step five hundred, [that's zero point two eight of the picture], [plus zero point nine six of noise].",
            m.AnimationGroup(m.FadeOut(m.VGroup(lv, rv, lt, rt, lcount, approx, st_l, st_r, st_note)),
                             m.FadeIn(f500, shift=0.1 * m.UP), m.FadeIn(part1, shift=0.1 * m.UP)),
            m.FadeIn(part2, shift=0.1 * m.UP),
        )
        self.say(
            "[In code, it's one line], [so training can pick any step at random]: [that's episode five].",
            m.AnimationGroup(m.FadeOut(m.Group(f500, part1, part2)), m.FadeIn(block, shift=0.1 * m.UP)),
            m.FadeIn(train, shift=0.1 * m.UP),
            m.FadeIn(ep5, shift=0.1 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 6 · the schedule

    def schedule(self):
        self.next_section("6 · Linear or cosine", notes=(
            "ᾱ_t for DDPM's linear schedule (ALPHA_BAR) and Nichol & Dhariwal's cosine schedule (ALPHA_BAR_COSINE: "
            "f(t)/f(0), f = cos²((t/T + 0.008)/1.008 · π/2), β clipped at 0.999), both from diffusion_reference.py and "
            "asserted against toy.json's table. ᾱ_500: 0.0786 vs 0.494; ᾱ_750: 0.00335 vs 0.144; ᾱ_1000: 4.0e-5 vs "
            "2.4e-9 (with the clip; notes Appendix A). Nichol & Dhariwal 2021, §3.2: for 64×64 and 32×32 images the end of "
            "the linear schedule 'is too noisy, and so doesn't contribute very much to sample quality'; Fig. 3: 'the last "
            "quarter of the linear schedule are almost purely noise'. Our digits are 28×28."))
        plot = Plot((0, 1000), (0, 1), width=6.6, height=3.7, center=(-2.75, -0.3, 0), x_ticks=[0, 250, 500, 750, 1000],
                    y_ticks=[0, 0.5, 1], x_label="step t", y_label="signal kept, ᾱ")
        tt = np.arange(T + 1)
        lin = plot.line(tt, AB, TIME, 5)
        cos = plot.line(tt, ABC, LIGHT, 5)
        lin_lab = text("linear (DDPM)", 22, TIME).move_to(plot.coords(160, 0.12))
        cos_lab = text("cosine", 22, LIGHT).move_to(plot.coords(650, 0.62))
        cos_src = cite("cosine: Nichol & Dhariwal, 2021")
        lin_dots = m.VGroup(*[plot.dot(t, AB[t], TIME, 0.07) for t in TABLE_T])
        cos_dots = m.VGroup(*[plot.dot(t, ABC[t], LIGHT, 0.07) for t in TABLE_T])

        box = card(4.3, 3.7, EDGE, CARD, radius=0.18, stroke=2).move_to([4.45, -0.15, 0])
        cols = [box.get_left()[0] + 0.85, box.get_left()[0] + 2.2, box.get_left()[0] + 3.5]
        top = box.get_top()[1] - 0.45
        head = m.VGroup(code("t", 22, MUTED), code("linear", 22, TIME, weight=m.BOLD), code("cosine", 22, LIGHT, weight=m.BOLD))
        for h, x in zip(head, cols):
            h.move_to([x, top, 0])
        sit(head[0], baseline(head[1]))
        rule = m.Line([box.get_left()[0] + 0.3, top - 0.32, 0], [box.get_right()[0] - 0.3, top - 0.32, 0]).set_stroke(EDGE, 2)
        rows_t, rows_l, rows_c = m.VGroup(), m.VGroup(), m.VGroup()
        for i, t in enumerate(TABLE_T):
            y = top - 0.85 - 0.5 * i
            rows_t.add(code(str(t), 22, INK).move_to([cols[0], y, 0]))
            rows_l.add(code(fmt_ab(AB[t]), 22, TIME).move_to([cols[1], y, 0]))
            rows_c.add(code(fmt_ab(ABC[t]), 22, LIGHT).move_to([cols[2], y, 0]))
        cos_src.next_to(box, m.DOWN, buff=0.2)
        table = m.VGroup(box, head, rule, rows_t)
        i250, i500, i750 = TABLE_T.index(250), TABLE_T.index(500), TABLE_T.index(750)
        row750 = m.SurroundingRectangle(m.VGroup(rows_t[i750], rows_c[i750]), buff=0.12)
        row750.set_stroke(INK, 2)

        a, b = plot.coords(750, 0), plot.coords(1000, 1)
        band = m.Rectangle(width=b[0] - a[0], height=b[1] - a[1]).set_stroke(width=0).set_fill(VIOLET, 0.16).move_to((a + b) / 2)
        band.set_z_index(-1)
        band_lab = text("almost pure noise", 20, NOISE).move_to(plot.coords(875, 0.9))
        quote = cite("Nichol & Dhariwal, 2021, §3.2: “too noisy … doesn’t contribute very much”").move_to([0, -3.4, 0])

        self.say(
            "[So the schedule decides how fast the picture fades.] [Here's alpha bar for the linear schedule].",
            m.AnimationGroup(self.new_header("06 · schedule", "Linear or cosine"), m.FadeIn(plot)),
            m.AnimationGroup(m.Create(lin, run_time=1.6), m.FadeIn(lin_lab)),
        )
        self.say(
            "[About half the signal is gone by step two hundred fifty], [and by step five hundred only eight percent is left].",
            m.Succession(m.AnimationGroup(m.FadeIn(table), m.FadeIn(rows_l), m.FadeIn(lin_dots), run_time=0.9),
                         m.Circumscribe(rows_l[i250], color=TIME, buff=0.08)),
            m.Circumscribe(rows_l[i500], color=TIME, buff=0.08),
        )
        self.say(
            "A cosine schedule from twenty twenty-one [fades more evenly]: [at step five hundred it still keeps about half].",
            m.AnimationGroup(m.Create(cos, run_time=1.6), m.FadeIn(cos_lab), m.FadeIn(cos_src)),
            m.Succession(m.AnimationGroup(m.FadeIn(rows_c), m.FadeIn(cos_dots), run_time=0.8),
                         m.Circumscribe(rows_c[i500], color=LIGHT, buff=0.08)),
        )
        self.say(
            "[Side by side]: [after step seven hundred fifty, the linear schedule has almost nothing left], "
            "[while cosine still holds fourteen percent].",
            m.Create(row750),
            m.Circumscribe(rows_l[i750], color=TIME, buff=0.08),
            m.Circumscribe(rows_c[i750], color=LIGHT, buff=0.08),
        )
        self.say(
            "[On small images like ours, the last quarter of the linear steps is almost pure noise], [and adds little to quality].",
            m.AnimationGroup(m.FadeIn(band), m.FadeIn(band_lab)),
            m.FadeIn(quote, shift=0.1 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 7 · signal to noise

    def signal_to_noise(self):
        self.next_section("7 · Signal to noise", notes=(
            "SNR_t = ᾱ_t/(1 − ᾱ_t) (Kingma et al. 2021; Lin et al. 2023 eq. 5), diffusion_reference.snr, on a log axis "
            f"from t = 1 (linear {snr(AB[1]):.0f}, cosine {snr(ABC[1]):.0f}) to t = 1000 (linear {snr(AB[T]):.2e}, cosine "
            f"{snr(ABC[T]):.2e}, the clipped β). SNR = 1 (ᾱ = ½) at t = {T_EVEN:.1f} (linear) and {T_EVEN_COS:.1f} (cosine), "
            "toy.json headline 259.1 / 496.0. The digit's frames: SNR 1.10 at t = 250 (teal), 0.085 at 500 (violet). "
            "Lin et al. 2023, Table 1: 'None of the schedules has zero SNR on the last timestep'."))
        plot = Plot((0, 1000), (1e-10, 1e5), width=6.6, height=4.0, center=(-2.6, -0.35, 0), log_y=True,
                    x_ticks=[0, 250, 500, 750, 1000], x_label="step t", y_label="signal-to-noise ratio (log scale)")
        ticks = m.VGroup()
        for e in (4, 0, -4, -8):
            p = plot.coords(0, 10.0 ** e)
            lab = tex("1" if e == 0 else rf"10^{{{e}}}", 30, MUTED).next_to(p, m.LEFT, buff=0.18)
            ticks.add(m.Line(p, p + [-0.09, 0, 0]).set_stroke(FAINT, 2), lab)
        tt = np.arange(1, T + 1)
        lin = plot.line(tt, snr(AB[1:]), TIME, 5)
        cos = plot.line(tt, snr(ABC[1:]), LIGHT, 5)
        lin_lab = text("linear", 22, TIME).move_to(plot.coords(660, 10 ** -3.5))
        cos_lab = text("cosine", 22, LIGHT).move_to(plot.coords(380, 10 ** 1.75))
        start = code("≈ 10,000 at t = 1", 20, TIME).move_to(plot.coords(170, 10 ** 4.2)).shift(0.6 * m.RIGHT)

        f = formula(r"\mathrm{SNR}_t", "=", r"\frac{\bar\alpha_t}{1-\bar\alpha_t}", size=48).move_to([RX + 0.9, 1.75, 0])
        f[2].set_color(TIME)
        f_sub = text("signal power ÷ noise power", 22, MUTED).next_to(f, m.DOWN, buff=0.25)

        one = plot.hline(1, INK, 2)
        one_lab = text("signal = noise", 20, INK).move_to(plot.coords(900, 1)).shift(0.24 * m.UP)
        d_lin, d_cos = plot.dot(T_EVEN, 1, TIME, 0.09), plot.dot(T_EVEN_COS, 1, LIGHT, 0.09)
        v_lin = m.DashedLine(plot.coords(T_EVEN, 1e-10), plot.coords(T_EVEN, 1), dash_length=0.08).set_stroke(TIME, 2)
        v_cos = m.DashedLine(plot.coords(T_EVEN_COS, 1e-10), plot.coords(T_EVEN_COS, 1), dash_length=0.08).set_stroke(LIGHT, 2)
        n_lin = code(f"t ≈ {T_EVEN:.0f}", 22, TIME, weight=m.BOLD).next_to(d_lin, m.DL, buff=0.08).shift(0.05 * m.LEFT)
        n_cos = code(f"t ≈ {T_EVEN_COS:.0f}", 22, LIGHT, weight=m.BOLD).next_to(d_cos, m.UR, buff=0.08)

        m250 = tagged(picture(frame_rgb(250), 1.15), m.VGroup(code("t = 250", 20, MUTED), code(f"SNR {SNR_250:.2f}", 22, TEAL))
                      .arrange(m.DOWN, buff=0.1))
        m500 = tagged(picture(frame_rgb(500), 1.15), m.VGroup(code("t = 500", 20, MUTED), code(f"SNR {SNR_500:.3f}", 22, VIOLET))
                      .arrange(m.DOWN, buff=0.1))
        minis = m.Group(m250, m500).arrange(m.RIGHT, buff=0.6).move_to([RX + 0.9, -0.5, 0])
        d_end = plot.dot(T, snr(AB[T]), TIME, 0.09)
        end_lab = m.VGroup(code("t = 1000:", 22, INK), tex(rf"\bar\alpha = {AB_T_M} \times 10^{{{AB_T_E}}}", 34, TIME))
        end_lab.arrange(m.RIGHT, buff=0.2).move_to([RX + 0.9, -2.45, 0])
        lin23 = cite("never exactly zero · Lin et al., 2023").next_to(end_lab, m.DOWN, buff=0.2)

        self.say(
            "One ratio sums up every step: [the signal-to-noise ratio], [alpha bar over one minus alpha bar].",
            m.AnimationGroup(self.new_header("07 · SNR", "Signal to noise"), m.FadeIn(f[0])),
            m.AnimationGroup(m.FadeIn(f[1:]), m.FadeIn(f_sub)),
        )
        self.say(
            "[On a log axis], [the linear schedule falls from about ten thousand], [while cosine stays higher for longer].",
            m.AnimationGroup(m.FadeIn(plot), m.FadeIn(ticks)),
            m.AnimationGroup(m.Create(lin, run_time=1.6), m.FadeIn(lin_lab), m.FadeIn(start)),
            m.AnimationGroup(m.Create(cos, run_time=1.6), m.FadeIn(cos_lab)),
        )
        self.say(
            "[Where a curve crosses one, signal and noise are equal]: [step two hundred fifty-nine for linear], "
            "[and four hundred ninety-six for cosine].",
            m.AnimationGroup(m.Create(one), m.FadeIn(one_lab)),
            m.AnimationGroup(m.Create(v_lin), m.FadeIn(d_lin, scale=0.5), m.FadeIn(n_lin)),
            m.AnimationGroup(m.Create(v_cos), m.FadeIn(d_cos, scale=0.5), m.FadeIn(n_cos)),
        )
        self.say(
            "[That crossing is why our digit turned violet] [between step two hundred fifty and step five hundred].",
            m.FadeIn(m250, shift=0.1 * m.UP),
            m.FadeIn(m500, shift=0.1 * m.UP),
        )
        self.say(
            "[And at step one thousand, alpha bar is four in a hundred thousand]: [not exactly zero], "
            "[but noise for every practical purpose].",
            m.AnimationGroup(m.FadeIn(d_end, scale=0.5), m.FadeIn(end_lab, shift=0.1 * m.UP)),
            m.FadeIn(lin23),
            m.Circumscribe(end_lab, color=TIME, buff=0.1),
        )
        self.wait(0.5)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes=(
            "Noise (Gaussian, a fresh draw per pixel), schedule (β_t, and ᾱ_t = how much signal survives), shortcut "
            "(x_t = √ᾱ_t·x0 + √(1 − ᾱ_t)·ε: any step in one jump). Module 1 ends; Module 2 (Episodes 5–8) learns to "
            "undo the noise."))
        halo = glow(VIOLET, radius=4.0)
        words = m.VGroup(*[title_text(w, 64, c) for w, c in (("Noise", NOISE), ("Schedule", TIME), ("Shortcut", DATA))])
        subs = m.VGroup(*[code(s, 20, MUTED) for s in ("Gaussian, every pixel", "how fast signal fades", "any step, one jump")])
        words.arrange(m.RIGHT, buff=1.0)
        on_baseline(words)
        for w, s in zip(words, subs):
            s.set_x(w.get_x())
        line_y = min(w.get_bottom()[1] for w in words) - 0.45
        for s in subs:
            s.set_y(line_y)
        mid = baseline(words[0]) + 0.3
        dots = m.VGroup(*[m.Dot([(l.get_right()[0] + r.get_left()[0]) / 2, mid, 0], radius=0.07, color=MUTED)
                          for l, r in zip(words, words[1:])])
        row = m.VGroup(words, dots, subs).move_to(0.9 * m.UP)
        f = formula(r"x_t", "=", r"\sqrt{\bar\alpha_t}", r"\,x_0", "+", r"\sqrt{1-\bar\alpha_t}", r"\,\varepsilon", size=44)
        for i, c in ((2, TIME), (3, DATA), (5, TIME), (6, NOISE)):
            f[i].set_color(c)
        f.next_to(row, m.DOWN, buff=0.6)
        done = pill("MODULE 1 · PICTURES FROM NOISE", VIOLET, VIOLET_FILL, 20, mono=True)
        tag = text("Destroying a picture is easy. Undoing it is the hard part.", 28, MUTED)
        nxt = pill("MODULE 2 · LEARNING TO DENOISE", MODEL, AMBER_FILL, 20, mono=True)
        mascot = pixel(0.42)
        foot = m.VGroup(done, tag, m.VGroup(mascot, nxt).arrange(m.RIGHT, buff=0.3)).arrange(m.DOWN, buff=0.3)
        foot.next_to(f, m.DOWN, buff=0.5)

        self.say(
            "[To sum up]: [noise], [schedule], [shortcut].",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.3 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(dots[0]), m.FadeIn(words[1], shift=0.3 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(dots[1]), m.FadeIn(words[2], shift=0.3 * m.UP), m.FadeIn(subs[2]), m.FadeIn(f)),
        )
        self.say(
            "[That closes Module one]: [destroying a picture is easy], [and needs no learning at all].",
            m.FadeIn(done, shift=0.1 * m.UP),
            m.Write(tag),
            m.Wait(0.1),
        )
        self.say(
            "[Undoing it is the hard part], [and that is what a network learns, starting next time].",
            m.Circumscribe(tag, color=MODEL, buff=0.1),
            m.AnimationGroup(m.FadeIn(mascot, shift=0.2 * m.UP), m.FadeIn(nxt, shift=0.2 * m.UP)),
        )
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, f, foot)], run_time=0.9)
