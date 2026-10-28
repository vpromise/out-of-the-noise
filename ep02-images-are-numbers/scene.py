"""Out of the Noise · Episode 2 · Images Are Numbers (Module 1 · Pictures from Noise)

A picture is a point in a space with millions of dimensions, and real pictures are rare. Running example: the digit
model's training data (MNIST, 60,000 training digits, load_mnist("train")); the swirl as the 2D stand-in.

Sections: pixels (training digit #15, a seven: 28 × 28 grey levels 0 … 255, rescaled to −1 … 1) · unrolled (the 28
rows laid end to end: one column of 784 numbers, x0.reshape(784)) · channels (red, green, blue; the lighthouse's
512 × 512 × 3 = 786,432 numbers ≈ 1,003× the digit; tensors; the digit model's batch of 256) · the space (two pixels
→ a point on a plane; 784 axes; every point is a picture) · random points (noise_image static; 256^784 ≈ 10^1888;
1,000 random pictures vs 1,000 test digits, nearest of the 60,000 training digits) · the thin set (the swirl; uniform
points near it: 1 in 3 in 2D, 1 in 46 in 3D, 1 in 200 billion in 10D; the manifold idea; 7–13 dimensions for MNIST;
the data distribution) · sampling (only examples; the average of 6,265 training sevens is a blur; the swirl's average
sits in the empty middle) · the way out (noise walking onto the swirl: DDIM, 50 steps, exact denoiser) · recap.

Sources:
- diffusion_reference.py: load_mnist, swirl, swirl_curve, nearest_on_swirl, mixture_density, gaussian_noise,
  exact_eps, ddim_sample (code excerpts use its names).
- research/experiments/logs/train_ddpm_config.json (the digit model's run on this Mac: images_per_epoch 60,000,
  img 28, batch 256). mac_runs.md / measured("mac_runs", …) did not exist yet when this episode was made, so the
  run's config file is read directly; the lighthouse's images are not shown (its run was not finished), only its
  output size 512 × 512 × 3.
- research/notes.md §2: MNIST 60,000 train / 10,000 test, 28 × 28 (LeCun, Cortes & Burges, MNIST page); licence per
  Appendix A ("freely available; CC BY-SA 3.0 per Keras"); pixel scaling to [−1, 1] (Ho et al., 2020, §3.3); 256^784
  = 10^1888.1 (worked example 2); the manifold hypothesis (Song & Ermon, 2019, §3.1, quoted); MNIST's intrinsic
  dimension 7–13 (Pope et al., 2021, Table 1, MLE with k = 3 … 20). Appendix C: 784; 60,000 / 10,000; 786,432.

Computed in this file (seeded, asserted): every count above; the nearest-training-digit distances (Euclidean, pixels
in [−1, 1]) of the first 1,000 test digits and of noise_image(seed=0 … 999) clipped to [−1, 1]; the share of uniform
random points in [−1.2, 1.2]^D within 0.1 of the swirl's centre line (the curve lies in a plane; 200,000 seeded points
in the plane, the extra D − 2 coordinates integrated exactly: the ball of radius √(0.1² − d²) over the cube's side);
the mean training seven; the swirl's mean; a DDIM trajectory with the exact denoiser (preview of Episodes 6–9).

Illustrative: the faint 8 × 8 grids drawn on the colour channels (a real channel has 512 × 512 pixels), the star of
axes standing for 784 dimensions, the path the point takes on the two-pixel plane, the five-picture stack standing for
a batch of 256.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the shared narrator serving
"""

import json
import sys
from math import gamma, pi
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403
from diffusion_reference import *  # noqa: E402,F403

# ---------------------------------------------------------------------------- the numbers

IMAGES, LABELS = load_mnist("train")
TEST_IMAGES, _TEST_LABELS = load_mnist("test")
N_TRAIN, N_TEST = len(IMAGES), len(TEST_IMAGES)
assert N_TRAIN == 60_000 and N_TEST == 10_000                     # "sixty thousand", "ten thousand more"
SIDE = IMAGES.shape[1]
N_PIX = SIDE * SIDE
assert IMAGES.shape[1:] == (28, 28) and N_PIX == 784               # "twenty-eight by twenty-eight", "784"

RUN = json.loads((EXPERIMENTS / "logs" / "train_ddpm_config.json").read_text())
assert RUN["images_per_epoch"] == N_TRAIN and RUN["model"]["img"] == SIDE   # the digit model trains on these
BATCH = RUN["batch"]
assert BATCH == 256                                                 # "two hundred and fifty-six digits at a time"

SEVEN = 15                                                          # training digit #15 is a bold seven
assert LABELS[SEVEN] == 7
DIGIT = IMAGES[SEVEN]
assert DIGIT.min() == 0 and DIGIT.max() == 255                      # "zero … up to two hundred and fifty-five"
X0 = DIGIT / 127.5 - 1                                              # load_mnist's own recipe: x / 127.5 - 1
assert X0.min() == -1 and X0.max() == 1                             # "from minus one to one"
N_DARK, N_INK = int((DIGIT == 0).sum()), int((DIGIT > 0).sum())
assert (N_DARK, N_INK) == (640, 144) and N_DARK + N_INK == N_PIX
WIN_R, WIN_C, WIN_N = 16, 12, 5                                     # the zoomed 5 × 5 window
WINDOW = DIGIT[WIN_R:WIN_R + WIN_N, WIN_C:WIN_C + WIN_N]
ZERO_AT = tuple(int(v) for v in np.argwhere(WINDOW == 0)[0])
FULL_AT = tuple(int(v) for v in np.argwhere(WINDOW == 255)[0])
FLAT = X0.reshape(N_PIX)                                            # row by row: x0[28·r + c] = pixel (r, c)
assert all(FLAT[SIDE * r + c] == X0[r, c] for r in (0, 9, 27) for c in (0, 14, 27))
SLICE_ROW, SLICE_C0 = 11, 14                                        # x0[322:328] runs into the stroke
SLICE = FLAT[SIDE * SLICE_ROW + SLICE_C0: SIDE * SLICE_ROW + SLICE_C0 + 6]
assert SLICE.min() == -1 and SLICE.max() > 0.9
BRIGHT_ROWS = (9, 20)                                               # a wide stroke row and a thin one
assert (DIGIT[9] > 128).sum() == 13 and (DIGIT[20] > 128).sum() == 3

LIGHTHOUSE = (512, 512, 3)
N_LH = int(np.prod(LIGHTHOUSE))
assert N_LH == 786_432                                              # "seven hundred and eighty-six thousand …"
RATIO = N_LH / N_PIX
assert round(RATIO) == 1003                                         # "about a thousand times the digit"
LARGE = (1024, 1024, 3)
N_LARGE = int(np.prod(LARGE))
assert N_LARGE == 3_145_728 and N_LARGE > 3_000_000                 # "over three million"

LOG10_GRIDS = N_PIX * np.log10(256)
DIGITS_LONG = int(np.floor(LOG10_GRIDS)) + 1
assert int(LOG10_GRIDS) == 1888 and DIGITS_LONG == 1889             # "nearly nineteen hundred digits long"

TRAIN_FLAT = IMAGES.reshape(N_TRAIN, N_PIX).astype(np.float32) / 127.5 - 1


def nearest_train_distance(queries):
    """Euclidean distance (pixels in [−1, 1]) from each query picture to its nearest of the 60,000 training digits."""
    q = np.asarray(queries, np.float32).reshape(len(queries), N_PIX)
    tt = (TRAIN_FLAT ** 2).sum(1)
    out = np.empty(len(q), np.float64)
    for i in range(0, len(q), 500):
        part = q[i:i + 500]
        d2 = (part ** 2).sum(1, keepdims=True) - 2 * part @ TRAIN_FLAT.T + tt
        out[i:i + 500] = np.sqrt(np.maximum(d2.min(1), 0))
    return out


N_CHECK = 1000
RANDOM_PICS = np.clip(np.stack([noise_image(seed=s) for s in range(N_CHECK)]), -1, 1)   # what the pictures show
D_RANDOM = nearest_train_distance(RANDOM_PICS)
D_TEST = nearest_train_distance(TEST_IMAGES[:N_CHECK].astype(np.float32) / 127.5 - 1)
GAP = D_RANDOM.min() / D_TEST.max()
assert D_RANDOM.min() > 2 * D_TEST.max()                            # "more than twice as far"
assert np.median(D_TEST) < 10 < 28 < D_RANDOM.min()

# the swirl, and uniform random points around it
SQ, NEAR = 1.2, 0.1                                                 # the square [−1.2, 1.2]², "near" = within 0.1
_U = np.random.default_rng(1).uniform(-SQ, SQ, (200_000, 2))
_D2 = nearest_on_swirl(_U)[0]


def tube_share(dims):
    """Share of uniform points in [−SQ, SQ]^dims within NEAR of the swirl's centre line (which lies in the first two
    coordinates): for a plane point at distance d, the other dims − 2 coordinates must fall in a ball of radius
    √(NEAR² − d²) — exact, as that ball fits inside the cube."""
    k = dims - 2
    rho = np.sqrt(np.maximum(NEAR ** 2 - _D2 ** 2, 0.0)) * (_D2 < NEAR)
    ball = pi ** (k / 2) / gamma(k / 2 + 1) * rho ** k if k else (_D2 < NEAR).astype(float)
    return float(np.mean(ball / (2 * SQ) ** k))


DIMS = (2, 3, 4, 5, 10)
ODDS = {k: 1 / tube_share(k) for k in DIMS}
assert round(ODDS[2]) == 3 and round(ODDS[3]) == 46                 # "one in three", "one in forty-six"
assert round(ODDS[4], -1) == 840 and round(ODDS[5], -3) == 17_000
assert 1.95e11 < ODDS[10] < 2.05e11                                 # "one in two hundred billion"
UNIFORM = _U[:2000]
NEAR_MASK = _D2[:2000] < NEAR
assert round(1 / NEAR_MASK.mean()) == 3                             # the 2,000 shown agree: about one in three

SWIRL_PTS = swirl(1500, seed=0)
SWIRL_MEAN = SWIRL_PTS.mean(0)
MEAN_GAP = float(nearest_on_swirl(SWIRL_MEAN[None])[0][0])
assert np.abs(SWIRL_MEAN).max() < 0.02 and 0.13 < MEAN_GAP < 0.16 and MEAN_GAP > NEAR  # "in the middle", "no data"

SEVENS = IMAGES[LABELS == 7]
N_SEVENS = len(SEVENS)
assert N_SEVENS == 6265 and N_SEVENS > 6000                         # "over six thousand training sevens"
MEAN_SEVEN = SEVENS.mean(0)

_Z = gaussian_noise(900, seed=5)
Z_START = _Z[np.abs(_Z).max(1) < 1.9][:500]                         # the noise points that fit the frame
assert len(Z_START) == 500
PATH = ddim_sample(exact_eps, Z_START, steps=50).xs                 # (51, 500, 2): noise → the swirl
assert PATH.shape == (51, 500, 2) and swirl_distance(PATH[-1]) < 0.05

# ---------------------------------------------------------------------------- look


def cite(s, size=20):
    return text(s, size, MUTED)


def hexrgb(rgb):
    return "#%02X%02X%02X" % tuple(int(v) for v in rgb)


def shade(value):
    """The TEAL tint of a pixel value in [−1, 1] (black → BG, full ink → TEAL), as tint() draws it."""
    return mix(BG, DATA, (float(value) + 1) / 2)


def pixel_squares(img, height, color=DATA, lo=None, hi=None):
    """A grey image as rows of filled squares (grid[r][c]), looking exactly like picture(tint(img)) but animatable."""
    rgb = tint(img, color, lo, hi)
    n_r, n_c = img.shape
    s = height / n_r
    grid = m.VGroup()
    for r in range(n_r):
        grid.add(m.VGroup(*[
            m.Square(s * 1.03).set_stroke(width=0).set_fill(hexrgb(rgb[r, c]), 1)
            .move_to([(c - (n_c - 1) / 2) * s, ((n_r - 1) / 2 - r) * s, 0]) for c in range(n_c)]))
    return grid


def boxed(grid, color=EDGE, buff=0.06, stroke=2.5):
    """A pixel_squares digit in a thin frame, like framed(picture): VGroup(frame, grid); grid keeps grid[r][c]."""
    frame = m.Rectangle(width=grid.width + 2 * buff, height=grid.height + 2 * buff).set_stroke(color, stroke)
    return m.VGroup(frame.move_to(grid), grid)


def grid_lines(target, n, color=EDGE, width=1.2):
    """n × n cell lines over a square mobject."""
    x0, x1 = target.get_left()[0], target.get_right()[0]
    y0, y1 = target.get_bottom()[1], target.get_top()[1]
    lines = m.VGroup()
    for i in range(n + 1):
        x = x0 + (x1 - x0) * i / n
        y = y0 + (y1 - y0) * i / n
        lines.add(m.Line([x, y0, 0], [x, y1, 0]).set_stroke(color, width))
        lines.add(m.Line([x0, y, 0], [x1, y, 0]).set_stroke(color, width))
    return lines


def number_cell(value, size, scaled=False):
    """One zoomed pixel: a square in the pixel's tint with its number (integer or rescaled)."""
    v = value / 127.5 - 1
    box = m.Square(size).set_stroke(EDGE, 1.5).set_fill(shade(v), 1)
    label = f"{v:.2f}".replace("-", "−") if scaled else f"{value}"
    words = code(label, 22, BG if value > 140 else INK).move_to(box)
    return m.VGroup(box, words)


def recolor_dots(cloud, c0, o0, colors, opacities, run_time=1.0):
    """Every dot of a Cloud from colour c0 / opacity o0 to its own colour and opacity."""

    def step(_, a):
        for d, c1, o1 in zip(cloud, colors, opacities):
            d.set_fill(mix(c0, c1, a), o0 + (o1 - o0) * a)

    return m.UpdateFromAlphaFunc(cloud, step, run_time=run_time)


def hist_bars(plot, values, color, lo=0.0, hi=36.0, width=1.0):
    """A histogram of values (share per bin of `width`) as bars on a Plot."""
    edges = np.arange(lo, hi + width, width)
    counts = np.histogram(values, edges)[0] / len(values)
    bars = m.VGroup()
    for left, c in zip(edges[:-1], counts):
        if c <= 0:
            continue
        p0, p1 = plot.coords(left, 0), plot.coords(left + width, c)
        rect = m.Rectangle(width=p1[0] - p0[0], height=p1[1] - p0[1]).set_stroke(color, 1.5)
        bars.add(rect.set_fill(FILLS[color], 1).move_to((p0 + p1) / 2))
    return bars


def density_image(plot, lo=-1.25, hi=1.25, n=260, alpha=210):
    """The swirl's data density p(x) on a grid, as a TEAL glow with transparency (computed by mixture_density)."""
    xs = np.linspace(lo, hi, n)
    gx, gy = np.meshgrid(xs, xs[::-1])
    dens = mixture_density(np.stack([gx.ravel(), gy.ravel()], 1), 0).reshape(n, n)
    level = np.sqrt(dens / dens.max())
    rgb = np.array([int(DATA[i:i + 2], 16) for i in (1, 3, 5)], np.uint8)
    rgba = np.zeros((n, n, 4), np.uint8)
    rgba[..., :3] = rgb
    rgba[..., 3] = (level * alpha).round().astype(np.uint8)
    image = m.ImageMobject(rgba)
    image.height = plot.coords(0, hi)[1] - plot.coords(0, lo)[1]
    return image.move_to(plot.coords(0, 0))


class Ep02ImagesAreNumbers(Episode):
    number = 2

    def lesson(self):
        self.pixels()
        self.unrolled()
        self.channels()
        self.space()
        self.random_points()
        self.thin_set()
        self.sampling()
        self.way_out()
        self.recap()

    def move_digit(self, height, center, run_time=1.4):
        """The seven (digit_box: frame + squares) to a new size and place, its frame staying 0.06 outside."""
        frame, grid = self.digit_box
        target = m.Rectangle(width=height + 0.12, height=height + 0.12).set_stroke(EDGE, 2.5).move_to(center)
        return m.AnimationGroup(grid.animate(run_time=run_time).scale(height / grid.height).move_to(center),
                                m.Transform(frame, target, run_time=run_time))

    # ------------------------------------------------------------------ 1 · pixels

    def pixels(self):
        self.next_section("1 · Pixels", notes=(
            "The digit model's training data: MNIST, 60,000 training digits + 10,000 test digits, 28 × 28 grey "
            "(LeCun, Cortes & Burges, MNIST page; notes §2). Licence: sources disagree (notes Appendix A) → 'freely "
            "available; CC BY-SA 3.0 per Keras'. The wall shows training digits 0-39; the zoomed one is #15, a seven. "
            "The 5 × 5 window is rows 16-20, columns 12-16 of that digit, its real values. Rescaling x = k / 127.5 − 1 "
            "is load_mnist's recipe and DDPM's convention (Ho et al., 2020, §3.3)."))
        dh = 0.95
        wall = m.Group()
        for i in range(40):
            if i == SEVEN:
                self.digit = pixel_squares(DIGIT, dh)
                self.digit_box = boxed(self.digit)
                wall.add(self.digit_box)
            else:
                wall.add(framed(picture(tint(IMAGES[i], DATA), dh), EDGE))
        wall.arrange_in_grid(rows=4, cols=10, buff=0.12).move_to([0, 0.3, 0])
        others = m.Group(*[p for i, p in enumerate(wall) if i != SEVEN])
        cap_a = code(f"MNIST training set · {N_TRAIN:,} digits", 22, DATA)
        cap_b = code(f"+ {N_TEST:,} kept for testing", 22, MUTED)
        caps = m.VGroup(cap_a, cap_b).arrange(m.RIGHT, buff=0.5).move_to([0, -2.45, 0])
        src = cite("LeCun, Cortes & Burges · MNIST · freely available, CC BY-SA 3.0 per Keras").move_to([0, -3.2, 0])

        big_h, big_c = 4.9, np.array([-3.75, -0.25, 0])
        s = big_h / SIDE
        lines = grid_lines(m.Square(big_h).move_to(big_c), SIDE)
        dim_top = code("28", 22, MUTED).next_to(m.Square(big_h + 0.12).move_to(big_c), m.UP, buff=0.15)
        dim_left = code("28", 22, MUTED).next_to(m.Square(big_h + 0.12).move_to(big_c), m.LEFT, buff=0.15)
        count = code("28 × 28 = 784 pixels", 22, MUTED).move_to([big_c[0], -3.25, 0])

        cell = 1.0
        win_c = big_c + np.array([(WIN_C + WIN_N / 2 - SIDE / 2) * s, (SIDE / 2 - WIN_R - WIN_N / 2) * s, 0])
        window = m.Square(WIN_N * s).set_stroke(INK, 3).move_to(win_c)
        panel_c = np.array([3.5, -0.2, 0])
        ints, scaled = m.VGroup(), m.VGroup()
        for r in range(WIN_N):
            for c in range(WIN_N):
                pos = panel_c + np.array([(c - (WIN_N - 1) / 2) * cell, ((WIN_N - 1) / 2 - r) * cell, 0])
                ints.add(number_cell(int(WINDOW[r, c]), cell).move_to(pos))
                scaled.add(number_cell(int(WINDOW[r, c]), cell, scaled=True)[1].move_to(pos))
        frame = m.Square(WIN_N * cell).set_stroke(INK, 3).move_to(panel_c)
        zoom = m.VGroup(
            m.Line(window.get_corner(m.UR), frame.get_corner(m.UL)).set_stroke(MUTED, 1.5),
            m.Line(window.get_corner(m.DR), frame.get_corner(m.DL)).set_stroke(MUTED, 1.5),
        )
        zero_cell = ints[ZERO_AT[0] * WIN_N + ZERO_AT[1]]
        full_cell = ints[FULL_AT[0] * WIN_N + FULL_AT[1]]
        rule = code("x = k / 127.5 − 1", 22, INK, t2c={r"k": MUTED, r"x": DATA})
        rule_src = cite("Ho et al., 2020")
        rule_row = m.VGroup(rule, rule_src).arrange(m.RIGHT, buff=0.4).move_to([panel_c[0], -3.25, 0])
        rule_src.align_to(rule, m.DOWN)
        self.pix = dict(lines=lines, dims=m.VGroup(dim_top, dim_left), count=count, window=window, zoom=zoom,
                        ints=ints, frame=frame, rule=rule_row)

        self.say(
            "[Our running example today is the digit model's training data]: [sixty thousand handwritten digits.]",
            m.AnimationGroup(self.new_header("01 · pixels", "A picture is a grid of numbers"),
                             m.LaggedStart(*[m.FadeIn(p, shift=0.1 * m.UP) for p in wall], lag_ratio=0.04)),
            m.FadeIn(cap_a, shift=0.1 * m.UP),
        )
        self.say(
            "[They come from the MNIST dataset], [with ten thousand more kept aside for testing.]",
            m.FadeIn(src),
            m.FadeIn(cap_b, shift=0.1 * m.UP),
        )
        self.say(
            "[Pick one seven], [and zoom in.]",
            m.Circumscribe(self.digit_box, color=DATA, buff=0.06),
            m.AnimationGroup(m.FadeOut(others), m.FadeOut(caps), m.FadeOut(src),
                             self.move_digit(big_h, big_c)),
        )
        self.say(
            "[It's a grid of twenty-eight by twenty-eight little squares], [called pixels.]",
            m.AnimationGroup(m.LaggedStart(*[m.Create(ln) for ln in lines], lag_ratio=0.02, run_time=1.6),
                             m.FadeIn(m.VGroup(dim_top, dim_left))),
            m.FadeIn(count, shift=0.1 * m.UP),
        )
        self.say(
            "[Each pixel holds one number]: [zero for the black background], [up to two hundred and fifty-five for full ink.]",
            m.AnimationGroup(m.Create(window), m.Create(zoom), m.FadeIn(ints), m.Create(frame)),
            m.Circumscribe(zero_cell, color=INK, buff=0.04),
            m.Circumscribe(full_cell, color=DATA, buff=0.04),
        )
        self.say(
            "[Diffusion models rescale every number], [so it runs from minus one to one.]",
            m.AnimationGroup(*[m.Transform(ints[i][1], scaled[i]) for i in range(len(ints))], run_time=1.2),
            m.FadeIn(rule_row, shift=0.1 * m.UP),
        )
        self.wait(0.3)

    # ------------------------------------------------------------------ 2 · unrolled

    def unrolled(self):
        self.next_section("2 · Unrolled", notes=(
            "The same seven, its 28 rows laid end to end (row-major, as numpy's reshape does): x0[28·r + c] is pixel "
            "(r, c). The column is drawn to scale: 784 cells of 1/784 of its height each. The readout x0[322:328] is "
            "row 11, columns 14-19 of the real digit. Code: an excerpt in diffusion_reference.py's names "
            "(load_mnist's docstring gives the x / 127.5 - 1 recipe)."))
        p = self.pix
        small_h, small_c = 2.7, np.array([-5.2, -0.35, 0])
        self.play(m.FadeOut(m.VGroup(p["lines"], p["dims"], p["count"], p["window"], p["zoom"], p["ints"],
                                     p["frame"], p["rule"])),
                  self.move_digit(small_h, small_c, run_time=0.9))
        digit = self.digit
        copy = digit.copy()
        self.add(copy)

        col_x, col_top, col_h, col_w = -2.55, 2.45, 5.8, 0.42
        cell_h = col_h / N_PIX
        column = m.VGroup()
        for r in range(SIDE):
            seg = m.VGroup()
            for c in range(SIDE):
                i = SIDE * r + c
                rect = m.Rectangle(width=col_w, height=cell_h * 1.08).set_stroke(width=0)
                seg.add(rect.set_fill(shade(FLAT[i]), 1).move_to([col_x, col_top - (i + 0.5) * cell_h, 0]))
            column.add(seg)
        col_frame = m.Rectangle(width=col_w + 0.08, height=col_h + 0.08).set_stroke(EDGE, 2)
        col_frame.move_to([col_x, col_top - col_h / 2, 0])
        first = code("x0[0]", 20, MUTED).next_to(col_frame.get_corner(m.UL), m.LEFT, buff=0.15).shift(0.12 * m.DOWN)
        last = code("x0[783]", 20, MUTED).next_to(col_frame.get_corner(m.DL), m.LEFT, buff=0.15).shift(0.12 * m.UP)
        brace = m.Brace(col_frame, m.RIGHT, buff=0.12).set_color(DATA)
        big = title_text(f"{N_PIX}", 72, DATA)
        big_sub = text("numbers", 24, MUTED)
        big_lab = m.VGroup(big, big_sub).arrange(m.DOWN, buff=0.1).next_to(brace, m.RIGHT, buff=0.25)
        times = code(f"{SIDE} × {SIDE}", 22, MUTED).next_to(big_lab, m.DOWN, buff=0.25)

        block = code_block(
            'images, labels = load_mnist("train")\n'
            f"x0 = images[{SEVEN}] / 127.5 - 1  # −1 … 1\n"
            f"x0 = x0.reshape({N_PIX})          # one column",
            20, title="diffusion_reference.py", accent=DATA)
        block.move_to([0, 1.45, 0]).align_to([6.5, 0, 0], m.RIGHT)
        a = SIDE * SLICE_ROW + SLICE_C0
        vals = " ".join(f"{v:.2f}".replace("-", "−") for v in SLICE)
        readout = m.VGroup(code(f"x0[{a}:{a + 6}]", 20, MUTED), code(vals, 20, INK))
        readout.arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.15).move_to([0, -1.75, 0]).align_to([6.5, 0, 0], m.RIGHT)
        readout_note = cite(f"row {SLICE_ROW}, running into the stroke").next_to(readout, m.DOWN, buff=0.2)
        readout_note.align_to(readout, m.LEFT)

        def carried(r, run_time):
            """Row r flies into its segment, with an outline travelling along (the top rows are all background)."""
            ring = m.SurroundingRectangle(copy[r], buff=0.02).set_stroke(INK, 2.5)
            goal = m.SurroundingRectangle(column[r], buff=0.02).set_stroke(INK, 2.5)
            return m.AnimationGroup(
                m.ReplacementTransform(copy[r], column[r], run_time=run_time),
                m.Succession(m.FadeIn(ring, run_time=0.2), m.Transform(ring, goal, run_time=run_time - 0.4),
                             m.FadeOut(ring, run_time=0.2)))

        self.say(
            "[Now take the top row], [then the next], [and lay all twenty-eight rows end to end.]",
            m.AnimationGroup(self.new_header("02 · unrolled", "784 numbers in a column"), m.FadeIn(col_frame),
                             carried(0, 1.2)),
            carried(1, 1.0),
            m.LaggedStart(*[m.ReplacementTransform(copy[r], column[r]) for r in range(2, SIDE)], lag_ratio=0.12,
                          run_time=3.4),
        )
        self.say(
            "[That makes one column of seven hundred and eighty-four numbers]: [twenty-eight times twenty-eight.]",
            m.AnimationGroup(m.GrowFromCenter(brace), m.FadeIn(big_lab, shift=0.1 * m.LEFT), m.FadeIn(first), m.FadeIn(last)),
            m.FadeIn(times, shift=0.1 * m.UP),
        )
        r1, r2 = BRIGHT_ROWS
        self.say(
            "[The bright bands are the pen strokes], [sliced row by row.]",
            m.AnimationGroup(m.Circumscribe(column[r1], color=INK, buff=0.05),
                             m.Circumscribe(digit[r1], color=INK, buff=0.03)),
            m.AnimationGroup(m.Circumscribe(column[r2], color=INK, buff=0.05),
                             m.Circumscribe(digit[r2], color=INK, buff=0.03)),
        )
        self.say(
            "[In code, it's one reshape], [and this column is all the model ever sees.]",
            m.FadeIn(block, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(readout, shift=0.1 * m.UP), m.FadeIn(readout_note)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 3 · channels

    def channels(self):
        self.next_section("3 · Channels", notes=(
            "Colour = three channels (red, green, blue), each a grid; the planes use the literal channel hues, and "
            "their 8 × 8 grids are illustrative. The lighthouse (the course's text-to-image run on this Mac, Episode 1) "
            "is 512 × 512 × 3 = 786,432 numbers = 1,003 × the digit's 784 (notes §2 table, Appendix C). Its images are "
            "not shown here. Tensors: shapes in the digit model's layout (MLX, NHWC: research/experiments/gpu/"
            "common.py); its batch of 256 from research/experiments/logs/train_ddpm_config.json."))
        side, c0 = 2.5, np.array([-3.9, 0.55, 0])
        off = np.array([0.42, 0.34, 0])
        hues = ((RED, RED_FILL, "R"), (GREEN, GREEN_FILL, "G"), (BLUE, BLUE_FILL, "B"))
        planes = []
        for k, (col, fill, letter) in enumerate(hues):
            sq = m.Square(side).set_stroke(col, 2.5).set_fill(fill, 1)
            g = grid_lines(sq, 8, color=mix(fill, col, 0.35), width=1)
            planes.append(m.VGroup(sq, g).move_to(c0 + (1 - k) * off * -1))
        stack = m.VGroup(*planes[::-1])                     # blue at the back, red in front
        spread_xs = (-5.45, -3.45, -1.45)
        spread_y = 0.75
        names = m.VGroup(*[text(n, 22, col) for n, (col, _, _) in zip(("red", "green", "blue"), hues)])
        for nm, x in zip(names, spread_xs):
            nm.move_to([x, spread_y - 1.15, 0])
        chan = pill("each one: a channel", MUTED, CARD, 20)
        chan.move_to([spread_xs[1], spread_y - 1.75, 0])

        dims = m.VGroup(*[code("512 × 512", 20, MUTED).move_to([x, spread_y + 1.12, 0]) for x in spread_xs])

        def row(name, shape, total, color):
            return m.VGroup(text(name, 24, color), code(shape, 22, INK), code(f"= {total}", 22, color, weight=m.BOLD))

        r_digit = row("digit", "28 × 28 × 1", f"{N_PIX:,}", DATA)
        r_light = row("lighthouse", "512 × 512 × 3", f"{N_LH:,}", INK)
        xs = (0.6, 2.3, 6.5)
        for rw, y in ((r_digit, 2.05), (r_light, 1.4)):
            rw[0].move_to([0, y, 0]).align_to([xs[0], 0, 0], m.LEFT)
            rw[1].move_to([0, y, 0]).align_to([xs[1], 0, 0], m.LEFT)
            rw[2].move_to([0, y, 0]).align_to([xs[2], 0, 0], m.RIGHT)
            rw[2].align_to(rw[1], m.DOWN)
        ratio = text(f"≈ {RATIO:,.0f} × as many numbers as the digit", 22, MUTED)
        ratio.move_to([0, 0.75, 0]).align_to([xs[2], 0, 0], m.RIGHT)

        def shape_chip(s, note, color):
            box_words = code(s, 24, INK)
            box = card(box_words.width + 0.5, 0.62, color, FILLS.get(color, CARD), radius=0.12, stroke=2)
            return m.VGroup(m.VGroup(box, box_words.move_to(box)), text(note, 20, MUTED)).arrange(m.DOWN, buff=0.12)

        t_title = text("tensors: grids with any number of axes", 24, INK).move_to([3.65, -0.25, 0])
        t1 = shape_chip("(28, 28, 1)", "height, width, channels", DATA).move_to([1.95, -1.15, 0])
        t2 = shape_chip("(512, 512, 3)", "the lighthouse", EDGE).move_to([5.1, -1.15, 0])
        t3 = shape_chip(f"({BATCH}, 28, 28, 1)", "batch, height, width, channels", MODEL).move_to([3.65, -2.6, 0])
        pics = m.Group(*[framed(picture(tint(IMAGES[i], DATA), 0.95), EDGE) for i in range(5)])
        for k, pic in enumerate(pics):
            pic.move_to([-4.75 + 0.26 * k, -1.75 - 0.15 * k, 0])
        times = code(f"× {BATCH}", 26, MODEL, weight=m.BOLD).next_to(pics, m.RIGHT, buff=0.35)
        run_note = cite("one training batch of the digit model · trained on this Mac").move_to([-3.45, -3.3, 0])

        self.say(
            "[A colour picture stores three numbers per pixel]: [red, green and blue.]",
            m.AnimationGroup(self.new_header("03 · channels", "Colour, and tensors"),
                             m.LaggedStart(*[m.FadeIn(p, shift=0.15 * m.UP) for p in stack], lag_ratio=0.3)),
            m.AnimationGroup(*[p.animate(run_time=1.4).scale(0.72).move_to([x, spread_y, 0])
                               for p, x in zip(planes, spread_xs)],
                             m.LaggedStart(*[m.FadeIn(nm, shift=0.1 * m.UP) for nm in names], lag_ratio=0.3)),
        )
        self.say(
            "[Each colour is a grid of its own], [called a channel.]",
            m.LaggedStart(*[m.Circumscribe(p[0], color=hue[0], buff=0.06) for p, hue in zip(planes, hues)], lag_ratio=0.3),
            m.FadeIn(chan, shift=0.1 * m.UP),
        )
        self.say(
            "[The lighthouse from episode one is five hundred and twelve pixels on each side], [with three channels.]",
            m.AnimationGroup(m.FadeIn(dims, shift=0.1 * m.DOWN), m.FadeIn(r_digit, shift=0.1 * m.UP)),
            m.FadeIn(r_light[:2], shift=0.1 * m.UP),
        )
        self.say(
            "[That's seven hundred and eighty-six thousand, four hundred and thirty-two numbers], "
            "[about a thousand times the digit.]",
            m.FadeIn(r_light[2], scale=1.2),
            m.FadeIn(ratio, shift=0.1 * m.UP),
        )
        self.say(
            "[Grids of numbers like these, with any number of axes, are called tensors.]",
            m.AnimationGroup(m.FadeIn(t_title), m.FadeIn(t1, shift=0.1 * m.UP), m.FadeIn(t2, shift=0.1 * m.UP)),
        )
        self.say(
            "[The digit model trains on a stack of two hundred and fifty-six digits at a time]: [a tensor with four axes.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(chan)), m.LaggedStart(*[m.FadeIn(p, shift=0.1 * m.DOWN) for p in pics],
                                                                    lag_ratio=0.2), m.FadeIn(times), m.FadeIn(run_note)),
            m.FadeIn(t3, shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 4 · the space

    def space(self):
        self.next_section("4 · The space", notes=(
            "Each number is a coordinate. Two real pixels of digit #15: a = (10, 19) in the stroke, b = (20, 20) in "
            "the background; the point's path on their plane is illustrative (the two squares are recoloured as it "
            "moves). The star of axes stands for 784 dimensions (illustrative). 512 × 512 × 3 = 786,432; "
            "1024 × 1024 × 3 = 3,145,728 (SDXL, SD3 and FLUX output size; notes §2 table)."))
        dh, dc = 3.0, np.array([-4.9, 0.55, 0])
        digit = pixel_squares(DIGIT, dh).move_to(dc)
        digit_box = boxed(digit)
        pa, pb = (10, 19), (20, 20)
        va, vb = float(X0[pa]), float(X0[pb])
        assert va > 0.9 and vb == -1
        sq_a, sq_b = digit[pa[0]][pa[1]], digit[pb[0]][pb[1]]
        s = dh / SIDE
        mark_a = m.Square(s * 1.9).set_stroke(INK, 3).move_to(sq_a)
        mark_b = m.Square(s * 1.9).set_stroke(INK, 3).move_to(sq_b)
        edge_x = digit_box.get_right()[0] + 0.45
        lab_a = code("pixel a", 20, INK).move_to([0, mark_a.get_y(), 0]).align_to([edge_x, 0, 0], m.LEFT)
        lab_b = code("pixel b", 20, INK).move_to([0, mark_b.get_y(), 0]).align_to([edge_x, 0, 0], m.LEFT)
        lead_a = m.Line(mark_a.get_right(), lab_a.get_left() + np.array([-0.1, 0, 0])).set_stroke(INK, 1.5)
        lead_b = m.Line(mark_b.get_right(), lab_b.get_left() + np.array([-0.1, 0, 0])).set_stroke(INK, 1.5)
        marks = m.VGroup(mark_a, mark_b, lab_a, lab_b, lead_a, lead_b)

        P = Plot((-1.25, 1.25), (-1.25, 1.25), width=4.0, height=4.0, center=(2.55, -0.25, 0),
                 x_ticks=[-1, 0, 1], y_ticks=[-1, 0, 1], x_label="pixel a", y_label="pixel b")
        dot = m.Dot(P.coords(va, vb), radius=0.12, color=DATA).set_z_index(3)
        gx = m.Line(P.coords(va, -1.25), P.coords(va, vb)).set_stroke(FAINT, 2)
        gy = m.Line(P.coords(-1.25, vb), P.coords(va, vb)).set_stroke(FAINT, 2)
        mover = m.VGroup(dot, gx, gy, sq_a, sq_b)
        route = [(va, vb), (-0.85, vb), (-0.85, 0.85), (va, 0.85), (va, vb)]

        def walk(_, alpha):
            seg = min(int(alpha * 4), 3)
            f = alpha * 4 - seg
            x = route[seg][0] + (route[seg + 1][0] - route[seg][0]) * f
            y = route[seg][1] + (route[seg + 1][1] - route[seg][1]) * f
            dot.move_to(P.coords(x, y))
            gx.put_start_and_end_on(P.coords(x, -1.25), P.coords(x, y))
            gy.put_start_and_end_on(P.coords(-1.25, y), P.coords(x, y))
            sq_a.set_fill(shade(x), 1)
            sq_b.set_fill(shade(y), 1)

        star_c = np.array([2.55, -0.25, 0])
        star = m.VGroup()
        for k in range(12):
            ang = k * PI_ / 12
            u = np.array([np.cos(ang), np.sin(ang), 0])
            star.add(m.Line(star_c - 2.25 * u, star_c + 2.25 * u).set_stroke(FAINT, 2))
        tips = [(0, "x0[0]"), (3, "x0[1]"), (6, "x0[2]"), (9, "x0[3]")]
        tip_labels = m.VGroup()
        for k, lab in tips:
            ang = k * PI_ / 12
            u = np.array([np.cos(ang), np.sin(ang), 0])
            tip_labels.add(code(lab, 20, MUTED).move_to(star_c + 2.75 * u))
        more = code(f"… {N_PIX} axes", 22, MUTED).move_to(star_c + np.array([0, -2.75, 0]))
        seven_link = m.DashedLine(digit_box.get_right() + np.array([0.12, 0, 0]), star_c + np.array([-0.18, 0, 0]),
                                  dash_length=0.1).set_stroke(DATA, 2.5)

        def thumb(img, color, at, lo=None, hi=None):
            spot = star_c + np.array(at + (0,))
            d = m.Dot(spot, radius=0.09, color=color).set_z_index(3)
            pic = framed(picture(tint(img, color, lo, hi), 0.8), color)
            pic.move_to(spot + np.array([0.0, 0.75, 0]) * (1 if at[1] > 0 else -1))
            return m.Group(d, pic)

        th_noise = thumb(RANDOM_PICS[0], NOISE, (-1.6, -0.95), -1, 1)
        th_other = thumb(IMAGES[0], DATA, (1.55, -0.95))

        big_lh = stat(f"{N_LH:,}", "axes", "the lighthouse · 512 × 512 × 3", INK, 44)
        big_lg = stat(f"{N_LARGE:,}", "axes", "a large photo · 1024 × 1024 × 3", INK, 44)
        big_lh.move_to([-4.6, -2.25, 0])
        big_lg.move_to([-4.6, -2.25, 0])

        self.say(
            "[Here's the big idea]: [treat every one of those numbers as a coordinate.]",
            m.AnimationGroup(self.new_header("04 · the space", "One picture, one point"), m.FadeIn(digit_box, shift=0.15 * m.UP)),
            m.AnimationGroup(m.Create(mark_a), m.Create(mark_b), m.Create(lead_a), m.Create(lead_b),
                             m.FadeIn(lab_a), m.FadeIn(lab_b)),
        )
        self.say(
            "[Take just two pixels], [and this seven becomes one point on a plane.]",
            m.FadeIn(P),
            m.AnimationGroup(m.FadeIn(dot, target_position=mark_a, scale=0.5), m.Create(gx), m.Create(gy)),
        )
        self.say(
            "[Move the point, and the two pixels change with it.]",
            m.UpdateFromAlphaFunc(mover, walk, run_time=4.0, rate_func=m.smooth),
        )
        self.say(
            "[With all seven hundred and eighty-four], [the seven is one point in a space with seven hundred and eighty-four axes.]",
            m.AnimationGroup(m.FadeOut(P), m.FadeOut(gx), m.FadeOut(gy), m.FadeOut(marks),
                             dot.animate(run_time=1.0).move_to(star_c)),
            m.AnimationGroup(m.LaggedStart(*[m.Create(ln) for ln in star], lag_ratio=0.08, run_time=1.8),
                             m.FadeIn(tip_labels), m.FadeIn(more), m.Create(seven_link)),
        )
        self.say(
            "[We can't draw that space], [but every point in it is a picture], [and every picture is a point.]",
            m.Circumscribe(m.VGroup(star, tip_labels), color=MUTED, buff=0.1, shape=m.Circle),
            m.FadeIn(th_noise, scale=0.8),
            m.FadeIn(th_other, scale=0.8),
        )
        self.say(
            "[The lighthouse lives in a space with seven hundred and eighty-six thousand axes], [a large photo in over three million.]",
            m.FadeIn(big_lh, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeOut(big_lh, shift=0.1 * m.UP), m.FadeIn(big_lg, shift=0.1 * m.UP)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 5 · random points

    def random_points(self):
        self.next_section("5 · Random points", notes=(
            "Random points are static: noise_image(seed = 0 … 14) (kit.py, standard Gaussian), shown clipped to [−1, 1] "
            "like a picture. Digit #15 has 640 pixels exactly 0 and 144 with ink. 256^784 = 10^1888.06, a number with "
            "1,889 digits (notes §2 worked example 2); no probability of hitting a digit is claimed. The check is "
            "computed here: nearest of the 60,000 training digits (Euclidean, pixels in [−1, 1]) for noise_image(seed "
            f"0 … 999) clipped: {D_RANDOM.min():.1f} … {D_RANDOM.max():.1f}; for the first 1,000 test digits: "
            f"{D_TEST.min():.1f} … {D_TEST.max():.1f} (median {np.median(D_TEST):.1f}); closest random ÷ farthest test = "
            f"{GAP:.2f}."))
        cols, rows_, ph = 5, 3, 0.95
        pics = m.Group(*[framed(picture(tint(RANDOM_PICS[i], NOISE, -1, 1), ph), mix(EDGE, NOISE, 0.4))
                         for i in range(cols * rows_)])
        pics.arrange_in_grid(rows=rows_, cols=cols, buff=0.15).move_to([-3.55, 0.05, 0])
        static = code("static", 20, NOISE).next_to(pics[0], m.UP, buff=0.12)
        grid_note = code(f"noise_image(seed) · {cols * rows_} of {N_CHECK:,}", 20, MUTED).next_to(pics, m.DOWN, buff=0.25)

        seven = framed(pixel_squares(DIGIT, 2.0), mix(EDGE, DATA, 0.5)).move_to([1.85, 0.8, 0])
        facts = m.VGroup(
            code(f"{N_DARK} pixels exactly 0", 20, MUTED),
            code(f"{N_INK} pixels with ink", 20, DATA),
        ).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.22).next_to(seven, m.RIGHT, buff=0.4)

        count = m.VGroup(tex(r"256^{784}", 50, NOISE), tex(r"\approx 10^{1888}", 50, INK)).arrange(m.RIGHT, buff=0.25)
        count.move_to([3.6, -1.45, 0])
        count_lab = text("possible 28 × 28 grids, at 256 shades each", 20, MUTED).next_to(count, m.DOWN, buff=0.2)
        count_vs = code(f"training digits: {N_TRAIN:,}", 22, DATA).next_to(count_lab, m.DOWN, buff=0.25)

        top = 0.4
        H = Plot((0, 36), (0, top), width=10.6, height=3.6, center=(0.0, -0.6, 0), x_ticks=[0, 10, 20, 30],
                 x_label=f"distance to the nearest of the {N_TRAIN:,} training digits")
        bars_r = hist_bars(H, D_RANDOM, NOISE, width=0.5)
        bars_t = hist_bars(H, D_TEST, DATA, width=0.5)
        lab_r = text(f"{N_CHECK:,} random pictures", 22, NOISE).next_to(bars_r, m.UP, buff=0.2)
        lab_t = text(f"{N_CHECK:,} new real digits (test set)", 22, DATA).next_to(bars_t, m.UP, buff=0.2)
        y_arrow = 0.62 * top
        gap_arrow = m.DoubleArrow(H.coords(D_TEST.max(), y_arrow), H.coords(D_RANDOM.min(), y_arrow), buff=0.06,
                                  stroke_width=3, max_tip_length_to_length_ratio=0.06, color=INK)
        gap_lab = m.VGroup(text("more than twice as far", 22, INK),
                           code(f"{D_RANDOM.min():.1f} vs {D_TEST.max():.1f}", 20, MUTED)).arrange(m.DOWN, buff=0.1)
        gap_lab.next_to(gap_arrow, m.UP, buff=0.15)
        units = cite("Euclidean distance, pixels in −1 … 1 · computed").next_to(H, m.DOWN, buff=0.15)

        self.say(
            "[So what does a random point look like?] [Pick every pixel at random.]",
            self.new_header("05 · random points", "Almost everything is static"),
            m.FadeIn(pics[0], scale=0.9),
        )
        self.say(
            "[Static.] [Try again, and again: always static.]",
            m.FadeIn(static, shift=0.1 * m.UP),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(p, scale=0.9) for p in pics[1:]], lag_ratio=0.15, run_time=2.4),
                             m.FadeIn(grid_note)),
        )
        self.say(
            "[A digit needs all of its pixels right at once]: [a dark background, and one connected stroke.]",
            m.FadeIn(seven, shift=0.1 * m.UP),
            m.LaggedStart(m.FadeIn(facts[0], shift=0.1 * m.UP), m.FadeIn(facts[1], shift=0.1 * m.UP), lag_ratio=0.5),
        )
        self.say(
            "[Count the possible grids]: [two hundred and fifty-six shades for each of seven hundred and eighty-four pixels] "
            "[gives a number nearly nineteen hundred digits long.]",
            m.FadeIn(count[0], shift=0.1 * m.UP),
            m.FadeIn(count_lab),
            m.AnimationGroup(m.FadeIn(count[1], shift=0.1 * m.UP), m.FadeIn(count_vs)),
        )
        self.clear()
        self.say(
            "[So we checked]: [a thousand random pictures], [each compared with all sixty thousand training digits.]",
            m.AnimationGroup(m.FadeIn(H), m.FadeIn(units)),
            m.AnimationGroup(m.LaggedStart(*[m.GrowFromEdge(b, m.DOWN) for b in bars_r], lag_ratio=0.1),
                             m.FadeIn(lab_r)),
            m.Circumscribe(H.x_label, color=MUTED, buff=0.08),
        )
        self.say(
            "[New digits from the test set land close to some training digit]; [the random ones stay more than twice as far.]",
            m.AnimationGroup(m.LaggedStart(*[m.GrowFromEdge(b, m.DOWN) for b in bars_t], lag_ratio=0.06),
                             m.FadeIn(lab_t)),
            m.AnimationGroup(m.GrowFromCenter(gap_arrow), m.FadeIn(gap_lab, shift=0.1 * m.UP)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 6 · the thin set

    def thin_set(self):
        self.next_section("6 · The thin set", notes=(
            "The swirl (diffusion_reference.swirl, 1,500 points, seed 0) as the 2D stand-in for the data. 2,000 of the "
            "200,000 seeded uniform points in [−1.2, 1.2]² are drawn; 'near' = within 0.1 of the swirl's centre line "
            "(off_swirl's threshold). Shares computed exactly in the extra dimensions (see tube_share): "
            + ", ".join(f"{k}D 1 in {ODDS[k]:,.3g}" for k in DIMS) + ". Manifold hypothesis: Song & Ermon, 2019, "
            "§3.1 (quoted). MNIST intrinsic dimension 7-13: Pope et al., 2021, Table 1 (MLE, k = 3 … 20). The glow is "
            "the swirl's exact density p(x) (mixture_density at t = 0, square-root scaled)."))
        S = Plot((-1.25, 1.25), (-1.25, 1.25), width=5.0, height=5.0, center=(-3.55, -0.45, 0), axes=False)
        box = m.Square(5.0).set_stroke(EDGE, 2).move_to(S.coords(0, 0))
        self.swirl_plot, self.swirl_box = S, box
        data = S.cloud(SWIRL_PTS, DATA, radius=0.03)
        self.swirl_dots = data
        curve = m.VGroup(*[S.line(arm[:, 0], arm[:, 1], DATA, 3) for arm in swirl_curve(300)])
        box_cap = code("the swirl · 2D stand-in", 20, MUTED).next_to(box, m.DOWN, buff=0.15)
        self.swirl_cap = box_cap
        uni = S.cloud(UNIFORM, NOISE, radius=0.03, opacity=0.85)
        near_cols = [NOISE if n else FAINT for n in NEAR_MASK]
        near_ops = [1.0 if n else 0.35 for n in NEAR_MASK]

        rx = 0.55
        head = text("random points within 0.1 of the swirl", 24, INK).move_to([0, 2.15, 0]).align_to([rx, 0, 0], m.LEFT)
        rows = []
        for k in DIMS:
            o = ODDS[k]
            if o < 100:
                odds = f"1 in {o:.0f}"
            elif o < 1e6:
                odds = f"1 in {float(f'{o:.2g}'):,.0f}"
            else:
                odds = f"1 in {float(f'{o:.2g}') / 1e9:.0f} billion"
            dim = code(f"{k:>2} dimensions", 24, MUTED)
            val = code(odds, 24, NOISE if k > 2 else INK, weight=m.BOLD)
            rows.append(m.VGroup(dim, val))
        for i, rw in enumerate(rows):
            y = 1.45 - 0.62 * i
            rw[0].move_to([0, y, 0]).align_to([rx, 0, 0], m.LEFT)
            rw[1].move_to([0, y, 0]).align_to([6.55, 0, 0], m.RIGHT)
        digit_row = m.VGroup(code(f"{N_PIX} dimensions", 24, DATA), text("a digit", 24, DATA))
        digit_row[0].move_to([0, 1.45 - 0.62 * len(rows), 0]).align_to([rx, 0, 0], m.LEFT)
        digit_row[1].move_to(digit_row[0]).align_to([6.55, 0, 0], m.RIGHT)
        table = m.VGroup(head, *rows, digit_row)

        idea = title_text("The manifold idea", 40, INK).move_to([0, 1.85, 0]).align_to([rx, 0, 0], m.LEFT)
        quote = m.VGroup(
            text("“…data in the real world tend to concentrate", 22, INK),
            text("on low dimensional manifolds embedded in", 22, INK),
            text("a high dimensional space”", 22, INK),
        ).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.1).next_to(idea, m.DOWN, buff=0.4).align_to(idea, m.LEFT)
        q_src = cite("Song & Ermon, 2019").next_to(quote, m.DOWN, buff=0.18).align_to(quote, m.LEFT)
        q_bar = m.Line(quote.get_corner(m.UL) + np.array([-0.22, 0.05, 0]), quote.get_corner(m.DL) + np.array([-0.22, -0.05, 0]))
        q_bar.set_stroke(EDGE, 4)
        q_card = m.VGroup(q_bar, quote, q_src)
        dim_stat = m.VGroup(title_text("7 – 13", 56, DATA), text(f"dimensions, out of {N_PIX}", 24, MUTED))
        dim_stat.arrange(m.RIGHT, buff=0.3).move_to([0, -1.55, 0]).align_to([rx, 0, 0], m.LEFT)
        dim_stat[1].align_to(dim_stat[0], m.DOWN)
        dim_src = cite("handwritten digits, estimated · Pope et al., 2021").next_to(dim_stat, m.DOWN, buff=0.18)
        dim_src.align_to(dim_stat, m.LEFT)
        glow_img = density_image(S)
        p_lab = m.VGroup(code("p_data(x)", 26, DATA, weight=m.BOLD), text("the data distribution", 22, MUTED))
        p_lab.arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.12).move_to([0, -2.85, 0]).align_to([rx, 0, 0], m.LEFT)

        self.say(
            "[We can't draw seven hundred and eighty-four dimensions], [so here is our stand-in, in just two]: [the swirl.]",
            m.AnimationGroup(self.new_header("06 · the thin set", "Real pictures are rare"), m.Create(box)),
            m.FadeIn(box_cap),
            m.LaggedStart(*[m.FadeIn(d, scale=0.5) for d in data], lag_ratio=0.002, run_time=1.6),
        )
        self.say(
            "[Each dot is one example, like one training digit], [and together they trace a thin spiral.]",
            m.Flash(data[40].get_center(), color=DATA, line_length=0.15, flash_radius=0.12),
            m.AnimationGroup(*[m.ShowPassingFlash(c.copy().set_stroke(INK, 5), time_width=0.5, run_time=2.0) for c in curve]),
        )
        self.say(
            "[Now scatter random points over the square.] [In two dimensions, about one in three lands near the swirl.]",
            m.LaggedStart(*[m.FadeIn(d, scale=0.5) for d in uni], lag_ratio=0.001, run_time=1.8),
            m.AnimationGroup(recolor_dots(uni, NOISE, 0.85, near_cols, near_ops, run_time=1.2), m.FadeIn(head), m.FadeIn(rows[0])),
        )
        self.say(
            "[Give the swirl a third dimension to miss in], [and it's one in forty-six.] "
            "[In ten dimensions, one in two hundred billion.]",
            m.FadeIn(rows[1][0], shift=0.1 * m.UP),
            m.FadeIn(rows[1][1], shift=0.1 * m.UP),
            m.LaggedStart(*[m.FadeIn(rw, shift=0.1 * m.UP) for rw in rows[2:]], lag_ratio=0.4),
        )
        self.say(
            "[Every extra dimension is one more way to miss], [and a digit has seven hundred and eighty-four.]",
            m.Circumscribe(m.VGroup(*[rw[1] for rw in rows]), color=NOISE, buff=0.1),
            m.FadeIn(digit_row, shift=0.1 * m.UP),
        )
        self.say(
            "[That's the manifold idea]: [real data sits on a thin, low-dimensional set] [inside a huge space.]",
            m.AnimationGroup(m.FadeOut(table), m.FadeIn(idea, shift=0.1 * m.UP)),
            m.FadeIn(m.VGroup(q_bar, quote), shift=0.1 * m.UP),
            m.FadeIn(q_src),
        )
        self.say(
            "[For handwritten digits, one estimate puts it at only seven to thirteen dimensions.]",
            m.AnimationGroup(m.FadeIn(dim_stat, shift=0.1 * m.UP), m.FadeIn(dim_src)),
        )
        self.say(
            "[Where real pictures sit, and how often each kind appears], [is called the data distribution.]",
            m.AnimationGroup(m.FadeOut(uni), m.FadeIn(glow_img)),
            m.FadeIn(p_lab, shift=0.1 * m.UP),
        )
        self.glow_img = glow_img
        self.wait(0.3)
        self.clear(box, glow_img, box_cap, data, *data)

    # ------------------------------------------------------------------ 7 · sampling

    def sampling(self):
        self.next_section("7 · Sampling", notes=(
            "To generate = to draw new samples from p_data, knowing it only through examples. The average of all "
            f"{N_SEVENS:,} training sevens is a blur (computed); the swirl's average ({len(SWIRL_PTS):,} points) sits at "
            f"({SWIRL_MEAN[0]:.3f}, {SWIRL_MEAN[1]:.3f}), {MEAN_GAP:.2f} from the nearest arm (the data's own spread across "
            "the curve is 0.03). Earlier generators and their blurry averages: Episode 3."))
        S = self.swirl_plot
        rx = 0.55
        we_have = text(f"we have: {N_TRAIN:,} examples", 24, DATA).move_to([0, 2.15, 0]).align_to([rx, 0, 0], m.LEFT)
        we_want = text("we want: new ones, drawn from p_data", 24, INK).next_to(we_have, m.DOWN, buff=0.25)
        we_want.align_to(we_have, m.LEFT)
        spot = S.coords(*swirl_curve(200)[0][150])
        ring = m.Circle(radius=0.13).set_stroke(INK, 3).move_to(spot).set_z_index(4)
        new_lab = code("new?", 20, INK).next_to(self.swirl_box, m.RIGHT, buff=0.2).set_y(spot[1])
        new_lead = m.Line(ring.get_right(), new_lab.get_left() + np.array([-0.08, 0, 0])).set_stroke(INK, 1.5)
        ask = m.VGroup(ring, new_lead, new_lab)

        reals = m.Group(*[framed(picture(tint(SEVENS[i], DATA), 1.05), EDGE) for i in range(2)])
        reals.arrange(m.RIGHT, buff=0.15)
        avg = framed(picture(tint(MEAN_SEVEN / 255.0, DATA, 0, 1), 1.3), DATA)
        eq = arrow([0, 0, 0], [0.7, 0, 0], MUTED, buff=0)
        row = m.Group(reals, eq, avg).arrange(m.RIGHT, buff=0.3).move_to([0, -0.45, 0]).align_to([rx, 0, 0], m.LEFT)
        r_lab = text("real sevens", 20, MUTED).next_to(reals, m.DOWN, buff=0.15)
        a_lab = text(f"average of {N_SEVENS:,}", 20, DATA).next_to(avg, m.DOWN, buff=0.15)
        blur = text("a blur nobody wrote", 22, RED).next_to(a_lab, m.DOWN, buff=0.15)

        cross = m.VGroup(m.Line([-0.14, -0.14, 0], [0.14, 0.14, 0]), m.Line([-0.14, 0.14, 0], [0.14, -0.14, 0]))
        cross.set_stroke(RED, 5).move_to(S.coords(*SWIRL_MEAN)).set_z_index(4)
        key = cross.copy().set_z_index(0).scale(0.8)
        mean_words = code(f"average point: {MEAN_GAP:.2f} from the swirl", 20, RED)
        mean_lab = m.VGroup(key, mean_words).arrange(m.RIGHT, buff=0.25)
        mean_lab.move_to([0, -2.5, 0]).align_to([rx, 0, 0], m.LEFT)
        link = cite("more on blurry averages: Episode 3").next_to(mean_lab, m.DOWN, buff=0.2)
        link.align_to(mean_lab, m.LEFT)

        self.say(
            "[To generate, we must draw new points from that distribution], [but all we have are examples.]",
            m.AnimationGroup(self.new_header("07 · sampling", "Why sampling is hard"), m.FadeIn(ask),
                             m.FadeIn(we_want, shift=0.1 * m.UP)),
            m.FadeIn(we_have, shift=0.1 * m.UP),
        )
        self.say(
            "[The easy guess fails]: [the average of over six thousand training sevens is a blur that nobody wrote.]",
            m.AnimationGroup(m.FadeIn(reals, shift=0.1 * m.UP), m.FadeIn(r_lab)),
            m.AnimationGroup(m.FadeIn(eq), m.FadeIn(avg, shift=0.1 * m.LEFT), m.FadeIn(a_lab), m.FadeIn(blur)),
        )
        self.say(
            "[On the swirl, the average lands in the middle], [where there is no data at all.]",
            m.AnimationGroup(m.FadeIn(cross, scale=0.5), m.FadeIn(mean_lab, shift=0.1 * m.UP)),
            m.AnimationGroup(m.Circumscribe(cross, color=RED, shape=m.Circle, buff=0.12), m.FadeIn(link)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 8 · the way out

    def way_out(self):
        self.next_section("8 · The way out", notes=(
            "Preview only: 500 Gaussian noise points (gaussian_noise(900, seed=5), the first 500 inside the frame) walked onto the "
            "swirl by DDIM with 50 steps and the swirl's exact denoiser (exact_eps; diffusion_reference.ddim_sample), "
            "computed. How earlier generators tried: Episode 3. Adding noise, the forward process: Episode 4. "
            "Denoisers, scores and samplers: Episodes 5-9."))
        W = Plot((-2.0, 2.0), (-2.0, 2.0), width=5.2, height=5.2, center=(-3.45, -0.45, 0), axes=False)
        box = m.Square(5.2).set_stroke(EDGE, 2).move_to(W.coords(0, 0))
        cloud = W.cloud(PATH[0], NOISE, radius=0.035)
        steps = len(PATH) - 1
        counter = code("step 0", 22, TIME).next_to(box, m.DOWN, buff=0.15)
        shown = [code(f"step {i}", 22, TIME).move_to(counter) for i in range(steps + 1)]
        prev_note = m.VGroup(text("a preview, computed", 22, INK),
                             cite(f"DDIM · {steps} steps · the swirl's exact denoiser"),
                             cite("noise in violet, the swirl in teal"))
        prev_note.arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.12).move_to([0, 2.05, 0]).align_to([0.75, 0, 0], m.LEFT)

        def walk(_, a):
            i = a * steps
            j = int(np.floor(i))
            k = min(j + 1, steps)
            xy = PATH[j] + (PATH[k] - PATH[j]) * (i - j)
            colour = mix(NOISE, DATA, min(1.0, a * 1.15))
            for d, (x, y) in zip(cloud, xy):
                d.move_to(W.coords(x, y))
                d.set_fill(colour, 0.9)
            counter.become(shown[int(round(a * steps))])

        def card_for(n, line):
            title, module = EPISODES[n]
            name, accent, fill = MODULES[module]
            box_ = card(5.6, 1.35, accent, CARD)
            words = m.VGroup(code(f"EP {n:02d} · {title}", 22, accent, weight=m.BOLD), text(line, 22, MUTED))
            words.arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.15).move_to(box_).align_to(box_, m.LEFT).shift(0.35 * m.RIGHT)
            return m.VGroup(box_, words)

        c3 = card_for(3, "how earlier generators tried").move_to([0, 0.05, 0]).align_to([0.75, 0, 0], m.LEFT)
        c4 = card_for(4, "diffusion's first half: adding noise").move_to([0, -1.65, 0]).align_to([0.75, 0, 0], m.LEFT)

        self.say(
            "[What we need is a way to walk from noise] [onto the thin set.]",
            m.AnimationGroup(self.new_header("08 · the way out", "From noise to the thin set"), m.Create(box),
                             m.FadeIn(cloud), m.FadeIn(counter), m.FadeIn(prev_note)),
            m.UpdateFromAlphaFunc(cloud, walk, run_time=4.2, rate_func=m.smooth),
        )
        self.say(
            "[Episode three shows how earlier generators tried], [and episode four starts diffusion's answer: adding noise.]",
            m.FadeIn(c3, shift=0.15 * m.UP),
            m.FadeIn(c4, shift=0.15 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 9 · recap

    def recap(self):
        self.next_section("9 · Recap", notes=(
            "Three words: numbers (a digit = 784, the lighthouse = 786,432), points (one picture = one point), rare "
            "(random points are static; real pictures sit on a thin set). Next: Episode 3, Before Diffusion."))
        halo = glow(NOISE, radius=3.4).move_to([0, 0.35, 0])
        words = m.VGroup(title_text("Numbers", 64, INK), title_text("Points", 64, DATA), title_text("Rare", 64, NOISE))
        words.arrange(m.RIGHT, buff=1.0).move_to([0, 0.35, 0])
        on_baseline(words)
        a1 = arrow(words[0].get_right(), words[1].get_left(), MUTED, buff=0.25)
        a2 = arrow(words[1].get_right(), words[2].get_left(), MUTED, buff=0.25)
        trio = m.VGroup(words, a1, a2)
        subs = m.VGroup(
            code(f"{SIDE} × {SIDE} = {N_PIX}", 20, MUTED).next_to(words[0], m.DOWN, buff=0.35),
            code("1 picture = 1 point", 20, MUTED).next_to(words[1], m.DOWN, buff=0.35),
            code("a thin set", 20, MUTED).next_to(words[2], m.DOWN, buff=0.35),
        )
        for sub in subs[1:]:
            sub.align_to(subs[0], m.DOWN)
        mascot = pixel(0.7).move_to([0, -2.45, 0])

        self.say(
            "[Three words to keep]: [numbers], [points], [rare.]",
            m.AnimationGroup(self.new_header("09 · recap", "Numbers, points, rare"), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.2 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.GrowArrow(a1), m.FadeIn(words[1], shift=0.2 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.GrowArrow(a2), m.FadeIn(words[2], shift=0.2 * m.UP), m.FadeIn(subs[2])),
        )
        self.say(
            "[A picture is a point in a space with millions of dimensions], [and real pictures fill only a thin sliver of it.]",
            m.Circumscribe(trio, color=MUTED, buff=0.2),
            m.FadeIn(mascot, shift=0.2 * m.UP),
        )
        self.wait(0.6)
        self.clear()
