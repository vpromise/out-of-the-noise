"""A tour of the "Out of the Noise" kit: every component on screen once.
VOICE=draft manimgx inspect kit_demo.py   → KitDemo.storyboard*.png"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kit import *  # noqa: E402,F403


def seven(n=28):
    """A crude hand-drawn 7 (illustration only)."""
    img = -np.ones((n, n))
    img[5:8, 6:22] = 1.0
    for r in range(8, 24):
        c = int(21 - (r - 8) * 0.55)
        img[r, c - 1:c + 2] = 1.0
    return img


def spiral(n=600, seed=0):
    rng = np.random.default_rng(seed)
    t = np.sqrt(rng.uniform(0.04, 1.0, n)) * 3.2 * np.pi
    xy = np.stack([t * np.cos(t), t * np.sin(t)], axis=1) / 10.0
    return xy + 0.03 * rng.standard_normal((n, 2))


class KitDemo(Episode):
    number = 4

    def lesson(self):
        # 1 · the loop and the mascot
        loop = denoise_loop(center=(0, -0.2, 0), width=9.4)
        mascot = pixel(0.8).move_to([5.6, 2.4, 0])
        self.say(
            "[The loop]: [a noisy picture goes in], [a little less noise comes out], [and round again.]",
            m.AnimationGroup(self.new_header("kit · 1", "The loop and Pixel"), m.FadeIn(mascot)),
            m.AnimationGroup(m.FadeIn(loop.left), m.GrowArrow(loop.go), m.FadeIn(loop.model)),
            m.AnimationGroup(m.GrowArrow(loop.out), m.FadeIn(loop.right)),
            m.Create(loop.back),
        )
        self.clear()

        # 2 · pictures from arrays: a digit dissolving into noise
        x0 = seven()
        eps = noise_image(28, 28, seed=1)
        abar = [1.0, 0.8, 0.5, 0.2, 0.05, 0.0]
        frames = [np.sqrt(a) * x0 + np.sqrt(1 - a) * eps for a in abar]
        row = strip([tint(f, TEAL if a > 0.5 else (VIOLET if a < 0.2 else INK), -2.2, 2.2) for f, a in zip(frames, abar)],
                    height=1.55, labels=[f"ᾱ = {a:g}" for a in abar])
        row.move_to([0, 0.9, 0])
        formula = tex(r"x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon", 40).move_to([0, -1.6, 0])
        formula[0][0:2].set_color(NOISE)
        self.say(
            "[Pictures from arrays], [a digit dissolving], [and the formula.]",
            m.AnimationGroup(self.new_header("kit · 2", "Pictures, strips, formulas"), m.FadeIn(row[0])),
            m.LaggedStart(*[m.FadeIn(f) for f in row[1:]], lag_ratio=0.25),
            m.Write(formula),
        )
        self.clear()

        # 3 · point clouds and an arrow field
        plot = Plot((-1.2, 1.2), (-1.2, 1.2), width=4.6, height=4.6, center=(-4.1, -0.4, 0), axes=False)
        frame = m.Square(4.6).set_stroke(EDGE, 2).move_to(plot.coords(0, 0))
        data = spiral()
        rng = np.random.default_rng(2)
        noise = rng.standard_normal(data.shape) * 0.5
        cloud = plot.cloud(noise, NOISE)
        field = plot.field(lambda x, y: (-y - 0.6 * x, x - 0.6 * y), nx=11, ny=11, max_len=0.38)
        net = UNet(sizes=(28, 14, 7), channels=(32, 64, 128), center=(3.0, -0.3, 0), scale=0.75)
        self.say(
            "[A cloud of noise], [an arrow field], [the cloud becoming a swirl], [and a U-Net.]",
            m.AnimationGroup(self.new_header("kit · 3", "Clouds, fields, the U-Net"), m.FadeIn(frame), m.FadeIn(cloud)),
            m.LaggedStart(*[m.GrowArrow(a) if isinstance(a, m.Arrow) else m.FadeIn(a) for a in field], lag_ratio=0.01),
            m.AnimationGroup(cloud.to(data, run_time=2.0), m.FadeOut(field)),
            m.AnimationGroup(m.FadeIn(net), net.flow()),
        )
        self.clear()

        # 4 · curves, bars, code
        plot2 = Plot((0, 1000), (0, 1), width=5.6, height=2.5, center=(-3.4, 0.55, 0), x_ticks=[0, 500, 1000],
                     y_ticks=[0, 0.5, 1.0], x_label="step t", y_label="signal kept, ᾱ")
        ts = np.linspace(0, 1000, 120)
        betas = np.linspace(1e-4, 0.02, 1000)
        ab = np.cumprod(1 - betas)
        lin = plot2.line(ts, np.interp(ts, np.arange(1000), ab), TIME)
        cosab = np.cos((ts / 1000 + 0.008) / 1.008 * np.pi / 2) ** 2 / np.cos(0.008 / 1.008 * np.pi / 2) ** 2
        cosl = plot2.line(ts, cosab, VIOLET)
        bars = Bars([0.62, 0.41, 0.27, 0.18], labels=["1", "3", "5", "8"], top=0.8, width=4.4, height=1.8,
                    center=(3.4, 1.2, 0), color=PROMPT)
        src = ("def q_sample(x0, t, eps, abar):\n"
               "    a = abar[t]\n"
               "    return sqrt(a) * x0 + sqrt(1 - a) * eps")
        block = code_block(src, 20, title="diffusion_reference.py").to_edge(m.RIGHT, buff=0.6).set_y(-2.55)
        self.say(
            "[A schedule], [two curves], [bars], [and code.]",
            m.AnimationGroup(self.new_header("kit · 4", "Curves, bars, code"), m.FadeIn(plot2)),
            m.AnimationGroup(m.Create(lin), m.Create(cosl)),
            m.FadeIn(bars),
            m.FadeIn(block),
        )
        self.clear()
