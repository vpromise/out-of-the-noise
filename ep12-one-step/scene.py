"""Out of the Noise · Episode 12 · One Step at a Time? (Module 3 · Faster Sampling — the module's last episode)

Distillation: a student learns to do in one step what the teacher does in many. Running example: the lighthouse —
SD-Turbo (Stable Diffusion 2.1 base distilled by adversarial diffusion distillation, the same 865.9 M-parameter
denoiser) against SD 2.1 itself, both run on this Mac (research/experiments/mac_runs.md §5). The mechanisms are shown
on the swirl (diffusion_reference: its exact denoiser as the teacher, DDIM with 64 steps); the reason distillation
exists is shown on the digit model at one step (mac_runs.md §4) and on the swirl (toy.md §4).

Every number on screen is computed below (diffusion_reference, the run's arrays and json) and asserted, or cited
with its source. Nothing was distilled for this episode: SD-Turbo was downloaded (Stability AI Community License).

Sources (research/notes.md §12, Appendix A row 12, Appendix C row 12; read October 2026):
- Salimans & Ho, "Progressive Distillation for Fast Sampling of Diffusion Models", arXiv:2202.00512 (Feb 2022),
  ICLR 2022: abstract (8192 → 4 steps "without losing much perceptual quality"); §5.2 (50 thousand parameter updates
  per halving, 100 thousand for 2 and 1 steps; cost to 4 steps "comparable or less than for training the original
  model"); Table 2 (CIFAR-10 FID: distilled 1/2/4/8 evaluations 9.12/4.51/3.00/2.57; DDIM (Song et al.) 10/20/50/100:
  13.36/6.84/4.67/4.16 — read in the PDF, Oct 2026); §7 (student = teacher's architecture and parameter count).
- Song, Dhariwal, Chen & Sutskever, "Consistency Models", arXiv:2303.01469 (Mar 2023), ICML 2023: f(x_t, t) maps any
  point of a probability-flow ODE trajectory to its start; consistency distillation, CIFAR-10 1-step FID 3.55.
- Luo, Tan, Huang, Li & Zhao, "Latent Consistency Models", arXiv:2310.04378 (Oct 2023): 2–4 steps, "32 A100 GPU
  hours" for a 768×768 model; LCM-LoRA arXiv:2311.05556 (Nov 2023).
- Sauer, Lorenz, Blattmann & Rombach, "Adversarial Diffusion Distillation", arXiv:2311.17042 (28 Nov 2023), SDXL
  Turbo: 1–4 steps; a discriminator plus a distillation loss to a frozen teacher; no classifier-free guidance at
  inference.
- research/experiments/mac_runs.md §4–5, §8 and mac_runs.json (digit_sampling, headline cfid_*, lighthouse.*);
  toy.md §4 and toy.json (flow/trained|exact/quality_vs_steps/1).

Illustrative (computed, but chosen): the eight swirl particles (rule in pick_particles); the folds are the IDEAL
student, read off the teacher's 64-step paths (a trained student only approximates them); the consistency arrows are
the ideal consistency function, again read off the paths; the hook's reveal is a cross-fade from decorative noise.

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

EP_DIR = Path(__file__).resolve().parent
warnings.filterwarnings("ignore", message=".*encountered in matmul")  # NumPy/Accelerate's spurious warnings

PREFIX = "ep12"
ARR = EXPERIMENTS / "arrays"
LIMG = EXPERIMENTS / "images" / "lighthouse"
HL = lambda *keys: measured("mac_runs", "headline", *keys)  # noqa: E731
LH = measured("mac_runs", "lighthouse")
MAC_MD = (EXPERIMENTS / "mac_runs.md").read_text()
NOTES = (EP_DIR.parent / "research" / "notes.md").read_text()
ORIGIN_ = np.zeros(3)

# ---------------------------------------------------------------------------- the lighthouse: teacher and student

SD_REPO, TB_REPO = "sd2-community/stable-diffusion-2-1-base", "stabilityai/sd-turbo"
MODELS = LH["models"]
assert MODELS[SD_REPO]["sampler"] == "SimpleEulerSampler" and MODELS[TB_REPO]["sampler"] == "SimpleEulerAncestralSampler"
UNET = MODELS[SD_REPO]["parts"]["unet"]["params"]
assert UNET == MODELS[TB_REPO]["parts"]["unet"]["params"] == 865_910_724            # "the very same denoiser"
assert f"{UNET / 1e6:.1f}" == f"{HL('lighthouse_unet_params_M'):.1f}" == "865.9"
assert "SD-Turbo is SD 2.1 distilled by Adversarial Diffusion Distillation" in MAC_MD
assert "Stability AI Community License" in MAC_MD


def _median(rows, model, steps):
    v = [r["total_s"] for r in rows if r["model"] == model and r["steps"] == steps]
    assert len(v) == 4, (model, steps, v)                                           # seeds 0–3
    return float(np.median(v))


assert all(r["cfg"] == 7.5 for r in LH["steps_sweep"]) and all(r["cfg"] == 0.0 for r in LH["turbo_sweep"])
SD_T = {s: _median(LH["steps_sweep"], SD_REPO, s) for s in (1, 4, 50)}
TB_T = {s: _median(LH["turbo_sweep"], TB_REPO, s) for s in (1, 2, 4)}
SD_TXT = {s: (f"{SD_T[s]:.1f} s" if SD_T[s] > 5 else f"{SD_T[s]:.2f} s") for s in SD_T}
TB_TXT = {s: f"{TB_T[s]:.2f} s" for s in TB_T}
assert SD_TXT == {1: "0.33 s", 4: "0.75 s", 50: "6.9 s"}, SD_TXT
assert TB_TXT == {1: "0.25 s", 2: "0.32 s", 4: "0.47 s"}, TB_TXT
assert HL("lighthouse_sd21_50steps_total_s_median_sweep") == 6.9 and HL("lighthouse_sd21_1steps_total_s_median_sweep") == 0.33
assert abs(TB_T[1] - HL("lighthouse_turbo_1step_total_s_median_sweep")) < 1e-3
assert HL("lighthouse_turbo_2step_total_s_median_sweep") == 0.32 and HL("lighthouse_turbo_4step_total_s_median_sweep") == 0.469


def sd_png(steps, seed=0):
    name = f"lighthouse_sd21_steps{steps:02d}_cfg7.5_seed{seed}.png"
    assert any(r["file"] == name for r in LH["steps_sweep"]) and (LIMG / name).exists(), name
    return LIMG / name


def tb_png(steps, seed=0):
    name = f"lighthouse_sdturbo_steps{steps}_seed{seed}.png"
    assert any(r["file"] == name for r in LH["turbo_sweep"]) and (LIMG / name).exists(), name
    return LIMG / name


# the timing runs (median of 3), with their breakdown
TM50, TM1 = LH["timing"]["cfg7.5_50steps"], LH["timing"]["turbo_1step"]
assert TM50["denoise_steps"] == 50 and TM50["unet_passes_per_step"] == 2                # 100 denoiser calls
assert TM1["denoise_steps"] == 1 and TM1["unet_passes_per_step"] == 1                   # 1 call
CALLS_50 = TM50["denoise_steps"] * TM50["unet_passes_per_step"]
assert round(TM50["total_s"], 2) == HL("lighthouse_total_s_50steps_cfg7.5") == 7.32
assert round(TM1["total_s"], 3) == HL("lighthouse_turbo_1step_total_s") == 0.364
PARTS50 = (TM50["text_encoder_s"], TM50["denoise_total_s"], TM50["vae_decode_s"])
PARTS1 = (TM1["text_encoder_s"], TM1["denoise_total_s"], TM1["vae_decode_s"])
assert abs(sum(PARTS50) - TM50["total_s"]) < 0.01 and abs(sum(PARTS1) - TM1["total_s"]) < 0.002
assert f"{PARTS50[1]:.2f}" == "7.04" and f"{PARTS50[2]:.3f}" == "0.265"
assert f"{PARTS1[1] * 1000:.0f}" == "76" and f"{PARTS1[2] * 1000:.0f}" == "271" and f"{PARTS1[0]:.3f}" == "0.017"
DECODE_SHARE = PARTS1[2] / TM1["total_s"]
assert 0.74 < DECODE_SHARE < 0.75                                                         # "now the decoder is the slow part"

# ---------------------------------------------------------------------------- the digit model at one step (Ep 9, 11)

GRID_IDX = list(range(10, 20))                                                          # samples 10 … 19 ask for 0 … 9
DIG = {}
for _key in ("ddpm_ddim1_w1", "flow_euler1_w1"):
    _f = np.load(ARR / f"digit_samples_{_key}.npz")
    assert "seed 1234" in str(_f["note"]) and list(_f["labels"][GRID_IDX]) == list(range(10))
    DIG[_key] = _f["samples"][GRID_IDX]
CF = {"ddim1": HL("cfid_ddim1"), "flow1": HL("cfid_flow1"), "floor": HL("cfid_real_data_floor")}
assert (f"{CF['ddim1']:.0f}", f"{CF['flow1']:.0f}", f"{CF['floor']:.2f}") == ("943", "886", "2.06")

# ---------------------------------------------------------------------------- the swirl: one step, and the teacher

Z = gaussian_noise(2000, seed=5)                                                         # toy.md's shared noise
DATA_MEAN = SWIRL.weights @ SWIRL.means
assert np.abs(DATA_MEAN).max() < 1e-12                                                    # the two arms balance
ONE_FLOW = flow_euler(exact_velocity, Z, 1).xs[-1]                                        # a perfect flow model, 1 step
assert np.abs(ONE_FLOW - DATA_MEAN).max() < 1e-9                                         # every point → the average
assert f"{swirl_distance(ONE_FLOW):.2f}" == f"{measured('toy', 'flow', 'exact', 'quality_vs_steps', '1', 'flow', 'dist'):.2f}" == "0.15"
MLP = load_swirl_model()
ONE_DDIM_MLP = swirl_distance(ddim_sample(MLP, Z, 1).xs[-1])                               # the trained ε-network, 1 step
assert abs(ONE_DDIM_MLP - measured("toy", "flow", "trained", "quality_vs_steps", "1", "ddim", "dist")) < 1e-3
assert f"{ONE_DDIM_MLP:.1f}" == "2.3"

N_TEACH = 64
FULL = ddim_sample(exact_eps, Z, N_TEACH)                                                 # the teacher: 64 DDIM steps
assert list(FULL.ts) == list(ddim_timesteps(N_TEACH))


def pick_particles(k=8):
    """Starts 0.9 < |z| < 1.9 whose 64-step path ends on an arm (< 0.04 from the curve), one per 45° of direction."""
    d = nearest_on_swirl(FULL.xs[-1])[0]
    r, ang = np.linalg.norm(Z, axis=1), np.arctan2(Z[:, 1], Z[:, 0])
    ok = np.where((d < 0.04) & (r > 0.9) & (r < 1.9))[0]
    picks = []
    for i in range(k):
        a0 = -np.pi + (i + 0.5) * 2 * np.pi / k
        picks.append(int(ok[np.argmin(np.abs(np.angle(np.exp(1j * (ang[ok] - a0)))))]))
    return np.array(picks)


PICKS = pick_particles()
assert len(set(PICKS.tolist())) == 8
PATHS = FULL.xs[:, PICKS]                                                                 # (65, 8, 2)
assert np.allclose(ddim_sample(exact_eps, Z[PICKS], N_TEACH).xs, PATHS)
TEACHER_ONE = ddim_sample(exact_eps, Z[PICKS], 1).xs[-1]                                  # the teacher's own single step
assert np.linalg.norm(TEACHER_ONE, axis=1).max() < 0.02                                   # … lands in the middle
END_DIST = nearest_on_swirl(PATHS[-1])[0]
assert END_DIST.max() < 0.04                                                              # … the path ends on an arm
LEVELS = [64, 32, 16, 8, 4, 2, 1]

# the consistency chart: the same teacher paths drawn as one coordinate against t (consistency models, Figure 2 style)
_d_end = nearest_on_swirl(FULL.xs[-1])[0]
_ok = np.where((np.abs(Z[:, 0]) < 1.85) & (_d_end < 0.04))[0]
_mid = [int(j) for j in _ok if 0.6 < abs(Z[j, 0]) < 1.3]
CH_J = _mid[int(np.argmax([abs(FULL.xs[-1, j, 0] - FULL.xs[0, j, 0]) for j in _mid]))]       # mid-height, the biggest swing
CH_BG = [int(j) for j in _ok[:28] if j != CH_J]                                             # the first 28 such paths
assert CH_J == 92 and f"{FULL.xs[0, CH_J, 0]:.2f}" == "-0.93"
CH_KS = (0, 16, 32, 44, 54)                                                                 # t = 1000, 750, 500, 312, 156
CH_NB = 44
assert [int(FULL.ts[k]) for k in CH_KS] == [1000, 750, 500, 312, 156]

# consistency: from any point of a path, the rest of the teacher's steps lead to the same end (deterministic DDIM)
for _k in (8, 32, 48, 60):
    _rest = ddim_sample(exact_eps, PATHS[_k], FULL.ts[_k:]).xs[-1]
    assert np.allclose(_rest, PATHS[-1], atol=1e-12)

# the papers (notes §12, verified)
PD = {1: 9.12, 2: 4.51, 4: 3.00, 8: 2.57}
DDIM_CIFAR = {10: 13.36, 20: 6.84, 50: 4.67, 100: 4.16}
assert "**1 → 9.12, 2 → 4.51, 4 → 3.00, 8 → 2.57**; DDIM 10 → 13.36, 20 → 6.84, 50 → 4.67, 100 → 4.16" in NOTES
assert "consistency distillation (CD, LPIPS) CIFAR-10 **1 step 3.55, 2 steps 2.93**" in NOTES
assert "a high-quality 768×768 2∼4-step LCM takes only 32 A100 GPU hours for training." in NOTES
assert "efficiently samples large-scale foundational image diffusion models in just 1–4 steps" in NOTES
assert '"During inference, our approach does not use classifier-free guidance"' in NOTES
assert "we start out with state-of-the-art samplers taking as many as 8192 steps" in NOTES

# ---------------------------------------------------------------------------- look

FLAT = set("abcdefhiklmnorstuvwxzABCDEFGHIKLMNOPRSTUVWXYZ0123456789")   # no J, Q: they descend in Baskerville
TEACH_C = MODEL          # the teacher (the original model)
STUD_C, STUD_F = BLUE, BLUE_FILL   # the student: this module's colour, "faster sampling"
PATH_C = FLOW            # the teacher's paths


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


def drive(mob, hooks, run_time=1.4, rate=m.linear):
    """Animate a progress a from 0 to 1, calling every hook(a) each frame."""

    def step(_, a):
        for hook in hooks:
            hook(a)

    return m.UpdateFromAlphaFunc(mob, step, run_time=run_time, rate_func=rate)


class Readout(m.VGroup):
    """A text display anchored at its left edge; .show(v) re-renders it when the value changes."""

    def __init__(self, left, y, fmt, value, size=30, color=TIME, display=False):
        super().__init__()
        self.anchor, self.fmt, self.size, self.color, self.display = np.array([left, y, 0.0]), fmt, size, color, display
        self.value = value
        self.words = self._make(value)
        self.add(self.words)

    def _make(self, value):
        s = self.fmt.format(value)
        words = title_text(s, self.size, self.color) if self.display else code(s, self.size, self.color, weight=m.BOLD)
        return words.move_to(self.anchor, aligned_edge=m.LEFT)

    def show(self, value):
        if value != self.value:
            self.value = value
            self.words.become(self._make(value))
        return self


def plane(center, size, lim):
    """A bare square plane for the swirl (no axes), data range [−lim, lim]²."""
    return Plot((-lim, lim), (-lim, lim), width=size, height=size, center=tuple(center), axes=False)


def at(plot, xy):
    return plot.coords(float(xy[0]), float(xy[1]))


def swirl_band(plot, width=9, opacity=0.2):
    """The swirl's two arms as wide faint TEAL strokes: where the data lives."""
    return m.VGroup(*[m.VMobject().set_points_smoothly([at(plot, p) for p in arm[::3]] + [at(plot, arm[-1])])
                      .set_stroke(DATA, width, opacity=opacity) for arm in swirl_curve(200)]).set_z_index(-1)


def polyline(plot, pts, color, width=2.6, opacity=0.95):
    pts = [at(plot, p) for p in pts]
    return m.VMobject().set_points_as_corners(pts if len(pts) > 1 else pts * 2).set_stroke(color, width, opacity=opacity)


def jump(a, b, color=STUD_C, width=3.2, tip=0.2):
    length = float(np.linalg.norm(np.asarray(b) - np.asarray(a))) or 1.0
    return m.Arrow(a, b, buff=0.04, color=color, stroke_width=width, max_tip_length_to_length_ratio=min(0.12, tip / length))


def arrow_to(a, b, color=MUTED, width=4, buff=0.12):
    return m.Arrow(a, b, buff=buff, color=color, stroke_width=width, max_tip_length_to_length_ratio=0.25)


def badge(label, color, fill, image):
    """A small role pill sitting on a picture's top-left corner."""
    p = pill(label, color, fill, 20)
    return p.move_to(image.get_corner(m.UL) + np.array([p.width / 2 + 0.14, -p.height / 2 - 0.14, 0]))


def role_card(title, sub, color, fill, width, center):
    words = m.VGroup(text(title, 24, INK, weight=m.BOLD), text(sub, 20, MUTED)).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
    box = card(width, words.height + 0.45, color, fill, radius=0.16, stroke=2.5)
    words.move_to(box).align_to(box, m.LEFT).shift(0.3 * m.RIGHT)
    return m.VGroup(box, words).move_to(center)


def project(p, a, b):
    """The closest point to p on the segment a–b."""
    ab = b - a
    n2 = float(ab @ ab)
    if n2 < 1e-18:
        return a.copy()
    return a + np.clip(float((p - a) @ ab) / n2, 0, 1) * ab


def level_colour(n):
    """The teacher's paths are LIME; each fold is a new student, a step closer to BLUE."""
    return mix(PATH_C, STUD_C, (6 - np.log2(n)) / 6)


def bump_height(length):
    return float(np.clip(0.42 * length, 0.12, 1.05))


class HopLine:
    """The teacher's network calls as arcs (hops) along the clock t = 1000 → 0; `.chain(n, m, a)` blends the arcs of
    n hops into those of m hops (a fold) — two neighbouring arcs merge into one twice as long."""

    def __init__(self, x_l, x_r, y, color):
        self.x_l, self.x_r, self.y = x_l, x_r, y
        self.xs = np.linspace(x_l, x_r, 769)
        self.curve = m.VMobject().set_stroke(color, 3.5)
        self.chain(64, 64, 0.0)

    def _heights(self, n):
        u = (self.xs - self.x_l) / (self.x_r - self.x_l) * n
        frac = np.clip(u - np.floor(np.minimum(u, n - 1e-9)), 0, 1)
        return bump_height((self.x_r - self.x_l) / n) * np.sin(np.pi * frac)

    def chain(self, n, m_, a):
        ys = (1 - a) * self._heights(n) + a * self._heights(m_)
        self.curve.set_points_as_corners([np.array([x, self.y + h, 0.0]) for x, h in zip(self.xs, ys)])
        return self.curve


# ---------------------------------------------------------------------------- the episode


class Ep12OneStepAtATime(Episode):
    number = 12

    def lesson(self):
        self.hook()
        self.one_big_step()
        self.the_idea()
        self.fold()
        self.paper_fold()
        self.consistency()
        self.adversarial()
        self.on_this_mac()
        self.where_time_goes()
        self.the_catch()
        self.recap()

    # ------------------------------------------------------------------ 0 · hook

    def hook(self):
        self.next_section("Hook", notes=(
            "Generated on this Mac (Apple M4 Max, MLX): SD 2.1 base (sd2-community mirror) with 50 steps, guidance 7.5, "
            "deterministic Euler; SD-Turbo (stabilityai/sd-turbo) with 1 step, no guidance, Euler ancestral. Prompt 'a "
            f"lighthouse on a rocky coast at dusk, oil painting', 512×512, seed 0 shown. Times: medians over seeds 0–3 of the "
            f"steps sweeps (mac_runs.json lighthouse.steps_sweep / turbo_sweep): {SD_T[50]:.3f} s vs {TB_T[1]:.3f} s. The "
            f"timing run (median of 3) gives {TM50['total_s']:.2f} s vs {TM1['total_s']:.3f} s (section 6). 50 steps × 2 "
            f"passes (guidance) = {CALLS_50} denoiser calls. The reveal is a cross-fade from decorative noise, not the run's "
            "intermediate frames. The Mac was shared while timing: ±5–10% jitter (mac_runs.md §8)."))
        h = 3.1
        xl, xr, yp = -3.35, 3.35, -0.1
        noise_l = framed(picture(tint(noise_image(64, 64, seed=3), NOISE, -2.5, 2.5), h)).move_to([xl, yp, 0])
        noise_r = framed(picture(tint(noise_image(64, 64, seed=4), NOISE, -2.5, 2.5), h)).move_to([xr, yp, 0])
        pic_l = picture(sd_png(50), h).move_to([xl, yp, 0])
        pic_r = picture(tb_png(1), h).move_to([xr, yp, 0])
        name_l = title_text("Stable Diffusion 2.1", 30, INK)
        name_r = title_text("SD-Turbo", 30, INK)
        sub_l = code("50 steps · guidance 7.5", 20, MUTED)
        sub_r = code("1 step · no guidance", 20, MUTED)
        for nm, sb, x in ((name_l, sub_l, xl), (name_r, sub_r, xr)):
            nm.move_to([x, 2.25, 0])
            sb.move_to([x, 1.82, 0])
        y_c = -2.2
        clock_l = Readout(xl - 1.55, y_c, "{:.1f} s", 0.0, 44, INK, display=True)
        clock_r = Readout(xr - 1.55, y_c, "{} s", "0.00", 44, INK, display=True)
        dx = title_text("0.25 s", 44).width + 0.3                       # the widest clock, plus a gap
        step_l = Readout(xl - 1.55 + dx, y_c + 0.2, "step {:d} of 50", 0, 20, TIME)
        step_r = Readout(xr - 1.55 + dx, y_c + 0.2, "step {:d} of 1", 0, 20, TIME)
        calls_l = code(f"{CALLS_50} denoiser calls", 20, MUTED).move_to([xl - 1.55 + dx, y_c - 0.22, 0], aligned_edge=m.LEFT)
        calls_r = code("1 denoiser call", 20, MUTED).move_to([xr - 1.55 + dx, y_c - 0.22, 0], aligned_edge=m.LEFT)
        assert calls_r.get_right()[0] < 6.6 and step_l.get_right()[0] < -0.2, (calls_r.get_right(), step_l.get_right())
        cap = cite("generated on this Mac · 512 × 512 · seed 0 · times: median of 4 seeds").move_to([0, -3.35, 0])
        mid = m.VGroup(title_text("50 → 1", 54, STUD_C), text("steps", 22, MUTED)).arrange(m.DOWN, buff=0.12).move_to([0, 0.35, 0])
        distil = title_text("distillation", 34, INK).move_to([0, -0.85, 0])
        b_l = badge("teacher", TEACH_C, AMBER_FILL, pic_l)
        b_r = badge("student", STUD_C, STUD_F, pic_r)
        assert mid.width < 2.9 and distil.width < 2.9, (mid.width, distil.width)

        def run_left(a):
            step_l.show(int(np.floor(a * 50 + 1e-9)))
            clock_l.show(round(a * SD_T[50], 1))

        self.say(
            "[Fifty steps with guidance] [take about seven seconds on this Mac] to paint our lighthouse.",
            m.AnimationGroup(self.new_header("00 · hook", "Fifty steps, or one?"), m.FadeIn(name_l), m.FadeIn(sub_l),
                             m.FadeIn(noise_l), m.FadeIn(step_l), m.FadeIn(clock_l), m.FadeIn(calls_l), m.FadeIn(cap)),
            m.AnimationGroup(drive(step_l, [run_left], run_time=3.2),
                             m.FadeIn(pic_l, run_time=3.2, rate_func=lambda a: float(a) ** 1.6)),
        )
        self.say(
            "[A distilled version of the same model] [paints] one in a single step: [a quarter of a second].",
            m.AnimationGroup(m.FadeIn(name_r), m.FadeIn(sub_r), m.FadeIn(noise_r), m.FadeIn(step_r), m.FadeIn(clock_r),
                             m.FadeIn(calls_r)),
            m.AnimationGroup(m.FadeIn(pic_r, run_time=0.35), drive(step_r, [lambda a: step_r.show(1 if a > 0.05 else 0)], 0.35)),
            drive(clock_r, [lambda a: clock_r.show(f"{a * TB_T[1]:.2f}")], run_time=0.6),
        )
        self.say(
            "[How can one step do the work of fifty]? [This is distillation]: [a student learns the teacher's shortcut].",
            m.FadeIn(mid, shift=0.15 * m.UP),
            m.FadeIn(distil, shift=0.15 * m.UP),
            m.AnimationGroup(m.FadeIn(b_l, shift=0.1 * m.DOWN), m.FadeIn(b_r, shift=0.1 * m.DOWN)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 1 · one big step

    def one_big_step(self):
        self.next_section("1 · Why not just jump?", notes=(
            "Digit model (mac_runs.md §4): samples 10–19 (labels 0–9), the same starting noise (numpy seed 1234) for both "
            f"rows. Noise-predicting U-Net, one DDIM step: cFID {CF['ddim1']:.1f}; flow-matching U-Net, one Euler step: "
            f"{CF['flow1']:.1f}; real digits {CF['floor']:.3f}. cFID = FID's formula on our own MNIST classifier's "
            "features, not Inception FID (Episode 19). The swirl (toy.md §4): 600 of toy.md's 2,000 shared noise points "
            "(gaussian_noise(2000, seed=5)); one Euler step of the EXACT flow velocity lands every point on the data's "
            f"mean (0, 0) to 1e-9 (mean distance to the curve {swirl_distance(ONE_FLOW):.3f}); one DDIM step of the "
            f"trained 45,954-parameter ε-MLP lands {ONE_DDIM_MLP:.3f} from the curve on average (toy 2.305): its clean guess "
            "divides by √ᾱ = 0.0064 (Episode 9). Paths: eight particles (illustrative choice: start radius 0.9–1.9, one per "
            "45° of direction, ending on an arm), DDIM with 64 steps and the exact denoiser — the teacher of the next "
            "sections."))
        pitch, dh = 0.6, 0.56
        x0 = -6.05
        y_a, y_b = 1.25, -0.55
        row_a = m.Group(*[picture(tint(img, DATA), dh).move_to([x0 + i * pitch, y_a, 0]) for i, img in enumerate(DIG["ddpm_ddim1_w1"])])
        row_b = m.Group(*[picture(tint(img, DATA), dh).move_to([x0 + i * pitch, y_b, 0]) for i, img in enumerate(DIG["flow_euler1_w1"])])

        def row_label(words, score, y):
            w = text(words, 22, INK, weight=m.SEMIBOLD).move_to([0, y, 0]).align_to([x0 - dh / 2, 0, 0], m.LEFT)
            s = code(f"cFID {score}", 22, RED, weight=m.BOLD)
            s.move_to([0, y, 0]).align_to([x0 + 9 * pitch + dh / 2, 0, 0], m.RIGHT)
            sit(s, baseline(w))
            assert w.get_right()[0] < s.get_left()[0] - 0.2, (w.get_right(), s.get_left())
            return m.VGroup(w, s)

        lab_a = row_label("noise-predicting model · 1 step", f"{CF['ddim1']:.0f}", y_a + 0.62)
        lab_b = row_label("flow model · 1 step", f"{CF['flow1']:.0f}", y_b + 0.62)
        tag_a = pill("speckles", RED, RED_FILL, 20).move_to([0, y_a - 0.62, 0]).align_to([x0 - dh / 2, 0, 0], m.LEFT)
        tag_b = pill("a blurry average digit", RED, RED_FILL, 20).move_to([0, y_b - 0.62, 0]).align_to([x0 - dh / 2, 0, 0], m.LEFT)
        cap_d = cite(f"generated on this Mac · digit model\ncFID: our FID-like score · real digits {CF['floor']:.2f}")
        cap_d.move_to([0, -1.9, 0]).align_to([x0 - dh / 2, 0, 0], m.LEFT)
        assert cap_d.get_right()[0] < 1.3, cap_d.get_right()
        ep_d = cite("Episodes 9 and 11").next_to(tag_a, m.RIGHT, buff=0.3)
        sit(ep_d, baseline(tag_a[1]))

        PC, PS, lim = np.array([3.75, -0.4, 0.0]), 4.3, 2.2
        pl = plane(PC, PS, lim)
        frame = m.Square(PS).set_stroke(EDGE, 2).move_to(PC)
        band = swirl_band(pl, 9, 0.2)
        t_r = m.VGroup(text("the swirl: 600 noise points", 22, INK, weight=m.BOLD),
                       cite("a perfect flow model, one Euler step")).arrange(m.DOWN, buff=0.08).next_to(frame, m.UP, buff=0.15)
        cloud = pl.cloud(Z[:600], NOISE, radius=0.03, opacity=0.85)
        ring = m.Circle(radius=0.16).set_stroke(RED, 4).move_to(at(pl, DATA_MEAN)).set_z_index(5)
        mean_l = text("every point lands on the data's average", 20, RED).next_to(frame, m.DOWN, buff=0.2)
        stat_n = title_text(f"{ONE_DDIM_MLP:.1f}", 54, RED)
        stat_l = text("average miss after one step:\nthe trained noise-predicting\nnetwork, on the swirl", 20, MUTED)
        stat = m.VGroup(stat_n, stat_l).arrange(m.RIGHT, buff=0.3).move_to([0, -2.8, 0]).align_to([x0 - dh / 2, 0, 0], m.LEFT)
        assert stat.get_right()[0] < frame.get_left()[0] - 0.3 and lab_a.get_right()[0] < frame.get_left()[0] - 0.3

        starts = m.VGroup(*[m.Dot(at(pl, q), radius=0.065, color=NOISE).set_z_index(4) for q in PATHS[0]])
        paths = m.VGroup(*[polyline(pl, PATHS[:, j], PATH_C, 2.8) for j in range(8)]).set_z_index(2)
        ends = m.VGroup(*[m.Dot(at(pl, q), radius=0.07, color=DATA).set_z_index(4) for q in PATHS[-1]])
        to_mid = m.VGroup(*[m.DashedLine(at(pl, q), at(pl, DATA_MEAN), dash_length=0.08).set_stroke(RED, 2, opacity=0.7)
                            for q in PATHS[0]]).set_z_index(1)
        path_l = text("the teacher's paths end on the arms", 20, PATH_C).next_to(frame, m.DOWN, buff=0.2)
        t_r2 = m.VGroup(text("the swirl: eight starting points", 22, INK, weight=m.BOLD),
                        cite("the teacher: 64 DDIM steps, exact denoiser")).arrange(m.DOWN, buff=0.08).next_to(frame, m.UP, buff=0.15)

        self.say(
            "[First, why not take one big step with the original model]? [Episodes nine and eleven tried it on our digits].",
            self.new_header("01 · one big step", "Why not just jump?"),
            m.AnimationGroup(m.FadeIn(lab_a), m.LaggedStart(*[m.FadeIn(p) for p in row_a], lag_ratio=0.08, run_time=1.0),
                             m.FadeIn(lab_b), m.LaggedStart(*[m.FadeIn(p) for p in row_b], lag_ratio=0.08, run_time=1.0),
                             m.FadeIn(cap_d)),
        )
        self.say(
            "[The noise-predicting model gives speckles], [and the flow model a blurry average digit].",
            m.AnimationGroup(m.FadeIn(tag_a, shift=0.1 * m.UP), m.FadeIn(ep_d)),
            m.FadeIn(tag_b, shift=0.1 * m.UP),
        )
        self.say(
            "[On the swirl, one step of a perfect flow model] [sends every point to the same place]: "
            "[the middle, the average of all the data].",
            m.AnimationGroup(m.FadeIn(frame), m.FadeIn(band), m.FadeIn(t_r), m.FadeIn(cloud)),
            cloud.to(ONE_FLOW[:600], run_time=1.6),
            m.AnimationGroup(m.Create(ring), m.FadeIn(mean_l)),
        )
        self.say(
            "[The trained noise-predicting network does no better]: [its one step lands two point three away, on average].",
            m.FadeIn(stat_n, scale=0.8),
            m.FadeIn(stat_l, shift=0.1 * m.LEFT),
        )
        self.say(
            "[Yet each starting point has a true answer]: [the end of the teacher's path, out on an arm of the swirl].",
            m.AnimationGroup(m.FadeOut(cloud), m.FadeOut(mean_l), m.FadeIn(starts, scale=0.5), m.Create(to_mid),
                             m.FadeOut(t_r), m.FadeIn(t_r2)),
            m.AnimationGroup(m.LaggedStart(*[m.Create(p) for p in paths], lag_ratio=0.12, run_time=1.8),
                             m.FadeIn(ends, scale=0.5), m.FadeIn(path_l)),
        )
        self.wait(0.3)
        self.p1 = dict(pl=pl, frame=frame, band=band, starts=starts, paths=paths, ends=ends, ring=ring, to_mid=to_mid,
                       t_r=t_r2, path_l=path_l)
        self.play(*[m.FadeOut(x) for x in (lab_a, lab_b, row_a, row_b, tag_a, tag_b, cap_d, ep_d, stat)], run_time=0.7)

    # ------------------------------------------------------------------ 2 · the idea

    def the_idea(self):
        self.next_section("2 · Teacher and student", notes=(
            "The idea of distillation (notes §12): a student learns to reproduce, in fewer network calls, where the "
            "teacher's sampler lands. Possible because DDIM's paths are deterministic (Episode 9): the same noise always "
            "gives the same picture, so 'noise → picture' is a function a network can learn. The student usually has the "
            "teacher's architecture and parameter count (Salimans & Ho 2022 §7: 'the student model has the same "
            "architecture and number of parameters as the teacher model'; SD-Turbo keeps SD 2.1's 865.9 M-parameter "
            "denoiser); what shrinks is the number of calls. BLUE arrows: the ideal student's single jumps, straight "
            "from each start to its path's end (illustration)."))
        p = self.p1
        pl = p["pl"]
        cx, w = -3.55, 5.6
        teacher = role_card("the teacher", "the original model: 64 DDIM steps per path", TEACH_C, AMBER_FILL, w, [cx, 1.65, 0])
        student = role_card("the student", "learns to land there in a single step", STUD_C, STUD_F, w, [cx, 0.3, 0])
        jumps = m.VGroup(*[jump(at(pl, a), at(pl, b)) for a, b in zip(PATHS[0], PATHS[-1])]).set_z_index(3)
        det = pill("deterministic paths: one answer per noise", PATH_C, LIME_FILL, 20).move_to([cx, -0.75, 0])
        det_ep = cite("Episode 9").next_to(det, m.DOWN, buff=0.12)
        same = m.VGroup(code("same network, same size", 20, INK), code("64 calls → 1 call", 20, STUD_C, weight=m.BOLD))
        same.arrange(m.DOWN, buff=0.14).move_to([cx, -1.95, 0])
        chips = m.VGroup(*[chip(wd, c, FILLS[c], width=1.6, height=0.66, size=26) for wd, c in
                           (("Fold", STUD_C), ("Map", PATH_C), ("Judge", TEACH_C))]).arrange(m.RIGHT, buff=0.3)
        chips.move_to([cx, -3.0, 0])
        for g in (teacher, student, det, same, chips):
            assert g.get_left()[0] > -6.62 and g.get_right()[0] < p["frame"].get_left()[0] - 0.2, (g.get_left(), g.get_right())
        hi = m.VGroup(m.Circle(radius=0.17).set_stroke(NOISE, 3).move_to(p["starts"][2]),
                      m.Circle(radius=0.17).set_stroke(DATA, 3).move_to(p["ends"][2])).set_z_index(6)

        self.say(
            "[The fix is called distillation]: [the slow model becomes a teacher], "
            "[and a student learns to land, in one step, where the teacher lands].",
            m.AnimationGroup(self.new_header("02 · the idea", "A student learns the shortcut"), m.FadeOut(p["path_l"])),
            m.AnimationGroup(m.FadeIn(teacher, shift=0.1 * m.UP), m.Circumscribe(p["paths"], color=TEACH_C, buff=0.08)),
            m.AnimationGroup(m.FadeIn(student, shift=0.1 * m.UP), m.FadeOut(p["to_mid"]),
                             m.LaggedStart(*[m.GrowArrow(j) for j in jumps], lag_ratio=0.1, run_time=1.4)),
        )
        self.say(
            "[That only works because the paths are deterministic, as in episode nine]: "
            "[each starting noise has exactly one answer].",
            m.AnimationGroup(m.FadeIn(det, shift=0.1 * m.UP), m.FadeIn(det_ep)),
            m.AnimationGroup(m.Create(hi), m.Circumscribe(jumps[2], color=STUD_C, buff=0.06)),
        )
        self.say(
            "[The student starts as a copy of the teacher, the same size]: [what shrinks is the number of calls], [not the network].",
            m.FadeIn(same[0], shift=0.1 * m.UP),
            m.FadeIn(same[1], shift=0.1 * m.UP),
            m.Circumscribe(same[0], color=INK, buff=0.08),
        )
        self.say(
            "[Three ideas make it work]: [fold the steps], [map any point to its end], [and add a critic].",
            m.Wait(0.1),
            m.FadeIn(chips[0], shift=0.15 * m.UP),
            m.FadeIn(chips[1], shift=0.15 * m.UP),
            m.FadeIn(chips[2], shift=0.15 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 3 · fold: progressive distillation

    def fold(self):
        self.next_section("3 · Fold the steps", notes=(
            "Progressive distillation (Salimans & Ho, arXiv:2202.00512, Feb 2022; ICLR 2022): a student learns to match "
            "TWO DDIM steps of the teacher with ONE of its own; then it becomes the teacher and the count halves again "
            "(their CIFAR-10 run: 8192 → … → 1; 50 thousand parameter updates per halving, 100 thousand for the 2- and "
            "1-step stages). Here: the teacher is the swirl's exact denoiser with 64 DDIM steps (t = 1000, 984, …, 0; "
            "ddim_timesteps(64)) on eight particles. Each fold draws the IDEAL student: its steps land exactly on every "
            "second point of the previous level's path (computed from the 64-step path); a trained student only "
            "approximates this, with errors that pile up over the rounds (the paper's 1-step FID 9.12 vs 2.57 at 8 "
            f"steps). The teacher's own single step lands in the middle (max |x| = {np.linalg.norm(TEACHER_ONE, axis=1).max():.4f}); "
            f"the ideal 1-step student lands on the arms (max distance to the curve {END_DIST.max():.3f}). Arcs: one per "
            "network call along the clock t."))
        PC, PS, lim = np.array([-3.8, -0.35, 0.0]), 4.75, 2.2
        pl = plane(PC, PS, lim)
        frame = m.Square(PS).set_stroke(EDGE, 2).move_to(PC)
        band = swirl_band(pl, 10, 0.18)
        top_cap = cite("teacher: exact denoiser, 64 DDIM steps").next_to(frame, m.UP, buff=0.15)
        lines = m.VGroup(*[polyline(pl, PATHS[:, j], PATH_C, 2.6) for j in range(8)]).set_z_index(2)
        vdots = [[m.Dot(at(pl, PATHS[i, j]), radius=0.032, color=INK).set_z_index(3) for i in range(N_TEACH + 1)] for j in range(8)]
        dots = m.VGroup(*[d for col in vdots for d in col])
        starts = m.VGroup(*[m.Dot(at(pl, q), radius=0.065, color=NOISE).set_z_index(4) for q in PATHS[0]])
        ends = m.VGroup(*[m.Dot(at(pl, q), radius=0.07, color=DATA).set_z_index(4) for q in PATHS[-1]])
        mid_ring = m.Circle(radius=0.16).set_stroke(RED, 4).move_to(at(pl, DATA_MEAN)).set_z_index(5)
        mid_l = text("the teacher's own one step", 20, RED).next_to(frame, m.DOWN, buff=0.18)
        illus = cite("illustration: the ideal student,\nread off the teacher's paths").next_to(frame, m.DOWN, buff=0.14)
        assert illus.get_left()[0] > -6.6 and top_cap.get_left()[0] > -6.6, (illus.get_left(), top_cap.get_left())

        # the right side: counter, ladder, the clock with one arc per call
        rx0, rx1 = -0.55, 6.35
        count = Readout(rx0, 1.85, "{:d}", 64, 64, STUD_C, display=True)
        count_l = text("steps", 24, MUTED)
        count_l.move_to([rx0 + count.words.width + 0.2, 1.85, 0], aligned_edge=m.LEFT)
        sit(count_l, baseline(count.words))
        ladder = m.VGroup(*[chip(str(n), EDGE, CARD, width=0.56, height=0.5, size=22, mono=True) for n in LEVELS])
        ladder.arrange(m.RIGHT, buff=0.08).move_to([0, 1.85, 0]).align_to([rx1, 0, 0], m.RIGHT)
        assert ladder.get_left()[0] > count_l.get_right()[0] + 0.25, (ladder.get_left(), count_l.get_right())
        lit = m.RoundedRectangle(corner_radius=0.12, width=0.7, height=0.64).set_stroke(STUD_C, 3.5).move_to(ladder[0]).set_z_index(3)
        hop = HopLine(rx0, rx1, 0.15, PATH_C)
        moving = m.VGroup(lines, dots, hop.curve)                     # one group on screen: the folds animate it
        axis = m.Line([rx0, 0.15, 0], [rx1, 0.15, 0]).set_stroke(EDGE, 3)
        t_l = code("t = 1000", 20, TIME).next_to([rx0, 0.15, 0], m.DOWN, buff=0.2).align_to([rx0, 0, 0], m.LEFT)
        t_r = code("0", 20, TIME).next_to([rx1, 0.15, 0], m.DOWN, buff=0.2).align_to([rx1, 0, 0], m.RIGHT)
        sit(t_r, baseline(t_l))
        arc_l = text("each arc: one network call", 20, MUTED).move_to([(rx0 + rx1) / 2, 0, 0])
        sit(arc_l, baseline(t_l))
        round1 = text("each round: a student learns two teacher steps in one", 22, INK).move_to([0, -1.05, 0]).align_to([rx0, 0, 0], m.LEFT)
        round2 = text("then the student becomes the next teacher", 22, MUTED).move_to([0, -1.55, 0]).align_to([rx0, 0, 0], m.LEFT)
        updates = cite("in the paper: 50,000 training updates per round").move_to([0, -2.1, 0]).align_to([rx0, 0, 0], m.LEFT)
        name = m.VGroup(title_text("Progressive distillation", 34, INK), cite("Salimans & Ho, 2022 · ICLR 2022")).arrange(
            m.DOWN, buff=0.1, aligned_edge=m.LEFT).move_to([0, -2.95, 0]).align_to([rx0, 0, 0], m.LEFT)
        for g in (round1, round2, updates, name, ladder, arc_l):
            assert g.get_right()[0] < 6.62 and g.get_left()[0] > frame.get_right()[0] + 0.25, (g.get_left(), g.get_right())
        pair = m.SurroundingRectangle(m.VGroup(m.Dot([rx0, 0.15, 0]), m.Dot([rx0 + 2 * (rx1 - rx0) / 64, 0.4, 0])),
                                      buff=0.08, corner_radius=0.06).set_stroke(STUD_C, 2.5)

        state = {"n": 64}

        def fold_to(n_from):
            n_to = n_from // 2
            stride = N_TEACH // n_from
            idx = np.arange(0, N_TEACH + 1, stride)                       # the level's vertices (path indices)
            pts = PATHS[idx]                                              # (n_from + 1, 8, 2)
            odd = np.arange(1, n_from, 2)
            target = pts.copy()
            for i in odd:
                for j in range(8):
                    target[i, j] = project(pts[i, j], pts[i - 1, j], pts[i + 1, j])
            col_from, col_to = level_colour(n_from), level_colour(n_to)
            k_to = LEVELS.index(n_to)

            def step(_, a):
                if a <= 0:
                    return
                now = (1 - a) * pts + a * target
                c = mix(col_from, col_to, a)
                for j, ln in enumerate(lines):
                    ln.set_points_as_corners([at(pl, q) for q in now[:, j]])
                    ln.set_stroke(c)
                    for i in odd:
                        d = vdots[j][idx[i]]
                        d.move_to(at(pl, now[i, j]))
                        d.set_opacity(1 - a)
                hop.chain(n_from, n_to, a)
                if a > 0.5:
                    count.show(n_to)
                    if n_to == 1 and count_l.text != "step":
                        count_l.become(sit(text("step", 24, MUTED).move_to(count_l, aligned_edge=m.LEFT), baseline(count_l)))
                    lit.move_to(ladder[k_to])
                    ladder[k_to][0].set_stroke(STUD_C)

            return m.UpdateFromAlphaFunc(moving, step, run_time=1.1, rate_func=m.smooth)

        self.say(
            "[Idea one: progressive distillation], [from Salimans and Ho in twenty twenty-two]. "
            "[Here is the teacher: sixty-four steps on each path].",
            m.AnimationGroup(self.new_header("03 · fold", "Halve the steps, again and again"), m.FadeIn(frame), m.FadeIn(band),
                             m.FadeIn(top_cap), m.FadeIn(name[0])),
            m.FadeIn(name[1]),
            m.AnimationGroup(m.FadeIn(starts), m.FadeIn(moving), m.FadeIn(ends), m.FadeIn(count), m.FadeIn(count_l),
                             m.FadeIn(ladder), m.FadeIn(lit), m.FadeIn(axis), m.FadeIn(t_l), m.FadeIn(t_r), m.FadeIn(arc_l)),
        )
        self.say(
            "[A student learns to cover two teacher steps with one], [so sixty-four steps become thirty-two].",
            m.AnimationGroup(m.FadeIn(round1, shift=0.1 * m.UP), m.Create(pair)),
            m.AnimationGroup(m.FadeOut(pair), fold_to(64)),
        )
        self.say(
            "[Then the student becomes the new teacher], [and the steps are cut in half again]: [sixteen], [eight].",
            m.FadeIn(round2, shift=0.1 * m.UP),
            m.FadeIn(updates),
            fold_to(32),
            fold_to(16),
        )
        self.say(
            "[Four], [two], [one]: [a single straight jump to where the teacher's sixty-four steps land].",
            fold_to(8),
            fold_to(4),
            fold_to(2),
            m.AnimationGroup(m.LaggedStart(*[m.Circumscribe(e, shape=m.Circle, color=STUD_C, buff=0.08, run_time=0.9) for e in ends],
                                           lag_ratio=0.1), m.Create(mid_ring), m.FadeIn(mid_l)),
        )
        self.say(
            "[Not the middle, where the teacher's own single step lands]. "
            "[These folds are drawn from the teacher's paths]: [a real student only learns to approximate them].",
            m.Circumscribe(mid_ring, shape=m.Circle, color=RED, buff=0.08),
            m.AnimationGroup(m.FadeOut(mid_l), m.FadeIn(illus)),
            m.Circumscribe(name, color=STUD_C, buff=0.1),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 3b · the paper's numbers

    def paper_fold(self):
        self.next_section("3b · The paper's numbers", notes=(
            "Salimans & Ho 2022, abstract: 'we start out with state-of-the-art samplers taking as many as 8192 steps, and "
            "are able to distill down to models taking as few as 4 steps without losing much perceptual quality; achieving, "
            "for example, a FID of 3.0 on CIFAR-10 in 4 steps.' Table 2 (CIFAR-10, Inception FID, read in the PDF): "
            "progressive distillation 1 / 2 / 4 / 8 model evaluations → 9.12 / 4.51 / 3.00 / 2.57; DDIM (Song et al., the "
            "undistilled deterministic sampler) 10 / 20 / 50 / 100 → 13.36 / 6.84 / 4.67 / 4.16. Notes Appendix A: the "
            "consistency-models paper lists 8.34 for PD at 1 step (another evaluation); we quote the paper's own 9.12."))
        lx = -6.45
        big = title_text("8,192 → 4", 54, STUD_C)
        big_l = text("steps, in the paper's\nCIFAR-10 runs", 22, INK)
        quote = text("“without losing much\nperceptual quality”", 22, INK)
        src = cite("Salimans & Ho, 2022")
        left = m.VGroup(big, big_l, quote, src).arrange(m.DOWN, buff=0.3, aligned_edge=m.LEFT).move_to([0, 0.1, 0]).align_to([lx, 0, 0], m.LEFT)
        vals = [PD[k] for k in (1, 2, 4, 8)] + [DDIM_CIFAR[k] for k in (10, 20, 50, 100)]
        bars = Bars(vals, labels=["1", "2", "4", "8", "10", "20", "50", "100"], top=14, width=7.6, height=3.4,
                    colors=[STUD_C] * 4 + [TEACH_C] * 4, gap=0.32, fmt="{:.2f}", center=(2.55, 0.1, 0))
        assert bars.get_left()[0] > left.get_right()[0] + 0.5, (bars.get_left(), left.get_right())
        y_grp = bars.y0 - 0.8
        g1 = text("distilled student", 22, STUD_C, weight=m.BOLD).move_to([(bars.xs[0] + bars.xs[3]) / 2, y_grp, 0])
        g2 = text("DDIM, not distilled", 22, TEACH_C, weight=m.BOLD).move_to([(bars.xs[4] + bars.xs[7]) / 2, y_grp, 0])
        sit(g2, baseline(g1))
        title = text("CIFAR-10 photos · FID, lower is better", 22, INK).move_to([2.55, 2.3, 0])
        src2 = cite("Salimans & Ho, 2022 · Table 2").move_to([2.55, -3.35, 0])
        assert g2.get_right()[0] < 6.62 and title.get_right()[0] < 6.62 and src2.get_right()[0] < 6.62
        steps_l = cite("steps").next_to(bars.labels, m.LEFT, buff=0.25)
        h4 = m.SurroundingRectangle(m.VGroup(bars.bars[2], bars.numbers[2]), buff=0.08, corner_radius=0.08).set_stroke(GOOD, 3)
        h50 = m.SurroundingRectangle(m.VGroup(bars.bars[6], bars.numbers[6]), buff=0.08, corner_radius=0.08).set_stroke(TEACH_C, 3)

        def grow(i):
            return m.AnimationGroup(m.GrowFromEdge(bars.bars[i], m.DOWN), m.FadeIn(bars.labels[i]), m.FadeIn(bars.numbers[i]))

        self.say(
            "[In the paper, samplers with eight thousand one hundred ninety-two steps came down to four], "
            "[without losing much quality].",
            m.AnimationGroup(self.new_header("03 · fold", "The paper's numbers"), m.FadeIn(big, shift=0.15 * m.UP), m.FadeIn(big_l)),
            m.AnimationGroup(m.FadeIn(quote, shift=0.1 * m.UP), m.FadeIn(src)),
        )
        self.say(
            "[On small colour photos, four distilled steps scored three point zero], "
            "[better than fifty DDIM steps at four point six seven]. [One step scored nine point one two].",
            m.AnimationGroup(m.FadeIn(title), m.FadeIn(bars.zero), m.FadeIn(g1), m.FadeIn(src2), m.FadeIn(steps_l),
                             m.LaggedStart(*[grow(i) for i in range(4)], lag_ratio=0.2, run_time=1.4), m.Create(h4)),
            m.AnimationGroup(m.FadeIn(g2), m.LaggedStart(*[grow(i) for i in range(4, 8)], lag_ratio=0.2, run_time=1.4), m.Create(h50)),
            m.Circumscribe(m.VGroup(bars.bars[0], bars.numbers[0]), color=RED, buff=0.08),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 4 · map: consistency models

    def consistency(self):
        self.next_section("4 · Map any point to its end", notes=(
            "Consistency models (Song, Dhariwal, Chen & Sutskever, arXiv:2303.01469, Mar 2023; ICML 2023): a network "
            "f_θ(x_t, t) that maps ANY point of a probability-flow ODE trajectory to the trajectory's start (the clean "
            "sample); trained by self-consistency, f(x_{t_{n+1}}, t_{n+1}) = f(x_{t_n}, t_n), with the neighbour from one "
            "teacher ODE-solver step (consistency distillation) or without a teacher (consistency training). CIFAR-10 "
            "one-step FID 3.55 (CD), 'the new state-of-the-art … for one-step generation' (abstract, 2023). On screen: the "
            "IDEAL consistency function, read off the swirl teacher's 64-step DDIM paths (exact denoiser; from any point, "
            "the remaining DDIM steps reach the same end — asserted), drawn as the first coordinate of each point against t "
            f"(like the paper's Figure 2): {len(CH_BG)} paths, the highlighted one starts at x = {FULL.xs[0, CH_J, 0]:.2f} "
            f"and ends at {FULL.xs[-1, CH_J, 0]:.2f}; points at t = {', '.join(str(int(FULL.ts[k])) for k in CH_KS)}; "
            f"neighbours at t = {int(FULL.ts[CH_NB])} and {int(FULL.ts[CH_NB + 1])}. Latent "
            "consistency models (Luo et al., arXiv:2310.04378, Oct 2023): consistency distillation in Stable Diffusion's "
            "latent space with guidance folded in; 2–4 steps; 'a high-quality 768×768 2∼4-step LCM takes only 32 A100 GPU "
            "hours for training'; LCM-LoRA (Nov 2023) packages it as an adapter."))
        CC, CW, CH = np.array([-3.4, -0.5, 0.0]), 5.7, 4.2
        ch = Plot((0, 1000), (-2.0, 2.0), width=CW, height=CH, center=tuple(CC), axes=False)
        tp = lambda t, x: ch.coords(1000 - float(t), float(x))  # noqa: E731  (time runs left to right: noise → data)
        frame = m.Rectangle(width=CW, height=CH).set_stroke(EDGE, 2).move_to(CC)
        zero = m.DashedLine(tp(1000, 0), tp(0, 0), dash_length=0.08).set_stroke(FAINT, 1.5)
        curve = lambda j: [tp(t, v) for t, v in zip(FULL.ts, FULL.xs[:, j, 0])]  # noqa: E731
        bg = m.VGroup(*[m.VMobject().set_points_as_corners(curve(j)).set_stroke(PATH_C, 1.6, opacity=0.38) for j in CH_BG]).set_z_index(1)
        hl = m.VMobject().set_points_as_corners(curve(CH_J)).set_stroke(PATH_C, 4).set_z_index(2)
        x_hl = FULL.xs[:, CH_J, 0]
        end_dot = m.Dot(tp(0, x_hl[-1]), radius=0.09, color=DATA).set_z_index(5)
        top_cap = cite("the teacher's paths against time:\none coordinate of each point").next_to(frame, m.UP, buff=0.15)
        lab_l = m.VGroup(code("t = 1000", 20, TIME), text("noise", 20, NOISE)).arrange(m.RIGHT, buff=0.2)
        lab_l.next_to(frame, m.DOWN, buff=0.16).align_to(frame, m.LEFT)
        lab_r = m.VGroup(text("data", 20, DATA), code("t = 0", 20, TIME)).arrange(m.RIGHT, buff=0.2)
        lab_r.next_to(frame, m.DOWN, buff=0.16).align_to(frame, m.RIGHT)
        sit(lab_l[1], baseline(lab_l[0]))
        sit(lab_r[0], baseline(lab_r[1]))
        sit(lab_r[1], baseline(lab_l[0]))
        sit(lab_r[0], baseline(lab_l[0]))
        fan_pts = m.VGroup(*[m.Dot(tp(FULL.ts[k], x_hl[k]), radius=0.075, color=mix(NOISE, DATA, i / 5)).set_z_index(4)
                             for i, k in enumerate(CH_KS)])
        fan_arrows = m.VGroup(*[jump(tp(FULL.ts[k], x_hl[k]), tp(0, x_hl[-1]), STUD_C, 3.0) for k in CH_KS]).set_z_index(3)
        others = m.VGroup()
        for j in CH_BG[::4]:
            xs_ = FULL.xs[:, j, 0]
            others.add(*[jump(tp(FULL.ts[k], xs_[k]), tp(0, xs_[-1]), STUD_C, 1.8, 0.14) for k in (0, 40)])
        others.set_opacity(0.55).set_z_index(3)
        k_n = CH_NB
        nb = m.VGroup(*[m.Circle(radius=0.14).set_stroke(INK, 3).move_to(tp(FULL.ts[k], x_hl[k])) for k in (k_n, k_n + 1)]).set_z_index(6)
        nb_arrows = m.VGroup(*[jump(tp(FULL.ts[k], x_hl[k]), tp(0, x_hl[-1]), INK, 2.6) for k in (k_n, k_n + 1)]).set_z_index(4)
        illus = cite("illustration: the ideal map").move_to([CC[0], -3.4, 0])

        rx0 = -0.15
        head1 = text("any point on a path → the path's end", 24, INK, weight=m.BOLD).move_to([0, 1.95, 0]).align_to([rx0, 0, 0], m.LEFT)
        f1 = m.VGroup(tex(r"f_\theta(x_t,\,t)", 40, STUD_C), text("= the end of its path", 24, INK)).arrange(m.RIGHT, buff=0.25)
        f1.move_to([0, 1.15, 0]).align_to([rx0, 0, 0], m.LEFT)
        one = code("1 call: pure noise → a sample", 22, STUD_C, weight=m.BOLD).move_to([0, 0.45, 0]).align_to([rx0, 0, 0], m.LEFT)
        f2 = tex(r"f_\theta(x_{t_{n+1}},\,t_{n+1}) \;=\; f_\theta(x_{t_n},\,t_n)", 36, INK)
        f2.move_to([0, -0.45, 0]).align_to([rx0, 0, 0], m.LEFT)
        f2_l = text("neighbours on one path must agree", 22, MUTED).move_to([0, -1.05, 0]).align_to([rx0, 0, 0], m.LEFT)
        how = cite("the neighbour: one teacher ODE step (distillation)\nor no teacher at all (consistency training)")
        how.move_to([0, -1.75, 0]).align_to([rx0, 0, 0], m.LEFT)
        stat_n = title_text("3.55", 54, GOOD)
        stat_l = text("FID with one step on CIFAR-10:\na one-step record in March 2023", 22, INK)
        stat = m.VGroup(stat_n, stat_l).arrange(m.RIGHT, buff=0.3).move_to([0, -2.75, 0]).align_to([rx0, 0, 0], m.LEFT)
        src = cite("Song, Dhariwal, Chen & Sutskever, 2023 · ICML 2023").move_to([0, -3.38, 0]).align_to([rx0, 0, 0], m.LEFT)
        for g in (head1, f1, one, f2, f2_l, how, stat, src):
            assert g.get_right()[0] < 6.62 and g.get_left()[0] > frame.get_right()[0] + 0.25, (g.get_left(), g.get_right())

        lcm_t = m.VGroup(title_text("Latent consistency models", 32, INK), cite("Luo et al., October 2023")).arrange(
            m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        lcm_rows = m.VGroup(text("the same idea in Stable Diffusion's\nlatent space, with guidance folded in", 22, INK),
                            code("2–4 steps", 24, STUD_C, weight=m.BOLD),
                            text("trained in 32 A100 GPU hours (768 × 768)", 22, INK),
                            cite("LCM-LoRA: the speed-up as a plug-in adapter")).arrange(m.DOWN, buff=0.2, aligned_edge=m.LEFT)
        lcm_body = m.VGroup(lcm_t, lcm_rows).arrange(m.DOWN, buff=0.3, aligned_edge=m.LEFT)
        lcm_box = card(lcm_body.width + 0.7, lcm_body.height + 0.6, STUD_C, CARD, radius=0.16, stroke=2.5).move_to(lcm_body)
        lcm = m.VGroup(lcm_box, lcm_body).move_to([0, -1.3, 0]).align_to([rx0, 0, 0], m.LEFT)
        assert lcm.get_right()[0] < 6.62 and lcm.get_bottom()[1] > -3.5 and lcm.get_top()[1] < f1.get_bottom()[1] - 0.15, \
            (lcm.get_right(), lcm.get_bottom(), lcm.get_top(), f1.get_bottom())

        self.say(
            "[Idea two: consistency models, by Song and colleagues in twenty twenty-three]. "
            "[Here are the teacher's paths, drawn against time].",
            m.AnimationGroup(self.new_header("04 · map", "Any point, straight to its end"), m.FadeIn(src)),
            m.AnimationGroup(m.FadeIn(frame), m.FadeIn(zero), m.FadeIn(top_cap), m.FadeIn(lab_l), m.FadeIn(lab_r),
                             m.LaggedStart(*[m.Create(c_) for c_ in bg], lag_ratio=0.04, run_time=1.8)),
        )
        self.say(
            "[Pick points anywhere along one path]: [early, middle or late, they all lead to the same end].",
            m.AnimationGroup(m.Create(hl, run_time=1.2), m.FadeIn(end_dot), m.FadeIn(head1),
                             m.LaggedStart(*[m.FadeIn(d, scale=0.5) for d in fan_pts], lag_ratio=0.2, run_time=1.2)),
            m.LaggedStart(*[m.GrowArrow(a_) for a_ in fan_arrows], lag_ratio=0.15, run_time=1.4),
        )
        self.say(
            "[So train one network to map any point straight to the end of its path]: "
            "[one call turns pure noise into a sample].",
            m.AnimationGroup(m.FadeIn(f1, shift=0.1 * m.UP), m.FadeIn(illus),
                             m.LaggedStart(*[m.GrowArrow(a_) for a_ in others], lag_ratio=0.05, run_time=1.4)),
            m.AnimationGroup(m.FadeIn(one, shift=0.1 * m.UP), m.Circumscribe(fan_arrows[0], color=STUD_C, buff=0.06)),
        )
        self.say(
            "[The training signal]: [two neighbouring points on a path must give the same answer].",
            m.AnimationGroup(m.FadeIn(f2, shift=0.1 * m.UP), m.FadeIn(f2_l), m.FadeOut(others), m.FadeOut(fan_arrows)),
            m.AnimationGroup(m.Create(nb), m.FadeIn(nb_arrows), m.FadeIn(how)),
        )
        self.say(
            "[On small colour photos, one step scored three point five five], [the best one-step score at the time].",
            m.FadeIn(stat_n, scale=0.8),
            m.FadeIn(stat_l, shift=0.1 * m.LEFT),
        )
        self.say(
            "[Latent consistency models brought this to Stable Diffusion in October twenty twenty-three]: "
            "[two to four steps], [after thirty-two GPU hours of training].",
            m.AnimationGroup(m.FadeOut(m.VGroup(f2, f2_l, how, stat, one, src)), m.FadeIn(lcm_box), m.FadeIn(lcm_t),
                             m.FadeIn(lcm_rows[0])),
            m.FadeIn(lcm_rows[1], shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(lcm_rows[2], shift=0.1 * m.UP), m.FadeIn(lcm_rows[3])),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 5 · judge: adversarial distillation

    def adversarial(self):
        self.next_section("5 · Add a critic", notes=(
            "Adversarial Diffusion Distillation (Sauer, Lorenz, Blattmann & Rombach, arXiv:2311.17042, 28 Nov 2023) → SDXL "
            "Turbo: the student generates in 1–4 steps and is trained with two signals: an adversarial loss from a "
            "discriminator (trainable heads on a frozen pretrained ViT feature network) for realism, and a distillation "
            "loss to a frozen teacher's denoised output (eq. 1: L = L_adv + λ·L_distill). 'During inference, our approach "
            "does not use classifier-free guidance' — so one network call per step, not two. Episode 3: GANs, the "
            "generator–critic tug of war. The picture is SD-Turbo's one-step lighthouse (seed 0), generated on this Mac: "
            "SD-Turbo is SD 2.1 trained with ADD (mac_runs.md §5)."))
        y0 = 0.35
        noise = framed(picture(tint(noise_image(64, 64, seed=4), NOISE, -2.5, 2.5), 1.5), NOISE).move_to([-5.8, y0, 0])
        noise_l = code("noise", 20, NOISE).next_to(noise, m.DOWN, buff=0.14)
        student = role_card("the student", "1 to 4 steps", STUD_C, STUD_F, 2.4, [-2.9, y0, 0])
        pic = framed(picture(tb_png(1), 2.0), STUD_C).move_to([0.45, y0, 0])
        pic_l = code("its picture", 20, MUTED).next_to(pic, m.DOWN, buff=0.14)
        critic = role_card("the critic", "does it look real?", INK, CARD_HI, 3.5, [4.65, 1.55, 0])
        teacher = role_card("the frozen teacher", "what would I have made?", TEACH_C, AMBER_FILL, 3.5, [4.65, -0.85, 0])
        a1 = arrow_to(noise.get_right(), student.get_left(), NOISE)
        a2 = arrow_to(student.get_right(), pic.get_left(), STUD_C)
        a3 = arrow_to(pic.get_right() + [0, 0.3, 0], critic.get_left(), MUTED)
        a4 = arrow_to(pic.get_right() + [0, -0.3, 0], teacher.get_left(), MUTED)
        back_c = m.CurvedArrow(critic.get_top() + [-0.6, 0.05, 0], student.get_top() + [0.0, 0.08, 0], angle=0.22 * PI_)
        back_c.set_stroke(INK, 3)
        back_t = m.CurvedArrow(teacher.get_bottom() + [-0.6, -0.05, 0], student.get_bottom() + [0.0, -0.08, 0], angle=-0.22 * PI_)
        back_t.set_stroke(TEACH_C, 3)
        lab_c = text("realism", 22, INK, weight=m.BOLD).move_to([0.75, 2.45, 0])
        lab_t = text("faithfulness", 22, TEACH_C, weight=m.BOLD).move_to([0.75, -2.0, 0])
        noguide = pill("no guidance: one call per step, not two", GOOD, GOOD_FILL, 20).move_to([0, -3.0, 0])
        src = cite("Sauer, Lorenz, Blattmann & Rombach, 2023 · SDXL Turbo, November 2023").move_to([0, -3.0, 0])
        ep3 = cite("Episode 3's adversarial game").next_to(critic, m.DOWN, buff=0.14)
        assert critic.get_right()[0] < 6.62 and ep3.get_right()[0] < 6.62, (critic.get_right(), ep3.get_right())

        self.say(
            "[Idea three: adversarial distillation, behind SDXL Turbo in November twenty twenty-three].",
            m.AnimationGroup(self.new_header("05 · judge", "A critic and a teacher"), m.FadeIn(src)),
        )
        self.say(
            "[The student paints in one to four steps]; [a critic judges whether the picture looks real], "
            "[and the frozen teacher shows what it would have painted].",
            m.AnimationGroup(m.FadeIn(noise), m.FadeIn(noise_l), m.GrowArrow(a1), m.FadeIn(student), m.GrowArrow(a2),
                             m.FadeIn(pic), m.FadeIn(pic_l)),
            m.AnimationGroup(m.GrowArrow(a3), m.FadeIn(critic, shift=0.1 * m.LEFT), m.FadeIn(ep3)),
            m.AnimationGroup(m.GrowArrow(a4), m.FadeIn(teacher, shift=0.1 * m.LEFT)),
        )
        self.say(
            "[Both train the student]: [realism from the critic, episode three's adversarial trick], [faithfulness from the teacher].",
            m.AnimationGroup(pulse(a3, INK), pulse(a4, TEACH_C)),
            m.AnimationGroup(m.Create(back_c), m.FadeIn(lab_c)),
            m.AnimationGroup(m.Create(back_t), m.FadeIn(lab_t)),
        )
        self.say(
            "[It needs no guidance]: [one network call per step, not two].",
            m.AnimationGroup(m.FadeOut(src), m.FadeIn(noguide, shift=0.1 * m.UP)),
            m.Circumscribe(pic, color=STUD_C, buff=0.08),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 6 · on this Mac

    def on_this_mac(self):
        self.next_section("6 · On this Mac", notes=(
            "Generated on this Mac (mac_runs.md §5): SD 2.1 base (sd2-community mirror of Stability's model, CreativeML "
            "Open RAIL++-M) with the library's deterministic Euler sampler and guidance 7.5; SD-Turbo (stabilityai/sd-turbo "
            "@ b261bac6, Stability AI Community License) with Euler ancestral and no guidance; both 512×512, seed 0 shown. "
            f"Same denoiser architecture: {UNET:,} parameters in both (fp16). Times: medians over seeds 0–3 (steps sweeps): "
            + ", ".join(f"SD 2.1 {s} → {SD_T[s]:.3f} s" for s in SD_T) + "; "
            + ", ".join(f"turbo {s} → {TB_T[s]:.3f} s" for s in TB_T)
            + ". SD 2.1 at 1 step still makes 2 passes (guidance). We did not distill anything: SD-Turbo was downloaded. "
            "Stability's card: 'SD-Turbo is a distilled version of Stable Diffusion 2.1' trained with ADD (the 2.1 variant "
            "is not stated); the base model we ran has the same architecture and parameter count, so it stands in for the "
            "teacher. The card: 'SD-Turbo does not make use of guidance_scale or negative_prompt'; 512×512 preferred."))
        h = 1.85
        xs = (-2.45, -0.1, 2.25)
        y1, y2 = 1.15, -1.45
        rows = {}
        for key, y, steps, png, times in (("sd", y1, (1, 4, 50), sd_png, SD_TXT), ("tb", y2, (1, 2, 4), tb_png, TB_TXT)):
            pics = m.Group()
            labs = m.VGroup()
            for x, s in zip(xs, steps):
                fr = framed(picture(png(s), h), STUD_C if key == "tb" else EDGE).move_to([x, y, 0])
                pics.add(fr)
                lab = text(f"{s} step{'s' if s > 1 else ''} · {times[s]}", 20, INK if key == "tb" else MUTED)
                labs.add(lab.next_to(fr, m.DOWN, buff=0.12))
            for lab in labs[1:]:
                sit(lab, baseline(labs[0]))
            rows[key] = (pics, labs)
        lx = -6.45

        def row_label(title, color, lines, y):
            g = m.VGroup(text(title, 24, color, weight=m.BOLD), *[code(s, 20, MUTED) for s in lines])
            g.arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT).move_to([0, y, 0]).align_to([lx, 0, 0], m.LEFT)
            return g

        lab1 = row_label("the teacher", TEACH_C, ("SD 2.1 base", "Euler", "guidance 7.5"), y1)
        lab2 = row_label("the student", STUD_C, ("SD-Turbo", "Euler ancestral", "no guidance"), y2)
        assert lab1.get_right()[0] < rows["sd"][0][0].get_left()[0] - 0.2 and lab2.get_right()[0] < rows["tb"][0][0].get_left()[0] - 0.2, \
            (lab1.get_right(), lab2.get_right(), rows["sd"][0][0].get_left())
        rx = rows["sd"][0][-1].get_right()[0] + 0.35
        top, bot = rows["sd"][0].get_top()[1], rows["tb"][0].get_bottom()[1]
        bracket = m.VMobject().set_points_as_corners([[rx, top, 0], [rx + 0.18, top, 0], [rx + 0.18, bot, 0], [rx, bot, 0]])
        bracket.set_stroke(TEACH_C, 3)
        same = m.VGroup(text("the same\ndenoiser", 24, INK, weight=m.BOLD), title_text(f"{UNET / 1e6:.1f} M", 34, TEACH_C),
                        code("parameters", 20, MUTED)).arrange(m.DOWN, buff=0.14, aligned_edge=m.LEFT)
        same.move_to([0, (top + bot) / 2 + 1.0, 0]).align_to([rx + 0.45, 0, 0], m.LEFT)
        dl = m.VGroup(text("downloaded,\nnot distilled\nby us", 20, INK), cite("Stability AI\nCommunity\nLicense"))
        dl.arrange(m.DOWN, buff=0.14, aligned_edge=m.LEFT).move_to([0, (top + bot) / 2 - 1.2, 0]).align_to([rx + 0.45, 0, 0], m.LEFT)
        assert same.get_right()[0] < 6.62 and dl.get_right()[0] < 6.62, (same.get_right(), dl.get_right())
        assert same.get_bottom()[1] > dl.get_top()[1] + 0.3, (same.get_bottom(), dl.get_top())
        cap = cite("generated on this Mac · 512 × 512 · seed 0 · times: median of 4 seeds").move_to([-0.2, -3.38, 0])
        assert rows["tb"][1].get_bottom()[1] > cap.get_top()[1] + 0.05, (rows["tb"][1].get_bottom(), cap.get_top())
        noguide = pill("no guidance", GOOD, GOOD_FILL, 20)

        sd_p, sd_l = rows["sd"]
        tb_p, tb_l = rows["tb"]
        self.say(
            "[Back to the lighthouse]. [The distilled model, SD-Turbo, is Stable Diffusion two point one after adversarial distillation], "
            "[with a denoiser of the very same size].",
            m.AnimationGroup(self.new_header("06 · on this Mac", "Fifty steps against one"), m.FadeIn(cap)),
            m.AnimationGroup(m.FadeIn(lab1), m.FadeIn(lab2), m.FadeIn(sd_p[2]), m.FadeIn(sd_l[2]), m.FadeIn(tb_p[0]), m.FadeIn(tb_l[0])),
            m.AnimationGroup(m.Create(bracket), m.FadeIn(same, shift=0.1 * m.LEFT)),
        )
        self.say(
            "[We downloaded it]: [we did not distill anything ourselves].",
            m.FadeIn(dl[0], shift=0.1 * m.UP),
            m.FadeIn(dl[1]),
        )
        self.say(
            "[The original model, at one step, paints a blur]; [at four, a soft lighthouse]; "
            "[at fifty, a sharp one, in about seven seconds].",
            m.AnimationGroup(m.FadeIn(sd_p[0]), m.FadeIn(sd_l[0])),
            m.AnimationGroup(m.FadeIn(sd_p[1]), m.FadeIn(sd_l[1])),
            m.Circumscribe(m.Group(sd_p[2], sd_l[2]), color=TEACH_C, buff=0.08),
        )
        self.say(
            "[The student paints a full lighthouse in one step], [in two], [or in four]: "
            "[a quarter to half a second], [with no guidance].",
            m.Circumscribe(m.Group(tb_p[0], tb_l[0]), color=STUD_C, buff=0.08),
            m.AnimationGroup(m.FadeIn(tb_p[1], scale=0.97), m.FadeIn(tb_l[1])),
            m.AnimationGroup(m.FadeIn(tb_p[2], scale=0.97), m.FadeIn(tb_l[2])),
            m.Circumscribe(tb_l, color=STUD_C, buff=0.1),
            m.Circumscribe(lab2[3], color=GOOD, buff=0.06),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 6b · where the time goes

    def where_time_goes(self):
        self.next_section("6b · Where the time goes", notes=(
            "The timing runs (mac_runs.md §5, median of 3 runs each, one image, 512×512): SD 2.1 base with 50 steps and "
            f"guidance (2 passes batched per step): text encoder {PARTS50[0]:.4f} s, denoiser {PARTS50[1]:.3f} s, VAE decode "
            f"{PARTS50[2]:.3f} s, total {TM50['total_s']:.3f} s. SD-Turbo, 1 step, no guidance: text {PARTS1[0]:.4f} s, "
            f"denoiser {PARTS1[1]:.4f} s, decode {PARTS1[2]:.4f} s, total {TM1['total_s']:.4f} s — the decoder is "
            f"{100 * DECODE_SHARE:.0f}% of it. The turbo sweep's 1-step median ({TB_T[1]:.3f} s) is lower: the VAE decode "
            "varied 0.17–0.27 s across runs (the Mac was shared; ±5–10% jitter). Colours: text PINK, denoiser AMBER, "
            "decoder TEAL."))
        x0, span = -4.55, 10.9
        scale = span / TM50["total_s"]
        cols = (PROMPT, MODEL, DATA)

        def stacked(parts, y, k=1.0, height=0.5):
            segs = m.VGroup()
            x = x0
            for v, c in zip(parts, cols):
                w = max(v * scale * k, 0.004)
                segs.add(m.Rectangle(width=w, height=height).set_stroke(c, 2).set_fill(FILLS[c], 1).move_to([x + w / 2, y, 0]))
                x += w
            return segs

        y1, y2, y3 = 1.4, -0.05, -1.75
        bar50 = stacked(PARTS50, y1)
        bar1 = stacked(PARTS1, y2)
        zoom = 20
        bar1z = stacked(PARTS1, y3, zoom, 0.62)
        assert abs(bar50.get_right()[0] - (x0 + span)) < 0.01 and bar1z.get_right()[0] < 6.62, bar1z.get_right()

        def head(words, total, y, color):
            t = text(words, 22, color, weight=m.BOLD).move_to([0, y + 0.52, 0]).align_to([x0, 0, 0], m.LEFT)
            n = code(total, 22, INK, weight=m.BOLD).next_to(t, m.RIGHT, buff=0.35)
            sit(n, baseline(t))
            return m.VGroup(t, n)

        h50 = head("the teacher · 50 steps with guidance", f"{TM50['total_s']:.2f} s", y1, TEACH_C)
        h1 = head("the student · 1 step", f"{TM1['total_s']:.3f} s", y2, STUD_C)
        in50 = code(f"denoiser {PARTS50[1]:.2f} s · {CALLS_50} calls", 22, INK).move_to(bar50[1])
        dec50 = code(f"decode {PARTS50[2]:.2f} s", 20, DATA).next_to(bar50, m.DOWN, buff=0.14).align_to(bar50, m.RIGHT)
        guides = m.VGroup(m.DashedLine(bar1.get_corner(m.DL), bar1z.get_corner(m.UL), dash_length=0.07).set_stroke(FAINT, 2),
                          m.DashedLine(bar1.get_corner(m.DR), bar1z.get_corner(m.UR), dash_length=0.07).set_stroke(FAINT, 2))
        zl = code(f"× {zoom}", 20, MUTED).next_to(bar1z, m.LEFT, buff=0.2)
        d1 = code(f"{PARTS1[1]:.3f} s", 22, INK, weight=m.BOLD).move_to(bar1z[1])
        v1 = code(f"decode {PARTS1[2]:.3f} s", 22, INK, weight=m.BOLD).move_to(bar1z[2])
        t1 = code(f"text {PARTS1[0]:.3f} s", 20, PROMPT).next_to(bar1z[0], m.DOWN, buff=0.14).align_to(bar1z, m.LEFT)
        share = text(f"the decoder: {100 * DECODE_SHARE:.0f}% of the time", 24, DATA, weight=m.BOLD)
        share.next_to(bar1z, m.DOWN, buff=0.55).align_to(bar1z, m.RIGHT)
        assert d1.width < bar1z[1].width - 0.1 and in50.width < bar50[1].width - 0.2, (d1.width, bar1z[1].width)
        legend = m.VGroup(*[m.VGroup(m.Square(0.22).set_stroke(c, 2).set_fill(FILLS[c], 1), code(w, 20, MUTED)).arrange(m.RIGHT, buff=0.12)
                            for c, w in zip(cols, ("text encoder", "denoiser", "decoder"))]).arrange(m.RIGHT, buff=0.45)
        legend.move_to([0, 2.35, 0]).align_to([x0, 0, 0], m.LEFT)
        cap = cite("timing runs on this Mac: median of 3 · the Mac was shared: ±5–10%").move_to([0, -3.38, 0])
        assert share.get_bottom()[1] > cap.get_top()[1] + 0.05

        self.say(
            "[Where does the time go]? [Fifty guided steps spend seven seconds in the denoiser], [and a quarter of a second decoding].",
            m.AnimationGroup(self.new_header("06 · on this Mac", "Where the time goes"), m.FadeIn(legend), m.FadeIn(cap)),
            m.AnimationGroup(m.FadeIn(h50), m.GrowFromEdge(bar50, m.LEFT), m.FadeIn(in50)),
            m.FadeIn(dec50, shift=0.1 * m.UP),
        )
        self.say(
            "[One step spends seventy-six milliseconds in the denoiser], [and about two hundred seventy decoding the picture]: "
            "[now the decoder is the slow part].",
            m.AnimationGroup(m.FadeIn(h1), m.GrowFromEdge(bar1, m.LEFT), m.Create(guides), m.FadeIn(zl),
                             m.GrowFromEdge(bar1z[0], m.LEFT), m.GrowFromEdge(bar1z[1], m.LEFT), m.FadeIn(d1), m.FadeIn(t1)),
            m.AnimationGroup(m.GrowFromEdge(bar1z[2], m.LEFT), m.FadeIn(v1)),
            m.AnimationGroup(m.FadeIn(share, shift=0.1 * m.UP), m.Circumscribe(bar1z[2], color=DATA, buff=0.06)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 6c · the catch

    def the_catch(self):
        self.next_section("6c · The catch", notes=(
            "Seeds 0–3, generated on this Mac: SD 2.1 base, 50 steps, guidance 7.5 (top) vs SD-Turbo, 1 step, no guidance "
            "(bottom). Seen in the images (our judgement, not a metric): turbo seed 3 has no lighthouse — the tower melts "
            "into a rock spire; seeds 0–2 do. The literature (notes §12): distilled models usually lose diversity and/or "
            "free use of guidance and negative prompts; ADD does not use classifier-free guidance at inference. Four seeds "
            "are too few to measure diversity. SD-Turbo is trained for 512×512."))
        h = 1.62
        xs = [-2.75 + i * 1.85 for i in range(4)]
        y1, y2 = 1.2, -1.15
        top = m.Group(*[framed(picture(sd_png(50, s), h)).move_to([x, y1, 0]) for s, x in enumerate(xs)])
        bot = m.Group(*[framed(picture(tb_png(1, s), h), STUD_C).move_to([x, y2, 0]) for s, x in enumerate(xs)])
        seeds = m.VGroup(*[code(f"seed {s}", 20, MUTED).next_to(bot[s], m.DOWN, buff=0.12) for s in range(4)])
        lx = -6.45
        l1 = m.VGroup(text("the teacher", 24, TEACH_C, weight=m.BOLD), code("50 steps", 20, MUTED)).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        l2 = m.VGroup(text("the student", 24, STUD_C, weight=m.BOLD), code("1 step", 20, MUTED)).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        l1.move_to([0, y1, 0]).align_to([lx, 0, 0], m.LEFT)
        l2.move_to([0, y2, 0]).align_to([lx, 0, 0], m.LEFT)
        assert l1.get_right()[0] < top[0].get_left()[0] - 0.2
        miss = m.SurroundingRectangle(bot[3], buff=0.06).set_stroke(RED, 4)
        miss_l = text("no lighthouse", 22, RED, weight=m.BOLD).next_to(bot[3], m.UP, buff=0.12)
        rx = top[-1].get_right()[0] + 0.45
        notes = m.VGroup(text("reported limits:", 22, INK, weight=m.BOLD),
                         text("less variety", 22, INK), text("no guidance or\nnegative prompts", 22, INK),
                         text("trained for\n512 × 512", 22, INK)).arrange(m.DOWN, buff=0.22, aligned_edge=m.LEFT)
        notes.move_to([0, 0.45, 0]).align_to([rx, 0, 0], m.LEFT)
        few = text("our 4 seeds:\ntoo few to\nmeasure variety", 20, MUTED).next_to(notes, m.DOWN, buff=0.4).align_to(notes, m.LEFT)
        assert notes.get_right()[0] < 6.62 and few.get_bottom()[1] > -3.5, (notes.get_right(), few.get_bottom())
        cap = cite("generated on this Mac · 512 × 512 · same prompt").move_to([-0.9, -3.38, 0])
        # the miss label sits in the gap between the rows; check it clears both
        assert miss_l.get_bottom()[1] > bot[3].get_top()[1] and miss_l.get_top()[1] < top[3].get_bottom()[1], (miss_l.get_top(), top[3].get_bottom())

        self.say(
            "[The catch]: across four seeds, one of the one-step pictures [lost its lighthouse in the rocks].",
            m.AnimationGroup(self.new_header("06 · on this Mac", "The catch"), m.FadeIn(l1), m.FadeIn(l2), m.FadeIn(cap),
                             m.LaggedStart(*[m.FadeIn(p_) for p_ in top], lag_ratio=0.15, run_time=1.2),
                             m.LaggedStart(*[m.FadeIn(p_) for p_ in bot], lag_ratio=0.15, run_time=1.2), m.FadeIn(seeds)),
            m.AnimationGroup(m.Create(miss), m.FadeIn(miss_l)),
        )
        self.say(
            "[Papers also report less variety, and no guidance or negative prompts]: [four seeds are too few for us to measure that].",
            m.FadeIn(notes, shift=0.1 * m.UP),
            m.FadeIn(few, shift=0.1 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ recap and the end of Module 3

    def recap(self):
        self.next_section("Recap · end of Module 3", notes=(
            "Fold (progressive distillation: halve the steps, round after round), map (consistency models: any point of a "
            "path → its end), judge (adversarial distillation: a critic plus a frozen teacher). On this Mac: the same "
            f"865.9 M-parameter denoiser, called once (SD-Turbo, {TB_T[1]:.2f} s) instead of {CALLS_50} times (SD 2.1, 50 "
            f"guided steps, {SD_T[50]:.1f} s; sweep medians). Module 3 (Faster Sampling): Episode 9 DDIM (skip steps, "
            "deterministic), Episode 10 solvers and schedules (better steps), Episode 11 flow matching (straighter "
            "training paths), Episode 12 distillation (learn the shortcut). Next: Module 4, Steering the Picture "
            "(Episode 13, Asking for a Digit)."))
        halo = glow(STUD_C, radius=4.0)
        words = m.VGroup(*[title_text(w, 58, c) for w, c in (("Fold", STUD_C), ("Map", PATH_C), ("Judge", TEACH_C))])
        subs = m.VGroup(*[code(s, 20, MUTED) for s in ("64 → 32 → … → 1", "any point → its end", "a critic + a teacher")])
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
        row = m.VGroup(words, dots, subs).move_to(1.75 * m.UP)
        assert row.width < 12.6 and all(a.get_right()[0] + 0.3 < b.get_left()[0] for a, b in zip(subs, subs[1:]))
        tag = text(f"The same denoiser, called once instead of {CALLS_50} times.", 28, INK).next_to(row, m.DOWN, buff=0.5)

        mod = text("Module 3 · Faster Sampling · Episodes 9–12", 22, STUD_C, weight=m.BOLD)
        steps = [("9", "skip steps"), ("10", "better steps"), ("11", "straighter paths"), ("12", "learn the shortcut")]
        chips_ = m.VGroup()
        for n, s in steps:
            words_ = m.VGroup(code(n, 20, STUD_C, weight=m.BOLD), text(s, 20, INK)).arrange(m.RIGHT, buff=0.16)
            box = card(words_.width + 0.45, 0.66, STUD_C if n == "12" else EDGE, STUD_F if n == "12" else CARD, radius=0.14, stroke=2)
            words_.move_to(box)
            sit(words_[0], baseline(words_[1]))
            chips_.add(m.VGroup(box, words_))
        chips_.arrange(m.RIGHT, buff=0.38)
        arrows = m.VGroup(*[arrow_to(a.get_right(), b.get_left(), MUTED, 3, 0.06) for a, b in zip(chips_, chips_[1:])])
        module = m.VGroup(mod, m.VGroup(chips_, arrows)).arrange(m.DOWN, buff=0.3).next_to(tag, m.DOWN, buff=0.7)
        assert module.width < 12.8, module.width
        mascot = pixel(0.42)
        nxt = pill("NEXT · MODULE 4 · STEERING THE PICTURE", PROMPT, PINK_FILL, 20, mono=True)
        foot = m.VGroup(mascot, nxt).arrange(m.RIGHT, buff=0.3).next_to(module, m.DOWN, buff=0.55)
        assert foot.get_bottom()[1] > -3.55, foot.get_bottom()

        self.say(
            "[To sum up]: [fold], [map], [judge].",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.3 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(dots[0]), m.FadeIn(words[1], shift=0.3 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(dots[1]), m.FadeIn(words[2], shift=0.3 * m.UP), m.FadeIn(subs[2])),
        )
        self.say(
            "[Three ways to teach a student the teacher's shortcut]: [the same denoiser, called once instead of a hundred times].",
            m.Circumscribe(row, color=STUD_C, buff=0.12),
            m.Write(tag),
        )
        self.say(
            "[That closes module three]: [skip steps], [take better steps], [straighten the paths], [and learn the shortcut].",
            m.FadeIn(mod, shift=0.1 * m.UP),
            m.FadeIn(chips_[0], shift=0.1 * m.UP),
            m.AnimationGroup(m.GrowArrow(arrows[0]), m.FadeIn(chips_[1], shift=0.1 * m.UP)),
            m.AnimationGroup(m.GrowArrow(arrows[1]), m.FadeIn(chips_[2], shift=0.1 * m.UP)),
            m.AnimationGroup(m.GrowArrow(arrows[2]), m.FadeIn(chips_[3], shift=0.1 * m.UP)),
        )
        self.say(
            "[Next, we stop asking how fast], [and start asking what to paint].",
            m.FadeIn(mascot, shift=0.2 * m.UP),
            m.FadeIn(nxt, shift=0.2 * m.UP),
        )
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, tag, module, foot)], run_time=0.9)
