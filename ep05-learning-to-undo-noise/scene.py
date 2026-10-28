"""Out of the Noise · Episode 5 · Learning to Undo Noise (Module 2 · Learning to Denoise — the module's first episode)

Training is a regression. Running example: the digit model, trained on this Mac (research/experiments/mac_runs.md):
one real training step on the first 7 in MNIST (training image 15, the digit Episode 4 dissolved) at t = 500, with the
run's own noise ε and the trained U-Net's own guess ε̂ (arrays/digit_unet_activations.npz); the loss of that step;
DDPM's Algorithm 1 as an 8-line loop and its eq. 14; why the target is the noise (x̂₀ follows in one line; the target
is unit-size static at every t); x₀- and v-prediction (Salimans & Ho 2022); the run itself (3,940,801 parameters,
10,000 steps × 256 = 42.7 epochs, 31.8 min with MLX, loss 0.81 → 0.0202: the real curve); the swirl's loss floor
(0.1581 exact vs 0.1588 trained); the model's clean guess x̂₀ along one generated 7 (blurry at high t, speckled at
t = 999); and why one jump from pure noise fails with ε-prediction (÷ √ᾱ_1000 ≈ 0.0064 magnifies errors 157×; one DDIM
step lands 2.3 away from the swirl; the flow model's one step lands on the average, Episode 11).

Every number on screen is computed below from diffusion_reference.py, the run's arrays or measured(...) and asserted,
or cited with its source.

Sources (research/notes.md §5, §0 row 6, §8, Appendix A rows "5, 11 · v" and "19 · DDPM FID reference set", Appendix C
row 5; read October 2026):
- Ho, Jain & Abbeel 2020 (DDPM, arXiv:2006.11239): Algorithm 1, eq. 14 (L_simple), §3.2, Table 2 (ε-prediction with
  L_simple: IS 9.46 / FID 3.17; with the true bound L: 7.67 / 13.51), App. B (35.7M-parameter CIFAR-10 U-Net, batch 128,
  800k steps, 10.6 h on a TPU v3-8). FID is against the training set (5.24 vs the test set).
- Salimans & Ho 2022 (Progressive Distillation, arXiv:2202.00512), §4: v ≡ α_t ε − σ_t x (ours: √ᾱ·ε − √(1 − ᾱ)·x₀);
  Table 1 (undistilled CIFAR-10, stochastic sampler, FID): ε with SNR weighting (= L_simple) 2.59, x with truncated SNR
  2.53, v with truncated SNR 2.45.
- research/experiments/mac_runs.md §3 and mac_runs.json (headline, digit_models.ddpm.loss_curve), arrays/
  digit_unet_activations.npz (the training step), arrays/digit_forward_process.npz (the run's ᾱ, 0-based t),
  arrays/digit_trajectory_7_more_seeds.npz (seed 37, DDPM 1,000 steps: x_t and x̂₀ at every step).
- research/experiments/toy.md §2 and §4, toy.json (headline loss_floor_eps, final_loss_eps, mlp_params,
  flow.trained.quality_vs_steps["1"], training.eps.loss_curve).

Illustrative (said "for example" or plainly a picture): the second training step's inputs in "One training step" (a
different MNIST digit, t = 180, fresh noise; no model output is shown for it); the random-pick wobble of the t marker.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403
from diffusion_reference import *  # noqa: E402,F403

PREFIX = "ep05"
ARR = EXPERIMENTS / "arrays"
H = lambda *keys: measured("mac_runs", "headline", *keys)  # noqa: E731
TOY = lambda *keys: measured("toy", "headline", *keys)  # noqa: E731
LIGHT_INK = mix(INK, BG, 0.15)

# ---------------------------------------------------------------------------- the run's schedule (0-based t)

FWD = np.load(ARR / "digit_forward_process.npz")
AB_RUN = FWD["abar"].astype(np.float64)                    # the run's ᾱ at code step t = 0 … 999
assert len(AB_RUN) == 1000 and np.allclose(AB_RUN, ALPHA_BAR[1:], rtol=1e-12)   # code t ≡ the paper's t + 1

# ---------------------------------------------------------------------------- one real training step (t = 500)

ACT = np.load(ARR / "digit_unet_activations.npz")
IMAGES, LABELS = load_mnist("train")
DIGIT_INDEX = int(np.flatnonzero(LABELS == 7)[0])
X0 = IMAGES[DIGIT_INDEX] / 127.5 - 1.0                     # the model's range, −1 … 1
T_STEP = int(ACT["t"])
EPS = ACT["eps_true"].astype(np.float64)
EPS_HAT = ACT["eps_hat"].astype(np.float64)
X_T = ACT["x_t"].astype(np.float64)
AB_STEP = float(AB_RUN[T_STEP])
SIG, NOI = np.sqrt(AB_STEP), np.sqrt(1 - AB_STEP)
assert DIGIT_INDEX == 15 and T_STEP == 500                                        # "the first seven", "t is 500"
assert np.abs(ACT["x0"] - X0).max() < 1e-6                                        # the run's digit is that 7
assert np.abs(X_T - (SIG * X0 + NOI * EPS)).max() < 1e-5                          # x_t = √ᾱ·x₀ + √(1 − ᾱ)·ε
assert f"{SIG:.2f}" == "0.28" and f"{NOI:.2f}" == "0.96"                          # "0.28 … plus 0.96"
assert X0.size == 784                                                             # "784 pixels"
SQ_ERR = (EPS - EPS_HAT) ** 2
LOSS_ONE = float(SQ_ERR.mean())
assert f"{LOSS_ONE:.4f}" == "0.0043"
ON_INK = X0 > 0                                                                   # pixels on the 7's strokes
assert SQ_ERR[ON_INK].mean() > 5 * SQ_ERR[X0 < -0.99].mean()                     # "the errors sit on the strokes"
X0_HAT = (X_T - NOI * EPS_HAT) / SIG                                              # the network's picture of the 7
assert np.mean((X0_HAT - X0) ** 2) < 0.06                                         # "a little blurred, but a seven"

# the second, illustrative step: another digit, another t, fresh noise (no model output shown)
NEXT_INDEX = int(np.flatnonzero(LABELS == 3)[0])
NEXT_T = 180
X0_NEXT = IMAGES[NEXT_INDEX] / 127.5 - 1.0
EPS_NEXT = np.random.default_rng(55).standard_normal((28, 28))
XT_NEXT = np.sqrt(AB_RUN[NEXT_T]) * X0_NEXT + np.sqrt(1 - AB_RUN[NEXT_T]) * EPS_NEXT

# the target is unit-size static at every step
COL_T = [100, 500, 900]
COL_EPS = [np.random.default_rng(20 + i).standard_normal((28, 28)) for i in range(3)]
COL_XT = [np.sqrt(AB_RUN[t]) * X0 + np.sqrt(1 - AB_RUN[t]) * e for t, e in zip(COL_T, COL_EPS)]
COL_STD = [float(e.std()) for e in COL_EPS]
assert all(abs(s - 1) < 0.05 for s in COL_STD)                                    # "unit-size static"

# x₀- and v-prediction targets for the same step
V_TGT = v_target(X0, EPS, AB_STEP)
assert np.allclose(SIG * X_T - NOI * V_TGT, X0)                                   # x₀ = √ᾱ·x_t − √(1 − ᾱ)·v
SH_FID = {"eps": 2.59, "x0": 2.53, "v": 2.45}                                     # Salimans & Ho 2022, Table 1 (cited)

# ---------------------------------------------------------------------------- DDPM (cited, notes §5 / §8)

DDPM_PARAMS_M, DDPM_STEPS, DDPM_BATCH, DDPM_HOURS = 35.7, 800_000, 128, 10.6
DDPM_FID, DDPM_IS, DDPM_FID_BOUND = 3.17, 9.46, 13.51

# ---------------------------------------------------------------------------- the digit model's run on this Mac

RUN = measured("mac_runs", "digit_models", "ddpm")
PARAMS = H("digit_unet_params")
STEPS, BATCH, EPOCHS = H("digit_train_steps"), H("digit_train_batch"), H("digit_train_epochs")
MINUTES, PEAK_GB, FINAL = H("digit_ddpm_train_minutes"), H("digit_train_peak_mem_gb"), H("digit_ddpm_final_loss")
assert PARAMS == RUN["params"] == 3_940_801 and f"{PARAMS / 1e6:.1f}" == "3.9"     # "three point nine million"
assert STEPS == 10_000 and BATCH == 256 and EPOCHS == 42.7                         # "about forty-three passes"
assert abs(STEPS * BATCH / 60_000 - EPOCHS) < 0.05 and round(EPOCHS) == 43
assert MINUTES == 31.8 and round(MINUTES) == 32 and abs(RUN["wall_min"] - MINUTES) < 0.05   # "thirty-two minutes"
assert round(DDPM_PARAMS_M * 1e6 / PARAMS) == 9                                   # "about nine times smaller"
assert RUN["ema"] == 0.999 and RUN["label_dropout"] == 0.1 and RUN["lr_peak"] == 5e-4   # "one label in ten"
CURVE = RUN["loss_curve"]
C_STEP, C_LOSS = np.array(CURVE["step"], float), np.array(CURVE["loss"], float)
assert f"{C_LOSS[0]:.2f}" == "0.81" and C_STEP[0] == 50                          # "starts at zero point eight one"
I150 = int(np.flatnonzero(C_STEP == 150)[0])
assert C_LOSS[I150] < 0.04 < C_LOSS[I150 - 1]                                     # "within 150 steps, below 0.04"
assert f"{C_LOSS[-1]:.3f}" == "0.020" and FINAL == 0.0202 and C_STEP[-1] == STEPS  # "zero point zero two zero"
assert f"{RUN['final_loss']:.4f}" == "0.0202"

# ---------------------------------------------------------------------------- the swirl's loss floor (toy.json)

FLOOR, MLP_LOSS, MLP_PARAMS = TOY("loss_floor_eps"), TOY("final_loss_eps"), TOY("mlp_params")
assert (FLOOR, MLP_LOSS) == (0.1581, 0.1588) and MLP_PARAMS == 45_954 == count_params(load_weights())
with np.errstate(all="ignore"):                       # (macOS numpy prints spurious matmul warnings)
    _rng = np.random.default_rng(11)
    _x0 = swirl(100_000, seed=11)
    _t = _rng.integers(1, T + 1, len(_x0))
    _eps = _rng.standard_normal(_x0.shape)
    FLOOR_CHECK = float(np.mean((_eps - exact_eps(q_sample(_x0, _t, _eps), _t)) ** 2))
assert abs(FLOOR_CHECK - FLOOR) < 0.002                                           # the floor, recomputed here
SW_CURVE = np.array(measured("toy", "training", "eps", "loss_curve"), float)
SW_STEPS = np.array(measured("toy", "training", "eps", "loss_curve_steps"), float)
SW_SMOOTH = np.convolve(SW_CURVE, np.ones(10) / 10, "valid")                     # rolling mean of 10 logged points
SW_SMOOTH_STEPS = SW_STEPS[9:]
assert abs(SW_SMOOTH[-1] - MLP_LOSS) < 0.001 and SW_SMOOTH.min() > FLOOR - 0.0005

# a noisy swirl point that could have come from either arm (the exact posterior, t = 100)
POST_T = 150
POST_X = np.array([[0.075, -0.875]])
with np.errstate(all="ignore"):
    POST = mixture_posterior(POST_X, np.sqrt(ALPHA_BAR[POST_T]), np.sqrt(1 - ALPHA_BAR[POST_T]))
POST_RESP = POST.resp[0]
POST_ARM0 = float(POST_RESP[SWIRL.labels == 0].sum())
POST_GUESS = POST.x0[0]
assert 0.4 < POST_ARM0 < 0.6 and nearest_on_swirl(POST_GUESS)[0][0] > 0.08        # split, and the average is off the arms
assert np.linalg.norm(POST_GUESS - POST_X[0]) > 0.3                               # the guess sits well away from x_t

# ---------------------------------------------------------------------------- the model's clean guess, one generated 7

TRAJ = np.load(ARR / "digit_trajectory_7_more_seeds.npz")
TRAJ_SEED = 37
TRAJ_TS = TRAJ["ddpm_ddpm1000_w1_t"].astype(int)
TRAJ_XT = TRAJ[f"seed{TRAJ_SEED}_ddpm_ddpm1000_w1_x_t"].astype(np.float64)
TRAJ_X0 = TRAJ[f"seed{TRAJ_SEED}_ddpm_ddpm1000_w1_x0_hat"].astype(np.float64)
TRAJ_FINAL = TRAJ[f"seed{TRAJ_SEED}_ddpm_ddpm1000_w1_final"].astype(np.float64)
assert int(TRAJ["label"]) == 7 and TRAJ_SEED in TRAJ["seeds"] and len(TRAJ_TS) == 1000 and TRAJ_TS[0] == 999
STRIP_T = [999, 900, 700, 500, 300, 100]
STRIP_I = [int(np.flatnonzero(TRAJ_TS == t)[0]) for t in STRIP_T]


def greyness(img):
    """Share of pixels that are neither ink nor background (|value| < 0.6): a blurred digit has many."""
    return float(np.mean(np.abs(np.clip(img, -1, 1)) < 0.6))


_grey = [greyness(TRAJ_X0[i]) for i in STRIP_I]
assert min(_grey[1], _grey[2]) > 3 * max(_grey[4], _grey[5])                      # "blurry at high noise, sharpens"
_bg = np.clip(TRAJ_FINAL, -1, 1) < -0.9                                          # the final digit's background
_speck = [float(np.clip(TRAJ_X0[i], -1, 1)[_bg].std()) for i in STRIP_I]
assert _speck[0] > 1.3 * _speck[1] and _speck[0] > 3 * _speck[3]                   # t = 999: "just speckles"

# ---------------------------------------------------------------------------- one jump from pure noise (the swirl)

SQ_AB_END = float(np.sqrt(ALPHA_BAR[T]))
GAIN = 1 / SQ_AB_END
assert f"{SQ_AB_END:.4f}" == "0.0064" and round(GAIN) == 157                      # "0.0064", "a hundred fifty-seven"
JUMP_Z = gaussian_noise(2000, seed=5)                                             # toy.md's fixed noise
with np.errstate(all="ignore"):
    _model = load_swirl_model()
    JUMP_EPS = ddim_sample(_model, JUMP_Z, steps=1).xs[-1]
    JUMP_FLOW = flow_euler(load_swirl_flow(), JUMP_Z, steps=1).xs[-1]
    EPS_ERR = float(np.linalg.norm(_model(JUMP_Z, T) - exact_eps(JUMP_Z, T), axis=1).mean())
    JUMP_DIST = swirl_distance(JUMP_EPS)
    FLOW_DIST = swirl_distance(JUMP_FLOW)
_q = measured("toy", "flow", "trained", "quality_vs_steps", "1")
assert abs(JUMP_DIST - _q["ddim"]["dist"]) < 1e-3 and f"{JUMP_DIST:.1f}" == "2.3"   # "two point three away"
assert abs(FLOW_DIST - _q["flow"]["dist"]) < 1e-3 and f"{FLOW_DIST:.2f}" == "0.13"
assert f"{EPS_ERR:.2f}" == "0.02"                                                 # "off by only zero point zero two"
N_SHOW, VIEW_R = 600, 10.0
_far = np.abs(JUMP_EPS[:N_SHOW]).max(1) > VIEW_R
N_BEYOND = int(_far.sum())
assert 0 < N_BEYOND <= 6
JUMP_SHOW = ~_far

# ---------------------------------------------------------------------------- look

FLAT = set("abcdefhiklmnorstuvwxzABCDEFGHIJKLMNOPRSTUVWXYZ0123456789")


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
    """Teal while the picture outweighs the noise (SNR > 1, the run's 0-based t), violet once the noise wins."""
    return TEAL if float(snr(AB_RUN[int(round(t))])) > 1 else VIOLET


def noisy_rgb(x, t):
    """A noisy picture as Episode 4 drew it: clipped to −1 … 1, tinted by which side is winning."""
    return tint(x, side(t), -1, 1)


def texture(rgb):
    """The RGBA texture picture() makes from a small image (crisp pixels), for swapping in place."""
    k = max(1, int(np.ceil(540 / max(rgb.shape[:2]))))
    arr = np.repeat(np.repeat(rgb, k, axis=0), k, axis=1)
    rgba = np.concatenate([arr, np.full(arr.shape[:2] + (1,), 255, np.uint8)], axis=-1)
    rgba.flags.writeable = False
    return rgba


def swap(image, rgb):
    image.paint = image.paint.but(texture=texture(rgb))


def drive(mob, t0, t1, hooks, run_time=1.4, rate=m.smooth):
    """Animate a step t from t0 to t1, calling every hook(t) each frame."""

    def step(_, a):
        t = t0 + (t1 - t0) * a
        for hook in hooks:
            hook(t)

    return m.UpdateFromAlphaFunc(mob, step, run_time=run_time, rate_func=rate)


class Counter(m.VGroup):
    """A step counter "t = 250" (mono, BLUE: time), left-aligned or centred at a fixed point."""

    def __init__(self, anchor, t=0, size=30, color=TIME, prefix="t = ", centred=False):
        super().__init__()
        self.anchor = np.array(anchor, dtype=float)
        self.size, self.color, self.prefix, self.centred = size, color, prefix, centred
        self.words = self._make(t)
        self.add(self.words)

    def _make(self, t):
        words = code(f"{self.prefix}{int(round(t))}", self.size, self.color, weight=m.BOLD)
        return words.move_to(self.anchor) if self.centred else words.move_to(self.anchor, aligned_edge=m.LEFT)

    def set_t(self, t):
        self.words.become(self._make(t))


class StepLine(m.VGroup):
    """The random step t on a BLUE number line from 0 to 1000, with a marker and a readout above it."""

    def __init__(self, left, right, y, t=0):
        super().__init__()
        self.lo, self.hi, self.y = left, right, y
        self.track = m.Line([left, y, 0], [right, y, 0]).set_stroke(TIME, 3)
        self.ends = m.VGroup()
        for v in (0, 1000):
            p = np.array([self.at(v), y, 0])
            self.ends.add(m.Line(p + [0, 0.09, 0], p - [0, 0.09, 0]).set_stroke(TIME, 3),
                          code(str(v), 20, MUTED).next_to(p, m.DOWN, buff=0.16))
        self.marker = m.Dot(radius=0.11, color=TIME).move_to([self.at(t), y, 0]).set_z_index(2)
        self.readout = Counter(((left + right) / 2, y + 0.45, 0), t, 26, centred=True)
        self.add(self.track, self.ends, self.marker, self.readout)

    def at(self, t):
        return self.lo + (self.hi - self.lo) * t / 1000

    def set_t(self, t):
        self.marker.move_to([self.at(t), self.y, 0])
        self.readout.set_t(t)


def formula(*parts, size=48, color=INK):
    """A formula made of separately coloured parts (kit's tex() takes one string)."""
    return m.MathTex(*parts, font_size=size, color=color)


def tagged(image, label, buff=0.16, color=EDGE):
    """A framed picture with a label (a formula or words) under it: Group(frame, image, label)."""
    pic = framed(image, color)
    label.next_to(pic, m.DOWN, buff=buff)
    pic.add(label)
    return pic


def step_chip(i, label, color, lit):
    words = text(f"{i} · {label}", 22, INK if lit else MUTED, weight=m.BOLD)
    box = card(words.width + 0.55, 0.52, color if lit else EDGE, FILLS.get(color, CARD) if lit else CARD, radius=0.26,
               stroke=2)
    return m.VGroup(box, words.move_to(box))


def network_card(width=2.5, height=1.6, scale=0.27):
    """The denoiser: a small U-Net drawing on a dark card outlined in AMBER (the model's colour)."""
    box = card(width, height, MODEL, CARD, radius=0.18, stroke=3)
    net = UNet(sizes=(28, 14, 7), channels=(32, 64, 128), scale=scale, labels=False).move_to(box)
    group = m.VGroup(box, net)
    group.box, group.net = box, net
    return group


class PlaneView(m.VGroup):
    """2D points drawn at center + scale·xy (a bare plane: no axes). `put(xy, scale)` moves them."""

    def __init__(self, xy, center, scale, color=DATA, radius=0.03, opacity=0.9):
        super().__init__()
        self.c, self.s = np.array(center, dtype=float), scale
        for _ in range(len(xy)):
            self.add(m.Dot(radius=radius, color=color).set_opacity(opacity))
        self.put(xy)

    def put(self, xy, scale=None):
        if scale is not None:
            self.s = scale
        for d, (x, y) in zip(self, np.asarray(xy, dtype=float)):
            d.move_to(self.c + self.s * np.array([x, y, 0.0]))
        return self


# ---------------------------------------------------------------------------- the episode


class Ep05LearningToUndoNoise(Episode):
    number = 5

    def lesson(self):
        self.hook()
        self.one_step()
        self.the_loop()
        self.why_noise()
        self.targets()
        self.on_this_mac()
        self.the_floor()
        self.the_guess()
        self.one_jump()
        self.recap()

    # ------------------------------------------------------------------ 0 · hook

    def hook(self):
        self.next_section("Hook", notes=(
            "Callback to Episode 4: the first 7 in the MNIST training set (index 15) dissolved with the run's own noise "
            "(the ε of the training step below), t = 0 … 999 (the run's 0-based steps), TEAL while SNR > 1, VIOLET after. "
            f"Right: a 7 generated on this Mac by the digit model (DDPM ancestral sampling, 1,000 steps, seed {TRAJ_SEED}, "
            "label 7; arrays/digit_trajectory_7_more_seeds.npz), every x_t frame played back. The forward process needs "
            "no learning (Ho et al. 2020, §2); the reverse needs the trained network."))
        fwd = picture(noisy_rgb(X0, 0), 2.4)
        left = framed(fwd).move_to([-4.4, 0.35, 0])
        rev = picture(noisy_rgb(TRAJ_XT[0], 999), 2.4)
        right = framed(rev, MODEL).move_to([4.4, 0.35, 0])
        l_top = text("add noise", 26, NOISE, weight=m.BOLD).next_to(left, m.UP, buff=0.25)
        r_top = text("remove noise", 26, MODEL, weight=m.BOLD).next_to(right, m.UP, buff=0.25)
        sit(r_top, baseline(l_top))
        l_count = Counter((left.get_x(), left.get_bottom()[1] - 0.38, 0), 0, 28, centred=True)
        r_count = Counter((right.get_x(), right.get_bottom()[1] - 0.38, 0), 999, 28, centred=True)
        l_cap = cite(f"MNIST training image {DIGIT_INDEX}").next_to(l_count, m.DOWN, buff=0.18)
        r_cap = cite(f"generated on this Mac\nDDPM, 1,000 steps · seed {TRAJ_SEED}").next_to(r_count, m.DOWN, buff=0.18)
        net = network_card(2.3, 1.5, 0.25).move_to([0, 0.35, 0])
        net_lab = text("a trained network", 22, MODEL).next_to(net, m.DOWN, buff=0.22)
        go_in = arrow(left[0].get_right(), net.box.get_left(), NOISE, buff=0.15)
        go_out = arrow(net.box.get_right(), right[0].get_left(), MODEL, buff=0.15)
        no_learn = pill("no learning needed", GREEN, GREEN_FILL, 20).move_to([-4.4, -2.75, 0])

        q1 = pill("plain regression", MODEL, AMBER_FILL, 24)
        q2 = pill("guess the noise, square the error", INK, CARD, 24)
        qs = m.VGroup(q1, q2).arrange(m.RIGHT, buff=0.4).move_to([0, -2.75, 0])

        def fwd_at(t):
            swap(fwd, noisy_rgb(np.sqrt(AB_RUN[int(t)]) * X0 + np.sqrt(1 - AB_RUN[int(t)]) * EPS, t))
            l_count.set_t(t)

        def rev_at(k):
            i = min(int(k), 999)
            swap(rev, noisy_rgb(np.clip(TRAJ_XT[i], -1, 1), TRAJ_TS[i]) if i < 999 else tint(TRAJ_FINAL, DATA, -1, 1))
            r_count.set_t(TRAJ_TS[i] if i < 999 else 0)

        self.say(
            "[Last time, we destroyed a picture with noise], [and that needed no learning at all].",
            m.AnimationGroup(self.new_header("00 · hook", "Learning to undo noise"),
                             m.FadeIn(m.Group(left, l_top, l_count, l_cap), shift=0.2 * m.UP)),
            m.Succession(drive(fwd, 0, 999, [fwd_at], run_time=2.2), m.FadeIn(no_learn, shift=0.1 * m.UP)),
        )
        self.say(
            "[Undoing it takes a network]. [Here is our digit model, trained on this Mac], [turning static back into a seven].",
            m.AnimationGroup(m.GrowArrow(go_in), m.FadeIn(net, scale=0.9), m.FadeIn(net_lab)),
            m.AnimationGroup(m.GrowArrow(go_out), m.FadeIn(m.Group(right, r_top, r_count, r_cap), shift=0.2 * m.UP)),
            drive(rev, 0, 999.9, [rev_at], run_time=3.2, rate=m.linear),
        )
        self.say(
            "[Today: how that network learns.] [The surprise is that it's plain regression]: "
            "[guess the noise, then square the error].",
            m.AnimationGroup(m.FadeOut(no_learn), net.net.animate(rate_func=m.there_and_back).scale(1.08)),
            m.FadeIn(q1, shift=0.1 * m.UP),
            m.FadeIn(q2, shift=0.1 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 1 · one training step

    def one_step(self):
        self.next_section("1 · One training step", notes=(
            f"A real training step of the digit model (arrays/digit_unet_activations.npz): MNIST training image "
            f"{DIGIT_INDEX} (the first 7) in the model's range −1…1, the run's 0-based step t = {T_STEP} (ᾱ = "
            f"{AB_STEP:.5f}, = ALPHA_BAR[{T_STEP + 1}] in the reference's 1-based steps), the run's noise ε, x_t = "
            f"{SIG:.4f}·x₀ + {NOI:.4f}·ε (asserted), and the trained (EMA) U-Net's ε̂. Loss = mean over the 784 pixels of "
            f"(ε − ε̂)² = {LOSS_ONE:.5f} (the run's loss is this mean, over 256 digits per step). Squared error per "
            f"pixel: mean {SQ_ERR[ON_INK].mean():.4f} on the 7's strokes vs {SQ_ERR[X0 < -0.99].mean():.5f} on plain "
            "background. Pictures: ε and ε̂ shown over −2.5…2.5; the error map over 0…0.05. The second step's inputs "
            f"(MNIST training image {NEXT_INDEX}, a 3, t = {NEXT_T}, noise seed 55) are illustrative: no model output "
            "is shown for it. The weight update is Adam on this loss (the run: lr 5e-4, grad-norm clip 1.0)."))
        names = [("picture", DATA), ("step t", TIME), ("noise", NOISE), ("guess", MODEL), ("compare", RED)]
        chips = m.VGroup(*[step_chip(i + 1, n, c, False) for i, (n, c) in enumerate(names)])
        chips.arrange(m.RIGHT, buff=0.3).move_to([0, 2.2, 0])
        lit = [step_chip(i + 1, n, c, True).move_to(chips[i]) for i, (n, c) in enumerate(names)]

        h = 1.4
        x0_img = picture(tint(X0, DATA, -1, 1), h)
        p_x0 = tagged(x0_img, tex(r"x_0", 34, DATA)).move_to([-5.45, 0.75, 0])
        eps_img = picture(tint(EPS, NOISE, -2.5, 2.5), h)
        p_eps = tagged(eps_img, tex(r"\varepsilon", 36, NOISE)).move_to([-5.45, -1.75, 0])
        xt_img = picture(noisy_rgb(X_T, T_STEP), 1.6)
        p_xt = tagged(xt_img, tex(r"x_{500}", 34)).move_to([-2.65, -0.62, 0])
        line = StepLine(-3.85, -1.45, 1.05, 0)
        mix_f = formula(r"x_{500}", "=", f"{SIG:.2f}", r"\,x_0", "+", f"{NOI:.2f}", r"\,\varepsilon", size=32)
        for i, c in ((2, TIME), (3, DATA), (5, TIME), (6, NOISE)):
            mix_f[i].set_color(c)
        mix_f.move_to([-2.65, -2.3, 0])
        a_x0 = arrow(p_x0[0].get_right(), p_xt[0].get_left() + [0, 0.45, 0], DATA, buff=0.12)
        a_eps = arrow(p_eps[0].get_right(), p_xt[0].get_left() + [0, -0.45, 0], NOISE, buff=0.12)

        net = network_card(2.4, 1.5, 0.26).move_to([0.25, -0.62, 0])
        net_lab = m.VGroup(code("U-Net", 20, MUTED), code(f"{PARAMS / 1e6:.1f} M parameters", 20, MUTED))
        net_lab.arrange(m.DOWN, buff=0.08).next_to(net, m.DOWN, buff=0.16)
        a_in = arrow(p_xt[0].get_right(), net.box.get_left(), MUTED, buff=0.1)
        t_in = code(f"t = {T_STEP}", 24, TIME, weight=m.BOLD).move_to([net.get_x(), 0.95, 0])
        a_t = arrow(t_in.get_bottom(), net.box.get_top(), TIME, buff=0.08)
        hat_img = picture(tint(EPS_HAT, MODEL, -2.5, 2.5), h)
        p_hat = tagged(hat_img, tex(r"\hat\varepsilon", 36, MODEL)).move_to([2.95, 0.75, 0])
        a_out = arrow(net.box.get_right(), p_hat[0].get_left() + [0, -0.3, 0], MODEL, buff=0.12)
        p_ans = tagged(picture(tint(EPS, NOISE, -2.5, 2.5), h), tex(r"\varepsilon", 36, NOISE)).move_to([2.95, -1.75, 0])
        ans_lab = cite("the noise we added").next_to(p_ans[2], m.RIGHT, buff=0.2)
        sit(ans_lab, p_ans[2].get_bottom()[1] + 0.02)
        err_img = picture(tint(SQ_ERR, RED, 0, 0.05), h)
        p_err = tagged(err_img, tex(r"(\varepsilon-\hat\varepsilon)^2", 32, RED)).move_to([5.5, -0.3, 0])
        a_e1 = arrow(p_hat[0].get_right(), p_err[0].get_left() + [0, 0.3, 0], RED, buff=0.1)
        a_e2 = arrow(p_ans[0].get_right() + [0, 0.25, 0], p_err[0].get_left() + [0, -0.35, 0], RED, buff=0.1)
        loss = code(f"loss = {LOSS_ONE:.4f}", 26, INK, weight=m.BOLD).next_to(p_err, m.DOWN, buff=0.3)
        loss_sub = cite("mean of 784 pixels").next_to(loss, m.DOWN, buff=0.12)
        strokes = m.SurroundingRectangle(p_err[0], buff=0.08).set_stroke(RED, 3)

        back = m.VGroup(m.Line(p_err[0].get_left() + [0, 0.3, 0], p_hat[0].get_right()),
                        m.Line(p_hat[0].get_left() + [0, -0.3, 0], net.box.get_right()))
        flashes = m.Succession(*[m.ShowPassingFlash(seg.copy().set_stroke(MODEL, 8), time_width=0.8, run_time=0.5)
                                 for seg in back])
        nudge = text("nudge every weight downhill", 20, MODEL).move_to([0.35, -2.75, 0])
        repeat = code(f"repeat · {STEPS:,} steps × {BATCH} pictures", 20, TIME, weight=m.BOLD).move_to([3.3, -3.2, 0])
        x180 = tex(rf"x_{{{NEXT_T}}}", 34).move_to(p_xt[2])
        outputs = m.Group(p_hat, a_out, p_ans, ans_lab, p_err, a_e1, a_e2, loss, loss_sub, strokes)

        def pick(_, a):
            keys = [0, 0.3, 0.55, 0.8, 1.0]
            line.set_t(float(np.interp(m.smooth(a), keys, [0, 830, 140, 690, T_STEP])))

        def pick_next(_, a):
            line.set_t(T_STEP + (NEXT_T - T_STEP) * m.smooth(a))

        def renew(_, a):
            if a > 0.5 and not getattr(renew, "done", False):
                swap(x0_img, tint(X0_NEXT, DATA, -1, 1))
                swap(eps_img, tint(EPS_NEXT, NOISE, -2.5, 2.5))
                swap(xt_img, noisy_rgb(XT_NEXT, NEXT_T))
                renew.done = True

        self.say(
            "[Here is one training step], [in five moves].",
            self.new_header("01 · one step", "One training step"),
            m.LaggedStart(*[m.FadeIn(c, shift=0.1 * m.DOWN) for c in chips], lag_ratio=0.2),
        )
        self.say(
            "[First, pick a training picture]: [the first seven in the training set], [the same one as in episode four].",
            m.Transform(chips[0], lit[0]),
            m.FadeIn(p_x0, shift=0.15 * m.RIGHT),
            m.Indicate(p_x0[2], color=DATA, scale_factor=1.2),
        )
        self.say(
            "[Second, pick a step at random]: [this time, t is five hundred].",
            m.AnimationGroup(m.Transform(chips[1], lit[1]), m.FadeIn(line, shift=0.1 * m.DOWN)),
            m.UpdateFromAlphaFunc(line, pick, run_time=1.6),
        )
        self.say(
            "[Third, draw fresh noise, epsilon], [and jump straight to x t]: "
            "[zero point two eight of the picture, plus zero point nine six of noise].",
            m.AnimationGroup(m.Transform(chips[2], lit[2]), m.FadeIn(p_eps, shift=0.15 * m.RIGHT)),
            m.AnimationGroup(m.GrowArrow(a_x0), m.GrowArrow(a_eps), m.FadeIn(p_xt, shift=0.15 * m.RIGHT)),
            m.Write(mix_f),
        )
        self.say(
            "[Fourth, show x t and the step to the network], [and it guesses the noise]: [epsilon hat].",
            m.AnimationGroup(m.Transform(chips[3], lit[3]), m.GrowArrow(a_in), m.FadeIn(t_in, target_position=line.readout),
                             m.GrowArrow(a_t), m.FadeIn(net, scale=0.9), m.FadeIn(net_lab)),
            net.net.flow(MODEL, run_time=1.4),
            m.AnimationGroup(m.GrowArrow(a_out), m.FadeIn(p_hat, shift=0.15 * m.RIGHT)),
        )
        self.say(
            "[Fifth, compare]: [square the difference between epsilon and epsilon hat], "
            "[averaged over all seven hundred eighty-four pixels].",
            m.AnimationGroup(m.Transform(chips[4], lit[4]), m.FadeIn(p_ans, target_position=p_eps, path_arc=-0.25),
                             m.FadeIn(ans_lab)),
            m.AnimationGroup(m.GrowArrow(a_e1), m.GrowArrow(a_e2), m.FadeIn(p_err, shift=0.15 * m.RIGHT)),
            m.FadeIn(m.VGroup(loss, loss_sub), shift=0.1 * m.UP),
        )
        self.say(
            "[The errors sit on the strokes of the seven]: [where the background is plain, the noise is easy to work out].",
            m.Create(strokes),
            m.Circumscribe(p_x0[0], color=DATA, buff=0.06),
        )
        self.say(
            "[Then nudge every weight a little downhill], "
            "[and repeat, with a new picture, a new step and new noise].",
            m.AnimationGroup(flashes, m.FadeIn(nudge, shift=0.1 * m.DOWN)),
            m.AnimationGroup(m.FadeOut(outputs), m.FadeOut(mix_f), m.FadeOut(t_in), m.FadeOut(a_t),
                             m.UpdateFromAlphaFunc(x0_img, renew, run_time=1.2), m.Transform(p_xt[2], x180),
                             m.UpdateFromAlphaFunc(line, pick_next, run_time=1.2), m.FadeIn(repeat, shift=0.1 * m.UP)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 2 · the loop

    def the_loop(self):
        self.next_section("2 · Algorithm 1", notes=(
            "DDPM's Algorithm 1 (Training) and eq. 14, L_simple = E_{t, x₀, ε} ‖ε − ε_θ(√ᾱ_t x₀ + √(1 − ᾱ_t) ε, t)‖², "
            "t ~ Uniform{1, …, T} (Ho et al. 2020). The code is our run's loop (research/experiments/gpu/train.py) in "
            "diffusion_reference.py's names, simplified: the run also passes the label (10% dropped), uses the run's "
            "0-based t and Adam with grad clipping. DDPM's numbers (notes §5, §8; Table 2; App. B): CIFAR-10 32×32, "
            f"{DDPM_PARAMS_M}M-parameter U-Net, {DDPM_STEPS:,} steps at batch {DDPM_BATCH}, {DDPM_HOURS} h on a TPU "
            f"v3-8, IS {DDPM_IS} / FID {DDPM_FID} (FID against the training set; 5.24 against the test set); the same "
            f"ε-model trained on the true variational bound L: FID {DDPM_FID_BOUND}."))
        f = formula(r"\mathcal{L}_{\mathrm{simple}}", "=", r"\mathbb{E}_{t,\,x_0,\,\varepsilon}", r"\big\|\,\varepsilon",
                    "-", r"\hat\varepsilon_\theta(x_t,\, t)", r"\,\big\|^2", size=46)
        f[3].set_color(NOISE)
        f[5].set_color(MODEL)
        f.move_to([0, 1.85, 0])
        eq14 = cite("Ho et al., 2020, eq. 14").next_to(f, m.DOWN, buff=0.12)
        src = "\n".join([
            "for step in range(10_000):",
            "    x0 = digits[rng.integers(0, 60_000, 256)]",
            "    t = rng.integers(1, T + 1, 256)",
            "    eps = rng.standard_normal(x0.shape)",
            "    x_t = q_sample(x0, t, eps)",
            "    eps_hat = eps_model(x_t, t)",
            "    loss = np.mean((eps - eps_hat) ** 2)",
            "    adam_step(loss)",
        ])
        block = code_block(src, 21, title="Algorithm 1 · our run, simplified", accent=MODEL)
        block.move_to([-2.85, -1.05, 0])
        body = block[1]
        rows = src.split("\n")
        starts = np.cumsum([0] + [len(r.replace(" ", "")) for r in rows])
        line_of = lambda k: body[starts[k]:starts[k + 1]]  # noqa: E731

        box = card(4.9, 3.75, MODEL, CARD, radius=0.18, stroke=2).move_to([3.9, -1.05, 0])
        head = code("DDPM · Ho et al., 2020", 20, MODEL, weight=m.BOLD).move_to(box.get_top() + [0, -0.38, 0])
        facts = m.VGroup(
            code("32 × 32 photos (CIFAR-10)", 20, INK),
            code(f"U-Net, {DDPM_PARAMS_M} M parameters", 20, INK),
            code(f"{DDPM_STEPS // 1000}k steps × {DDPM_BATCH} images", 20, INK),
            code(f"{DDPM_HOURS} h on a TPU v3-8", 20, INK),
        ).arrange(m.DOWN, buff=0.16, aligned_edge=m.LEFT).next_to(head, m.DOWN, buff=0.28)
        facts.set_x(box.get_x())
        fid = title_text(f"FID {DDPM_FID}", 44, GREEN)
        isc = code(f"IS {DDPM_IS}", 22, MUTED)
        score = m.VGroup(fid, isc).arrange(m.RIGHT, buff=0.35).next_to(facts, m.DOWN, buff=0.3).set_x(box.get_x())
        sit(isc, baseline(fid))
        bound = code(f"full bound: FID {DDPM_FID_BOUND}", 20, RED).next_to(score, m.DOWN, buff=0.2)
        bound.set_x(box.get_x())
        lower = cite("lower FID = better").next_to(box, m.DOWN, buff=0.15)

        self.say(
            "[That's the whole loop.] [The DDPM paper calls it Algorithm one], [and it fits in eight lines].",
            m.AnimationGroup(self.new_header("02 · the loop", "Algorithm 1"), m.FadeIn(block[0]), m.FadeIn(block[2])),
            m.AddTextLetterByLetter(body, time_per_char=0.012),
            m.Circumscribe(m.VGroup(line_of(5), line_of(6)), color=MODEL, buff=0.08),
        )
        self.say(
            "[Its loss is equation fourteen]: [the squared error between the true noise and the guess], "
            "[averaged over pictures, steps and noise].",
            m.AnimationGroup(m.Write(f), m.FadeIn(eq14)),
            m.Circumscribe(m.VGroup(f[3], f[4], f[5], f[6]), color=MODEL, buff=0.08),
            m.Circumscribe(f[2], color=TIME, buff=0.08),
        )
        self.say(
            "[With this simple loss], [the paper's network, about thirty-six million parameters, reached an FID of three "
            "point one seven], [on tiny thirty-two by thirty-two photos].",
            m.AnimationGroup(m.FadeIn(box), m.FadeIn(head)),
            m.AnimationGroup(m.FadeIn(facts[1], shift=0.1 * m.UP), m.FadeIn(score, shift=0.1 * m.UP), m.FadeIn(lower)),
            m.AnimationGroup(m.FadeIn(facts[0], shift=0.1 * m.UP), m.FadeIn(facts[2:], shift=0.1 * m.UP)),
        )
        self.say(
            "[Trained on the full, weighted bound instead, it scored thirteen point five one]: [here, simpler was better].",
            m.FadeIn(bound, shift=0.1 * m.UP),
            m.Circumscribe(fid, color=GREEN, buff=0.1),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 3 · why the noise

    def why_noise(self):
        self.next_section("3 · Why predict the noise?", notes=(
            "x̂₀ = (x_t − √(1 − ᾱ_t)·ε̂)/√ᾱ_t (diffusion_reference.x0_from_eps), applied to the training step's real ε̂ "
            f"at t = {T_STEP}: × {NOI:.4f}, ÷ {SIG:.4f}; MSE to the true digit {np.mean((X0_HAT - X0) ** 2):.4f}. "
            f"Unit-size target: three fresh noise draws (seeds 20–22) at t = {', '.join(map(str, COL_T))}, std "
            f"{', '.join(f'{s:.3f}' for s in COL_STD)}; the inputs x_t change from a clear 7 to static. Ho et al. 2020 "
            "§3.2: predicting x₀ led 'to worse sample quality early in our experiments'; Table 2: ε-prediction with "
            "L_simple was best."))
        f = formula(r"\hat{x}_0", "=", r"\frac{x_t - \sqrt{1-\bar\alpha_t}\;\hat\varepsilon}{\sqrt{\bar\alpha_t}}", size=50)
        f[0].set_color(MODEL)
        f.move_to([0, 1.6, 0])
        f_sub = text("knowing the noise = knowing the picture", 24, INK).next_to(f, m.DOWN, buff=0.3)
        h = 1.5
        xt = tagged(picture(noisy_rgb(X_T, T_STEP), h), tex(r"x_{500}", 34))
        minus = tex(rf"-\;{NOI:.2f}\,\times", 40, TIME)
        hat = tagged(picture(tint(EPS_HAT, MODEL, -2.5, 2.5), h), tex(r"\hat\varepsilon", 36, MODEL))
        inner = m.Group(xt, minus, hat).arrange(m.RIGHT, buff=0.3)
        minus.set_y(xt[0].get_y())
        brack = m.SurroundingRectangle(inner, buff=0.18, corner_radius=0.15).set_stroke(EDGE, 2.5)
        div = tex(rf"\div\;{SIG:.2f}\;=", 44, TIME)
        guess = tagged(picture(tint(X0_HAT, MODEL, -1, 1), h), tex(r"\hat{x}_0", 36, MODEL), color=MODEL)
        true_pic = tagged(picture(tint(X0, DATA, -1, 1), 0.95), code("the real one", 20, MUTED))
        row = m.Group(m.Group(inner, brack), div, guess, true_pic).arrange(m.RIGHT, buff=0.35).move_to([0, -1.35, 0])
        div.set_y(xt[0].get_y())
        true_pic.align_to(guess[0], m.DOWN)

        cols = m.Group()
        for t, x, e, s in zip(COL_T, COL_XT, COL_EPS, COL_STD):
            top = framed(picture(noisy_rgb(x, t), 1.35))
            bot = framed(picture(tint(e, NOISE, -2.5, 2.5), 1.35))
            lab = code(f"t = {t}", 22, TIME, weight=m.BOLD)
            std = code(f"std {s:.2f}", 20, NOISE)
            col = m.Group(lab, top, bot, std).arrange(m.DOWN, buff=0.18)
            cols.add(col)
        cols.arrange(m.RIGHT, buff=1.0).move_to([0.9, -1.3, 0])
        r_in = text("input", 22, MUTED).next_to(cols[0][1], m.LEFT, buff=0.45)
        r_out = text("target", 22, NOISE).next_to(cols[0][2], m.LEFT, buff=0.45)
        r_out.align_to(r_in, m.RIGHT)

        self.say(
            "[Why guess the noise], [and not the clean picture]? [Because knowing one gives you the other].",
            m.AnimationGroup(self.new_header("03 · why noise", "Why predict the noise?"), m.Write(f)),
            m.Wait(0.1),
            m.FadeIn(f_sub, shift=0.1 * m.UP),
        )
        self.say(
            "[Put the network's guess for our seven into this formula], [and out comes its picture]: "
            "[a little blurred, but clearly a seven].",
            m.FadeIn(m.Group(inner, brack), shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(div), m.FadeIn(guess, shift=0.15 * m.RIGHT)),
            m.FadeIn(true_pic, shift=0.1 * m.LEFT),
        )
        self.say(
            "[And the noise is a tidy target]: [unit-size static at every step], "
            "[whether the picture is lightly noised or almost gone].",
            m.AnimationGroup(m.FadeOut(row, shift=0.1 * m.DOWN), m.FadeOut(f_sub)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(c, shift=0.1 * m.UP) for c in cols], lag_ratio=0.25),
                             m.FadeIn(r_in), m.FadeIn(r_out)),
            m.LaggedStart(*[m.Circumscribe(c[1], color=TIME, buff=0.06) for c in cols], lag_ratio=0.3),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 4 · other targets

    def targets(self):
        self.next_section("4 · Other targets", notes=(
            "Three targets for the same step (the 7 at t = 500): ε (DDPM), x₀, and v = √ᾱ·ε − √(1 − ᾱ)·x₀ "
            "(Salimans & Ho 2022, §4, in course notation: their α_t = √ᾱ_t, σ_t = √(1 − ᾱ_t); "
            "diffusion_reference.v_target; x₀ = √ᾱ·x_t − √(1 − ᾱ)·v asserted). The v picture is shown over −1.5…1.5. "
            "FID: Salimans & Ho Table 1, undistilled CIFAR-10 models, stochastic sampler: ε (SNR weighting = L_simple) "
            f"{SH_FID['eps']}, x (truncated SNR) {SH_FID['x0']}, v (truncated SNR) {SH_FID['v']} — 'all to work well'. "
            "v-prediction is not flow matching's velocity ε − x₀ (Episode 11; notes Appendix A)."))
        xs = [-4.65, -0.95, 2.65, 5.25]
        ys = [1.15, -0.25, -1.65]
        head = m.VGroup(text("the network predicts", 22, MUTED), text("target", 22, MUTED),
                        text("our 7 at t = 500", 22, MUTED), text("FID", 22, MUTED))
        for w, x in zip(head, xs):
            w.move_to([x, 2.2, 0])
        for w in head[1:]:
            sit(w, baseline(head[0]))
        rule = m.Line([-6.5, 1.92, 0], [6.5, 1.92, 0]).set_stroke(EDGE, 2)
        names = [("the noise", NOISE), ("the clean picture", DATA), ("v, a blend", INK)]
        forms = [tex(r"\varepsilon", 44, NOISE), tex(r"x_0", 44, DATA),
                 formula(r"\sqrt{\bar\alpha_t}\,", r"\varepsilon", "-", r"\sqrt{1-\bar\alpha_t}\,", "x_0", size=40)]
        forms[2][0].set_color(TIME)
        forms[2][1].set_color(NOISE)
        forms[2][3].set_color(TIME)
        forms[2][4].set_color(DATA)
        pics = [picture(tint(EPS, NOISE, -2.5, 2.5), 1.05), picture(tint(X0, DATA, -1, 1), 1.05),
                picture(tint(V_TGT, INK, -1.5, 1.5), 1.05)]
        fids = [SH_FID["eps"], SH_FID["x0"], SH_FID["v"]]
        rows = []
        for (name, c), fo, pic, fd, y in zip(names, forms, pics, fids, ys):
            lab = text(name, 26, c, weight=m.BOLD).move_to([xs[0], y, 0])
            fo.move_to([xs[1], y, 0])
            fr = framed(pic).move_to([xs[2], y, 0])
            num = code(f"{fd:.2f}", 26, INK, weight=m.BOLD).move_to([xs[3], y, 0])
            rows.append((lab, fo, fr, num))
        ddpm = cite("DDPM").next_to(rows[0][0], m.DOWN, buff=0.08)
        sh = cite("Salimans & Ho, 2022").next_to(rows[2][0], m.DOWN, buff=0.08)
        src = cite("FID on CIFAR-10 · Salimans & Ho, 2022, Table 1 · lower is better").move_to([2.3, -2.55, 0])
        warn = pill("v here is not flow matching's velocity ε − x₀ · Episode 11", RED, RED_FILL, 22).move_to([0, -3.2, 0])

        self.say(
            "[Two other targets are common.] [Some models predict x zero, the clean picture, directly].",
            m.AnimationGroup(self.new_header("04 · targets", "Noise, picture, or a blend"), m.FadeIn(head), m.Create(rule),
                             m.FadeIn(m.Group(*rows[0][:3], ddpm), shift=0.1 * m.UP)),
            m.FadeIn(m.Group(*rows[1][:3]), shift=0.1 * m.UP),
        )
        self.say(
            "[Others predict v, a blend of noise and picture], [introduced by Salimans and Ho in twenty twenty-two].",
            m.FadeIn(m.Group(*rows[2][:3]), shift=0.1 * m.UP),
            m.FadeIn(sh, shift=0.1 * m.UP),
        )
        self.say(
            "[In their tests, all three trained about equally well]. [And v is not flow matching's velocity], "
            "[which comes in episode eleven].",
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(r[3], shift=0.1 * m.UP) for r in rows], lag_ratio=0.2), m.FadeIn(src)),
            m.FadeIn(warn, shift=0.1 * m.UP),
            m.Circumscribe(warn, color=RED, buff=0.08),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 5 · trained on this Mac

    def on_this_mac(self):
        self.next_section("5 · Trained on this Mac", notes=(
            "research/experiments/mac_runs.md §3 and mac_runs.json (headline; digit_models.ddpm): a class-conditional "
            f"U-Net, {PARAMS:,} parameters; {STEPS:,} steps × batch {BATCH} = {STEPS * BATCH:,} images = {EPOCHS} epochs "
            f"of MNIST's 60,000; {MINUTES} min wall time with MLX 0.32.4 on an M4 Max (peak {PEAK_GB} GB); Adam, peak "
            "lr 5e-4 (warm-up 500, cosine to 5e-5), grad-norm clip 1.0, EMA 0.999, 10% label dropout, seed 0. The loss "
            f"curve is the run's own (200 points, each the mean training loss of the preceding 50 steps): {C_LOSS[0]:.4f} "
            f"at step 50, {C_LOSS[I150]:.4f} at 150, {C_LOSS[-1]:.4f} at 10,000 (headline final {FINAL}). DDPM's "
            f"U-Net: {DDPM_PARAMS_M}M (notes §8), {DDPM_PARAMS_M * 1e6 / PARAMS:.1f}× ours. The first 2,000 steps were "
            "slowed ~6% by other jobs on the Mac (mac_runs.md §8)."))
        stats = m.VGroup(
            stat(f"{PARAMS:,}", "", "parameters · our U-Net", MODEL, 50),
            stat(f"{STEPS:,} × {BATCH}", "", "steps × digits per step", INK, 50),
            stat(f"{EPOCHS}", "epochs", "passes over MNIST's 60,000 digits", INK, 50),
            stat(f"{MINUTES}", "min", f"on this Mac · MLX · peak {PEAK_GB} GB", INK, 50),
        ).arrange(m.DOWN, buff=0.38, aligned_edge=m.LEFT).move_to([-4.35, -0.6, 0])
        vs = cite(f"DDPM's: {DDPM_PARAMS_M} M").next_to(stats[0][1], m.RIGHT, buff=0.3)

        plot = Plot((0, STEPS), (0.01, 1), width=5.6, height=3.9, center=(2.75, -0.35, 0), log_y=True,
                    x_ticks=[0, 5000, 10000], x_fmt="{:,.0f}", x_label="training step", y_label="loss (log scale)")
        ticks = m.VGroup()
        for v in (1, 0.1, 0.01):
            p = plot.coords(0, v)
            ticks.add(m.Line(p, p + [-0.09, 0, 0]).set_stroke(FAINT, 2),
                      code(f"{v:g}", 20, MUTED).next_to(p, m.LEFT, buff=0.16))
        k = I150 + 1
        early = plot.line(C_STEP[:k], C_LOSS[:k], MODEL, 5)
        late = plot.line(C_STEP[I150:], C_LOSS[I150:], MODEL, 5)
        d0 = plot.dot(C_STEP[0], C_LOSS[0], MODEL, 0.09)
        l0 = code(f"{C_LOSS[0]:.2f}", 24, MODEL, weight=m.BOLD).next_to(d0, m.RIGHT, buff=0.15)
        d150 = plot.dot(C_STEP[I150], C_LOSS[I150], MODEL, 0.08)
        l150 = code(f"{C_LOSS[I150]:.3f} after {int(C_STEP[I150])} steps", 20, INK).next_to(d150, m.UR, buff=0.08)
        l150.shift(0.25 * m.RIGHT + 0.3 * m.UP)
        d_end = plot.dot(C_STEP[-1], C_LOSS[-1], MODEL, 0.09)
        l_end = code(f"{FINAL}", 24, MODEL, weight=m.BOLD).next_to(d_end, m.UP, buff=0.2)
        tag = cite(f"trained on this Mac · seed 0 · each point: mean of 50 steps").next_to(plot, m.DOWN, buff=0.15)

        recipe = m.VGroup(code("the recipe", 20, MUTED),
                          code("Adam, lr 5e-4", 20, INK),
                          code("weights averaged (EMA 0.999)", 20, INK),
                          code("1 label in 10 hidden (Ep 14)", 20, PROMPT)).arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT)
        r_box = card(recipe.width + 0.5, recipe.height + 0.4, EDGE, CARD, radius=0.14, stroke=2)
        recipe.move_to(r_box)
        r_card = m.VGroup(r_box, recipe).move_to(plot.coords(6600, 0.32))

        self.say(
            "[Now our own run.] [The digit model is a U-Net with three point nine million parameters], "
            "[about nine times smaller than DDPM's].",
            self.new_header("05 · this Mac", "The digit model's loss"),
            m.FadeIn(stats[0], shift=0.15 * m.RIGHT),
            m.FadeIn(vs),
        )
        self.say(
            "[It trained for ten thousand steps of two hundred fifty-six digits]: [about forty-three passes over MNIST], "
            "[in thirty-two minutes on this Mac, with MLX].",
            m.FadeIn(stats[1], shift=0.15 * m.RIGHT),
            m.FadeIn(stats[2], shift=0.15 * m.RIGHT),
            m.FadeIn(stats[3], shift=0.15 * m.RIGHT),
        )
        self.say(
            "[The loss starts at zero point eight one], [and within a hundred fifty steps it's below zero point zero four].",
            m.AnimationGroup(m.FadeIn(plot), m.FadeIn(ticks), m.FadeIn(tag), m.FadeIn(d0, scale=0.5), m.FadeIn(l0)),
            m.Succession(m.Create(early, run_time=1.0), m.AnimationGroup(m.FadeIn(d150, scale=0.5), m.FadeIn(l150))),
        )
        self.say(
            "[Then it creeps down slowly], [to zero point zero two zero after ten thousand steps].",
            m.Create(late, run_time=2.4, rate_func=m.linear),
            m.AnimationGroup(m.FadeIn(d_end, scale=0.5), m.FadeIn(l_end, shift=0.1 * m.UP)),
        )
        self.say(
            "[It used a few standard extras]: [the Adam optimizer, an average of recent weights], "
            "[and one label in ten hidden, for episode fourteen].",
            m.FadeIn(r_box),
            m.FadeIn(recipe[:3], shift=0.1 * m.UP),
            m.FadeIn(recipe[3], shift=0.1 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 6 · the floor

    def the_floor(self):
        self.next_section("6 · The loss floor", notes=(
            f"The swirl (diffusion_reference.SWIRL, 256 round Gaussians). A noisy point x_t = ({POST_X[0, 0]}, {POST_X[0, 1]}) at t = "
            f"{POST_T}: the exact posterior p(component | x_t) (mixture_posterior) gives {POST_ARM0:.0%} to arm 1 and "
            f"{1 - POST_ARM0:.0%} to arm 2; brightness of each centre and of its line to x_t ∝ √(p/p_max) (x_t is drawn in "
            "the same plane as the clean swirl, a simplification: it lives in the √ᾱ-scaled picture). The perfect "
            "denoiser E[x₀ | x_t] = "
            f"({POST_GUESS[0]:.2f}, {POST_GUESS[1]:.2f}) lies {nearest_on_swirl(POST_GUESS)[0][0]:.2f} off the arms. "
            f"Loss floor (best possible ε-loss, t ~ U{{1..1000}}): toy.json {FLOOR} (exact denoiser, 200,000 points; "
            f"recomputed here with 100,000: {FLOOR_CHECK:.4f}); the tiny MLP ({MLP_PARAMS:,} parameters; inputs x, sine "
            "features of x and t; 40,000 steps × 512) finished at "
            f"{MLP_LOSS}. Curve: toy.json training.eps.loss_curve, drawn as a rolling mean of 10 logged points (single "
            "logged values scatter ±0.002). Without the sine features of x a plain MLP blurred the thin arms "
            "(toy.md §2 caveats; Tancik et al. 2020)."))
        c, s = np.array([-3.7, -0.3, 0.0]), 1.85
        means = SWIRL.means
        w = np.sqrt(POST_RESP / POST_RESP.max())
        faint = PlaneView(means, c, s, DATA, 0.035, 0.28)
        keep = [k for k in range(len(means)) if w[k] > 0.4]
        p_x = c + s * np.array([*POST_X[0], 0])
        bright = m.VGroup(*[m.Dot(c + s * np.array([*means[k], 0]), radius=0.045, color=DATA).set_opacity(float(w[k]))
                            for k in keep])
        links = m.VGroup(*[m.Line(p_x, c + s * np.array([*means[k], 0])).set_stroke(DATA, 1.5, opacity=0.55 * float(w[k]))
                           for k in keep])
        pt = m.Dot(p_x, radius=0.1, color=NOISE).set_z_index(3)
        pt_lab = tex(r"x_t", 36, NOISE).next_to(pt, m.RIGHT, buff=0.14)
        avg = m.Dot(c + s * np.array([*POST_GUESS, 0]), radius=0.1, color=MODEL).set_z_index(3)
        avg_lab = text("best possible guess:\nthe average", 22, MODEL).move_to([-5.35, -2.85, 0])
        avg_arrow = arrow(avg_lab.get_top() + [0.35, 0.02, 0], avg.get_center() + [-0.08, -0.08, 0], MODEL, buff=0.1)
        could = text("could have come from\neither arm", 22, DATA).move_to(c + [-0.5, 2.35, 0])
        plot = Plot((0, 40_000), (0.15, 0.2), width=5.0, height=3.4, center=(3.45, -0.25, 0), x_ticks=[0, 20_000, 40_000],
                    y_ticks=[0.15, 0.2], x_fmt="{:,.0f}", y_fmt="{:.2f}", x_label="training step", y_label="ε-loss on the swirl")
        floor = plot.hline(FLOOR, GREEN, 3)
        floor_lab = code(f"floor {FLOOR} (exact)", 22, GREEN, weight=m.BOLD).move_to(plot.coords(26_000, FLOOR)).shift(0.3 * m.DOWN)
        curve = plot.line(SW_SMOOTH_STEPS, SW_SMOOTH, MODEL, 4)
        d_end = plot.dot(SW_SMOOTH_STEPS[-1], SW_SMOOTH[-1], MODEL, 0.08)
        mlp_lab = code(f"tiny network {MLP_LOSS}", 22, MODEL, weight=m.BOLD).move_to(plot.coords(26_000, 0.17))
        info = cite(f"{MLP_PARAMS:,} parameters · inputs: x, t and sine waves of x").next_to(plot, m.DOWN, buff=0.18)

        self.say(
            "[Why doesn't the loss reach zero?] [Because a noisy point could have come from many places].",
            m.AnimationGroup(self.new_header("06 · the floor", "Why the loss never hits zero"),
                             m.LaggedStart(*[m.FadeIn(d, scale=0.5) for d in faint], lag_ratio=0.003, run_time=1.0),
                             m.FadeIn(pt, scale=0.5), m.FadeIn(pt_lab)),
            m.AnimationGroup(m.LaggedStart(*[m.Create(l) for l in links], lag_ratio=0.02),
                             m.LaggedStart(*[m.FadeIn(d, scale=0.4) for d in bright], lag_ratio=0.02), m.FadeIn(could)),
        )
        self.say(
            "[So even a perfect model can only predict the average]: [the loss has a floor].",
            m.AnimationGroup(m.FadeIn(avg, scale=0.5), m.GrowArrow(avg_arrow), m.FadeIn(avg_lab)),
            m.AnimationGroup(m.FadeIn(plot), m.Create(floor), m.FadeIn(floor_lab)),
        )
        self.say(
            "[On the swirl, that floor can be computed exactly]: [zero point one five eight one]. "
            "[Our tiny network reaches zero point one five eight eight].",
            m.Circumscribe(floor_lab, color=GREEN, buff=0.08),
            m.Indicate(floor, color=GREEN, scale_factor=1.0),
            m.Succession(m.Create(curve, run_time=1.8), m.AnimationGroup(m.FadeIn(d_end, scale=0.5), m.FadeIn(mlp_lab))),
        )
        self.say(
            "[It has forty-six thousand parameters], [and needed sine waves of its input] [to keep the thin arms sharp].",
            m.FadeIn(info, shift=0.1 * m.UP),
            m.Circumscribe(info, color=INK, buff=0.08),
            m.Circumscribe(bright, color=DATA, buff=0.1),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 7 · the model's guess

    def the_guess(self):
        self.next_section("7 · The model's clean guess", notes=(
            f"One 7 generated on this Mac by the digit model (DDPM ancestral sampling, 1,000 steps, w = 1, starting noise "
            f"np.random.default_rng({TRAJ_SEED})): arrays/digit_trajectory_7_more_seeds.npz. Top: x_t, the input at step t "
            "(0-based); bottom: the model's clean guess x̂₀ = (x_t − √(1 − ᾱ)·ε̂)/√ᾱ at that step (AMBER), both shown "
            f"clipped to −1…1. Blur (share of grey pixels, |value| < 0.6) of x̂₀ at t = {', '.join(map(str, STRIP_T))}: "
            f"{', '.join(f'{v:.3f}' for v in _grey)}. At t = 999, √ᾱ = {np.sqrt(AB_RUN[999]):.4f}: the guess is "
            f"speckles (std over the final digit's background {_speck[0]:.2f}, vs {_speck[1]:.2f} at 900 and "
            f"{_speck[3]:.2f} at 500; Section 8)."))
        h = 1.45
        cols = m.Group()
        for t, i in zip(STRIP_T, STRIP_I):
            lab = code(f"t = {t}", 22, TIME, weight=m.BOLD)
            top = framed(picture(noisy_rgb(np.clip(TRAJ_XT[i], -1, 1), t), h))
            bot = framed(picture(tint(TRAJ_X0[i], MODEL, -1, 1), h), MODEL)
            cols.add(m.Group(lab, top, bot).arrange(m.DOWN, buff=0.2))
        cols.arrange(m.RIGHT, buff=0.32).move_to([0.45, -0.35, 0])
        r_in = tex(r"x_t", 40).next_to(cols[0][1], m.LEFT, buff=0.35)
        r_out = tex(r"\hat{x}_0", 40, MODEL).next_to(cols[0][2], m.LEFT, buff=0.35)
        cap = cite(f"generated on this Mac · digit model, DDPM 1,000 steps · seed {TRAJ_SEED} · label 7")
        cap.next_to(cols, m.DOWN, buff=0.28)
        blur = m.Brace(m.Group(cols[1][2], cols[2][2]), direction=m.DOWN, color=MODEL, buff=0.08)
        sharp = m.Brace(m.Group(cols[4][2], cols[5][2]), direction=m.DOWN, color=MODEL, buff=0.08)
        blur_lab = text("blurry: an average", 22, MODEL).next_to(blur, m.DOWN, buff=0.1)
        sharp_lab = text("sharp", 22, MODEL).next_to(sharp, m.DOWN, buff=0.1)
        sit(sharp_lab, baseline(blur_lab))
        bad = m.SurroundingRectangle(cols[0][2], buff=0.1).set_stroke(RED, 4)
        bad_lab = text("speckles", 22, RED, weight=m.BOLD).next_to(bad, m.DOWN, buff=0.12)

        self.say(
            "[Turn each noise guess into a picture guess], [and you can watch what the model believes].",
            m.AnimationGroup(self.new_header("07 · the guess", "What the model believes"),
                             m.FadeIn(r_in), m.FadeIn(r_out), m.FadeIn(cap),
                             m.LaggedStart(*[m.FadeIn(c[:2], shift=0.1 * m.DOWN) for c in cols], lag_ratio=0.15)),
            m.LaggedStart(*[m.FadeIn(c[2], target_position=c[1]) for c in cols], lag_ratio=0.2),
        )
        self.say(
            "[At high noise, it guesses a blurry seven]: [an average of every seven that could be hiding there]. "
            "[As the noise falls, the guess sharpens].",
            m.AnimationGroup(m.FadeOut(cap), m.GrowFromCenter(blur), m.FadeIn(blur_lab)),
            m.Circumscribe(m.Group(cols[1][2], cols[2][2]), color=MODEL, buff=0.08),
            m.AnimationGroup(m.GrowFromCenter(sharp), m.FadeIn(sharp_lab)),
        )
        self.say(
            "[But at step one thousand, the guess is just speckles].",
            m.AnimationGroup(m.Create(bad), m.FadeIn(bad_lab, shift=0.1 * m.UP)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 8 · one jump

    def one_jump(self):
        self.next_section("8 · Why not one jump?", notes=(
            f"At t = 1000, √ᾱ = {SQ_AB_END:.5f} (ALPHA_BAR[1000] = {ALPHA_BAR[T]:.3e}), so x̂₀ = (x − √(1 − ᾱ)·ε̂)/√ᾱ "
            f"multiplies any error in ε̂ by {GAIN:.1f}. The swirl: the trained ε-MLP vs the exact ε at 2,000 points of "
            f"pure noise (gaussian_noise(2000, seed=5), toy.md's): mean error |ε̂ − ε_exact| = {EPS_ERR:.4f}. One DDIM step "
            f"(ddim_sample(model, z, steps=1)) lands {JUMP_DIST:.3f} from the swirl on average (toy.json "
            f"flow.trained.quality_vs_steps['1'].ddim.dist = 2.305; with the exact ε: 0.149, at the middle). The view "
            f"zooms out to ±{VIEW_R:g} to show where they land: the first {N_SHOW} points are drawn; {N_BEYOND} land beyond "
            f"the view and are hidden. The trained flow model (Episode 11), one Euler step from the same noise: "
            f"{FLOW_DIST:.4f} (toy.json 0.1346), every point at the data's average. Salimans & Ho 2022 §4: ε-prediction "
            "gives zero weight at SNR = 0; one reason for x₀/v targets."))
        f = formula(r"\hat{x}_0", "=",
                    r"\frac{x_{1000} - \sqrt{1-\bar\alpha_{1000}}\;\hat\varepsilon}{\sqrt{\bar\alpha_{1000}}}", size=44)
        f[0].set_color(MODEL)
        f.move_to([-3.45, 1.65, 0])
        small = formula(r"\sqrt{\bar\alpha_{1000}}", "=", f"{SQ_AB_END:.4f}", size=40)
        small[0].set_color(TIME)
        small[2].set_color(TIME)
        small.move_to([-3.45, 0.45, 0])
        g1 = code(f"÷ {SQ_AB_END:.4f}", 30, TIME, weight=m.BOLD)
        g2 = code(f"× {GAIN:.0f}", 40, RED, weight=m.BOLD)
        m.VGroup(g1, g2).arrange(m.RIGHT, buff=1.1).move_to([-3.45, -0.35, 0])
        sit(g1, baseline(g2))
        g_arrow = arrow(g1.get_right() + [0.1, 0.12, 0], g2.get_left() + [-0.1, 0.12, 0], MUTED, buff=0.05)
        gain = m.VGroup(g1, g_arrow, g2)
        err_line = code(f"noise guess off by {EPS_ERR:.2f}", 22, INK).move_to([-3.45, -1.05, 0])
        land = m.VGroup(title_text(f"{JUMP_DIST:.1f}", 54, RED), text("away from the swirl,\non average", 22, INK))
        land.arrange(m.RIGHT, buff=0.3).move_to([-3.45, -1.85, 0])

        c = np.array([3.4, -0.45, 0.0])
        s0, s1 = 0.75, 2.6 / VIEW_R
        sw = swirl(400, seed=3)
        data = PlaneView(sw, c, s0, DATA, 0.03, 0.9).set_z_index(2)
        ring = m.Circle(radius=s0).set_stroke(FAINT, 2.5).move_to(c).set_z_index(2)
        z_show = JUMP_Z[:N_SHOW][JUMP_SHOW]
        e_show = JUMP_EPS[:N_SHOW][JUMP_SHOW]
        f_show = JUMP_FLOW[:N_SHOW][JUMP_SHOW]
        pts = PlaneView(z_show, c, s0, NOISE, 0.03, 0.85)
        top = text(f"one jump from pure noise · {N_SHOW} of 2,000 points", 22, MUTED).move_to([c[0], 2.4, 0])
        hidden = cite(f"{N_BEYOND} of {N_SHOW} land beyond this view").move_to([c[0], -3.35, 0])
        flow_pts = PlaneView(z_show, c, s1, FLOW, 0.035, 0.95).set_z_index(3)
        flow_txt = text(f"Episode 11's flow model: one step\nlands on the average ({FLOW_DIST:.2f} away)", 22, FLOW)
        flow_box = card(flow_txt.width + 0.5, flow_txt.height + 0.35, FLOW, LIME_FILL, radius=0.14, stroke=2)
        flow_lab = m.VGroup(flow_box, flow_txt.move_to(flow_box)).move_to([-3.45, -2.95, 0])

        def jump(_, a):
            sc = s0 + (s1 - s0) * m.smooth(min(1.0, a * 1.3))
            pts.put(z_show + (e_show - z_show) * m.smooth(a), sc)
            shade = mix(NOISE, MODEL, m.smooth(a))
            for d in pts:
                d.set_color(shade)
            data.put(sw, sc)
            ring.become(m.Circle(radius=sc).set_stroke(FAINT, 2.5).move_to(c).set_z_index(2))

        def to_flow(_, a):
            flow_pts.put(z_show + (f_show - z_show) * m.smooth(a), s1)

        self.say(
            "[So why not jump from pure noise to a picture in one step]? [Look at the formula at step one thousand].",
            m.AnimationGroup(self.new_header("08 · one jump", "Why not one big jump?"),
                             m.FadeIn(m.VGroup(data, ring)), m.FadeIn(pts), m.FadeIn(top)),
            m.Write(f),
        )
        self.say(
            "Here, root alpha bar is [zero point zero zero six four], [so dividing by it multiplies any error in the "
            "noise guess] [by a hundred fifty-seven].",
            m.FadeIn(small, shift=0.1 * m.UP),
            m.FadeIn(gain[:2], shift=0.1 * m.UP),
            m.FadeIn(gain[2], scale=0.7),
        )
        self.say(
            "[On the swirl, the noise guess is off by only zero point zero two], "
            "[yet one jump lands the points two point three away].",
            m.FadeIn(err_line, shift=0.1 * m.UP),
            m.AnimationGroup(m.UpdateFromAlphaFunc(pts, jump, run_time=2.2),
                             m.Succession(m.Wait(1.6), m.AnimationGroup(m.FadeIn(land, shift=0.1 * m.UP), m.FadeIn(hidden)))),
        )
        self.say(
            "[That's one reason for the x zero and v targets], [and for the flow model of episode eleven], "
            "[whose one step lands on the average instead].",
            m.Circumscribe(f, color=MODEL, buff=0.1),
            m.AnimationGroup(m.FadeIn(flow_lab, shift=0.1 * m.UP), m.FadeIn(flow_pts)),
            m.Succession(m.UpdateFromAlphaFunc(flow_pts, to_flow, run_time=1.6),
                         m.Create(m.Circle(radius=0.2).set_stroke(FLOW, 3).move_to(c).set_z_index(4), run_time=0.5)),
        )
        self.wait(0.5)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes=(
            "Noise (a random step t and fresh ε, x_t in one jump), predict (the network's ε̂), compare (‖ε − ε̂‖², then "
            "one gradient step). L_simple, Ho et al. 2020 eq. 14. Next: Episode 6, the score — predicting the noise is "
            "estimating the score, ∇log p = −ε/√(1 − ᾱ)."))
        halo = glow(MODEL, radius=4.0)
        words = m.VGroup(*[title_text(w, 64, c) for w, c in (("Noise", NOISE), ("Predict", MODEL), ("Compare", INK))])
        subs = m.VGroup(*[code(s, 20, MUTED) for s in ("random t, fresh ε", "the network's ε̂", "squared error, step")])
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
        row = m.VGroup(words, dots, subs).move_to(0.95 * m.UP)
        f = formula(r"\mathcal{L}_{\mathrm{simple}}", "=", r"\mathbb{E}\,\big\|\,\varepsilon", "-",
                    r"\hat\varepsilon_\theta(x_t,\, t)", r"\,\big\|^2", size=44)
        f[4].set_color(MODEL)
        f.next_to(row, m.DOWN, buff=0.6)
        tag = text("Training a diffusion model is regression on the noise.", 28, MUTED)
        mascot = pixel(0.42)
        nxt = pill("next · Episode 6 · The Score", SCORE, LIME_FILL, 20, mono=True)
        foot = m.VGroup(tag, m.VGroup(mascot, nxt).arrange(m.RIGHT, buff=0.3)).arrange(m.DOWN, buff=0.35)
        foot.next_to(f, m.DOWN, buff=0.55)

        self.say(
            "[To sum up]: [add noise], [predict it], [compare].",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.3 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(dots[0]), m.FadeIn(words[1], shift=0.3 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(dots[1]), m.FadeIn(words[2], shift=0.3 * m.UP), m.FadeIn(subs[2]), m.FadeIn(f)),
        )
        self.say(
            "[Training is just regression on the noise]. [And that noise guess hides a direction]: "
            "[which way the more likely pictures lie].",
            m.Write(tag),
            m.FadeIn(mascot, shift=0.2 * m.UP),
            m.FadeIn(nxt, shift=0.2 * m.UP),
        )
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, f, foot)], run_time=0.9)
