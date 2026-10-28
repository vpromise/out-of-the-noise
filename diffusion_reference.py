"""The canonical code and numbers for "Out of the Noise", a course on diffusion models. Episodes that show a
formula, a sampler or a number about the swirl compute it here, so names and values match across the course.
Excerpts on screen are short (≤ 10 lines) and may simplify, but never rename. NumPy only; nothing heavy runs
at import. `python diffusion_reference.py` runs the self-checks (~3 s) and prints "all checks passed".

CONVENTIONS (read these once)
  • DDPM time: integer steps t = 0 … T (T = 1000). Schedule arrays have length T + 1 and are indexed by t:
    betas[0] = 0 and alpha_bar[0] = 1 is the clean data; alpha_bar[T] ≈ 0 is (almost) pure noise.
        x_t = √ᾱ_t · x0 + √(1 − ᾱ_t) · ε        ("variance preserving": x_t stays about unit scale)
    Models may be called at a fractional t; ᾱ is then interpolated between neighbouring steps.
  • σ view (EDM, "variance exploding"): x_t / √ᾱ_t = x0 + σ·ε with σ = √((1 − ᾱ)/ᾱ) = 1/√SNR. The ODE
    samplers integrate in σ but return their paths in the x_t picture, where everything stays on screen.
  • Flow-matching time: t ∈ [0, 1], t = 0 data, t = 1 noise, x_t = (1 − t)·x0 + t·ε, velocity v = ε − x0
    (the rectified-flow / Stable Diffusion 3 / FLUX convention). Flow samplers run t from 1 down to 0.
  • Models are plain functions of (x, t) with x of shape (n, 2) and t a number (or one per row):
        eps_model(x, t) → ε̂ (t a DDPM step)    v_model(x, t) → v̂ (t flow time)    score_fn(x, sigma) (σ view)
    exact_eps / exact_velocity / mixture_score_sigma are such functions for the swirl, with no training.
  • Every sampler takes starting points x (usually gaussian_noise(n, seed)) and returns a Trajectory:
    xs (steps + 1, n, 2) with xs[0] the noise and xs[-1] the samples; ts the time of each state; x0s the
    model's guess of the clean point at each step. All randomness is np.random.default_rng(seed).

THE SWIRL: two spiral arms, each 1¼ turns from radius 0.15 out to 1.0, so the swirl fits in the unit disc
the way pixel values are scaled to [−1, 1] (std ≈ 0.5 per coordinate, like images). It IS a mixture of 256
small round Gaussians (std 0.03) evenly spaced along the arms, so its noised density, score, perfect
denoiser and flow velocity are exact closed forms at every noise level, and the arms are its two labels.
A second labelled toy, blobs_mixture() (three round blobs), shows guidance pushing samples beyond the data.
Measured results (training, samplers, guidance …) are in research/experiments/*.json: use measured(...).
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import numpy as np

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE / "research" / "experiments"
WEIGHTS = EXPERIMENTS / "weights"

# ============================================================================ noise schedules (Ep 4, 10)


def linear_betas(T=1000, beta_start=1e-4, beta_end=0.02):
    """DDPM's schedule (Ho et al. 2020): β grows linearly from 1e-4 to 0.02. betas[0] = 0 (t = 0 is clean)."""
    return np.concatenate([[0.0], np.linspace(beta_start, beta_end, T)])


def alphas_cumprod(betas):
    """ᾱ_t = (1 − β_1)(1 − β_2)…(1 − β_t): the share of the signal's variance that survives to step t."""
    return np.cumprod(1.0 - np.asarray(betas, dtype=float))


def betas_from_alpha_bar(alpha_bar, max_beta=0.999):
    """Per-step β_t = 1 − ᾱ_t/ᾱ_{t−1} from a ᾱ curve, clipped at max_beta (Nichol & Dhariwal's guard)."""
    ab = np.asarray(alpha_bar, dtype=float)
    return np.clip(np.concatenate([[0.0], 1 - ab[1:] / ab[:-1]]), 0.0, max_beta)


def cosine_alpha_bar(T=1000, s=0.008, max_beta=0.999):
    """Nichol & Dhariwal (2021): ᾱ_t = f(t)/f(0) with f(t) = cos²(π/2 · (t/T + s)/(1 + s)); it removes the
    signal more evenly than the linear schedule. β is clipped at 0.999 as in their code (changes only t = T)."""
    f = np.cos((np.arange(T + 1) / T + s) / (1 + s) * np.pi / 2) ** 2
    return alphas_cumprod(betas_from_alpha_bar(f / f[0], max_beta))


ALPHA_BAR = alphas_cumprod(linear_betas())      # the default schedule everywhere: DDPM's linear, T = 1000
ALPHA_BAR_COSINE = cosine_alpha_bar()
T = len(ALPHA_BAR) - 1


def snr(alpha_bar):
    """Signal-to-noise ratio ᾱ/(1 − ᾱ): ∞ at t = 0, 1 where signal and noise are equal, → 0 at t = T."""
    ab = np.asarray(alpha_bar, dtype=float)
    with np.errstate(divide="ignore"):
        return ab / (1 - ab)


def log_snr(alpha_bar):
    """λ = log(ᾱ/(1 − ᾱ)) (Kingma et al. 2021): falls from +∞ at t = 0 to about −10 at t = T (linear)."""
    ab = np.asarray(alpha_bar, dtype=float)
    with np.errstate(divide="ignore"):
        return np.log(ab) - np.log1p(-ab)


def sigma_from_alpha_bar(alpha_bar):
    """σ = √((1 − ᾱ)/ᾱ) = 1/√SNR: the noise level when x_t is rescaled to x0 + σ·ε."""
    ab = np.asarray(alpha_bar, dtype=float)
    return np.sqrt((1 - ab) / ab)


def alpha_bar_from_sigma(sigma):
    """ᾱ = 1/(1 + σ²)."""
    return 1.0 / (1.0 + np.asarray(sigma, dtype=float) ** 2)


def alpha_bar_at(t, alpha_bar=ALPHA_BAR):
    """ᾱ at step t (a number or an array; a fractional step interpolates between its neighbours)."""
    return np.interp(np.asarray(t, dtype=float), np.arange(len(alpha_bar)), alpha_bar)


def timestep_from_sigma(sigma, alpha_bar=ALPHA_BAR):
    """The (fractional) DDPM step whose noise level is σ, so a DDPM model can run on any σ grid.
    σ = 0 → t = 0; a σ beyond the schedule's last step is clipped to t = T."""
    return np.interp(alpha_bar_from_sigma(sigma), alpha_bar[::-1], np.arange(len(alpha_bar))[::-1])


def karras_sigmas(n, sigma_min=0.002, sigma_max=80.0, rho=7.0):
    """Karras et al. (2022, "EDM") step placement: n noise levels from σ_max down to σ_min, evenly spaced in
    σ^(1/ρ) so steps crowd in at low noise where the detail is decided. A final 0 is appended (n + 1 values)."""
    hi, lo = sigma_max ** (1 / rho), sigma_min ** (1 / rho)
    return np.append((hi + np.linspace(0, 1, n) * (lo - hi)) ** rho, 0.0)


def _beta_rate(t, alpha_bar):
    """β(t) = −d log ᾱ/dt, the continuous-time rate of the schedule (≈ β_t for small β)."""
    return np.interp(t, np.arange(1, len(alpha_bar)), -np.diff(np.log(alpha_bar)))


# ============================================================================ the forward process (Ep 4, 5, 11)


def gaussian_noise(n, seed=0, d=2):
    """n points of standard Gaussian noise ε ~ N(0, I): where every sampler starts."""
    return np.random.default_rng(seed).standard_normal((n, d))


def _col(v, x):
    """A number stays a number; one value per row of x becomes a column that broadcasts against x."""
    v = np.asarray(v, dtype=float)
    return v.reshape(v.shape + (1,) * (np.ndim(x) - v.ndim)) if v.ndim else v


def q_sample(x0, t, eps, alpha_bar=ALPHA_BAR):
    """Jump straight to step t, no loop needed: x_t = √ᾱ_t · x0 + √(1 − ᾱ_t) · ε (t: one step or one per row)."""
    ab = _col(alpha_bar_at(t, alpha_bar), x0)
    return np.sqrt(ab) * x0 + np.sqrt(1 - ab) * eps


def x0_from_eps(x_t, eps, ab):
    """Undo the jump if you know the noise: x0 = (x_t − √(1 − ᾱ)·ε)/√ᾱ. (ab = ᾱ_t, a number or column.)"""
    return (x_t - np.sqrt(1 - ab) * eps) / np.sqrt(ab)


def eps_from_x0(x_t, x0, ab):
    """The noise that must have been added: ε = (x_t − √ᾱ·x0)/√(1 − ᾱ)."""
    return (x_t - np.sqrt(ab) * x0) / np.sqrt(1 - ab)


def score_from_eps(eps, ab):
    """The score from a noise prediction: ∇ log p_t(x_t) = −ε̂/√(1 − ᾱ_t) (Vincent 2011; Song et al. 2021)."""
    return -eps / np.sqrt(1 - ab)


def v_target(x0, eps, ab):
    """v-prediction's target (Salimans & Ho 2022): v = √ᾱ·ε − √(1 − ᾱ)·x0. Not flow matching's velocity."""
    return np.sqrt(ab) * eps - np.sqrt(1 - ab) * x0


def flow_interpolant(x0, t, eps):
    """Flow matching's straight path from data (t = 0) to noise (t = 1): x_t = (1 − t)·x0 + t·ε."""
    t = _col(t, x0)
    return (1 - t) * x0 + t * eps


def flow_velocity(x0, eps):
    """d x_t/dt along that path: v = ε − x0, the same at every t (that is what makes the path straight)."""
    return eps - x0


# ============================================================================ the swirl and its exact scores (Ep 2–7)

SWIRL_TURNS, SWIRL_R0, SWIRL_R1, SWIRL_NOISE = 1.25, 0.15, 1.0, 0.03


def swirl_curve(n=200):
    """The centre line of the two arms, shape (2, n, 2), evenly spaced by arc length from the centre out.
    Arm 0 starts at (0.15, 0) and turns anticlockwise; arm 1 is arm 0 rotated by half a turn."""
    theta = np.linspace(0, 2 * np.pi * SWIRL_TURNS, 4001)
    r = SWIRL_R0 + (SWIRL_R1 - SWIRL_R0) * theta / theta[-1]
    dense = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=-1)
    length = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))])
    s = np.linspace(0, length[-1], n)
    arm = np.stack([np.interp(s, length, dense[:, 0]), np.interp(s, length, dense[:, 1])], axis=-1)
    return np.stack([arm, -arm])


class Mixture(NamedTuple):
    """A mixture of round Gaussians: component k has centre means[k], std stds[k], weight weights[k], class labels[k]."""
    means: np.ndarray
    stds: np.ndarray
    weights: np.ndarray
    labels: np.ndarray

    def sample(self, n, seed=0):
        """n points and their class labels: pick a component by weight, then add its round noise."""
        rng = np.random.default_rng(seed)
        k = rng.choice(len(self.weights), size=n, p=self.weights)
        return self.means[k] + self.stds[k, None] * rng.standard_normal((n, self.means.shape[1])), self.labels[k]

    def given(self, label):
        """The same mixture restricted to one class, weights renormalised: p(x0 | label)."""
        keep = self.labels == label
        return Mixture(self.means[keep], self.stds[keep], self.weights[keep] / self.weights[keep].sum(),
                       self.labels[keep])


@lru_cache(maxsize=8)
def swirl_mixture(noise=SWIRL_NOISE, per_arm=128):
    """The swirl as a mixture: per_arm round Gaussians of std `noise` evenly spaced along each arm (spacing
    0.036, close enough that the density along an arm is flat to 1e-11). Labels: 0 and 1 for the two arms."""
    means = swirl_curve(per_arm).reshape(-1, 2)
    k = len(means)
    mix = Mixture(means, np.full(k, float(noise)), np.full(k, 1.0 / k), np.repeat([0, 1], per_arm))
    for array in mix:
        array.setflags(write=False)
    return mix


SWIRL = swirl_mixture()


def swirl(n, seed=0, noise=SWIRL_NOISE):
    """n points of the swirl, the course's 2D dataset (exactly samples of the mixture SWIRL)."""
    return swirl_mixture(noise).sample(n, seed)[0]


def swirl_labelled(n, seed=0, noise=SWIRL_NOISE):
    """The same points as swirl(n, seed) plus the arm (0 or 1) each came from: the conditional toy."""
    return swirl_mixture(noise).sample(n, seed)


class Posterior(NamedTuple):
    resp: np.ndarray         # (n, K) p(component k | x)
    score: np.ndarray        # (n, d) ∇ log p(x)
    x0: np.ndarray           # (n, d) E[x0 | x]
    log_density: np.ndarray  # (n,)   log p(x)


def mixture_posterior(x, a, b, mix=SWIRL):
    """Everything exact about x = a·x0 + b·ε with x0 ~ mix, ε ~ N(0, I). Component k becomes a round Gaussian
    with centre a·μ_k and variance a²s_k² + b², so the noisy data is still a mixture: Bayes gives the
    responsibilities, the score is their weighted pull toward each centre, and E[x0 | x] is Gaussian conditioning.
    a, b: numbers, or one per row of x (DDPM: a = √ᾱ, b = √(1 − ᾱ); σ view: a = 1, b = σ; flow: 1 − t, t)."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    a, b = _col(a, x), _col(b, x)
    mu, s2 = mix.means, mix.stds ** 2
    var = a ** 2 * s2 + b ** 2                                                      # (K,) or (n, K)
    sq = (x ** 2).sum(1, keepdims=True) - 2 * a * (x @ mu.T) + a ** 2 * (mu ** 2).sum(1)  # ‖x − a·μ_k‖²
    logits = np.log(mix.weights) - 0.5 * x.shape[1] * np.log(2 * np.pi * var) - sq / (2 * var)
    top = logits.max(1, keepdims=True)
    p = np.exp(logits - top)
    resp = p / p.sum(1, keepdims=True)
    pull = resp / var
    score = a * (pull @ mu) - x * pull.sum(1, keepdims=True)                         # Σ_k r_k (a·μ_k − x)/var_k
    g = resp * s2 / var
    x0 = resp @ mu + a * (g.sum(1, keepdims=True) * x - a * (g @ mu))               # Σ_k r_k (μ_k + a s²/var (x − aμ_k))
    return Posterior(resp, score, x0, (top + np.log(p.sum(1, keepdims=True)))[:, 0])


def _vp(t, alpha_bar):
    ab = alpha_bar_at(t, alpha_bar)
    return np.sqrt(ab), np.sqrt(1 - ab)


def mixture_density(x, t, mix=SWIRL, alpha_bar=ALPHA_BAR):
    """p_t(x), the exact density of the noised data at step t (for heat maps)."""
    return np.exp(mixture_posterior(x, *_vp(t, alpha_bar), mix).log_density)


def mixture_score(x, t, mix=SWIRL, alpha_bar=ALPHA_BAR):
    """The exact score ∇ log p_t(x) at step t: the arrow toward more likely noisy data. No training involved."""
    return mixture_posterior(x, *_vp(t, alpha_bar), mix).score


def exact_denoiser(x, t, mix=SWIRL, alpha_bar=ALPHA_BAR):
    """The perfect denoiser E[x0 | x_t]: the average of every clean point that could have produced x_t."""
    return mixture_posterior(x, *_vp(t, alpha_bar), mix).x0


def exact_eps(x, t, mix=SWIRL, alpha_bar=ALPHA_BAR):
    """What a perfect ε-prediction network outputs: E[ε | x_t] = −√(1 − ᾱ_t) · ∇ log p_t(x_t)."""
    return -_col(np.sqrt(1 - alpha_bar_at(t, alpha_bar)), x) * mixture_score(x, t, mix, alpha_bar)


def mixture_score_sigma(x, sigma, mix=SWIRL):
    """The exact score of the σ view, x = x0 + σ·ε (used by Langevin dynamics and NCSN)."""
    return mixture_posterior(x, 1.0, sigma, mix).score


def exact_velocity(x, t, mix=SWIRL):
    """The exact flow-matching velocity E[ε − x0 | x_t] for x_t = (1 − t)·x0 + t·ε, 0 < t ≤ 1. Since
    x_t = x0 + t·v, it is (x_t − E[x0 | x_t])/t."""
    t = np.asarray(t, dtype=float)
    return (x - mixture_posterior(x, 1 - t, t, mix).x0) / _col(t, x)


def arm_probability(x, t, arm=0, mix=SWIRL, alpha_bar=ALPHA_BAR):
    """p(arm | x_t): the exact chance that a noisy point came from this arm (a perfect classifier)."""
    resp = mixture_posterior(x, *_vp(t, alpha_bar), mix).resp
    return resp[:, mix.labels == arm].sum(1)


# ============================================================================ guidance (Ep 13, 14)


def cfg(eps_uncond, eps_cond, w):
    """Classifier-free guidance (Ho & Salimans 2022), in the "guidance scale" form of Stable Diffusion and
    diffusers:  ε̃ = ε_u + w·(ε_c − ε_u).  w = 0 ignores the condition, w = 1 is the plain conditional model,
    w > 1 pushes past it. The paper writes ε̃ = (1 + w′)·ε_c − w′·ε_u, the same thing with w′ = w − 1
    (so the paper's w′ = 0 is this w = 1). This course quotes w in the guidance-scale form unless it says so."""
    return eps_uncond + w * (eps_cond - eps_uncond)


@lru_cache(maxsize=4)
def blobs_mixture(radius=0.6, std=0.15):
    """The second conditional toy (as in Ho & Salimans' Figure 2): three labelled round blobs, std 0.15, centred
    at radius 0.6 at 90°, 210° and 330° (class 0 at the top). Guidance on the swirl's arms only concentrates the
    samples (the thin arm catches them again); on blobs, high w pushes them out beyond their class's data."""
    angle = np.radians([90.0, 210.0, 330.0])
    means = radius * np.stack([np.cos(angle), np.sin(angle)], axis=1)
    mix = Mixture(means, np.full(3, float(std)), np.full(3, 1 / 3), np.arange(3))
    for array in mix:
        array.setflags(write=False)
    return mix


def blobs(n, seed=0):
    """n labelled points of the three blobs: (points, labels)."""
    return blobs_mixture().sample(n, seed)


def guided_eps(w, label=0, mix=SWIRL, alpha_bar=ALPHA_BAR):
    """An ε-model for the exactly guided swirl: cfg(ε̂(x_t), ε̂(x_t | arm = label), w), both computed exactly."""
    cond = mix.given(label)

    def eps_model(x, t):
        return cfg(exact_eps(x, t, mix, alpha_bar), exact_eps(x, t, cond, alpha_bar), w)
    return eps_model


# ============================================================================ the tiny trained denoiser (Ep 5, 8, 11)

EMB_DIM = 32


def timestep_embedding(t, dim=EMB_DIM, max_period=10000.0):
    """Sinusoidal time embedding (DDPM, after the Transformer's positions): dim/2 frequencies from 1 down to
    1/max_period, returning [sin(t·f), cos(t·f)] → shape (len(t), dim). DDPM's code divides by half − 1."""
    half = dim // 2
    freqs = np.exp(-np.log(max_period) * np.arange(half) / half)
    args = np.asarray(t, dtype=float).reshape(-1, 1) * freqs
    return np.concatenate([np.sin(args), np.cos(args)], axis=1)


def silu(z):
    """SiLU / swish, z·sigmoid(z): the activation of the DDPM U-Net (written with tanh: no overflow)."""
    return z * 0.5 * (1 + np.tanh(z / 2))


def mlp_input(x, t):
    """What the tiny MLP sees (98 numbers): x itself, the same sinusoidal trick applied to each coordinate
    (wavelengths 0.1 … 9, so it can resolve the swirl's thin arms; without them it blurs the last steps), and
    the time embedding."""
    t = np.broadcast_to(np.asarray(t, dtype=float), (len(x),))
    coords = [timestep_embedding(64 * x[:, j], EMB_DIM, max_period=128) for j in range(x.shape[1])]
    return np.concatenate([x, *coords, timestep_embedding(t)], axis=1)


def mlp_forward(params, x, t):
    """The tiny denoiser: mlp_input(x, t) (98 numbers) → 3 × (Linear 128 → SiLU) → Linear → 2 numbers."""
    h = mlp_input(x, t)
    layers = len(params) // 2
    for i in range(layers):
        h = h @ params[f"W{i}"] + params[f"b{i}"]
        if i < layers - 1:
            h = silu(h)
    return h


def count_params(params):
    """Number of trainable numbers in a dict of arrays (the tiny MLP: 45,954)."""
    return int(sum(np.size(v) for v in params.values()))


@lru_cache(maxsize=4)
def _load_npz(name):
    path = WEIGHTS / name
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; train it with research/experiments/scripts/toy_train.py")
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def load_weights(name="swirl_mlp.npz"):
    """The trained weights only (W0, b0, …), e.g. for count_params; the .npz also holds its training log."""
    return {k: v for k, v in _load_npz(name).items() if re.fullmatch(r"[Wb]\d+", k)}


def load_swirl_model():
    """The tiny MLP trained with the simple ε-loss (Ho et al. 2020) on the swirl → eps_model(x, t)."""
    params = load_weights("swirl_mlp.npz")
    return lambda x, t: mlp_forward(params, x, t)


def load_swirl_flow():
    """The same MLP trained by flow matching → v_model(x, t), t ∈ [0, 1] (embedded as 1000·t)."""
    params = load_weights("swirl_flow.npz")
    return lambda x, t: mlp_forward(params, x, 1000 * np.asarray(t, dtype=float))


# ============================================================================ samplers (Ep 6, 7, 9, 10, 11)


class Trajectory(NamedTuple):
    xs: np.ndarray    # (steps + 1, n, d): xs[0] the starting noise, xs[-1] the samples
    ts: np.ndarray    # (steps + 1,): the time of each state (DDPM step, flow time, or σ for Langevin)
    x0s: np.ndarray   # (steps, n, d): the model's guess of the clean point at each step


def ddim_timesteps(steps, T=T):
    """`steps` evenly spaced DDPM steps from T down to 0 ("trailing" spacing): 4 → [1000, 750, 500, 250, 0]."""
    return np.round(np.linspace(T, 0, steps + 1)).astype(int)


def _grid(steps, alpha_bar):
    return ddim_timesteps(steps, len(alpha_bar) - 1) if np.ndim(steps) == 0 else np.asarray(steps)


def ddpm_sample(eps_model, x, alpha_bar=ALPHA_BAR, variance="beta", seed=0, steps=None):
    """Ancestral sampling (Ho et al. 2020, Algorithm 2), from t = T down to 1:
        x_{t−1} = (x_t − β_t/√(1 − ᾱ_t) · ε̂) / √(1 − β_t)  +  σ_t·z,   z fresh noise (none on the last step)
    variance="beta": σ_t² = β_t;  "beta_tilde": σ_t² = β̃_t = β_t·(1 − ᾱ_{t−1})/(1 − ᾱ_t), the true
    posterior's. Optional `steps` skips steps; a jump t → s then uses β = 1 − ᾱ_t/ᾱ_s."""
    if variance not in ("beta", "beta_tilde"):
        raise ValueError("variance must be 'beta' or 'beta_tilde'")
    rng = np.random.default_rng(seed)
    ts = _grid(len(alpha_bar) - 1 if steps is None else steps, alpha_bar)
    xs, x0s = [x], []
    for t, s in zip(ts[:-1], ts[1:]):
        ab_t, ab_s = alpha_bar_at(t, alpha_bar), alpha_bar_at(s, alpha_bar)
        beta = 1 - ab_t / ab_s
        eps = eps_model(x, t)
        x0s.append(x0_from_eps(x, eps, ab_t))
        var = beta if variance == "beta" else beta * (1 - ab_s) / (1 - ab_t)
        z = rng.standard_normal(x.shape)
        x = (x - beta / np.sqrt(1 - ab_t) * eps) / np.sqrt(1 - beta) + np.sqrt(var) * z * (s > 0)
        xs.append(x)
    return Trajectory(np.array(xs), ts, np.array(x0s))


def ddim_sample(eps_model, x, steps=50, eta=0.0, alpha_bar=ALPHA_BAR, seed=0):
    """DDIM (Song, Meng & Ermon 2021): guess x0, then re-noise it to the next step s with the SAME noise guess:
        x_s = √ᾱ_s · x̂0 + √(1 − ᾱ_s − σ²) · ε̂ + σ·z,   σ = η·√((1 − ᾱ_s)/(1 − ᾱ_t) · (1 − ᾱ_t/ᾱ_s))
    η = 0: deterministic, the same noise always gives the same sample; η = 1: DDPM with β̃.
    `steps`: how many (evenly spaced from T to 0) or an explicit descending list of steps."""
    rng = np.random.default_rng(seed)
    ts = _grid(steps, alpha_bar)
    xs, x0s = [x], []
    for t, s in zip(ts[:-1], ts[1:]):
        ab_t, ab_s = alpha_bar_at(t, alpha_bar), alpha_bar_at(s, alpha_bar)
        eps = eps_model(x, t)
        x0 = x0_from_eps(x, eps, ab_t)
        sigma = eta * np.sqrt((1 - ab_s) / (1 - ab_t) * (1 - ab_t / ab_s))
        z = rng.standard_normal(x.shape)
        x = np.sqrt(ab_s) * x0 + np.sqrt(1 - ab_s - sigma ** 2) * eps + sigma * z
        xs.append(x)
        x0s.append(x0)
    return Trajectory(np.array(xs), ts, np.array(x0s))


def _probability_flow(eps_model, x, steps, alpha_bar, space, heun):
    ts = _grid(steps, alpha_bar)
    ab = alpha_bar_at(ts, alpha_bar)
    if space == "sigma":    # state y = x_t/√ᾱ_t = x0 + σ·ε, clock σ, slope dy/dσ = ε̂
        clock, scale = sigma_from_alpha_bar(ab), np.sqrt(ab)
    elif space == "t":      # state x_t itself, clock t, slope dx/dt = −½β(t)·(x − ε̂/√(1 − ᾱ_t))
        clock, scale, rate = ts.astype(float), np.ones_like(ab), _beta_rate(ts, alpha_bar)
    else:
        raise ValueError("space must be 'sigma' or 't'")

    def slope(y, i):
        eps = eps_model(y * scale[i], ts[i])
        d = eps if space == "sigma" else -0.5 * rate[i] * (y - eps / np.sqrt(1 - ab[i]))
        return d, x0_from_eps(y * scale[i], eps, ab[i])

    y, xs, x0s = x / scale[0], [x], []
    for i in range(len(ts) - 1):
        d, x0 = slope(y, i)
        h = clock[i + 1] - clock[i]
        y_next = y + h * d                                   # Euler
        if heun and ab[i + 1] < 1:                           # Heun: average with the slope at the far end
            y_next = y + h * (d + slope(y_next, i + 1)[0]) / 2
        y = y_next
        xs.append(y * scale[i + 1])
        x0s.append(x0)
    return Trajectory(np.array(xs), ts, np.array(x0s))


def euler_ode(eps_model, x, steps=50, alpha_bar=ALPHA_BAR, space="sigma"):
    """Euler steps on the probability-flow ODE (Song et al. 2021): no fresh noise, one model call per step.
    space="sigma": in the x0 + σ·ε view the ODE is simply dy/dσ = ε̂, and Euler there IS DDIM (η = 0).
    space="t": the same ODE in DDPM time, dx/dt = −½β(t)·(x − ε̂/√(1 − ᾱ_t)) (Song et al.'s VP form; on the swirl
    it is about as accurate as DDIM, so step placement, not this choice, is the lesson there).
    `steps`: a count (DDPM steps evenly spaced) or a descending list, e.g. timestep_from_sigma(karras_sigmas(…))."""
    return _probability_flow(eps_model, x, steps, alpha_bar, space, heun=False)


def heun_ode(eps_model, x, steps=25, alpha_bar=ALPHA_BAR, space="sigma"):
    """Heun's method (Karras et al. 2022, Algorithm 1): take the Euler step, look at the slope where it lands,
    redo the step with the average slope. 2 model calls per step, 1 on the last (σ = 0): NFE = 2·steps − 1."""
    return _probability_flow(eps_model, x, steps, alpha_bar, space, heun=True)


def flow_euler(v_model, x, steps=16):
    """Flow matching's sampler: Euler steps along the velocity from t = 1 (noise) to t = 0 (data),
    x ← x + (t_next − t)·v̂(x, t). One step sends every point to the data's mean; straighter paths need fewer."""
    ts = np.linspace(1.0, 0.0, steps + 1) if np.ndim(steps) == 0 else np.asarray(steps, dtype=float)
    xs, x0s = [x], []
    for t, s in zip(ts[:-1], ts[1:]):
        v = v_model(x, t)
        x0s.append(x - t * v)                                # x_t = x0 + t·v, so the clean guess is x_t − t·v̂
        x = x + (s - t) * v
        xs.append(x)
    return Trajectory(np.array(xs), ts, np.array(x0s))


def langevin_sample(score_fn, x, sigmas, steps_per_level=100, step_size=2e-5, seed=0):
    """Annealed Langevin dynamics (Song & Ermon 2019, Algorithm 1): at each noise level σ, largest first, take
    steps  x ← x + (α/2)·score(x, σ) + √α·z  with α = step_size·σ²/σ_last². A single σ is plain Langevin.
    Paths are in the σ view (x0 + σ·ε); x0s holds Tweedie's clean guess x + σ²·score."""
    rng = np.random.default_rng(seed)
    sigmas = np.atleast_1d(np.asarray(sigmas, dtype=float))
    xs, ts, x0s = [x], [sigmas[0]], []
    for sigma in sigmas:
        alpha = step_size * (sigma / sigmas[-1]) ** 2
        for _ in range(steps_per_level):
            score = score_fn(x, sigma)
            x0s.append(x + sigma ** 2 * score)
            x = x + alpha / 2 * score + np.sqrt(alpha) * rng.standard_normal(x.shape)
            xs.append(x)
            ts.append(sigma)
    return Trajectory(np.array(xs), np.array(ts), np.array(x0s))


# ============================================================================ metrics (Ep 10, 19)


def gaussian_stats(points):
    """Mean and covariance of a cloud of feature vectors: all that FID keeps of each set."""
    p = np.asarray(points, dtype=float)
    return p.mean(0), np.atleast_2d(np.cov(p, rowvar=False))


def _sqrt_psd(c):
    lam, vec = np.linalg.eigh((c + c.T) / 2)
    return (vec * np.sqrt(np.clip(lam, 0, None))) @ vec.T


def frechet_distance(mu1, cov1, mu2, cov2):
    """FID's formula (Heusel et al. 2017), the Fréchet distance between N(μ1, C1) and N(μ2, C2):
        ‖μ1 − μ2‖² + Tr(C1) + Tr(C2) − 2·Tr((C1·C2)^½)
    C1·C2 is not symmetric, but it is similar to √C1·C2·√C1, which is symmetric and ≥ 0, so both have the same
    eigenvalues λ and Tr((C1·C2)^½) = Σ √λ. Two symmetric eigh calls; no scipy.linalg.sqrtm needed."""
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    cov1, cov2 = np.atleast_2d(cov1), np.atleast_2d(cov2)
    root1 = _sqrt_psd(cov1)
    middle = root1 @ cov2 @ root1
    lam = np.linalg.eigvalsh((middle + middle.T) / 2)
    tr_sqrt = np.sqrt(np.clip(lam, 0, None)).sum()
    return float(((mu1 - mu2) ** 2).sum() + np.trace(cov1) + np.trace(cov2) - 2 * tr_sqrt)


def fd_from_samples(a, b):
    """The Fréchet distance between two clouds of features (FID when the features come from Inception)."""
    return frechet_distance(*gaussian_stats(a), *gaussian_stats(b))


@lru_cache(maxsize=1)
def _dense_curve():
    return swirl_curve(2000)        # spacing ≈ 0.0023 along each arm


def nearest_on_swirl(points):
    """For each point: distance to the swirl's centre line, which arm is nearest, and where along it (0 → 1)."""
    p = np.atleast_2d(np.asarray(points, dtype=float))
    dense = _dense_curve()
    curve = dense.reshape(-1, 2)
    cc = (curve ** 2).sum(1)
    d2, idx = np.empty(len(p)), np.empty(len(p), dtype=int)
    for i in range(0, len(p), 512):
        q = p[i:i + 512]
        D = (q ** 2).sum(1, keepdims=True) - 2 * q @ curve.T + cc
        j = D.argmin(1)
        idx[i:i + 512], d2[i:i + 512] = j, D[np.arange(len(q)), j]
    return np.sqrt(np.maximum(d2, 0)), idx // dense.shape[1], (idx % dense.shape[1]) / (dense.shape[1] - 1)


def swirl_distance(points):
    """Sample quality: mean distance from each sample to the swirl's centre line. The target is the data's own
    0.024 (its noise across the curve, 0.03·√(2/π)), not 0: deterministic samplers with few steps score BELOW it
    (too tidy: the last step returns the denoiser's average guess), and points filling the gaps between the arms
    score only ≈ 0.08, so read it with off_swirl() and swirl_coverage(). One step to the mean scores 0.15."""
    return float(nearest_on_swirl(points)[0].mean())


def off_swirl(points, far=0.1):
    """Stragglers: the fraction of samples farther than `far` (3⅓ noise widths) from the curve. Data: 0.1 %."""
    return float(np.mean(nearest_on_swirl(points)[0] > far))


def swirl_profile(points, pieces=20, near=0.1):
    """Share of the samples on each of `pieces` equal-length pieces of each arm, shape (2, pieces); only samples
    within `near` of the curve count. The data gives 1/(2·pieces) everywhere."""
    d, arm, u = nearest_on_swirl(points)
    piece = arm * pieces + np.minimum((u * pieces).astype(int), pieces - 1)
    return np.bincount(piece[d < near], minlength=2 * pieces).reshape(2, pieces) / len(d)


def swirl_coverage(points, pieces=20, arm=None, near=0.1):
    """Diversity: the fraction of the curve's pieces (both arms, or one `arm`) that get at least a quarter of
    their fair share of samples. The data scores 1.0; a sampler that only makes one arm scores 0.5."""
    profile = swirl_profile(points, pieces, near)
    profile = profile if arm is None else profile[[arm]]
    return float(np.mean(profile >= 0.25 / profile.size))


# ============================================================================ measured results


_JSON_CACHE = {}


def _experiment_files():
    data = {}
    for path in sorted(EXPERIMENTS.glob("*.json")):
        stamp = path.stat().st_mtime_ns
        if _JSON_CACHE.get(path, (None,))[0] != stamp:
            _JSON_CACHE[path] = (stamp, json.loads(path.read_text()))
        data[path.stem] = _JSON_CACHE[path][1]
    return data


def measured(*keys):
    """A result from research/experiments/*.json, e.g. measured("toy", "headline", "ddpm1000_exact_dist").
    The first key names the file ("toy", "mac_runs"); it may be left out when the next key is in only one file.
    A missing key raises KeyError listing what is there."""
    data = _experiment_files()
    if keys and keys[0] not in data:
        owners = [stem for stem, d in data.items() if isinstance(d, dict) and keys[0] in d]
        if len(owners) > 1:
            raise KeyError(f"{keys[0]!r} is in several files {owners}; name the file first")
        keys = tuple(owners) + keys
    value, path = data, "research/experiments"
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        elif isinstance(value, list) and isinstance(key, int) and -len(value) <= key < len(value):
            value = value[key]
        else:
            there = sorted(map(str, value)) if isinstance(value, dict) else f"a list of {len(value)}" \
                if isinstance(value, list) else repr(value)
            raise KeyError(f"no {key!r} in {path}; available: {there}")
        path += f"/{key}"
    return value


def load_mnist(split="train", limit=None):
    """MNIST as (images, labels): images uint8 (n, 28, 28) with 0 = black background and 255 = ink, labels uint8 (n,).
    split "train" (60,000) or "test" (10,000), read from research/experiments/data/mnist/*-ubyte.gz (LeCun's IDX
    format: a 16-byte header for images, 8 bytes for labels). Scale to the model's range with x / 127.5 - 1."""
    import gzip
    prefix = {"train": "train", "test": "t10k"}[split]
    folder = HERE / "research" / "experiments" / "data" / "mnist"
    with gzip.open(folder / f"{prefix}-images-idx3-ubyte.gz", "rb") as f:
        images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
    with gzip.open(folder / f"{prefix}-labels-idx1-ubyte.gz", "rb") as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    if limit is not None:
        images, labels = images[:limit], labels[:limit]
    return images, labels


# ============================================================================ self-checks


def _self_check():
    global EXPERIMENTS                                    # swapped for a temp dir to test measured()
    import subprocess
    import sys
    import time

    start = time.perf_counter()
    ok = lambda cond, msg: (print(f"  ok  {msg}") if cond else (_ for _ in ()).throw(AssertionError(msg)))
    rng = np.random.default_rng(0)
    ab, abc = ALPHA_BAR, ALPHA_BAR_COSINE

    print("schedules")
    ok(len(ab) == T + 1 == 1001 and ab[0] == 1 and np.all(np.diff(ab) < 0), "ᾱ has T + 1 entries, starts at 1, falls")
    ok(abs(ab[T] - 4.035e-5) < 0.01e-5, f"linear ᾱ_1000 = {ab[T]:.4g} (the known 4.04e-5)")
    f = np.cos((np.arange(T + 1) / T + 0.008) / 1.008 * np.pi / 2) ** 2
    ok(np.allclose(abc[:T], (f / f[0])[:T], rtol=1e-12, atol=0), "cosine ᾱ matches f(t)/f(0) up to the clipped last step")
    ok(np.isclose(snr(ab[500]), np.exp(log_snr(ab[500]))) and snr(ab)[0] == np.inf, "snr = exp(log_snr), ∞ at t = 0")
    sig = sigma_from_alpha_bar(ab[1:])
    ok(np.allclose(alpha_bar_from_sigma(sig), ab[1:]) and np.allclose(sig, 1 / np.sqrt(snr(ab[1:]))), "σ ↔ ᾱ round trip; σ = 1/√SNR")
    ok(np.allclose(timestep_from_sigma(sig), np.arange(1, T + 1)), "timestep_from_sigma inverts the schedule")
    ks = karras_sigmas(10, 0.002, 80)
    ok(len(ks) == 11 and np.isclose(ks[0], 80) and np.isclose(ks[-2], 0.002) and ks[-1] == 0 and np.all(np.diff(ks) < 0),
       "karras_sigmas: σ_max … σ_min, then 0")

    print("forward process and the swirl")
    x0 = swirl(20000, seed=1)
    r = np.linalg.norm(x0, axis=1)
    ok(np.mean(r < 1.05) > 0.99 and abs(x0.std() - 0.5) < 0.05, f"swirl fits the unit disc, std {x0.std():.3f} per coordinate")
    m, c = gaussian_stats(x0)
    mix_mean = SWIRL.weights @ SWIRL.means
    mix_cov = (SWIRL.weights[:, None] * SWIRL.means).T @ SWIRL.means - np.outer(mix_mean, mix_mean) + 0.03 ** 2 * np.eye(2)
    ok(np.allclose(m, mix_mean, atol=0.01) and np.allclose(c, mix_cov, atol=0.01), "swirl samples have the mixture's mean and covariance")
    ok(np.array_equal(swirl(100, 3), swirl_labelled(100, 3)[0]), "swirl and swirl_labelled give the same points")
    eps = rng.standard_normal(x0.shape)
    xt = q_sample(x0, 500, eps)
    ok(np.allclose(xt.var(0), ab[500] * x0.var(0) + 1 - ab[500], rtol=0.05), "q_sample: Var x_t = ᾱ·Var x0 + (1 − ᾱ)")
    tt = rng.integers(0, T + 1, len(x0))
    ok(np.allclose(q_sample(x0, tt, eps)[5], q_sample(x0[5:6], tt[5], eps[5:6])[0]), "q_sample takes one t per row")
    ok(np.allclose(x0_from_eps(xt, eps, ab[500]), x0) and np.allclose(eps_from_x0(xt, x0, ab[500]), eps), "x0 ↔ ε conversions")
    fl = flow_interpolant(x0, 0.3, eps)
    ok(np.allclose(fl, x0 + 0.3 * flow_velocity(x0, eps)), "flow path x_t = x0 + t·(ε − x0)")
    vt = v_target(x0, eps, ab[500])
    ok(np.allclose(np.sqrt(ab[500]) * xt - np.sqrt(1 - ab[500]) * vt, x0), "v-prediction: x0 = √ᾱ·x_t − √(1 − ᾱ)·v")
    ok(np.isclose(swirl_distance(swirl_curve(50).reshape(-1, 2)), 0, atol=1e-3) and
       abs(swirl_distance(x0) - 0.03 * np.sqrt(2 / np.pi)) < 0.002, f"swirl_distance: curve 0, data {swirl_distance(x0):.4f}")
    ok(swirl_coverage(x0) == 1.0 and swirl_coverage(x0[swirl_labelled(20000, 1)[1] == 0]) == 0.5, "coverage: data 1.0, one arm 0.5")

    print("exact scores, denoiser, velocity")
    pts = rng.uniform(-1.3, 1.3, (200, 2))
    for t in (0, 30, 300, 900):
        a_, b_ = np.sqrt(ab[t]), np.sqrt(1 - ab[t])
        h = 1e-5
        fd = np.stack([(mixture_posterior(pts + h * e, a_, b_).log_density - mixture_posterior(pts - h * e, a_, b_).log_density) / (2 * h)
                       for e in np.eye(2)], axis=1)
        s = mixture_score(pts, t)
        ok(np.allclose(fd, s, rtol=1e-4, atol=1e-4 * np.abs(s).max()), f"t = {t}: score = finite-difference ∇ log p_t")
        if t:
            tweedie = (pts + (1 - ab[t]) * s) / np.sqrt(ab[t])
            ok(np.allclose(exact_denoiser(pts, t), tweedie), f"t = {t}: E[x0 | x_t] = Tweedie (x + (1 − ᾱ)·score)/√ᾱ")
    grid = np.stack(np.meshgrid(np.linspace(-3, 3, 241), np.linspace(-3, 3, 241)), -1).reshape(-1, 2)
    ok(abs(mixture_density(grid, 100).sum() * (6 / 240) ** 2 - 1) < 2e-3, "p_t integrates to 1")
    t = 300
    xt = q_sample(x0, t, eps)
    D = exact_denoiser(xt, t)
    ok(abs(np.mean(np.sum((x0 - D) * D, 1))) < 2e-3 and np.mean((x0 - D) ** 2) < np.mean((x0 - D * 1.05) ** 2),
       "the denoiser is the MMSE estimate (error ⟂ estimate; scaling it is worse)")
    ok(np.allclose(exact_eps(xt, t), (xt - np.sqrt(ab[t]) * D) / np.sqrt(1 - ab[t])), "exact_eps = (x_t − √ᾱ·E[x0|x_t])/√(1 − ᾱ)")
    tv = np.array([0.2, 0.7])
    ok(np.allclose(exact_velocity(pts[:2], tv), [exact_velocity(pts[i:i + 1], tv[i])[0] for i in range(2)]), "velocity takes one t per row")
    ok(np.allclose(exact_velocity(pts, 1.0), pts - mix_mean), "at t = 1 the velocity points from the data mean to x")
    sig = 0.4
    ok(np.allclose(mixture_score_sigma(pts, sig), np.sqrt(alpha_bar_from_sigma(sig)) *
                   mixture_posterior(pts * np.sqrt(alpha_bar_from_sigma(sig)), np.sqrt(alpha_bar_from_sigma(sig)),
                                     np.sqrt(1 - alpha_bar_from_sigma(sig))).score), "σ-view score = √ᾱ × x_t-view score")

    print("conditioning and guidance")
    lab = SWIRL.given(0)
    pa = arm_probability(pts, 200)
    bayes = mixture_score(pts, 200) + np.stack([(np.log(arm_probability(pts + 1e-5 * e, 200)) -
                                                 np.log(arm_probability(pts - 1e-5 * e, 200))) / 2e-5 for e in np.eye(2)], 1)
    ok(np.allclose(mixture_score(pts, 200, lab), bayes, rtol=1e-4, atol=1e-4), "∇log p(x|arm) = ∇log p(x) + ∇log p(arm|x) (Bayes)")
    ok(np.all((pa > 0) & (pa < 1)) and np.allclose(arm_probability(pts, 200, 1) + pa, 1), "arm probabilities sum to 1")
    eu, ec = rng.standard_normal((2, 5, 2))
    ok(np.allclose(cfg(eu, ec, 0), eu) and np.allclose(cfg(eu, ec, 1), ec) and
       np.allclose(cfg(eu, ec, 4), (1 + 3) * ec - 3 * eu), "cfg: w = 0 uncond, 1 cond, w = 1 + w′ of Ho & Salimans")

    print("samplers")
    z = gaussian_noise(300, seed=5)
    dd = ddim_sample(exact_eps, z, steps=20)
    eu_ = euler_ode(exact_eps, z, steps=20)
    ok(dd.xs.shape == (21, 300, 2) and np.allclose(dd.xs, eu_.xs, atol=1e-9), "DDIM (η = 0) = Euler on the PF-ODE in σ, step by step")
    d1 = ddim_sample(exact_eps, z, steps=40, eta=1.0, seed=9)
    p1 = ddpm_sample(exact_eps, z, variance="beta_tilde", seed=9, steps=40)
    ok(np.allclose(d1.xs, p1.xs, atol=1e-9), "DDIM with η = 1 = DDPM ancestral with β̃ (same noise)")
    p = ddpm_sample(exact_eps, z, seed=1)
    ok(p.xs.shape == (1001, 300, 2) and swirl_distance(p.xs[-1]) < 0.035 and swirl_coverage(p.xs[-1]) > 0.9,
       f"DDPM 1000 steps, exact ε: distance {swirl_distance(p.xs[-1]):.3f}, coverage {swirl_coverage(p.xs[-1]):.2f}")
    d50 = ddim_sample(exact_eps, z, steps=50).xs[-1]
    ok(swirl_distance(d50) < 0.035 and off_swirl(d50) < 0.1,
       f"DDIM 50 steps, exact ε: distance {swirl_distance(d50):.3f}, off the swirl {off_swirl(d50):.1%}")
    karras = lambda n: timestep_from_sigma(karras_sigmas(n, sigma_from_alpha_bar(ab[1]), sigma_from_alpha_bar(ab[T])))
    ref = heun_ode(exact_eps, z, karras(400)).xs[-1]
    err = lambda tr: np.linalg.norm(tr.xs[-1] - ref, axis=1).mean()
    ok(np.isclose(karras(25)[0], T) and karras(25)[-1] == 0 and err(heun_ode(exact_eps, z, karras(25))) < 0.5 * err(euler_ode(exact_eps, z, 49)),
       "Heun on a Karras σ grid (NFE 49) beats Euler/DDIM (NFE 49) by 2×+")
    ok(err(euler_ode(exact_eps, z, 400, space="t")) < 0.01 and err(euler_ode(exact_eps, z, 400)) < 0.01, "Euler in t and in σ converge to the same ODE solution")
    fe = flow_euler(exact_velocity, z, steps=16)
    tf = np.linspace(1, 0, 17)
    with np.errstate(divide="ignore"):
        same = ddim_sample(exact_eps, z, timestep_from_sigma(tf / (1 - tf))).xs[-1]
    ok(np.abs(fe.xs[-1] - same).max() < 2e-3, "exact flow Euler = DDIM on the σ grid σ = t/(1 − t)")
    ok(np.allclose(flow_euler(exact_velocity, z, 1).xs[-1], mix_mean), "flow Euler with 1 step sends every point to the data mean")
    sig = np.geomspace(1.0, 0.01, 8)
    lv = langevin_sample(mixture_score_sigma, rng.uniform(-1, 1, (300, 2)), sig, 60, 2e-5, seed=2)
    ok(lv.xs.shape == (481, 300, 2) and swirl_distance(lv.xs[-1]) < 0.04, f"annealed Langevin lands on the swirl ({swirl_distance(lv.xs[-1]):.3f})")
    g = ddim_sample(guided_eps(1.0), z, steps=50).xs[-1]
    ok(np.mean(nearest_on_swirl(g)[1] == 0) > 0.97, "guidance w = 1 (exact conditional) makes arm 0 only")

    print("metrics and helpers")
    mu1, mu2, a1, a2 = np.array([1.0, 2.0]), np.array([0.0, -1.0]), np.array([0.5, 2.0]), np.array([1.5, 0.3])
    closed = np.sum((mu1 - mu2) ** 2) + np.sum((np.sqrt(a1) - np.sqrt(a2)) ** 2)
    ok(np.isclose(frechet_distance(mu1, np.diag(a1), mu2, np.diag(a2)), closed), "FD = ‖Δμ‖² + Σ(√a − √b)² for diagonal covariances")
    A, B = rng.standard_normal((2, 6, 6))
    C1, C2 = A @ A.T + 0.1 * np.eye(6), B @ B.T + 0.1 * np.eye(6)
    direct = np.trace(C1) + np.trace(C2) - 2 * np.sqrt(np.linalg.eigvals(C1 @ C2).real.clip(0)).sum()
    ok(np.isclose(frechet_distance(np.zeros(6), C1, np.zeros(6), C2), direct) and
       np.isclose(frechet_distance(np.zeros(6), C1, np.zeros(6), C2), frechet_distance(np.zeros(6), C2, np.zeros(6), C1)) and
       abs(frechet_distance(np.zeros(6), C1, np.zeros(6), C1)) < 1e-9, "FD: matches eig(C1·C2), symmetric, 0 for identical")
    emb = timestep_embedding(np.array([0, 10, 999]), 32)
    ok(emb.shape == (3, 32) and np.allclose(emb[0], np.r_[np.zeros(16), np.ones(16)]) and
       np.allclose(np.linalg.norm(emb, axis=1), 4), "timestep_embedding: (n, dim), [sin, cos], norm √(dim/2)")
    ok(count_params({"W": np.zeros((34, 128)), "b": np.zeros(128)}) == 4480, "count_params")

    print("trained models and measured results")
    if (WEIGHTS / "swirl_mlp.npz").exists():
        model = load_swirl_model()
        ok(count_params(load_weights()) == 45954, "the tiny MLP has 45,954 parameters")
        xt = q_sample(x0[:2000], 300, eps[:2000])
        rel = np.mean((model(xt, 300) - exact_eps(xt, 300)) ** 2) / np.mean(exact_eps(xt, 300) ** 2)
        ok(rel < 0.1, f"trained ε-model ≈ exact ε at t = 300 (relative error {rel:.1e})")
        ok(swirl_distance(ddim_sample(model, z, 50).xs[-1]) < 0.06, "trained ε-model + DDIM 50 lands near the swirl")
    if (WEIGHTS / "swirl_flow.npz").exists():
        ok(swirl_distance(flow_euler(load_swirl_flow(), z, 32).xs[-1]) < 0.06, "trained flow model + 32 Euler steps lands near the swirl")
    if (EXPERIMENTS / "toy.json").exists():
        ok(measured("toy", "headline", "mlp_params") == 45954, "measured('toy', 'headline', …) reads toy.json")
    import tempfile
    real = EXPERIMENTS
    with tempfile.TemporaryDirectory() as tmp:            # the merge rules, on two stand-in files
        EXPERIMENTS = Path(tmp)
        (EXPERIMENTS / "toy.json").write_text('{"headline": {"a": 1}, "only_toy": {"b": [5, 6]}}')
        (EXPERIMENTS / "mac_runs.json").write_text('{"headline": {"a": 2}, "steps": {"50": 3.5}}')
        try:
            ok(measured("toy", "headline", "a") == 1 and measured("mac_runs", "headline", "a") == 2 and
               measured("only_toy", "b", 1) == 6 and measured("steps", "50") == 3.5, "measured: namespaced and short forms")
            for bad, hint in ((("headline", "a"), "several files"), (("toy", "headline", "zz"), "available: ['a']")):
                try:
                    measured(*bad)
                    ok(False, f"measured{bad} should raise")
                except KeyError as e:
                    ok(hint in str(e), f"measured{bad} raises KeyError ({hint})")
        finally:
            EXPERIMENTS = real
    if (HERE / "research" / "experiments" / "data" / "mnist" / "t10k-images-idx3-ubyte.gz").exists():
        xs, ys = load_mnist("test")
        ok(xs.shape == (10000, 28, 28) and ys.shape == (10000,) and int(ys.max()) == 9 and int(xs.max()) == 255,
           "load_mnist('test'): 10,000 digits of 28×28, labels 0–9")
    code = "import time; t = time.perf_counter(); import diffusion_reference; print(time.perf_counter() - t)"
    seconds = min(float(subprocess.run([sys.executable, "-c", code], cwd=HERE, capture_output=True, text=True,
                                       check=True).stdout) for _ in range(3))
    ok(seconds < 0.3, f"import takes {seconds:.3f} s (numpy included)")
    print(f"all checks passed in {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    _self_check()
