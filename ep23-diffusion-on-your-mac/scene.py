"""Out of the Noise · Episode 23 · Diffusion on Your Mac (Module 6 · Video and Your Mac)

What it takes to run a text-to-image model locally, and where the time goes. Running example: the lighthouse,
generated on this Mac Studio (Apple M4 Max, 64 GB unified memory) with Stable Diffusion 2.1 base in MLX
(research/experiments/mac_runs.md §5), with the digit model (mac_runs.md §1, §3, §4) as the small-scale counterpart.
Every timing and size on screen is MEASURED on this Mac (mac_runs.json, loaded with measured(...)) or arithmetic on
those numbers; the published Apple-silicon numbers are cited and dated ("published, not ours").

Sections: hook (7.32 s, 8.13 GB) · one generation on a clock, played in real time (text encoder 0.017 s for 13 tokens,
denoiser 50 steps × 2 batched passes 7.04 s = 96.2%, decoder 0.265 s; one step 0.141 s vs 0.083 s unguided, 1.7×,
Episode 14) · steps × time per step (the steps sweep: 0.33 … 6.90 s, a least-squares line: slope 0.134 s per step,
intercept 0.22 s; SD-Turbo one step 0.25 s, its timing run 0.364 s where the decoder is 75%, Episode 12) · memory
(bytes = parameters × bytes per number: 1.73 + 0.68 + 0.33 = 2.75 GB per model, 5.50 GB downloaded for two; peak 8.13
GB, 10.88 GB with both models; 64 GB unified; FLUX.1 16-bit weights 33.73 GB, published) · the speed limit (Tokens per
Second Episode 23: bandwidth set the speed of language models; here reading the denoiser's weights once would take
3.2 ms at 546 GB/s, one pass takes 83 ms, a batch of two 141 ms: mostly arithmetic) · framework (MLX vs PyTorch-MPS on
the digit network: training 1,455 vs 942 images/s = 1.55×; sampling 5,603 vs 5,259 ≈ even; fp16 no speed-up;
GroupNorm ≈ 25%) · small scale (the digit model: 31.8 min, 6.32 GB; DDIM-50 latency per step at batch 1 / 10 / 100 /
1,000: 4.1 / 4.8 / 23.6 / 177 ms) · published (Apple Core ML Nov 2022, SD 1.4, 50 steps, M1 Ultra 9 s; Core ML SDXL,
20 steps, M2 Ultra 20 s; mflux FLUX.1 schnell, 2 steps, M4 Max ≈ 19 s incl. loading) · recap (steps, passes, bytes).

Sources (research/notes.md §23, §17, Appendix A rows 17/23, Appendix C row 23; read October 2026):
- research/experiments/mac_runs.md §1 (framework benchmark), §3 (training), §4 (sampling, latency), §5 (lighthouse:
  models table, timing breakdown, steps sweep, turbo), §8 (caveats: shared machine, ±5–10% jitter, VAE decode
  0.17–0.27 s), and the 2026-10-27 erratum (13 tokens, not 77). Numbers from mac_runs.json; the HF cache is never
  read (the downloaded models are deleted at the end of the course).
- Apple, ml-stable-diffusion README at its first commit 583cc04 (29 Nov 2022), read Oct 2026: "Standard
  CompVis/stable-diffusion-v1-4 Benchmark", "50 inference steps, 512x512 output image resolution, 77 text token sequence
  length, classifier-free guidance (batch size of 2 for unet)": Mac Studio (M1 Ultra, 64-core GPU) 9 s; "conducted by
  Apple using public beta versions of iOS 16.2, iPadOS 16.2 and macOS 13.1 in November 2022". Current README (read Oct
  2026; the SDXL table dates from July 2023, notes §23): SDXL base 1.0, 1024x1024, 20 steps, classifier-free guidance,
  float16: Mac Studio (M2 Ultra) 20 s, 1.11 it/s; "these numbers do not represent peak HW capability".
- mflux README (community-reported table, versions Nov 2024 – Jan 2026, via notes §23; not on the current main page):
  FLUX.1 [schnell], 2 steps, 1024x1024, not quantised, including model loading: M4 Max (128 GB) ≈ 19 s. FLUX.1 full
  pipeline in 16 bits: 33.73 GB (notes §23 table: ≈ 16.87 B parameters × 2 bytes; = mflux "33.73GB").
- Memory bandwidth of the M4 Max (40-core GPU): 546 GB/s, Apple's specification, as used by the sister course "Tokens
  per Second" (its research/notes.md §2 hardware table; Episode 23 "Your Mac as an Inference Server"). The 3.2 ms is
  arithmetic (1.73 GB ÷ 546 GB/s), not a measurement; the passes are measured.

Illustrative (noted per section): the hook's reveal (pixel-space noise mixed into the finished picture by the course's
DDPM schedule with q_sample — not the run's own frames); the latent tile in the denoiser card (the 30-step run's latent,
channel 1, shown at the matching fraction of the 50 steps); the cross-fade of the picture during the decode.

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
PREFIX = "ep23"
ARR = EXPERIMENTS / "arrays"
IMG = EXPERIMENTS / "images"
LIMG = IMG / "lighthouse"
PX = 135                                          # pixels per scene unit at 1080p
HL = lambda *keys: measured("mac_runs", "headline", *keys)  # noqa: E731
LH = measured("mac_runs", "lighthouse")
MAC_MD = (EXPERIMENTS / "mac_runs.md").read_text()
META = measured("mac_runs", "meta")
MEMORY = LIME                                     # this module's accent: memory in this episode

# ---------------------------------------------------------------------------- the machine

assert "Apple M4 Max" in META["machine"] and "64 GB unified memory" in META["machine"] and META["mlx"] == "0.32.4"
RAM_GB = 64                                                                         # "sixty-four gigabytes"
assert "Mac Studio" in META["machine"]

# ---------------------------------------------------------------------------- one generation (timing run, median of 3)

SD_REPO, TB_REPO = "sd2-community/stable-diffusion-2-1-base", "stabilityai/sd-turbo"
TM = LH["timing"]["cfg7.5_50steps"]
T1 = LH["timing"]["cfg1_50steps_no_cfg"]
TB = LH["timing"]["turbo_1step"]
TXT_S, DEN_S, DEC_S, TOT_S = TM["text_encoder_s"], TM["denoise_total_s"], TM["vae_decode_s"], TM["total_s"]
STEPS, PASSES = TM["denoise_steps"], TM["unet_passes_per_step"]
assert (STEPS, PASSES, TM["latent_shape"], TM["image_shape"]) == (50, 2, [1, 64, 64, 4], [512, 512, 3])
assert (f"{TXT_S:.3f}", f"{DEN_S:.2f}", f"{DEC_S:.3f}", f"{TOT_S:.2f}") == ("0.017", "7.04", "0.265", "7.32")
assert abs(TXT_S + DEN_S + DEC_S - TOT_S) < 1e-3 and round(TOT_S, 2) == HL("lighthouse_total_s_50steps_cfg7.5")
REPEATS = TM["repeat_totals_s"]
assert [f"{r:.2f}" for r in REPEATS] == ["7.35", "7.32", "7.31"] and sorted(REPEATS)[1] == TOT_S  # the median
SHARE = {k: 100 * v / TOT_S for k, v in (("txt", TXT_S), ("den", DEN_S), ("dec", DEC_S))}
assert (f"{SHARE['txt']:.1f}", f"{SHARE['den']:.1f}", f"{SHARE['dec']:.1f}") == ("0.2", "96.2", "3.6")
STEP_TIMES = np.array(TM["step_s"])                                                  # every step of the timed run
assert len(STEP_TIMES) == STEPS and abs(STEP_TIMES.sum() - DEN_S) < 0.01
STEP_ENDS = TXT_S + np.cumsum(STEP_TIMES) * DEN_S / STEP_TIMES.sum()               # when each step finished
STEP_CFG, STEP_ONE = TM["mean_step_s"], T1["mean_step_s"]
assert (f"{STEP_CFG:.3f}", f"{STEP_ONE:.3f}") == ("0.141", "0.083")                 # "a hundred forty-one", "eighty-three"
assert HL("lighthouse_unet_step_s_cfg") == 0.141 and HL("lighthouse_unet_step_s_nocfg") == 0.083
assert f"{STEP_CFG / STEP_ONE:.1f}" == "1.7" and 2 * STEP_ONE > STEP_CFG           # "one point seven times, not two"
assert T1["unet_passes_per_step"] == 1 and f"{T1['total_s']:.2f}" == "4.43"
PEAK = TM["peak_mem_gb"]
assert f"{PEAK:.2f}" == "8.13" and HL("lighthouse_peak_mem_gb") == 8.13              # "just over eight gigabytes"
PROMPT_TEXT = "a lighthouse on a rocky coast at dusk, oil painting"
TOKENS = ["start", "a", "lighthouse", "on", "a", "rocky", "coast", "at", "dusk", ",", "oil", "painting", "end"]
assert f'"{PROMPT_TEXT}"' in MAC_MD and "encoded as **13\ntokens** (start + 11 + end)" in MAC_MD
assert len(TOKENS) == 13                                                            # "thirteen tokens"
assert "text encoder (13 tokens, prompt + empty negative; see erratum)" in MAC_MD
SEED = 0
PIC_FILE = "lighthouse_sd21_cfg7.5_steps50_seed0.png"
GEN_CITE = "generated on this Mac · Stable Diffusion 2.1 base · 512 × 512 · 50 steps · guidance 7.5 · seed 0"

# ---------------------------------------------------------------------------- steps × time per step (the sweep)


def _median(rows, model, steps):
    v = [r["total_s"] for r in rows if r["model"] == model and r["steps"] == steps]
    assert len(v) == 4, (model, steps, v)                                           # seeds 0–3
    return float(np.median(v))


assert all(r["cfg"] == 7.5 for r in LH["steps_sweep"]) and all(r["cfg"] == 0.0 for r in LH["turbo_sweep"])
SWEEP = [1, 2, 4, 8, 20, 50]
SD_T = {s: _median(LH["steps_sweep"], SD_REPO, s) for s in SWEEP}
assert [f"{SD_T[s]:.2f}" for s in SWEEP] == ["0.33", "0.48", "0.75", "1.34", "2.93", "6.90"]
assert HL("lighthouse_sd21_4steps_total_s_median_sweep") == 0.75 and HL("lighthouse_sd21_20steps_total_s_median_sweep") == 2.93
SLOPE, INTER = np.polyfit(SWEEP, [SD_T[s] for s in SWEEP], 1)                       # least squares
assert f"{SLOPE * 1000:.0f}" == "134" and f"{INTER:.2f}" == "0.22"                  # "a hundred thirty-four", "a fifth"
FIXED = [r["total_s"] - r["denoise_s"] for r in LH["steps_sweep"]]                  # text + decode, run by run
assert 0.17 < float(np.median(FIXED)) < 0.20 and 0.15 < INTER < 0.25               # the intercept is the fixed cost
TB_T1 = _median(LH["turbo_sweep"], TB_REPO, 1)
assert f"{TB_T1:.2f}" == "0.25" and abs(TB_T1 - HL("lighthouse_turbo_1step_total_s_median_sweep")) < 1e-3
assert TB["denoise_steps"] == 1 and TB["unet_passes_per_step"] == 1
TB_PARTS = (TB["text_encoder_s"], TB["denoise_total_s"], TB["vae_decode_s"])
assert abs(sum(TB_PARTS) - TB["total_s"]) < 0.002 and f"{TB['total_s']:.3f}" == "0.364"
assert (f"{TB_PARTS[0]:.3f}", f"{TB_PARTS[1] * 1000:.0f}", f"{TB_PARTS[2] * 1000:.0f}") == ("0.017", "76", "271")
TB_DEC_SHARE = TB_PARTS[2] / TB["total_s"]
assert f"{100 * TB_DEC_SHARE:.0f}" == "75"                                          # "most of the time"
assert "The VAE decode varied from 0.17 to 0.27 s" in MAC_MD and "about ±5–10%" in MAC_MD


def sd_png(steps):
    name = f"lighthouse_sd21_steps{steps:02d}_cfg7.5_seed{SEED}.png"
    assert any(r["file"] == name for r in LH["steps_sweep"]) and (LIMG / name).exists(), name
    return LIMG / name


def tb_png(steps):
    name = f"lighthouse_sdturbo_steps{steps}_seed{SEED}.png"
    assert any(r["file"] == name for r in LH["turbo_sweep"]) and (LIMG / name).exists(), name
    return LIMG / name


# ---------------------------------------------------------------------------- sizes: bytes = parameters × bytes per number

PARTS = LH["models"][SD_REPO]["parts"]
assert PARTS == LH["models"][TB_REPO]["parts"]                                       # the same three networks' shapes
BPN = {"float16": 2, "float32": 4}
NETS = [  # key, name, colour, dtype word
    ("unet", "denoiser", MODEL, "fp16"),
    ("text_encoder", "text encoder", PROMPT, "fp16"),
    ("vae", "autoencoder", DATA, "fp32"),
]
SIZE = {}
for _k, _n, _c, _d in NETS:
    _p = PARTS[_k]
    assert _p["params"] * BPN[_p["dtype"]] == _p["bytes_in_memory"] and _d == {"float16": "fp16", "float32": "fp32"}[_p["dtype"]]
    SIZE[_k] = _p["bytes_in_memory"] / 1e9
P_M = {k: PARTS[k]["params"] / 1e6 for k, *_ in NETS}
assert [f"{P_M[k]:.1f}" for k in ("unet", "text_encoder", "vae")] == ["865.9", "340.4", "83.7"]
assert [f"{SIZE[k]:.2f}" for k in ("unet", "text_encoder", "vae")] == ["1.73", "0.68", "0.33"]
WEIGHTS = sum(SIZE.values())
assert f"{WEIGHTS:.2f}" == "2.75" and abs(WEIGHTS - LH["models"][SD_REPO]["weights_active_gb"]) < 0.001
DL = measured("mac_runs", "lighthouse_downloads")
DL_ONE = DL[SD_REPO]["total_bytes"] / 1e9
DL_BOTH = (DL[SD_REPO]["total_bytes"] + DL[TB_REPO]["total_bytes"]) / 1e9
assert f"{DL_ONE:.2f}" == "2.75" and f"{DL_BOTH:.2f}" == "5.50" and HL("lighthouse_download_gb_total") == 5.5
PEAK_BOTH = TB["peak_mem_gb"]
assert f"{PEAK_BOTH:.1f}" == "10.9"                                                 # "ten point nine" (both models loaded)
assert "Peak 10.9 GB, with both\n  models loaded" in MAC_MD
FLUX_GB = 33.73                    # notes §23: FLUX.1 full pipeline ≈ 16.87 B params × 2 B = 33.73 GB (= mflux "33.73GB")
assert round(FLUX_GB) == 34 and FLUX_GB < RAM_GB                                     # "thirty-four gigabytes", fits

# ---------------------------------------------------------------------------- the speed limit (derived)

BW = 546e9                         # Apple's figure for the M4 Max (40-core GPU); Tokens per Second notes §2 hardware table
READ_MS = PARTS["unet"]["bytes_in_memory"] / BW * 1e3
PASS_MS, PAIR_MS = STEP_ONE * 1e3, STEP_CFG * 1e3
assert f"{READ_MS:.1f}" == "3.2" and round(READ_MS) == 3                             # "about three milliseconds"
assert f"{PASS_MS:.0f}" == "83" and PASS_MS / READ_MS > 25                           # one pass: 26× the weight read

# ---------------------------------------------------------------------------- framework (digit network, MLX vs PyTorch-MPS)

FB = measured("mac_runs", "framework_benchmark")
F_MLX, F_PT = FB["mlx"]["mlx_ch32_m124"], FB["torch_mps"]["torch_ch32_m124"]
assert F_MLX["params"] == F_PT["params"] == HL("digit_unet_params") == 3_940_801      # the same network
assert (FB["batch"], FB["infer_batch"], FB["torch_version"]) == (256, 1000, "2.14.1")
TR = (F_MLX["train_img_per_s"], F_PT["train_img_per_s"])
INF = (F_MLX["infer_img_per_s"], F_PT["infer_img_per_s"])
assert [f"{int(v):,}" for v in TR + INF] == ["1,455", "942", "5,603", "5,259"]        # whole images per second (floor)
assert f"{TR[0] / TR[1]:.2f}" == "1.55" and f"{INF[0] / INF[1]:.2f}" == "1.07"        # "one point five five", "about even"
assert "fp16/bf16 inference gave no speed-up for the digit model; GroupNorm costs about 25% of a forward pass" in MAC_MD
assert "compiled MLX step vs eager PyTorch" in MAC_MD

# ---------------------------------------------------------------------------- the digit model: training and latency

DM = measured("mac_runs", "digit_models", "ddpm")
assert (DM["steps"], DM["batch"], f"{DM['wall_min']:.1f}", f"{DM['peak_mem_gb']:.2f}") == (10000, 256, "31.8", "6.32")
assert HL("digit_ddpm_train_minutes") == 31.8 and HL("digit_train_peak_mem_gb") == 6.32
LAT_ROWS = [r for r in measured("mac_runs", "latency", "rows") if r["cfg"] == "ddpm_ddim50_w1"]
LAT_MS = {r["batch"]: r["ms_per_step"] for r in LAT_ROWS}                            # one batch, DDIM 50 steps
DS = measured("mac_runs", "digit_sampling", "configs", "ddpm_ddim50_w1")
assert DS["steps"] == 50 and DS["w"] == 1.0
LAT_MS[1000] = DS["s_per_1000"] / DS["steps"] * 1e3                                  # batch 1,000: 8.87 s for 50 steps
BATCHES = [1, 10, 100, 1000]
assert [f"{LAT_MS[b]:.1f}" for b in BATCHES] == ["4.1", "4.8", "23.6", "177.4"]       # "four … twenty-four … a hundred seventy-seven"
ONE_DIGIT = LAT_MS[1] * DS["steps"] / 1e3
PER_DIGIT_1000 = DS["s_per_1000"]                                                   # ms per digit at batch 1,000
assert f"{ONE_DIGIT:.3f}" == "0.205" and f"{PER_DIGIT_1000:.1f}" == "8.9"            # "a fifth of a second", "nine ms"
assert round(ONE_DIGIT * 1e3 / PER_DIGIT_1000) == 23
assert "At batch 1 the GPU is mostly idle" in MAC_MD
DIG = np.load(ARR / "digit_samples_ddpm_ddim50_w1.npz")
assert "seed 1234" in str(DIG["note"]) and list(DIG["labels"][10:20]) == list(range(10))
DIGITS = DIG["samples"][10:20]

# ---------------------------------------------------------------------------- published (not ours), dated

PUB = [  # when, who, model · settings, Mac, time, sub
    ("Nov 2022", "Apple · Core ML", "SD 1.4 · 512² · 50 steps", "M1 Ultra", "9 s", "guidance on"),
    ("Jul 2023", "Apple · Core ML", "SDXL · 1024² · 20 steps", "M2 Ultra", "20 s", "guidance on"),
    ("2024–26", "mflux · community table", "FLUX.1 schnell · 1024² · 2 steps", "M4 Max · 128 GB", "≈ 19 s", "with loading"),
]
OURS = ("Oct 2026", "this course · MLX", "SD 2.1 base · 512² · 50 steps", "M4 Max · 64 GB", f"{TOT_S:.1f} s", "measured")
assert OURS[4] == "7.3 s"

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
    """An RGB array → an RGBA texture about px pixels wide (small arrays with crisp pixels, large ones resized)."""
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
    nearest = max(rgb.shape[:2]) <= 64 if nearest is None else nearest
    image = m.ImageMobject(texture_of(rgb, int(np.ceil(height * PX)), nearest))
    image.height = height
    return image


class Reel:
    """A picture that steps through frames: `.image`, `.show(k)`."""

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


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))


def rgb_of(hexcode):
    return np.array([int(hexcode[i:i + 2], 16) for i in (1, 3, 5)], float)


def hbar(x0, y, w, h, color, fill=None, stroke=2):
    w = max(w, 0.004)
    return m.Rectangle(width=w, height=h).set_stroke(color, stroke).set_fill(fill or FILLS.get(color, CARD), 1).move_to([x0 + w / 2, y, 0])


def stage(name, sub, color, width, height, center):
    box = card(width, height, color, FILLS[color], radius=0.16, stroke=2.5).move_to(center)
    words = m.VGroup(text(name, 24, INK, weight=m.BOLD), text(sub, 20, MUTED)).arrange(m.DOWN, buff=0.08).move_to(box)
    return m.VGroup(box, words)


PIC50 = load_rgb(LIMG / PIC_FILE)
assert PIC50.shape == (512, 512, 3)

# the hook's reveal (illustrative): pixel-space noise mixed in by the course's DDPM schedule (q_sample)
HOOK_H = 3.3
_hpx = int(np.ceil(HOOK_H * PX))
_x0 = np.asarray(Image.fromarray(PIC50).resize((_hpx, _hpx), Image.LANCZOS)).astype(np.float64) / 127.5 - 1
_eps = tint(np.random.default_rng(23).standard_normal((_hpx, _hpx)), NOISE, -2.5, 2.5).astype(np.float64) / 127.5 - 1  # VIOLET static
_shares = np.linspace(0.0, 1.0, 25) ** 1.6                                         # ᾱ from 0 (noise) to 1 (picture)
HOOK_T = np.interp(_shares, ALPHA_BAR[::-1], np.arange(T + 1)[::-1])                 # the matching DDPM steps
assert HOOK_T[0] > 990 and HOOK_T[-1] == 0
HOOK_FRAMES = [((np.clip(q_sample(_x0, t, _eps), -1, 1) + 1) * 127.5).round().astype(np.uint8) for t in HOOK_T]

# the decode's cross-fade and the denoiser card's latent (the 30-step run, channel 1; illustrative pacing)
_card = rgb_of(CARD)
DECODE_FRAMES = [(_card + (PIC50 - _card) * a).round().astype(np.uint8) for a in np.linspace(0, 1, 10)]
LAT = np.load(ARR / "lighthouse_latents_30steps.npz")
assert "30 steps, cfg 7.5, seed 0" in str(LAT["note"])
XT = LAT["x_t"].astype(np.float64)
SIG = list(LAT["sigma"].astype(np.float64)) + [0.0]
assert XT.shape == (31, 64, 64, 4)
_lo, _hi = np.percentile(XT[30][..., 0], [1, 99])
LAT_FRAMES = [tint(np.clip((XT[k][..., 0] - _lo) / (_hi - _lo), 0, 1), mix(NOISE, DATA, 1 / (1 + SIG[k] ** 2)), 0, 1)
              for k in range(31)]

# ---------------------------------------------------------------------------- the episode


class Ep23DiffusionOnYourMac(Episode):
    number = 23

    def lesson(self):
        self.hook()
        self.timeline()
        self.steps()
        self.memory()
        self.speed_limit()
        self.framework()
        self.small_scale()
        self.published()
        self.recap()

    # ------------------------------------------------------------------ 0 · hook

    def hook(self):
        self.next_section("Hook", notes=(
            f"The lighthouse ({PIC_FILE}), generated on this Mac: SD 2.1 base in MLX, 512×512, 50 steps, guidance 7.5, seed "
            f"0. Timing run (mac_runs §5, median of 3: 7.35 / 7.32 / 7.31 s) {TOT_S:.2f} s; peak memory {PEAK:.2f} GB "
            "(MLX's peak). Machine: Mac Studio, Apple M4 Max, 64 GB unified memory (mac_runs.json meta). The reveal is "
            "illustrative: pixel-space noise (VIOLET) mixed into the finished picture with q_sample and the course's DDPM "
            "schedule, ᾱ from 0 to 1 — the run itself denoised a 64×64×4 latent (Episode 17). The clock counts the measured "
            "7.32 s, faster than real time."))
        reel = Reel(HOOK_FRAMES, HOOK_H)
        pic_f = framed(reel.image, DATA).move_to([-3.45, 0.55, 0])
        gen = cite(GEN_CITE).move_to([0, -3.32, 0])
        assert gen.width < 12.8
        clock_ic = icon(recolored("clock", TIME, BLUE_FILL, PREFIX), 0.8)
        num = title_text("0.00 s", 66, TIME)
        cx = 0.75

        def clock_text(v):
            return title_text(f"{v:.2f} s", 66, TIME).move_to([cx, 1.5, 0], aligned_edge=m.LEFT)

        num = clock_text(0)
        clock_ic.next_to(num, m.LEFT, buff=0.3)
        clock_ic.set_x(cx - 0.75)
        num_sub = text("one picture, start to finish", 22, MUTED).next_to(num, m.DOWN, buff=0.12).align_to(num, m.LEFT)
        mem = title_text(f"{PEAK:.2f} GB", 66, MEMORY).move_to([cx, 0.05, 0], aligned_edge=m.LEFT)
        sit(mem, baseline(num) - 1.45)
        mem_sub = text("memory at the peak", 22, MUTED).next_to(mem, m.DOWN, buff=0.12).align_to(mem, m.LEFT)
        mac = m.VGroup(text("on a Mac Studio with an Apple M4 Max", 22, INK), code("64 GB of unified memory", 22, INK))
        mac.arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT)
        mac.next_to(mem_sub, m.DOWN, buff=0.35).align_to(mem, m.LEFT)
        assert mac.get_right()[0] < 6.5 and mac.get_bottom()[1] > -1.8, (mac.get_right(), mac.get_bottom())
        agenda = m.VGroup(pill("where the seconds go", TIME, BLUE_FILL, 22), pill("what has to fit", MEMORY, LIME_FILL, 22),
                          pill("the small digit model", DATA, TEAL_FILL, 22)).arrange(m.RIGHT, buff=0.35)
        agenda.move_to([-0.45, -2.35, 0])
        mascot = pixel(0.45).next_to(agenda, m.RIGHT, buff=0.45)
        assert mascot.get_right()[0] < 6.5 and agenda.get_left()[0] > -6.5
        assert agenda.get_top()[1] < min(pic_f.get_bottom()[1], mac.get_bottom()[1]) - 0.15 and agenda.get_bottom()[1] > gen.get_top()[1] + 0.15
        dummy = m.VMobject()

        def tick(v):
            num.become(clock_text(v))

        self.say(
            "[Our lighthouse, generated on this Mac], [took seven point three seconds].",
            m.AnimationGroup(self.new_header("00 · hook", "Seven seconds, eight gigabytes"),
                             m.FadeIn(pic_f, shift=0.15 * m.UP), m.FadeIn(gen), m.FadeIn(clock_ic), m.FadeIn(num)),
            m.AnimationGroup(drive(reel.image, 0, len(HOOK_FRAMES) - 1, [reel.show], run_time=3.0, rate=m.smooth),
                             drive(dummy, 0, TOT_S, [tick], run_time=3.0), m.FadeIn(num_sub, shift=0.1 * m.UP)),
        )
        self.remove(dummy)
        self.say(
            "[At its peak, it used just over eight gigabytes of memory], [on a Mac Studio with sixty-four].",
            m.AnimationGroup(m.FadeIn(mem, shift=0.15 * m.UP), m.FadeIn(mem_sub)),
            m.FadeIn(mac, shift=0.1 * m.UP),
        )
        self.say(
            "[Today: where those seconds go], [what has to fit in memory], [and how our small digit model compares].",
            m.AnimationGroup(m.FadeIn(agenda[0], shift=0.1 * m.UP), m.FadeIn(mascot, shift=0.2 * m.UP)),
            m.FadeIn(agenda[1], shift=0.1 * m.UP),
            m.FadeIn(agenda[2], shift=0.1 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 1 · one generation, on a clock

    def timeline(self):
        self.next_section("1 · One generation", notes=(
            "The timing run (mac_runs §5): SD 2.1 base, 512×512, 50 steps, guidance 7.5, median of 3 runs (7.35 / 7.32 / "
            f"7.31 s). Text encoder {TXT_S:.4f} s for 13 tokens (prompt + empty negative; the MLX pipeline pads only to the "
            "longest prompt — start + 11 + end — where diffusers pads to 77; mac_runs erratum 2026-10-27). Denoiser: 50 steps × "
            f"2 passes batched, {DEN_S:.3f} s ({SHARE['den']:.1f}%); the tick marks are the run's own 50 step times (they "
            f"alternate about 0.128 / 0.159 s). Decoder (VAE, float32) {DEC_S:.3f} s ({SHARE['dec']:.1f}%). The bar is played "
            "in real time (7.32 s), to scale. One step: guided 0.141 s (two passes in one batch), unguided 0.083 s (guidance "
            f"1, one pass; 50 steps {T1['total_s']:.2f} s): 1.7×, not 2× — Episode 14. Dashed: two passes one after the other "
            f"(2 × 0.083 = {2 * STEP_ONE:.3f} s). Illustrative: the latent tile in the denoiser card (the 30-step run's latent, "
            "channel 1, at the matching fraction of the run; VIOLET → TEAL by 1/(1 + σ²)) and the picture's cross-fade "
            "during the decode. Caveat (mac_runs §8): the Mac was shared, ±5–10% jitter; the decode varied 0.17–0.27 s."))
        cy, ch = 1.85, 1.1
        txt_card = stage("text encoder", "the prompt, once", PROMPT, 2.45, ch, [-4.85, cy, 0])
        den_box = card(3.5, ch, MODEL, AMBER_FILL, radius=0.16, stroke=2.5).move_to([-1.15, cy, 0])
        tile = Reel(LAT_FRAMES, 0.82)
        tile_f = framed(tile.image, EDGE, buff=0.03, stroke=1.5).move_to(den_box.get_left() + [0.6, 0, 0])

        def counter_text(k):
            return code(f"step {k:>2}/{STEPS}", 22, TIME, weight=m.BOLD).move_to([-0.6, cy - 0.22, 0])

        den_name = text("denoiser", 24, INK, weight=m.BOLD).move_to([-0.6, cy + 0.2, 0])
        counter = counter_text(0)
        den_card = m.Group(den_box, tile_f, den_name, counter)
        dec_card = stage("decoder", "once, at the end", DATA, 2.3, ch, [2.45, cy, 0])
        pic = Reel(DECODE_FRAMES, ch + 0.15)
        pic_f = framed(pic.image, DATA).move_to([4.65 + 0.55, cy, 0])
        arrows = m.VGroup(arrow(txt_card.get_right(), den_box.get_left(), PROMPT, buff=0.1),
                          arrow(den_box.get_right(), dec_card.get_left(), MODEL, buff=0.1),
                          arrow(dec_card.get_right(), pic_f.get_left(), DATA, buff=0.1))
        assert pic_f.get_right()[0] < 6.5 and txt_card.get_left()[0] > -6.5
        assert all(a.get_length() > 0.45 for a in arrows), [a.get_length() for a in arrows]

        # the clock: 0 … 7.5 s over 12 units
        left, span, secs = -6.0, 12.0, 7.5
        X = lambda t: left + span * t / secs  # noqa: E731
        by, bh = -0.7, 0.5
        axis = m.Line([left - 0.1, by - bh / 2, 0], [left + span + 0.1, by - bh / 2, 0]).set_stroke(TIME, 3)
        marks = m.VGroup()
        for s in range(8):
            p = np.array([X(s), by - bh / 2, 0])
            marks.add(m.Line(p, p + [0, -0.1, 0]).set_stroke(TIME, 2), code(f"{s} s", 20, MUTED).next_to(p, m.DOWN, buff=0.16))
        t_end = TXT_S + DEN_S + DEC_S
        assert abs(X(t_end) - (left + span * TOT_S / secs)) < 0.01 and X(t_end) < 5.8

        def seg(t0, t1, color, solid=False):
            w = span * (t1 - t0) / secs
            r = m.Rectangle(width=max(w, 0.002), height=bh).move_to([X(t0) + max(w, 0.002) / 2, by, 0])
            if w <= 0.001:
                return r.set_stroke(color, 0, opacity=0).set_fill(color, 0)
            return r.set_stroke(color, 0 if solid else 2).set_fill(color if solid else FILLS[color], 1)

        seg_txt, seg_den, seg_dec = seg(0, 0, PROMPT, True), seg(TXT_S, TXT_S, MODEL), seg(TXT_S + DEN_S, TXT_S + DEN_S, DATA, True)
        ticks = m.VGroup(*[m.Line([X(t), by - bh / 2 + 0.07, 0], [X(t), by + bh / 2 - 0.07, 0]).set_stroke(MODEL, 1.4, opacity=0)
                           for t in STEP_ENDS[:-1]])

        def head_at(t):
            x = X(t)
            line = m.Line([x, by - bh / 2 - 0.05, 0], [x, by + bh / 2 + 0.08, 0]).set_stroke(TIME, 3)
            words = code(f"{t:.2f} s", 24, TIME, weight=m.BOLD).next_to([x, by + bh / 2 + 0.08, 0], m.UP, buff=0.1)
            return m.VGroup(line, words)

        head = head_at(0)
        base_fill = {id(txt_card[0]): PINK_FILL, id(den_box): AMBER_FILL, id(dec_card[0]): TEAL_FILL}

        def light(box, color, on):
            box.set_stroke(color, 5 if on else 2.5)
            box.set_fill(mix(base_fill[id(box)], color, 0.22) if on else base_fill[id(box)], 1)

        def run(t):
            seg_txt.become(seg(0, min(t, TXT_S), PROMPT, True))
            seg_den.become(seg(TXT_S, float(np.clip(t, TXT_S, TXT_S + DEN_S)), MODEL))
            seg_dec.become(seg(TXT_S + DEN_S, float(np.clip(t, TXT_S + DEN_S, t_end)), DATA, True))
            for tk, te in zip(ticks, STEP_ENDS[:-1]):
                tk.set_stroke(MODEL, 1.4, opacity=0.7 if t >= te else 0)
            head.become(head_at(t))
            k = 0 if t <= TXT_S else min(STEPS, int(np.searchsorted(STEP_ENDS, t, side="right")) + 1)
            counter.become(counter_text(k))
            tile.show(round(30 * k / STEPS))
            pic.show(0 if t <= TXT_S + DEN_S else 9 * (t - TXT_S - DEN_S) / DEC_S)
            light(txt_card[0], PROMPT, t < TXT_S + 0.01 and t > 0)
            light(den_box, MODEL, TXT_S <= t < TXT_S + DEN_S)
            light(dec_card[0], DATA, TXT_S + DEN_S <= t < t_end - 1e-6)

        # labels under the bar
        lab_den = code(f"denoiser · {STEPS} steps × {PASSES} passes · {DEN_S:.2f} s · {SHARE['den']:.1f}%", 20, MODEL)
        lab_den.move_to([X(TXT_S + DEN_S / 2), -1.82, 0])
        lab_txt = code(f"text encoder · 13 tokens · {TXT_S:.3f} s · {SHARE['txt']:.1f}%", 20, PROMPT)
        lab_txt.move_to([0, -2.3, 0]).align_to([left, 0, 0], m.LEFT)
        lab_dec = code(f"decoder · {DEC_S:.3f} s · {SHARE['dec']:.1f}%", 20, DATA)
        lab_dec.move_to([0, -2.3, 0]).align_to([X(t_end), 0, 0], m.RIGHT)
        sit(lab_dec, baseline(lab_txt))
        assert lab_txt.get_right()[0] < lab_dec.get_left()[0] - 0.4 and lab_den.get_left()[0] > -6.5
        sliver_box = m.Rectangle(width=0.32, height=bh + 0.12).set_stroke(PROMPT, 2.5).set_fill(opacity=0).move_to([X(0) + 0.02, by, 0])
        cap = cite("measured on this Mac · SD 2.1 base · 50 steps · guidance 7.5 · median of 3 runs: 7.35, 7.32, 7.31 s")
        cap.move_to([0, -3.32, 0])
        assert cap.width < 12.6 and lab_txt.get_bottom()[1] > cap.get_top()[1] + 0.3, (cap.width, lab_txt.get_bottom())

        # the prompt's tokens (13: start + 11 + end)
        toks = token_row(TOKENS, PROMPT, 20, max_width=13.0)
        for i in (0, len(TOKENS) - 1):
            toks[i][1].set_color(MUTED)
        comma = TOKENS.index(",")
        toks[comma][1].become(code(",", 34, INK).move_to(toks[comma][0]).shift(0.07 * m.DOWN))   # a bigger comma (≥ 20 px)
        toks.move_to([0, 0.6, 0])
        assert toks.width < 12.7 and abs(toks[0].get_y() - toks[-1].get_y()) < 0.01   # one row
        leader = m.DashedLine([X(0) + 0.02, by + bh / 2 + 0.05, 0], [X(0) + 0.02, toks.get_bottom()[1] - 0.05, 0],
                              dash_length=0.06).set_stroke(PROMPT, 2)
        assert toks.get_top()[1] < cy - ch / 2 - 0.15, toks.get_top()

        # one step, zoomed (Episode 14)
        S = 25.0                                                                       # units per second in the inset
        bx = -3.55
        g_y, u_y, lane = 0.76, -0.04, 0.34
        lanes = m.VGroup(hbar(bx, g_y + lane / 2, S * STEP_CFG, lane, MODEL), hbar(bx, g_y - lane / 2, S * STEP_CFG, lane, MODEL))
        lane_w = m.VGroup(code("with the prompt", 20, INK).move_to(lanes[0]), code("empty prompt", 20, INK).move_to(lanes[1]))
        assert lane_w[0].width < lanes[0].width - 0.15 and lane_w[0].height < lane - 0.04, (lane_w[0].width, lane_w[0].height)
        gy = g_y - lane - 0.13
        ghost = m.DashedLine([bx, gy, 0], [bx + 2 * S * STEP_ONE, gy, 0], dash_length=0.08).set_stroke(MUTED, 2)
        ghost_end = m.Line([bx + 2 * S * STEP_ONE, gy - 0.08, 0], [bx + 2 * S * STEP_ONE, gy + 0.08, 0]).set_stroke(MUTED, 2)
        g_val = code(f"{STEP_CFG:.3f} s", 22, INK, weight=m.BOLD).next_to(lanes, m.RIGHT, buff=0.2)
        g_lab = text("one guided step", 20, INK).move_to([0, g_y, 0]).align_to([left, 0, 0], m.LEFT)
        one = hbar(bx, u_y, S * STEP_ONE, lane, MODEL)
        one_w = code("one pass", 20, INK).move_to(one)
        u_val = code(f"{STEP_ONE:.3f} s", 22, INK, weight=m.BOLD).next_to(one, m.RIGHT, buff=0.2)
        u_lab = text("unguided step", 20, INK).move_to([0, u_y, 0]).align_to([left, 0, 0], m.LEFT)
        sit(u_lab, baseline(u_val))
        sit(g_lab, baseline(g_val))
        gh_lab = cite(f"in a row: {2 * STEP_ONE:.3f} s").next_to(ghost_end, m.RIGHT, buff=0.15)
        ratio = m.VGroup(pill("1.7×, not 2×", GOOD, GOOD_FILL, 22), cite("as in Episode 14")).arrange(m.DOWN, buff=0.12)
        ratio.move_to([0, 0.55, 0]).align_to([6.4, 0, 0], m.RIGHT)
        assert g_lab.get_right()[0] < bx - 0.2 and u_lab.get_right()[0] < bx - 0.2
        assert g_val.get_right()[0] < ratio.get_left()[0] - 0.2 and gh_lab.get_right()[0] < ratio.get_left()[0] - 0.15
        assert lanes.get_top()[1] < cy - ch / 2 - 0.15 and one.get_top()[1] < gy - 0.1
        assert gh_lab.get_right()[0] < ratio.get_left()[0] - 0.2, (gh_lab.get_right(), ratio.get_left())
        k25 = 24
        step_box = m.Rectangle(width=span * STEP_TIMES[k25] / secs + 0.06, height=bh + 0.2).set_stroke(INK, 2.5).set_fill(opacity=0)
        step_box.move_to([(X(STEP_ENDS[k25 - 1]) + X(STEP_ENDS[k25])) / 2, by, 0])
        inset = m.VGroup(lanes, lane_w, g_val, g_lab, one, one_w, u_val, u_lab)

        dyn = m.VGroup(seg_txt, seg_den, seg_dec, ticks)
        dummy = m.VMobject()
        self.say(
            "[Here is that one generation, laid out on a clock]: "
            "[five hundred twelve pixels square, fifty steps, guidance seven point five].",
            m.AnimationGroup(self.new_header("01 · one generation", "Where the seconds go"), m.Create(axis), m.FadeIn(marks),
                             m.FadeIn(txt_card, shift=0.1 * m.UP), m.FadeIn(den_card, shift=0.1 * m.UP),
                             m.FadeIn(dec_card, shift=0.1 * m.UP), m.FadeIn(pic_f, shift=0.1 * m.UP),
                             m.LaggedStart(*[m.GrowArrow(a) for a in arrows], lag_ratio=0.3)),
            m.AnimationGroup(m.FadeIn(cap), m.FadeIn(head)),
        )
        self.add(dyn)
        self.say(
            "[Watch it in real time]: the prompt is read in a blink, the denoiser takes fifty steps, "
            "and at the very end, the decoder paints the picture.",
            drive(dummy, 0, t_end, [run], run_time=t_end),
        )
        run(t_end)
        self.remove(dummy)
        self.say(
            "[The text encoder read thirteen tokens in seventeen milliseconds]: [too thin to see on this bar].",
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(t, shift=0.1 * m.UP) for t in toks], lag_ratio=0.06),
                             m.Create(leader)),
            m.AnimationGroup(m.Create(sliver_box), m.FadeIn(lab_txt, shift=0.1 * m.UP)),
        )
        self.say(
            "[The denoiser ran fifty steps, two passes each]: [seven seconds, ninety-six percent of the whole].",
            m.AnimationGroup(m.FadeOut(sliver_box), m.FadeIn(lab_den, shift=0.1 * m.UP),
                             m.ShowPassingFlash(seg_den.copy().set_stroke(MODEL, 6).set_fill(opacity=0), time_width=0.5)),
            m.Circumscribe(seg_den, color=MODEL, buff=0.08),
        )
        self.say(
            "[The decoder ran once, for a quarter of a second]: [only then did the picture exist].",
            m.FadeIn(lab_dec, shift=0.1 * m.UP),
            m.AnimationGroup(m.Circumscribe(pic_f, color=DATA, buff=0.06), m.Circumscribe(seg_dec, color=DATA, buff=0.06)),
        )
        self.say(
            "[Zoom into one step]: [two passes, with and without the prompt, run as one batch]: "
            "[a hundred forty-one milliseconds].",
            m.AnimationGroup(m.FadeOut(toks), m.FadeOut(leader), m.Create(step_box)),
            m.AnimationGroup(m.GrowFromEdge(lanes, m.LEFT), m.FadeIn(g_lab)),
            m.AnimationGroup(m.FadeIn(lane_w), m.FadeIn(g_val, shift=0.1 * m.LEFT)),
        )
        self.say(
            "[A single pass takes eighty-three], [so guidance costs one point seven times, not two], "
            "[as episode fourteen showed].",
            m.AnimationGroup(m.GrowFromEdge(one, m.LEFT), m.FadeIn(u_lab), m.FadeIn(u_val),
                             m.Succession(m.Wait(0.6), m.FadeIn(one_w, run_time=0.4))),
            m.AnimationGroup(m.Create(ghost), m.Create(ghost_end), m.FadeIn(gh_lab)),
            m.FadeIn(ratio, shift=0.1 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 2 · steps × time per step

    def steps(self):
        self.next_section("2 · Steps", notes=(
            "The steps sweep (mac_runs §5): SD 2.1 base, guidance 7.5, seeds 0–3, total seconds per picture, median of 4 seeds: "
            + ", ".join(f"{s}: {SD_T[s]:.3f}" for s in SWEEP) + f". Least-squares line: {SLOPE:.4f} s per step, intercept "
            f"{INTER:.3f} s; measured fixed cost (total − denoise) median {np.median(FIXED):.3f} s across the sweep; the timing "
            f"run's text + decode was {TXT_S + DEC_S:.3f} s (the decode varied 0.17–0.27 s). The sweep's 50-step median "
            f"(6.90 s) is a little faster than the timing run's 7.32 s: run-to-run jitter on a shared machine (±5–10%). "
            "Thumbnails: seed 0 at each step count (a lighthouse is recognisable by 4 steps). SD-Turbo (SD 2.1 distilled by "
            f"adversarial diffusion distillation, the same 865.9 M-parameter denoiser; Episode 12): 1 step, no guidance, median "
            f"of 4 seeds {TB_T1:.3f} s; its timing run (median of 3) {TB['total_s']:.3f} s = text {TB_PARTS[0]:.3f} + denoiser "
            f"{TB_PARTS[1]:.3f} + decode {TB_PARTS[2]:.3f} s: the decoder is {100 * TB_DEC_SHARE:.0f}%."))
        plot = Plot((0, 50), (0, 8), width=6.2, height=3.7, center=(-2.85, -0.25, 0), x_ticks=[0, 10, 20, 30, 40, 50],
                    y_ticks=[0, 2, 4, 6, 8], x_label="denoising steps (guidance 7.5)", y_label="seconds for one picture")
        assert plot.get_top()[1] < 2.62 and plot.get_bottom()[1] > -3.0 and plot.get_left()[0] > -6.6, (plot.get_top(), plot.get_bottom())
        dots = m.VGroup(*[plot.dot(s, SD_T[s], TIME, 0.085) for s in SWEEP])
        fit = plot.line([0, 50], [INTER, INTER + 50 * SLOPE], MODEL, 4)
        p20, p40 = plot.coords(20, INTER + 20 * SLOPE), plot.coords(40, INTER + 40 * SLOPE)
        corner = np.array([p40[0], p20[1], 0])
        tri = m.VGroup(m.Line(p20, corner).set_stroke(MUTED, 2.5), m.Line(corner, p40).set_stroke(MUTED, 2.5))
        tri_h = code("20 steps", 20, MUTED).next_to(m.Line(p20, corner), m.DOWN, buff=0.12)
        tri_v = code(f"+{20 * SLOPE:.2f} s", 20, MUTED).next_to(m.Line(corner, p40), m.RIGHT, buff=0.12)
        slope_lab = code(f"{SLOPE * 1000:.0f} ms per step", 22, MODEL, weight=m.BOLD)
        slope_lab.move_to(plot.coords(5, 4.8), aligned_edge=m.LEFT)
        assert slope_lab.get_bottom()[1] > plot.coords(0, INTER + 27 * SLOPE)[1] + 0.15, slope_lab.get_bottom()
        i_dot = m.Dot(plot.coords(0, INTER), radius=0.09, color=DATA).set_z_index(4)
        i_lab = m.VGroup(code(f"{INTER:.2f} s at zero steps", 20, DATA, weight=m.BOLD), cite("text encoder + decoder, once"))
        i_lab.arrange(m.DOWN, buff=0.06, aligned_edge=m.LEFT).move_to(plot.coords(12, 0.95), aligned_edge=m.LEFT)
        i_lead = m.DashedLine(plot.coords(0.6, INTER + 0.05), i_lab.get_left() + [-0.08, 0, 0], dash_length=0.06).set_stroke(DATA, 2)
        assert i_lab.get_right()[0] < plot.coords(50, 0)[0] + 0.3 and tri_v.get_right()[0] < 1.3
        assert i_lab.get_bottom()[1] > plot.coords(0, 0)[1] + 0.1 and i_lab.get_top()[1] < plot.coords(14, INTER + 14 * SLOPE)[1]

        # thumbnails: what each step count gives (seed 0)
        col_x, row_y, ph = [2.35, 3.95, 5.55], [1.4, -0.75], 1.25
        thumbs = m.Group()
        for i, s in enumerate(SWEEP):
            x, y = col_x[i % 3], row_y[i // 3]
            img = framed(photo(load_rgb(sd_png(s)), ph), EDGE, buff=0.04, stroke=2).move_to([x, y, 0])
            l1 = code(f"{s} step{'s' if s > 1 else ''}", 20, INK).next_to(img, m.DOWN, buff=0.12)
            l2 = code(f"{SD_T[s]:.2f} s", 20, TIME).next_to(l1, m.DOWN, buff=0.08)
            thumbs.add(m.Group(img, l1, l2))
        th_t = cite("generated on this Mac · seed 0").move_to([col_x[1], 2.35, 0])
        assert thumbs.get_bottom()[1] > -3.0 and thumbs.get_right()[0] < 6.5 and thumbs.get_left()[0] > plot.get_right()[0] + 0.2

        # SD-Turbo, one step (Episode 12)
        tp = framed(photo(load_rgb(tb_png(1)), 1.75), GOOD).move_to([2.8, 1.3, 0])
        sp = framed(photo(load_rgb(sd_png(50)), 1.75), EDGE).move_to([5.2, 1.3, 0])
        tl = m.VGroup(code("SD-Turbo", 20, GOOD), code("1 step", 20, MUTED), code(f"{TB_T1:.2f} s", 22, INK, weight=m.BOLD))
        sl = m.VGroup(code("SD 2.1 base", 20, MUTED), code("50 steps", 20, MUTED), code(f"{SD_T[50]:.2f} s", 22, INK, weight=m.BOLD))
        tl.arrange(m.DOWN, buff=0.07)
        sl.arrange(m.DOWN, buff=0.07)
        tl.next_to(tp, m.DOWN, buff=0.14)
        sl.next_to(sp, m.DOWN, buff=0.14)
        x0b, wb = 1.75, 4.75
        kb = wb / TB["total_s"]
        tb_bar = m.VGroup()
        xx = x0b
        for v, c in zip(TB_PARTS, (PROMPT, MODEL, DATA)):
            tb_bar.add(hbar(xx, -1.68, v * kb, 0.5, c))
            xx += v * kb
        tb_head = text(f"the one-step picture, {TB['total_s']:.3f} s in all", 20, INK).next_to(tb_bar, m.UP, buff=0.14).align_to(tb_bar, m.LEFT)
        tb_den = code(f"{TB_PARTS[1] * 1000:.0f} ms", 20, INK).move_to(tb_bar[1])
        tb_dec = code(f"decoder {TB_PARTS[2] * 1000:.0f} ms · {100 * TB_DEC_SHARE:.0f}%", 20, INK).move_to(tb_bar[2])
        tb_cap = cite("timing run, median of 3").next_to(tb_bar, m.DOWN, buff=0.16).align_to(tb_bar, m.LEFT)
        assert tb_den.width < tb_bar[1].width - 0.08 and tb_dec.width < tb_bar[2].width - 0.15, (tb_den.width, tb_bar[1].width)
        assert tb_bar.get_right()[0] < 6.55
        assert tl.get_bottom()[1] > tb_head.get_top()[1] + 0.1 and tb_cap.get_bottom()[1] > -3.0
        cap = cite("measured on this Mac · median of 4 seeds · the Mac was shared: ±5–10% jitter · line: least squares")
        cap.move_to([0, -3.32, 0])
        assert cap.width < 12.8

        self.say(
            "[Now vary the number of steps]: [one, two, four, eight, twenty, and fifty].",
            m.AnimationGroup(self.new_header("02 · steps", "Steps times time per step"), m.FadeIn(plot), m.FadeIn(th_t)),
            m.LaggedStart(*[m.AnimationGroup(m.FadeIn(d, scale=0.5), m.FadeIn(t, shift=0.1 * m.UP)) for d, t in zip(dots, thumbs)],
                          lag_ratio=0.35),
        )
        self.say(
            "[The points fall on a straight line], [give or take the jitter of a shared machine].",
            m.Create(fit, run_time=1.4),
            m.FadeIn(cap),
        )
        self.say(
            "[Its slope is the price of one step]: [about a hundred thirty-four milliseconds].",
            m.AnimationGroup(m.Create(tri), m.FadeIn(tri_h), m.FadeIn(tri_v)),
            m.FadeIn(slope_lab, shift=0.1 * m.UP),
        )
        self.say(
            "[Where it meets zero steps, about a fifth of a second is left]: [the text encoder and the decoder, paid once].",
            m.AnimationGroup(m.FadeIn(i_dot, scale=0.5), m.Flash(i_dot, color=DATA)),
            m.AnimationGroup(m.Create(i_lead), m.FadeIn(i_lab, shift=0.1 * m.UP)),
        )
        self.say(
            "[The turbo model from episode twelve needs one step]: [a quarter of a second for the whole picture].",
            m.AnimationGroup(m.FadeOut(thumbs), m.FadeOut(th_t), m.FadeIn(sp, shift=0.1 * m.UP), m.FadeIn(sl)),
            m.AnimationGroup(m.FadeIn(tp, shift=0.1 * m.UP), m.FadeIn(tl, shift=0.1 * m.UP)),
        )
        self.say(
            "[At one step, the decoder takes most of the time]: "
            "[two hundred seventy milliseconds, against seventy-six for the denoiser].",
            m.AnimationGroup(m.FadeIn(tb_head), *[m.GrowFromEdge(b, m.LEFT) for b in tb_bar], m.FadeIn(tb_cap)),
            m.AnimationGroup(m.FadeIn(tb_dec), m.FadeIn(tb_den), m.Circumscribe(tb_bar[2], color=DATA, buff=0.06)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 3 · memory

    def memory(self):
        self.next_section("3 · Memory", notes=(
            "Sizes from mac_runs §5 / mac_runs.json lighthouse.models (counted on this Mac): denoiser 865,910,724 parameters, "
            "float16; OpenCLIP-H text encoder 340,387,840, float16; VAE 83,653,863 (encoder 34.2 M + decoder 49.5 M), "
            "float32 (the pipeline defaults). Bytes = parameters × bytes per number: 1.732 + 0.681 + 0.335 = 2.747 GB. "
            f"Downloads (pinned revisions, lighthouse_downloads): {DL_ONE:.3f} GB per model (the same files: fp16 denoiser and "
            f"text encoder, fp32 VAE, configs, tokenizer), {DL_BOTH:.2f} GB for SD 2.1 base + SD-Turbo. Peak memory (MLX): "
            f"{PEAK:.2f} GB for one 50-step guided picture with SD 2.1 loaded; {PEAK_BOTH:.2f} GB in the turbo run with both "
            "models loaded. The rest above the weights is working memory for the computation (activations, temporaries). "
            "Unified memory: CPU, GPU and Neural Engine share one pool (notes §23), so the GPU reads the weights where they lie. "
            "FLUX.1 (Black Forest Labs, Aug 2024), full pipeline in 16 bits: 33.73 GB (notes §23: ≈ 16.87 B parameters × 2 bytes; "
            "= the mflux README's 'Original (16 bit) 33.73GB', Jan 2026) — published, not run here. Bars to scale (64 GB = full)."))
        formula = code("bytes = parameters × bytes per number", 26, INK, t2c={r"^bytes": MEMORY})
        formula.move_to([0, 2.2, 0]).align_to([-6.2, 0, 0], m.LEFT)
        rows, bars = [], []
        ys = [1.5, 0.98, 0.46]
        for (k, name, col, dw), y in zip(NETS, ys):
            sw = m.Square(0.24).set_stroke(col, 2).set_fill(FILLS[col], 1).move_to([-6.05, y, 0])
            nm = text(name, 22, INK).move_to([0, y, 0]).align_to([-5.75, 0, 0], m.LEFT)
            pm = code(f"{P_M[k]:.1f} M", 22, INK).move_to([0, y, 0]).align_to([-1.45, 0, 0], m.RIGHT)
            mul = code(f"× {BPN[PARTS[k]['dtype']]} bytes", 22, INK).move_to([0, y, 0]).align_to([-1.15, 0, 0], m.LEFT)
            dt = code(dw, 20, MUTED).move_to([0, y, 0]).align_to([0.85, 0, 0], m.LEFT)
            eq = code(f"= {SIZE[k]:.2f} GB", 22, col, weight=m.BOLD).move_to([0, y, 0]).align_to([3.95, 0, 0], m.RIGHT)
            for w in (pm, mul, dt, eq):
                sit(w, baseline(nm))
            bar = hbar(4.25, y, SIZE[k] * 1.1, 0.3, col)
            rows.append(m.VGroup(sw, nm, pm, mul, dt, eq))
            bars.append(bar)
        assert all(r[3].get_right()[0] < r[4].get_left()[0] - 0.2 for r in rows) and max(b.get_right()[0] for b in bars) < 6.4
        rule = m.Line([-6.2, 0.12, 0], [6.2, 0.12, 0]).set_stroke(EDGE, 2)
        tot = code(f"= {WEIGHTS:.2f} GB", 24, INK, weight=m.BOLD).move_to([0, -0.22, 0]).align_to([3.95, 0, 0], m.RIGHT)
        tot_l = text("the weights of one model", 22, INK, weight=m.BOLD).move_to([0, -0.22, 0]).align_to([-5.75, 0, 0], m.LEFT)
        sit(tot, baseline(tot_l))
        dl = cite(f"the same on disk: {DL_ONE:.2f} GB per model, {DL_BOTH:.2f} GB downloaded for both")
        dl.next_to(tot_l, m.DOWN, buff=0.12).align_to(tot_l, m.LEFT)
        weights_block = m.VGroup(tot_l, tot)

        # the 64 GB pool
        trk_l, trk_w = -1.35, 5.9
        pool = text(f"this Mac: {RAM_GB} GB of unified memory", 22, INK, weight=m.BOLD).move_to([0, -1.15, 0]).align_to([-6.2, 0, 0], m.LEFT)
        one_pool = pill("one pool: the GPU reads it directly", MEMORY, LIME_FILL, 20).move_to([0, -1.05, 0])
        mrows = []
        for label, v, col, y, pub in (("peak, making one picture", PEAK, MEMORY, -1.66, False),
                                      ("both models loaded", PEAK_BOTH, MEMORY, -2.12, False),
                                      ("FLUX.1 weights, 16-bit", FLUX_GB, MUTED, -2.58, True)):
            lab = text(label, 20, INK if not pub else MUTED).move_to([0, y, 0]).align_to([-6.2, 0, 0], m.LEFT)
            mt = Meter(trk_w, v / RAM_GB, col, height=0.24).move_to([trk_l + trk_w / 2, y, 0])
            val = code(f"{v:.2f} GB" if v < 20 else f"{v:.1f} GB", 20, col if not pub else INK).next_to(mt, m.RIGHT, buff=0.18)
            sit(val, baseline(lab))
            mrows.append((lab, mt, val, v))
        assert all(r[0].get_right()[0] < trk_l - 0.2 and r[2].get_right()[0] < 6.55 for r in mrows)
        pub_c = cite("published, not ours").next_to(mrows[2][0], m.DOWN, buff=0.05).align_to(mrows[2][0], m.LEFT)
        one_pool.move_to([0, -1.15, 0]).align_to([6.45, 0, 0], m.RIGHT)
        assert one_pool.get_left()[0] > pool.get_right()[0] + 0.2, (one_pool.get_left(), pool.get_right())
        cap = cite("measured on this Mac · MLX peak memory, 512 × 512 · FLUX.1: notes §23, mflux README, Jan 2026")
        cap.move_to([0, -3.34, 0])
        assert pub_c.get_bottom()[1] > cap.get_top()[1] + 0.08 and cap.width < 12.6, (pub_c.get_bottom(), cap.width)

        self.say(
            "[What has to fit in memory]? [First, the weights of all three networks].",
            m.AnimationGroup(self.new_header("03 · memory", "What has to fit"), m.FadeIn(formula, shift=0.1 * m.UP)),
            m.LaggedStart(*[m.FadeIn(m.VGroup(r[0], r[1], r[2]), shift=0.1 * m.RIGHT) for r in rows], lag_ratio=0.25),
        )
        self.say(
            "[Bytes are parameters times bytes per number]: [the denoiser's eight hundred sixty-six million weights, "
            "at two bytes each, make one point seven gigabytes].",
            m.Circumscribe(formula, color=MEMORY, buff=0.08),
            m.AnimationGroup(m.FadeIn(rows[0][3]), m.FadeIn(rows[0][4]), m.FadeIn(rows[0][5], shift=0.1 * m.LEFT),
                             m.GrowFromEdge(bars[0], m.LEFT)),
        )
        self.say(
            "[The text encoder adds zero point seven], "
            "[and the autoencoder, kept at four bytes per number, a third of a gigabyte].",
            m.AnimationGroup(m.FadeIn(rows[1][3]), m.FadeIn(rows[1][4]), m.FadeIn(rows[1][5], shift=0.1 * m.LEFT),
                             m.GrowFromEdge(bars[1], m.LEFT)),
            m.AnimationGroup(m.FadeIn(rows[2][3]), m.FadeIn(rows[2][4]), m.FadeIn(rows[2][5], shift=0.1 * m.LEFT),
                             m.GrowFromEdge(bars[2], m.LEFT)),
        )
        self.say(
            "[In all, two point seven five gigabytes, the same on disk as in memory]: "
            "[we downloaded five and a half for two models].",
            m.AnimationGroup(m.Create(rule), m.FadeIn(weights_block, shift=0.1 * m.UP)),
            m.FadeIn(dl, shift=0.1 * m.UP),
        )
        self.say(
            "[While it ran, memory peaked at eight point one gigabytes], "
            "[and at ten point nine with the turbo model loaded too].",
            m.AnimationGroup(m.FadeIn(pool), m.FadeIn(mrows[0][0]), m.FadeIn(mrows[0][1].track),
                             m.GrowFromEdge(mrows[0][1].bar, m.LEFT), m.FadeIn(mrows[0][2]), m.FadeIn(cap)),
            m.AnimationGroup(m.FadeIn(mrows[1][0]), m.FadeIn(mrows[1][1].track), m.GrowFromEdge(mrows[1][1].bar, m.LEFT),
                             m.FadeIn(mrows[1][2])),
        )
        self.say(
            "[This Mac has sixty-four], [in one pool that the processor and the GPU share: nothing is copied between them].",
            m.Circumscribe(pool, color=MEMORY, buff=0.08),
            m.FadeIn(one_pool),
        )
        self.say(
            "[Even FLUX one, at sixteen bits, would fit]: [its weights take thirty-four gigabytes].",
            m.AnimationGroup(m.FadeIn(mrows[2][0]), m.FadeIn(mrows[2][1].track), m.FadeIn(pub_c)),
            m.AnimationGroup(m.GrowFromEdge(mrows[2][1].bar, m.LEFT, run_time=1.2), m.FadeIn(mrows[2][2])),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 4 · the speed limit

    def speed_limit(self):
        self.next_section("4 · Speed limit", notes=(
            "Sister course 'Tokens per Second', Episode 23 'Your Mac as an Inference Server': for language models, decoding "
            "one token reads every weight once, so memory bandwidth ÷ bytes of weights sets the speed. Here (DERIVED): the "
            f"denoiser's fp16 weights are {PARTS['unet']['bytes_in_memory'] / 1e9:.3f} GB; at the M4 Max's 546 GB/s (Apple's "
            f"figure, as in Tokens per Second notes §2) one read takes {READ_MS:.2f} ms. One unguided pass takes {PASS_MS:.1f} "
            f"ms (measured, mean of 50 steps); a batch of two (guided) {PAIR_MS:.1f} ms = {STEP_CFG / STEP_ONE:.2f}×. If reading "
            "the weights set the pace, a pass would be near 3 ms and a batch of two would cost about the same as one. Neither "
            "is true: the time is spent on arithmetic over the 64×64 latent (convolutions and attention over 4,096 "
            "positions), so a diffusion step is closer to compute-bound. Levers: fewer steps (Episodes 9–12), fewer passes "
            "(no guidance or guidance-distilled models, Episode 14), a smaller latent (Episode 17). Bars to scale (0–150 ms)."))
        sis = card(12.4, 1.15, EDGE, CARD, radius=0.16, stroke=2).move_to([0, 1.85, 0])
        sis_t = m.VGroup(text("Tokens per Second · Episode 23", 22, INK, weight=m.BOLD),
                         text("our sister course · Your Mac as an Inference Server", 20, MUTED)).arrange(m.DOWN, buff=0.06, aligned_edge=m.LEFT)
        sis_t.move_to(sis).align_to(sis.get_left() + [0.35, 0, 0], m.LEFT)
        sis_r = m.VGroup(text("language models:", 20, MUTED), text("reading the weights sets the speed", 22, INK, weight=m.BOLD))
        sis_r.arrange(m.DOWN, buff=0.06, aligned_edge=m.LEFT).move_to(sis).align_to(sis.get_right() - [0.35, 0, 0], m.RIGHT)
        assert sis_r.get_left()[0] > sis_t.get_right()[0] + 0.3, (sis_r.get_left(), sis_t.get_right())
        x0, k = -6.0, 10.4 / 150                                                     # 150 ms over 10.4 units
        specs = (("reading the denoiser's weights once: 1.73 GB ÷ 546 GB/s", READ_MS, MEMORY, 0.62),
                 (f"one denoiser pass, measured: {PASS_MS / READ_MS:.0f}× longer", PASS_MS, MODEL, -0.28),
                 (f"a batch of two passes, measured: {STEP_CFG / STEP_ONE:.1f}× one pass", PAIR_MS, MODEL, -1.18))
        brs = []
        for label, v, col, y in specs:
            lab = text(label, 20, INK).move_to([0, y + 0.36, 0]).align_to([x0, 0, 0], m.LEFT)
            b = hbar(x0, y, v * k, 0.34, col)
            val = code(f"{v:.1f} ms" if v < 10 else f"{v:.0f} ms", 22, col, weight=m.BOLD).next_to(b, m.RIGHT, buff=0.18)
            brs.append((lab, b, val))
        assert brs[2][2].get_right()[0] < 6.5 and brs[0][0].get_top()[1] < sis.get_bottom()[1] - 0.12
        verdict = pill("a diffusion step is mostly arithmetic, not memory", MODEL, AMBER_FILL, 22).move_to([0, -2.05, 0])
        levers = m.VGroup(pill("fewer steps · Ep 9–12", TIME, BLUE_FILL, 20), pill("fewer passes · Ep 14", PROMPT, PINK_FILL, 20),
                          pill("a smaller latent · Ep 17", DATA, TEAL_FILL, 20)).arrange(m.RIGHT, buff=0.35).move_to([0, -2.72, 0])
        cap = cite("546 GB/s: Apple's figure for the M4 Max · 3.2 ms is arithmetic · the passes are measured on this Mac")
        cap.move_to([0, -3.35, 0])
        assert cap.width < 12.6 and levers.width < 12.6, (cap.width, levers.width)
        assert verdict.get_top()[1] < brs[2][1].get_bottom()[1] - 0.15, (verdict.get_top(), brs[2][1].get_bottom())

        self.say(
            "[In our sister course, Tokens per Second, episode twenty-three], "
            "[reading the weights from memory set the speed of language models].",
            m.AnimationGroup(self.new_header("04 · speed limit", "Arithmetic, not memory"), m.FadeIn(sis),
                             m.FadeIn(sis_t, shift=0.1 * m.RIGHT)),
            m.FadeIn(sis_r, shift=0.1 * m.LEFT),
        )
        self.say(
            "[Here, reading the denoiser's weights once would take about three milliseconds]. [One pass takes eighty-three].",
            m.AnimationGroup(m.FadeIn(brs[0][0]), m.GrowFromEdge(brs[0][1], m.LEFT), m.FadeIn(brs[0][2]), m.FadeIn(cap)),
            m.AnimationGroup(m.FadeIn(brs[1][0]), m.GrowFromEdge(brs[1][1], m.LEFT, run_time=1.2), m.FadeIn(brs[1][2])),
        )
        self.say(
            "[And a batch of two passes took one point seven times as long]: "
            "[the time grows with the work, so a diffusion step is mostly arithmetic].",
            m.AnimationGroup(m.FadeIn(brs[2][0]), m.GrowFromEdge(brs[2][1], m.LEFT, run_time=1.4), m.FadeIn(brs[2][2])),
            m.FadeIn(verdict, shift=0.1 * m.UP),
        )
        self.say(
            "[So the way to go faster is less arithmetic]: [fewer steps], [fewer passes], [or a smaller latent].",
            m.Circumscribe(verdict, color=MODEL, buff=0.08),
            m.FadeIn(levers[0], shift=0.1 * m.UP),
            m.FadeIn(levers[1], shift=0.1 * m.UP),
            m.FadeIn(levers[2], shift=0.1 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 5 · framework

    def framework(self):
        self.next_section("5 · Framework", notes=(
            "mac_runs §1: the same digit U-Net (ch 32, mults 1-2-4, 3,940,801 parameters in both) written in MLX 0.32.4 and "
            "PyTorch 2.14.1 on the MPS backend, timed back to back (gpu/bench_frameworks.py). Training at batch 256 (compiled "
            f"MLX step vs eager PyTorch): {TR[0]:,.1f} vs {TR[1]:,.1f} images/s = {TR[0] / TR[1]:.3f}×. Sampling at batch 1,000: "
            f"{INF[0]:,.1f} vs {INF[1]:,.1f} images/s = {INF[0] / INF[1]:.3f}× (about even). fp16/bf16 inference gave no speed-up "
            "for the digit model; GroupNorm costs about 25% of a forward pass. Larger configurations: MLX ahead in training "
            "(1.13–1.28×); sampling mixed (ch 64, mults 1-2-2: PyTorch faster). MLX is also what the lighthouse pipeline uses "
            "(mlx-examples stable_diffusion, commit 796f5b5)."))
        tr = Bars(list(TR), ["MLX", "PyTorch"], top=1700, width=3.4, height=2.5, colors=[MEMORY, MUTED], gap=0.6,
                  fmt="{:,.0f}", size=20, center=(-3.3, 0.3))
        tr.numbers.become(m.VGroup(*[tr._number(i, int(v)) for i, v in enumerate(TR)]))
        inf = Bars(list(INF), ["MLX", "PyTorch"], top=6500, width=3.4, height=2.5, colors=[MEMORY, MUTED], gap=0.6,
                   fmt="{:,.0f}", size=20, center=(3.3, 0.3))
        t1 = m.VGroup(text("training", 24, INK, weight=m.BOLD), cite("images per second · batch 256")).arrange(m.DOWN, buff=0.06)
        t2 = m.VGroup(text("sampling", 24, INK, weight=m.BOLD), cite("images per second · batch 1,000")).arrange(m.DOWN, buff=0.06)
        t1.move_to([-3.3, 2.15, 0])
        t2.move_to([3.3, 2.15, 0])
        assert t1.get_bottom()[1] > tr.numbers.get_top()[1] + 0.1 and t2.get_bottom()[1] > inf.numbers.get_top()[1] + 0.1
        r1 = pill(f"MLX {TR[0] / TR[1]:.2f}× faster", GOOD, GOOD_FILL, 22).move_to([-3.3, -1.85, 0])
        r2 = pill(f"about even: {INF[0] / INF[1]:.2f}×", MUTED, CARD, 22).move_to([3.3, -1.85, 0])
        assert r1.get_top()[1] < tr.labels.get_bottom()[1] - 0.1
        n1 = m.VGroup(text("half precision:", 20, MUTED), text("no speed-up for this small model", 20, INK)).arrange(m.RIGHT, buff=0.15)
        n2 = m.VGroup(text("group normalization:", 20, MUTED), text("about a quarter of every pass", 20, INK)).arrange(m.RIGHT, buff=0.15)
        for n in (n1, n2):
            sit(n[1], baseline(n[0]))
        notes_row = m.VGroup(n1, n2).arrange(m.DOWN, buff=0.12).move_to([0, -2.6, 0])
        assert notes_row.width < 12.6 and notes_row.get_top()[1] < r1.get_bottom()[1] - 0.12, notes_row.get_top()
        cap = cite("measured on this Mac · one 3.94 M digit network · MLX 0.32.4, compiled · PyTorch 2.14.1, eager")
        cap.move_to([0, -3.32, 0])
        assert cap.width < 12.6, cap.width

        self.say(
            "[Which software should run it]? [We built the same digit network twice]: "
            "[in MLX, Apple's array framework, and in PyTorch].",
            m.AnimationGroup(self.new_header("05 · framework", "MLX or PyTorch?"), m.FadeIn(t1), m.FadeIn(t2)),
            m.AnimationGroup(m.FadeIn(tr.zero), m.FadeIn(inf.zero), m.FadeIn(cap)),
            m.AnimationGroup(m.FadeIn(tr.labels), m.FadeIn(inf.labels)),
        )
        self.say(
            "[For training, MLX was one point five five times faster]: "
            "[fourteen hundred fifty-five images a second, against nine hundred forty-two].",
            m.AnimationGroup(*[m.GrowFromEdge(b, m.DOWN) for b in tr.bars], m.FadeIn(r1, shift=0.1 * m.UP)),
            m.FadeIn(tr.numbers, shift=0.1 * m.UP),
        )
        self.say(
            "[For sampling, the two were about even].",
            m.AnimationGroup(*[m.GrowFromEdge(b, m.DOWN) for b in inf.bars], m.FadeIn(inf.numbers, shift=0.1 * m.UP),
                             m.FadeIn(r2, shift=0.1 * m.UP)),
        )
        self.say(
            "[Two more lessons]: [half precision gave this small model no speed-up], "
            "[and group normalization took about a quarter of every pass].",
            m.Circumscribe(m.VGroup(tr, inf), color=EDGE, buff=0.1),
            m.FadeIn(n1, shift=0.1 * m.UP),
            m.FadeIn(n2, shift=0.1 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 6 · the digit model, small scale

    def small_scale(self):
        self.next_section("6 · Small scale", notes=(
            "The digit model (mac_runs §3–4): class-conditional DDPM-style U-Net, 3,940,801 parameters, trained on this Mac in "
            f"{DM['wall_min']:.1f} min ({DM['wall_s']:.0f} s): 10,000 steps × 256 digits, peak {DM['peak_mem_gb']:.2f} GB. The "
            "digits: generated on this Mac, DDIM 50, w = 1, noise seed 1234, samples 10–19 (labels 0–9). Latency, one batch, "
            "DDIM 50 (latency table: median wall time, compile excluded): batch 1 "
            + ", ".join(f"{b}: {LAT_MS[b]:.1f} ms" for b in BATCHES[1:3]) + f", batch 1: {LAT_MS[1]:.2f} ms per step; batch "
            f"1,000 from the 10,000-sample run (mean of batches 2–10): {DS['s_per_1000']:.3f} s / 50 = {LAT_MS[1000]:.1f} ms per "
            f"step. Per digit: {ONE_DIGIT * 1e3:.0f} ms alone vs {PER_DIGIT_1000:.2f} ms in a batch of 1,000 "
            f"({ONE_DIGIT * 1e3 / PER_DIGIT_1000:.0f}× less). Dashed line: cost proportional to the batch, through the "
            "1,000-digit point. mac_runs: 'At batch 1 the GPU is mostly idle'. Contrast: the lighthouse's batch of two passes "
            "already costs 1.7× one pass. The Mac was shared (±5–10%)."))
        row = m.Group(*[framed(picture(tint(d, DATA), 0.5), EDGE, buff=0.03, stroke=1.5) for d in DIGITS]).arrange(m.RIGHT, buff=0.07)
        row.move_to([0, 1.95, 0]).align_to([-6.2, 0, 0], m.LEFT)
        row_c = cite("generated on this Mac · DDIM, 50 steps · seed 1234").next_to(row, m.DOWN, buff=0.14).align_to(row, m.LEFT)
        s1 = m.VGroup(title_text("3.9 M", 44, MODEL), text("weights", 22, MUTED)).arrange(m.RIGHT, buff=0.2)
        s2 = m.VGroup(title_text(f"{DM['wall_min']:.1f} min", 44, TIME), text("training: 10,000 steps × 256", 22, MUTED)).arrange(m.RIGHT, buff=0.2)
        s3 = m.VGroup(title_text(f"{DM['peak_mem_gb']:.2f} GB", 44, MEMORY), text("at the peak", 22, MUTED)).arrange(m.RIGHT, buff=0.2)
        for s in (s1, s2, s3):
            sit(s[1], baseline(s[0]))
        stats = m.VGroup(s1, s2, s3).arrange(m.DOWN, buff=0.2, aligned_edge=m.LEFT).move_to([0, -0.08, 0]).align_to([-6.2, 0, 0], m.LEFT)
        assert stats.get_top()[1] < row_c.get_bottom()[1] - 0.15

        plot = Plot((1, 1000), (1, 1000), width=4.4, height=3.3, center=(3.55, 0.05, 0), x_ticks=[1, 10, 100, 1000],
                    y_ticks=[1, 10, 100, 1000], x_label="digits per batch", y_label="milliseconds per step",
                    log_x=True, log_y=True, x_fmt="{:,.0f}", y_fmt="{:,.0f}")
        assert plot.get_top()[1] < 2.62 and plot.get_left()[0] > stats.get_right()[0] + 0.2, (plot.get_left(), stats.get_right())
        pts = [plot.dot(b, LAT_MS[b], MODEL, 0.085) for b in BATCHES]
        curve = plot.line(BATCHES, [LAT_MS[b] for b in BATCHES], MODEL, 3.5)
        k1000 = LAT_MS[1000] / 1000
        prop = m.DashedLine(plot.coords(1 / k1000, 1), plot.coords(1000, LAT_MS[1000]), dash_length=0.08).set_stroke(MUTED, 2)
        legend = m.VGroup(cite("flat at first:"), cite("the GPU is waiting"),
                          m.VGroup(m.DashedLine([0, 0, 0], [0.45, 0, 0], dash_length=0.08).set_stroke(MUTED, 2),
                                   cite("cost ∝ batch")).arrange(m.RIGHT, buff=0.15))
        legend.arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT).move_to(plot.coords(1.35, 900), aligned_edge=m.UL)
        lab1 = code(f"{LAT_MS[1]:.1f} ms", 20, INK).next_to(pts[0], m.UR, buff=0.08)
        lab3 = code(f"{LAT_MS[100]:.1f}", 20, INK).next_to(pts[2], m.UL, buff=0.06)
        lab4 = code(f"{LAT_MS[1000]:.0f} ms", 20, INK).next_to(pts[3], m.LEFT, buff=0.15)
        assert legend.get_right()[0] < lab4.get_left()[0] - 0.1 or legend.get_bottom()[1] > lab4.get_top()[1] + 0.05
        per = m.VGroup(
            m.VGroup(text("one digit alone:", 20, MUTED), code(f"{ONE_DIGIT:.3f} s", 22, INK, weight=m.BOLD)).arrange(m.RIGHT, buff=0.15),
            m.VGroup(text("each, in a batch of 1,000:", 20, MUTED), code(f"{PER_DIGIT_1000:.1f} ms", 22, GOOD, weight=m.BOLD)).arrange(m.RIGHT, buff=0.15),
        ).arrange(m.DOWN, buff=0.14, aligned_edge=m.LEFT)
        for p in per:
            sit(p[1], baseline(p[0]))
        per.move_to([0, -2.05, 0]).align_to([-6.2, 0, 0], m.LEFT)
        lh = pill("the lighthouse: a batch of two already costs 1.7×", MODEL, AMBER_FILL, 20)
        lh.move_to([0, -2.86, 0]).align_to([6.45, 0, 0], m.RIGHT)
        assert lh.get_left()[0] > per.get_right()[0] + 0.25, (lh.get_left(), per.get_right())
        assert per.get_top()[1] < stats.get_bottom()[1] - 0.15 and plot.get_bottom()[1] > lh.get_top()[1] + 0.1, (plot.get_bottom(), lh.get_top())
        cap = cite("measured on this Mac · DDIM, 50 steps · one batch, median, compile excluded · the Mac was shared").move_to([0, -3.32, 0])

        self.say(
            "[The digit model is the small counterpart]: [three point nine million weights], "
            "[trained here in thirty-two minutes, at a peak of six point three gigabytes].",
            m.AnimationGroup(self.new_header("06 · small scale", "The digit model, by comparison"),
                             m.LaggedStart(*[m.FadeIn(d, shift=0.1 * m.UP) for d in row], lag_ratio=0.08), m.FadeIn(row_c)),
            m.FadeIn(s1, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(s2, shift=0.1 * m.UP), m.FadeIn(s3, shift=0.1 * m.UP)),
        )
        self.say(
            "[Making one digit with fifty steps takes a fifth of a second]: [about four milliseconds per step].",
            m.AnimationGroup(m.FadeIn(plot), m.FadeIn(per[0], shift=0.1 * m.UP), m.FadeIn(cap)),
            m.AnimationGroup(m.FadeIn(pts[0], scale=0.5), m.FadeIn(lab1)),
        )
        self.say(
            "[Ten digits at once take barely longer], [a hundred take twenty-four milliseconds], "
            "[and a thousand, a hundred seventy-seven].",
            m.FadeIn(pts[1], scale=0.5),
            m.AnimationGroup(m.FadeIn(pts[2], scale=0.5), m.FadeIn(lab3)),
            m.AnimationGroup(m.FadeIn(pts[3], scale=0.5), m.FadeIn(lab4), m.Create(curve)),
        )
        self.say(
            "[At batch one, the GPU is mostly idle]: [in batches of a thousand, each digit costs just nine milliseconds].",
            m.AnimationGroup(m.FadeIn(legend), m.Create(prop)),
            m.FadeIn(per[1], shift=0.1 * m.UP),
        )
        self.say(
            "[The lighthouse is the opposite]: [one picture already keeps the GPU busy].",
            m.FadeIn(lh, shift=0.1 * m.UP),
            m.Circumscribe(lh, color=MODEL, buff=0.08),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 7 · published, not ours

    def published(self):
        self.next_section("7 · Published", notes=(
            "Published Apple-silicon numbers (notes §23, read October 2026; each with its own date and settings — not ours): "
            "(1) apple/ml-stable-diffusion README at its first commit (29 Nov 2022): CompVis SD v1.4, 50 steps, 512×512, 77 "
            "tokens, classifier-free guidance (batch 2 for the U-Net), Core ML, Mac Studio M1 Ultra (64-core GPU) 9 s; 'conducted "
            "by Apple … in November 2022'. (2) The same README's SDXL base 1.0 table (added July 2023, still current in Oct "
            "2026): 1024×1024, 20 steps, guidance, float16, Mac Studio M2 Ultra 20 s (1.11 it/s). (3) mflux README "
            "(community-reported table, versions Nov 2024 – Jan 2026): FLUX.1 [schnell], 2 steps, 1024×1024, not quantised, "
            "INCLUDING model loading: M4 Max (128 GB) ≈ 19 s. Ours: SD 2.1 base in MLX, 512×512, 50 steps, guidance 7.5, "
            f"{TOT_S:.2f} s, median of 3 (mac_runs §5). Apple's README: 'these numbers do not represent peak HW capability'. "
            "Models, sizes, step counts, precision and software differ, so the rows are not a ranking."))
        cols = (-6.15, -4.35, 1.35, 6.15)
        heads = m.VGroup(cite("when"), cite("who · model · settings"), cite("Mac"), cite("time"))
        for h, x in zip(heads[:3], cols[:3]):
            h.move_to([0, 1.88, 0]).align_to([x, 0, 0], m.LEFT)
        heads[3].move_to([0, 1.88, 0]).align_to([cols[3], 0, 0], m.RIGHT)
        for h in heads[1:]:
            sit(h, baseline(heads[0]))
        tag = pill("published, not ours · dated", MUTED, CARD, 20).move_to([0, 2.33, 0]).align_to([cols[3], 0, 0], m.RIGHT)
        assert tag.get_top()[1] < 2.62 and tag.get_bottom()[1] > heads.get_top()[1] + 0.02, (tag.get_bottom(), heads.get_top())

        def row(spec, y, ours=False):
            when, who, setting, mac, t, sub = spec
            c = LIME if ours else INK
            w1 = code(when, 20, MUTED).move_to([0, y + 0.17, 0]).align_to([cols[0], 0, 0], m.LEFT)
            w2 = text(who, 22, c, weight=m.BOLD).move_to([0, y + 0.17, 0]).align_to([cols[1], 0, 0], m.LEFT)
            w3 = code(setting, 20, MUTED).move_to([0, y - 0.2, 0]).align_to([cols[1], 0, 0], m.LEFT)
            w4 = code(mac, 20, INK).move_to([0, y + 0.17, 0]).align_to([cols[2], 0, 0], m.LEFT)
            w5 = title_text(t, 36, c).move_to([0, y + 0.1, 0]).align_to([cols[3], 0, 0], m.RIGHT)
            w6 = cite(sub).move_to([0, y - 0.24, 0]).align_to([cols[3], 0, 0], m.RIGHT)
            for w in (w1, w4):
                sit(w, baseline(w2))
            return m.VGroup(w1, w2, w3, w4, w5, w6)

        rows = [row(p, y) for p, y in zip(PUB, (1.25, 0.25, -0.75))]
        ours = row(OURS, -1.82, True)
        ours_box = card(12.75, 0.95, LIME, LIME_FILL, radius=0.14, stroke=2).move_to([0, -1.85, 0]).set_z_index(-1)
        rules = m.VGroup(*[m.Line([-6.3, y, 0], [6.3, y, 0]).set_stroke(EDGE, 1.5) for y in (1.68, 0.75, -0.25)])
        for r in rows + [ours]:
            assert r[2].get_right()[0] < cols[2] - 0.2 and r[3].get_right()[0] < r[4].get_left()[0] - 0.2, r[2].text
        warn = text("different models, sizes, steps and software: compare with care", 22, INK).move_to([0, -2.75, 0])
        cap = cite("Apple, ml-stable-diffusion README (Nov 2022, Jul 2023) · mflux README (2024–26) · read October 2026")
        cap.move_to([0, -3.32, 0])
        assert cap.width < 12.6, cap.width

        self.say(
            "[Others have published Mac timings too]: [these are theirs, not ours, each with its date].",
            m.AnimationGroup(self.new_header("07 · published", "Other Macs, other models"), m.FadeIn(heads), m.FadeIn(cap)),
            m.FadeIn(tag, shift=0.1 * m.UP),
        )
        self.say(
            "[Back in November twenty twenty-two, Apple's own release made a fifty-step picture in nine seconds], "
            "[on an M1 Ultra].",
            m.AnimationGroup(m.FadeIn(rows[0][:3], shift=0.1 * m.RIGHT), m.FadeIn(rows[0][4:], shift=0.1 * m.LEFT), m.Create(rules[0])),
            m.FadeIn(rows[0][3]),
        )
        self.say(
            "[By mid twenty twenty-three, the larger SDXL took twenty seconds on an M2 Ultra]; "
            "[a community port of FLUX reports about nineteen, loading included].",
            m.AnimationGroup(m.FadeIn(rows[1], shift=0.1 * m.RIGHT), m.Create(rules[1])),
            m.AnimationGroup(m.FadeIn(rows[2], shift=0.1 * m.RIGHT), m.Create(rules[2])),
        )
        self.say(
            "[Ours took seven point three], [but the models, sizes and settings all differ]: [compare them with care].",
            m.AnimationGroup(m.FadeIn(ours_box), m.FadeIn(ours, shift=0.1 * m.UP)),
            m.AnimationGroup(*[m.Circumscribe(r[2], color=MUTED, buff=0.05) for r in rows + [ours]]),
            m.FadeIn(warn, shift=0.1 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes=(
            "Steps × passes × the cost of one pass, plus a fixed cost (text encoder + decoder) paid once; memory = parameters × "
            "bytes per number, plus working space. On this Mac (mac_runs §5): the lighthouse in 7.32 s (50 guided steps) at a "
            f"peak of {PEAK:.2f} GB; SD-Turbo in {TB_T1:.2f} s (one step, median of 4 seeds). Next: Episode 24, the course "
            "revisited."))
        halo = glow(MEMORY, radius=4.0)
        words = m.VGroup(*[title_text(w, 64, c) for w, c in (("Steps", TIME), ("Passes", MODEL), ("Bytes", MEMORY))])
        subs = m.VGroup(*[text(s, 22, MUTED) for s in ("× the price of one step", "2 per step with guidance",
                                                       "2 per weight in fp16")])
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
        top = m.VGroup(words, dots, subs).move_to(1.45 * m.UP)
        assert top.width < 12.6 and all(a.get_right()[0] + 0.3 < b.get_left()[0] for a, b in zip(subs, subs[1:])), [x.width for x in subs]
        f1 = code("time   ≈ steps × passes × one pass + a fixed cost", 22, INK, t2c={r"steps": TIME, r"passes": MODEL})
        f2 = code("memory ≈ parameters × bytes + working space", 22, INK, t2c={r"bytes": MEMORY})
        forms = m.VGroup(f1, f2).arrange(m.DOWN, buff=0.22, aligned_edge=m.LEFT).move_to([0, -0.45, 0])
        mac = m.VGroup(text("on this Mac:", 22, MUTED), code(f"{TOT_S:.1f} s · {PEAK:.1f} GB", 24, INK, weight=m.BOLD),
                       text("· one step:", 22, MUTED), code(f"{TB_T1:.2f} s", 24, GOOD, weight=m.BOLD)).arrange(m.RIGHT, buff=0.2)
        for w in mac[1:]:
            sit(w, baseline(mac[0]))
        mac.move_to([0, -1.65, 0])
        mascot = pixel(0.42)
        nxt = pill("NEXT · EPISODE 24 · OUT OF THE NOISE, REVISITED", MEMORY, LIME_FILL, 20, mono=True)
        foot = m.VGroup(mascot, nxt).arrange(m.RIGHT, buff=0.3).move_to([0, -2.65, 0])
        assert forms.width < 12.6 and foot.width < 12.6 and mac.width < 12.6

        self.say(
            "[To sum up]: [steps], [passes], [bytes].",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.3 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(dots[0]), m.FadeIn(words[1], shift=0.3 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(dots[1]), m.FadeIn(words[2], shift=0.3 * m.UP), m.FadeIn(subs[2])),
        )
        self.say(
            "[Time is steps times passes times the cost of one pass], [plus a little paid once]. "
            "[Memory starts with parameters times bytes].",
            m.AnimationGroup(m.FadeIn(f1, shift=0.1 * m.UP), m.Circumscribe(words[0], color=TIME, buff=0.1),
                             m.Circumscribe(words[1], color=MODEL, buff=0.1)),
            m.Circumscribe(glyphs(f1, "a fixed cost"), color=DATA, buff=0.06),
            m.AnimationGroup(m.FadeIn(f2, shift=0.1 * m.UP), m.Circumscribe(words[2], color=MEMORY, buff=0.1)),
        )
        self.say(
            "[On this Mac, our lighthouse took seven seconds and eight gigabytes], [or a quarter of a second, in one step].",
            m.FadeIn(mac[:2], shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(mac[2:], shift=0.1 * m.UP), m.FadeIn(mascot, shift=0.2 * m.UP),
                             m.FadeIn(nxt, shift=0.2 * m.UP)),
        )
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, top, forms, mac, foot)], run_time=0.9)
