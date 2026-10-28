"""Out of the Noise · Episode 15 · Words to Pictures (Module 4 · Steering the Picture)

How a prompt steers the denoiser: a tokenizer cuts it into tokens, a frozen text encoder (trained contrastively on
image–caption pairs, CLIP-style) turns them into vectors, and the denoiser reads those vectors through
cross-attention, in many layers at every step. Running example: the lighthouse prompt, "a lighthouse on a rocky coast
at dusk, oil painting", word by word, and the lighthouse model run on this Mac (Stable Diffusion 2.1-base, MLX;
research/experiments/mac_runs.md §5). Episodes 13 (a label added to the time embedding) and 14 (classifier-free
guidance, the empty prompt) are linked, not re-taught.

Every number on screen is computed below (diffusion_reference, the run's json, the model's own config values copied
with their provenance) and asserted, or cited with its source. Nothing is downloaded or run: the token ids were
computed by the course coordinator with the model's tokenizer from the local cache (offline) and re-checked against
its vocab.json; the encoder and denoiser sizes come from the model's config files in the same cache.

Sources (research/notes.md §15, §8, §11, §17; Appendix A row 15, Appendix C row 15; read October 2026):
- Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (CLIP), arXiv:2103.00020
  (Feb 2021): 400 million (image, text) pairs; contrastive (symmetric InfoNCE); batch 32,768; BPE vocab 49,152 + specials.
- SD 1.x text encoder: CLIP ViT-L/14, 123,060,480 parameters, 77 × 768 (notes §15). SD 2.x: OpenCLIP ViT-H, context
  1024 (SDXL Table 1). Our model's text_encoder/config.json: 23 layers of width 1024; OpenCLIP's ViT-H-14 config has 24
  text layers (github.com/mlfoundations/open_clip, model_configs/ViT-H-14.json, read 2026-10-27) → the second-to-last
  ("penultimate") layer, as SDXL states for its own encoders.
- Saharia et al., Imagen, arXiv:2205.11487 (May 2022): frozen T5-XXL (4.6B); "increasing the size of the language model
  in Imagen boosts both sample fidelity and image-text alignment much more than increasing the size of the image
  diffusion model".
- Podell et al., SDXL, arXiv:2307.01952 (Jul 2023): CLIP ViT-L + OpenCLIP ViT-bigG, penultimate outputs concatenated
  (77 × 2048); "The text encoders have a total size of 817M parameters."
- Esser et al., SD3, arXiv:2403.03206 (Mar 2024): CLIP-L + OpenCLIP bigG + T5-v1.1-XXL → 154 × 4096 context; "4.7B
  parameter T5 text encoder".
- Hertz et al., Prompt-to-Prompt, arXiv:2208.01626 (Aug 2022): "the cross-attention layers are the key to controlling the
  relation between the spatial layout of the image to each word in the prompt."
- Rombach et al., LDM (2022) §3.3: cross-attention, Q from the image features, K and V from the text encoder.

Illustrative (labelled on screen): the encoder's vector values (shading only); the attention weights of one position
and the three word maps (hand-drawn: our run saved no attention maps); the contrastive toy (4 pairs, 2-number vectors,
trained here in numpy, starting angles chosen).

    VOICE=draft manimgx inspect scene.py      # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import re
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403
from diffusion_reference import *  # noqa: E402,F403

EP_DIR = Path(__file__).resolve().parent          # set AFTER the import: diffusion_reference exports HERE (the series)
warnings.filterwarnings("ignore", message=".*encountered in matmul")

PREFIX = "ep15"
IMG = EXPERIMENTS / "images"
LIMG = IMG / "lighthouse"
FRAMES = IMG / "lighthouse_frames30"
ARR = EXPERIMENTS / "arrays"
HL = lambda *keys: measured("mac_runs", "headline", *keys)  # noqa: E731
LH = measured("mac_runs", "lighthouse")
MAC_MD = (EXPERIMENTS / "mac_runs.md").read_text()
NOTES = (EP_DIR.parent / "research" / "notes.md").read_text()
SD_REPO = "sd2-community/stable-diffusion-2-1-base"

# ---------------------------------------------------------------------------- the prompt, token by token

PROMPT_TEXT = "a lighthouse on a rocky coast at dusk, oil painting"
assert f'Prompt: **"{PROMPT_TEXT}"**' in MAC_MD
# Token ids: computed by the course coordinator with the SD 2.1-base tokenizer from the local Hugging Face cache
# (sd2-community/stable-diffusion-2-1-base @ 4e63672c, tokenizer/vocab.json + merges.txt, offline, 2026-10-27), and
# re-checked for this episode against that vocab.json ("a</w>" 320, "lighthouse</w>" 13717, …). Nothing is run here.
TOKENS = ["<|startoftext|>", "a", "lighthouse", "on", "a", "rocky", "coast", "at", "dusk", ",", "oil", "painting",
          "<|endoftext|>"]
IDS = [49406, 320, 13717, 525, 320, 9648, 3720, 536, 19708, 267, 2870, 3086, 49407]
VOCAB = 49_408                                   # text_encoder/config.json vocab_size (CLIP: 49,152 BPE + 256 bytes)
N_TOK = len(TOKENS)
assert N_TOK == len(IDS) == 13
assert TOKENS[1:-1] == re.findall(r"[a-z]+|,", PROMPT_TEXT)        # every word is one token, and so is the comma
assert all(0 <= i < VOCAB for i in IDS) and IDS[0] == VOCAB - 2 and IDS[-1] == VOCAB - 1
assert IDS[1] == IDS[4]                                            # "a" twice: the same row of the vocabulary
SHOWN = ["<start>"] + [t if t != "," else '","' for t in TOKENS[1:-1]] + ["<end>"]   # the comma quoted: a bare "," is too small to read
VOCAB_ROWS = [('"a</w>"', 320), ('"on</w>"', 525), ('"lighthouse</w>"', 13717)]
assert [v for _, v in VOCAB_ROWS] == [IDS[1], IDS[3], IDS[2]]

# ---------------------------------------------------------------------------- the text encoder (our model and SD 1.x)

# SD 2.1-base text_encoder/config.json (the model we ran; local cache, read 2026-10-27): CLIPTextModel, hidden 1024,
# intermediate 4096, 16 heads, 23 layers, 77 positions, vocab 49,408. OpenCLIP ViT-H-14 has 24 text layers: SD 2 reads
# the second-to-last one. SD 1.x (openai/clip-vit-large-patch14, notes §15): 12 layers, width 768, 77 positions.
TE = dict(vocab=VOCAB, ctx=77, width=1024, ff=4096, layers=23, heads=16)
OPENCLIP_H_TEXT_LAYERS = 24
SD1_TE = dict(vocab=VOCAB, ctx=77, width=768, ff=3072, layers=12, heads=12)
CONTEXT = TE["ctx"]


def clip_text_params(vocab, ctx, width, ff, layers, **_):
    """A CLIP text transformer's parameter count: token + position tables, `layers` blocks (attention with biases,
    an MLP, two layer norms), a final layer norm."""
    block = 4 * (width * width + width) + (width * ff + ff) + (ff * width + width) + 2 * 2 * width
    return vocab * width + ctx * width + layers * block + 2 * width


TE_PARAMS = clip_text_params(**TE)
assert TE_PARAMS == LH["models"][SD_REPO]["parts"]["text_encoder"]["params"] == 340_387_840      # counted in the run
assert f"{TE_PARAMS / 1e6:.1f}" == f"{HL('lighthouse_text_encoder_params_M'):.1f}" == "340.4"
assert "340.4 M OpenCLIP-H text" in MAC_MD
SD1_PARAMS = clip_text_params(**SD1_TE)
assert SD1_PARAMS == 123_060_480 and "text encoder alone 123,060,480" in NOTES
assert "OpenCLIP ViT-H (context dim 1024)" in NOTES and TE["width"] == 1024
assert "[DERIVED] SD 1.x uses the final-layer token outputs, 77 × 768" in NOTES
assert TE["layers"] == OPENCLIP_H_TEXT_LAYERS - 1                       # the penultimate layer
OUR_NUMBERS = N_TOK * TE["width"]
assert OUR_NUMBERS == 13_312
SD1_NUMBERS = SD1_TE["ctx"] * SD1_TE["width"]
assert SD1_NUMBERS == 59_136 and "77 × 768 numbers = **59,136** values" in NOTES

# the MLX pipeline we ran pads a batch to its longest prompt (13), not to 77: gpu/third_party/mlx-examples/
# stable_diffusion/__init__.py, _tokenize(); its text model is causal, so the 13 vectors equal those of a 77-padded run.
_mlx = (EXPERIMENTS / "gpu" / "third_party" / "mlx-examples" / "stable_diffusion" / "stable_diffusion")
assert "N = max(lengths)" in (_mlx / "__init__.py").read_text()
assert "mask = indices[:, None] < indices[None]" in (_mlx / "clip.py").read_text()

# ---------------------------------------------------------------------------- where the time goes (the timing run)

TM = LH["timing"]["cfg7.5_50steps"]
TXT_S, DEN_S, VAE_S, TOTAL_S = TM["text_encoder_s"], TM["denoise_total_s"], TM["vae_decode_s"], TM["total_s"]
STEPS, PASSES = TM["denoise_steps"], TM["unet_passes_per_step"]
assert (STEPS, PASSES) == (50, 2)
assert f"{TXT_S:.3f}" == "0.017" and HL("lighthouse_text_encoder_s") == 0.017
assert round(TOTAL_S, 2) == HL("lighthouse_total_s_50steps_cfg7.5") == 7.32
assert abs(TXT_S + DEN_S + VAE_S - TOTAL_S) < 0.01
TXT_SHARE = TXT_S / TOTAL_S
assert f"{100 * TXT_SHARE:.1f}" == "0.2"                                 # "about a fifth of a percent"
assert "| text encoder (13 tokens, prompt + empty negative; see erratum) | **0.017 s** | 0.2% |" in MAC_MD  # row corrected 2026-10-27

# ---------------------------------------------------------------------------- the denoiser's cross-attention layers

# SD 2.1-base unet/config.json (the model we ran; local cache, read 2026-10-27). attention_head_dim is diffusers'
# old name for the number of heads in SD 2: 5/10/20/20 heads of 64 channels.
UNET_CFG = dict(block_out_channels=(320, 640, 1280, 1280), layers_per_block=2, heads=(5, 10, 20, 20),
                down=("CrossAttnDownBlock2D",) * 3 + ("DownBlock2D",), up=("UpBlock2D",) + ("CrossAttnUpBlock2D",) * 3,
                cross_attention_dim=1024, sample_size=64)
DOWN_X = [UNET_CFG["layers_per_block"] if b.startswith("CrossAttn") else 0 for b in UNET_CFG["down"]]       # 64→8
UP_X = [UNET_CFG["layers_per_block"] + 1 if b.startswith("CrossAttn") else 0 for b in UNET_CFG["up"]][::-1]  # 64→8
MID_X = 1                                                     # the middle block: ResBlock, transformer, ResBlock
N_XATTN = sum(DOWN_X) + MID_X + sum(UP_X)
assert DOWN_X == [2, 2, 2, 0] and UP_X == [3, 3, 3, 0] and N_XATTN == 16
SIZES = [UNET_CFG["sample_size"] // 2 ** k for k in range(4)]
assert SIZES == [64, 32, 16, 8] and HL("lighthouse_latent_h") == HL("lighthouse_latent_w") == 64
POSITIONS = SIZES[0] ** 2
assert POSITIONS == 4096
HEAD_DIM = {c // h for c, h in zip(UNET_CFG["block_out_channels"], UNET_CFG["heads"])}
assert HEAD_DIM == {64}                                       # softmax(q·k / √64)
assert UNET_CFG["cross_attention_dim"] == TE["width"]         # keys and values are made from the 1,024-wide vectors
assert "Q has 4,096 rows (one per latent position), K and V have 77 rows (one per token)" in NOTES

# ---------------------------------------------------------------------------- the papers (notes §15, §11, §17)

assert "dataset of 400 million (image, text) pairs" in NOTES
assert "a very large minibatch size of 32,768" in NOTES
assert "T5-XXL (4.6B parameters)" in NOTES
assert "much more than increasing the size of the image diffusion model" in NOTES
assert "The text encoders have a total size of 817M parameters." in NOTES and "Context 77 × 2048 (768 + 1280)" in NOTES
assert "c_ctxt ∈ ℝ^{154×4096}" in NOTES and "4.7B parameter T5 text encoder" in NOTES
assert "the cross-attention layers are the key to controlling the relation between the spatial layout of the image to " \
       "each word in the prompt" in NOTES
assert "SD public release 22 Aug 2022" in NOTES and "| SD 2.1 | 7 Dec 2022 |" in NOTES
assert "(Saharia et al., arXiv:2205.11487, May 2022)" in NOTES or "Saharia et al., arXiv:2205.11487, May 2022" in NOTES
assert "**SDXL** (Podell et al., arXiv:2307.01952, Jul 2023)" in NOTES
assert "arXiv:2403.03206, 5 Mar 2024" in NOTES
SDXL_B, T5_IMAGEN_B, T5_SD3_B = 0.817, 4.6, 4.7
SDXL_WIDTH = 768 + 1280                                            # CLIP ViT-L + OpenCLIP bigG, concatenated
assert SDXL_WIDTH == 2048 == 2 * TE["width"]                      # "vectors twice as wide as ours"
SDXL_BIGG_B = SDXL_B - SD1_PARAMS / 1e9                           # [DERIVED] 817 M minus the CLIP ViT-L text encoder
SD3_B = SDXL_B + T5_SD3_B                                         # [DERIVED] the same two CLIPs + T5-XXL
assert f"{SD3_B:.1f}" == "5.5" and f"{SDXL_BIGG_B * 1000:.0f}" == "694"

# ---------------------------------------------------------------------------- the lighthouse pictures

FR = LH["frames"]
assert (FR["steps"], FR["cfg"], FR["seed"], FR["n_xt_frames"]) == (30, 7.5, 0, 31)
XT = [FRAMES / f"xt_{k:02d}.png" for k in range(31)]
assert all(p.exists() for p in XT)
SEEDS = [LIMG / f"lighthouse_sd21_cfg7.5_steps50_seed{s}.png" for s in range(4)]
assert all(p.exists() for p in SEEDS)
assert sum(1 for r in LH["guidance_sweep"] if r["cfg"] == 7.5 and r["file"] in [p.name for p in SEEDS]) == 4
SDE = LH["sdedit"]
assert SDE["prompt"] == "a lighthouse on a rocky coast at dusk in winter, snow, oil painting"
assert SDE["source"] == SEEDS[0].name and SDE["cfg"] == 7.5
SDE_09 = [r for r in SDE["rows"] if r["strength"] == 0.9][0]
assert SDE_09["steps_run"] == 45 and (LIMG / SDE_09["file"]).exists()
assert "no snow appears" in MAC_MD
# where the tower stands in each seed (read off the pictures by eye, as fractions of width and height): the rings in §6
TOWER = {0: (0.66, 0.36, 0.16, 0.54), 1: (0.925, 0.42, 0.13, 0.36), 2: (0.94, 0.29, 0.09, 0.14), 3: (0.845, 0.43, 0.24, 0.72)}

# ---------------------------------------------------------------------------- the contrastive toy (computed here)

CAPTIONS = ["lighthouse", "a seven", "a swirl", "static"]
TOY_SCALE, TOY_LR, TOY_STEPS = 6.0, 0.01, 600


def clip_toy(a, b, scale=TOY_SCALE, lr=TOY_LR, steps=TOY_STEPS):
    """CLIP's symmetric contrastive loss on a toy: picture i and caption j are unit 2-vectors at angles a_i, b_j; the
    logits are scale × cos(a_i − b_j); the loss is the mean of the row-wise and column-wise cross-entropies with the
    matching pair as the answer (Radford et al. 2021, Fig. 3). Plain gradient descent on the angles."""
    a, b = np.array(a, float), np.array(b, float)
    n, eye = len(a), np.eye(len(a))
    path, losses = [np.concatenate([a, b])], []
    for _ in range(steps):
        d = a[:, None] - b[None, :]
        z = scale * np.cos(d)
        pr = np.exp(z - z.max(1, keepdims=True))
        pr /= pr.sum(1, keepdims=True)
        pc = np.exp(z - z.max(0, keepdims=True))
        pc /= pc.sum(0, keepdims=True)
        losses.append(-0.5 * (np.log(np.diag(pr)).mean() + np.log(np.diag(pc)).mean()))
        g = scale * 0.5 * ((pr - eye) + (pc - eye)) / n            # d loss / d cos
        a = a - lr * n * (g * -np.sin(d)).sum(1)
        b = b - lr * n * (g * np.sin(d)).sum(0)
        path.append(np.concatenate([a, b]))
    return np.array(path), np.array(losses)


_DEG = np.pi / 180
TOY_PATH, TOY_LOSS = clip_toy(np.array([20, 110, 200, 290]) * _DEG, np.array([235, 300, 60, 140]) * _DEG)
TOY_PATH = TOY_PATH + (np.pi / 2 - TOY_PATH[-1, 0])               # turn the plane: the lighthouse pair ends on top
_MOVE = np.concatenate([[0.0], np.cumsum(np.abs(np.diff(TOY_PATH, axis=0)).sum(1))])
TOY_ARC = _MOVE / _MOVE[-1]                                        # replay by distance moved, not by step


def toy_sims(k):
    a, b = TOY_PATH[k, :4], TOY_PATH[k, 4:]
    return np.cos(a[:, None] - b[None, :])


S0, S1 = toy_sims(0), toy_sims(-1)
assert (np.diag(S0) < 0).all() and S0.max() > 0.8                  # at first, every match is wrong
assert (np.diag(S1) > 0.99).all() and (S1 - np.diag(np.diag(S1)) < 0.25).all()     # after: pairs close, others apart
assert TOY_LOSS[0] > 10 and TOY_LOSS[-1] < 0.01

# ---------------------------------------------------------------------------- illustrative attention (not measured)

ILLU_W = [0.10, 0.02, 0.46, 0.02, 0.02, 0.05, 0.07, 0.02, 0.06, 0.01, 0.04, 0.08, 0.05]   # for one tower position
assert len(ILLU_W) == N_TOK and abs(sum(ILLU_W) - 1) < 1e-9 and int(np.argmax(ILLU_W)) == TOKENS.index("lighthouse")
CELL = (2, 5)                                                       # row, column of the 8 × 8 drawing: on the tower

# ---------------------------------------------------------------------------- look

SHADE_BIGG, SHADE_T5 = mix(PROMPT, BG, 0.35), mix(PROMPT, INK, 0.6)    # one family of text encoders, three shades of PINK
FLAT = set("abcdefhiklmnorstuvwxzABCDEFGHIKLMNOPRSTUVWXYZ0123456789")


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


def drive(mob, hooks, run_time=1.4, rate=m.linear):
    """Animate a progress a from 0 to 1, calling every hook(a) each frame."""

    def step(_, a):
        for hook in hooks:
            hook(a)

    return m.UpdateFromAlphaFunc(mob, step, run_time=run_time, rate_func=rate)


class Readout(m.VGroup):
    """A mono text display at a fixed point (centre or left edge); .show(v) re-renders it when the value changes."""

    def __init__(self, point, fmt, value, size=22, color=INK, edge=None):
        super().__init__()
        self.anchor, self.fmt, self.size, self.color, self.edge = np.array(point, float), fmt, size, color, edge
        self.value = value
        self.words = self._make(value)
        self.add(self.words)

    def _make(self, value):
        words = code(self.fmt.format(value), self.size, self.color, weight=m.BOLD)
        return words.move_to(self.anchor, aligned_edge=self.edge) if self.edge is not None else words.move_to(self.anchor)

    def show(self, value):
        if value != self.value:
            self.value = value
            self.words.become(self._make(value))
        return self


def arrow_to(a, b, color=MUTED, width=4, buff=0.12):
    return m.Arrow(a, b, buff=buff, color=color, stroke_width=width, max_tip_length_to_length_ratio=0.25)


def tchip(label, size=20, color=PROMPT, fill=PINK_FILL, min_w=0.5, ink=INK):
    """A token as a mono chip; all chips share one height and one baseline."""
    ref = code("Mx", size)
    words = code(label, size, ink)
    box = card(max(words.width + 0.2, min_w), ref.height + 0.3, color, fill, radius=0.08, stroke=1.8)
    words.move_to(box)
    chars = label.replace(" ", "")
    if len(words) == len(chars):
        sitting = [g for g, ch in zip(words, chars) if ch not in "gjpqy,;_()[]{}|<>\""] or list(words)
        words.shift((box.get_y() - ref.height / 2 - min(g.get_bottom()[1] for g in sitting)) * m.UP)
    return m.VGroup(box, words)


def chip_row(labels, y, size=20, gap=0.08, min_w=0.5, x=0.0):
    row = m.VGroup(*[tchip(lab, size, min_w=min_w) for lab in labels]).arrange(m.RIGHT, buff=gap)
    return row.move_to([x, y, 0])


def span_glyphs(words, s, a, b):
    """The glyphs of mono Text `words` (made from string s) that draw s[a:b]."""
    skip = sum(c.isspace() for c in s[:a])
    width = sum(not c.isspace() for c in s[a:b])
    return words[a - skip: a - skip + width]


def info_card(title, lines, color, fill, width, title_color=None, size=20, num_color=None):
    head = text(title, 22, title_color or color, weight=m.BOLD)
    body = m.VGroup(*[code(s, size, INK, t2c={r"(?<![/\w])\d[\d,.]*\b": num_color or color}) for s in lines])
    body.arrange(m.DOWN, buff=0.13, aligned_edge=m.LEFT)
    inner = m.VGroup(head, body).arrange(m.DOWN, buff=0.18, aligned_edge=m.LEFT)
    box = card(width, inner.height + 0.45, color, fill, radius=0.16, stroke=2.2)
    inner.move_to(box).align_to(box, m.LEFT).shift(0.3 * m.RIGHT)
    return m.VGroup(box, head, body)


def rgb(hexcode):
    return np.array([int(hexcode[i:i + 2], 16) for i in (1, 3, 5)], float)


def load_rgb(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB")).astype(float)


def _blob(u, v, cu, cv, su, sv):
    return np.exp(-(((u - cu) / su) ** 2 + ((v - cv) / sv) ** 2) / 2)


def _upsample(a, n):
    x = (np.arange(n) + 0.5) / n * a.shape[0] - 0.5
    rows = np.array([np.interp(x, np.arange(a.shape[1]), r) for r in a])
    return np.array([np.interp(x, np.arange(a.shape[0]), c) for c in rows.T]).T


def word_map(word, n=256):
    """A HAND-DRAWN stand-in for a word's cross-attention map on the seed-0 lighthouse (no map was saved in our run):
    soft blobs placed by eye, pooled to 16 × 16 like a coarse attention layer, then smoothed back up."""
    u, v = np.meshgrid((np.arange(512) + 0.5) / 512, (np.arange(512) + 0.5) / 512)
    tower = np.maximum(_blob(u, v, 0.665, 0.40, 0.045, 0.17), _blob(u, v, 0.66, 0.17, 0.045, 0.05))
    if word == "lighthouse":
        mask = tower
    elif word == "rocky":
        mask = np.maximum.reduce([_blob(u, v, 0.80, 0.67, 0.15, 0.10), _blob(u, v, 0.90, 0.86, 0.10, 0.10),
                                  _blob(u, v, 0.17, 0.86, 0.15, 0.08), _blob(u, v, 0.45, 0.90, 0.08, 0.06)])
    else:  # dusk: the sky
        mask = np.clip(1.0 - v / 0.55, 0, 1) ** 0.8 * (1 - 0.9 * tower)
    pooled = mask.reshape(16, 32, 16, 32).mean(axis=(1, 3))
    return np.clip(_upsample(pooled / pooled.max(), n), 0, 1)


def overlay(path, mask, color=PROMPT):
    """The picture dimmed, with `mask` painted over it in `color` (a heat map look)."""
    img = load_rgb(path)
    k = img.shape[0] // mask.shape[0]
    small = img.reshape(mask.shape[0], k, mask.shape[1], k, 3).mean(axis=(1, 3))
    a = (0.82 * mask ** 1.1)[..., None]
    out = small * (0.30 + 0.25 * mask[..., None]) * (1 - a) + rgb(color) * a
    return np.clip(out, 0, 255).astype(np.uint8)


def swirl_thumb(n=40):
    pts = swirl(4000, seed=0)
    hist, _, _ = np.histogram2d(pts[:, 1], pts[:, 0], bins=n, range=[[-1.1, 1.1], [-1.1, 1.1]])
    return tint(np.flipud(hist), DATA, 0, np.percentile(hist, 97))


def seven_thumb():
    f = np.load(ARR / "digit_samples_ddpm_ddim50_w1.npz")
    assert int(f["labels"][27]) == 7 and "seed 1234" in str(f["note"])
    return tint(f["samples"][27], DATA)


# ---------------------------------------------------------------------------- the episode


class Ep15WordsToPictures(Episode):
    number = 15

    def lesson(self):
        self.hook()
        self.tokens()
        self.encoder()
        self.read_once()
        self.contrastive()
        self.cross_attention()
        self.where_words_land()
        self.scale()
        self.sensitivity()
        self.recap()

    # ------------------------------------------------------------------ 0 · hook

    def hook(self):
        self.next_section("Hook", notes=(
            "Generated on this Mac (Apple M4 Max, MLX): Stable Diffusion 2.1-base (sd2-community mirror), 30 steps, "
            "guidance 7.5, seed 0, deterministic Euler; frames are the noisy latent x_t decoded at each step "
            "(research/experiments/images/lighthouse_frames30/xt_00..30.png, mac_runs.md §5). The prompt is the course's "
            "lighthouse prompt. 'Numbers in, numbers out': the denoiser takes and returns 64×64×4 latents (Episode 17)."))
        h, yp = 3.35, -0.95
        pics = [picture(p, h).move_to([0, yp, 0]) for p in XT]
        rim = m.Rectangle(width=h + 0.12, height=h + 0.12).set_stroke(EDGE, 2.5).move_to([0, yp, 0])
        ptxt = code(PROMPT_TEXT, 24, INK)
        plab = code("prompt", 20, PROMPT, weight=m.BOLD)
        inner = m.VGroup(plab, ptxt).arrange(m.RIGHT, buff=0.35)
        pbox = card(inner.width + 0.7, inner.height + 0.5, PROMPT, CARD, radius=0.16, stroke=2.5)
        inner.move_to(pbox)
        sit(plab, baseline(ptxt))
        m.VGroup(pbox, inner).move_to([0, 1.75, 0])
        step = Readout([h / 2 + 0.3, yp - h / 2 + 0.12, 0], "step {:d} of 30", 0, 22, TIME, edge=m.LEFT)
        cap = cite("generated on this Mac · Stable Diffusion 2.1-base · 30 steps · guidance 7.5 · seed 0").move_to([0, -3.3, 0])

        _dw = m.VGroup(text("denoiser", 28, INK, weight=m.BOLD), text("numbers in, numbers out", 20, MUTED)).arrange(m.DOWN, buff=0.12)
        den = m.VGroup(card(_dw.width + 0.6, 1.45, MODEL, AMBER_FILL), _dw)
        den[1].move_to(den[0])
        den.move_to([-4.65, yp + 0.4, 0])
        link = arrow_to(pbox.get_corner(m.DL) + np.array([1.0, -0.05, 0]), den[0].get_top() + np.array([0.3, 0.05, 0]),
                        PROMPT, 4, 0.1)
        qmark = title_text("?", 60, PROMPT).next_to(link.get_center(), m.LEFT, buff=0.25)
        plan = m.VGroup()
        for i, s in enumerate(("words → vectors", "learned from captions", "pixels listen to words")):
            words = m.VGroup(code(str(i + 1), 22, PROMPT, weight=m.BOLD), text(s, 22, INK)).arrange(m.RIGHT, buff=0.22)
            box = card(3.75, 0.72, PROMPT if i == 0 else EDGE, CARD, radius=0.14, stroke=2)
            words.move_to(box).align_to(box, m.LEFT).shift(0.25 * m.RIGHT)
            sit(words[0], baseline(words[1]))
            plan.add(m.VGroup(box, words))
        plan.arrange(m.DOWN, buff=0.28).move_to([4.6, yp + 0.25, 0])
        assert plan.get_right()[0] < 6.6 and rim.get_right()[0] < plan.get_left()[0] - 0.3
        assert den.get_left()[0] > -6.6 and den.get_right()[0] < rim.get_left()[0] - 0.3

        self.say(
            "[Type a sentence], [and out of the noise comes a picture].",
            m.AnimationGroup(self.new_header("00 · hook", "From a sentence to a picture"), m.FadeIn(pbox), m.FadeIn(plab),
                             m.AddTextLetterByLetter(ptxt, time_per_char=0.03)),
            m.AnimationGroup(m.FadeIn(rim), m.FadeIn(step), m.FadeIn(cap),
                             m.Succession(m.FadeIn(pics[0], run_time=0.3), *[m.FadeIn(p, run_time=0.1) for p in pics[1:]]),
                             drive(step, [lambda a: step.show(int(round(a * 30)))], run_time=3.3)),
        )
        self.remove(*pics[:-1])
        self.say(
            "[But the denoiser only understands numbers]. [So how do words steer it]?",
            m.FadeIn(den, shift=0.2 * m.RIGHT),
            m.AnimationGroup(m.GrowArrow(link), m.FadeIn(qmark, scale=0.6)),
        )
        self.say(
            "[Today]: [words become vectors], [vectors learn from captions], [and pixels listen to words].",
            m.Circumscribe(m.VGroup(pbox, inner), color=PROMPT, buff=0.08),
            m.FadeIn(plan[0], shift=0.15 * m.LEFT),
            m.FadeIn(plan[1], shift=0.15 * m.LEFT),
            m.FadeIn(plan[2], shift=0.15 * m.LEFT),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 1 · tokens

    def tokens(self):
        self.next_section("1 · Tokens", notes=(
            f"Real token ids of the lighthouse prompt for SD 2.1-base's tokenizer (CLIP BPE, lower-cased; vocab {VOCAB:,}): "
            + ", ".join(f"{t} {i}" for t, i in zip(TOKENS, IDS)) + ". Computed by the course coordinator from the local "
            "tokenizer files (offline) and re-checked here against vocab.json; '</w>' marks the end of a word. Every word "
            f"is one token here; a rarer word would split into pieces. Context: {CONTEXT} positions (CLIP; notes App. A: "
            "'77 tokens' incl. start and end). Our MLX pipeline pads a batch only to its longest prompt (13 here), not to "
            "77 (mlx-examples stable_diffusion _tokenize); its text model is causal, so the 13 vectors are the same as in a "
            "77-padded run. mac_runs.md's timing row said '77 tokens' until its erratum of 2026-10-27: the pipeline code says 13."))
        ptxt = code(PROMPT_TEXT, 26, INK).move_to([0, 2.0, 0])
        final = chip_row(SHOWN, 0.95, gap=0.1)
        assert final.width < 12.9, final.width
        mids = final[1:-1]
        dx = -m.VGroup(*mids).get_center()[0]
        finals = [c.get_center().copy() for c in final]
        for c in mids:
            c.shift(dx * m.RIGHT)
        ids = m.VGroup()
        for i, (n_, c) in enumerate(zip(IDS, final)):
            num = code(str(n_), 20, INK)
            assert num.width < c.width + 0.02, (TOKENS[i], num.width, c.width)
            num.move_to([c.get_x(), 0.38 if i % 2 == 0 else 0.04, 0])      # staggered: neighbours never crowd
            tick = m.Line([c.get_x(), c.get_bottom()[1] - 0.05, 0], [c.get_x(), num.get_top()[1] + 0.07, 0]).set_stroke(PROMPT, 2, opacity=0.7)
            ids.add(m.VGroup(tick, num))       # id i sits in chip i's own column: ids are narrower than their chips
        spans = [mt.span() for mt in re.finditer(r"[a-z]+|,", PROMPT_TEXT)]
        assert len(spans) == len(mids)

        vrows = m.VGroup(*[code(f"{k}: {v}", 20, INK, t2c={r'"[^"]*"': PROMPT}) for k, v in VOCAB_ROWS])
        vrows.arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT)
        vtitle = code(f"vocab.json · {VOCAB:,} entries", 20, MUTED)
        vin = m.VGroup(vtitle, vrows).arrange(m.DOWN, buff=0.16, aligned_edge=m.LEFT)
        vbox = card(vin.width + 0.6, vin.height + 0.45, EDGE, CARD, radius=0.16, stroke=2)
        vin.move_to(vbox)
        vocab = m.VGroup(vbox, vin).move_to([-3.3, -1.2, 0])
        big = m.VGroup(title_text(str(N_TOK), 72, PROMPT), text("tokens", 26, MUTED)).arrange(m.RIGHT, buff=0.2)
        sit(big[1], baseline(big[0]))
        big.move_to([3.3, -1.2, 0])

        n_slots = CONTEXT
        slots = m.VGroup(*[m.Square(0.13).set_stroke(PROMPT if i < N_TOK else EDGE, 1.2)
                           .set_fill(PROMPT if i < N_TOK else CARD, 0.85 if i < N_TOK else 1) for i in range(n_slots)])
        slots.arrange(m.RIGHT, buff=0.03).move_to([0, -2.78, 0])
        s_lab = code(f"{CONTEXT} positions: the most it reads", 20, MUTED).next_to(slots, m.UP, buff=0.14).align_to(slots, m.LEFT)
        used = code(f"{N_TOK} used", 20, PROMPT).next_to(slots[:N_TOK], m.DOWN, buff=0.14).align_to(slots, m.LEFT)
        cut = m.Line(slots.get_corner(m.UR) + [0.09, 0.14, 0], slots.get_corner(m.DR) + [0.09, -0.14, 0]).set_stroke(RED, 4)
        cut_l = text("longer prompts are cut here", 20, RED).next_to(slots, m.UP, buff=0.14).align_to(cut, m.RIGHT).shift(0.2 * m.LEFT)
        ours = cite("our MLX run encoded just these 13 · the reference code pads to 77").next_to(slots, m.DOWN, buff=0.14).align_to(slots, m.RIGHT)
        assert slots.get_left()[0] > -6.6 and cut.get_right()[0] < 6.6 and ours.get_bottom()[1] > -3.55
        assert used.get_right()[0] < ours.get_left()[0] - 0.3 and s_lab.get_right()[0] < cut_l.get_left()[0] - 0.3
        assert vocab.get_bottom()[1] > s_lab.get_top()[1] + 0.15 and vocab.get_top()[1] < ids.get_bottom()[1] - 0.15

        self.say(
            "[First, a tokenizer] [cuts the prompt into pieces called tokens].",
            m.AnimationGroup(self.new_header("01 · tokens", "Words become numbers"), m.FadeIn(ptxt, shift=0.15 * m.DOWN)),
            m.LaggedStart(*[m.FadeIn(c, target_position=span_glyphs(ptxt, PROMPT_TEXT, a, b))
                            for c, (a, b) in zip(mids, spans)], lag_ratio=0.12, run_time=1.8),
        )
        self.say(
            "[Here, every word is one token], [and so is the comma].",
            m.LaggedStart(*[m.Indicate(c, color=PROMPT, scale_factor=1.08) for c in mids], lag_ratio=0.08, run_time=1.6),
            m.Circumscribe(mids[TOKENS.index(",") - 1], color=PROMPT, buff=0.06),
        )
        self.say(
            "[Each token is a number]: [its row in a vocabulary of forty-nine thousand, four hundred and eight].",
            m.LaggedStart(*[m.FadeIn(ids[i + 1], shift=0.15 * m.DOWN) for i in range(len(mids))], lag_ratio=0.08, run_time=1.4),
            m.FadeIn(vocab, shift=0.15 * m.UP),
        )
        self.say(
            "[Two special tokens mark the start and the end]: [thirteen in all].",
            m.AnimationGroup(*[c.animate.move_to(finals[i + 1]) for i, c in enumerate(mids)],
                             *[ids[i + 1].animate.move_to([finals[i + 1][0], ids[i + 1].get_y(), 0]) for i in range(len(mids))],
                             m.FadeIn(final[0], shift=0.3 * m.RIGHT), m.FadeIn(final[-1], shift=0.3 * m.LEFT),
                             m.FadeIn(ids[0], shift=0.3 * m.RIGHT), m.FadeIn(ids[-1], shift=0.3 * m.LEFT), run_time=1.2),
            m.FadeIn(big, shift=0.15 * m.UP),
        )
        self.say(
            "[The encoder reads seventy-seven at most]; [anything longer is cut off].",
            m.AnimationGroup(m.FadeIn(s_lab), m.LaggedStart(*[m.FadeIn(s) for s in slots], lag_ratio=0.02, run_time=1.4),
                             m.FadeIn(used), m.FadeIn(ours)),
            m.AnimationGroup(m.Create(cut), m.FadeIn(cut_l)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 2 · the encoder

    def encoder(self):
        self.next_section("2 · Text encoder", notes=(
            f"Our model (SD 2.1-base): OpenCLIP ViT-H/14's text tower, {TE_PARAMS:,} parameters (counted in the run, "
            f"mac_runs.json lighthouse.models; recomputed here from its config: vocab {VOCAB:,} × {TE['width']}, "
            f"{CONTEXT} positions, {TE['layers']} layers of width {TE['width']} with MLP {TE['ff']}). OpenCLIP's ViT-H "
            f"text tower has {OPENCLIP_H_TEXT_LAYERS} layers; SD 2 keeps {TE['layers']}: it reads the second-to-last "
            f"(penultimate) layer. Output for our prompt: {N_TOK} × {TE['width']} = {OUR_NUMBERS:,} numbers. SD 1.x: "
            f"CLIP ViT-L/14, {SD1_PARAMS:,} parameters, final layer, {SD1_TE['ctx']} × {SD1_TE['width']} = "
            f"{SD1_NUMBERS:,} (notes §15). The encoder is causal: each token's vector depends on itself and the tokens "
            "before it. Vector values on screen are shading only, for example (the encoder was not run for this episode)."))
        row = chip_row(SHOWN, 2.05)
        band = card(row.width + 0.3, 0.62, PROMPT, PINK_FILL, radius=0.14, stroke=2.2).move_to([row.get_x(), 1.25, 0])
        band_t = m.VGroup(text("text encoder", 22, INK, weight=m.BOLD), code("frozen · runs once", 20, MUTED)).arrange(m.RIGHT, buff=0.35)
        band_t.move_to(band)
        sit(band_t[1], baseline(band_t[0]))
        rng = np.random.default_rng(15)
        cols = m.VGroup()
        for c in row:
            vals = rng.uniform(0.1, 1.0, 7)
            cells = m.VGroup(*[m.Rectangle(width=0.3, height=0.17).set_stroke(PROMPT, 1.2).set_fill(mix(CARD, PROMPT, 0.85 * v), 1)
                               for v in vals]).arrange(m.DOWN, buff=0.04)
            dots = code("⋮", 20, MUTED)
            col = m.VGroup(cells, dots).arrange(m.DOWN, buff=0.06)
            cols.add(col.move_to([c.get_x(), 0.0, 0]).align_to([0, 0.72, 0], m.UP))
        back = m.Arrow(row[8].get_top() + [0.0, 0.08, 0], row[0].get_top() + [0.0, 0.08, 0], buff=0.0, color=PROMPT,
                       stroke_width=3, max_tip_length_to_length_ratio=0.03)
        back.shift(0.22 * m.UP)
        back_l = text("read with the words before it", 20, PROMPT)
        back_l.next_to(back, m.RIGHT, buff=0.2)
        assert back_l.get_right()[0] < 6.6 and back.get_top()[1] < 2.75, (back_l.get_right(), back.get_top())
        x_r = cols[-1].get_right()[0] + 0.15
        brk = m.VGroup(m.Line([x_r, 0.72, 0], [x_r, cols[-1].get_bottom()[1], 0]),
                       m.Line([x_r - 0.1, 0.72, 0], [x_r, 0.72, 0]),
                       m.Line([x_r - 0.1, cols[-1].get_bottom()[1], 0], [x_r, cols[-1].get_bottom()[1], 0])).set_stroke(INK, 2.5)
        size_l = code(f"{N_TOK} vectors × {TE['width']:,} numbers = {OUR_NUMBERS:,}", 22, INK, t2c={r"[\d,]+": PROMPT})
        size_l.move_to([0, cols.get_bottom()[1] - 0.32, 0]).align_to([x_r, 0, 0], m.RIGHT)

        ours = info_card("ours · Stable Diffusion 2.1-base",
                         ["OpenCLIP ViT-H/14 · text half", f"{TE_PARAMS / 1e6:.1f} M parameters · width {TE['width']:,}",
                          f"output of layer {TE['layers']} of {OPENCLIP_H_TEXT_LAYERS}"],
                         PROMPT, PINK_FILL, 6.15).move_to([-3.25, -2.55, 0])
        sd1 = info_card("Stable Diffusion 1.x · August 2022",
                        ["CLIP ViT-L/14 (OpenAI) · text half", f"{SD1_PARAMS / 1e6:.1f} M parameters · width {SD1_TE['width']}",
                         f"{SD1_TE['ctx']} × {SD1_TE['width']} = {SD1_NUMBERS:,} numbers"],
                        EDGE, CARD, 6.15, title_color=INK, num_color=INK).move_to([3.25, -2.55, 0])
        assert ours.get_top()[1] < size_l.get_bottom()[1] - 0.12 and ours.get_bottom()[1] > -3.55, (ours.get_top(), size_l.get_bottom())
        assert sd1.get_right()[0] < 6.62 and ours.get_left()[0] > -6.62

        self.say(
            "[Next, a text encoder] [turns each token into a vector of numbers], [each read in the light of the words before it].",
            m.AnimationGroup(self.new_header("02 · encoder", "Tokens become vectors"),
                             m.LaggedStart(*[m.FadeIn(c) for c in row], lag_ratio=0.05, run_time=0.9),
                             m.FadeIn(band), m.FadeIn(band_t)),
            m.LaggedStart(*[m.FadeIn(col, shift=0.35 * m.DOWN) for col in cols], lag_ratio=0.07, run_time=1.6),
            m.AnimationGroup(m.GrowArrow(back), m.FadeIn(back_l)),
        )
        self.say(
            "[Our lighthouse model], Stable Diffusion two point one, [uses the text half of OpenCLIP]: "
            "[three hundred forty million parameters].",
            m.AnimationGroup(m.FadeIn(ours[0]), m.FadeIn(ours[1])),
            m.FadeIn(ours[2][0], shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(ours[2][1:], shift=0.1 * m.UP)),
        )
        self.say(
            "[Thirteen tokens in], [thirteen vectors out], [each one thousand and twenty-four numbers long].",
            m.LaggedStart(*[m.Indicate(c, color=PROMPT, scale_factor=1.06) for c in row], lag_ratio=0.05, run_time=1.0),
            m.LaggedStart(*[m.Indicate(col, color=PROMPT, scale_factor=1.04) for col in cols], lag_ratio=0.05, run_time=1.0),
            m.AnimationGroup(m.Create(brk), m.FadeIn(size_l, shift=0.1 * m.UP)),
        )
        self.say(
            "[Back in Stable Diffusion one], [the encoder was OpenAI's smaller CLIP]: "
            "[seven hundred sixty-eight numbers per token].",
            m.AnimationGroup(m.FadeIn(sd1[0]), m.FadeIn(sd1[1])),
            m.FadeIn(sd1[2][0:2], shift=0.1 * m.UP),
            m.FadeIn(sd1[2][2], shift=0.1 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 2b · read once

    def read_once(self):
        self.next_section("2b · Read once", notes=(
            f"Timing run on this Mac (median of 3; mac_runs.md §5): text encoder {TXT_S:.3f} s, denoiser {DEN_S:.2f} s "
            f"({STEPS} steps × {PASSES} passes, guidance 7.5), decode {VAE_S:.3f} s, total {TOTAL_S:.2f} s; the text "
            f"encoder is {100 * TXT_SHARE:.2f}% of it. It encodes the prompt and the empty negative prompt once (one batch "
            "of two); the denoiser re-reads both sets of vectors at every step (Episode 14). The Mac was shared: ±5–10%."))
        x0, x1 = -6.0, 6.0
        k = (x1 - x0) / TOTAL_S
        y = 1.2
        widths = [max(TXT_S * k, 0.05), 0.0, VAE_S * k]
        widths[1] = (x1 - x0) - widths[0] - widths[2]                    # the sliver is drawn wider than 17 ms
        segs, x = [], x0
        for w, color in zip(widths, (PROMPT, MODEL, DATA)):
            segs.append(m.Rectangle(width=w, height=0.55).set_stroke(color, 2).set_fill(FILLS[color], 1).move_to([x + w / 2, y, 0]))
            x += w
        bar = m.VGroup(*segs)
        head_l = m.VGroup(text("one picture", 24, INK, weight=m.BOLD), code(f"{TOTAL_S:.2f} s · {STEPS} steps · guidance 7.5", 20, MUTED))
        head_l.arrange(m.RIGHT, buff=0.3).next_to(bar, m.UP, buff=0.22).align_to(bar, m.LEFT)
        sit(head_l[1], baseline(head_l[0]))
        den_l = code(f"denoiser {DEN_S:.2f} s", 22, INK).move_to(segs[1])
        dec_l = code(f"decode {VAE_S:.2f} s", 20, DATA).next_to(segs[2], m.DOWN, buff=0.16).align_to(segs[2], m.RIGHT)
        txt_l = code(f"text encoder {TXT_S * 1000:.0f} ms = {100 * TXT_SHARE:.1f}%", 22, PROMPT, weight=m.BOLD)
        txt_l.next_to(segs[0], m.DOWN, buff=0.16).align_to(bar, m.LEFT)
        cap = cite("timing run on this Mac · median of 3 · the Mac was shared: ±5–10%").move_to([0, -3.35, 0])

        yb = -1.2
        _ol = code(f"prompt · {N_TOK} × {TE['width']:,}", 20, INK)
        ours_b = m.VGroup(card(_ol.width + 0.45, 0.66, PROMPT, PINK_FILL, radius=0.12, stroke=2.2), _ol)
        ours_b[1].move_to(ours_b[0])
        empty_b = m.VGroup(card(_ol.width + 0.45, 0.66, PROMPT, CARD, radius=0.12, stroke=2.2), code("empty prompt", 20, INK))
        empty_b[1].move_to(empty_b[0])
        blocks = m.VGroup(ours_b, empty_b).arrange(m.DOWN, buff=0.22).move_to([0, yb, 0]).align_to([-6.4, 0, 0], m.LEFT)
        ep14 = code("Episode 14: guidance", 20, MUTED).next_to(empty_b, m.DOWN, buff=0.14)
        n = STEPS
        cw, gap = 0.14, 0.03
        cells = m.VGroup(*[m.Rectangle(width=cw, height=0.66).set_stroke(MODEL, 1.4).set_fill(AMBER_FILL, 1) for _ in range(n)])
        cells.arrange(m.RIGHT, buff=gap).move_to([0, yb, 0]).align_to([blocks.get_right()[0] + 0.7, 0, 0], m.LEFT)
        assert cells.get_right()[0] < 6.6, cells.get_right()
        rail = m.VGroup(*[m.Line(b_.get_right() + [0.1, 0, 0], cells.get_left() + [-0.1, 0, 0]).set_stroke(PROMPT, 3) for b_ in blocks])
        steps_l = code(f"{STEPS} denoising steps · {PASSES} passes each", 20, MUTED).next_to(cells, m.DOWN, buff=0.18).align_to(cells, m.LEFT)
        reread = text("every step reads the same vectors", 22, PROMPT).next_to(cells, m.UP, buff=0.2).align_to(cells, m.LEFT)
        assert txt_l.get_bottom()[1] > reread.get_top()[1] + 0.3 and dec_l.get_bottom()[1] > reread.get_top()[1] + 0.2

        self.say(
            "[On this Mac, encoding the prompt took seventeen milliseconds]: "
            "[about a fifth of a percent of the picture's seven point three seconds].",
            m.AnimationGroup(self.new_header("02 · encoder", "Read once, used at every step"), m.FadeIn(head_l),
                             m.GrowFromEdge(segs[0], m.LEFT), m.FadeIn(txt_l), m.FadeIn(cap)),
            m.AnimationGroup(m.GrowFromEdge(segs[1], m.LEFT, run_time=1.2), m.FadeIn(den_l),
                             m.GrowFromEdge(segs[2], m.LEFT), m.FadeIn(dec_l)),
        )
        self.say(
            "[It runs once], [for our prompt and for the empty one guidance needs]; "
            "[the denoiser rereads the vectors at every step].",
            m.FadeIn(ours_b, shift=0.15 * m.RIGHT),
            m.AnimationGroup(m.FadeIn(empty_b, shift=0.15 * m.RIGHT), m.FadeIn(ep14)),
            m.AnimationGroup(m.FadeIn(steps_l), m.FadeIn(reread),
                             m.Succession(m.AnimationGroup(m.Create(rail), m.FadeIn(cells), run_time=0.6),
                                          m.LaggedStart(*[c.animate(rate_func=m.there_and_back, run_time=0.35).set_fill(PROMPT, 1) for c in cells],
                                                        lag_ratio=0.12, run_time=2.4))),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 3 · contrastive training

    def contrastive(self):
        self.next_section("3 · Contrastive training", notes=(
            "CLIP (Radford et al., arXiv:2103.00020, Feb 2021, OpenAI): a text encoder and an image encoder trained together "
            "on 400 million web image–caption pairs with a symmetric contrastive loss; batches of 32,768 pairs (notes §15). "
            "OpenCLIP is an open implementation; our model's encoder is OpenCLIP's ViT-H text tower, kept frozen in "
            "Stable Diffusion (model card: 'a fixed, pretrained text encoder (OpenCLIP-ViT/H)'). The toy is computed here: "
            f"4 pairs, unit 2-vectors (angles), logits {TOY_SCALE:g} × cosine, the same symmetric loss, {TOY_STEPS} steps of "
            f"gradient descent; loss {TOY_LOSS[0]:.1f} → {TOY_LOSS[-1]:.3f}. Starting angles chosen for the picture (every "
            "pair starts wrong); the plane is turned so the lighthouse pair ends on top. Pictures: our lighthouse (SD 2.1, "
            "seed 0), a generated 7 (the digit model), the swirl, static."))
        # the grid: one column per picture, one row per caption; cells[i][j] = picture i against caption j
        cw, ch, gx, gy = 1.0, 0.62, 0.08, 0.08
        xs = [-4.05 + i * (cw + gx) for i in range(4)]
        ys = [0.85 - j * (ch + gy) for j in range(4)]
        thumbs_src = [SEEDS[0], seven_thumb(), swirl_thumb(), tint(noise_image(24, 24, seed=2), NOISE, -2.5, 2.5)]
        heads = m.Group(*[framed(picture(s_, 0.62)).move_to([x, ys[0] + ch / 2 + 0.5, 0]) for s_, x in zip(thumbs_src, xs)])
        caps = m.VGroup(*[tchip(c, 20).move_to([0, y, 0]).align_to([xs[0] - cw / 2 - 0.16, 0, 0], m.RIGHT)
                          for c, y in zip(CAPTIONS, ys)])
        cells = m.VGroup(*[m.VGroup(*[m.Rectangle(width=cw, height=ch).set_stroke(GREEN if i == j else EDGE, 2.5 if i == j else 1.5)
                                      .set_fill(heat(S0[i, j], -1, 1, 0.8, high=GREEN, low=RED), 1).move_to([x, y, 0])
                                      for j, y in enumerate(ys)]) for i, x in enumerate(xs)])
        fmt = lambda v: f"{0.0 if abs(v) < 0.005 else v:.2f}".replace("-", "−")  # noqa: E731
        nums = [[Readout([x, y, 0], "{}", fmt(S0[i, j]), 20, INK) for j, y in enumerate(ys)] for i, x in enumerate(xs)]
        num_g = m.VGroup(*[n_ for r in nums for n_ in r])
        diag = m.VGroup(*[cells[i][i] for i in range(4)])
        assert caps.get_left()[0] > -6.6 and heads.get_top()[1] < 2.6, (caps.get_left(), heads.get_top())
        grid_right = xs[-1] + cw / 2

        cx, cy, r_in, r_gap = 3.4, -0.35, 0.95, 1.4
        c0 = np.array([cx, cy, 0.0])
        ring = m.Circle(radius=r_in).set_stroke(EDGE, 2).move_to(c0)
        origin = m.Dot(c0, radius=0.06, color=MUTED)
        dots_img = m.Group(*[framed(picture(s, 0.5), DATA, buff=0.04, stroke=2) for s in thumbs_src])
        dots_txt = m.VGroup(*[tchip(c, 20) for c in CAPTIONS])
        rays_img = m.VGroup(*[m.Line(c0, c0 + [0.5, 0, 0]).set_stroke(DATA, 3) for _ in range(4)])
        rays_txt = m.VGroup(*[m.Line(c0, c0 + [0.5, 0, 0]).set_stroke(PROMPT, 3) for _ in range(4)])

        def extent(mob, d):
            return mob.width / 2 * abs(d[0]) + mob.height / 2 * abs(d[1])

        def place(k):
            angles = TOY_PATH[k]
            for i in range(4):
                for which, th in ((0, angles[i]), (1, angles[4 + i])):
                    d = np.array([np.cos(th), np.sin(th), 0.0])
                    if which == 0:
                        dots_img[i].move_to(c0 + r_in * d)
                        rays_img[i].put_start_and_end_on(c0 + 0.08 * d, c0 + (r_in - extent(dots_img[i], d)) * d)
                    else:
                        dots_txt[i].move_to(c0 + (r_gap + extent(dots_txt[i], d)) * d)
                        rays_txt[i].put_start_and_end_on(c0 + 0.08 * d, c0 + (r_gap - 0.02) * d)

        def at_alpha(a):
            return int(np.clip(np.searchsorted(TOY_ARC, a), 0, len(TOY_ARC) - 1))

        place(0)
        panel = m.Group(rays_txt, rays_img, ring, origin, dots_img, dots_txt)
        for k_ in (0, len(TOY_PATH) - 1):
            place(k_)
            bb = m.Group(dots_img, dots_txt)
            assert bb.get_right()[0] < 6.6 and bb.get_left()[0] > grid_right + 0.3 and bb.get_top()[1] < 1.75 \
                and bb.get_bottom()[1] > -2.4, (k_, bb.get_right(), bb.get_left(), bb.get_top(), bb.get_bottom())
            for i in range(4):
                for j in range(4):
                    if i != j:
                        a_, b_ = dots_txt[i], dots_txt[j]
                        assert abs(a_.get_x() - b_.get_x()) > (a_.width + b_.width) / 2 + 0.05 or \
                            abs(a_.get_y() - b_.get_y()) > (a_.height + b_.height) / 2 + 0.05, (k_, i, j)
        place(0)
        leg = m.VGroup(m.VGroup(m.Square(0.2).set_stroke(DATA, 2).set_fill(TEAL_FILL, 1), text("picture → image encoder → vector", 20, INK)).arrange(m.RIGHT, buff=0.15),
                       m.VGroup(m.Square(0.2).set_stroke(PROMPT, 2).set_fill(PINK_FILL, 1), text("caption → text encoder → vector", 20, INK)).arrange(m.RIGHT, buff=0.15))
        leg.arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT).move_to([cx, 2.15, 0])
        assert leg.get_right()[0] < 6.6 and leg.get_left()[0] > grid_right + 0.2, (leg.get_left(), leg.get_right())
        toy = cite("toy, trained here: 4 pairs, vectors of 2 numbers · similarity = cosine of the angle · pictures from our runs").move_to([0, -2.75, 0])
        src = cite("CLIP: 400 M picture–caption pairs, batches of 32,768 · Radford et al., 2021").move_to([0, -3.25, 0])
        lock = icon(recolored("lock", PROMPT, PINK_FILL, PREFIX), 0.42)
        frozen = m.VGroup(lock, text("kept frozen inside Stable Diffusion", 22, PROMPT, weight=m.BOLD)).arrange(m.RIGHT, buff=0.2)
        frozen.move_to([0, -2.2, 0]).align_to(caps, m.LEFT)
        assert frozen.get_bottom()[1] > toy.get_top()[1] + 0.08 and frozen.get_top()[1] < cells.get_bottom()[1] - 0.1

        def train(a):
            k_ = at_alpha(a)
            place(k_)
            s = toy_sims(k_)
            for i in range(4):
                for j in range(4):
                    cells[i][j].set_fill(heat(s[i, j], -1, 1, 0.8, high=GREEN, low=RED), 1)
            step_n = int(a * 40)
            if step_n != getattr(train, "last", -1):
                train.last = step_n
                for i in range(4):
                    for j in range(4):
                        nums[i][j].show(fmt(s[i, j]))

        self.say(
            "[Where do these vectors come from]? [From CLIP, trained by OpenAI in twenty twenty-one], "
            "[on four hundred million pictures with their captions].",
            self.new_header("03 · contrastive", "Learned from captions"),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(h_) for h_ in heads], lag_ratio=0.15), m.LaggedStart(*[m.FadeIn(c) for c in caps], lag_ratio=0.15)),
            m.FadeIn(src),
        )
        self.say(
            "[Two encoders, one for pictures and one for text], [turn each into a direction in one shared space].",
            m.FadeIn(leg),
            m.AnimationGroup(m.FadeIn(ring), m.FadeIn(origin), m.Create(rays_img), m.Create(rays_txt),
                             m.FadeIn(dots_img), m.FadeIn(dots_txt), m.FadeIn(toy)),
        )
        self.say(
            "[Training scores every picture against every caption in the batch]: [matching pairs sit on the diagonal].",
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(c) for row_ in cells for c in row_], lag_ratio=0.04, run_time=1.2),
                             m.LaggedStart(*[m.FadeIn(n_) for n_ in num_g], lag_ratio=0.04, run_time=1.2)),
            m.Circumscribe(diag, color=GREEN, buff=0.06),
        )
        self.say(
            "[It pulls each matching pair together, and pushes all the others apart].",
            drive(panel, [train], run_time=4.2, rate=m.smooth),
        )
        self.say(
            "[Our lighthouse model keeps such an encoder frozen]: [its vectors already describe what pictures look like].",
            m.FadeIn(frozen, shift=0.15 * m.UP),
            m.AnimationGroup(m.Circumscribe(diag, color=GREEN, buff=0.06), m.Indicate(caps[0], color=PROMPT, scale_factor=1.12)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 4 · cross-attention

    def cross_attention(self):
        self.next_section("4 · Cross-attention", notes=(
            "Cross-attention (LDM §3.3): queries from the image features at each position, keys and values from the text "
            f"vectors; weights = softmax(q·k/√d), d = {next(iter(HEAD_DIM))} per head in SD 2.1 (320 channels / 5 "
            "heads, 640/10, 1280/20); the weighted sum of the values is added back to the position's features (a residual "
            f"add). Our run: {N_TOK} tokens; the largest level is 64×64 = {POSITIONS:,} positions. Drawn here on an 8×8 "
            "grid over the seed-0 picture (SD 2.1-base, 50 steps, guidance 7.5) for clarity: the denoiser works on the "
            "64×64×4 latent, not the picture. The 13 weights for the tower position are an ILLUSTRATION (our run saved no "
            "attention maps). Self-attention: Episode 8."))
        h, xp, yp = 3.3, -4.25, -0.85
        pic = picture(SEEDS[0], h).move_to([xp, yp, 0])
        rim = m.Rectangle(width=h + 0.1, height=h + 0.1).set_stroke(EDGE, 2.5).move_to(pic)
        csz = h / 8
        ul = pic.get_corner(m.UL)
        cells = m.VGroup(*[m.Square(csz).set_stroke(INK, 1.2, opacity=0.28).set_fill(PROMPT, 0)
                           .move_to(ul + np.array([(c + 0.5) * csz, -(r + 0.5) * csz, 0])) for r in range(8) for c in range(8)])
        me = cells[CELL[0] * 8 + CELL[1]]
        p0 = me.get_center()
        hi = m.Square(csz).set_stroke(DATA, 5).move_to(p0).set_z_index(3)
        others = [(0, 1), (1, 6), (4, 2), (6, 6), (5, 0), (7, 3)]
        arcs = m.VGroup(*[m.VGroup(m.ArcBetweenPoints(p0, cells[r * 8 + c].get_center(), angle=-0.45).set_stroke(DATA, 3, opacity=0.9),
                                   m.Dot(cells[r * 8 + c].get_center(), radius=0.06, color=DATA))
                          for r, c in others])
        sa_l = text("self-attention (Episode 8): pixels look at pixels", 20, DATA).move_to([xp + 0.9, -2.85, 0])
        ca_l = text("cross-attention: pixels look at the words", 20, PROMPT, weight=m.BOLD).move_to(sa_l).align_to(rim, m.LEFT)
        sa_l.align_to(rim, m.LEFT)
        row = chip_row(SHOWN, 1.95)
        assert row.get_left()[0] > -6.6 and row.get_right()[0] < 6.6, row.width
        fan = m.VGroup(*[m.Line(p0, c.get_bottom() + [0, -0.04, 0]).set_stroke(PROMPT, 2, opacity=0.55) for c in row])
        heavy = m.VGroup(*[m.Line(p0, c.get_bottom() + [0, -0.04, 0]).set_stroke(PROMPT, 1.2 + 14 * w, opacity=0.3 + 0.7 * w / max(ILLU_W))
                           for c, w in zip(row, ILLU_W)])
        bars = m.VGroup(*[m.Rectangle(width=0.32, height=max(0.03, 0.9 * w)).set_stroke(PROMPT, 1.5).set_fill(PINK_FILL, 1)
                          .move_to([c.get_x(), 0, 0]).align_to([0, c.get_top()[1] + 0.06, 0], m.DOWN) for c, w in zip(row, ILLU_W)])
        assert bars.get_top()[1] < 2.82, bars.get_top()

        src = ("q = W_Q · pixel        # a question\n"
               "k = W_K · token        # to match\n"
               "v = W_V · token        # to pass on\n"
               "w = softmax(q·k / √64) # sums to 1\n"
               "pixel += Σ w·v")
        rules = {r"#[^\n]*": MUTED, r"\bq\b|pixel": DATA, r"\b[kv]\b|token": PROMPT, r"\bw\b": INK}
        lines = src.split("\n")
        body = m.VGroup(*[code(s, 20, INK, t2c=rules) for s in lines]).arrange(m.DOWN, buff=0.13, aligned_edge=m.LEFT)
        title = code("cross-attention · one position", 20, MUTED)
        fbox = card(max(body.width, title.width) + 0.7, body.height + title.height + 0.62, EDGE, CARD, radius=0.16, stroke=2)
        m.VGroup(title, body).arrange(m.DOWN, buff=0.18, aligned_edge=m.LEFT).move_to(fbox).align_to(fbox, m.LEFT).shift(0.35 * m.RIGHT)
        formula = m.VGroup(fbox, title, body).move_to([3.35, -1.3, 0])
        assert formula.get_right()[0] < 6.6 and formula.get_left()[0] > rim.get_right()[0] + 0.3, (formula.get_left(), formula.get_right())
        # the fan of lines must pass above the card
        for ln in fan:
            a_, b_ = ln.get_start(), ln.get_end()
            x_ = formula.get_left()[0]
            if b_[0] > x_:
                yy = a_[1] + (b_[1] - a_[1]) * (x_ - a_[0]) / (b_[0] - a_[0])
                assert yy > formula.get_top()[1] + 0.15, (yy, formula.get_top())
        illu = m.VGroup(cite("weights: an illustration (our run saved no attention maps)"),
                        cite("picture: generated on this Mac · SD 2.1-base · 50 steps · seed 0"))
        illu.arrange(m.DOWN, buff=0.08, aligned_edge=m.RIGHT).next_to(formula, m.DOWN, buff=0.1).align_to(formula, m.RIGHT)
        assert illu.get_bottom()[1] > -3.55, illu.get_bottom()
        pos_l = text(f"{POSITIONS:,} positions at the largest size ({SIZES[0]} × {SIZES[0]})\ndrawn here at 8 × 8", 20, PROMPT)
        pos_l.move_to([0, -2.98, 0]).align_to(rim, m.LEFT)
        assert pos_l.get_right()[0] < formula.get_left()[0] - 0.2 or pos_l.get_bottom()[1] < formula.get_bottom()[1], pos_l.get_right()

        self.say(
            "[Now, how does the denoiser listen]? [Episode eight's attention] [let every pixel look at the others].",
            m.AnimationGroup(self.new_header("04 · cross-attention", "Pixels ask, words answer"), m.FadeIn(pic), m.FadeIn(rim)),
            m.AnimationGroup(m.FadeIn(cells), m.FadeIn(hi)),
            m.AnimationGroup(m.LaggedStart(*[m.Create(a_) for a_ in arcs], lag_ratio=0.12), m.FadeIn(sa_l)),
        )
        self.say(
            "[Cross-attention lets each pixel look at the words instead].",
            m.AnimationGroup(m.FadeOut(arcs), m.FadeOut(sa_l), m.FadeIn(ca_l),
                             m.LaggedStart(*[m.FadeIn(c, shift=0.15 * m.DOWN) for c in row], lag_ratio=0.05),
                             m.LaggedStart(*[m.Create(f_) for f_ in fan], lag_ratio=0.05)),
        )
        self.say(
            "[Each image position asks with a query]; [each token answers with a key], [and offers a value].",
            m.AnimationGroup(m.FadeIn(fbox), m.FadeIn(title), m.FadeIn(body[0], shift=0.1 * m.UP), m.Indicate(hi, color=DATA, scale_factor=1.25)),
            m.AnimationGroup(m.FadeIn(body[1], shift=0.1 * m.UP), m.LaggedStart(*[m.Indicate(c[0], color=PROMPT, scale_factor=1.1) for c in row], lag_ratio=0.04)),
            m.FadeIn(body[2], shift=0.1 * m.UP),
        )
        self.say(
            "[Query against key gives each token a score], "
            "[and softmax turns the scores into weights that add up to one].",
            m.LaggedStart(*[m.ShowPassingFlash(f_.copy().set_stroke(PROMPT, 5, opacity=1), time_width=0.6) for f_ in fan], lag_ratio=0.05, run_time=1.4),
            m.AnimationGroup(m.FadeIn(body[3], shift=0.1 * m.UP), m.FadeIn(illu),
                             *[m.Transform(f_, g_) for f_, g_ in zip(fan, heavy)],
                             m.LaggedStart(*[m.GrowFromEdge(b_, m.DOWN) for b_ in bars], lag_ratio=0.05)),
        )
        self.say(
            "[The weighted sum of the values] [is added back into that position].",
            m.LaggedStart(*[m.ShowPassingFlash(m.Line(c.get_bottom() + [0, -0.04, 0], p0).set_stroke(PROMPT, 3 + 12 * w), time_width=0.7)
                            for c, w in zip(row, ILLU_W)], lag_ratio=0.03, run_time=1.3),
            m.AnimationGroup(m.FadeIn(body[4], shift=0.1 * m.UP), m.Flash(hi, color=PROMPT, flash_radius=0.42, line_length=0.18),
                             me.animate(rate_func=m.there_and_back, run_time=0.9).set_fill(PROMPT, 0.75)),
        )
        self.say(
            "[Every position does this at once]: [four thousand and ninety-six of them, at the largest size].",
            m.LaggedStart(*[c.animate(rate_func=m.there_and_back, run_time=0.5).set_fill(PROMPT, 0.6) for c in cells],
                          lag_ratio=0.03, run_time=2.2),
            m.AnimationGroup(m.FadeOut(ca_l), m.FadeIn(pos_l)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 4b · in the denoiser, and where words land

    def where_words_land(self):
        self.next_section("4b · Where words land", notes=(
            f"SD 2.1-base's denoiser (unet/config.json of the model we ran): cross-attention in the down blocks at 64, 32, "
            f"16 ({DOWN_X[:3]} transformer blocks), the middle block at 8×8 (1) and the up blocks at 16, 32, 64 "
            f"({UP_X[2::-1]}): {N_XATTN} cross-attention layers; channels 320/640/1280/1280, keys and values from the "
            f"{TE['width']}-wide text vectors. With guidance, each of the {STEPS} steps runs it twice. The three word maps are "
            "HAND-DRAWN illustrations on the seed-0 lighthouse (blobs placed by eye, pooled to 16×16): our run saved no "
            "attention maps. Real maps are averaged over heads, layers and steps (notes §15) and are not explanations. "
            "Prompt-to-Prompt: Hertz et al., arXiv:2208.01626 (Aug 2022; ICLR 2023)."))
        unet = UNet(sizes=tuple(SIZES), channels=UNET_CFG["block_out_channels"], center=(-0.45, -0.55), scale=0.74, size=22)
        assert unet.get_left()[0] > -6.6 and unet.get_right()[0] < 6.6, (unet.get_left(), unet.get_right())
        top_y = 2.1
        pbar = m.VGroup(card(unet.get_right()[0] - unet.down[0].get_left()[0] + 0.3, 0.55, PROMPT, PINK_FILL, radius=0.12, stroke=2.2),
                        code(f"prompt vectors · {N_TOK} × {TE['width']:,}", 20, INK))
        pbar[0].move_to([(unet.down[0].get_left()[0] + unet.get_right()[0]) / 2, top_y, 0])
        pbar[1].move_to(pbar[0])
        targets = [(unet.down[k], DOWN_X[k]) for k in range(3)] + [(unet.mid, MID_X)] + [(unet.up[k], UP_X[k]) for k in range(3)]
        drops, badges = m.VGroup(), m.VGroup()
        for blk, n_ in targets:
            start = np.array([blk.get_x(), pbar[0].get_bottom()[1] - 0.04, 0])
            drops.add(m.Arrow(start, blk.get_top() + [0, 0.04, 0], buff=0.02, color=PROMPT, stroke_width=3,
                              max_tip_length_to_length_ratio=0.08))
            bd = code(f"×{n_}", 20, PROMPT, weight=m.BOLD)
            if blk is unet.mid:
                bd.next_to(blk, m.DOWN, buff=0.08)
            elif any(blk is u_ for u_ in unet.up):
                bd.next_to(blk.get_corner(m.UL), m.LEFT, buff=0.06)
            else:
                bd.next_to(blk.get_corner(m.UR), m.RIGHT, buff=0.06)
            badges.add(bd)
        total = m.VGroup(title_text(str(N_XATTN), 60, PROMPT), text("cross-attention\nlayers", 20, MUTED)).arrange(m.RIGHT, buff=0.18)
        total.move_to([5.2, 0.35, 0])
        per = code(f"× {STEPS} steps", 22, TIME).next_to(total, m.DOWN, buff=0.3).align_to(total, m.LEFT)
        assert total.get_right()[0] < 6.6 and total.get_left()[0] > unet.up[0].get_right()[0] + 0.35, (total.get_left(), unet.up[0].get_right())

        words = ["lighthouse", "rocky", "dusk"]
        hm = 2.75
        xs = [-4.35, 0.0, 4.35]
        maps = m.Group()
        for wd, x in zip(words, xs):
            base = framed(picture(overlay(SEEDS[0], np.zeros((256, 256))), hm), PROMPT, stroke=2).move_to([x, -0.15, 0])
            glow = picture(overlay(SEEDS[0], word_map(wd)), hm).move_to(base[1])      # the same picture, lit where the word lands
            chip_ = tchip(wd, 22).next_to(base, m.UP, buff=0.2)
            maps.add(m.Group(base, glow, chip_))
        illu = text("illustration: sketched by hand on our seed-0 picture · our run saved no attention maps", 20, MUTED)
        illu.next_to(maps, m.DOWN, buff=0.22)
        quote = text("“the cross-attention layers are the key to controlling the relation between\n"
                     "the spatial layout of the image to each word in the prompt”", 22, INK)
        qsrc = cite("Hertz et al., 2022 · Prompt-to-Prompt").next_to(quote, m.DOWN, buff=0.12).align_to(quote, m.RIGHT)
        qg = m.VGroup(quote, qsrc).move_to([0, -2.85, 0])
        assert maps.get_top()[1] < 2.6 and qg.get_bottom()[1] > -3.55 and illu.get_bottom()[1] > qg.get_top()[1] + 0.1, \
            (maps.get_top(), qg.get_bottom(), illu.get_bottom(), qg.get_top())

        self.say(
            "[In our denoiser], [this happens in sixteen layers, at four sizes], [and again at every step].",
            m.AnimationGroup(self.new_header("04 · cross-attention", "Sixteen layers, every step"), m.FadeIn(unet)),
            m.AnimationGroup(m.FadeIn(pbar, shift=0.1 * m.DOWN), m.LaggedStart(*[m.GrowArrow(d_) for d_ in drops], lag_ratio=0.1),
                             m.LaggedStart(*[m.FadeIn(b_) for b_ in badges], lag_ratio=0.1), m.FadeIn(total, shift=0.1 * m.LEFT)),
            m.AnimationGroup(unet.flow(PROMPT, run_time=1.8), m.FadeIn(per, shift=0.1 * m.UP)),
        )
        group_u = m.VGroup(unet, pbar, drops, badges, total, per)
        self.say(
            "[Collect one word's weights over every position], [and you get a map]. "
            "[Sketched by hand]: [lighthouse on the tower], [rocky on the rocks], [dusk in the sky].",
            m.AnimationGroup(self.new_header("04 · cross-attention", "Where words land"), m.FadeOut(group_u),
                             *[m.FadeIn(mp[0]) for mp in maps], *[m.FadeIn(mp[2], shift=0.1 * m.DOWN) for mp in maps]),
            m.LaggedStart(*[m.Indicate(mp[2], color=PROMPT, scale_factor=1.1) for mp in maps], lag_ratio=0.2),
            m.FadeIn(illu),
            m.FadeIn(maps[0][1], run_time=1.0),
            m.FadeIn(maps[1][1], run_time=1.0),
            m.FadeIn(maps[2][1], run_time=1.0),
        )
        self.say(
            "[A twenty twenty-two paper, Prompt-to-Prompt], [showed that these maps decide where each word lands in the picture].",
            m.FadeIn(qsrc),
            m.FadeIn(quote, shift=0.1 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 5 · scale

    def scale(self):
        self.next_section("5 · Bigger readers", notes=(
            f"Text-encoder sizes (notes §15, §11): SD 1.x CLIP ViT-L/14 {SD1_PARAMS / 1e6:.1f} M (Aug 2022); SD 2.1 "
            f"OpenCLIP ViT-H text {TE_PARAMS / 1e6:.1f} M (Dec 2022; ours, counted in our run); Imagen (Saharia et al., "
            "May 2022): frozen T5-XXL, 4.6 B, text-only; SDXL (Podell et al., Jul 2023): CLIP ViT-L + OpenCLIP ViT-bigG, "
            f"'a total size of 817M', context 77 × 2048 (bigG ≈ {SDXL_BIGG_B * 1000:.0f} M, derived); SD3 (Esser et al., "
            f"Mar 2024): CLIP-L + OpenCLIP bigG + T5-XXL ('4.7B'), context 154 × 4096; total ≈ {SD3_B:.1f} B [derived: "
            "SDXL's two + 4.7 B]. Imagen's quote: abstract. Dates as in notes §1/§17 (October 2026)."))
        k = 0.95                                   # scene units per billion parameters
        x0 = -2.05
        rows = [
            ("Stable Diffusion 1", "Aug 2022", [(SD1_PARAMS / 1e9, PROMPT)], "CLIP ViT-L/14", f"{SD1_PARAMS / 1e6:.0f} M", "77 × 768"),
            ("Stable Diffusion 2.1", "Dec 2022 · ours", [(TE_PARAMS / 1e9, PROMPT)], "OpenCLIP ViT-H/14", f"{TE_PARAMS / 1e6:.0f} M", "77 × 1024"),
            ("Imagen (Google)", "May 2022", [(T5_IMAGEN_B, SHADE_T5)], "T5-XXL · text only", f"{T5_IMAGEN_B:.1f} B", ""),
            ("SDXL", "Jul 2023", [(SD1_PARAMS / 1e9, PROMPT), (SDXL_BIGG_B, SHADE_BIGG)],
             "CLIP ViT-L + OpenCLIP ViT-bigG", f"{SDXL_B * 1000:.0f} M", f"77 × {SDXL_WIDTH}"),
            ("Stable Diffusion 3", "Mar 2024", [(SD1_PARAMS / 1e9, PROMPT), (SDXL_BIGG_B, SHADE_BIGG), (T5_SD3_B, SHADE_T5)],
             "two CLIPs + T5-XXL", f"≈ {SD3_B:.1f} B", "154 × 4096"),
        ]
        ys = [1.75, 0.9, 0.05, -0.8, -1.65]
        col_h = m.VGroup(code("text encoder size", 20, MUTED).move_to([0, 2.38, 0]).align_to([x0, 0, 0], m.LEFT),
                         code("tokens × width", 20, MUTED).move_to([0, 2.38, 0]).align_to([6.5, 0, 0], m.RIGHT))
        built = []
        for (name, date, segs, enc, val, ctx), y in zip(rows, ys):
            nm = text(name, 22, INK, weight=m.BOLD).move_to([0, y + 0.17, 0]).align_to([-6.5, 0, 0], m.LEFT)
            dt = code(date, 20, PROMPT if "ours" in date else MUTED).move_to([0, y - 0.2, 0]).align_to([-6.5, 0, 0], m.LEFT)
            bars, x = m.VGroup(), x0
            for b_, col in segs:
                w = max(b_ * k, 0.04)
                bars.add(m.Rectangle(width=w, height=0.36).set_stroke(col, 2).set_fill(mix(CARD, col, 0.55), 1).move_to([x + w / 2, y + 0.17, 0]))
                x += w
            vl = code(val, 20, INK, weight=m.BOLD).next_to(bars, m.RIGHT, buff=0.15)
            en = code(enc, 20, MUTED).move_to([0, y - 0.2, 0]).align_to([x0, 0, 0], m.LEFT)
            cx_ = code(ctx, 20, INK).move_to([0, y + 0.17, 0]).align_to([6.5, 0, 0], m.RIGHT) if ctx else m.VGroup()
            assert nm.get_right()[0] < x0 - 0.2 and dt.get_right()[0] < x0 - 0.2, name
            if ctx:
                assert vl.get_right()[0] < cx_.get_left()[0] - 0.25, (name, vl.get_right(), cx_.get_left())
            built.append(dict(nm=nm, dt=dt, bars=bars, vl=vl, en=en, cx=cx_))
        ours_box = m.SurroundingRectangle(m.VGroup(built[1]["nm"], built[1]["dt"], built[1]["cx"], built[1]["bars"]), buff=0.1)
        ours_box.set_stroke(PROMPT, 2, opacity=0.8)
        quote = text("“increasing the size of the language model in Imagen boosts both sample fidelity and\n"
                     "image-text alignment much more than increasing the size of the image diffusion model”", 22, INK)
        qsrc = cite("Saharia et al., 2022 · Imagen").next_to(quote, m.DOWN, buff=0.1).align_to(quote, m.RIGHT)
        qg = m.VGroup(quote, qsrc).move_to([0, -2.85, 0])
        assert qg.width < 13.1 and qg.get_top()[1] < built[-1]["en"].get_bottom()[1] - 0.1, (qg.width, qg.get_top())

        def show(i, extra=()):
            b = built[i]
            return m.AnimationGroup(m.FadeIn(b["nm"]), m.FadeIn(b["dt"]), m.LaggedStart(*[m.GrowFromEdge(s, m.LEFT) for s in b["bars"]], lag_ratio=0.5),
                                    m.FadeIn(b["vl"]), m.FadeIn(b["en"]), m.FadeIn(b["cx"]), *extra)

        self.say(
            "[Text encoders keep growing]: [from a hundred twenty-three million parameters in Stable Diffusion one], "
            "[to our three hundred forty million].",
            m.AnimationGroup(self.new_header("05 · scale", "Bigger readers"), m.FadeIn(col_h)),
            show(0),
            show(1, (m.Create(ours_box),)),
        )
        self.say(
            "[In twenty twenty-two, Google's Imagen read prompts with T5], [a language model trained on text alone]: "
            "[four point six billion parameters].",
            m.AnimationGroup(m.FadeIn(built[2]["nm"]), m.FadeIn(built[2]["dt"])),
            m.FadeIn(built[2]["en"]),
            m.AnimationGroup(m.GrowFromEdge(built[2]["bars"][0], m.LEFT, run_time=1.2), m.FadeIn(built[2]["vl"])),
        )
        self.say(
            "[Its finding]: [a bigger text encoder helped the pictures more than a bigger denoiser].",
            m.FadeIn(qsrc),
            m.FadeIn(quote, shift=0.1 * m.UP),
        )
        self.say(
            "[Then SDXL, in twenty twenty-three], [joined two CLIP encoders side by side]: "
            "[vectors twice as wide as ours].",
            m.AnimationGroup(m.FadeIn(built[3]["nm"]), m.FadeIn(built[3]["dt"])),
            m.AnimationGroup(m.LaggedStart(*[m.GrowFromEdge(s, m.LEFT) for s in built[3]["bars"]], lag_ratio=0.6),
                             m.FadeIn(built[3]["en"]), m.FadeIn(built[3]["vl"])),
            m.FadeIn(built[3]["cx"], shift=0.1 * m.LEFT),
        )
        self.say(
            "[And Stable Diffusion three, in twenty twenty-four, uses three]: [two CLIPs, plus T5].",
            m.AnimationGroup(m.FadeIn(built[4]["nm"]), m.FadeIn(built[4]["dt"]), m.FadeIn(built[4]["cx"])),
            m.AnimationGroup(m.LaggedStart(*[m.GrowFromEdge(s, m.LEFT) for s in built[4]["bars"]], lag_ratio=0.5, run_time=1.6),
                             m.FadeIn(built[4]["en"]), m.FadeIn(built[4]["vl"])),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ 6 · sensitivity

    def sensitivity(self):
        self.next_section("6 · Same words, different pictures", notes=(
            "Generated on this Mac: SD 2.1-base, 50 steps, guidance 7.5, seeds 0–3, the same prompt (mac_runs.json "
            "lighthouse.guidance_sweep). Rings mark where the tower landed, placed by eye. Our runs have only this one "
            "text-to-image prompt. The ignored prompt: SDEdit (Episode 16) from the seed-0 picture at strength 0.9 "
            f"({SDE_09['steps_run']} steps, guidance 7.5) with the prompt '{SDE['prompt']}': the layout stays and no snow "
            "appears (mac_runs.md §5; a more explicit prompt did add snow — Episode 16)."))
        hh = 2.55
        xs = [-4.65, -1.55, 1.55, 4.65]
        yp = 0.3
        pics = m.Group(*[framed(picture(p, hh)).move_to([x, yp, 0]) for p, x in zip(SEEDS, xs)])
        seeds = m.VGroup(*[code(f"seed {s}", 20, MUTED).next_to(pics[s], m.DOWN, buff=0.14) for s in range(4)])
        ptxt = code(f'"{PROMPT_TEXT}"', 22, PROMPT).move_to([0, 2.15, 0])
        rings = m.VGroup()
        for s in range(4):
            u, v, w_, h_ = TOWER[s]
            img = pics[s][1]
            ctr = img.get_corner(m.UL) + np.array([u * hh, -v * hh, 0])
            rings.add(m.Ellipse(width=w_ * hh, height=h_ * hh).set_stroke(PROMPT, 3.5).move_to(ctr))
        cap = cite("generated on this Mac · Stable Diffusion 2.1-base · 50 steps · guidance 7.5 · same 13 tokens").move_to([0, -3.3, 0])
        what = m.VGroup(text("the words: what to paint", 24, PROMPT, weight=m.BOLD), text("the noise: where, and how big", 24, NOISE, weight=m.BOLD))
        what.arrange(m.RIGHT, buff=1.2).move_to([0, -2.1, 0])
        assert pics.get_left()[0] > -6.6 and pics.get_right()[0] < 6.6 and what.width < 12.5

        hb = 2.8
        src_p = framed(picture(SEEDS[0], hb)).move_to([-3.85, -0.35, 0])
        out_p = framed(picture(LIMG / SDE_09["file"], hb), RED).move_to([3.85, -0.35, 0])
        go = arrow_to(src_p.get_right(), out_p.get_left(), PROMPT, 5, 0.25)
        ask = m.VGroup(text("repaint, strength 0.9", 22, INK, weight=m.BOLD), code('"… in winter, snow, …"', 20, PROMPT))
        ask.arrange(m.DOWN, buff=0.12).next_to(go, m.UP, buff=0.2)
        none_l = text("no snow", 30, RED, weight=m.BOLD).next_to(out_p, m.DOWN, buff=0.18)
        src_l = code("seed 0", 20, MUTED).next_to(src_p, m.DOWN, buff=0.18)
        ep16 = cite("SDEdit, Episode 16 · generated on this Mac · 45 steps · guidance 7.5").move_to([0, -3.3, 0])
        assert ask.get_left()[0] > src_p.get_right()[0] + 0.15 and ask.get_right()[0] < out_p.get_left()[0] - 0.15

        self.say(
            "[Same thirteen tokens], [four different starting noises]: [four different lighthouses].",
            m.AnimationGroup(self.new_header("06 · sensitivity", "Same words, different pictures"), m.FadeIn(ptxt)),
            m.AnimationGroup(m.FadeIn(seeds), m.FadeIn(cap)),
            m.LaggedStart(*[m.FadeIn(p, shift=0.15 * m.UP) for p in pics], lag_ratio=0.2, run_time=1.6),
        )
        self.say(
            "[The words fix what to paint]; [the noise picks where], [and how big].",
            m.FadeIn(what[0], shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(what[1], shift=0.1 * m.UP), m.LaggedStart(*[m.Create(r_) for r_ in rings], lag_ratio=0.15)),
            m.LaggedStart(*[m.Indicate(r_, color=PROMPT, scale_factor=1.15) for r_ in rings], lag_ratio=0.1),
        )
        self.say(
            "[Words can also be ignored]: [in episode sixteen, a repaint asked for winter and snow], [and got none].",
            m.AnimationGroup(m.FadeOut(pics), m.FadeOut(seeds), m.FadeOut(rings), m.FadeOut(what), m.FadeOut(cap), m.FadeOut(ptxt),
                             m.FadeIn(src_p), m.FadeIn(src_l), m.FadeIn(ep16)),
            m.AnimationGroup(m.GrowArrow(go), m.FadeIn(ask), m.FadeIn(out_p)),
            m.FadeIn(none_l, scale=0.8),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes=(
            f"Split (a tokenizer: {N_TOK} tokens for the lighthouse prompt, vocabulary {VOCAB:,}, room for {CONTEXT}), "
            f"encode (a frozen, contrastively trained text encoder: {N_TOK} × {TE['width']} numbers, {TXT_S * 1000:.0f} ms on "
            f"this Mac), attend (cross-attention in {N_XATTN} layers of the denoiser, at every step). Next: Episode 16, "
            "Editing and Control."))
        halo = glow(PROMPT, radius=4.0)
        words = m.VGroup(*[title_text(w, 58, c) for w, c in (("Split", PROMPT), ("Encode", PROMPT), ("Attend", MODEL))])
        subs = m.VGroup(*[code(s, 20, MUTED) for s in (f"{N_TOK} tokens", f"{N_TOK} × {TE['width']:,} numbers",
                                                        f"{N_XATTN} layers · every step")])
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
        row = m.VGroup(words, dots, subs).move_to(0.85 * m.UP)
        assert row.width < 12.6
        tag = text("The prompt is read once; the denoiser listens at every step.", 28, INK).next_to(row, m.DOWN, buff=0.7)

        chain = [("prompt", PROMPT, PINK_FILL), (f"{N_TOK} tokens", PROMPT, PINK_FILL),
                 (f"{N_TOK} × {TE['width']:,}", PROMPT, PINK_FILL), ("denoiser", MODEL, AMBER_FILL)]
        pills = m.VGroup(*[pill(s, c, f, 20, mono=True) for s, c, f in chain]).arrange(m.RIGHT, buff=0.55)
        links = m.VGroup(*[arrow_to(a.get_right(), b.get_left(), MUTED, 3, 0.08) for a, b in zip(pills, pills[1:])])
        mascot = pixel(0.42)
        foot = m.VGroup(m.VGroup(pills, links), mascot).arrange(m.RIGHT, buff=0.5).next_to(tag, m.DOWN, buff=0.9)
        assert foot.width < 12.8 and foot.get_bottom()[1] > -3.5, (foot.width, foot.get_bottom())

        self.say(
            "[To sum up]: [split], [encode], [attend].",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.3 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(dots[0]), m.FadeIn(words[1], shift=0.3 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(dots[1]), m.FadeIn(words[2], shift=0.3 * m.UP), m.FadeIn(subs[2])),
        )
        self.say(
            "[The prompt is read once]; [the denoiser listens to it in sixteen layers, at every step].",
            m.Write(tag),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(p_, shift=0.1 * m.RIGHT) for p_ in pills], lag_ratio=0.25),
                             m.LaggedStart(*[m.GrowArrow(l_) for l_ in links], lag_ratio=0.25)),
        )
        self.say(
            "[Next, we keep the words], [and start editing the picture].",
            m.FadeIn(mascot, shift=0.2 * m.UP),
            m.Indicate(pills[0], color=PROMPT, scale_factor=1.1),
        )
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, tag, foot)], run_time=0.9)
