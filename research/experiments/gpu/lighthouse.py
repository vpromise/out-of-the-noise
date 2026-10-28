"""The lighthouse: a real latent text-to-image model run locally with MLX.

Pipeline: ml-explore/mlx-examples stable_diffusion (pinned commit in gpu/third_party/mlx-examples, see mac_runs.md),
models: sd2-community/stable-diffusion-2-1-base (mirror of Stability's SD 2.1-base, CreativeML Open RAIL++-M) and
stabilityai/sd-turbo (ADD-distilled from SD 2.1, Stability AI Community License). UNet + text encoder in float16,
VAE in float32 (the pipeline's defaults). Run with HF_HUB_OFFLINE=1 after gpu/download_lighthouse_models.py.

  HF_HUB_OFFLINE=1 python gpu/lighthouse.py [stage ...]
  stages: load timing steps guidance turbo frames sdedit vae   (default: all, in that order)

The denoising loop below is a transcription of StableDiffusion.generate_latents/_denoising_step from the pinned
mlx-examples code (same seeding, sampler and CFG formula: eps = eps_neg + w (eps_text - eps_neg), CFG off for w <= 1),
instrumented to time each stage and to expose eps for x0-hat previews. `check` verifies it bit-matches the library.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "third_party", "mlx-examples", "stable_diffusion"))
import mlx.core as mx  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402
from PIL import Image  # noqa: E402

from common import ARRAYS, IMAGES, LOGS, read_result, write_result  # noqa: E402
from stable_diffusion import StableDiffusion, model_io  # noqa: E402
from stable_diffusion.sampler import SimpleEulerAncestralSampler  # noqa: E402

SD21 = "sd2-community/stable-diffusion-2-1-base"
TURBO = "stabilityai/sd-turbo"
FILES = {
    "unet_config": "unet/config.json", "unet": "unet/diffusion_pytorch_model.fp16.safetensors",
    "text_encoder_config": "text_encoder/config.json", "text_encoder": "text_encoder/model.fp16.safetensors",
    "vae_config": "vae/config.json", "vae": "vae/diffusion_pytorch_model.safetensors",
    "diffusion_config": "scheduler/scheduler_config.json",
    "tokenizer_vocab": "tokenizer/vocab.json", "tokenizer_merges": "tokenizer/merges.txt",
}
model_io._MODELS[SD21] = dict(FILES)
model_io._MODELS[TURBO] = dict(FILES)
# pinned Hugging Face revisions (commit SHAs) = exactly what gpu/download_lighthouse_models.py fetched
REVISIONS = {SD21: "4e63672c03103b6c636b8fb4119ba982469b2955", TURBO: "b261bac6fd2cf515557d5d0707481eafa0485ec2"}
_hf_download = model_io.hf_hub_download
model_io.hf_hub_download = lambda repo, filename, **kw: _hf_download(repo, filename, revision=REVISIONS[repo], **kw)
# the fp16 text-encoder files also store the non-learned `position_ids` buffer (= arange(77)); skip it
_map_clip = model_io.map_clip_text_encoder_weights
model_io.map_clip_text_encoder_weights = lambda k, v: [] if k.endswith("position_ids") else _map_clip(k, v)
# the mirror's fp32 VAE uses diffusers' legacy attention names (query/key/value/proj_attn); same Linear layers as
# to_q/to_k/to_v/to_out.0 (sd-turbo's file), so rename before the pipeline's own mapping
import re  # noqa: E402
# sd-turbo's unet/config.json writes transformer_layers_per_block as the int 1; the pipeline wants one per block
_UNetConfig = model_io.UNetConfig


def _unet_config(**kw):
    t = kw.get("transformer_layers_per_block")
    if isinstance(t, int):
        kw["transformer_layers_per_block"] = (t,) * len(kw["block_out_channels"])
    return _UNetConfig(**kw)


model_io.UNetConfig = _unet_config
_LEGACY = {"query": "to_q", "key": "to_k", "value": "to_v", "proj_attn": "to_out.0"}
_map_vae = model_io.map_vae_weights
model_io.map_vae_weights = lambda k, v: _map_vae(
    re.sub(r"(attentions\.\d+\.)(query|key|value|proj_attn)\.", lambda m: m.group(1) + _LEGACY[m.group(2)] + ".", k), v)

PROMPT = "a lighthouse on a rocky coast at dusk, oil painting"
PROMPT_WINTER = "a lighthouse on a rocky coast at dusk in winter, snow, oil painting"
SEEDS = [0, 1, 2, 3]
LAT = 64  # latent side for 512x512 images
LH = os.path.join(IMAGES, "lighthouse")
reset_peak = getattr(mx, "reset_peak_memory", None) or mx.metal.reset_peak_memory
get_peak = getattr(mx, "get_peak_memory", None) or mx.metal.get_peak_memory
get_active = getattr(mx, "get_active_memory", None) or mx.metal.get_active_memory
_sd = {}
RES = read_result("lighthouse") or {}


def save_res():
    write_result("lighthouse", RES)


def load(repo):
    if repo not in _sd:
        mx.clear_cache()
        a0 = get_active()
        t0 = time.perf_counter()
        sd = StableDiffusion(repo, float16=True)
        sd.ensure_models_are_loaded()
        dt = time.perf_counter() - t0
        if repo == TURBO:
            sd.sampler = SimpleEulerAncestralSampler(sd.diffusion_config)  # as mlx-examples does for SDXL-Turbo
        _sd[repo] = sd
        parts = {}
        for name, mod in [("unet", sd.unet), ("text_encoder", sd.text_encoder), ("vae", sd.autoencoder)]:
            flat = tree_flatten(mod.parameters())
            parts[name] = dict(params=int(sum(v.size for _, v in flat)), dtype=str(flat[0][1].dtype).split(".")[-1],
                               bytes_in_memory=int(sum(v.nbytes for _, v in flat)))
        enc = sum(v.size for k, v in tree_flatten(sd.autoencoder.parameters()) if k.startswith(("encoder", "quant_proj")))
        parts["vae"]["encoder_params"] = int(enc)
        parts["vae"]["decoder_params"] = parts["vae"]["params"] - int(enc)
        RES.setdefault("models", {})[repo] = dict(load_s_first=dt, parts=parts,
                                                   weights_active_gb=(get_active() - a0) / 1e9,
                                                   sampler=type(sd.sampler).__name__)
        save_res()
        print("loaded", repo, round(dt, 1), "s", {k: v["params"] for k, v in parts.items()}, flush=True)
    return _sd[repo]


def to_img(x):
    """decoded [1,H,W,3] in [0,1] -> uint8 HxWx3"""
    return (np.array(x[0].astype(mx.float32)) * 255).round().clip(0, 255).astype(np.uint8)


def generate(sd, prompt, steps, cfg, seed, record=False, x0_latent=None, strength=None):
    """Returns dict(img, latent, timing, [frames]). If x0_latent is given: SDEdit from that latent at `strength`."""
    tm = {}
    mx.random.seed(seed)
    t0 = time.perf_counter()
    cond = sd._get_text_conditioning(prompt, 1, cfg, "")
    mx.eval(cond)
    tm["text_encoder_s"] = time.perf_counter() - t0
    if x0_latent is None:
        start = sd.sampler.max_time
        x = sd.sampler.sample_prior((1, LAT, LAT, sd.autoencoder.latent_channels), dtype=sd.dtype)
    else:
        start = sd.sampler.max_time * strength
        steps = int(steps * strength)
        x = sd.sampler.add_noise(x0_latent, mx.array(start))
    mx.eval(x)
    x_start = x
    frames = dict(t=[], sigma=[], x_t=[np.array(x[0].astype(mx.float32))], x0_hat=[])
    step_s = []
    for t, t_prev in sd.sampler.timesteps(steps, start_time=start, dtype=sd.dtype):
        ts = time.perf_counter()
        x_in = mx.concatenate([x] * 2, axis=0) if cfg > 1 else x
        eps = sd.unet(x_in, mx.broadcast_to(t, [len(x_in)]), encoder_x=cond, text_time=None)
        if cfg > 1:
            e_text, e_neg = eps.split(2)
            eps = e_neg + cfg * (e_text - e_neg)
        if record:
            s = sd.sampler.sigmas(t).astype(eps.dtype)
            x0h = (s.square() + 1).sqrt() * x - s * eps
        x = sd.sampler.step(eps, x, t, t_prev)
        mx.eval(x)
        step_s.append(time.perf_counter() - ts)
        if record:
            frames["t"].append(float(t)); frames["sigma"].append(float(sd.sampler.sigmas(t)))
            frames["x_t"].append(np.array(x[0].astype(mx.float32))); frames["x0_hat"].append(np.array(x0h[0].astype(mx.float32)))
    td = time.perf_counter()
    img = sd.decode(x)
    mx.eval(img)
    tm["vae_decode_s"] = time.perf_counter() - td
    tm["denoise_steps"] = len(step_s)
    tm["unet_passes_per_step"] = 2 if cfg > 1 else 1
    tm["step_s"] = [round(v, 4) for v in step_s]
    tm["denoise_total_s"] = float(sum(step_s))
    tm["total_s"] = tm["text_encoder_s"] + float(np.sum(step_s)) + tm["vae_decode_s"]
    out = dict(img=to_img(img), latent=x, timing=tm, x_start=x_start)
    if record:
        out["frames"] = frames
    return out


def save_png(arr, name):
    os.makedirs(LH, exist_ok=True)
    Image.fromarray(arr).save(os.path.join(LH, name))


def grid(images, ncol, path, thumb=256, pad=8):
    n = len(images)
    nr = (n + ncol - 1) // ncol
    G = Image.new("RGB", (ncol * (thumb + pad) + pad, nr * (thumb + pad) + pad), (13, 13, 13))
    for i, im in enumerate(images):
        r, c = divmod(i, ncol)
        G.paste(Image.fromarray(im).resize((thumb, thumb), Image.LANCZOS), (pad + c * (thumb + pad), pad + r * (thumb + pad)))
    G.save(path)


# ----------------------------------------------------------------------------- stages
def st_check():
    """Verify the instrumented loop reproduces the library's own generate_latents bit-for-bit."""
    sd = load(SD21)
    lib = None
    t0 = time.perf_counter()
    for lib in sd.generate_latents(PROMPT, n_images=1, num_steps=4, cfg_weight=7.5, seed=0):
        mx.eval(lib)
    RES["cold_first_denoise_4steps_cfg_s"] = time.perf_counter() - t0  # includes Metal kernel compilation
    mine = generate(sd, PROMPT, 4, 7.5, 0)["latent"]
    d = float(mx.abs(lib.astype(mx.float32) - mine.astype(mx.float32)).max())
    RES["loop_matches_library_max_abs_diff"] = d
    save_res()
    print("check max abs diff", d, flush=True)


def st_timing():
    sd = load(SD21)
    warm = generate(sd, PROMPT, 3, 7.5, 0)  # warm-up (kernel compilation)
    RES["first_generation_3steps_s"] = warm["timing"]["total_s"]
    rows = {}
    for name, cfg in [("cfg7.5_50steps", 7.5), ("cfg1_50steps_no_cfg", 1.0)]:
        runs = []
        for _ in range(3):  # the Mac is shared with other jobs: repeat and keep the median run
            mx.clear_cache(); reset_peak()
            g = generate(sd, PROMPT, 50, cfg, 0)
            g["timing"]["peak_mem_gb"] = get_peak() / 1e9
            runs.append(g)
        totals = [r["timing"]["total_s"] for r in runs]
        g = runs[int(np.argsort(totals)[1])]
        tm = g["timing"]
        tm["repeat_totals_s"] = totals
        tm["mean_step_s"] = float(np.mean(tm["step_s"][1:]))
        tm["latent_shape"] = list(g["latent"].shape)
        tm["image_shape"] = list(g["img"].shape)
        rows[name] = tm
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in tm.items() if k != "step_s"}, flush=True)
    sdt = load(TURBO)
    generate(sdt, PROMPT, 1, 0.0, 0)  # warm-up
    runs = []
    for _ in range(3):
        mx.clear_cache(); reset_peak()
        g = generate(sdt, PROMPT, 1, 0.0, 0)
        g["timing"]["peak_mem_gb"] = get_peak() / 1e9
        runs.append(g["timing"])
    totals = [r["total_s"] for r in runs]
    rows["turbo_1step"] = dict(runs[int(np.argsort(totals)[1])], repeat_totals_s=totals)
    print("turbo_1step", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in g["timing"].items()}, flush=True)
    RES["timing"] = rows
    save_res()


def st_steps():
    sd = load(SD21)
    rows, imgs = [], []
    for seed in SEEDS:
        for n in [1, 2, 4, 8, 20, 50]:
            g = generate(sd, PROMPT, n, 7.5, seed)
            fn = f"lighthouse_sd21_steps{n:02d}_cfg7.5_seed{seed}.png"
            save_png(g["img"], fn)
            imgs.append(g["img"])
            rows.append(dict(model=SD21, steps=n, cfg=7.5, seed=seed, file=fn, total_s=g["timing"]["total_s"],
                             denoise_s=g["timing"]["denoise_total_s"]))
            print("steps", seed, n, round(g["timing"]["total_s"], 2), flush=True)
    grid(imgs, 6, os.path.join(IMAGES, "lighthouse_grid_sd21_steps_1-2-4-8-20-50_cfg7.5_seeds0-3.png"))
    RES["steps_sweep"] = rows
    save_res()


def st_guidance():
    sd = load(SD21)
    rows, imgs = [], []
    for seed in SEEDS:
        for w in [1.0, 3.0, 5.0, 7.5, 10.0, 15.0]:
            g = generate(sd, PROMPT, 50, w, seed)
            fn = f"lighthouse_sd21_cfg{w:g}_steps50_seed{seed}.png"
            save_png(g["img"], fn)
            imgs.append(g["img"])
            im = g["img"].astype(np.float32) / 255
            mx_, mn_ = im.max(-1), im.min(-1)
            sat = np.where(mx_ > 0, (mx_ - mn_) / np.maximum(mx_, 1e-6), 0)
            rows.append(dict(model=SD21, steps=50, cfg=w, seed=seed, file=fn, total_s=g["timing"]["total_s"],
                             mean_saturation=float(sat.mean()), mean_brightness=float(im.mean()),
                             frac_clipped_pixels=float(((g["img"] == 0) | (g["img"] == 255)).any(-1).mean())))
            print("guidance", seed, w, round(g["timing"]["total_s"], 2), flush=True)
    grid(imgs, 6, os.path.join(IMAGES, "lighthouse_grid_sd21_cfg_1-3-5-7.5-10-15_steps50_seeds0-3.png"))
    RES["guidance_sweep"] = rows
    save_res()


def st_turbo():
    sdt = load(TURBO)
    rows, imgs = [], []
    for seed in SEEDS:
        for n in [1, 2, 4]:
            g = generate(sdt, PROMPT, n, 0.0, seed)
            fn = f"lighthouse_sdturbo_steps{n}_seed{seed}.png"
            save_png(g["img"], fn)
            imgs.append(g["img"])
            rows.append(dict(model=TURBO, steps=n, cfg=0.0, seed=seed, file=fn, total_s=g["timing"]["total_s"],
                             denoise_s=g["timing"]["denoise_total_s"]))
            print("turbo", seed, n, round(g["timing"]["total_s"], 2), flush=True)
    grid(imgs, 3, os.path.join(IMAGES, "lighthouse_grid_sdturbo_steps_1-2-4_seeds0-3.png"))
    RES["turbo_sweep"] = rows
    save_res()


def st_frames():
    sd = load(SD21)
    g = generate(sd, PROMPT, 30, 7.5, 0, record=True)
    fr = g["frames"]
    d = os.path.join(IMAGES, "lighthouse_frames30")
    os.makedirs(d, exist_ok=True)
    dec_s = []
    xt_imgs, x0_imgs = [], []
    for i, lat in enumerate(fr["x_t"]):  # 31 latents: x_T and after each of 30 steps
        t0 = time.perf_counter()
        im = to_img(sd.decode(mx.array(lat[None]).astype(mx.float32)))
        dec_s.append(time.perf_counter() - t0)
        Image.fromarray(im).save(os.path.join(d, f"xt_{i:02d}.png"))
        xt_imgs.append(im)
    for i, lat in enumerate(fr["x0_hat"]):  # 30 clean-image guesses
        im = to_img(sd.decode(mx.array(lat[None]).astype(mx.float32)))
        Image.fromarray(im).save(os.path.join(d, f"x0hat_{i:02d}.png"))
        x0_imgs.append(im)
    save_png(g["img"], "lighthouse_sd21_steps30_cfg7.5_seed0_final.png")
    np.savez_compressed(os.path.join(ARRAYS, "lighthouse_latents_30steps.npz"),
                        x_t=np.stack(fr["x_t"]).astype(np.float16), x0_hat=np.stack(fr["x0_hat"]).astype(np.float16),
                        t=np.array(fr["t"]), sigma=np.array(fr["sigma"]),
                        note=f"SD 2.1-base (mlx Euler), prompt '{PROMPT}', 30 steps, cfg 7.5, seed 0. x_t[0] = x_T, "
                             "x_t[i] = latent after step i; x0_hat[i] = clean-latent guess at step i (before the step). "
                             "Latents are [64,64,4] (NHWC) for a 512x512x3 image; frames in images/lighthouse_frames30/")
    pick = np.round(np.linspace(0, 29, 10)).astype(int)
    grid([x0_imgs[i] for i in pick] + [xt_imgs[i + 1] for i in pick], 10,
         os.path.join(IMAGES, "lighthouse_frames30_strip_x0hat_top_xt_bottom.png"), thumb=192)
    RES["frames"] = dict(steps=30, cfg=7.5, seed=0, n_xt_frames=len(fr["x_t"]), n_x0hat_frames=len(fr["x0_hat"]),
                         latent_shape=list(fr["x_t"][0].shape), image_shape=list(g["img"].shape),
                         vae_decode_s_median=float(np.median(dec_s[1:])), dir="images/lighthouse_frames30",
                         timing=g["timing"])
    save_res()


def st_sdedit():
    sd = load(SD21)
    src = np.array(Image.open(os.path.join(LH, "lighthouse_sd21_cfg7.5_steps50_seed0.png")).convert("RGB"))
    img = mx.array(src).astype(mx.float32) / 255 * 2 - 1
    x0, _ = sd.autoencoder.encode(img[None])
    mx.eval(x0)
    rows, imgs = [], [src]
    for s in [0.3, 0.5, 0.7, 0.9]:
        g = generate(sd, PROMPT_WINTER, 50, 7.5, 0, x0_latent=x0, strength=s)
        fn = f"lighthouse_sdedit_winter_strength{s:g}.png"
        save_png(g["img"], fn)
        noisy = to_img(sd.decode(g["x_start"].astype(mx.float32)))
        save_png(noisy, f"lighthouse_sdedit_winter_strength{s:g}_noised_start.png")
        imgs.append(g["img"])
        rows.append(dict(strength=s, steps_run=g["timing"]["denoise_steps"], start_t=1000 * s, file=fn,
                         total_s=g["timing"]["total_s"],
                         mean_abs_pixel_change=float(np.abs(g["img"].astype(np.float32) - src).mean() / 255)))
        print("sdedit", s, rows[-1]["steps_run"], flush=True)
    grid(imgs, 5, os.path.join(IMAGES, "lighthouse_grid_sdedit_winter_source_0.3_0.5_0.7_0.9.png"))
    RES["sdedit"] = dict(source="lighthouse_sd21_cfg7.5_steps50_seed0.png", prompt=PROMPT_WINTER, cfg=7.5, seed=0,
                         base_steps=50, rows=rows,
                         note="strength s: encode the source, add noise to t = 1000*s, run int(50*s) Euler steps")
    save_res()


PROMPT_SNOW = "a lighthouse on a snowy rocky coast in winter, snow-covered rocks, falling snow, at dusk, oil painting"


def st_sdedit2():
    """Second SDEdit series: a more explicit snow prompt, plus strength 1.0 (= text-to-image from almost pure noise)."""
    sd = load(SD21)
    src = np.array(Image.open(os.path.join(LH, "lighthouse_sd21_cfg7.5_steps50_seed0.png")).convert("RGB"))
    img = mx.array(src).astype(mx.float32) / 255 * 2 - 1
    x0, _ = sd.autoencoder.encode(img[None])
    rows, imgs = [], [src]
    for s in [0.3, 0.5, 0.7, 0.9, 1.0]:
        g = generate(sd, PROMPT_SNOW, 50, 7.5, 0, x0_latent=x0, strength=s)
        fn = f"lighthouse_sdedit_snow_strength{s:g}.png"
        save_png(g["img"], fn)
        imgs.append(g["img"])
        rows.append(dict(strength=s, steps_run=g["timing"]["denoise_steps"], start_t=1000 * s, file=fn,
                         total_s=g["timing"]["total_s"],
                         mean_abs_pixel_change=float(np.abs(g["img"].astype(np.float32) - src).mean() / 255)))
        print("sdedit2", s, rows[-1]["mean_abs_pixel_change"], flush=True)
    grid(imgs, 6, os.path.join(IMAGES, "lighthouse_grid_sdedit_snow_source_0.3_0.5_0.7_0.9_1.0.png"))
    RES["sdedit_snow"] = dict(source="lighthouse_sd21_cfg7.5_steps50_seed0.png", prompt=PROMPT_SNOW, cfg=7.5, seed=0,
                              base_steps=50, rows=rows)
    save_res()


def st_vae():
    sd = load(SD21)
    src = np.array(Image.open(os.path.join(LH, "lighthouse_sd21_cfg7.5_steps50_seed0.png")).convert("RGB"))
    img = mx.array(src).astype(mx.float32) / 255 * 2 - 1
    t0 = time.perf_counter()
    mean, logvar = sd.autoencoder.encode(img[None])
    mx.eval(mean)
    enc_s = time.perf_counter() - t0
    rec = to_img(sd.decode(mean))
    mse = float(((rec.astype(np.float64) - src) ** 2).mean())
    psnr = 10 * np.log10(255 ** 2 / mse)
    lat = np.array(mean[0])
    chans = []
    for c in range(lat.shape[-1]):
        a = lat[..., c]
        chans.append(((a - a.min()) / (a.max() - a.min() + 1e-8) * 255).astype(np.uint8))
    Image.fromarray(np.concatenate(chans, axis=1)).resize((64 * 4 * 4, 64 * 4), Image.NEAREST).save(
        os.path.join(IMAGES, "lighthouse_latent_4channels_64x64.png"))
    save_png(rec, "lighthouse_vae_roundtrip_reconstruction.png")
    np.savez_compressed(os.path.join(ARRAYS, "lighthouse_vae_latent.npz"), latent=lat.astype(np.float16),
                        note="VAE encoder mean for lighthouse_sd21_cfg7.5_steps50_seed0.png (512x512x3 -> 64x64x4); "
                             "the pipeline multiplies by the 0.18215 scaling factor inside encode()")
    RES["vae"] = dict(image_shape=list(src.shape), latent_shape=list(lat.shape),
                      compression_ratio=float(src.size / lat.size), encode_s=enc_s, roundtrip_psnr_db=float(psnr),
                      latent_std=float(lat.std()), latent_mean=float(lat.mean()))
    save_res()
    print("vae", RES["vae"], flush=True)


if __name__ == "__main__":
    os.makedirs(LH, exist_ok=True)
    stages = sys.argv[1:] or ["check", "timing", "steps", "guidance", "turbo", "frames", "sdedit", "vae"]
    reset_peak()
    for s in stages:
        t0 = time.perf_counter()
        globals()["st_" + s]()
        RES.setdefault("stage_wall_s", {})[s] = time.perf_counter() - t0
        save_res()
    RES["peak_mem_gb_overall"] = get_peak() / 1e9
    RES["mlx"] = mx.__version__
    save_res()
    print("LIGHTHOUSE_DONE", flush=True)
