"""Merge all run outputs into research/experiments/mac_runs.json and print markdown tables (logs/tables.md).

  python gpu/build_report.py
"""
import json
import os
import platform
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ARRAYS, EXP, IMAGES, LOGS, MNIST_MD5, WEIGHTS, read_result  # noqa: E402


def jload(p):
    return json.load(open(p)) if os.path.exists(p) else None


def curve(name):
    p = os.path.join(LOGS, f"train_{name}.jsonl")
    if not os.path.exists(p):
        return None
    rows = [json.loads(l) for l in open(p)]
    if len(rows) > 200:
        idx = np.round(np.linspace(0, len(rows) - 1, 200)).astype(int)
        rows = [rows[i] for i in idx]
    return dict(step=[r["step"] for r in rows], loss=[round(r["loss"], 5) for r in rows],
                lr=[float(f"{r['lr']:.3g}") for r in rows], wall_s=[r["wall_s"] for r in rows],
                note="each point = mean training loss over the preceding 50 steps")


def r4(x, n=4):
    return None if x is None else float(round(x, n))


def main():
    out = {}
    out["meta"] = dict(
        machine="Mac Studio (Mac16,9), Apple M4 Max: 16-core CPU (12P+4E), 40-core GPU, 64 GB unified memory",
        os=f"macOS {platform.mac_ver()[0]}", python=platform.python_version(),
        mlx=__import__("mlx.core").core.__version__, numpy=np.__version__,
        mlx_examples_commit=subprocess.run(["git", "-C", os.path.join(EXP, "gpu/third_party/mlx-examples"), "log", "-1",
                                            "--format=%H"], capture_output=True, text=True).stdout.strip() or None,
        date="2026-10-25/26", venv="research/experiments/.venv-gpu (uv, Python 3.12)")
    out["data"] = dict(name="MNIST", source="https://ossci-datasets.s3.amazonaws.com/mnist/ (torchvision mirror of "
                                            "http://yann.lecun.com/exdb/mnist/)",
                       license="CC BY-SA 3.0 (Yann LeCun and Corinna Cortes)", md5=MNIST_MD5,
                       train=60000, test=10000, preprocessing="pixels scaled to [-1, 1]; no augmentation")
    b1, b2 = jload(os.path.join(LOGS, "bench_frameworks_mlx.json")), jload(os.path.join(LOGS, "bench_frameworks_torch.json"))
    if b1 and b2:
        out["framework_benchmark"] = dict(batch=b1["batch"], infer_batch=b1["infer_batch"],
                                          mlx=b1["results"], torch_mps=b2["results"],
                                          torch_version=__import__("torch").__version__)
    models = {}
    for name in ["ddpm", "flow", "dit"]:
        meta = jload(os.path.join(WEIGHTS, f"digit_{name}_ema.json"))
        if not meta:
            continue
        cfg = jload(os.path.join(LOGS, f"train_{name}_config.json"))
        c = curve(name)
        models[name] = dict(params=meta["params"], architecture=meta["config"], objective=meta["objective"],
                            steps=meta["steps"], batch=meta["batch"], epochs=meta["steps"] * meta["batch"] / 60000,
                            lr_peak=meta["lr"], lr_schedule=f"linear warm-up {cfg['warmup']} steps, cosine to 10%",
                            optimizer="Adam (beta 0.9/0.999), grad-norm clip 1.0", ema=meta["ema"],
                            label_dropout=meta["p_uncond"], seed=meta["seed"], wall_s=meta["wall_s"],
                            wall_min=meta["wall_s"] / 60, s_per_step=meta["wall_s"] / meta["steps"],
                            peak_mem_gb=meta["peak_mem_gb"], final_loss=c["loss"][-1] if c else None,
                            loss_curve=c, weights=f"weights/digit_{name}_ema.safetensors")
    out["digit_models"] = models
    out["classifier"] = read_result("classifier")
    out["architecture"] = read_result("architecture")
    met = read_result("metrics") or {}
    gens = {}
    for fn in sorted(os.listdir(os.path.join(EXP, "results"))):
        if fn.startswith("gen_"):
            g = json.load(open(os.path.join(EXP, "results", fn)))
            m = met.get(g["cfg"], {})
            gens[g["cfg"]] = dict(model=g["model"], sampler=g["sampler"], steps=g["steps"], w=g["w"],
                                  nfe_per_sample=g["nfe_per_sample"], s_per_1000=r4(g["s_per_1000"], 3),
                                  ms_per_sample=r4(g["ms_per_sample"], 3), first_batch_s=r4(g["first_batch_s"], 2),
                                  peak_mem_gb=r4(g["peak_mem_gb"], 3), cfid=r4(m.get("cfid")),
                                  cond_acc=r4(m.get("cond_acc")), precision=r4(m.get("precision")),
                                  recall=r4(m.get("recall")), diversity=r4(m.get("diversity")),
                                  mean_confidence=r4(m.get("mean_confidence")),
                                  label_entropy_bits=r4(m.get("predicted_label_entropy_bits")),
                                  cond_acc_per_class=m.get("cond_acc_per_class"),
                                  samples=f"arrays/digit_samples_{g['cfg']}.npz",
                                  grid=f"images/digit_grid_{g['cfg']}.png")
    out["digit_sampling"] = dict(
        n_samples=10000, batch=1000, labels="sample i has label i % 10 (1,000 per class)",
        initial_noise="np.random.default_rng(1234), shared by every config (same seeds)",
        guidance_convention="pred = uncond + w (cond - uncond); w=0 unconditional, w=1 plain conditional (1 pass), "
                            "w>1 classifier-free guidance (2 passes); Ho & Salimans' w_HS = w - 1",
        timing_note="s_per_1000 = mean wall time of batches 2..10 (batch 1 includes compilation), batch 1000 on GPU",
        configs=gens, real_reference=met.get("real_train10k_vs_test"))
    out["nearest_neighbours"] = read_result("nearest_neighbours")
    out["latency"] = read_result("latency")
    out["loss_vs_t"] = read_result("loss_vs_t")
    out["lighthouse"] = read_result("lighthouse")
    out["lighthouse_downloads"] = jload(os.path.join(LOGS, "lighthouse_downloads.json"))
    out["cfid_uncertainty"] = read_result("cfid_uncertainty")
    vt = read_result("video_toy")
    if vt:
        out["video_toy_stacked_frames_attempt"] = dict(
            status="FAILED - undertrained (2,500 steps x 64 clips, ~4 min per model); samples are noise-ridden and its "
                   "flicker/slope numbers are meaningless. Do NOT use on screen. Outputs moved to "
                   "logs/video_toy_stacked_undertrained/.", raw=vt)
    out["video_toy_noise_correlation"] = read_result("video_toy_noise")

    # ------------------------------------------------------------------ headline
    H = {}

    def g(cfg, k):
        return gens.get(cfg, {}).get(k)

    if "ddpm" in models:
        H["digit_unet_params"] = models["ddpm"]["params"]
        H["digit_train_steps"] = models["ddpm"]["steps"]
        H["digit_train_batch"] = models["ddpm"]["batch"]
        H["digit_train_epochs"] = r4(models["ddpm"]["epochs"], 1)
        H["digit_ddpm_train_minutes"] = r4(models["ddpm"]["wall_min"], 1)
        H["digit_train_peak_mem_gb"] = r4(models["ddpm"]["peak_mem_gb"], 2)
        H["digit_ddpm_final_loss"] = r4(models["ddpm"]["final_loss"])
    if "flow" in models:
        H["digit_flow_train_minutes"] = r4(models["flow"]["wall_min"], 1)
    if out["classifier"]:
        H["classifier_test_accuracy"] = r4(out["classifier"]["test_accuracy"])
    if met.get("real_train10k_vs_test"):
        H["cfid_real_data_floor"] = r4(met["real_train10k_vs_test"]["cfid"], 3)
    for cfg, key in [("ddpm_ddpm1000_w1", "ddpm1000"), ("ddpm_ddim100_w1", "ddim100"), ("ddpm_ddim50_w1", "ddim50"),
                     ("ddpm_ddim20_w1", "ddim20"), ("ddpm_ddim10_w1", "ddim10"), ("ddpm_ddim5_w1", "ddim5"),
                     ("ddpm_ddim2_w1", "ddim2"), ("ddpm_ddim1_w1", "ddim1"), ("flow_euler50_w1", "flow50"), ("flow_euler10_w1", "flow10"),
                     ("flow_euler5_w1", "flow5"), ("flow_euler2_w1", "flow2"), ("flow_euler1_w1", "flow1")]:
        if g(cfg, "cfid") is not None:
            H[f"cfid_{key}"] = r4(g(cfg, "cfid"), 3)
    for cfg, key in [("ddpm_ddpm1000_w1", "ddpm1000"), ("ddpm_ddim50_w1", "ddim50"), ("ddpm_ddim10_w1", "ddim10"),
                     ("flow_euler10_w1", "flow10"), ("flow_euler1_w1", "flow1"), ("ddpm_ddim50_w3", "ddim50_w3")]:
        if g(cfg, "s_per_1000") is not None:
            H[f"seconds_per_1000_digits_{key}"] = r4(g(cfg, "s_per_1000"), 2)
    for w in ["0", "1", "2", "3", "5", "8"]:
        cfg = f"ddpm_ddim50_w{w}"
        if g(cfg, "cfid") is not None:
            H[f"guidance_w{w}_cfid"] = r4(g(cfg, "cfid"), 3)
            H[f"guidance_w{w}_cond_acc"] = r4(g(cfg, "cond_acc"), 4)
            H[f"guidance_w{w}_diversity"] = r4(g(cfg, "diversity"), 3)
    nn_ = out["nearest_neighbours"]
    if nn_ and "ddpm_ddim50_w1" in nn_:
        H["nn_dist_generated_median"] = r4(nn_["ddpm_ddim50_w1"]["median"], 2)
        H["nn_dist_test_median"] = r4(nn_["test_to_train"]["median"], 2)
        H["nn_dist_random_pair_median"] = r4(nn_["random_pair_median"], 2)
    if "dit" in models:
        H["dit_params"] = models["dit"]["params"]
        H["dit_train_minutes"] = r4(models["dit"]["wall_min"], 1)
        if g("dit_ddim50_w1", "cfid") is not None:
            H["cfid_dit_ddim50"] = r4(g("dit_ddim50_w1", "cfid"), 3)
            H["seconds_per_1000_digits_dit_ddim50"] = r4(g("dit_ddim50_w1", "s_per_1000"), 2)
    cu = out["cfid_uncertainty"]
    if cu:
        H["cfid_real_floor_std_5_subsets"] = r4(cu["real_floor_5_subsets"]["std"], 3)
        H["cfid_bootstrap_std_ddim50"] = r4(cu["ddpm_ddim50_w1"]["bootstrap_std"], 3)
    for w in ["0", "2", "3", "5", "8"]:
        if g(f"flow_euler20_w{w}", "cfid") is not None:
            H[f"flow_euler20_guidance_w{w}_cfid"] = r4(g(f"flow_euler20_w{w}", "cfid"), 3)
    for w in ["1", "3", "8"]:
        if g(f"ddpm_ddim50_w{w}", "recall") is not None:
            H[f"guidance_w{w}_precision"] = r4(g(f"ddpm_ddim50_w{w}", "precision"), 3)
            H[f"guidance_w{w}_recall"] = r4(g(f"ddpm_ddim50_w{w}", "recall"), 3)
    if g("dit_ddim50_w1", "cond_acc") is not None:
        H["dit_ddim50_cond_acc"] = r4(g("dit_ddim50_w1", "cond_acc"), 4)
    vn = out["video_toy_noise_correlation"]
    if vn:
        for row in vn["rows"]:
            H[f"video_flicker_rho{row['rho']:g}"] = r4(row["flicker"], 4)
    L = out["lighthouse"] or {}
    if "timing" in L:
        t = L["timing"]["cfg7.5_50steps"]
        H["lighthouse_total_s_50steps_cfg7.5"] = r4(t["total_s"], 2)
        H["lighthouse_text_encoder_s"] = r4(t["text_encoder_s"], 3)
        H["lighthouse_unet_step_s_cfg"] = r4(t["mean_step_s"], 3)
        H["lighthouse_unet_step_s_nocfg"] = r4(L["timing"]["cfg1_50steps_no_cfg"]["mean_step_s"], 3)
        H["lighthouse_vae_decode_s"] = r4(t["vae_decode_s"], 3)
        H["lighthouse_peak_mem_gb"] = r4(t["peak_mem_gb"], 2)
        H["lighthouse_turbo_1step_total_s"] = r4(L["timing"]["turbo_1step"]["total_s"], 3)
    if "turbo_sweep" in L:
        for n in [1, 2, 4]:
            v = [r["total_s"] for r in L["turbo_sweep"] if r["steps"] == n]
            H[f"lighthouse_turbo_{n}step_total_s_median_sweep"] = r4(float(np.median(v)), 3)
    if "steps_sweep" in L:
        for n in [1, 4, 20, 50]:
            v = [r["total_s"] for r in L["steps_sweep"] if r["steps"] == n]
            H[f"lighthouse_sd21_{n}steps_total_s_median_sweep"] = r4(float(np.median(v)), 2)
    if "guidance_sweep" in L:
        for w in [1.0, 7.5, 15.0]:
            v = [r["mean_saturation"] for r in L["guidance_sweep"] if r["cfg"] == w]
            H[f"lighthouse_cfg{w:g}_mean_saturation"] = r4(float(np.mean(v)), 3)
    if "models" in L and "sd2-community/stable-diffusion-2-1-base" in L["models"]:
        p = L["models"]["sd2-community/stable-diffusion-2-1-base"]["parts"]
        H["lighthouse_unet_params_M"] = r4(p["unet"]["params"] / 1e6, 1)
        H["lighthouse_text_encoder_params_M"] = r4(p["text_encoder"]["params"] / 1e6, 1)
        H["lighthouse_vae_params_M"] = r4(p["vae"]["params"] / 1e6, 1)
    if "vae" in L:
        H["lighthouse_image_px"] = L["vae"]["image_shape"][0]
        H["lighthouse_latent_h"], H["lighthouse_latent_w"], H["lighthouse_latent_c"] = L["vae"]["latent_shape"]
        H["lighthouse_vae_compression_ratio"] = r4(L["vae"]["compression_ratio"], 1)
        H["lighthouse_vae_roundtrip_psnr_db"] = r4(L["vae"]["roundtrip_psnr_db"], 2)
    dl = out["lighthouse_downloads"]
    if dl:
        H["lighthouse_download_gb_total"] = r4(sum(v["total_bytes"] for v in dl.values()) / 1e9, 2)
    out = dict(headline=H, **out)
    out["files"] = dict(images=sorted(os.listdir(IMAGES)), arrays=sorted(os.listdir(ARRAYS)),
                        weights=sorted(f for f in os.listdir(WEIGHTS) if f.startswith("digit_")))
    with open(os.path.join(EXP, "mac_runs.json"), "w") as f:
        json.dump(out, f, indent=1)

    # ------------------------------------------------------------------ tables
    T = ["| config | NFE/sample | s per 1,000 | cFID | cond. acc | precision | recall | diversity |",
         "|---|---|---|---|---|---|---|---|"]
    for cfg, v in gens.items():
        T.append(f"| {cfg} | {v['nfe_per_sample']} | {v['s_per_1000']} | {v['cfid']} | {v['cond_acc']} | "
                 f"{v['precision']} | {v['recall']} | {v['diversity']} |")
    open(os.path.join(LOGS, "tables.md"), "w").write("\n".join(T) + "\n")
    print(json.dumps(H, indent=0))


if __name__ == "__main__":
    main()
