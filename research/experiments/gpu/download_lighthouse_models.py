"""Download exactly the files the lighthouse runs need into the default Hugging Face cache."""
import json, os, sys
from huggingface_hub import hf_hub_download, HfApi
FILES = ["unet/config.json", "unet/diffusion_pytorch_model.fp16.safetensors",
         "text_encoder/config.json", "text_encoder/model.fp16.safetensors",
         "vae/config.json", "vae/diffusion_pytorch_model.safetensors",
         "scheduler/scheduler_config.json", "tokenizer/vocab.json", "tokenizer/merges.txt"]
REPOS = ["sd2-community/stable-diffusion-2-1-base", "stabilityai/sd-turbo"]
out = {}
for r in REPOS:
    sha = HfApi().model_info(r).sha
    out[r] = dict(revision=sha, files={})
    for f in FILES:
        p = hf_hub_download(r, f, revision=sha)
        out[r]["files"][f] = os.path.getsize(os.path.realpath(p))
        print(r, f, round(out[r]["files"][f] / 1e6, 1), "MB", flush=True)
    out[r]["total_bytes"] = sum(out[r]["files"].values())
json.dump(out, open(sys.argv[1], "w"), indent=1)
print("DOWNLOAD_DONE", {r: round(v["total_bytes"] / 1e9, 3) for r, v in out.items()})
