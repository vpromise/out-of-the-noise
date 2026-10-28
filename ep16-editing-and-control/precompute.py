"""Episode 16 · one-off: read the SHAPES of SD 2.1-base's attention projections from its safetensors header (a JSON
block at the start of the file; no weights are loaded, no model is run) so the LoRA arithmetic on screen is exact.
Writes sd21_attention_shapes.json beside this file; the scene asserts against it (the Hugging Face cache that holds
the weights is deleted at the end of the course, research/experiments/mac_runs.md §10).

    ~/.local/share/uv/tools/manimgx/bin/python precompute.py
"""

import json
import struct
from pathlib import Path

SNAP = Path.home() / ".cache/huggingface/hub/models--sd2-community--stable-diffusion-2-1-base/snapshots"
REV = "4e63672c03103b6c636b8fb4119ba982469b2955"          # the revision mac_runs.md §5 pinned
UNET = SNAP / REV / "unet" / "diffusion_pytorch_model.fp16.safetensors"
PROJ = ("to_q.weight", "to_k.weight", "to_v.weight", "to_out.0.weight")

with open(UNET, "rb") as f:
    size = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(size))
header.pop("__metadata__", None)
count = lambda shape: int(eval("*".join(map(str, shape)) or "1"))  # noqa: E731
attn = {k: v["shape"] for k, v in header.items() if k.endswith(PROJ)}
out = {
    "source": f"sd2-community/stable-diffusion-2-1-base@{REV} unet/diffusion_pytorch_model.fp16.safetensors (header only)",
    "unet_params": sum(count(v["shape"]) for v in header.values()),
    "unet_file_bytes": UNET.stat().st_size,
    "attention_projections": attn,
    "scheduler": json.loads((SNAP / REV / "scheduler" / "scheduler_config.json").read_text()),
}
(Path(__file__).resolve().parent / "sd21_attention_shapes.json").write_text(json.dumps(out, indent=1))
print(len(attn), "projections;", out["unet_params"], "parameters")
