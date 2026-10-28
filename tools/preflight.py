"""Pronunciation pre-flight for "Trial and Reward": speak one sentence per risky term through the running
narrator (which reads lexicon.json), then transcribe each clip with Whisper large and small, and report
which terms both listeners heard. Run with the narrator env, from the series folder:
    ~/rag-video/tts/.venv/bin/python tools/preflight.py [out_dir]"""

import base64
import io
import json
import re
import sys
import urllib.request
from pathlib import Path

import librosa
import mlx_whisper
import soundfile

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "research/preflight")
OUT.mkdir(parents=True, exist_ok=True)

# (term as written in the script, sentence, regex a correct transcript must match)
TESTS = [
    ("DDPM", "Then DDPM made diffusion work on real images.", r"\bddpm\b|d ?d ?p ?m"),
    ("DDIM", "With DDIM the same model needs far fewer steps.", r"\bddim\b|d ?d ?i ?m"),
    ("U-Net", "The U-Net shrinks the picture and grows it back.", r"\bu[- ]?net\b"),
    ("VAE", "The VAE squeezes the image into a small latent.", r"\bvae\b|v ?a ?e"),
    ("GAN", "A GAN pits a generator against a critic.", r"\bgan\b|g ?a ?n"),
    ("CLIP", "The CLIP text encoder turns words into vectors.", r"\bclip\b"),
    ("FID", "A lower FID means the samples look more like real data.", r"\bfid\b|f ?i ?d"),
    ("SDXL", "Then SDXL generated larger images.", r"\bsdxl\b|s ?d ?x ?l"),
    ("DiT", "A DiT replaces the U-Net with a transformer.", r"\bdit\b|d ?i ?t|\bdit\b"),
    ("MNIST", "The MNIST dataset holds seventy thousand handwritten digits.", r"\bmnist\b|m ?nist|em ?nist"),
    ("MLX", "We ran it with MLX on this Mac.", r"\bmlx\b|m ?l ?x"),
    ("ControlNet", "With ControlNet the model follows a sketch.", r"control ?net"),
    ("LoRA", "A small LoRA adapter teaches it a new style.", r"\blora\b|\blaura\b"),
    ("DreamBooth", "With DreamBooth it learns one object from a few photos.", r"dream ?booth"),
    ("SDEdit", "With SDEdit you add noise to a picture and denoise it again.", r"sd ?edit|s ?d ?edit"),
    ("Stable Diffusion", "Stable Diffusion runs on a laptop.", r"stable diffusion"),
    ("DALL-E", "Then DALL-E 3 trained on better captions.", r"dall[- ]?e|dolly|dali"),
    ("Sora", "Then Sora turned diffusion into video.", r"\bsora\b"),
    ("FLUX", "The FLUX model uses flow matching.", r"\bflux\b"),
    ("Imagen", "Then Imagen used a large language model as its text encoder.", r"imagen|imogen"),
    ("Langevin", "Langevin dynamics follows the score with a little noise.", r"langevin|langeven"),
    ("Tweedie", "Tweedie's formula turns a noisy point into a clean guess.", r"tweedie"),
    ("Heun", "The Heun solver takes two looks per step.", r"\bheun\b|\bhoin\b|\bhoyne\b|\bhain\b"),
    ("Euler", "An Euler step follows the slope in a straight line.", r"euler|oiler"),
    ("Karras", "Then Karras and colleagues tuned the schedule.", r"karras|caras|karas"),
    ("epsilon", "The network predicts epsilon, the noise.", r"epsilon"),
    ("alpha bar", "Alpha bar tells how much signal is left.", r"alpha bar"),
    ("ODE", "Deterministic sampling solves an ODE.", r"\bode\b|o ?d ?e"),
    ("SDE", "The forward process is an SDE.", r"\bsde\b|s ?d ?e"),
    ("latent", "The latent is sixty-four by sixty-four by four.", r"latent"),
    ("Pixel", "Pixel is half noise and half picture.", r"\bpixel\b"),
    ("CFG", "With CFG the model follows the prompt more closely.", r"\bcfg\b|c ?f ?g"),
    ("Inception", "FID uses features from an Inception network.", r"inception"),
    ("T5", "Then Imagen used T5 as its text encoder.", r"\bt5\b|t ?five"),
    ("Dhariwal", "Dhariwal and Nichol showed diffusion beats GANs.", r"dhariwal|dariwal|dariwal"),
]

LISTENERS = {"large": "mlx-community/whisper-large-v3-turbo", "small": "mlx-community/whisper-small-mlx"}


def speak(sentence):
    settings = dict(voice_ref="", exaggeration=0.5, cfg_weight=0.3, temperature=0.8)  # the same voice as kit.narrator
    req = urllib.request.Request("http://127.0.0.1:8765/speak", json.dumps({"text": sentence, **settings}).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as reply:
        return base64.b64decode(json.load(reply)["audio"])


rows = []
for i, (term, sentence, pattern) in enumerate(TESTS):
    wav = OUT / f"{i:02d}-{re.sub(r'[^A-Za-z0-9]+', '_', term)}.wav"
    if not wav.exists():
        wav.write_bytes(speak(sentence))
    audio, _ = librosa.load(io.BytesIO(wav.read_bytes()), sr=16000)
    heard = {name: mlx_whisper.transcribe(audio, path_or_hf_repo=repo, language="en")["text"].strip()
             for name, repo in LISTENERS.items()}
    ok = {name: bool(re.search(pattern, t.lower())) for name, t in heard.items()}
    rows.append(dict(term=term, sentence=sentence, heard=heard, ok=ok))
    mark = "ok  " if all(ok.values()) else ("half" if any(ok.values()) else "FAIL")
    print(f"{mark} {term:15} large: {heard['large'][:70]!r} | small: {heard['small'][:70]!r}", flush=True)

(OUT / "preflight.json").write_text(json.dumps(rows, indent=1))
print(f"{sum(all(r['ok'].values()) for r in rows)}/{len(rows)} terms heard by both listeners")
