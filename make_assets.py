"""Writes assets/ for "Out of the Noise": the backdrop (a dark darkroom grey, a soft vignette, a fine grid) and
the base icons — the "Agents, Animated" set recoloured into this palette, plus a camera, a film strip, a picture
and a brush. Run once with a Python that has numpy and Pillow."""
import re
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
SOURCE = Path.home() / "videos" / "agent-series" / "assets"
ASSETS.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------- backdrop
W, H = 1920, 1080
y, x = np.mgrid[0:H, 0:W].astype(np.float32)
base = np.array([16, 17, 20], np.float32)            # #101114
lift = np.array([30, 32, 40], np.float32)            # a cool grey centre lift
r = np.sqrt(((x - W * 0.5) / (W * 0.62)) ** 2 + ((y - H * 0.45) / (H * 0.70)) ** 2)
t = np.clip(1 - r, 0, 1) ** 2
img = base + (lift - base) * t[..., None] * 0.55
img *= (1 - 0.35 * np.clip(r - 0.75, 0, 1))[..., None]
step = 1920 / 14.222 * 0.5                           # a line every half unit
minor = (np.abs((x + 0.5) % step) < 1.0) | (np.abs((y + 0.5) % step) < 1.0)
major = (np.abs((x + 0.5) % (step * 4)) < 1.2) | (np.abs((y + 0.5) % (step * 4)) < 1.2)
img += minor[..., None] * np.array([6, 6, 8], np.float32) * (0.6 + 0.4 * t[..., None])
img += major[..., None] * np.array([5, 5, 7], np.float32) * (0.6 + 0.4 * t[..., None])
Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(ASSETS / "backdrop.png")

# ---------------------------------------------------------------------------- recoloured base icons
AMBER, AMBER_FILL, BLUE, BLUE_FILL = "#FBBF24", "#3A2C0A", "#60A5FA", "#132744"
PINK, PINK_FILL, TEAL, TEAL_FILL = "#F472B6", "#3B1530", "#2DD4BF", "#0E2E2A"
VIOLET, VIOLET_FILL, LIME, LIME_FILL = "#A78BFA", "#241C3D", "#BEF264", "#26330E"
GREEN, GREEN_FILL, RED, RED_FILL = "#4ADE80", "#103320", "#F87171", "#3B1515"
REMAP = {
    "#FF5C7A": RED, "#3A1520": RED_FILL,                                          # rose → red
    "#3DDC97": GREEN, "#12301F": GREEN_FILL, "#1B6B47": "#1C6B3F",                 # mint → green
    "#8EF0C4": "#A7F3C8", "#1B5A44": "#1B5536",
    "#FFC857": AMBER, "#3A2E14": AMBER_FILL, "#B7791F": "#B8861A", "#FFD98A": "#FDE08A",  # sun → amber
    "#5AB8FF": BLUE, "#132838": BLUE_FILL, "#D6ECFF": "#DBEAFE", "#4AA8F0": "#4C8EF0",  # azure → blue
    "#C792EA": VIOLET, "#2C1F38": VIOLET_FILL, "#E7C4F5": "#DDD6FE", "#EEDDFB": "#EDE9FE",  # orchid → violet
    "#261B30": "#1F1A33",
    "#FF7A59": PINK,                                                              # coral → pink
    "#7A7166": "#6B7280", "#A9A094": "#9AA1AE", "#DDD5C9": "#E2E5EA",            # warm greys → neutral greys
    "#1E1B17": "#181A20", "#14120F": "#101114",
}
for svg in sorted(SOURCE.glob("*.svg")):
    if re.match(r"ep\d", svg.name):
        continue
    out = re.sub(r"#[0-9A-Fa-f]{6}", lambda mt: REMAP.get(mt.group(0).upper(), mt.group(0)), svg.read_text())
    left = {c.upper() for c in re.findall(r"#[0-9A-Fa-f]{6}", out)} - {v.upper() for v in REMAP.values()}
    assert not left, (svg.name, left)
    (ASSETS / svg.name).write_text(out)

# ---------------------------------------------------------------------------- new icons (flat, 2-tone)
LT = "#E2E5EA"
NEW = {
    "camera": f'''<rect x="10" y="30" width="80" height="54" rx="10" fill="{TEAL_FILL}" stroke="{TEAL}" stroke-width="5"/>
  <path d="M34 30 L40 18 H60 L66 30" fill="{TEAL_FILL}" stroke="{TEAL}" stroke-width="5" stroke-linejoin="round"/>
  <circle cx="50" cy="57" r="16" fill="#101114" stroke="{TEAL}" stroke-width="5"/><circle cx="50" cy="57" r="6" fill="{TEAL}"/>''',
    "film": f'''<rect x="12" y="14" width="76" height="72" rx="6" fill="{LIME_FILL}" stroke="{LIME}" stroke-width="5"/>
  {"".join(f'<rect x="18" y="{22 + 13 * i}" width="7" height="7" rx="1.5" fill="{LIME}"/><rect x="75" y="{22 + 13 * i}" width="7" height="7" rx="1.5" fill="{LIME}"/>' for i in range(5))}
  <rect x="32" y="22" width="36" height="25" rx="3" fill="{LIME}"/><rect x="32" y="53" width="36" height="25" rx="3" fill="{LIME}" opacity="0.55"/>''',
    "picture": f'''<rect x="12" y="16" width="76" height="68" rx="8" fill="{TEAL_FILL}" stroke="{TEAL}" stroke-width="5"/>
  <circle cx="34" cy="36" r="8" fill="{AMBER}"/><path d="M18 76 L42 50 L56 64 L66 54 L84 76 Z" fill="{TEAL}"/>''',
    "noise": f'''<rect x="12" y="12" width="76" height="76" rx="8" fill="{VIOLET_FILL}" stroke="{VIOLET}" stroke-width="5"/>
  {"".join(f'<rect x="{20 + 12 * (k % 6)}" y="{20 + 12 * (k // 6)}" width="10" height="10" fill="{VIOLET}" opacity="{(0.25, 0.9, 0.5, 0.15, 0.7, 0.35, 1.0)[k % 7]}"/>' for k in range(36))}''',
    "brush": f'''<path d="M70 14 L86 30 L48 68 L32 52 Z" fill="{PINK_FILL}" stroke="{PINK}" stroke-width="5" stroke-linejoin="round"/>
  <path d="M32 52 L48 68 C44 80 30 88 14 86 C18 74 20 60 32 52 Z" fill="{PINK}"/>''',
}
for name, body in NEW.items():
    (ASSETS / f"{name}.svg").write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">\n  {body}\n</svg>\n')
print(f"{len(list(ASSETS.glob('*.svg')))} icons + backdrop.png in {ASSETS}")
