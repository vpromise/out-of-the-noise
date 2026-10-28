"""Write research/experiments/toy.md from toy.json (so every number in the prose matches the JSON).
Run from the series folder:  research/experiments/.venv-toy/bin/python research/experiments/scripts/toy_report.py
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
r = json.loads((HERE / "toy.json").read_text())


def pct(v, digits=1):
    return f"{100 * v:.{digits}f} %"


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


m = r["about"]["machine"]
wall = r["about"]["wall_seconds"]
tr = r["training"]
fw = r["forward"]
sc = r["score"]
sp = r["samplers"]
fl = r["flow"]
gd = r["guidance"]
lv = r["langevin"]
fd = r["fid"]
h = r["headline"]
se, sm = sp["exact"]["runs"], sp["mlp"]["runs"]
data = sp["data"]

md = []
add = md.append

add(f"""# Toy experiments: the swirl in 2D

Everything about the swirl that scenes show: the forward process, exact scores vs a trained MLP, every sampler,
flow matching, guidance, Langevin dynamics and FID's formula. All NumPy, all on this Mac's CPU. Results are in
`toy.json`; scenes read them with `measured("toy", …)` and recompute anything else live with
`diffusion_reference.py` (exact scores and the trained MLPs are fast enough to run inside a scene).

**Machine:** {m['cpu']}, {m['memory_gb']} GB, {m['os']}; Python {m['python']}, NumPy {m['numpy']}; {m['date']}.
**Wall times:** training {wall['train_eps']:.0f} s (ε-model) + {wall['train_flow']:.0f} s (flow model); experiments
{wall['total']:.0f} s in total (samplers {wall['samplers']:.0f} s, Langevin {wall['langevin']:.0f} s, guidance
{wall['guidance']:.0f} s, flow {wall['flow']:.0f} s, the rest < 2 s each). `python diffusion_reference.py` (the self-checks) takes ≈ 3 s.

```
research/experiments/.venv-toy/bin/python research/experiments/scripts/toy_train.py        # ~3 min, writes weights/swirl_*.npz
research/experiments/.venv-toy/bin/python research/experiments/scripts/toy_experiments.py  # ~100 s, writes toy.json
research/experiments/.venv-toy/bin/python research/experiments/scripts/toy_report.py       # writes this file
```
(`.venv-toy` is a uv venv with numpy, Pillow and matplotlib; the scripts need only numpy.)

## Setup

- **The swirl** (`dr.swirl(n, seed)`): two spiral arms, each 1¼ turns from radius 0.15 out to 1.0, so it fits the unit
  disc the way pixels are scaled to [−1, 1]; std ≈ 0.5 per coordinate. It *is* a mixture of 256 round Gaussians
  (std 0.03) evenly spaced along the arms, so the noised density, score, perfect denoiser E[x₀ | x_t] and flow
  velocity are exact at every noise level (`dr.mixture_score`, `dr.exact_denoiser`, `dr.exact_eps`,
  `dr.exact_velocity`). The two arms are the labels for guidance.
- **Schedule:** DDPM's linear β (1e-4 → 0.02, T = 1000) unless "cosine" is named. t = 0 is clean data.
- **Trained models:** one tiny MLP, trained twice in NumPy ({tr['eps']['params']:,} parameters; {tr['eps']['architecture']}).
  The input includes sinusoidal features of x itself (see caveats). ε-model: DDPM's simple loss, {tr['eps']['steps']:,} Adam steps ×
  batch {tr['eps']['batch']} of fresh swirl points ({tr['eps']['points_seen'] / 1e6:.1f} M points), EMA {tr['eps']['ema_decay']},
  {tr['eps']['train_seconds']:.0f} s. Flow model: the same with the flow-matching loss, {tr['flow']['train_seconds']:.0f} s.
- **Shared noise:** every sampler comparison starts from the same {sp['noise']}.
- **Metrics** (`dr.swirl_distance`, `dr.off_swirl`, `dr.swirl_coverage`):
  - *dist* is the mean distance to the swirl's centre line. The data scores {data['dist']}, and that is the target, not 0.
  - *off* is the fraction of samples more than 0.1 from the curve (stragglers between the arms). The data scores {pct(data['off'])}.
  - *coverage* is the fraction of the 40 equal pieces of the curve that get at least a quarter of their fair share. The data scores {data['coverage']}.
  - *fd* is FID's formula on raw 2D coordinates against 20,000 data points.
  - *ode error* is the mean distance to the exact ODE solution from the same noise (Heun, 1,000 Karras steps).
""")

# ------------------------------------------------------------------------------------------------ 1
lin = {row["t"]: row for row in fw["linear"]["table"]}
cos = {row["t"]: row for row in fw["cosine"]["table"]}
rows = []
for t in fw["t"]:
    a, c = lin[t], cos[t]
    rows.append([t, a["beta"], a["alpha_bar"], a["signal"], a["noise"], a["snr"], a["sigma"], a["arm_accuracy"],
                 c["alpha_bar"], c["snr"], c["arm_accuracy"]])
add(f"""## 1. The forward process

x_t = √ᾱ_t·x₀ + √(1 − ᾱ_t)·ε. *arm accuracy* = how often a perfect detective (the exact posterior p(arm | x_t))
names the arm a noisy point came from; 0.5 is a coin flip. (20,000 points.)

{table(["t", "β_t", "ᾱ_t", "√ᾱ (signal)", "√(1−ᾱ) (noise)", "SNR", "σ", "arm accuracy", "cosine ᾱ_t", "cosine SNR", "cosine arm acc."], rows)}

- Signal and noise are equal (SNR = 1, ᾱ = ½) at **t = {fw['linear']['t_snr_1']:.0f}** (linear) and **t = {fw['cosine']['t_snr_1']:.0f}** (cosine).
  ᾱ drops below 1 % at t = {fw['linear']['t_alpha_bar_below_1pct']} (linear) and t = {fw['cosine']['t_alpha_bar_below_1pct']} (cosine).
- Linear ᾱ_1000 = {fw['linear']['alpha_bar_T']} (pure noise to 4 parts in 100,000); cosine ᾱ_1000 = {fw['cosine']['alpha_bar_T']}
  (exactly 0 in the formula; Nichol & Dhariwal clip β at 0.999, which is why the last cosine β is 0.999).
- Snapshots at t = {', '.join(map(str, fw['snapshot_t']))} for both schedules: `toy.json → forward/<schedule>/snapshots/<t>`
  (1,000 points, the same x₀ and ε at every t). Recipe to recompute them exactly: `{fw['snapshot_recipe']}`.
  Curves of ᾱ_t and log SNR every 10 steps: `forward/<schedule>/curve_alpha_bar`, `curve_log_snr`.

**On screen:** "Around step 260 of 1,000 the swirl is half signal, half noise. By step 500 only 8 % of the signal
variance is left, and even a perfect detective can tell which arm a point came from just {pct(lin[500]['arm_accuracy'], 0)} of the
time, barely better than a coin. The cosine schedule keeps the picture alive twice as long: at step 500 it still has half its signal."

**Caveats:** "8 % of the signal" means ᾱ (signal *variance*); the signal *amplitude* √ᾱ is 28 %. Arm accuracy is
already down to {pct(lin[50]['arm_accuracy'], 0)} at t = 50 because the arms interleave: a little noise is enough to confuse
neighbouring turns.
""")

# ------------------------------------------------------------------------------------------------ 2
rows = [[row["t"], row["sigma"], row["loss_floor"], row["loss_mlp"], row["excess"], pct(row["rel_score_error"]), f"{row['median_angle_deg']}°"]
        for row in sc["on_data"]]
grid_rows = [[g["t"], f"{g['median_angle_deg']}°", f"{g['median_angle_high_density_deg']}°", f"{g['median_angle_low_density_deg']}°"] for g in sc["grid_error"]]
add(f"""## 2. The exact score vs the trained MLP

The MLP predicts ε; its score is −ε̂/√(1 − ᾱ). Errors on 20,000 noisy points x_t drawn from p_t (where the model is
trained and used):

{table(["t", "σ", "ε-loss floor (perfect model)", "ε-loss of the MLP", "excess", "score error (relative RMS)", "median arrow angle"], rows)}

- **The loss can't reach 0.** Averaged over t ~ U{{1..1000}} the best possible ε-loss is **{sc['loss_floor_eps']}** (computed with
  the exact denoiser on 200,000 points); the MLP finished at **{tr['eps']['final_loss']}**, within {pct((tr['eps']['final_loss'] - sc['loss_floor_eps']) / sc['loss_floor_eps'])} of it.
  The flow loss floor is {sc['loss_floor_flow']} (MLP: {tr['flow']['final_loss']}). The floor is set by what the noisy point
  hides: which arm, and where across it, the point began.
- Loss curves (200 points each): `training/eps/loss_curve` with `loss_curve_steps`; the floor is the line to draw under them.
- On a 21 × 21 grid over [−1.5, 1.5]² (everywhere, including where no data goes) the arrows agree to:

{table(["t", "median angle (all grid)", "high-density 75 %", "lowest-density 25 %"], grid_rows)}

  Both fields at t = 10, 100 and 500 are stored as arrows: `score/fields/<t>/exact` and `…/mlp` (scores; the grid is
  `score/grid/axis`, all (x, y) pairs with x fastest). Normalise the arrows before drawing (on the grid the median arrow
  is ~100× longer at t = 10 than at t = 500).

**On screen:** "The arrows the network learned point the same way as the exact ones, within a few degrees. At low
noise the network's arrows are about {pct(sc['on_data'][1]['rel_score_error'], 0)} off in length and direction, and at high noise under 1 %."

**Caveats:** without the sinusoidal features of x in its input, the same MLP learned the score well above t ≈ 30 but
blurred it below (excess error 25 % of the floor at t = 10 vs about 1 % now), and its DDPM samples filled the gaps between the
arms (*dist* 0.056, 21 % off-swirl). Training it 8× longer (5,000 → 40,000 steps) only partly helped (*dist* 0.087 → 0.056). This is the "spectral bias" of plain MLPs; the fix
(Fourier features, Tancik et al. 2020) is the same sinusoid trick as the time embedding. Low-density regions were *not*
noticeably worse here: on a 2D toy the network generalises smoothly into the corners.
""")

# ------------------------------------------------------------------------------------------------ 3
order = ["ddpm_1000", "ddpm_1000_beta_tilde", "ddim_1000", "ddim_100", "ddim_50", "ddim_20", "ddim_10", "ddim_5",
         "euler_karras_100", "euler_karras_50", "euler_karras_20", "euler_karras_10", "euler_karras_5",
         "euler_t_100", "euler_t_50", "euler_t_20", "euler_t_10", "euler_t_5",
         "heun_karras_99", "heun_karras_49", "heun_karras_19", "heun_karras_9", "heun_karras_5", "heun_99", "heun_49", "heun_19", "heun_9", "heun_5"]
names = {"ddpm_1000": "DDPM, σ²=β", "ddpm_1000_beta_tilde": "DDPM, σ²=β̃", "ddim": "DDIM η=0 (= Euler in σ)",
         "euler_karras": "Euler, Karras σ grid", "euler_t": "Euler in t (VP ODE)", "heun_karras": "Heun, Karras σ grid", "heun": "Heun, uniform t"}


def label(key):
    if key.startswith("ddpm"):
        return names[key]
    base = key.rsplit("_", 1)[0]
    return names[base]


rows = []
for k in order:
    e, mm = se[k], sm[k]
    rows.append([label(k), e["nfe"], e["dist"], pct(e["off"]), e["coverage"], e.get("ode_error", "–"),
                 mm["dist"], pct(mm["off"]), mm["coverage"], mm.get("ode_error", "–"), mm["fd"], mm["seconds"]])
det_e, det_m = sp["exact"]["determinism"], sp["mlp"]["determinism"]
add(f"""## 3. Samplers: DDPM, DDIM, Euler and Heun

2,000 samples from the same noise; exact ε (left) and the trained MLP (right). Data: dist {data['dist']}, off {pct(data['off'])},
coverage {data['coverage']}, fd {data['fd']}. NFE = model calls. Karras grid: {sp['karras_grid']}.

{table(["sampler", "NFE", "exact: dist", "off", "cov.", "ode err", "MLP: dist", "off", "cov.", "ode err", "MLP fd", "MLP s"], rows)}

**Same noise, same point?** (MLP; exact in brackets)
- DDIM is deterministic: running DDIM-100 twice gives identical points (max difference {det_m['ddim100_repeat']}).
  DDIM-100 and DDIM-50 from the same noise land {det_m['ddim100_vs_ddim50']} apart on average ({det_e['ddim100_vs_ddim50']}), on the same arm
  {pct(det_m['ddim100_vs_ddim50_same_arm'], 0)} of the time; DDIM-10 lands {det_m['ddim100_vs_ddim10']} away, same arm {pct(det_m['ddim100_vs_ddim10_same_arm'], 0)}.
- DDPM forgets its starting noise: two runs from the same noise with different fresh noise land {det_m['ddpm_seed1_vs_seed2']} apart
  ({det_e['ddpm_seed1_vs_seed2']}), the same as two random swirl points ({det_m['random_pair_baseline']}), and on the same arm
  {pct(det_m['ddpm_seed1_vs_seed2_same_arm'], 0)} of the time, a coin flip.
- Trajectories of the first 8 noise points (≤ 200 states each): `samplers/<exact|mlp>/trajectories/<ddpm_1000|ddim_100|ddim_50|ddim_10|heun_karras_19>`
  (`xs` per particle, `ts` the DDPM step of each state); DDPM's clean guesses x̂₀ along the way: `samplers/exact/ddpm_x0_guesses`.
  End clouds (1,000 points): `samplers/<model>/clouds/<run>`.

**On screen:**
- "A thousand small noisy steps rebuild the swirl: the trained model's samples sit {sm['ddpm_1000']['dist']} from the curve on
  average, the data itself {data['dist']}."
- "DDIM takes the same model and walks deterministically: 100 steps come close ({pct(sm['ddim_100']['off'], 1)} stranded vs
  {pct(sm['ddpm_1000']['off'], 1)} for DDPM's 1,000); at 50 steps {pct(sm['ddim_50']['off'], 0)} of points
  get stranded between the arms, at 20 steps {pct(sm['ddim_20']['off'], 0)}, at 10 steps {pct(sm['ddim_10']['off'], 0)}."
- "Where you put the steps matters as much as how many: with Karras's spacing, 20 Euler steps leave only
  {pct(sm['euler_karras_20']['off'], 0)} stranded, and Heun with 49 model calls ({pct(sm['heun_karras_49']['off'], 1)}) is as clean as DDPM's 1,000."

**Caveats:**
- *dist below the data's {data['dist']} is not better.* DDIM-100 scores {sm['ddim_100']['dist']} because its last step returns the
  denoiser's average guess, which is tidier than real data (fine detail at the scale of the last step is averaged away;
  for images this is the slightly smooth look of deterministic samplers). DDPM with σ² = β̃ is also slightly too tidy
  ({sm['ddpm_1000_beta_tilde']['dist']}); σ² = β matches the data.
- *ode error vs quality:* Euler on the Karras grid has the same ODE error as DDIM at the same NFE, but far fewer stragglers.
  Its errors move points *along* the arms, which you cannot see; DDIM's big last jump (from σ = 0.17 at 20 steps) drops them
  *between* the arms.
- *Heun is not always better.* On the Karras grid it is the best solver from NFE ≈ 49 up (ODE error {se['heun_karras_49']['ode_error']} vs
  DDIM-50's {se['ddim_50']['ode_error']}), but at NFE 9 and 5 it diverges ({se['heun_karras_9']['dist']} and {se['heun_karras_5']['dist']} from the curve): its
  second look at a very low σ, where the swirl's score is steep, overshoots. On the uniform t grid Heun is worse than
  DDIM at equal NFE, because half as many steps means a bigger final Euler jump.
- *Euler in t vs in σ:* on this toy, Euler on the VP ODE written in DDPM time is *not* worse than DDIM by ODE error
  (it is somewhat worse by stragglers at NFE ≥ 50). Don't claim on screen that the σ coordinate is what makes DDIM good;
  the robust lesson here is step placement.
- *fd on raw 2D coordinates is a poor judge:* Euler-in-t with 5 steps scores fd {sm['euler_t_5']['fd']} (as good as DDPM) while
  {pct(sm['euler_t_5']['off'], 0)} of its points are off the swirl. See §7.
""")

# ------------------------------------------------------------------------------------------------ 4
tp = fl["training_paths"]
rows = []
for s in ["1", "2", "4", "8", "16", "32"]:
    e, t_ = fl["exact"]["quality_vs_steps"][s], fl["trained"]["quality_vs_steps"][s]
    rows.append([s, e["flow"]["dist"], pct(e["flow"]["off"]), e["ddim"]["dist"], pct(e["ddim"]["off"]),
                 t_["flow"]["dist"], pct(t_["flow"]["off"]), t_["flow"]["coverage"], t_["ddim"]["dist"], pct(t_["ddim"]["off"]), t_["ddim"]["coverage"]])
fe, ft = fl["exact"], fl["trained"]
add(f"""## 4. Flow matching vs the DDPM-trained model

Same noise, Euler steps evenly spaced in each model's own time (flow: t from 1 to 0; DDIM: DDPM steps from 1000 to 0).

{table(["Euler steps", "exact flow: dist", "off", "exact DDIM: dist", "off", "trained flow: dist", "off", "cov.", "trained DDIM: dist", "off", "cov."], rows)}

**Path straightness** (path length ÷ distance between its ends; 1 = a straight line):
- *Training paths* (what each loss sees; 5,000 (x₀, ε) pairs): flow matching's are exactly straight
  ({tp['straightness_flow']['mean']}); DDPM's are arcs, x_t = √ᾱ·x₀ + √(1−ᾱ)·ε: median {tp['straightness_ddpm']['median']}, mean {tp['straightness_ddpm']['mean']},
  10 % above {tp['straightness_ddpm']['p90']} (some loop when x₀ and ε point the same way). 8 of each: `flow/training_paths/paths_flow`, `paths_ddpm`.
- *Sampling paths* (200 Euler steps; trained models): flow median {ft['straightness_flow']['median']} (mean {ft['straightness_flow']['mean']}), DDPM ODE in its
  x_t picture median {ft['straightness_ddpm_ode']['median']} (mean {ft['straightness_ddpm_ode']['mean']}); exact fields: {fe['straightness_flow']['median']} vs
  {fe['straightness_ddpm_ode']['median']}. Paths of 8 particles: `flow/<exact|trained>/paths_flow`, `paths_ddpm_ode`.
- With exact fields, flow Euler **is** DDIM on a different step grid (σ = t/(1 − t)): after 16 steps the two differ by at most
  {fl['exact_flow16_vs_ddim_on_sigma_grid_max_diff']} (the self-checks test this). Both ODEs carry each noise point to the same place
  along the same route, drawn in different coordinates.

**On screen:** "Flow matching trains on straight lines from noise to data, while DDPM's training paths are arcs. Sampled with 8 or
more plain Euler steps, the flow model is much cleaner: with 8 steps {pct(ft['quality_vs_steps']['8']['flow']['off'], 0)} of its points are stranded vs
{pct(ft['quality_vs_steps']['8']['ddim']['off'], 0)} for the DDPM model, with 16 steps {pct(ft['quality_vs_steps']['16']['flow']['off'], 0)} vs {pct(ft['quality_vs_steps']['16']['ddim']['off'], 0)}, with 32 steps {pct(ft['quality_vs_steps']['32']['flow']['off'], 0)} vs {pct(ft['quality_vs_steps']['32']['ddim']['off'], 0)}.
One step sends every point to the middle, the average of the data (dist {fe['quality_vs_steps']['1']['flow']['dist']})."

**Caveats (important for Ep 11):**
- **The sampling paths are not straighter.** The flow model's actual sampling paths bend about as much as the DDPM ODE's
  (by length ÷ chord they bend *more* in the flow picture, median {ft['straightness_flow']['median']} vs {ft['straightness_ddpm_ode']['median']}), because every path first heads for the data's average and
  then turns toward its arm. Show "straight vs curved" with the *training* paths, or say "straighter on average, as trained".
  Truly straight sampling paths need reflow (rectified flow's second round) or distillation (Ep 12).
- With 2 and 4 steps both are poor (40–60 % stranded), and at 4 steps the DDPM model is slightly ahead; don't say "better at every budget".
- One DDIM step with the *trained* ε-model lands {ft['quality_vs_steps']['1']['ddim']['dist']} from the swirl (the exact ε: {fe['quality_vs_steps']['1']['ddim']['dist']}).
  The clean guess at t = 1000 is (x − √(1−ᾱ)·ε̂)/√ᾱ with √ᾱ = 0.0064, so a tiny error in ε̂ is magnified 157×. ε-prediction is
  ill-conditioned at pure noise; the flow model's velocity is not (one step: {ft['quality_vs_steps']['1']['flow']['dist']}). This is one reason for v-prediction and flow
  targets (Ep 5, 11). The same effect makes the trained model's Heun-5 runs blow up in §3.
- With perfect models the per-step advantage comes entirely from *where the steps fall*: uniform flow time puts more steps
  at low noise than uniform DDPM time under the linear β schedule. The trained models track the exact results closely,
  so on this toy the learned fields are not what differs.
""")

# ------------------------------------------------------------------------------------------------ 5
g = gd["runs"]
outer = {w: sum(p[10:]) / max(sum(p), 1e-12) for w, p in gd["profiles"].items()}
rows = [[w, pct(g[w]["on_arm0"], 0), pct(g[w]["on_arm1"], 0), pct(g[w]["off"], 1), g[w]["coverage_arm0"], pct(outer[w], 0), g[w]["spread"],
         g[w]["max_radius"], pct(g[w]["ddpm_check_on_arm0"], 0)] for w in ["0", "1", "3", "7"]]
prof7 = g["7"]
gb = gd["blobs"]
rowsb = [[w, pct(gb["runs"][w]["class0"], 0), gb["runs"][w]["centre_y"], gb["runs"][w]["std"], pct(gb["runs"][w]["beyond_2sd"], 0), pct(gb["runs"][w]["beyond_3sd"], 1)]
         for w in ["0", "1", "3", "7"]]
add(f"""## 5. Classifier-free guidance, exact

ε̃ = ε_u + w·(ε_c − ε_u), with both ε's exact (`dr.guided_eps(w, label)`). w is the *guidance scale* (Stable Diffusion,
diffusers): w = 0 ignores the label, w = 1 is the plain conditional model. Ho & Salimans' paper writes (1 + w′)·ε_c − w′·ε_u, so
**their w′ = this w − 1** (the course's w = 3 is their 2). {gd['sampler']}; DDPM-1000 (500 points) confirms the arm shares.

**The swirl, asking for arm 0:**

{table(["w", "on arm 0", "on arm 1", "off", "coverage of arm 0", "arm-0 samples on its outer half", "spread (√trace cov)", "max radius", "DDPM: on arm 0"], rows)}

Arm 0's own data: spread {gd['arm0_data']['spread']}, coverage {gd['arm0_data']['coverage_arm0']}. Where the samples sit along arm 0
(20 equal pieces, centre → tip, as shares of all samples; `guidance/profiles/<w>`): at w = 7, {pct(outer['7'], 0)} sit on the outer half
(data: 50 %), the part of arm 0 that looks least like arm 1, and the outermost piece (the tip) is nearly empty. Clouds: `guidance/clouds/<w>`.

**Three labelled blobs, asking for class 0 (the top blob):** ({gb['setup']})

{table(["w", "class 0", "centre height (data 0.60)", "std (data " + str(gb['data_class0']['std']) + ")", "beyond 2 std of the class", "beyond 3 std"], rowsb)}

Class 0's own data: {pct(gb['data_class0']['beyond_2sd'], 0)} beyond 2 std, {pct(gb['data_class0']['beyond_3sd'], 1)} beyond 3 std. Clouds: `guidance/blobs/clouds/<w>`,
data `guidance/blobs/data_cloud`.

**On screen:** "Guidance trades variety for certainty. At w = 1 we get arm 0, all of it. At w = 7 we still get arm 0,
but only its outer half ({pct(outer['7'], 0)} of the samples): they crowd where the arm looks least like the other one, and their spread falls from
{g['1']['spread']} to {g['7']['spread']}. With blobs you can see the overshoot: at w = 7 the samples' centre moves from height {gb['runs']['1']['centre_y']} to {gb['runs']['7']['centre_y']}, and
{pct(gb['runs']['7']['beyond_2sd'], 0)} of them lie outside where the class's own data lives ({pct(gb['data_class0']['beyond_2sd'], 0)} for real data). They are pushed beyond the data,
away from the other classes. That is the over-saturated look of high guidance."

**Caveats:** on the swirl high guidance does *not* push points off the data (off stays 0, max radius {prof7['max_radius']}): the thin arm
re-captures them at low noise, where the classifier is certain and its push vanishes. Use the blobs for "beyond the data".
The DDIM-100 samples are a little tidier than the data (blob std {gb['runs']['1']['std']} at w = 1 vs {gb['data_class0']['std']}), as in §3.
""")

# ------------------------------------------------------------------------------------------------ 6
rows = []
for k, name in (("annealed", "annealed, 10 levels σ = 1 → 0.01"), ("plain_small", "plain, σ = 0.01"), ("plain_large", "plain, σ = 0.3")):
    v = lv["runs"][k]
    rows.append([name, v["steps"], v["dist"], pct(v["off"]), v["coverage"], pct(v["outer_half_share"], 0), v["profile_tv"], pct(v["arm0_share"], 0)])
add(f"""## 6. Langevin dynamics with the exact score

x ← x + (α/2)·∇log p_σ(x) + √α·z, starting from {lv['start']}. Settings: annealed {lv['settings']['annealed']};
plain {lv['settings']['plain_small']}; blurry {lv['settings']['plain_large']}.

{table(["run", "steps", "dist", "off", "coverage", "share on the outer half (data " + pct(lv['data_outer_half_share'], 0) + ")", "TV from even (data " + str(lv['data_profile_tv']) + ")", "arm 0 share"], rows)}

Clouds and 8 particle paths (≤ 200 states): `langevin/clouds/<run>`, `langevin/paths/<run>`.

**On screen:** "Follow the exact arrows at one small noise level and every particle slides onto the nearest bit of
swirl, but the outer turns, which are closest to most starting points, collect {pct(lv['runs']['plain_small']['outer_half_share'], 0)} of them instead of half. The proportions
are wrong. Start with big noise and turn it down slowly, and the particles
first spread over the right broad shape, then sharpen: {pct(lv['runs']['annealed']['outer_half_share'], 0)} on the outer half, the same as the data."

**Caveats:** with the *exact* score, plain Langevin at small σ still lands on the curve (dist {lv['runs']['plain_small']['dist']}); its failure here is
the proportions. With a *learned* score it would also be steered wrongly in empty regions (Song & Ermon's other reason
for annealing); our MLP is accurate even there (§2), so this toy does not show that part. A single large σ samples the
blurred swirl ({pct(lv['runs']['plain_large']['off'], 0)} off).
""")

# ------------------------------------------------------------------------------------------------ 7
s = fd["sanity"]
rows = []
for d in ("d2", "d64"):
    for n, v in fd["bias"][d]["rows"].items():
        rows.append([d[1:], f"{int(n):,}", f"{v['vs_truth_mean']} ± {v['vs_truth_std']}", f"{v['two_sample_mean']} ± {v['two_sample_std']}"])
bs = fd["blind_spot"]
add(f"""## 7. FID's formula on Gaussians

`dr.frechet_distance(μ1, C1, μ2, C2)` = ‖μ1 − μ2‖² + Tr(C1 + C2 − 2(C1·C2)^½). The matrix square root uses eigh on the
symmetric matrix √C1·C2·√C1, which has the same eigenvalues as C1·C2.

- Sanity: {s['case']}: closed form ‖Δμ‖² + Σ(√a − √b)² = **{s['closed_form']}**, computed **{s['computed']}**. Rotating the second
  covariance by 0.6 rad (so the matrices no longer commute): {s['rotated_case_computed']} by our formula, {s['rotated_case_via_eig_of_product']} via eigenvalues of C1·C2.
- **Bias with few samples** (true distribution N(0, I); FD between the samples' mean/covariance and the truth, or between
  two sample sets of the same size; mean ± std over {fd['bias']['d2']['repeats']} (2D) / {fd['bias']['d64']['repeats']} (64-D) repeats). The true value is 0 in every row:

{table(["dims", "samples", "FD(sample, truth)", "FD(sample, sample)"], rows)}

- **The blind spot:** a Gaussian with exactly the swirl's mean and covariance scores FD **{bs['fd_swirl_vs_matched_gaussian']}** against 10,000
  swirl points (another 10,000 real swirl points: {bs['fd_swirl_vs_swirl']}); both are inside the estimator's own noise at n = 10,000. Yet
  {pct(bs['gaussian_quality']['off'], 0)} of the Gaussian's points are off the swirl (dist {bs['gaussian_quality']['dist']}). Cloud: `fid/blind_spot/gaussian_cloud`.

**On screen:** "FID only compares averages and spreads. A shapeless blob with the swirl's average and spread scores as well as the
swirl itself. And it is biased: with 100 samples, two identical distributions already score {fd['bias']['d2']['rows']['100']['vs_truth_mean']} in 2D. In 64 dimensions
the bias is {fd['bias']['d64']['rows']['1000']['vs_truth_mean']} even with 1,000 samples, which is why FID is quoted on 50,000 images and only compared at equal sample counts."

**Caveats:** for N(0, I) the expected bias is ≈ (d + d(d + 1)/4)/n (0.0035 for 2-D and 1.1 for 64-D at n = 1,000, matching the
table): it falls as 1/n and grows like d², and Inception features have d = 2048. The two-sample estimate is about twice as
biased as comparing with exact statistics.
""")

# ------------------------------------------------------------------------------------------------ headline
add("## Headline numbers (`measured(\"toy\", \"headline\", key)`)\n")
add(table(["key", "value"], [[f"`{k}`", v] for k, v in h.items()]))
add(f"""

## General caveats

- Every number is from one fixed noise draw ({sp['noise']}) or seed; sampler comparisons share the noise, so differences
  between rows are much more reliable than any single value. Re-running the scripts reproduces every number in toy.json
  except the wall times (checked; same NumPy build).
- The swirl is 2D and its data is an exact mixture, so "the exact denoiser" really exists here; for images it never does.
  Use the toy for mechanisms (which way arrows point, why steps strand points, what guidance does), not for image-model numbers.
- The DDPM step grids are "trailing" (S steps: t = round(linspace(1000, 0, S + 1))). Other spacings (diffusers' "leading")
  change the low-step numbers.
""")

(HERE / "toy.md").write_text("\n".join(md))
print(f"wrote {HERE / 'toy.md'}")
