"""Write one subagent brief per episode (briefs/epNN.md) for "Out of the Noise", from the rows of COURSE.md,
the per-episode notes in SPEC (briefs/spec.py), briefs/pronunciation.txt and the series notes below.
Run: python3 briefs/make_briefs.py [N ...]"""

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERIES_DIR = HERE.parent
VIDEOS = SERIES_DIR.parent
sys.path.insert(0, str(HERE))
from spec import SPEC  # noqa: E402

NARRATOR_PYTHON = "~/rag-video/tts/.venv/bin/python"
MODULE_NAMES = {1: "PICTURES FROM NOISE", 2: "LEARNING TO DENOISE", 3: "FASTER SAMPLING", 4: "STEERING THE PICTURE",
                5: "SCALING UP", 6: "VIDEO AND YOUR MAC"}


def _scene(series, n):
    hit = sorted(VIDEOS.glob(f"{series}/ep{n:02d}-*/scene.py"))
    return str(hit[0]) if hit else None


AI = {35: "diffusion in one episode (the sister course's overview)", 24: "attention & transformers", 25: "CLIP / multimodal"}
AG = {}
IN = {2: "a measured number paired with its ceiling", 13: "steps vs quality trade-offs, measured", 23: "everything measured on this Mac",
      24: "plots and a step-by-step calculation"}
RL = {1: "an opener with a course map", 9: "a spark between neighbours, measured curves", 20: "real model outputs as token chips",
      23: "one real group from a run, assertions against logs", 26: "a finale built on a real run, course-closing card"}


def references(keys):
    out = []
    for key in keys:
        kind, n = key[:2], int(key[2:])
        if kind == "ag":
            path, what, label = _scene("agent-series", n), AG.get(n, ""), "'Agents, Animated' (sister course, OLD palette)"
        elif kind == "ai":
            path, what, label = _scene("ai-series", n), AI.get(n, ""), "'AI, Explained' (sister course, OLD palette)"
        elif kind == "in":
            path, what, label = _scene("inference-series", n), IN.get(n, ""), "'Tokens per Second' (sister course, OTHER palette)"
        elif kind == "rl":
            path, what, label = _scene("rl-series", n), RL.get(n, ""), "'Trial and Reward' (the latest sister course: match its polish, NOT its palette)"
        else:
            path, what = _scene("diffusion-series", n), "THIS course, THIS palette — match it"
            label = f"this course's Episode {n}"
            if not path:
                out.append(f"{label} (not made yet — skip)")
                continue
        out.append(f"{path} — {label}: {what}")
    return "; ".join(out)


TEMPLATE = """You are producing ONE narrated animated tutorial for the course "Out of the Noise" — how diffusion models make pictures and video (Mac, Apple Silicon). Work fully autonomously and finish with a short report.

YOUR EPISODE: Episode {N} · "{TITLE}" (Module {M} · {MODULE_NAME}). Folder: {SERIES_DIR}/ep{NN}-{slug}/ (create it). Scene class: `Ep{NN}{ClassName}(Episode)` with `number = {N}`.
Goal (what the viewer will understand): {GOAL}
Key concepts: {CONCEPTS}
Mechanism to animate: {MECHANISM}
Running example: {EXAMPLE}
EPISODE NOTES: {NOTES}

READ FIRST, in this order:
1. ~/.claude/skills/tutorial-video/SKILL.md, then reference/narration.md, reference/design.md, reference/manimgx.md (gotchas 19–31 matter), reference/series.md.
2. {SERIES_DIR}/COURSE.md (the whole course: what comes before and after yours — link to other episodes by number, never repeat them at length).
3. {SERIES_DIR}/kit.py — the visual system (dark "darkroom" stage; Baskerville titles, Avenir Next text, Fira Mono code) and the diffusion components: picture(array, height) (an image array or PNG as an ImageMobject; small images get crisp pixels), tint(grey, colour), framed(), strip() (a row of framed pictures), noise_image(), Plot (curves, bands, log axes, .cloud(xy) → Cloud with .to(xy)/.replay(path) for 2D point clouds, .field(fn) arrow fields), Bars, NeuralNet, UNet (.flow()), denoise_loop(), pixel() (the mascot), heat/mix, plus code_block/terminal/json_card/message/token_row/stat/cell_grid/Meter/tex; header via self.new_header(tag, title). Look at its storyboards ({SERIES_DIR}/KitDemo.storyboard*.png, already current — do not re-run kit_demo.py, it writes outside your folder). Images are Group members (not VMobjects): put them in m.Group, not m.VGroup.
4. {SERIES_DIR}/diffusion_reference.py — the canonical algorithms and numbers (numpy, seeded, self-checked: schedules, q_sample, the swirl and its exact score/denoiser, a trained swirl MLP, samplers DDPM/DDIM/Euler/Heun/flow/Langevin with trajectories, cfg, guided_eps and the three-blob toy, frechet_distance, load_mnist(split) for real MNIST digits, measured()). Read its docstring and the functions you use. Import it in your scene (`from diffusion_reference import *` after the sys.path line) and COMPUTE every number you show from it (fast, seeded calls are fine at render time) or load it with measured(...); any code you show is an excerpt (≤ 10 lines) in its style and names.
5. {SERIES_DIR}/research/notes.md — sections: {NOTES_SECTIONS}; plus its Appendix A (where sources disagree — use the version it recommends) and Appendix C (your episode's must-get-right numbers). And the runs on THIS Mac (Apple M4 Max, 64 GB): {SERIES_DIR}/research/experiments/toy.md (the swirl) and mac_runs.md (the digit model and the lighthouse; images in research/experiments/images/, arrays in research/experiments/arrays/ — load with numpy) — never re-run the experiments yourself. Open the original sources the notes cite when you need more detail (WebFetch).
6. Style references (read their scene.py and look at a few storyboard PNGs; sister courses share the engine and pacing but use OTHER palettes — copy structure and care, NOT colours or fonts): {REFERENCE_EPISODES}.

Match the series: one running example carried through the episode (the swirl, the digit model or the lighthouse — whichever teaches best; say which), a header (mono tag + Baskerville title via self.new_header) per section, sections via self.next_section(...), ~38–46 narrated self.say(...) lines, 3.5–4.5 minutes, a three-word recap near the end. Show mechanisms MOVING: noise dissolving pictures and pictures emerging from noise, point clouds flowing, arrow fields, curves being drawn, steps counting. Every number on screen is computed in the scene (from diffusion_reference / measured() / the run's arrays) and asserted, or cited with its source; invented data is said "for example" and recorded as illustrative in the section notes. Real generated images from our runs are labelled "generated on this Mac" (with the model and step count).

HARD RULES (other agents work in the same series folder at the same time):
- Write only inside your episode folder. Do NOT edit kit.py, diffusion_reference.py, COURSE.md, lexicon.json, research/, other episodes, or existing files in {SERIES_DIR}/assets/. You MAY add NEW icon SVGs to assets/ named `ep{NN}-*.svg` (flat 2-tone, this palette — see kit.py), and recoloured copies via kit's recolored(name, color, fill, prefix="ep{NN}").
- Do not run training, models or benchmarks (no new experiments, no downloads): the runs are done and shared. Calling diffusion_reference functions in the scene is fine (keep render-time compute under ~10 s, or precompute into your folder and assert).
- The narrator is ALREADY RUNNING at http://127.0.0.1:8765/speak, shared by several agents (requests queue). NEVER start, stop or restart it. If a request fails, wait a minute and retry. Its log is {SERIES_DIR}/narrator.log (shared, appended).
- Python with Whisper: {NARRATOR_PYTHON}. ManimGX CLI: manimgx. Keep each response and tool output compact (pipe long logs through tail/grep): long turns time out. In the shell, `ls` is aliased to eza, which can hang in a background shell: use `/bin/ls`.

PROCESS:
1. Write the script + scene. 2. Draft pass: `cd <folder> && VOICE=draft manimgx inspect scene.py` until exit 0 and "layout: no problems in N samples"; READ EVERY storyboard PNG and fix what you see (overlaps, cramped or empty frames, text over lines, misaligned baselines, faint or half-drawn objects, anything closer than 0.5 units to the frame edge). 3. Voice pass: `manimgx inspect scene.py`. For a line below 100% after 4 takes, transcribe the clip (librosa 16 kHz → mlx_whisper, path_or_hf_repo="mlx-community/whisper-large-v3-turbo") to decide: checker false alarm (keep) or real misread (reword, `python3 ~/.claude/skills/tutorial-video/scripts/drop_lines.py voice '<regex>'`, re-run inspect). 4. `python3 ~/.claude/skills/tutorial-video/scripts/drop_lines.py voice . --voice-id draft_voice`. 5. `manimgx render scene.py`, then `manimgx present scene.py --no-open`. 6. From INSIDE your folder: `{NARRATOR_PYTHON} ~/.claude/skills/tutorial-video/scripts/verify_audio.py <Scene>.mp4 --terms AI,<every ALL-CAPS acronym you say>`; and a sync check of 2 cues with `manimgx inspect scene.py -t a,b -o /tmp/ep{NN}_sync.png` (Read it).

FINAL REPORT (short): paths of MP4/SRT/HTML/scene.py; duration; layout result; verify_audio summary; lines reworded or below 100% and why; new icons/helpers; facts and numbers you checked (with sources) and anything you are unsure about.
"""

COMMON = """
SERIES NOTES (apply to every episode):
- Visual language (keep it consistent — viewers learn it once): the colours carry meaning across the course — noise VIOLET · data and clean images TEAL · the model / denoiser AMBER · time and the noise schedule BLUE · text and conditioning PINK · scores, velocities and flows LIME · good GREEN · bad RED · structure EDGE/MUTED. kit.py names them NOISE, DATA, MODEL, TIME, PROMPT, FLOW (also SCHEDULE, CONDITION, SCORE, VELOCITY, DENOISER). Headers use your module's accent (kit picks it). Display text (titles, big numbers, recap words) in title_text (Baskerville Bold); reading text in text() (Avenir Next); code, units, tables of numbers, commands and exact strings in code() / code_block / json_card / terminal (Fira Mono). Grey images can be tinted with tint(img, colour): noise in VIOLET, clean digits in TEAL; real colour images (the lighthouse) are shown as they are.
- The three running examples: the swirl (2D points on a spiral; exact scores and denoisers from diffusion_reference), the digit model (a small class-conditional diffusion model trained on MNIST on this Mac — research/experiments/mac_runs.md) and the lighthouse (a real latent text-to-image model run on this Mac, prompt "a lighthouse on a rocky coast at dusk, oil painting"). The mascot Pixel (pixel()) appears small, as a presence, not in every frame.
- Facts: prefer research/notes.md (each fact is sourced and dated) and primary sources (papers). Show "Source, Year" in small MUTED text (font_size ≥ 20) near a claim that needs one, e.g. "Ho et al., 2020", "Song et al., 2021", "Rombach et al., 2022". Fast-moving facts (model releases, video models) are dated (October 2026) in the section notes.
- Measured runs: load them with measured(...) or the run's arrays and say "trained on this Mac" / "generated on this Mac" on screen, with the seed and step counts. Never round a measured number into a stronger claim; if a result is surprising or weak (a small model's blurry digits, a metric that is only "FID-like"), say so honestly — that is part of the lesson.
- No real logos, screenshots or photographs of people; show our own generated images, our own arrays and simple drawings. Product and paper names are fine as text.
- Numbers as words in narration ("a thousand steps", "zero point five", "sixty-four by sixty-four by four"); digits are fine on screen. Say symbols as words: "x zero", "x t", "epsilon", "alpha bar", "sigma", "the guidance scale w".
{PRONUNCIATION}
- ManimGX: kit.py already fixes TransformFromCopy/FadeTransform early-show and parts animated after their group's intro in the same line. Still: never Indicate/Wiggle on a LATER cue an object whose intro may still be running (use Circumscribe); probe any doubtful line with inspect -t just before/after its cue. Within one line, fade out the SAME group object you faded in (gotcha 24). Chained .animate.shift(v) in one Succession land wrong: use absolute move_to (gotcha 25). Never name a scene method or attribute voice, film, say or clock (gotchas 26, 31), and never override a mobject method you didn't check (ManimGX uses names like _place internally). Axes ranges are cut to whole steps (gotcha 27) — prefer kit's Plot. Store your own geometry under your own names, never in .width/.height (gotcha 30). ManimGX has no DashedArrow. `from diffusion_reference import *` exports HERE (the SERIES folder) and EXPERIMENTS: name your episode folder something else (e.g. EP_DIR = Path(__file__).resolve().parent, set AFTER the import) — two episodes wrote files into the series folder this way. The downloaded Stable Diffusion models (HF cache) are DELETED at the end of the course: a scene must never read the HF cache at render time — hard-code any value you read there (with a provenance comment) or save it to a small json in your episode folder.
- Text: Text() strips leading spaces of a single line (indent by position; multi-line strings keep indentation). Baskerville and Avenir merge "fi"/"fl" into one glyph: glyph-indexing helpers (glyphs(), paint()) are for mono text only. code() / code_block / json_card / terminal colour code by regex already.
- Margins: keep every object ≥ 0.5 units inside the frame edges (the layout check only catches cut-offs, not text flush against the edge), and make sure cards placed late never cover earlier arrows or labels still on screen.
"""


def course_rows():
    rows = {}
    for line in (SERIES_DIR / "COURSE.md").read_text().splitlines():
        cells = [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", line.strip().strip("|"))]
        if len(cells) == 7 and cells[0].isdigit():
            rows[int(cells[0])] = dict(title=cells[1].strip("*"), goal=cells[2], concepts=cells[3], mechanism=cells[4],
                                       example=cells[5])
    return rows


def module_of(n):
    src = (SERIES_DIR / "kit.py").read_text()
    return {int(a): int(c) for a, b, c in re.findall(r"(\d+): \(\"(.+?)\", (\d)\)", src)}[n]


def brief(n, rows):
    slug, cls, refs, sections, notes = SPEC[n]
    row, module = rows[n], module_of(n)
    pron = (HERE / "pronunciation.txt").read_text().strip() if (HERE / "pronunciation.txt").exists() else ""
    fill = {
        "{N}": str(n), "{NN}": f"{n:02d}", "{TITLE}": row["title"], "{M}": str(module),
        "{MODULE_NAME}": MODULE_NAMES[module], "{SERIES_DIR}": str(SERIES_DIR), "{slug}": slug,
        "{ClassName}": cls, "{GOAL}": row["goal"], "{CONCEPTS}": row["concepts"],
        "{MECHANISM}": row["mechanism"], "{EXAMPLE}": row["example"], "{NOTES}": notes,
        "{NOTES_SECTIONS}": sections, "{REFERENCE_EPISODES}": references(refs), "{NARRATOR_PYTHON}": NARRATOR_PYTHON,
    }
    text = TEMPLATE
    for key, value in fill.items():
        text = text.replace(key, value)
    leftover = re.findall(r"\{(?:[A-Z_]+|slug|ClassName)\}", text)
    assert not leftover, (n, leftover)
    return text.rstrip() + "\n" + COMMON.replace("{PRONUNCIATION}", pron)


def main():
    rows = course_rows()
    for n in [int(a) for a in sys.argv[1:]] or sorted(SPEC):
        path = HERE / f"ep{n:02d}.md"
        path.write_text(brief(n, rows))
        print(path.name, len(path.read_text().split()), "words")


if __name__ == "__main__":
    main()
