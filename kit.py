"""The series kit for "Out of the Noise" (how diffusion models make pictures and video): design system, diffusion
components (pictures from arrays, point clouds, arrow fields, a U-Net diagram, the denoising loop), voice, and the
Episode base class with the series intro and next-episode outro. Episodes import it:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from kit import *  # noqa: F403
    from diffusion_reference import *  # noqa: F403  (the course's algorithms, numbers and measured runs)

    class Ep04AddingNoise(Episode):
        number = 4
        def lesson(self): ...

Run from an episode folder: `VOICE=draft manimgx inspect scene.py` (silent draft) or, with the narrator
serving from the series folder, `manimgx render scene.py`. A tour of every component: `kit_demo.py`.
"""

import base64
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

import manimgx as m
from manimgx.audio import Word

SERIES = Path(__file__).resolve().parent
ASSETS = SERIES / "assets"
TITLE_WIDTH = 12.5  # the widest an intro/outro card title may be (the frame is about 14.2 wide)

# ---------------------------------------------------------------------------- engine fixes
# (carried over from "AI, Explained" and "Agents, Animated": each fixes a ManimGX timing fault, see the
# tutorial-video skill's reference/manimgx.md, gotchas 19–29)


def _from_copy(mobject, target_mobject, **kwargs):
    """TransformFromCopy shows its finished target from the start of the line; a FadeIn from the source waits."""
    return m.FadeIn(target_mobject, target_position=mobject, **kwargs)


def _fade_transform(mobject, target_mobject, stretch=True, dim_to_match=1, **kwargs):
    """FadeTransform(Pieces) show the target early too; a cross-fade where each side glides waits for its cue."""
    return m.AnimationGroup(
        m.FadeOut(mobject, target_position=target_mobject),
        m.FadeIn(target_mobject, target_position=mobject),
        **kwargs,
    )


m.TransformFromCopy = _from_copy
m.FadeTransform = _fade_transform
m.FadeTransformPieces = _fade_transform


# A line that introduces a group and also moves or highlights a PART of it would show that part early:
# marking those animations as introducers keeps play() from adding them before their cue.
def _leaves(animation):
    subs = getattr(animation, "animations", None)
    if subs:
        for sub in subs:
            yield from _leaves(sub)
    else:
        yield animation


def _cue_safe(scene, animations):
    on_screen = {id(x) for x in scene.get_mobject_family_members()}
    roots, family = set(), set()
    for animation in animations:
        for leaf in _leaves(animation):
            if leaf.is_introducer() and getattr(leaf, "mobject", None) is not None:
                roots.add(id(leaf.mobject))
                family |= {id(x) for x in leaf.mobject.get_family()}
    for animation in animations:
        for leaf in _leaves(animation):
            mob = getattr(leaf, "mobject", None)
            if mob is None or leaf.is_introducer() or id(mob) in roots or id(mob) in on_screen:
                continue
            if id(mob) in family:
                leaf.introducer = True
    return animations


# Compiled under ManimGX's package path so play labels still point at the episode's own scene.py line.
_SAY = """
def say(self, script, *animations, voice=None):
    return _scene_say(self, script, *_cue_safe(self, animations), voice=voice)
"""
_say_namespace = {"_scene_say": m.Scene.say, "_cue_safe": _cue_safe}
exec(compile(_SAY, str(Path(m.__file__).parent / "_series_kit_say.py"), "exec"), _say_namespace)



# ---------------------------------------------------------------------------- design system
# A dark "darkroom" stage. Colours carry meaning across the whole course — learn them once:
#   noise VIOLET · data and clean images TEAL · the model / denoiser AMBER · time and the noise schedule BLUE
#   text and conditioning PINK · scores, velocities and flows LIME · good GREEN · bad RED · structure EDGE/MUTED

BG = "#101114"
CARD = "#181A20"
CARD_HI = "#1F222A"
EDGE = "#2E323C"
INK = "#EEF0F4"
MUTED = "#9AA1AE"
FAINT = "#565C68"
VIOLET, VIOLET_FILL = "#A78BFA", "#241C3D"
TEAL, TEAL_FILL = "#2DD4BF", "#0E2E2A"
AMBER, AMBER_FILL = "#FBBF24", "#3A2C0A"
BLUE, BLUE_FILL = "#60A5FA", "#132744"
PINK, PINK_FILL = "#F472B6", "#3B1530"
LIME, LIME_FILL = "#BEF264", "#26330E"
GREEN, GREEN_FILL = "#4ADE80", "#103320"
RED, RED_FILL = "#F87171", "#3B1515"
GOOD, GOOD_FILL = GREEN, GREEN_FILL
BAD, BAD_FILL = RED, RED_FILL
FILLS = {VIOLET: VIOLET_FILL, TEAL: TEAL_FILL, AMBER: AMBER_FILL, BLUE: BLUE_FILL, PINK: PINK_FILL,
         LIME: LIME_FILL, GREEN: GREEN_FILL, RED: RED_FILL, MUTED: CARD, EDGE: CARD, INK: CARD}

# what each colour means (use these names in scenes so the meaning stays fixed)
NOISE, DATA, MODEL, TIME, PROMPT, FLOW = VIOLET, TEAL, AMBER, BLUE, PINK, LIME
SCHEDULE, CONDITION, SCORE, VELOCITY, DENOISER = TIME, PROMPT, FLOW, FLOW, MODEL

ROLES = {"system": (MUTED, CARD), "user": (PINK, PINK_FILL), "assistant": (TEAL, TEAL_FILL)}

HEAD = "Baskerville"               # titles, headers, recap words (bold)
SANS = "Avenir Next"               # everything you read
MONO = "Fira Mono for Powerline"   # code, numbers in tables, units, labels: no ligatures, so "->", "!=" stay as typed

m.config.background_color = BG

MASCOT = "Pixel"  # the course's mascot (drawn by pixel()): half noise, half picture

# the course: module → (name, accent, fill); episode → (title, module). See COURSE.md.
MODULES = {
    1: ("PICTURES FROM NOISE", VIOLET, VIOLET_FILL),
    2: ("LEARNING TO DENOISE", AMBER, AMBER_FILL),
    3: ("FASTER SAMPLING", BLUE, BLUE_FILL),
    4: ("STEERING THE PICTURE", PINK, PINK_FILL),
    5: ("SCALING UP", TEAL, TEAL_FILL),
    6: ("VIDEO AND YOUR MAC", LIME, LIME_FILL),
}
EPISODES = {
    1: ("Out of the Noise", 1), 2: ("Images Are Numbers", 1), 3: ("Before Diffusion", 1), 4: ("Adding Noise", 1),
    5: ("Learning to Undo Noise", 2), 6: ("The Score", 2), 7: ("Sampling Step by Step", 2), 8: ("The U-Net", 2),
    9: ("Fewer Steps", 3), 10: ("Solvers and Schedules", 3), 11: ("Flow Matching", 3), 12: ("One Step at a Time?", 3),
    13: ("Asking for a Digit", 4), 14: ("Classifier-Free Guidance", 4), 15: ("Words to Pictures", 4),
    16: ("Editing and Control", 4),
    17: ("Latent Diffusion", 5), 18: ("Diffusion Transformers", 5), 19: ("Measuring Pictures", 5),
    20: ("Data and Alignment", 5),
    21: ("From Pictures to Video", 6), 22: ("Keeping Frames Together", 6), 23: ("Diffusion on Your Mac", 6),
    24: ("Out of the Noise, Revisited", 6),
}
LEVELS = MODULES  # older kits' name

_SMALL = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()


def spoken_title(title):
    """A title as the narrator should read it: no symbols (the lexicon splits CamelCase names)."""
    return title.rstrip("?.").replace(" & ", " and ").replace(":", ",").replace(" vs ", " versus ")


def say_number(n):
    """A number under a hundred as words, for the narrator."""
    return _SMALL[n] if n < 20 else _TENS[n // 10 - 2] + ("" if n % 10 == 0 else "-" + _SMALL[n % 10])


# ---------------------------------------------------------------------------- the voice

NARRATOR = os.environ.get("NARRATOR_URL", "http://127.0.0.1:8765/speak")
VOICE_REF = ""  # a clean 10–20 s WAV of a voice to copy; "" = the built-in voice


def narrator(text, voice_ref=VOICE_REF, exaggeration=0.5, cfg_weight=0.3, temperature=0.8):
    """Chatterbox, run locally by the tutorial-video skill's tts_server.py, timed by Whisper.
    Start it from the series folder: ~/.claude/skills/tutorial-video/scripts/narrator.sh"""
    settings = dict(voice_ref=voice_ref, exaggeration=exaggeration, cfg_weight=cfg_weight, temperature=temperature)
    request = urllib.request.Request(
        NARRATOR, json.dumps({"text": text, **settings}).encode(), {"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=1800) as reply:
            said = json.load(reply)
    except urllib.error.URLError as error:
        raise RuntimeError(
            f"no narrator at {NARRATOR}: start ~/.claude/skills/tutorial-video/scripts/narrator.sh"
            " in the series folder (or render with VOICE=draft)"
        ) from error
    return m.Speech(base64.b64decode(said["audio"]), text=text, words=[Word(*w) for w in said["words"]])


def draft_voice(text, seconds_per_word=0.3):
    """A silent stand-in at about the real voice's pace: for layout and timing passes."""
    words = re.findall(r"\S+", text)
    rate, lead = 24000, 0.15
    timed = [Word(w, lead + i * seconds_per_word, lead + (i + 1) * seconds_per_word) for i, w in enumerate(words)]
    silence = np.zeros(int((2 * lead + seconds_per_word * len(words)) * rate), dtype=np.float32)
    return m.Speech(silence, text=text, words=timed, rate=rate)



# ---------------------------------------------------------------------------- basic components


def text(s, size=32, color=INK, weight=m.NORMAL, **kw):
    """Body text (Avenir Next)."""
    return m.Text(s, font=SANS, font_size=size, color=color, weight=weight, **kw)


def title_text(s, size=40, color=INK, **kw):
    """Display text (Baskerville Bold): titles, big numbers, recap words."""
    return m.Text(s, font=HEAD, font_size=size, color=color, weight=m.BOLD, **kw)


def code(s, size=22, color=INK, t2c=None, **kw):
    """Monospace text (Fira Mono). `t2c={"regex": colour}` colours matching parts (ManimGX's own t2c
    takes literal strings only, so the regexes are matched here; a key that matches nothing as a regex is
    tried as a literal). The first rule to claim a character wins."""
    words = m.Text(s, font=MONO, font_size=size, color=color, **kw)
    if t2c:
        paint(words, s, t2c)
    return words


def paint(words, s, rules):
    """Colour the glyphs of Text `words` (made from string `s`) whose characters match `rules`
    ({regex: colour}, in priority order). Whitespace draws no glyph, so indices skip it."""
    glyph_of, n = {}, 0
    for i, ch in enumerate(s):
        if not ch.isspace():
            glyph_of[i] = n
            n += 1
    if n != len(words):  # a ligature or emoji changed the glyph count: leave the text plain
        return words
    claimed = set()
    for pattern, colour in rules.items():
        try:
            spans = [mt.span() for mt in re.finditer(pattern, s, re.MULTILINE)]
        except re.error:
            spans = []
        if not spans:
            spans = [mt.span() for mt in re.finditer(re.escape(pattern), s)]
        for a, b in spans:
            for i in range(a, b):
                if i in glyph_of and i not in claimed:
                    claimed.add(i)
                    words[glyph_of[i]].set_color(colour)
    return words


def tex(s, size=40, color=INK, **kw):
    """A formula (LaTeX math)."""
    return m.MathTex(s, font_size=size, color=color, **kw)


def glyphs(words, key):
    """The glyphs that draw the first `key` in a Text (spaces and newlines draw none). Mono text only:
    ligatures (fi, fl) in the sans font merge two letters into one glyph."""
    start = words.text.index(key)
    skip = sum(c.isspace() for c in words.text[:start])
    width = sum(not c.isspace() for c in key)
    return words[start - skip : start - skip + width]


def icon(name, height):
    return m.SVGMobject(str(ASSETS / f"{name}.svg"), height=height)


def card(width, height, color=EDGE, fill=CARD, radius=0.2, stroke=2.5):
    return m.RoundedRectangle(
        corner_radius=radius, width=width, height=height,
        stroke_color=color, stroke_width=stroke, fill_color=fill, fill_opacity=1,
    )


def pill(label, color, fill, size=22, mono=False):
    words = code(label, size - 2, color, weight=m.BOLD) if mono else text(label, size, color, weight=m.BOLD)
    height = words.height + 0.3
    box = card(words.width + 0.55, height, color, fill, radius=height / 2, stroke=2)
    return m.VGroup(box, words.move_to(box))


def chip(word, color=TEAL, fill=TEAL_FILL, width=1.5, height=0.72, size=28, mono=False):
    """A word in a small rounded box, on the same baseline as its neighbours'."""
    box = card(width, height, color, fill, radius=0.14, stroke=2.5)
    label = (code(word, size - 4, INK) if mono else text(word, size, INK, weight=m.BOLD)).move_to(box)
    chars = word.replace(" ", "")
    if len(label) == len(chars):
        sitting = [g for g, c in zip(label, chars) if c not in "gjpqy,;_()[]{}"] or list(label)
        bottom = min(g.get_bottom()[1] for g in sitting)
    else:  # a ligature (fi, fl) merged two letters into one glyph: most glyphs sit on the baseline
        bottom = float(np.median([g.get_bottom()[1] for g in label]))
    baseline = box.get_center()[1] - 0.449 * size / 48 / 2
    label.shift((baseline - bottom) * m.UP)
    return m.VGroup(box, label)


def header(tag, title, color, fill):
    """The section header: an accent bar, a mono tag, a Baskerville title (top left)."""
    tag_words = code(tag.upper(), 20, color, weight=m.BOLD)
    title_words = title_text(title, 36)
    bar = m.RoundedRectangle(corner_radius=0.04, width=0.1, height=0.62, stroke_width=0, fill_color=color, fill_opacity=1)
    row = m.VGroup(bar, tag_words, title_words).arrange(m.RIGHT, buff=0.3)
    tag_words.align_to(title_words, m.DOWN)
    return row.to_corner(m.UL, buff=0.5)


def bubble(words, color, fill, tail="left"):
    box = card(words.width + 0.7, words.height + 0.55, color, fill, radius=0.3)
    words.move_to(box)
    side = -1 if tail == "left" else 1
    base = box.get_bottom() + np.array([side * (box.width / 2 - 0.6), 0.15, 0])
    nub = m.Polygon(
        base + [-0.2, 0, 0], base + [0.2, 0, 0], base + [side * 0.5, -0.55, 0],
        stroke_color=color, stroke_width=2.5, fill_color=fill, fill_opacity=1,
    )
    return m.VGroup(nub, box, words)


def node(name, label, sub, color, fill, width=2.1, height=2.1, label_size=26):
    box = card(width, height, color, fill)
    body = m.VGroup(icon(name, 0.8), text(label, label_size, INK, weight=m.BOLD), text(sub, 20, MUTED))
    body.arrange(m.DOWN, buff=0.13).move_to(box)
    return m.VGroup(box, body)


def tile(name, title, desc, color, width=6.3, height=2.1):
    box = card(width, height, color, CARD, stroke=2)
    words = m.VGroup(text(title, 30, INK, weight=m.BOLD), text(desc, 22, MUTED))
    words.arrange(m.DOWN, buff=0.18, aligned_edge=m.LEFT)
    body = m.VGroup(icon(name, 1.05), words).arrange(m.RIGHT, buff=0.45)
    body.move_to(box).align_to(box, m.LEFT).shift(0.45 * m.RIGHT)
    return m.VGroup(box, body)


def arrow(start, end, color=MUTED, buff=0.1):
    return m.Arrow(start, end, buff=buff, color=color, stroke_width=5, max_tip_length_to_length_ratio=0.3)


def pulse(a, color, width=10):
    return m.ShowPassingFlash(m.Line(a.get_start(), a.get_end()).set_stroke(color, width=width), time_width=0.7)


def on_baseline(words):
    for w in words[1:]:
        w.shift((words[0][0].get_bottom()[1] - w[0].get_bottom()[1]) * m.UP)
    return words


def glow(color, radius=3.0, layers=28):
    return m.VGroup(
        *[m.Circle(radius=radius * (1 - i / layers), stroke_width=0).set_fill(color, opacity=0.009) for i in range(layers)]
    )


def backdrop():
    image = m.ImageMobject(str(ASSETS / "backdrop.png"))
    image.height = 8
    return image


def axes(x_range, y_range, x_length, y_length, color=FAINT):
    """Quiet axes for the dark stage: thin, no numbers (label them with text()). Note: a range is cut to
    whole steps, so pick a step that divides it (gotcha 27)."""
    return m.Axes(
        x_range=x_range, y_range=y_range, x_length=x_length, y_length=y_length,
        axis_config={"color": color, "stroke_width": 2.5, "include_tip": True, "tip_length": 0.18},
    )


def recolored(name, color, fill, prefix):
    """An icon in another accent, written as assets/<prefix>-<name>-<hex>.svg (pass your episode's prefix,
    e.g. "ep07", so parallel episodes never write the same file). Returns the new name."""

    def swap(match):
        hexcode = match.group(0)
        if hexcode.upper() in (BG.upper(), CARD.upper()):
            return hexcode
        r, g, b = (int(hexcode[i : i + 2], 16) / 255 for i in (1, 3, 5))
        return color if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.3 else fill

    variant = f"{prefix}-{name}-{color.lstrip('#').lower()}"
    out = re.sub(r"#[0-9A-Fa-f]{6}", swap, (ASSETS / f"{name}.svg").read_text())
    path = ASSETS / f"{variant}.svg"
    if not path.exists() or path.read_text() != out:
        path.write_text(out)
    return variant


# ---------------------------------------------------------------------------- code and data cards

_PY_KEYWORDS = r"\b(def|while|for|in|if|elif|else|return|import|from|as|with|try|except|raise|class|break|continue|not|and|or|is|None|True|False|lambda|yield|await|async)\b"


def code_block(src, size=20, width=None, title=None, accent=EDGE):
    """Python (or pseudo-code) in a dark card, lightly highlighted: keywords VIOLET, strings GREEN, comments
    FAINT, numbers AMBER, called names TEAL. Keep lines short (about 44 characters at size 20 fill 7 units)."""
    t2c = {
        r"#[^\n]*": FAINT,
        r"\"[^\"\n]*\"|'[^'\n]*'": GREEN,
        _PY_KEYWORDS: VIOLET,
        r"\b\d+(\.\d+)?(e\d+)?\b": AMBER,
        r"\b[a-z_][a-z0-9_]*(?=\()": TEAL,
    }
    body = code(src, size, INK, t2c=t2c)
    w = max(width or 0, body.width + 0.7)
    top, parts = 0.0, []
    if title:
        cap = code(title, 20, MUTED)
        top = cap.height + 0.25
        parts.append(cap)
    box = card(w, body.height + 0.6 + top, accent, CARD, radius=0.16, stroke=2)
    body.move_to(box).align_to(box, m.LEFT).shift(0.35 * m.RIGHT)
    if title:
        body.shift(-top / 2 * m.UP)
        parts[0].align_to(box, m.UL).shift(0.35 * m.RIGHT + 0.18 * m.DOWN)
    return m.VGroup(box, body, *parts)


def terminal(lines, width=6.5, size=20, title="terminal"):
    """A terminal window. Lines starting with "$ " are commands (the prompt in GREEN); lines starting
    with "! " are errors (RED); others are output (MUTED)."""
    rows = []
    for line in lines:
        if line.startswith("$ "):
            rows.append(code(line, size, INK, t2c={r"^\$": GREEN}))
        elif line.startswith("! "):
            rows.append(code(line[2:], size, RED))
        else:
            rows.append(code(line or " ", size, MUTED))
    body = m.VGroup(*rows).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.14)
    bar_h = 0.42
    w = max(width, body.width + 0.7)
    box = card(w, body.height + 0.55 + bar_h, EDGE, "#0B0C0F", radius=0.16, stroke=2)
    bar = m.Line(box.get_corner(m.UL) + [0.0, -bar_h, 0], box.get_corner(m.UR) + [0.0, -bar_h, 0]).set_stroke(EDGE, 2)
    dots = m.VGroup(*[m.Dot(radius=0.06, color=c) for c in (RED, AMBER, GREEN)]).arrange(m.RIGHT, buff=0.12)
    dots.move_to(box.get_corner(m.UL) + [0.4, -bar_h / 2, 0])
    cap = code(title, 20, FAINT).move_to(box.get_top() + [0, -bar_h / 2, 0])
    body.next_to(bar, m.DOWN, buff=0.28).align_to(box, m.LEFT).shift(0.35 * m.RIGHT)
    return m.VGroup(box, bar, dots, cap, body)


def json_card(obj, size=19, accent=TEAL, fill=CARD, width=None, title=None):
    """A JSON object as a card: keys TEAL, strings GREEN, numbers AMBER, true/false/null VIOLET."""
    src = json.dumps(obj, indent=2, ensure_ascii=False)
    t2c = {
        r"\"[^\"\n]*\"(?=\s*:)": TEAL,
        r":\s*\"[^\"\n]*\"": GREEN,
        r"\b-?\d+(\.\d+)?\b": AMBER,
        r"\b(true|false|null)\b": VIOLET,
    }
    body = code(src, size, INK, t2c=t2c)
    top, cap = 0.0, None
    if title:
        cap = code(title, 20, accent, weight=m.BOLD)
        top = cap.height + 0.22
    w = max(width or 0, body.width + 0.7, (cap.width + 0.7) if cap else 0)
    box = card(w, body.height + 0.55 + top, accent, fill, radius=0.16, stroke=2)
    body.move_to(box).align_to(box, m.LEFT).shift(0.35 * m.RIGHT + top / 2 * m.DOWN)
    group = m.VGroup(box, body)
    if cap:
        cap.align_to(box, m.UL).shift(0.35 * m.RIGHT + 0.17 * m.DOWN)
        group.add(cap)
    return group


def message(role, body, width=5.2, size=20, mono=False):
    """One chat message: a role label in its colour (system MUTED, user PINK, assistant TEAL)."""
    color, fill = ROLES[role]
    label = code(role, 22, color, weight=m.BOLD)
    words = code(body, size - 2, INK) if mono else text(body, size, INK)
    inner = m.VGroup(label, words).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.12)
    if words.width > width - 0.6:
        words.scale((width - 0.6) / words.width)
        inner.arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.12)
    box = card(width, inner.height + 0.36, EDGE, CARD, radius=0.12, stroke=1.5)
    stripe = m.RoundedRectangle(corner_radius=0.05, width=0.09, height=box.height - 0.16, stroke_width=0,
                                fill_color=color, fill_opacity=1)
    stripe.move_to(box).align_to(box, m.LEFT).shift(0.1 * m.RIGHT)
    inner.move_to(box).align_to(box, m.LEFT).shift(0.32 * m.RIGHT)
    return m.VGroup(box, stripe, inner)


def loop_ring(labels, radius=1.6, colors=None, size=24, center=(0, 0, 0)):
    """A cycle of steps around a circle with arrows between them. Returns VGroup(nodes, arcs)."""
    n = len(labels)
    colors = colors or [VIOLET, AMBER, TEAL, PINK, BLUE][:n]
    center = np.array(center, dtype=float)
    angles = [np.pi / 2 - 2 * np.pi * i / n for i in range(n)]
    nodes = m.VGroup()
    for label, color, a in zip(labels, colors, angles):
        words = text(label, size, INK, weight=m.BOLD)
        c = chip(label, color, FILLS.get(color, CARD), width=max(1.4, words.width + 0.5), height=0.66, size=size)
        nodes.add(c.move_to(center + radius * np.array([np.cos(a), np.sin(a), 0])))

    def clearance(chip_, a):
        half_w, half_h = chip_.width / 2 + 0.14, chip_.height / 2 + 0.14
        reach = half_w * abs(np.sin(a)) + half_h * abs(np.cos(a))
        return min(reach / radius, 0.45 * 2 * np.pi / n)

    arcs = m.VGroup()
    for i in range(n):
        a0, a1 = angles[i], angles[(i + 1) % n]
        if a1 > a0:
            a1 -= 2 * np.pi
        start, end = a0 - clearance(nodes[i], a0), a1 + clearance(nodes[(i + 1) % n], a1)
        arc = m.Arc(radius=radius, start_angle=start, angle=end - start, arc_center=center)
        arcs.add(arc.set_stroke(MUTED, 4).add_tip(tip_length=0.2))
    return m.VGroup(nodes, arcs)

# ---------------------------------------------------------------------------- small data components


def token_row(tokens, color=PROMPT, size=20, gap=0.08, max_width=12.0, fill=None):
    """Tokens as small mono chips in a row (wrapping at max_width). Returns a VGroup of chips."""
    ref = code("Mx", size)  # every chip has the same height and baseline, whatever its letters
    chips = []
    for tok in tokens:
        shown = tok if tok.strip() else "␣"
        words = code(shown, size, INK if tok.strip() else MUTED)
        box = card(max(words.width + 0.26, 0.44), ref.height + 0.3, color, fill or FILLS.get(color, CARD),
                   radius=0.08, stroke=1.8)
        words.move_to(box)
        sitting = [g for g, ch in zip(words, shown.replace(" ", "")) if ch not in "gjpqy,;_()[]{}"] or list(words)
        if len(words) == len(shown.replace(" ", "")):
            words.shift((box.get_y() - ref.height / 2 - min(g.get_bottom()[1] for g in sitting)) * m.UP)
        chips.append(m.VGroup(box, words))
    rows, row, width = [], [], 0.0
    for c in chips:
        if row and width + c.width > max_width:
            rows.append(m.VGroup(*row).arrange(m.RIGHT, buff=gap))
            row, width = [], 0.0
        row.append(c)
        width += c.width + gap
    if row:
        rows.append(m.VGroup(*row).arrange(m.RIGHT, buff=gap))
    m.VGroup(*rows).arrange(m.DOWN, aligned_edge=m.LEFT, buff=gap)
    return m.VGroup(*chips)


def stat(value, unit="", label="", color=INK, size=72):
    """A big number (DIN) with a small unit and a MUTED label under it."""
    number = title_text(value, size, color)
    parts = [number]
    if unit:
        u = code(unit, max(20, size // 3), MUTED)
        u.next_to(number, m.RIGHT, buff=0.15).align_to(number, m.DOWN)
        parts.append(u)
    group = m.VGroup(*parts)
    if label:
        group.add(text(label, 22, MUTED).next_to(group, m.DOWN, buff=0.18).align_to(number, m.LEFT))
    return group


def cell_grid(rows, cols, size=0.34, gap=0.06, color=EDGE, fill=CARD):
    """A grid of small squares (KV blocks, batch slots × steps …). grid[r][c] is one cell."""
    grid = m.VGroup(*[
        m.VGroup(*[m.Square(size).set_stroke(color, 1.5).set_fill(fill, 1) for _ in range(cols)]).arrange(m.RIGHT, buff=gap)
        for _ in range(rows)
    ]).arrange(m.DOWN, buff=gap)
    return grid


class Meter(m.VGroup):
    """A horizontal gauge (memory used, a budget …). `fill_to(f)` animates it; it turns RED above 90%."""

    def __init__(self, width=4.0, fraction=0.0, color=TEAL, height=0.24):
        super().__init__()
        self.track_width, self.bar_height, self.color = width, height, color
        self.track = m.RoundedRectangle(corner_radius=height / 2, width=width, height=height,
                                        stroke_color=EDGE, stroke_width=2, fill_color=CARD, fill_opacity=1)
        self.bar = self._bar(fraction)
        self.add(self.track, self.bar)

    def _bar(self, fraction):
        f = float(np.clip(fraction, 0.004, 1.0))
        bar = m.RoundedRectangle(corner_radius=self.bar_height / 2, width=self.track_width * f, height=self.bar_height,
                                 stroke_width=0, fill_color=RED if fraction > 0.9 else self.color, fill_opacity=1)
        return bar.align_to(self.track, m.LEFT).set_y(self.track.get_y())

    def fill_to(self, fraction, **kwargs):
        return m.Transform(self.bar, self._bar(fraction).align_to(self.track, m.LEFT), **kwargs)





# ---------------------------------------------------------------------------- diffusion components


def pixel(size=1.0, seed=7):
    """Pixel, the course's mascot: a rounded square face whose left half is noise (VIOLET speckles) and whose
    right half is clean picture (TEAL), with two eyes. Parts: [0] face, [1] noise cells, [2] eyes."""
    s = size
    face = m.RoundedRectangle(corner_radius=0.22 * s, width=1.2 * s, height=1.2 * s,
                              stroke_color=TEAL, stroke_width=4 * s, fill_color=TEAL_FILL, fill_opacity=1)
    rng = np.random.default_rng(seed)
    cell = 0.16 * s
    cells = m.VGroup()
    for i in range(3):
        for j in range(6):
            shade = mix(VIOLET_FILL, VIOLET, float(rng.uniform(0.15, 0.95)))
            sq = m.Square(cell * 0.92).set_stroke(width=0).set_fill(shade, 1)
            cells.add(sq.move_to(face.get_corner(m.UL) + np.array([(i + 0.9) * cell, -(j + 0.9) * cell, 0])))
    eye = lambda: m.VGroup(m.Circle(radius=0.13 * s).set_stroke(INK, 3 * s).set_fill(BG, 1), m.Dot(radius=0.05 * s, color=INK))
    eyes = m.VGroup(eye(), eye()).arrange(m.RIGHT, buff=0.12 * s).move_to(face).shift(0.18 * s * m.RIGHT + 0.1 * s * m.UP)
    return m.VGroup(face, cells, eyes)


def mix(a, b, t):
    """The hex colour a fraction t of the way from hex colour a to b."""
    ca, cb = (np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], float) for h in (a, b))
    r, g, bl = (ca + (cb - ca) * float(np.clip(t, 0, 1))).round().astype(int)
    return f"#{r:02X}{g:02X}{bl:02X}"


def heat(value, lo=-1.0, hi=1.0, strength=0.75, high=TEAL, low=RED):
    """A fill colour for a value: toward `high` above zero, toward `low` below zero, CARD near zero."""
    if value >= 0:
        return mix(CARD, high, strength * (0.0 if hi <= 0 else value / hi))
    return mix(CARD, low, strength * (0.0 if lo >= 0 else value / lo))


def to_rgb8(img, lo=None, hi=None):
    """An image array → H×W×3 uint8. Accepts H×W (grey) or H×W×3/4 arrays, uint8 or float. Floats are mapped from
    [lo, hi]: by default [-1, 1] if the array has negative values (the model's range), else [0, 1]."""
    a = np.asarray(img)
    if a.dtype != np.uint8:
        a = a.astype(np.float64)
        if lo is None:
            lo, hi = (-1.0, 1.0) if a.min() < 0 else (0.0, 1.0)
        a = np.clip((a - lo) / (hi - lo), 0, 1) * 255
        a = a.round().astype(np.uint8)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    if a.shape[-1] == 4:
        a = a[..., :3]
    return a


def tint(img, color=INK, lo=None, hi=None):
    """A grey image → RGB in one colour (black → BG, white → `color`), e.g. digits in TEAL, noise in VIOLET."""
    g = to_rgb8(img, lo, hi)[..., 0].astype(np.float64) / 255
    c = np.array([int(color[i:i + 2], 16) for i in (1, 3, 5)], float)
    b = np.array([int(BG[i:i + 2], 16) for i in (1, 3, 5)], float)
    return (b + (c - b) * g[..., None]).round().astype(np.uint8)


def picture(img, height=2.0, pixelated=None, lo=None, hi=None):
    """An image array (or a PNG path) as an ImageMobject `height` units tall. Small images (≤ 64 px) are shown with
    crisp square pixels (nearest neighbour) unless pixelated=False. Grey arrays show grey; use tint() for colour."""
    if isinstance(img, (str, Path)):
        from PIL import Image
        arr = np.asarray(Image.open(img).convert("RGB"))
    else:
        arr = to_rgb8(img, lo, hi)
    if pixelated is None:
        pixelated = max(arr.shape[:2]) <= 64
    if pixelated:
        k = max(1, int(np.ceil(540 / max(arr.shape[:2]))))
        arr = np.repeat(np.repeat(arr, k, axis=0), k, axis=1)
    rgba = np.concatenate([arr, np.full(arr.shape[:2] + (1,), 255, np.uint8)], axis=-1)
    image = m.ImageMobject(rgba)
    image.height = height
    return image


def framed(image, color=EDGE, buff=0.06, stroke=2.5, label=None, size=20, label_color=MUTED):
    """A picture in a thin frame, with an optional caption under it. Returns Group(frame, image[, label])."""
    frame = m.Rectangle(width=image.width + 2 * buff, height=image.height + 2 * buff).set_stroke(color, stroke)
    frame.set_fill(opacity=0).move_to(image)
    parts = [frame, image]
    if label is not None:
        parts.append(code(label, size, label_color).next_to(frame, m.DOWN, buff=0.14))
    return m.Group(*parts)


def noise_image(h=28, w=28, seed=0, channels=1):
    """Standard Gaussian noise of the given shape (the model's range: mostly within ±3)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((h, w) if channels == 1 else (h, w, channels))


def strip(images, height=1.4, buff=0.18, labels=None, size=20, color=EDGE):
    """A row of framed pictures (e.g. a trajectory x_T … x_0) with optional labels under each."""
    labels = labels or [None] * len(images)
    row = m.Group(*[framed(picture(im, height) if not isinstance(im, m.Mobject) else im, color, label=lab, size=size)
                    for im, lab in zip(images, labels)])
    return row.arrange(m.RIGHT, buff=buff)


class Bars(m.VGroup):
    """Vertical bars on a baseline (probabilities, counts, scores). Data range [bottom, top] maps to the height;
    negative values hang below the zero line. `.set(values)` animates heights and numbers; `.level(v)` is the scene
    y of data value v; `.marker(i, v, color)` a dashed tick across bar i at v; `.bars[i]`, `.labels[i]`, `.numbers[i]`."""

    def __init__(self, values, labels=None, top=1.0, bottom=0.0, width=6.0, height=3.0, color=MODEL, colors=None,
                 gap=0.3, show_values=True, fmt="{:.2f}", size=20, center=(0, 0, 0)):
        super().__init__()
        n = len(values)
        self.top_v, self.bottom_v, self.h, self.fmt, self.size = top, bottom, height, fmt, size
        self.colors = list(colors) if colors else [color] * n
        self.bar_w = (width - gap * (n - 1)) / n
        c = np.array(center, dtype=float)
        self.y0 = c[1] - height / 2
        self.xs = [c[0] - width / 2 + self.bar_w / 2 + i * (self.bar_w + gap) for i in range(n)]
        self.zero = m.Line([c[0] - width / 2 - 0.15, self.level(0), 0], [c[0] + width / 2 + 0.15, self.level(0), 0]).set_stroke(FAINT, 2)
        self.bars = m.VGroup(*[self._rect(i, v) for i, v in enumerate(values)])
        self.labels = m.VGroup(*[code(str(lab), size, MUTED).move_to([x, self.y0 - 0.32, 0])
                                 for x, lab in zip(self.xs, labels or [""] * n)])
        self.numbers = m.VGroup(*[self._number(i, v) for i, v in enumerate(values)]) if show_values else m.VGroup()
        self.add(self.zero, self.bars, self.labels, self.numbers)

    def level(self, v):
        return self.y0 + self.h * (v - self.bottom_v) / (self.top_v - self.bottom_v)

    def _rect(self, i, v):
        y_zero, y_v = self.level(0), self.level(float(np.clip(v, self.bottom_v, self.top_v)))
        hgt = max(abs(y_v - y_zero), 0.015)
        col = self.colors[i]
        rect = m.Rectangle(width=self.bar_w, height=hgt).set_stroke(col, 2).set_fill(FILLS.get(col, CARD), 1)
        return rect.move_to([self.xs[i], (y_zero + y_v) / 2 if abs(y_v - y_zero) > 0.015 else y_zero, 0])

    def _number(self, i, v):
        words = code(self.fmt.format(v).replace("-", "−"), self.size, self.colors[i])
        y = max(self.level(max(v, 0)), self.level(0)) + 0.24
        return words.move_to([self.xs[i], y, 0])

    def set(self, values, run_time=0.8):
        anims = [m.Transform(self.bars[i], self._rect(i, v)) for i, v in enumerate(values)]
        if len(self.numbers):
            anims += [m.Transform(self.numbers[i], self._number(i, v)) for i, v in enumerate(values)]
        return m.AnimationGroup(*anims, run_time=run_time)

    def marker(self, i, v, color=INK):
        y = self.level(v)
        return m.DashedLine([self.xs[i] - self.bar_w / 2 - 0.08, y, 0], [self.xs[i] + self.bar_w / 2 + 0.08, y, 0],
                            dash_length=0.08).set_stroke(color, 3)


class Plot(m.VGroup):
    """A quiet chart for curves and 2D point clouds. Axes with a few labelled ticks; `.coords(x, y)` maps data to the
    scene; `.line(xs, ys, color)` a curve (draw it with Create); `.dot(x, y)`; `.hline(y)` / `.vline(x)` dashed guides;
    `.band(xs, lo, hi, color)` a shaded spread; `.area(xs, ys, color)` the region under a curve; `.cloud(xy, color)` a
    Cloud of points in data coordinates; `.field(fn, …)` an arrow field. `log_x=True` / `log_y=True` for log axes;
    `axes=False` draws no axes (a bare plane for point clouds). The chart, labels included, is centred on `center`."""

    def __init__(self, x_range, y_range, width=7.0, height=4.0, center=(0, 0, 0), x_ticks=None, y_ticks=None,
                 x_label="", y_label="", log_x=False, log_y=False, x_fmt="{:g}", y_fmt="{:g}", size=20, axes=True):
        super().__init__()
        (self.x0, self.x1), (self.y0, self.y1) = x_range, y_range
        self.w, self.h, self.log_x, self.log_y = width, height, log_x, log_y
        self.origin = np.zeros(3)
        x_axis = m.Arrow(self.origin, self.origin + [width + 0.45, 0, 0], buff=0, color=FAINT, stroke_width=3,
                         max_tip_length_to_length_ratio=0.035)
        y_axis = m.Arrow(self.origin, self.origin + [0, height + 0.4, 0], buff=0, color=FAINT, stroke_width=3,
                         max_tip_length_to_length_ratio=0.06)
        self.axes = m.VGroup(x_axis, y_axis) if axes else m.VGroup()
        self.ticks = m.VGroup()
        for x in x_ticks or []:
            p = self.coords(x, self.y0)
            self.ticks.add(m.Line(p, p + [0, -0.09, 0]).set_stroke(FAINT, 2),
                           code(x_fmt.format(x), size, MUTED).next_to(p, m.DOWN, buff=0.16))
        for y in y_ticks or []:
            p = self.coords(self.x0, y)
            self.ticks.add(m.Line(p, p + [-0.09, 0, 0]).set_stroke(FAINT, 2),
                           code(y_fmt.format(y).replace("-", "−"), size, MUTED).next_to(p, m.LEFT, buff=0.16))
        frame = m.Rectangle(width=width, height=height).set_stroke(width=0).set_fill(opacity=0)
        frame.move_to([width / 2, height / 2, 0])
        self.add(frame, self.axes, self.ticks)
        if x_label:
            self.x_label = text(x_label, size + 2, MUTED)
            self.x_label.next_to(self.ticks if len(self.ticks) else frame, m.DOWN, buff=0.16).set_x(width / 2)
            self.add(self.x_label)
        if y_label:
            self.y_label = text(y_label, size + 2, MUTED).next_to(frame.get_corner(m.UL) + [0, 0.4, 0], m.UP, buff=0.12)
            self.y_label.align_to(frame, m.LEFT)
            self.add(self.y_label)
        offset = np.array(center, dtype=float) - self.get_center()
        self.shift(offset)
        self.origin = self.origin + offset

    def coords(self, x, y):
        fx = ((np.log10(x) - np.log10(self.x0)) / (np.log10(self.x1) - np.log10(self.x0)) if self.log_x
              else (x - self.x0) / (self.x1 - self.x0))
        fy = ((np.log10(y) - np.log10(self.y0)) / (np.log10(self.y1) - np.log10(self.y0)) if self.log_y
              else (y - self.y0) / (self.y1 - self.y0))
        return self.origin + np.array([fx * self.w, fy * self.h, 0])

    def _clip(self, y):
        return float(np.clip(y, self.y0, self.y1))

    def line(self, xs, ys, color=MODEL, width=4):
        pts = [self.coords(x, self._clip(y)) for x, y in zip(xs, ys)]
        return m.VMobject().set_points_as_corners(pts).set_stroke(color, width).set_fill(opacity=0)

    def dot(self, x, y, color=INK, radius=0.08):
        return m.Dot(self.coords(x, y), radius=radius, color=color).set_z_index(3)

    def hline(self, y, color=MUTED, width=2):
        return m.DashedLine(self.coords(self.x0, y), self.coords(self.x1, y), dash_length=0.1).set_stroke(color, width)

    def vline(self, x, color=MUTED, width=2):
        return m.DashedLine(self.coords(x, self.y0), self.coords(x, self.y1), dash_length=0.1).set_stroke(color, width)

    def band(self, xs, lo, hi, color=MODEL, opacity=0.22):
        pts = [self.coords(x, self._clip(y)) for x, y in zip(xs, hi)]
        pts += [self.coords(x, self._clip(y)) for x, y in zip(xs[::-1], lo[::-1])]
        return m.Polygon(*pts, stroke_width=0, fill_color=color, fill_opacity=opacity)

    def area(self, xs, ys, color=MODEL, opacity=0.25):
        return self.band(xs, [self.y0] * len(xs), ys, color, opacity)

    def cloud(self, xy, color=DATA, radius=0.035, opacity=0.9):
        return Cloud(self, xy, color, radius, opacity)

    def field(self, fn, nx=13, ny=9, color=FLOW, max_len=0.42, width=3, margin=0.06):
        """Arrows of fn(x, y) → (u, v) on an nx × ny grid inside the plot, all scaled by one factor so the longest
        is max_len scene units (directions and relative sizes are true). Returns a VGroup of arrows."""
        xs = np.linspace(self.x0 + margin * (self.x1 - self.x0), self.x1 - margin * (self.x1 - self.x0), nx)
        ys = np.linspace(self.y0 + margin * (self.y1 - self.y0), self.y1 - margin * (self.y1 - self.y0), ny)
        pts = np.array([(x, y) for y in ys for x in xs])
        vec = np.array([fn(x, y) for x, y in pts], dtype=float)
        sx, sy = self.w / (self.x1 - self.x0), self.h / (self.y1 - self.y0)
        scene_vec = vec * np.array([sx, sy])
        longest = np.max(np.linalg.norm(scene_vec, axis=1)) or 1.0
        arrows = m.VGroup()
        for (x, y), v in zip(pts, scene_vec * (max_len / longest)):
            p = self.coords(x, y)
            if np.linalg.norm(v) < 0.04:
                arrows.add(m.Dot(p, radius=0.02, color=color))
                continue
            arrows.add(m.Arrow(p, p + np.array([v[0], v[1], 0]), buff=0, color=color, stroke_width=width,
                               max_tip_length_to_length_ratio=0.35))
        return arrows


class Cloud(m.VGroup):
    """2D points drawn as small dots in a Plot's data coordinates. `.positions(xy)` moves them in place; `.to(xy)` is
    the animation to new positions; `.replay(path, seconds)` walks through a trajectory array (steps × n × 2), e.g. a
    sampler's output. The number of points is fixed at creation."""

    def __init__(self, plot, xy, color=DATA, radius=0.035, opacity=0.9):
        super().__init__()
        self.plot = plot
        xy = np.asarray(xy, dtype=float)
        for x, y in xy:
            self.add(m.Dot(plot.coords(x, y), radius=radius, color=color).set_opacity(opacity))

    def _put_xy(self, xy):
        for d, (x, y) in zip(self, np.asarray(xy, dtype=float)):
            d.move_to(self.plot.coords(x, y))
        return self

    def positions(self, xy):
        return self._put_xy(xy)

    def to(self, xy, run_time=1.2):
        start = np.array([[*self._data_of(d)] for d in self])
        end = np.asarray(xy, dtype=float)

        def step(_, a):
            self._put_xy(start + (end - start) * a)

        return m.UpdateFromAlphaFunc(self, step, run_time=run_time)

    def _data_of(self, dot):
        p = dot.get_center() - self.plot.origin
        fx, fy = p[0] / self.plot.w, p[1] / self.plot.h
        return (self.plot.x0 + fx * (self.plot.x1 - self.plot.x0), self.plot.y0 + fy * (self.plot.y1 - self.plot.y0))

    def replay(self, path, seconds=3.0):
        path = np.asarray(path, dtype=float)

        def step(_, a):
            i = a * (len(path) - 1)
            j = int(np.floor(i))
            k = min(j + 1, len(path) - 1)
            self._put_xy(path[j] + (path[k] - path[j]) * (i - j))

        return m.UpdateFromAlphaFunc(self, step, run_time=seconds, rate_func=m.linear)


class NeuralNet(m.VGroup):
    """A small network diagram: columns of nodes joined by thin lines. `layers` = nodes per column (more than
    `max_nodes` are capped). `.columns[i]`, `.edges[i]` (between i and i+1); `.forward(color)` a flash left to right."""

    def __init__(self, layers=(4, 8, 8, 2), width=4.5, height=3.0, center=(0, 0, 0), color=MODEL, radius=0.11, max_nodes=8):
        super().__init__()
        c = np.array(center, dtype=float)
        n = len(layers)
        self.columns, self.edges = m.VGroup(), m.VGroup()
        for i, k in enumerate(layers):
            shown = min(k, max_nodes)
            x = c[0] - width / 2 + width * i / max(n - 1, 1)
            ys = np.linspace(c[1] + height / 2, c[1] - height / 2, shown) if shown > 1 else [c[1]]
            col = m.VGroup(*[m.Circle(radius=radius).set_stroke(color, 2.5).set_fill(BG, 1).move_to([x, y, 0]) for y in ys])
            self.columns.add(col)
        for a, b in zip(self.columns[:-1], self.columns[1:]):
            self.edges.add(m.VGroup(*[m.Line(p.get_center(), q.get_center(), buff=radius).set_stroke(EDGE, 1.5)
                                      for p in a for q in b]))
        self.add(self.edges, self.columns)

    def forward(self, color=MODEL, run_time=1.2):
        flashes = [m.ShowPassingFlash(e.copy().set_stroke(color, 3), time_width=0.6) for e in self.edges]
        return m.LaggedStart(*flashes, lag_ratio=0.7, run_time=run_time)


class UNet(m.VGroup):
    """A U-Net diagram: down blocks shrinking left to right-down, a bottleneck, up blocks growing back up, with
    skip arrows across. `sizes` = feature-map sides per level (e.g. (28, 14, 7)), `channels` = channels per level
    (labels). `.down[i]`, `.up[i]`, `.mid`, `.skips[i]`; `.flow(color)` a flash down the U and back up."""

    def __init__(self, sizes=(28, 14, 7), channels=(64, 128, 256), center=(0, 0, 0), scale=1.0, color=MODEL,
                 labels=True, size=20):
        super().__init__()
        c = np.array(center, dtype=float)
        n = len(sizes)
        side = lambda k: (0.5 + 1.3 * sizes[k] / sizes[0]) * scale
        dx, dy = 1.25 * scale, 0.95 * scale
        x_left = c[0] - dx * n
        self.down, self.up, self.skips, self.lab = m.VGroup(), m.VGroup(), m.VGroup(), m.VGroup()
        for k in range(n):
            y = c[1] + dy * (n - 1) / 2 - dy * k
            for group, x in ((self.down, x_left + dx * k), (self.up, c[0] + dx * n - dx * k)):
                h = side(k)
                block = m.Rectangle(width=0.32 * scale + 0.06 * k * scale, height=h).set_stroke(color, 2.5)
                block.set_fill(FILLS.get(color, CARD), 1).move_to([x, y, 0])
                group.add(block)
            if labels:
                self.lab.add(code(f"{sizes[k]}×{sizes[k]}·{channels[k]}", size - 2, MUTED).move_to([0, y, 0]))
        y_mid = c[1] + dy * (n - 1) / 2 - dy * n
        self.mid = m.Rectangle(width=0.5 * scale, height=side(n - 1) * 0.7).set_stroke(color, 2.5)
        self.mid.set_fill(FILLS.get(color, CARD), 1).move_to([c[0], y_mid, 0])
        self.path = m.VGroup()
        chain = list(self.down) + [self.mid] + list(self.up)[::-1]
        for a, b in zip(chain[:-1], chain[1:]):
            self.path.add(m.Arrow(a.get_center(), b.get_center(), buff=0.35 * scale, color=MUTED, stroke_width=3,
                                  max_tip_length_to_length_ratio=0.2))
        for k in range(n):
            a, b = self.down[k].get_right() + [0.12, 0, 0], self.up[k].get_left() - [0.12, 0, 0]
            line = m.DashedLine(a, b - [0.16, 0, 0], dash_length=0.1).set_stroke(FAINT, 2.5)
            tip = m.Triangle().set_stroke(width=0).set_fill(FAINT, 1).scale(0.09).rotate(-PI_ / 2).move_to(b - [0.08, 0, 0])
            self.skips.add(m.VGroup(line, tip))
        if labels:
            right = self.down[0].get_left()[0] - 0.25
            for lab in self.lab:
                lab.set_x(right - lab.width / 2)
        self.add(self.skips, self.path, self.down, self.mid, self.up, self.lab)

    def flow(self, color=MODEL, run_time=1.6):
        flashes = [m.ShowPassingFlash(a.copy().set_stroke(color, 6), time_width=0.8) for a in self.path]
        return m.LaggedStart(*flashes, lag_ratio=0.6, run_time=run_time)


def denoise_loop(center=(0, 0, 0), width=9.0, size=1.0, labels=("noisy picture", "denoiser", "less noise")):
    """The course's loop: a noisy picture goes into the denoiser (AMBER), a slightly cleaner picture comes out and
    goes round again. Returns a Group with .left (input card), .model, .right (output card), .go, .out, .back."""
    c = np.array(center, dtype=float)
    s = size
    rng = np.random.default_rng(3)

    def panel(img, color, label):
        words = text(label, 20, MUTED)
        box = card(max(1.9 * s, words.width + 0.4), 2.0 * s, color, CARD)
        image = picture(img, 1.3 * s).move_to(box).shift(0.15 * s * m.UP)
        words.move_to(box.get_bottom() + [0, 0.28 * s, 0])
        return m.Group(box, image, words)

    left = panel(tint(rng.standard_normal((12, 12)), VIOLET, -2.5, 2.5), NOISE, labels[0])
    model = m.VGroup(card(2.2 * s, 1.4 * s, MODEL, AMBER_FILL), text(labels[1], 26, INK, weight=m.BOLD))
    model[1].move_to(model[0])
    clean = 0.55 * rng.standard_normal((12, 12)) + np.where((np.add.outer(np.arange(12), np.arange(12)) % 6) < 3, 1.2, -1.2)
    right = panel(tint(clean, TEAL, -2.5, 2.5), DATA, labels[2])
    left.move_to(c + [-width / 2 + left[0].width / 2, 0.25 * s, 0])
    model.move_to(c + [0, 0.25 * s, 0])
    right.move_to(c + [width / 2 - right[0].width / 2, 0.25 * s, 0])
    go = m.Arrow(left[0].get_right(), model[0].get_left(), buff=0.12, color=NOISE, stroke_width=6)
    out = m.Arrow(model[0].get_right(), right[0].get_left(), buff=0.12, color=DATA, stroke_width=6)
    back = m.CurvedArrow(right[0].get_bottom() + [0, -0.1, 0], left[0].get_bottom() + [0, -0.1, 0], angle=-0.35 * PI_)
    back.set_stroke(MUTED, 3)
    loop = m.Group(left, model, right, go, out, back)
    loop.left, loop.model, loop.right, loop.go, loop.out, loop.back = left, model, right, go, out, back
    return loop


PI_ = float(np.pi)


# ---------------------------------------------------------------------------- the episode


class Episode(m.Scene):
    """An episode of the course: its intro card, its lesson (`lesson()`), the next-episode card."""

    number = 0
    voice = draft_voice if os.environ.get("VOICE") == "draft" else narrator
    say = _say_namespace["say"]

    def construct(self) -> None:
        self.head = None
        self.add(backdrop().set_z_index(-100))
        self.series_intro()
        self.lesson()
        self.series_outro()

    def lesson(self) -> None:
        raise NotImplementedError

    @property
    def module(self):
        return MODULES[EPISODES[self.number][1]]

    level = module

    def new_header(self, tag, title, color=None, fill=None):
        """The animation that swaps the header: pass it to say() with a short [span]."""
        _, accent, tint = self.module
        new = header(tag, title, color or accent, fill or tint)
        parts = [m.FadeIn(new, shift=0.25 * m.UP)]
        if self.head:
            parts.insert(0, m.FadeOut(self.head, shift=0.25 * m.UP))
        self.head = new
        return m.AnimationGroup(*parts, run_time=0.9)

    def clear(self, *keep, run_time=0.8):
        gone = [mob for mob in self.mobjects[1:] if mob not in keep and mob is not self.head]
        if gone:
            self.play(*[m.FadeOut(mob) for mob in gone], run_time=run_time)

    def drop_header(self):
        old, self.head = self.head, None
        return m.FadeOut(old, shift=0.25 * m.UP) if old else m.Wait(0.01)

    def _fit(self, heading):
        if heading.width > TITLE_WIDTH:
            heading.scale(TITLE_WIDTH / heading.width)
        return heading

    def series_intro(self):
        title, module = EPISODES[self.number]
        name, accent, tint = MODULES[module]
        self.next_section(f"Episode {self.number} · {title}", notes="Series card.")
        halo = glow(accent, radius=4.2).move_to(0.1 * m.DOWN)
        prompt = code(f"$ denoise --episode {self.number:02d}", 22, MUTED, t2c={r"^\$": GREEN})
        brand = title_text("Out of the Noise", 46, INK, gradient=(VIOLET, TEAL))
        badge = pill(f"MODULE {module} · {name}", accent, tint, 20, mono=True)
        heading = self._fit(title_text(title, 76 if len(title) < 22 else 60))
        top = m.VGroup(prompt, brand).arrange(m.DOWN, buff=0.22)
        group = m.VGroup(top, badge, heading).arrange(m.DOWN, buff=0.45).move_to(0.25 * m.UP)
        half = min(heading.width, 6) / 2
        rule = m.Line([-half, 0, 0], [half, 0, 0]).set_stroke(accent, width=5).next_to(heading, m.DOWN, buff=0.35)
        mascot = pixel(0.5).next_to(rule, m.DOWN, buff=0.45)

        self.wait(0.3)
        self.say(
            f"[Welcome to Out of the Noise.] [Episode {say_number(self.number)}]: [{spoken_title(title)}.]",
            m.AnimationGroup(m.FadeIn(halo, scale=0.6), m.AddTextLetterByLetter(prompt), m.FadeIn(brand, shift=0.2 * m.DOWN), run_time=1.2),
            m.AnimationGroup(m.FadeIn(badge, shift=0.2 * m.UP), m.FadeIn(mascot, shift=0.2 * m.UP)),
            m.AnimationGroup(m.Write(heading), m.GrowFromCenter(rule)),
        )
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, group, rule, mascot)], run_time=0.8)

    def series_outro(self):
        upcoming = self.number + 1
        self.next_section("Next episode", notes="Next-episode card.")
        if self.head:
            self.play(self.drop_header(), run_time=0.6)
        if upcoming not in EPISODES:
            self.say("Thanks for watching!")
            return
        title, module = EPISODES[upcoming]
        name, accent, tint = MODULES[module]
        halo = glow(accent, radius=4.0)
        kicker = code("next episode", 24, MUTED)
        number = pill(f"EP {upcoming:02d} · {name}", accent, tint, 20, mono=True)
        heading = self._fit(title_text(title, 64 if len(title) < 26 else 50))
        group = m.VGroup(kicker, number, heading).arrange(m.DOWN, buff=0.4).move_to(0.2 * m.UP)
        self.say(
            f"[Next time, in episode {say_number(upcoming)}]: [{spoken_title(title)}.] Thanks for watching!",
            m.AnimationGroup(m.FadeIn(halo, scale=0.6), m.FadeIn(kicker), m.FadeIn(number, shift=0.2 * m.UP)),
            m.Write(heading),
        )
        self.wait(1.0)
        self.play(*[m.FadeOut(mob) for mob in (halo, group)], run_time=1.0)
        self.wait(0.5)
