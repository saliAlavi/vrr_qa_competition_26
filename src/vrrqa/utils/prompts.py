"""Prompt construction and answer parsing for multiple-choice video QA.

Handles variable option counts (2..8 options, keys A..H) and robust letter parsing.
"""
from __future__ import annotations
import re
from typing import Dict, List

SYSTEM_PROMPT = (
    "You are an expert at visual reasoning over video. You watch the video carefully, "
    "reason about spatial relationships, motion, depth, viewpoint, counting, and implicit "
    "social/physical cues that may not be directly visible in any single frame, then choose "
    "the single best answer."
)


DESCRIBE_PROMPT = (
    "You are watching a video clip shown as an ordered sequence of frames. Write a "
    "thorough, objective description of what happens. Be especially precise about:\n"
    "1. the setting and environment;\n"
    "2. every person, animal, and notable object, and where each is located "
    "(left/right, foreground/background, above/below) and their relative distance/depth;\n"
    "3. who or what is facing or able to see whom (line of sight, occlusions, viewpoint);\n"
    "4. every movement and its direction across the frames (who moves toward/away from what);\n"
    "5. counts of relevant repeated items or events;\n"
    "6. actions and interactions between entities, and any implied cause/intent.\n"
    "Do not answer any question. Just describe what can be seen and inferred."
)


def build_decompose_prompt(question: str, options: Dict[str, str]) -> str:
    opts = format_options(options)
    return (
        f"You are given a hard multiple-choice question about a video:\n"
        f"Question: {question}\nOptions:\n{opts}\n\n"
        f"Break this into 2 to 4 SIMPLE, ATOMIC visual sub-questions that must be "
        f"answered first to determine the correct option. Each sub-question should ask "
        f"about exactly one concrete thing (e.g. the position of an entity, a direction "
        f"of motion, a count, who is doing what, who can see whom). "
        f"Output ONLY the sub-questions, one per line, prefixed 'Q1:', 'Q2:', ..."
    )


def build_subanswer_prompt(subquestions: str) -> str:
    return (
        f"Answer each of the following sub-questions about the video concisely and "
        f"factually, based only on what is visible/inferable across the frames:\n"
        f"{subquestions}\n\n"
        f"Reply with one line per sub-question, formatted 'Q1: <answer>'."
    )


def build_aggregate_prompt(question: str, options: Dict[str, str], findings: str) -> str:
    opts = format_options(options)
    letters = ", ".join(sorted(options.keys()))
    return (
        f"Findings from sub-questions about the video:\n{findings}\n\n"
        f"Using these findings together with the video frames, answer the main question.\n"
        f"Question: {question}\nOptions:\n{opts}\n\n"
        f"Reason briefly, then end with a line in exactly this format:\n"
        f"Answer: <one letter from {letters}>"
    )


def build_answer_from_desc(question: str, options: Dict[str, str], description: str,
                           has_video: bool) -> str:
    opts = format_options(options)
    letters = ", ".join(sorted(options.keys()))
    src = ("the video frames and the description below" if has_video
           else "the description below")
    return (
        f"Detailed description of the video clip:\n\"{description}\"\n\n"
        f"Question: {question}\n\nOptions:\n{opts}\n\n"
        f"Using {src}, reason step by step, then end with a line in exactly this format:\n"
        f"Answer: <one letter from {letters}>"
    )


def format_options(options: Dict[str, str]) -> str:
    return "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))


SPATIAL_HINT = (
    "First describe the spatial layout of the scene (relative left/right, up/down, "
    "front/behind, depth/proximity, and who/what can see what), tracking how it "
    "changes across frames, then reason to the answer. "
)


def build_question_prompt(question: str, options: Dict[str, str], cot: bool = True,
                          style: str = "plain", transcript: str = "") -> str:
    opts = format_options(options)
    letters = ", ".join(sorted(options.keys()))
    tx = ""
    if transcript:
        tx = (f"Audio/dialogue transcript of the clip (may help, may be irrelevant): "
              f"\"{transcript[:1200]}\"\n\n")

    if style == "reasoner":
        # Matches the Video-R1 / VideoChat-R1 training format (think/answer tags).
        return (
            f"{tx}{question}\n\nOptions:\n{opts}\n\n"
            f"Provide your detailed step-by-step reasoning between <think> </think> tags, "
            f"and then give the single option letter (one of {letters}) between "
            f"<answer> </answer> tags."
        )
    if style == "spatial":
        return (
            f"{tx}Question: {question}\n\nOptions:\n{opts}\n\n"
            f"{SPATIAL_HINT}"
            f"End your response with a line in exactly this format:\n"
            f"Answer: <one letter from {letters}>"
        )
    if cot:
        return (
            f"{tx}Question: {question}\n\nOptions:\n{opts}\n\n"
            f"Think step by step about what the video shows, then decide. "
            f"End your response with a line in exactly this format:\n"
            f"Answer: <one letter from {letters}>"
        )
    return (
        f"{tx}Question: {question}\n\nOptions:\n{opts}\n\n"
        f"Answer with the option's letter from the given choices directly "
        f"(one of {letters})."
    )


DEPTH_SYSTEM_PROMPT = (
    "You are an expert at spatial reasoning over video. You judge 3D spatial "
    "relationships from 2D frames using monocular depth cues, and you reason about "
    "ORDER and RELATIONS, never absolute distances. You never invent measurements. "
    "You answer multiple-choice questions by mapping a carefully reconstructed 3D "
    "layout onto exactly one of the given options."
)

# Depth/proximity/relative-position + POV cluster — the categories with the highest
# error rate. Detector is keyword-based and high-precision (answer-blind).
_DEPTH_RE = re.compile(
    r"\b(relative to|located relative|point of view|from .{0,20}perspective|"
    r"in front|behind|nearer|closer|proximity|distance from|how far|"
    r"between the|next to|in relation to)\b", re.IGNORECASE)


# Causal/temporal stems: even if they mention spatial words ("why did X come in front
# of Y"), they are NOT depth questions and must keep the baseline prompt.
_NOT_DEPTH_RE = re.compile(r"^\s*(why|what caused|what made|how did .* (cause|make)|"
                           r"what happens after|because)\b", re.IGNORECASE)


def is_depth_question(question: str) -> bool:
    """True for depth/proximity/relative-position/POV questions (the hardest cluster).
    Excludes causal 'why/what-caused' questions that merely mention spatial words."""
    q = question or ""
    if _NOT_DEPTH_RE.search(q):
        return False
    return bool(_DEPTH_RE.search(q))


def build_depth_prompt(question: str, options: Dict[str, str]) -> str:
    """Cue-guided ordinal depth reasoning (Version A): pairwise occlusion first,
    monocular cue priority, reference-frame pinning, then map to one option."""
    opts = format_options(options)
    letters = ", ".join(sorted(options.keys()))
    return (
        f"Question: {question}\n\nOptions:\n{opts}\n\n"
        "This question is about depth / proximity / relative position. Follow this "
        "procedure exactly. Reason in relative, ordinal terms only — never estimate "
        "distances in meters or any absolute units.\n\n"
        "STEP 0 — Pin the reference frame. Decide which is asked: (a) proximity to the "
        "CAMERA/viewer; (b) the relation BETWEEN two scene objects in 3D; or (c) position "
        "from a NAMED CHARACTER'S OWN egocentric viewpoint (their own left/right/front/back "
        "as they face, NOT the screen's). State the frame in one sentence. Remember: two "
        "things can be adjacent in the image yet far apart in depth — keep image position "
        "and depth separate.\n\n"
        "STEP 1 — Pairwise occlusion (STRONGEST cue, near-certain). For each relevant pair "
        "of target objects, decide whether one object's contour passes unbroken in front of "
        "the other; the unbroken-outline object is NEARER. State who occludes whom.\n\n"
        "STEP 2 — Where occlusion is unavailable, use the highest-ranked cue that applies: "
        "(1) occlusion; (2) relative size for SAME-CLASS objects (larger = nearer); "
        "(3) ground-contact height for objects on a common ground plane (lower in frame = "
        "nearer); (4) sharper texture/focus = nearer; (5) linear perspective (closer to the "
        "vanishing point = farther).\n\n"
        "STEP 3 — Resolve a single front-to-back (nearer→farther) order from the pairwise "
        "judgments. If they form a contradiction/cycle, say so and re-decide that pair with "
        "the next cue down the list.\n\n"
        "STEP 4 — State each target's left/right and above/below image position, transformed "
        "into the Step-0 reference frame (for case (c), rotate screen-left/right into the "
        "character's own left/right).\n\n"
        "STEP 5 — Map the resolved depth order + image position to exactly ONE option. If two "
        "options seem close, pick the one consistent with the STRONGEST cue (occlusion wins).\n\n"
        "If the targets never share a frame, reconstruct their relation from a shot that "
        "establishes the layout or from consistent eyelines/camera moves; do NOT assume "
        "separate shots mean the objects are far apart or facing away.\n\n"
        f"End your response with a line in exactly this format:\nAnswer: <one letter from {letters}>"
    )


def parse_choice(text: str, valid: List[str], fallback: str = "B") -> str:
    """Extract the chosen option letter from free-form model output."""
    if not text:
        return fallback if fallback in valid else valid[0]
    valid_set = set(valid)
    t = text.strip()

    # 0) RL reasoners emit <answer>X</answer> (optionally after <think>...</think>)
    m0 = re.findall(r"<answer>\s*\(?([A-H])\)?", t, flags=re.IGNORECASE)
    for c in reversed(m0):
        if c.upper() in valid_set:
            return c.upper()
    # strip a <think>...</think> block so its letters don't pollute parsing
    t = re.sub(r"<think>.*?</think>", " ", t, flags=re.S | re.IGNORECASE).strip() or text.strip()

    # 1) explicit "Answer: X"
    m = re.findall(r"answer\s*[:\-]?\s*\*{0,2}\(?([A-H])\)?", t, flags=re.IGNORECASE)
    for c in reversed(m):
        if c.upper() in valid_set:
            return c.upper()

    # 2) a standalone letter on the last non-empty line
    for line in reversed([l.strip() for l in t.splitlines() if l.strip()]):
        m2 = re.fullmatch(r"\(?([A-H])\)?[.):]?", line)
        if m2 and m2.group(1).upper() in valid_set:
            return m2.group(1).upper()

    # 3) first valid "(X)" or "X." pattern anywhere
    for c in re.findall(r"\(?\b([A-H])\b\)?[.):]", t):
        if c.upper() in valid_set:
            return c.upper()

    # 4) any valid bare letter, last one wins
    for c in reversed(re.findall(r"\b([A-H])\b", t)):
        if c.upper() in valid_set:
            return c.upper()

    return fallback if fallback in valid_set else valid[0]
