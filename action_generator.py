"""
Action Generator — maps scene understanding + optional instruction
to a structured JSON action.

Action selection is deterministic / rule-based on top of CLIP scores,
keeping latency near zero and output fully structured.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Action schema
# ---------------------------------------------------------------------------

ACTIONS = {
    "pick_object":    ["bottle", "cup", "book", "phone", "pen", "remote control",
                       "bag", "box", "plate", "bowl", "knife", "fork", "spoon"],
    "navigate_to":    ["door", "window", "chair", "table", "kitchen",
                       "living room", "office"],
    "observe_scene":  ["empty scene", "cluttered desk", "outdoor scene",
                       "person", "laptop", "keyboard"],
    "interact_with":  ["laptop", "keyboard", "phone", "remote control"],
}

# Instruction verb → action override
VERB_MAP = {
    r"\b(pick|grab|take|get|fetch)\b": "pick_object",
    r"\b(go|move|navigate|walk|approach)\b": "navigate_to",
    r"\b(look|observe|scan|check|inspect)\b": "observe_scene",
    r"\b(use|interact|press|click|type)\b": "interact_with",
}


def _action_for_object(obj: str) -> str:
    for action, targets in ACTIONS.items():
        if any(t in obj for t in targets):
            return action
    return "observe_scene"


def _action_from_instruction(instruction: str) -> Optional[str]:
    low = instruction.lower()
    for pattern, action in VERB_MAP.items():
        if re.search(pattern, low):
            return action
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_action(
    scene_info: Dict,
    instruction: Optional[str] = None,
) -> Dict:
    """
    Returns a structured action dict:
    {
        "action": str,
        "target": str,
        "confidence": float,
        "alternatives": [ {action, target, confidence}, ... ],
        "instruction_alignment": float | None
    }
    """
    top_labels: List[Dict] = scene_info["top_labels"]
    dominant: str = scene_info["dominant_object"]
    inst_score: Optional[float] = scene_info.get("instruction_score")

    # Primary action
    primary_action = _action_for_object(dominant)

    # Override with instruction verb if present
    if instruction:
        verb_action = _action_from_instruction(instruction)
        if verb_action:
            primary_action = verb_action

    primary_conf = float(top_labels[0]["score"])

    # Alternatives from remaining top labels
    alternatives = []
    for item in top_labels[1:3]:
        alt_obj = item["label"].lstrip("a ").lstrip("an ").strip()
        alternatives.append({
            "action": _action_for_object(alt_obj),
            "target": alt_obj,
            "confidence": round(float(item["score"]), 4),
        })

    return {
        "action": primary_action,
        "target": dominant,
        "confidence": round(primary_conf, 4),
        "alternatives": alternatives,
        "instruction_alignment": round(inst_score, 4) if inst_score is not None else None,
    }
