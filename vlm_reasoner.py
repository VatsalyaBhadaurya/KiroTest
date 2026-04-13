"""
VLM Reasoner — uses CLIP text tower to score candidate scene labels
and instruction alignment against the image embedding.

No autoregressive LLM → stays CPU-fast and well under 1.5 GB.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import open_clip

from vision_encoder import _load_model, encode_image
from PIL import Image


# ---------------------------------------------------------------------------
# Scene vocabulary — extend as needed
# ---------------------------------------------------------------------------

SCENE_LABELS: List[str] = [
    "a bottle", "a cup", "a book", "a phone", "a keyboard",
    "a chair", "a table", "a person", "a door", "a window",
    "a box", "a bag", "a pen", "a laptop", "a remote control",
    "a plate", "a bowl", "a knife", "a fork", "a spoon",
    "an empty scene", "a cluttered desk", "an outdoor scene",
    "a kitchen", "a living room", "an office",
]

_text_features_cache: Optional[torch.Tensor] = None


def _get_text_features() -> torch.Tensor:
    """Encode all scene labels once and cache."""
    global _text_features_cache
    if _text_features_cache is None:
        model, _ = _load_model()
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        tokens = tokenizer(SCENE_LABELS)
        with torch.no_grad():
            feats = model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        _text_features_cache = feats
    return _text_features_cache


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def reason(
    image: Image.Image,
    instruction: Optional[str] = None,
) -> Tuple[Dict, float]:
    """
    Returns (scene_info dict, total_latency_ms).

    scene_info keys:
        top_labels   : list of {label, score} sorted by score desc
        instruction_score : cosine similarity of instruction to image (or None)
        dominant_object   : best matching object label (stripped)
    """
    img_emb, img_ms = encode_image(image)
    img_tensor = torch.from_numpy(img_emb).unsqueeze(0)  # [1, 512]

    t0 = time.perf_counter()
    text_feats = _get_text_features()                    # [N, 512]

    # Cosine similarities
    sims = (img_tensor @ text_feats.T).squeeze(0)        # [N]
    sims_np = sims.numpy()

    top_k = 5
    top_idx = np.argsort(sims_np)[::-1][:top_k]
    top_labels = [
        {"label": SCENE_LABELS[i], "score": float(sims_np[i])}
        for i in top_idx
    ]

    # Instruction alignment
    instruction_score = None
    if instruction:
        model, _ = _load_model()
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        tokens = tokenizer([instruction])
        with torch.no_grad():
            inst_feat = model.encode_text(tokens)
            inst_feat = inst_feat / inst_feat.norm(dim=-1, keepdim=True)
        instruction_score = float((img_tensor @ inst_feat.T).squeeze())

    reason_ms = (time.perf_counter() - t0) * 1000

    # Dominant object: strip leading article
    raw = top_labels[0]["label"]
    dominant = raw.lstrip("a ").lstrip("an ").strip()

    return {
        "top_labels": top_labels,
        "dominant_object": dominant,
        "instruction_score": instruction_score,
    }, img_ms + reason_ms
