"""
Vision Encoder — CLIP ViT-B/32 image tower (CPU, cached, lazy-loaded).
Produces a 512-dim L2-normalized embedding from a PIL image.
"""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Tuple

import numpy as np
import torch
import open_clip
from PIL import Image


# ---------------------------------------------------------------------------
# Singleton model cache — loaded once, reused across requests
# ---------------------------------------------------------------------------

_model = None
_preprocess = None
_device = torch.device("cpu")


def _load_model():
    global _model, _preprocess
    if _model is None:
        print("[VisionEncoder] Loading CLIP ViT-B/32 …")
        _model, _, _preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        _model.eval()
        # Quantize to INT8 for faster CPU inference
        _model = torch.quantization.quantize_dynamic(
            _model, {torch.nn.Linear}, dtype=torch.qint8
        )
        print("[VisionEncoder] Model ready (INT8 quantized).")
    return _model, _preprocess


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def encode_image(image: Image.Image) -> Tuple[np.ndarray, float]:
    """
    Returns (embedding: float32 ndarray shape [512], latency_ms: float).
    """
    model, preprocess = _load_model()

    t0 = time.perf_counter()
    tensor = preprocess(image).unsqueeze(0)          # [1, 3, 224, 224]
    with torch.no_grad():
        features = model.encode_image(tensor)        # [1, 512]
        features = features / features.norm(dim=-1, keepdim=True)  # L2 norm
    latency_ms = (time.perf_counter() - t0) * 1000

    return features.squeeze(0).numpy().astype(np.float32), latency_ms
