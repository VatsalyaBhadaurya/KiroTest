"""
VLA Agent — FastAPI backend
Endpoints:
  POST /predict   — image + optional instruction → structured action JSON
  GET  /health    — liveness check
  GET  /          — serves the web UI
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from vlm_reasoner import reason
from action_generator import generate_action

app = FastAPI(title="VLA Agent", version="1.0.0")

# Serve static UI
UI_DIR = Path(__file__).parent / "app"
app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
    html = (UI_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    instruction: Optional[str] = Form(default=None),
):
    # Validate image
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    raw = await image.read()
    try:
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot decode image: {e}")

    t_start = time.perf_counter()

    # Pipeline
    scene_info, reasoning_ms = reason(pil_img, instruction or None)
    action = generate_action(scene_info, instruction or None)

    total_ms = (time.perf_counter() - t_start) * 1000

    return JSONResponse({
        "action": action,
        "scene": {
            "top_labels": scene_info["top_labels"][:3],
            "dominant_object": scene_info["dominant_object"],
        },
        "meta": {
            "reasoning_ms": round(reasoning_ms, 1),
            "total_ms": round(total_ms, 1),
            "instruction": instruction,
        },
    })
