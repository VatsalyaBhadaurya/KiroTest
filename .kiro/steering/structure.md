# Project Structure

```
/
├── main.py                  # FastAPI app, routes: GET /, GET /health, POST /predict
├── vision_encoder.py        # CLIP image tower — encode_image() → (ndarray[512], latency_ms)
├── vlm_reasoner.py          # CLIP text tower — reason() → (scene_info dict, latency_ms)
├── action_generator.py      # Rule-based action mapping — generate_action() → action dict
├── profile_inference.py     # Standalone latency profiler (cold + warm runs)
├── requirements.txt         # Pinned Python dependencies
├── app/
│   └── index.html           # Single-file web UI (vanilla JS, no build step)
└── tests/
    ├── __init__.py
    └── test_vision_encoder.py  # pytest unit tests for vision_encoder
```

## Module Responsibilities

- **`vision_encoder.py`**: Owns model loading and caching (`_model`, `_preprocess` singletons). Always resizes input to 224×224 before preprocessing. Returns L2-normalized float32 embeddings.
- **`vlm_reasoner.py`**: Imports `_load_model` and `encode_image` from `vision_encoder`. Caches text features for `SCENE_LABELS` in `_text_features_cache`. Extend `SCENE_LABELS` list to add new scene vocabulary.
- **`action_generator.py`**: Purely rule-based, no ML. Extend `ACTIONS` dict to map new objects, or `VERB_MAP` to handle new instruction verbs.
- **`main.py`**: Thin API layer — validates input, calls `reason()` then `generate_action()`, returns structured JSON.

## Conventions

- `from __future__ import annotations` at the top of every Python module
- Module-level singletons for expensive resources (model, text feature cache) — load once, reuse across requests
- Public functions return typed tuples where latency is included: `(result, latency_ms)`
- Tests use `unittest.mock.patch` to avoid loading real CLIP weights; reset module globals in `autouse` fixtures
- Test classes group by requirement (e.g. `TestEncodeImageOutput`, `TestResizeGuard`)
- No GPU-specific code; never call `.cuda()` or check `torch.cuda.is_available()`
efqeeaf


