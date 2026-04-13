# Product: VLA Agent

A CPU-optimized Vision-Language Action Agent that takes an image and an optional text instruction, then returns a structured JSON action (e.g. `pick_object`, `navigate_to`).

## Core pipeline

```
image + instruction → vision_encoder → vlm_reasoner → action_generator → JSON action
```

- **Vision encoder**: CLIP ViT-B/32 (OpenAI), INT8 quantized, produces a 512-dim L2-normalized embedding
- **VLM reasoner**: CLIP text tower scores candidate scene labels against the image embedding; no autoregressive LLM
- **Action generator**: Deterministic rule-based mapping from scene understanding to structured output

## Key constraints

- Total model footprint ≤ 1.5 GB (target ~350 MB)
- Warm inference latency target: 200–600 ms on a standard laptop CPU
- No GPU required; all inference runs on CPU
- Output is always a structured JSON action with `action`, `target`, `confidence`, `alternatives`, and `instruction_alignment` fields
