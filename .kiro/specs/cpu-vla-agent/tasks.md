# Implementation Plan: CPU-Optimized VLA Agent

## Overview

Bring the existing module skeletons (`vision_encoder.py`, `vlm_reasoner.py`, `action_generator.py`, `main.py`) to full spec compliance. Work proceeds module-by-module, wiring everything together in `main.py`, then polishing the Web UI.

## Tasks

- [ ] 1. Harden `vision_encoder.py` — image preprocessing and model cache
  - [x] 1.1 Enforce 224×224 resize/normalize in `encode_image` before passing to CLIP preprocess pipeline
    - Confirm `open_clip` preprocess already handles this; add explicit `image.resize((224, 224))` guard if not
    - _Requirements: 1.5_
  - [x] 1.2 Verify INT8 dynamic quantization is applied to all `torch.nn.Linear` layers at load time
    - Confirm `quantize_dynamic` call covers the full model graph; add assertion or log confirming quantization
    - _Requirements: 5.2_
  - [x] 1.3 Confirm lazy-load singleton: model weights must not load until first `encode_image` call
    - No top-level model instantiation; `_load_model()` called only inside `encode_image`
    - _Requirements: 5.4_
  - [-] 1.4 Write unit tests for `encode_image`
    - Test output shape is `(512,)`, dtype `float32`, L2-norm ≈ 1.0
    - Test with JPEG, PNG, and WebP inputs
    - Test that a second call reuses the cached model (mock `_load_model`)
    - _Requirements: 1.5, 3.5, 5.2_

- [ ] 2. Harden `vlm_reasoner.py` — scene scoring and text-feature cache
  - [ ] 2.1 Ensure `reason()` returns exactly the top 5 labels sorted by descending cosine similarity
    - Validate `top_k = 5` and sort order; add assertion in tests
    - _Requirements: 3.1, 3.2_
  - [ ] 2.2 Ensure `dominant_object` strips leading articles ("a", "an") correctly for all vocabulary entries
    - Cover edge cases: "an empty scene", "a cluttered desk"
    - _Requirements: 3.3_
  - [ ] 2.3 Verify text-feature cache (`_text_features_cache`) is populated after first call and reused on subsequent calls
    - _Requirements: 3.4_
  - [ ] 2.4 Ensure `instruction_score` is included in returned dict when instruction is provided, and absent/`None` when not
    - _Requirements: 2.2, 2.3_
  - [ ] 2.5 Write unit tests for `reason()`
    - Test top-5 label count and descending order
    - Test `dominant_object` article stripping
    - Test `instruction_score` present vs. absent
    - Test text-feature cache hit (mock `_load_model`)
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 2.2, 2.3_

- [ ] 3. Checkpoint — Ensure all encoder and reasoner tests pass
  - Run the test suite; ask the user if any questions arise before proceeding.

- [ ] 4. Harden `action_generator.py` — structured action output
  - [ ] 4.1 Ensure `generate_action` returns all required fields: `action`, `target`, `confidence`, `alternatives`, `instruction_alignment`
    - `action` must be one of `"pick_object"`, `"navigate_to"`, `"observe_scene"`, `"interact_with"`
    - `confidence` and each alternative's `confidence` rounded to 4 decimal places
    - `alternatives` list contains at most 2 entries
    - `instruction_alignment` is `None` when no instruction provided
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_
  - [ ] 4.2 Implement instruction verb override: when instruction contains a recognized verb, override object-derived action
    - Verify `VERB_MAP` regex patterns cover all required verbs
    - _Requirements: 2.4_
  - [ ] 4.3 Write unit tests for `generate_action`
    - Test each of the 4 action values is reachable
    - Test verb override takes precedence over object-derived action
    - Test `instruction_alignment` is `None` when no instruction
    - Test `alternatives` length ≤ 2 and all fields present
    - Test `confidence` values are rounded to 4 decimal places
    - _Requirements: 4.2, 4.3, 4.4, 4.5, 4.6, 2.4_

- [ ] 5. Harden `main.py` — API server compliance
  - [ ] 5.1 Rename endpoint from `/predict` to `/infer` and accept `multipart/form-data` POST
    - Update route decorator and any internal references
    - _Requirements: 1.2_
  - [ ] 5.2 Add file format validation: reject non-image MIME types with HTTP 422 and descriptive message
    - Accept `image/jpeg`, `image/png`, `image/webp` only
    - _Requirements: 1.3_
  - [ ] 5.3 Add file size validation: reject uploads > 10 MB with HTTP 413 and descriptive message
    - Read `Content-Length` header or measure raw bytes before decoding
    - _Requirements: 1.4_
  - [ ] 5.4 Add instruction length validation: reject instructions > 512 characters with HTTP 422 and descriptive message
    - _Requirements: 2.5_
  - [ ] 5.5 Ensure successful response body matches `Structured_Action` schema exactly
    - Fields: `action`, `target`, `confidence`, `alternatives`, `instruction_alignment`, `latency_ms`
    - `latency_ms` is the measured end-to-end inference duration in ms
    - _Requirements: 4.1, 4.7, 6.2_
  - [ ] 5.6 Add per-request INFO-level stdout logging of latency breakdown: image encoding ms, reasoning ms, total ms
    - Use Python `logging` at `INFO` level
    - _Requirements: 9.3_
  - [ ] 5.7 Serve Web UI HTML at root path `/` from `app/index.html`
    - Confirm `UI_DIR` points to `app/` and the `GET /` route returns the file contents
    - _Requirements: 7.1, 8.5_
  - [ ] 5.8 Write integration tests for `/infer` endpoint using `httpx.AsyncClient` + FastAPI `TestClient`
    - Test 200 response with valid JPEG upload
    - Test 422 for non-image file
    - Test 413 for oversized file
    - Test 422 for instruction > 512 chars
    - Test response body contains all required `Structured_Action` fields
    - _Requirements: 1.2, 1.3, 1.4, 2.5, 4.1_

- [ ] 6. Checkpoint — Ensure all API tests pass
  - Run the test suite; ask the user if any questions arise before proceeding.

- [ ] 7. Update Web UI (`app/index.html`)
  - [ ] 7.1 Add image upload control accepting JPEG, PNG, WebP (`accept="image/jpeg,image/png,image/webp"`)
    - _Requirements: 1.1_
  - [ ] 7.2 Add image preview: display selected image in an `<img>` element before submission
    - Use `FileReader` API to render preview client-side
    - _Requirements: 7.2_
  - [ ] 7.3 Add optional instruction text input field
    - _Requirements: 2.1_
  - [ ] 7.4 On form submit, POST to `/infer` via `fetch`, display returned JSON as formatted output
    - Pretty-print the `Structured_Action` JSON in a `<pre>` block
    - _Requirements: 7.3_
  - [ ] 7.5 Show loading indicator and disable submit button while request is in flight; re-enable on completion
    - _Requirements: 7.4_
  - [ ] 7.6 Display error message to user when API returns a non-2xx response
    - Parse error detail from response body and render it visibly
    - _Requirements: 7.5_

- [ ] 8. Update `requirements.txt` — pin all runtime dependencies
  - Verify all imports used across modules are covered with pinned versions
  - Confirm `uvicorn[standard]`, `fastapi`, `python-multipart`, `open-clip-torch`, `torch`, `torchvision`, `Pillow`, `numpy` are present
  - Add `httpx` for test client if not already listed
  - _Requirements: 10.1, 10.2, 10.3_

- [ ] 9. Final checkpoint — full pipeline smoke test
  - Ensure all unit and integration tests pass
  - Verify `uvicorn main:app --host 0.0.0.0 --port 8000` starts without errors
  - Ask the user if any questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Each task references specific requirements for traceability
- The `/infer` rename (task 5.1) is a breaking change from the current `/predict` route — update the Web UI fetch call accordingly
- Lazy loading (requirement 5.4) means the first request will be slower; this is expected and acceptable
