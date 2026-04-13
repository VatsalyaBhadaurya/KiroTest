# Requirements Document

## Introduction

This document defines requirements for a minimal, production-grade CPU-optimized Vision-Language Action (VLA) Agent. The system accepts an image and an optional text instruction via a web UI, uses a lightweight Vision-Language Model to understand the scene, and produces structured JSON actions. All inference must run efficiently on a standard laptop CPU with no GPU dependency, total model weight under 1.5 GB, and end-to-end latency under 2 seconds.

The existing codebase provides `vision_encoder.py` (CLIP ViT-B/32, INT8 quantized), `vlm_reasoner.py` (CLIP text-tower scene scoring), and `action_generator.py` (rule-based action mapping). The requirements below cover the complete system including the web UI, API layer, and all optimization constraints.

---

## Glossary

- **VLA_Agent**: The end-to-end CPU-optimized Vision-Language Action Agent system described in this document.
- **Vision_Encoder**: The module (`vision_encoder.py`) that converts an input image into a 512-dimensional L2-normalized embedding using CLIP ViT-B/32 with INT8 dynamic quantization.
- **VLM_Reasoner**: The module (`vlm_reasoner.py`) that scores candidate scene labels and instruction alignment against the image embedding using the CLIP text tower.
- **Action_Generator**: The module (`action_generator.py`) that maps scene understanding and an optional instruction to a structured JSON action.
- **Web_UI**: The browser-based front-end that accepts image and instruction inputs and displays the structured action output.
- **API_Server**: The FastAPI-based HTTP server (`main.py`) that exposes the inference endpoint and serves the Web_UI.
- **Structured_Action**: A JSON object with at minimum the fields `action` (string), `target` (string), and `confidence` (float in [0, 1]).
- **Instruction**: An optional natural-language string provided by the user to modify the agent's behavior.
- **Model_Cache**: The in-process singleton that holds loaded model weights to avoid repeated disk I/O across requests.
- **Quantization**: The process of reducing model weight precision (INT8 or lower) to decrease memory footprint and improve CPU inference speed.
- **ONNX_Runtime**: The cross-platform inference engine used as an optional backend for exported ONNX models.

---

## Requirements

### Requirement 1: Image Input Acceptance

**User Story:** As a user, I want to upload an image through the web UI, so that the VLA Agent can analyze the scene depicted in it.

#### Acceptance Criteria

1. THE Web_UI SHALL provide an image upload control that accepts JPEG, PNG, and WebP formats.
2. WHEN a user submits an image, THE API_Server SHALL accept the image as a multipart/form-data POST request to the `/infer` endpoint.
3. IF the uploaded file is not a valid image format, THEN THE API_Server SHALL return an HTTP 422 response with a descriptive error message.
4. IF the uploaded image exceeds 10 MB in size, THEN THE API_Server SHALL return an HTTP 413 response with a descriptive error message.
5. THE Vision_Encoder SHALL resize and normalize the input image to 224×224 pixels before encoding, regardless of the original image dimensions.

---

### Requirement 2: Optional Text Instruction

**User Story:** As a user, I want to provide an optional text instruction alongside my image, so that I can guide the agent toward a specific action.

#### Acceptance Criteria

1. THE Web_UI SHALL provide a text input field for an optional instruction string.
2. WHEN an instruction is provided, THE VLM_Reasoner SHALL compute the cosine similarity between the instruction embedding and the image embedding and include the result as `instruction_score` in the scene information.
3. WHEN no instruction is provided, THE Action_Generator SHALL select the primary action based solely on the dominant scene label.
4. WHEN an instruction is provided and contains a recognized action verb, THE Action_Generator SHALL override the object-derived action with the verb-derived action.
5. IF the instruction string exceeds 512 characters, THEN THE API_Server SHALL return an HTTP 422 response with a descriptive error message.

---

### Requirement 3: Scene Understanding

**User Story:** As a developer, I want the system to identify the dominant objects and context in an image, so that the Action_Generator has accurate scene information to work with.

#### Acceptance Criteria

1. WHEN an image is encoded, THE VLM_Reasoner SHALL score the image embedding against all entries in the scene vocabulary using cosine similarity.
2. THE VLM_Reasoner SHALL return the top 5 scene labels sorted by descending cosine similarity score.
3. THE VLM_Reasoner SHALL identify the `dominant_object` as the label with the highest cosine similarity score, with leading articles ("a", "an") stripped.
4. THE VLM_Reasoner SHALL cache the encoded text features for all scene labels after the first inference request, so that subsequent requests do not re-encode the vocabulary.
5. THE Vision_Encoder SHALL cache the loaded model weights in memory after the first load, so that subsequent requests do not reload weights from disk.

---

### Requirement 4: Structured Action Output

**User Story:** As a downstream system, I want the agent to return a structured JSON action, so that I can programmatically consume and execute the recommended action.

#### Acceptance Criteria

1. WHEN inference completes successfully, THE API_Server SHALL return an HTTP 200 response whose body is a JSON object conforming to the Structured_Action schema.
2. THE Structured_Action SHALL contain the field `action` with one of the values: `"pick_object"`, `"navigate_to"`, `"observe_scene"`, or `"interact_with"`.
3. THE Structured_Action SHALL contain the field `target` as a non-empty string identifying the primary object or scene context.
4. THE Structured_Action SHALL contain the field `confidence` as a float rounded to 4 decimal places in the range [0.0, 1.0].
5. THE Structured_Action SHALL contain the field `alternatives` as a list of up to 2 alternative action objects, each with `action`, `target`, and `confidence` fields.
6. THE Structured_Action SHALL contain the field `instruction_alignment` as a float rounded to 4 decimal places when an instruction was provided, and `null` otherwise.
7. THE Structured_Action SHALL contain the field `latency_ms` as a float representing the total end-to-end inference time in milliseconds.

---

### Requirement 5: CPU-Only Inference

**User Story:** As an operator, I want the system to run entirely on CPU, so that it can be deployed on standard laptops and servers without GPU hardware.

#### Acceptance Criteria

1. THE VLA_Agent SHALL perform all model inference on the CPU device without requiring CUDA, ROCm, or any GPU runtime.
2. THE Vision_Encoder SHALL apply INT8 dynamic quantization to all `torch.nn.Linear` layers at model load time using `torch.quantization.quantize_dynamic`.
3. THE VLA_Agent SHALL load no single model whose unquantized weight file exceeds 1.5 GB.
4. WHEN the API_Server starts, THE VLA_Agent SHALL not load any model weights until the first inference request is received (lazy loading).

---

### Requirement 6: Inference Latency

**User Story:** As a user, I want inference to complete quickly, so that I receive action results without noticeable delay.

#### Acceptance Criteria

1. WHEN a warm inference request is made (model already loaded), THE VLA_Agent SHALL complete end-to-end inference in under 2000 ms on a standard laptop CPU (defined as a dual-core x86-64 processor at 2.0 GHz or faster with 8 GB RAM).
2. THE API_Server SHALL include the field `latency_ms` in every successful response, reporting the measured end-to-end inference duration.
3. THE VLM_Reasoner SHALL report the combined image encoding and reasoning latency as a single float in milliseconds.

---

### Requirement 7: Web UI

**User Story:** As a user, I want a browser-based interface, so that I can interact with the VLA Agent without writing code.

#### Acceptance Criteria

1. THE Web_UI SHALL be served by the API_Server at the root path `/`.
2. THE Web_UI SHALL display an image preview after the user selects an image file.
3. WHEN the user submits the form, THE Web_UI SHALL send the image and optional instruction to the `/infer` endpoint and display the returned Structured_Action as formatted JSON.
4. WHILE a request is in flight, THE Web_UI SHALL display a loading indicator and disable the submit button.
5. IF the API_Server returns an error response, THEN THE Web_UI SHALL display the error message to the user.

---

### Requirement 8: Modular Architecture

**User Story:** As a developer, I want the system split into clearly separated modules, so that each component can be tested, replaced, or optimized independently.

#### Acceptance Criteria

1. THE VLA_Agent SHALL implement the vision encoding logic exclusively in `vision_encoder.py`, exposing a public `encode_image(image: PIL.Image) -> (np.ndarray, float)` function.
2. THE VLA_Agent SHALL implement the scene reasoning logic exclusively in `vlm_reasoner.py`, exposing a public `reason(image: PIL.Image, instruction: Optional[str]) -> (dict, float)` function.
3. THE VLA_Agent SHALL implement the action mapping logic exclusively in `action_generator.py`, exposing a public `generate_action(scene_info: dict, instruction: Optional[str]) -> dict` function.
4. THE VLA_Agent SHALL implement the HTTP server and request orchestration exclusively in `main.py`.
5. THE VLA_Agent SHALL serve the Web_UI from a static HTML file at `static/index.html`.

---

### Requirement 9: Inference Profiling

**User Story:** As a developer, I want per-stage timing information, so that I can identify and optimize bottlenecks in the inference pipeline.

#### Acceptance Criteria

1. THE Vision_Encoder SHALL measure and return the image encoding latency in milliseconds as the second element of its return tuple.
2. THE VLM_Reasoner SHALL measure and return the combined reasoning latency (image encoding + text scoring) in milliseconds as the second element of its return tuple.
3. THE API_Server SHALL log the per-request latency breakdown (image encoding ms, reasoning ms, total ms) to stdout at INFO level on every successful inference.

---

### Requirement 10: Dependency and Environment

**User Story:** As a developer, I want all dependencies declared in a single file, so that the environment can be reproduced reliably.

#### Acceptance Criteria

1. THE VLA_Agent SHALL declare all runtime Python dependencies with pinned versions in `requirements.txt`.
2. THE VLA_Agent SHALL be executable with `uvicorn main:app --host 0.0.0.0 --port 8000` after installing `requirements.txt` in a Python 3.10+ environment.
3. THE VLA_Agent SHALL not require any dependency that is unavailable on PyPI or that requires compilation from source on a standard x86-64 Linux, macOS, or Windows environment.
