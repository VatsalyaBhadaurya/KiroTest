# Tech Stack

## Language & Runtime
- Python 3.x
- All inference runs on CPU (no CUDA dependency)

## Key Libraries
| Library | Version | Purpose |
|---|---|---|
| fastapi | 0.111.0 | REST API framework |
| uvicorn | 0.29.0 | ASGI server |
| open-clip-torch | 2.24.0 | CLIP model (ViT-B/32) |
| torch | 2.2.2 | Tensor ops + INT8 quantization |
| torchvision | 0.17.2 | Image transforms |
| Pillow | 10.3.0 | Image loading/resizing |
| numpy | 1.26.4 | Embedding math |
| onnxruntime | 1.18.0 | Available for future ONNX export |
| scipy | 1.13.0 | Scientific utilities |
| pytest | (dev) | Unit testing |

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the development server
uvicorn main:app --host 0.0.0.0 --port 8000

# Run tests
pytest tests/

# Profile inference latency
python profile_inference.py
```

## Quantization
- INT8 dynamic quantization applied to all `torch.nn.Linear` layers via `torch.quantization.quantize_dynamic`
- Applied once at model load time; quantized model is cached as a module-level singleton

## Frontend
- Single-page vanilla JS/HTML (`app/index.html`) served directly by FastAPI
- No build step required — no bundler, no npm
