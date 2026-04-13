"""
Unit tests for vision_encoder.encode_image.

Requirements: 1.5, 3.5, 5.2
"""

from __future__ import annotations

import io
import importlib
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pil(width: int = 320, height: int = 240, mode: str = "RGB") -> Image.Image:
    """Return a synthetic solid-colour PIL image."""
    return Image.new(mode, (width, height), color=(128, 64, 32))


def _pil_from_bytes(fmt: str) -> Image.Image:
    """Round-trip a synthetic image through a specific format's bytes."""
    buf = io.BytesIO()
    img = _make_pil()
    img.save(buf, format=fmt)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _make_mock_model() -> MagicMock:
    """Return a mock CLIP model whose encode_image returns a random [1,512] tensor."""
    mock_model = MagicMock()
    mock_model.encode_image.return_value = torch.randn(1, 512)
    return mock_model


def _make_mock_preprocess() -> MagicMock:
    """Return a mock preprocess that returns a [3,224,224] tensor."""
    mock_pre = MagicMock(return_value=torch.randn(3, 224, 224))
    return mock_pre


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_module_globals():
    """Reset vision_encoder module-level singletons before each test."""
    import vision_encoder
    vision_encoder._model = None
    vision_encoder._preprocess = None
    yield
    vision_encoder._model = None
    vision_encoder._preprocess = None


# ---------------------------------------------------------------------------
# Tests: output shape, dtype, L2-norm
# ---------------------------------------------------------------------------

class TestEncodeImageOutput:
    """Validates Requirements 1.5, 3.5."""

    def _run_encode(self, image: Image.Image):
        mock_model = _make_mock_model()
        mock_pre = _make_mock_preprocess()

        with patch("vision_encoder._load_model", return_value=(mock_model, mock_pre)):
            import vision_encoder
            embedding, latency_ms = vision_encoder.encode_image(image)
        return embedding, latency_ms

    def test_output_shape(self):
        embedding, _ = self._run_encode(_make_pil())
        assert embedding.shape == (512,), f"Expected (512,), got {embedding.shape}"

    def test_output_dtype(self):
        embedding, _ = self._run_encode(_make_pil())
        assert embedding.dtype == np.float32, f"Expected float32, got {embedding.dtype}"

    def test_output_l2_norm_approx_one(self):
        embedding, _ = self._run_encode(_make_pil())
        norm = float(np.linalg.norm(embedding))
        assert abs(norm - 1.0) < 1e-5, f"L2 norm expected ≈1.0, got {norm}"

    def test_latency_ms_is_positive_float(self):
        _, latency_ms = self._run_encode(_make_pil())
        assert isinstance(latency_ms, float)
        assert latency_ms >= 0.0


# ---------------------------------------------------------------------------
# Tests: JPEG, PNG, WebP format inputs
# ---------------------------------------------------------------------------

class TestEncodeImageFormats:
    """Validates Requirement 1.5 — accepts JPEG, PNG, WebP."""

    def _run_encode(self, image: Image.Image):
        mock_model = _make_mock_model()
        mock_pre = _make_mock_preprocess()
        with patch("vision_encoder._load_model", return_value=(mock_model, mock_pre)):
            import vision_encoder
            return vision_encoder.encode_image(image)

    def test_jpeg_input(self):
        img = _pil_from_bytes("JPEG")
        embedding, _ = self._run_encode(img)
        assert embedding.shape == (512,)

    def test_png_input(self):
        img = _pil_from_bytes("PNG")
        embedding, _ = self._run_encode(img)
        assert embedding.shape == (512,)

    def test_webp_input(self):
        img = _pil_from_bytes("WEBP")
        embedding, _ = self._run_encode(img)
        assert embedding.shape == (512,)


# ---------------------------------------------------------------------------
# Tests: 224×224 resize guard
# ---------------------------------------------------------------------------

class TestResizeGuard:
    """Validates Requirement 1.5 — image is resized to 224×224 before preprocess."""

    def test_non_224_image_is_resized(self):
        """Preprocess should receive a 224×224 image even when input is larger."""
        received_sizes = []

        def capturing_preprocess(img):
            received_sizes.append(img.size)
            return torch.randn(3, 224, 224)

        mock_model = _make_mock_model()

        with patch("vision_encoder._load_model", return_value=(mock_model, capturing_preprocess)):
            import vision_encoder
            vision_encoder.encode_image(_make_pil(640, 480))

        assert received_sizes == [(224, 224)], f"Expected [(224, 224)], got {received_sizes}"

    def test_already_224_image_not_resized(self):
        """A 224×224 image should pass through without an extra resize call."""
        received_sizes = []

        def capturing_preprocess(img):
            received_sizes.append(img.size)
            return torch.randn(3, 224, 224)

        mock_model = _make_mock_model()

        with patch("vision_encoder._load_model", return_value=(mock_model, capturing_preprocess)):
            import vision_encoder
            vision_encoder.encode_image(_make_pil(224, 224))

        assert received_sizes == [(224, 224)]


# ---------------------------------------------------------------------------
# Tests: model cache reuse
# ---------------------------------------------------------------------------

class TestModelCacheReuse:
    """Validates Requirements 3.5, 5.4 — _load_model called only once."""

    def test_second_call_reuses_cached_model(self):
        mock_model = _make_mock_model()
        mock_pre = _make_mock_preprocess()

        import vision_encoder

        with patch("vision_encoder._load_model", return_value=(mock_model, mock_pre)) as mock_load:
            vision_encoder.encode_image(_make_pil())
            vision_encoder.encode_image(_make_pil())
            assert mock_load.call_count == 2  # called per encode_image invocation

        # Now test the real singleton: _load_model itself only loads weights once
        call_count = 0
        original_create = None

        def fake_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            fake_model = _make_mock_model()
            fake_model.named_modules.return_value = []
            return fake_model, None, _make_mock_preprocess()

        with patch("open_clip.create_model_and_transforms", side_effect=fake_create):
            with patch("torch.quantization.quantize_dynamic", side_effect=lambda m, *a, **kw: m):
                vision_encoder._model = None
                vision_encoder._preprocess = None
                vision_encoder._load_model()
                vision_encoder._load_model()

        assert call_count == 1, "open_clip.create_model_and_transforms should be called only once"
