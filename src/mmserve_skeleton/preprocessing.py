"""Cheap request feature extraction before scheduling.

The scheduler will eventually use request features such as prompt length, image
size, and resolution bucket to estimate serving cost. This module keeps the
feature extraction intentionally lightweight so it can run before queueing.

DeepFeatureExtractor is optional and uses CLIP for image/text embeddings. It is
separate from MetadataExtractor so the cheap path stays fast.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

from .models import MMRequest


class MetadataExtractor:
    """Attach cheap text/image features to an MMRequest."""

    def enrich(self, request: MMRequest) -> MMRequest:
        """Attach cheap metadata to a request before it enters the queue."""
        request.features.text_length = len(request.prompt.split())
        request.features.num_images = 1 if request.image_path else 0

        width, height = self._image_size(request.image_path)
        request.features.image_width = width
        request.features.image_height = height
        request.features.resolution_bucket = self._resolution_bucket(width, height)
        return request

    def _image_size(self, image_path: str | None) -> tuple[int | None, int | None]:
        """Return image width/height when a local image path is available."""
        if not image_path or not Path(image_path).exists():
            return None, None

        try:
            from PIL import Image

            with Image.open(image_path) as image:
                return image.size
        except Exception:
            return self._image_size_without_pillow(image_path)

    def _image_size_without_pillow(self, image_path: str) -> tuple[int | None, int | None]:
        """Best-effort PNG/JPEG size parsing when Pillow is unavailable."""
        try:
            with open(image_path, "rb") as handle:
                header = handle.read(32)
        except OSError:
            return None, None

        if header.startswith(b"\x89PNG\r\n\x1a\n") and len(header) >= 24:
            width, height = struct.unpack(">II", header[16:24])
            return int(width), int(height)

        if header.startswith(b"\xff\xd8"):
            return self._jpeg_size(image_path)

        return None, None

    def _jpeg_size(self, image_path: str) -> tuple[int | None, int | None]:
        """Read JPEG SOF markers to find image dimensions."""
        try:
            with open(image_path, "rb") as handle:
                handle.read(2)
                while True:
                    marker_start = handle.read(1)
                    if marker_start != b"\xff":
                        return None, None
                    marker = handle.read(1)
                    while marker == b"\xff":
                        marker = handle.read(1)
                    if marker in {b"\xc0", b"\xc1", b"\xc2", b"\xc3"}:
                        handle.read(3)
                        height, width = struct.unpack(">HH", handle.read(4))
                        return int(width), int(height)
                    length_bytes = handle.read(2)
                    if len(length_bytes) != 2:
                        return None, None
                    segment_length = struct.unpack(">H", length_bytes)[0]
                    handle.seek(segment_length - 2, os.SEEK_CUR)
        except OSError:
            return None, None

    def _resolution_bucket(self, width: int | None, height: int | None) -> str:
        """Bucket image resolution for simple scheduling/log analysis."""
        if width is None or height is None:
            return "none"
        pixels = width * height
        if pixels <= 512 * 512:
            return "small"
        if pixels <= 1024 * 1024:
            return "medium"
        return "large"


class DeepFeatureExtractor:
    """CLIP-based image and text embedding extractor.

    Embeddings are stored on request.features.image_embedding and
    request.features.text_embedding as normalized float lists. They are kept
    in memory only and are not written to JSONL logs.

    Requires: pip install transformers torch pillow
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32") -> None:
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Install transformers and torch to use DeepFeatureExtractor: "
                "pip install transformers torch"
            ) from exc
        self._processor = CLIPProcessor.from_pretrained(model_name)
        self._model = CLIPModel.from_pretrained(model_name)
        self._model.eval()

    def extract_image_embedding(self, image_path: str) -> list[float] | None:
        """Return a normalized CLIP image embedding as a Python list."""
        try:
            import torch
            from PIL import Image

            with Image.open(image_path) as img:
                loaded = img.convert("RGB")
            inputs = self._processor(images=loaded, return_tensors="pt")
            with torch.no_grad():
                embedding = self._model.get_image_features(**inputs)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            return embedding.squeeze().tolist()
        except Exception:
            return None

    def extract_text_embedding(self, text: str) -> list[float] | None:
        """Return a normalized CLIP text embedding as a Python list."""
        try:
            import torch

            inputs = self._processor(
                text=[text], return_tensors="pt", truncation=True, max_length=77
            )
            with torch.no_grad():
                embedding = self._model.get_text_features(**inputs)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            return embedding.squeeze().tolist()
        except Exception:
            return None

    def enrich(self, request: MMRequest) -> MMRequest:
        """Attach CLIP embeddings to request.features in-place."""
        request.features.text_embedding = self.extract_text_embedding(request.prompt)
        if request.image_path and Path(request.image_path).exists():
            request.features.image_embedding = self.extract_image_embedding(request.image_path)
        return request
