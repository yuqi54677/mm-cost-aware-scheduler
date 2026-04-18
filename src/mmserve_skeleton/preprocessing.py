"""
metadata extraction.

The scheduler will eventually need cheap request features such as prompt length,
image size, and resolution bucket. This skeleton computes prompt length now and
keeps the image metadata code stubbed for when r eal image files are connected.

Author: ()
"""

from __future__ import annotations

from mmserve_skeleton.models import MMRequest


class MetadataExtractor:
    """implement image feature extraction"""

    def enrich(self, request: MMRequest) -> MMRequest:
        """Attach cheap metadata to a request before it enters the queue."""
        request.features.text_length = len(request.prompt)
        # request.features.num_images = 1 if request.image_path else 0

        # width, height = self._image_size(request.image_path)
        # request.features.image_width = width
        # request.features.image_height = height
        # request.features.resolution_bucket = self._resolution_bucket(width, height)
        print("add extracted metadata to request feature") # just to see the workflow
        print(request.prompt)
        return request

    # def _image_size(self, image_path: str | None) -> tuple[int | None, int | None]:
    #     if not image_path or not os.path.exists(image_path):
    #         return None, None

    #     try:
    #         from PIL import Image

    #         with Image.open(image_path) as image:
    #             return image.size
    #     except Exception:
    #         return self._image_size_without_pillow(image_path)

    # def _image_size_without_pillow(self, image_path: str) -> tuple[int | None, int | None]:
    #     try:
    #         with open(image_path, "rb") as handle:
    #             header = handle.read(32)
    #     except OSError:
    #         return None, None

    #     if header.startswith(b"\x89PNG\r\n\x1a\n") and len(header) >= 24:
    #         width, height = struct.unpack(">II", header[16:24])
    #         return int(width), int(height)

    #     if header.startswith(b"\xff\xd8"):
    #         return self._jpeg_size(image_path)

    #     return None, None

    # def _jpeg_size(self, image_path: str) -> tuple[int | None, int | None]:
    #     try:
    #         with open(image_path, "rb") as handle:
    #             handle.read(2)
    #             while True:
    #                 marker_start = handle.read(1)
    #                 if marker_start != b"\xff":
    #                     return None, None
    #                 marker = handle.read(1)
    #                 while marker == b"\xff":
    #                     marker = handle.read(1)
    #                 if marker in {b"\xc0", b"\xc1", b"\xc2", b"\xc3"}:
    #                     handle.read(3)
    #                     height, width = struct.unpack(">HH", handle.read(4))
    #                     return int(width), int(height)
    #                 length_bytes = handle.read(2)
    #                 if len(length_bytes) != 2:
    #                     return None, None
    #                 segment_length = struct.unpack(">H", length_bytes)[0]
    #                 handle.seek(segment_length - 2, os.SEEK_CUR)
    #     except OSError:
    #         return None, None

    # def _resolution_bucket(self, width: int | None, height: int | None) -> str:
    #     if width is None or height is None:
    #         return "unknown"
    #     pixels = width * height
    #     if pixels <= 512 * 512:
    #         return "small"
    #     if pixels <= 1024 * 1024:
    #         return "medium"
    #     return "large"
