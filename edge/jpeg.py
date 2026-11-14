"""JPEG encode/decode helpers (OpenCV, lazily imported).

Kept in one tiny module so tests can inject fakes and never need cv2.
"""

from __future__ import annotations

from io import BytesIO

import numpy as np


def encode_jpeg(frame: np.ndarray, quality: int = 85) -> bytes | None:
    import cv2  # lazy: heavy, not needed on the fake/test path

    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buffer.tobytes() if ok else None


def decode_bgr(data: bytes) -> np.ndarray | None:
    import cv2  # lazy: heavy, not needed on the fake/test path
    from PIL import Image, UnidentifiedImageError

    if not data or len(data) > 8 * 1024 * 1024:
        return None
    try:
        # Inspect dimensions before OpenCV allocates the decompressed pixel buffer.
        with Image.open(BytesIO(data)) as header:
            if header.format not in {"JPEG", "PNG"} or header.width * header.height > 20_000_000:
                return None
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, cv2.error):
        return None
    return image
