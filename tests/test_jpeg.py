import struct
import zlib

import numpy as np
import pytest

from edge.jpeg import decode_bgr, encode_jpeg


def test_real_jpeg_roundtrip_and_invalid_input() -> None:
    pytest.importorskip("cv2")
    pytest.importorskip("PIL")
    frame = np.full((16, 16, 3), 100, dtype=np.uint8)
    encoded = encode_jpeg(frame)
    assert encoded is not None
    decoded = decode_bgr(encoded)
    assert decoded is not None and decoded.shape == frame.shape
    assert decode_bgr(b"") is None
    assert decode_bgr(b"not an image") is None


def test_oversized_header_rejected_before_opencv_decode(monkeypatch: pytest.MonkeyPatch) -> None:
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("PIL")

    def chunk(kind: bytes, payload: bytes) -> bytes:
        data = kind + payload
        return struct.pack(">I", len(payload)) + data + struct.pack(">I", zlib.crc32(data))

    data = b"\x89PNG\r\n\x1a\n"
    data += chunk(b"IHDR", struct.pack(">IIBBBBB", 5000, 5000, 8, 2, 0, 0, 0))
    data += chunk(b"IDAT", b"") + chunk(b"IEND", b"")

    def unexpected_decode(*args: object) -> None:
        pytest.fail("Large images must be rejected before allocating decoded pixels")

    monkeypatch.setattr(cv2, "imdecode", unexpected_decode)
    assert decode_bgr(data) is None
