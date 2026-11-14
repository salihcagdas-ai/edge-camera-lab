"""Frame sources.

``SyntheticSource`` is dependency-light (tests/dev). ``RTSPSource`` (OpenCV) and
``PiCameraSource`` (picamera2, the default on the Pi 5 + Camera Module 3) import
their heavy deps lazily so the fake path runs anywhere.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator

import numpy as np


class SyntheticSource:
    def __init__(self, frames: Iterable[np.ndarray]) -> None:
        self._frames = list(frames)

    def frames(self) -> Iterator[np.ndarray]:
        yield from self._frames


class RTSPSource:
    def __init__(self, url: str, fps: float = 5.0) -> None:
        self._url = url
        self._interval = 1.0 / fps if fps > 0 else 0.0

    def frames(self) -> Iterator[np.ndarray]:
        import cv2  # lazy: heavy, only needed for a real IP camera

        capture = cv2.VideoCapture(self._url)
        try:
            last = 0.0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                now = time.monotonic()
                if now - last < self._interval:
                    continue
                last = now
                yield frame
        finally:
            capture.release()


class PiCameraSource:
    """Pi Camera Module 3 over CSI via picamera2 (Raspberry Pi OS).

    Note: libcamera's ``RGB888`` is byte-ordered B,G,R in memory — i.e. exactly
    the OpenCV/InsightFace convention — so frames feed the pipeline unconverted.
    """

    def __init__(self, width: int = 1280, height: int = 720, fps: float = 10.0) -> None:
        self._size = (width, height)
        self._interval = 1.0 / fps if fps > 0 else 0.0

    def frames(self) -> Iterator[np.ndarray]:
        from picamera2 import Picamera2  # lazy: Pi-only dependency

        camera = Picamera2()
        config = camera.create_video_configuration(main={"size": self._size, "format": "RGB888"})
        camera.configure(config)
        camera.start()
        try:
            while True:
                frame = camera.capture_array()
                yield frame
                if self._interval:
                    time.sleep(self._interval)
        finally:
            camera.stop()
            camera.close()
