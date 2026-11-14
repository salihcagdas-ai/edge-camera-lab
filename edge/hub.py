"""Latest-frame hub shared between the capture thread, the analysis loop and
the MJPEG stream. Holding only the newest frame (with a sequence number) gives
natural backpressure: slow consumers skip frames instead of queueing them."""

from __future__ import annotations

import threading

import numpy as np


class FrameHub:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._seq = 0

    def publish(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame
            self._seq += 1

    def latest(self) -> tuple[np.ndarray | None, int]:
        with self._lock:
            return self._frame, self._seq
