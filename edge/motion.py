"""Cheap motion gate.

Face detection is the expensive step on the Pi 5 CPU, so idle frames are
skipped: recognition only runs while the scene is changing (plus a cooldown so
a person who stops moving is still analyzed). Pure numpy — no OpenCV — so it is
fully unit-testable and adds ~zero dependency weight.
"""

from __future__ import annotations

import numpy as np


class MotionGate:
    def __init__(
        self,
        min_change: float = 0.01,
        cooldown_s: float = 10.0,
        downscale: int = 4,
        pixel_threshold: int = 25,
        enabled: bool = True,
    ) -> None:
        if downscale < 1:
            raise ValueError("downscale must be >= 1")
        self._min_change = min_change
        self._cooldown_s = cooldown_s
        self._downscale = downscale
        self._pixel_threshold = pixel_threshold
        self._enabled = enabled
        self._previous: np.ndarray | None = None
        self._last_motion: float | None = None

    def _grayscale_small(self, frame: np.ndarray) -> np.ndarray:
        small = frame[:: self._downscale, :: self._downscale]
        if small.ndim == 3:
            small = small.mean(axis=2)
        return small.astype(np.int16)

    def active(self, frame: np.ndarray, now: float) -> bool:
        """Feed one frame; return True while recognition should run."""
        if not self._enabled:
            return True
        current = self._grayscale_small(frame)
        previous, self._previous = self._previous, current
        if previous is None or previous.shape != current.shape:
            # First frame / resolution change: treat as motion (fail open).
            self._last_motion = now
            return True
        changed = np.abs(current - previous) > self._pixel_threshold
        if float(changed.mean()) >= self._min_change:
            self._last_motion = now
        return self._last_motion is not None and (now - self._last_motion) <= self._cooldown_s
