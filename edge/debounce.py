"""Low-false-positive confirmation + notification rate limiting.

``ConsecutiveConfirmer`` tracks a single subject stream (kept for simple
callers/tests). ``MultiTrackConfirmer`` generalizes it to several concurrent
identities per frame — including the pseudo-identity "unknown" — emitting each
identity once per continuous appearance. ``Cooldown`` additionally rate-limits
emails per subject. All pure logic — fully unit-testable.
"""

from __future__ import annotations


class ConsecutiveConfirmer:
    def __init__(self, required: int = 2) -> None:
        if required < 1:
            raise ValueError("required must be >= 1")
        self._required = required
        self._current: str | None = None
        self._count = 0
        self._emitted: str | None = None

    def observe(self, subject_id: str | None) -> str | None:
        """Feed one frame's match (subject_id or None). Return the subject_id
        exactly once, on the frame where it reaches the consecutive threshold."""
        if subject_id is None:
            self._reset()
            return None
        if subject_id == self._current:
            self._count += 1
        else:
            self._current = subject_id
            self._count = 1
            self._emitted = None
        if self._count >= self._required and self._emitted != subject_id:
            self._emitted = subject_id
            return subject_id
        return None

    def _reset(self) -> None:
        self._current = None
        self._count = 0
        self._emitted = None


class MultiTrackConfirmer:
    """Per-identity N-consecutive-frames confirmation for multi-face frames.

    An identity is confirmed once it appears in ``required`` consecutive
    observations; it is emitted exactly once and re-armed only after it
    disappears (so one visitor = one event, however long they stand there).
    """

    def __init__(self, required: int = 2) -> None:
        if required < 1:
            raise ValueError("required must be >= 1")
        self._required = required
        self._counts: dict[str, int] = {}
        self._emitted: set[str] = set()

    def observe(self, present: set[str]) -> set[str]:
        """Feed the identities seen in one frame; return newly confirmed ones."""
        confirmed: set[str] = set()
        for key in present:
            count = self._counts.get(key, 0) + 1
            self._counts[key] = count
            if count >= self._required and key not in self._emitted:
                self._emitted.add(key)
                confirmed.add(key)
        for key in list(self._counts):
            if key not in present:
                del self._counts[key]
                self._emitted.discard(key)
        return confirmed


class Cooldown:
    """Per-key rate limiter (e.g. one alarm email per subject per N seconds)."""

    def __init__(self, seconds: float) -> None:
        self._seconds = seconds
        self._last: dict[str, float] = {}

    def ready(self, key: str, now: float) -> bool:
        """Return True (and start the window) if ``key`` is outside its cooldown."""
        last = self._last.get(key)
        if last is not None and (now - last) < self._seconds:
            return False
        self._last[key] = now
        return True
