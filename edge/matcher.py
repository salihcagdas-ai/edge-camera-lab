"""Open-set 1:N matching over the local gallery (in-memory, numpy).

Embeddings are unit vectors, so cosine distance = 1 - dot. If the closest
gallery vector is farther than ``max_distance`` we return None ("unknown")
rather than a low-confidence guess — for a security camera the dangerous error
is a stranger classified as a known person.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from edge.store import GalleryEntry


@dataclass(slots=True)
class Match:
    person_id: int
    name: str
    is_owner: bool
    distance: float


class OpenSetMatcher:
    def __init__(self, entries: list[GalleryEntry], max_distance: float) -> None:
        if not np.isfinite(max_distance) or not 0 <= max_distance <= 2:
            raise ValueError("max_distance must be finite and between zero and two")
        self._entries = entries
        self._max_distance = max_distance
        if entries:
            matrix = np.stack([e.vector for e in entries]).astype(np.float64)
            if matrix.shape != (len(entries), 512) or not np.isfinite(matrix).all():
                raise ValueError("gallery embeddings must be finite 512-dimensional vectors")
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            if not np.isfinite(norms).all() or np.any(norms <= 0):
                raise ValueError("gallery embeddings must have a finite non-zero norm")
            self._matrix: np.ndarray | None = matrix / norms
        else:
            self._matrix = None

    @property
    def size(self) -> int:
        return len(self._entries)

    def match(self, embedding: list[float] | np.ndarray) -> Match | None:
        if self._matrix is None:
            return None
        try:
            probe = np.asarray(embedding, dtype=np.float64)
        except (ValueError, TypeError):
            return None
        if probe.shape != (512,) or not np.isfinite(probe).all():
            return None
        norm = float(np.linalg.norm(probe))
        if not np.isfinite(norm) or norm <= 0:
            return None
        probe = probe / norm
        distances = 1.0 - (self._matrix @ probe)
        best = int(np.argmin(distances))
        distance = float(np.clip(distances[best], 0, 2))
        if not np.isfinite(distance) or distance > self._max_distance:
            return None
        entry = self._entries[best]
        return Match(
            person_id=entry.person_id,
            name=entry.name,
            is_owner=entry.is_owner,
            distance=distance,
        )


class MatcherRef:
    """Mutable holder so enrollment (web/CLI) can swap in a rebuilt matcher
    while the analysis loop keeps a stable reference."""

    def __init__(self, matcher: OpenSetMatcher) -> None:
        self.current = matcher

    def rebuild(self, entries: list[GalleryEntry], max_distance: float) -> None:
        self.current = OpenSetMatcher(entries, max_distance)
