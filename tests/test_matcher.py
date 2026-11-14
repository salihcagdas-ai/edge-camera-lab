"""Open-set matcher tests (numpy only)."""

from __future__ import annotations

import numpy as np
import pytest

from edge.matcher import MatcherRef, OpenSetMatcher
from edge.store import GalleryEntry


def _unit(index: int) -> np.ndarray:
    vec = np.zeros(512, dtype=np.float32)
    vec[index] = 1.0
    return vec


def _entry(person_id: int, name: str, index: int, *, is_owner: bool = False) -> GalleryEntry:
    return GalleryEntry(person_id=person_id, name=name, is_owner=is_owner, vector=_unit(index))


def test_empty_gallery_matches_nothing() -> None:
    matcher = OpenSetMatcher([], max_distance=0.55)
    assert matcher.size == 0
    assert matcher.match(_unit(0)) is None


def test_exact_vector_matches_with_zero_distance() -> None:
    matcher = OpenSetMatcher([_entry(1, "Demo Person", 0, is_owner=True)], max_distance=0.55)
    match = matcher.match(_unit(0))
    assert match is not None
    assert match.name == "Demo Person"
    assert match.is_owner is True
    assert match.distance < 1e-6


def test_far_probe_is_open_set_rejected() -> None:
    # Orthogonal vectors -> cosine distance 1.0 > threshold -> "unknown".
    matcher = OpenSetMatcher([_entry(1, "Demo Person", 0)], max_distance=0.55)
    assert matcher.match(_unit(1)) is None


def test_nearest_of_multiple_people_wins() -> None:
    matcher = OpenSetMatcher([_entry(1, "Demo Person", 0), _entry(2, "Ali", 1)], max_distance=0.55)
    probe = (0.9 * _unit(1) + 0.1 * _unit(0)).astype(np.float32)
    match = matcher.match(probe)
    assert match is not None
    assert match.name == "Ali"


def test_zero_probe_matches_nothing() -> None:
    matcher = OpenSetMatcher([_entry(1, "Demo Person", 0)], max_distance=0.55)
    assert matcher.match(np.zeros(512, dtype=np.float32)) is None


def test_matcher_ref_rebuild_swaps_instance() -> None:
    ref = MatcherRef(OpenSetMatcher([], max_distance=0.55))
    assert ref.current.size == 0
    ref.rebuild([_entry(1, "Demo Person", 0)], max_distance=0.55)
    assert ref.current.size == 1
    assert ref.current.match(_unit(0)) is not None


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_probe_cannot_match_an_owner(value: float) -> None:
    matcher = OpenSetMatcher([_entry(1, "Owner", 0, is_owner=True)], max_distance=0.55)
    probe = _unit(0)
    probe[1] = value
    assert matcher.match(probe) is None


@pytest.mark.parametrize("probe", [[], [1.0], [[1.0] * 512], [0.0] * 512])
def test_malformed_or_zero_probe_is_rejected(probe: list) -> None:
    matcher = OpenSetMatcher([_entry(1, "Owner", 0, is_owner=True)], max_distance=0.55)
    assert matcher.match(probe) is None


@pytest.mark.parametrize("threshold", [np.nan, np.inf, -0.1, 2.1])
def test_invalid_match_threshold_is_rejected(threshold: float) -> None:
    with pytest.raises(ValueError):
        OpenSetMatcher([], max_distance=threshold)


@pytest.mark.parametrize("bad", [np.zeros(512), np.full(512, np.nan), np.ones(511)])
def test_invalid_gallery_vector_is_rejected(bad: np.ndarray) -> None:
    entry = GalleryEntry(person_id=1, name="Owner", is_owner=True, vector=bad)
    with pytest.raises(ValueError):
        OpenSetMatcher([entry], max_distance=0.55)
