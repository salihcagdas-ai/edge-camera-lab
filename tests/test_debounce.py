"""Unit tests for confirmation + cooldown logic."""

from __future__ import annotations

from edge.debounce import ConsecutiveConfirmer, Cooldown, MultiTrackConfirmer

# --- ConsecutiveConfirmer (single stream) ---


def test_emits_after_required_consecutive() -> None:
    c = ConsecutiveConfirmer(required=2)
    assert c.observe("a") is None
    assert c.observe("a") == "a"  # threshold reached -> emit once
    assert c.observe("a") is None  # already emitted this run


def test_resets_on_none() -> None:
    c = ConsecutiveConfirmer(required=2)
    c.observe("a")
    assert c.observe(None) is None
    assert c.observe("a") is None  # streak restarted
    assert c.observe("a") == "a"


def test_switching_subject_restarts_streak() -> None:
    c = ConsecutiveConfirmer(required=2)
    c.observe("a")
    assert c.observe("b") is None
    assert c.observe("b") == "b"


def test_reemits_after_a_break() -> None:
    c = ConsecutiveConfirmer(required=2)
    c.observe("a")
    assert c.observe("a") == "a"
    c.observe(None)
    c.observe("a")
    assert c.observe("a") == "a"  # a fresh run re-emits


# --- MultiTrackConfirmer (per-identity, multi-face frames) ---


def test_multi_emits_each_identity_once() -> None:
    c = MultiTrackConfirmer(required=2)
    assert c.observe({"a"}) == set()
    assert c.observe({"a"}) == {"a"}
    assert c.observe({"a"}) == set()  # continuous presence -> no re-emit


def test_multi_absence_resets_and_rearms() -> None:
    c = MultiTrackConfirmer(required=2)
    c.observe({"a"})
    assert c.observe(set()) == set()  # streak broken
    c.observe({"a"})
    assert c.observe({"a"}) == {"a"}  # fresh appearance re-emits


def test_multi_tracks_identities_independently() -> None:
    c = MultiTrackConfirmer(required=2)
    assert c.observe({"a", "unknown"}) == set()
    assert c.observe({"a", "unknown"}) == {"a", "unknown"}
    # "a" leaves, unknown stays: only "a" re-arms
    assert c.observe({"unknown"}) == set()
    c.observe({"a", "unknown"})
    assert c.observe({"a", "unknown"}) == {"a"}


def test_multi_requires_consecutive_not_total() -> None:
    c = MultiTrackConfirmer(required=3)
    c.observe({"a"})
    c.observe({"a"})
    c.observe(set())  # gap
    c.observe({"a"})
    c.observe({"a"})
    assert c.observe({"a"}) == {"a"}


# --- Cooldown ---


def test_cooldown_blocks_within_window() -> None:
    cd = Cooldown(seconds=60)
    assert cd.ready("x", now=0.0) is True
    assert cd.ready("x", now=30.0) is False
    assert cd.ready("x", now=61.0) is True


def test_cooldown_keys_are_independent() -> None:
    cd = Cooldown(seconds=60)
    assert cd.ready("x", now=0.0) is True
    assert cd.ready("y", now=1.0) is True
