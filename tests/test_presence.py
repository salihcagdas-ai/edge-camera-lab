"""Presence (home/away) tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from edge.presence import MODE_AWAY, MODE_HOME, PresenceState
from edge.store import HomeStore


def test_defaults_to_home(tmp_path: Path) -> None:
    store = HomeStore(tmp_path)
    assert PresenceState(store).mode == MODE_HOME
    store.close()


def test_mode_persists_across_restart(tmp_path: Path) -> None:
    store = HomeStore(tmp_path)
    PresenceState(store).set_mode(MODE_AWAY)
    assert PresenceState(store).mode == MODE_AWAY  # fresh instance, same store
    store.close()


def test_owner_seen_flips_away_to_home(tmp_path: Path) -> None:
    store = HomeStore(tmp_path)
    presence = PresenceState(store)
    presence.set_mode(MODE_AWAY)
    assert presence.on_owner_seen() is True
    assert presence.mode == MODE_HOME
    assert presence.on_owner_seen() is False  # already home
    kinds = [e.kind for e in store.recent_events()]
    assert kinds.count("mode_change") == 2  # away + back home
    store.close()


def test_setting_same_mode_logs_nothing(tmp_path: Path) -> None:
    store = HomeStore(tmp_path)
    presence = PresenceState(store)
    presence.set_mode(MODE_HOME)  # already home
    assert store.recent_events() == []
    store.close()


def test_invalid_mode_rejected(tmp_path: Path) -> None:
    store = HomeStore(tmp_path)
    with pytest.raises(ValueError):
        PresenceState(store).set_mode("vacation")
    store.close()
