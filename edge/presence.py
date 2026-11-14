"""Home/away presence.

The mode changes alert wording/urgency: while "away", any person at home is
worth an email. Owner recognition flips the mode to "home" automatically;
"away" is set manually in the web UI (auto-away is a TODO). Persisted in the
kv table so a restart keeps the mode.
"""

from __future__ import annotations

from edge.store import HomeStore

MODE_HOME = "home"
MODE_AWAY = "away"
_KEY = "presence_mode"


class PresenceState:
    def __init__(self, store: HomeStore) -> None:
        self._store = store
        self._mode = store.get_setting(_KEY, MODE_HOME)

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def away(self) -> bool:
        return self._mode == MODE_AWAY

    def set_mode(self, mode: str) -> None:
        if mode not in (MODE_HOME, MODE_AWAY):
            raise ValueError(f"invalid mode: {mode!r}")
        if mode == self._mode:
            return
        self._mode = mode
        self._store.set_setting(_KEY, mode)
        self._store.log_event("mode_change", detail=mode)

    def on_owner_seen(self) -> bool:
        """Auto-flip to home when the owner is recognized. Returns True if flipped."""
        if self._mode == MODE_AWAY:
            self.set_mode(MODE_HOME)
            return True
        return False
