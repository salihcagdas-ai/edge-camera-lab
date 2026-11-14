"""Motion gate tests (numpy only)."""

from __future__ import annotations

import numpy as np

from edge.motion import MotionGate


def _static() -> np.ndarray:
    return np.zeros((32, 32, 3), dtype=np.uint8)


def _moving() -> np.ndarray:
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    frame[:16, :, :] = 200  # half the pixels change vs the static frame
    return frame


def test_disabled_gate_always_active() -> None:
    gate = MotionGate(enabled=False)
    assert gate.active(_static(), now=0.0) is True
    assert gate.active(_static(), now=1000.0) is True


def test_first_frame_fails_open() -> None:
    gate = MotionGate(min_change=0.01, cooldown_s=10.0, downscale=1)
    assert gate.active(_static(), now=0.0) is True  # no baseline yet


def test_idle_scene_goes_inactive_after_cooldown() -> None:
    gate = MotionGate(min_change=0.01, cooldown_s=10.0, downscale=1)
    gate.active(_static(), now=0.0)
    assert gate.active(_static(), now=5.0) is True  # inside cooldown
    assert gate.active(_static(), now=20.0) is False  # cooldown expired, no change


def test_change_reactivates_the_gate() -> None:
    gate = MotionGate(min_change=0.01, cooldown_s=10.0, downscale=1)
    gate.active(_static(), now=0.0)
    gate.active(_static(), now=20.0)  # now inactive
    assert gate.active(_moving(), now=21.0) is True  # scene changed
    assert gate.active(_moving(), now=25.0) is True  # cooldown holds it active
