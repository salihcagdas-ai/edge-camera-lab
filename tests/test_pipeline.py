"""End-to-end pipeline tests with fakes (numpy only — no model/camera/SMTP).

``FakeEmbedder`` is deterministic per frame content, so enrolling the embedding
of frame(10) makes frame(10) a "known" face while frame(20) stays "unknown"
(independent random unit vectors are ~orthogonal in 512-d).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from edge.clips import ClipRecorder
from edge.debounce import Cooldown, MultiTrackConfirmer
from edge.detect import Detection, FakeObjectDetector
from edge.embedder import FakeEmbedder
from edge.matcher import MatcherRef, OpenSetMatcher
from edge.motion import MotionGate
from edge.notify import NullNotifier
from edge.pipeline import RecognitionPipeline, run
from edge.presence import MODE_AWAY, MODE_HOME, PresenceState
from edge.store import HomeStore


def _frame(value: int) -> np.ndarray:
    return np.full((4, 4, 3), value, dtype=np.uint8)


def _embedding_of(value: int) -> list[float]:
    faces = FakeEmbedder().embed_all(_frame(value))
    assert faces, "fake embedder must detect a face on a non-zero frame"
    return faces[0][0]


class _Fixture:
    def __init__(self, tmp_path: Path, **overrides: object) -> None:
        self.store = HomeStore(tmp_path)
        self.notifier = NullNotifier(capture_for_tests=True)
        self.presence = PresenceState(self.store)
        self.matcher_ref = MatcherRef(OpenSetMatcher([], max_distance=0.55))
        kwargs: dict = {
            "embedder": FakeEmbedder(),
            "matcher": lambda: self.matcher_ref.current,
            "confirmer": MultiTrackConfirmer(required=2),
            "motion": MotionGate(enabled=False),
            "presence": self.presence,
            "store": self.store,
            "notifier": self.notifier,
            "notify_cooldown": Cooldown(seconds=600),
            "encode_jpeg": lambda frame: b"fake-jpeg",
        }
        kwargs.update(overrides)
        self.pipeline = RecognitionPipeline(**kwargs)

    def enroll(self, name: str, value: int, *, is_owner: bool = False) -> None:
        person_id = self.store.add_person(name, is_owner=is_owner)
        self.store.add_embedding(person_id, _embedding_of(value))
        self.matcher_ref.rebuild(self.store.gallery_entries(), max_distance=0.55)


async def test_no_face_frames_do_nothing(tmp_path: Path) -> None:
    fx = _Fixture(tmp_path)
    outcome = await fx.pipeline.process(np.zeros((4, 4, 3), dtype=np.uint8))
    assert outcome.known == [] and outcome.unknown_confirmed is False
    assert fx.store.recent_events() == []
    assert fx.notifier.sent == []


async def test_owner_recognition_confirms_notifies_and_flips_home(tmp_path: Path) -> None:
    fx = _Fixture(tmp_path)
    fx.enroll("Demo Person", 10, is_owner=True)
    fx.presence.set_mode(MODE_AWAY)

    first = await fx.pipeline.process(_frame(10))
    assert first.known == []  # 2-consecutive-frame guard: not yet
    second = await fx.pipeline.process(_frame(10))
    assert [m.name for m in second.known] == ["Demo Person"]

    assert fx.presence.mode == MODE_HOME  # auto-flip on owner
    kinds = [e.kind for e in fx.store.recent_events()]
    assert "known_person" in kinds
    assert len(fx.notifier.sent) == 1
    subject, _body, has_photo = fx.notifier.sent[0]
    assert "Demo Person" in subject and has_photo is True

    event = next(e for e in fx.store.recent_events() if e.kind == "known_person")
    assert event.snapshot is not None
    path = fx.store.snapshot_path(event.snapshot)
    assert path is not None and path.read_bytes() == b"fake-jpeg"


async def test_unknown_face_alarms_once_per_appearance(tmp_path: Path) -> None:
    fx = _Fixture(tmp_path)  # empty gallery -> everything is unknown
    await run([_frame(20), _frame(20), _frame(20)], fx.pipeline)
    events = [e for e in fx.store.recent_events() if e.kind == "unknown_person"]
    assert len(events) == 1  # confirmed once per continuous appearance
    assert len(fx.notifier.sent) == 1
    assert "ALARM" in fx.notifier.sent[0][0]


async def test_alarm_email_respects_cooldown_but_events_still_log(tmp_path: Path) -> None:
    fx = _Fixture(tmp_path)
    # Appearance 1 (confirmed), gap, appearance 2 (confirmed again, same run).
    await run(
        [_frame(20), _frame(20), np.zeros((4, 4, 3), dtype=np.uint8), _frame(20), _frame(20)],
        fx.pipeline,
    )
    events = [e for e in fx.store.recent_events() if e.kind == "unknown_person"]
    assert len(events) == 2  # both appearances logged
    assert len(fx.notifier.sent) == 1  # second email suppressed by cooldown


async def test_known_friend_while_away_mentions_absence(tmp_path: Path) -> None:
    fx = _Fixture(tmp_path)
    fx.enroll("Ali", 30)
    fx.presence.set_mode(MODE_AWAY)
    await run([_frame(30), _frame(30)], fx.pipeline)
    assert fx.presence.mode == MODE_AWAY  # only the owner flips presence
    assert len(fx.notifier.sent) == 1
    subject, body, _ = fx.notifier.sent[0]
    assert "Ali" in subject
    assert "dışarıdayken" in body


async def test_motion_gate_skips_recognition(tmp_path: Path) -> None:
    fx = _Fixture(
        tmp_path,
        motion=MotionGate(min_change=0.01, cooldown_s=0.0, downscale=1),
        clock=iter(range(100)).__next__,
    )
    fx.enroll("Demo Person", 10, is_owner=True)
    await fx.pipeline.process(_frame(10))  # t=0: fail-open baseline frame
    # t=1..2: identical frames, zero change, cooldown 0 -> gated, never embedded
    await fx.pipeline.process(_frame(10))
    await fx.pipeline.process(_frame(10))
    assert fx.store.recent_events() == []  # confirmation never reached 2 frames


class _ScriptedMotion(MotionGate):
    def __init__(self, active: list[bool]) -> None:
        super().__init__()
        self._script = iter(active)

    def active(self, frame: np.ndarray, now: float) -> bool:
        return next(self._script)


async def test_motion_gap_breaks_evidence_and_rearms_a_later_appearance(tmp_path: Path) -> None:
    fx = _Fixture(tmp_path, motion=_ScriptedMotion([True, False, True, True, False, True, True]))
    fx.enroll("Owner", 10, is_owner=True)
    for _ in range(3):
        assert (await fx.pipeline.process(_frame(10))).known == []
    assert len((await fx.pipeline.process(_frame(10))).known) == 1
    for _ in range(2):
        assert (await fx.pipeline.process(_frame(10))).known == []
    assert len((await fx.pipeline.process(_frame(10))).known) == 1


class _InvalidEmbedder(FakeEmbedder):
    def embed_all(self, frame: np.ndarray) -> list[tuple[list[float], float]]:
        return [([float("nan")] * 512, 0.99)]


async def test_invalid_probe_cannot_disarm_the_camera(tmp_path: Path) -> None:
    fx = _Fixture(tmp_path, embedder=_InvalidEmbedder())
    fx.enroll("Owner", 10, is_owner=True)
    fx.presence.set_mode(MODE_AWAY)
    await run([_frame(10), _frame(10)], fx.pipeline)
    assert fx.presence.mode == MODE_AWAY
    assert not any(event.kind == "known_person" for event in fx.store.recent_events())


# --- Object detector policy (person / cat) ---


def _person(confidence: float = 0.9) -> Detection:
    return Detection(class_id=0, label="person", confidence=confidence, bbox=(0, 0, 10, 20))


def _cat(confidence: float = 0.8) -> Detection:
    return Detection(class_id=15, label="cat", confidence=confidence, bbox=(0, 0, 5, 5))


async def test_person_without_face_alarms_only_while_away(tmp_path: Path) -> None:
    fx = _Fixture(tmp_path, detector=FakeObjectDetector([[_person()]]))
    fx.presence.set_mode(MODE_AWAY)
    empty = np.zeros((4, 4, 3), dtype=np.uint8)  # no face in frame
    first = await fx.pipeline.process(empty)
    assert first.person_detected is False  # 2-consecutive guard
    second = await fx.pipeline.process(empty)
    assert second.person_detected is True
    events = [e for e in fx.store.recent_events() if e.kind == "person_detected"]
    assert len(events) == 1
    assert "confidence=0.90" in (events[0].detail or "")
    assert any("İnsan algılandı" in subject for subject, _, _ in fx.notifier.sent)


async def test_person_detection_is_silent_at_home(tmp_path: Path) -> None:
    fx = _Fixture(tmp_path, detector=FakeObjectDetector([[_person()]]))
    assert fx.presence.mode == MODE_HOME
    empty = np.zeros((4, 4, 3), dtype=np.uint8)
    await run([empty, empty, empty], fx.pipeline)
    assert [e for e in fx.store.recent_events() if e.kind == "person_detected"] == []
    assert fx.notifier.sent == []
    assert list((tmp_path / "snapshots").rglob("*.jpg")) == []


async def test_person_seen_at_home_can_trigger_after_arming_away(tmp_path: Path) -> None:
    fx = _Fixture(tmp_path, detector=FakeObjectDetector([[_person()]]))
    empty = np.zeros((4, 4, 3), dtype=np.uint8)
    await run([empty, empty, empty], fx.pipeline)
    assert list((tmp_path / "snapshots").rglob("*.jpg")) == []
    fx.presence.set_mode(MODE_AWAY)
    assert not (await fx.pipeline.process(empty)).person_detected
    assert (await fx.pipeline.process(empty)).person_detected
    assert len([e for e in fx.store.recent_events() if e.kind == "person_detected"]) == 1


async def test_person_alarm_suppressed_when_a_known_face_matches(tmp_path: Path) -> None:
    fx = _Fixture(tmp_path, detector=FakeObjectDetector([[_person()]]))
    fx.enroll("Ali", 30)
    fx.presence.set_mode(MODE_AWAY)
    outcome = None
    for _ in range(2):
        outcome = await fx.pipeline.process(_frame(30))
    assert outcome is not None and outcome.person_detected is False
    kinds = {e.kind for e in fx.store.recent_events()}
    assert "known_person" in kinds and "person_detected" not in kinds


async def test_cat_logs_activity_without_mail_by_default(tmp_path: Path) -> None:
    fx = _Fixture(tmp_path, detector=FakeObjectDetector([[_cat()]]))
    empty = np.zeros((4, 4, 3), dtype=np.uint8)
    await run([empty, empty], fx.pipeline)
    events = [e for e in fx.store.recent_events() if e.kind == "cat_activity"]
    assert len(events) == 1
    assert fx.notifier.sent == []  # notify_on_cat defaults to off


# --- event clips wired through the pipeline ---


class _FakeWriter:
    def __init__(self) -> None:
        self.calls: list[list[tuple[float, bytes]]] = []

    def write(self, frames: list[tuple[float, bytes]], ts: float) -> str | None:
        self.calls.append(list(frames))
        return "clips/test.mp4"


async def test_unknown_event_produces_a_clip_with_pre_and_post_roll(tmp_path: Path) -> None:
    writer = _FakeWriter()
    store = HomeStore(tmp_path)
    recorder = ClipRecorder(
        writer=writer,
        attach=store.set_event_clip,
        pre_seconds=30.0,
        post_seconds=30.0,
        max_seconds=300.0,
    )
    fx = _Fixture(
        tmp_path,
        store=store,
        recorder=recorder,
        clock=iter([0.0, 10.0, 20.0, 60.0]).__next__,
    )
    empty = np.zeros((4, 4, 3), dtype=np.uint8)
    await fx.pipeline.process(_frame(20))  # t=0: unknown face seen (1/2)
    await fx.pipeline.process(_frame(20))  # t=10: confirmed -> clip starts
    assert recorder.recording
    await fx.pipeline.process(empty)  # t=20: face gone, post-roll running
    await fx.pipeline.process(empty)  # t=60: 50 s silent > 30 s -> finalized
    assert not recorder.recording
    (frames,) = writer.calls
    assert [ts for ts, _ in frames] == [0.0, 10.0, 20.0, 60.0]  # pre-roll included
    event = next(e for e in fx.store.recent_events() if e.kind == "unknown_person")
    assert event.clip == "clips/test.mp4"
