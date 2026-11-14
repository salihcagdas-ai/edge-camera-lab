"""Clip recorder state-machine tests (pure logic — fake writer, manual clock).

The agreed contract: pre_seconds of pre-roll, no countdown while a subject is
still seen, post_seconds after the last sighting, max_seconds hard cap, and
one continuous incident = one clip with every trigger event attached.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from edge.clips import ClipRecorder, Mp4ClipWriter


class FakeWriter:
    def __init__(self) -> None:
        self.calls: list[list[tuple[float, bytes]]] = []

    def write(self, frames: list[tuple[float, bytes]], ts: float) -> str | None:
        self.calls.append(list(frames))
        return f"clips/fake_{len(self.calls)}.mp4"


def _recorder(**kwargs: float) -> tuple[ClipRecorder, FakeWriter, list[tuple[int, str]]]:
    writer = FakeWriter()
    attached: list[tuple[int, str]] = []
    recorder = ClipRecorder(
        writer=writer,
        attach=lambda event_id, path: attached.append((event_id, path)),
        pre_seconds=kwargs.get("pre_seconds", 30.0),
        post_seconds=kwargs.get("post_seconds", 30.0),
        max_seconds=kwargs.get("max_seconds", 300.0),
    )
    return recorder, writer, attached


def test_pre_roll_keeps_only_the_last_pre_seconds() -> None:
    recorder, writer, _ = _recorder()
    for t in range(0, 101):  # 100 s of idle footage, 1 fps
        recorder.tick(float(t), f"f{t}".encode(), subjects_seen=False)
    recorder.tick(101.0, b"f101", subjects_seen=True, trigger_event_ids=[1])
    assert recorder.recording
    # Recording starts from the buffered pre-roll: nothing older than t-30.
    recorder.tick(140.0, b"end", subjects_seen=False)  # 39 s silent -> finalize
    assert not recorder.recording
    (frames,) = writer.calls
    timestamps = [ts for ts, _ in frames]
    assert min(timestamps) >= 71.0
    assert 101.0 in timestamps and 140.0 in timestamps


def test_post_roll_waits_while_subject_stays_in_view() -> None:
    recorder, writer, attached = _recorder()
    recorder.tick(0.0, b"a", subjects_seen=True, trigger_event_ids=[7])
    # Subject remains in view far past post_seconds: the timer must not run.
    for t in range(1, 121):
        recorder.tick(float(t), b"b", subjects_seen=True)
    assert recorder.recording
    # Subject leaves; 30 s of empty frames then finalize.
    for t in range(121, 152):
        recorder.tick(float(t), b"c", subjects_seen=False)
    assert not recorder.recording
    assert len(writer.calls) == 1
    assert attached == [(7, "clips/fake_1.mp4")]


def test_second_trigger_merges_into_the_open_clip() -> None:
    recorder, writer, attached = _recorder()
    recorder.tick(0.0, b"a", subjects_seen=True, trigger_event_ids=[1])
    recorder.tick(5.0, b"b", subjects_seen=True, trigger_event_ids=[2])
    recorder.tick(40.0, b"c", subjects_seen=False)  # 35 s after last sighting
    assert len(writer.calls) == 1  # one incident, one video
    assert [event_id for event_id, _ in attached] == [1, 2]


def test_max_seconds_caps_a_parked_subject() -> None:
    recorder, writer, _ = _recorder(max_seconds=60.0)
    recorder.tick(0.0, b"a", subjects_seen=True, trigger_event_ids=[1])
    for t in range(1, 61):  # subject never leaves
        recorder.tick(float(t), b"b", subjects_seen=True)
    assert not recorder.recording  # cap reached
    assert len(writer.calls) == 1


def test_none_frames_are_skipped_and_writer_failure_does_not_wedge() -> None:
    class ExplodingWriter:
        def write(self, frames: list[tuple[float, bytes]], ts: float) -> str | None:
            raise OSError("disk full")

    attached: list[tuple[int, str]] = []
    recorder = ClipRecorder(
        writer=ExplodingWriter(),
        attach=lambda event_id, path: attached.append((event_id, path)),
        pre_seconds=30.0,
        post_seconds=10.0,
        max_seconds=300.0,
    )
    recorder.tick(0.0, None, subjects_seen=False)  # encode failed: no frame buffered
    recorder.tick(1.0, b"a", subjects_seen=True, trigger_event_ids=[3])
    recorder.tick(20.0, b"b", subjects_seen=False)  # finalize -> writer explodes
    assert not recorder.recording  # recorder recovered
    assert attached == []  # nothing attached on failure
    # And it can record again afterwards.
    recorder.tick(30.0, b"c", subjects_seen=True, trigger_event_ids=[4])
    assert recorder.recording


def test_clip_path_attaches_to_store_event(tmp_path: Path) -> None:
    from edge.store import HomeStore

    store = HomeStore(tmp_path)
    event_id = store.log_event("unknown_person", ts=100.0)
    recorder = ClipRecorder(
        writer=FakeWriter(),
        attach=store.set_event_clip,  # the real wiring used by __main__
        pre_seconds=30.0,
        post_seconds=5.0,
        max_seconds=300.0,
    )
    recorder.tick(100.0, b"a", subjects_seen=True, trigger_event_ids=[event_id])
    recorder.tick(110.0, b"b", subjects_seen=False)
    (event,) = store.recent_events()
    assert event.clip == "clips/fake_1.mp4"
    store.close()


def test_shutdown_flushes_pending_clip_once() -> None:
    recorder, writer, attached = _recorder()
    recorder.tick(0, b"frame", subjects_seen=True, trigger_event_ids=[1])
    recorder.flush(1)
    recorder.flush(2)
    assert not recorder.recording
    assert len(writer.calls) == 1 and len(attached) == 1


@pytest.mark.parametrize(
    "kwargs", [{"pre_seconds": -1}, {"post_seconds": float("nan")}, {"max_seconds": 0}]
)
def test_invalid_clip_durations_are_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        _recorder(**kwargs)


def test_closed_codec_never_attaches_a_successful_clip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cv2 = pytest.importorskip("cv2")
    ok, jpeg = cv2.imencode(".jpg", np.zeros((32, 32, 3), dtype=np.uint8))
    assert ok
    released = []

    class ClosedWriter:
        fourcc = staticmethod(lambda *_: 0)

        def __init__(self, *_: object) -> None:
            pass

        def isOpened(self) -> bool:
            return False

        def release(self) -> None:
            released.append(True)

        def write(self, *_: object) -> None:
            pytest.fail("must never write to an unopened codec")

    monkeypatch.setattr(cv2, "VideoWriter", ClosedWriter)
    with pytest.raises(RuntimeError, match="encoder is unavailable"):
        Mp4ClipWriter(tmp_path).write([(0, jpeg.tobytes())], 0)
    assert released == [True]
    assert not list(tmp_path.rglob("*.mp4"))


def test_real_mp4_roundtrip_uses_only_synthetic_frames(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    frames = []
    for index in range(3):
        ok, data = cv2.imencode(".jpg", np.full((32, 32, 3), index * 60, dtype=np.uint8))
        assert ok
        frames.append((float(index), data.tobytes()))
    relative = Mp4ClipWriter(tmp_path).write(frames, 0)
    assert relative is not None
    cap = cv2.VideoCapture(str(tmp_path / relative))
    count = 0
    try:
        while cap.read()[0]:
            count += 1
    finally:
        cap.release()
    assert count == 3


def test_corrupt_jpeg_cannot_leave_a_published_partial_clip(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    with pytest.raises(ValueError, match="undecodable"):
        Mp4ClipWriter(tmp_path).write([(0, b"not a JPEG")], 0)
    assert not list(tmp_path.rglob("*.mp4"))
