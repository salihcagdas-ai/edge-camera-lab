"""SQLite store tests (tmp dir, stdlib sqlite3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from edge.store import HomeStore


def test_add_person_is_idempotent_by_name(tmp_path: Path) -> None:
    store = HomeStore(tmp_path)
    first = store.add_person("Example Owner", is_owner=True)
    second = store.add_person("Example Owner")
    assert first == second
    assert len(store.people()) == 1
    store.close()


def test_embeddings_roundtrip_unit_normalized(tmp_path: Path) -> None:
    store = HomeStore(tmp_path)
    person_id = store.add_person("Example Owner", is_owner=True)
    store.add_embedding(person_id, [3.0, 4.0] + [0.0] * 510)
    entries = store.gallery_entries()
    assert len(entries) == 1
    entry = entries[0]
    assert entry.name == "Example Owner"
    assert entry.is_owner is True
    assert entry.vector.dtype == np.float32
    assert abs(float(np.linalg.norm(entry.vector)) - 1.0) < 1e-6
    assert abs(float(entry.vector[0]) - 0.6) < 1e-6  # 3/5 after normalization
    store.close()


def test_remove_person_cascades_embeddings(tmp_path: Path) -> None:
    store = HomeStore(tmp_path)
    person_id = store.add_person("Example Visitor")
    store.add_embedding(person_id, [1.0] * 512)
    store.remove_person(person_id)
    assert store.people() == []
    assert store.gallery_entries() == []
    store.close()


def test_events_are_recent_first_and_limited(tmp_path: Path) -> None:
    store = HomeStore(tmp_path)
    store.log_event("known_person", person_name="Example Owner", ts=100.0)
    store.log_event("unknown_person", ts=200.0)
    store.log_event("mode_change", detail="away", ts=300.0)
    events = store.recent_events(limit=2)
    assert [e.kind for e in events] == ["mode_change", "unknown_person"]
    store.close()


def test_snapshot_save_and_safe_resolution(tmp_path: Path) -> None:
    store = HomeStore(tmp_path)
    relative = store.save_snapshot(b"jpeg-bytes", ts=1_700_000_000.0)
    resolved = store.snapshot_path(relative)
    assert resolved is not None
    assert resolved.read_bytes() == b"jpeg-bytes"
    # Path traversal outside data_dir must not resolve.
    assert store.snapshot_path("../secret.txt") is None
    assert store.snapshot_path("nonexistent/file.jpg") is None
    store.close()


def test_event_clip_attach_roundtrip(tmp_path: Path) -> None:
    store = HomeStore(tmp_path)
    event_id = store.log_event("unknown_person", ts=100.0)
    assert store.recent_events()[0].clip is None
    store.set_event_clip(event_id, "clips/20260702/olay.mp4")
    assert store.recent_events()[0].clip == "clips/20260702/olay.mp4"
    store.close()


def test_clip_column_is_migrated_into_a_pre_2_1_database(tmp_path: Path) -> None:
    import sqlite3

    # Simulate a database created before the clip column existed.
    conn = sqlite3.connect(tmp_path / "camera.db")
    conn.execute(
        "CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, "
        "kind TEXT NOT NULL, person_name TEXT, detail TEXT, snapshot TEXT)"
    )
    conn.execute("INSERT INTO events (ts, kind) VALUES (1.0, 'unknown_person')")
    conn.commit()
    conn.close()

    store = HomeStore(tmp_path)  # opens the old DB -> ALTER TABLE adds clip
    (event,) = store.recent_events()
    assert event.kind == "unknown_person" and event.clip is None
    store.set_event_clip(event.id, "clips/x.mp4")
    assert store.recent_events()[0].clip == "clips/x.mp4"
    store.close()


def test_purge_older_than_deletes_old_events_and_their_files(tmp_path: Path) -> None:
    store = HomeStore(tmp_path)
    old_snap = store.save_snapshot(b"old", ts=100.0)
    new_snap = store.save_snapshot(b"new", ts=1_000_000_000.0)
    store.log_event("unknown_person", snapshot=old_snap, ts=100.0)
    store.log_event("unknown_person", snapshot=new_snap, ts=1_000_000_000.0)

    removed = store.purge_older_than(cutoff_ts=1000.0)
    assert removed == 1
    kinds_ts = [(e.kind, e.ts) for e in store.recent_events()]
    assert kinds_ts == [("unknown_person", 1_000_000_000.0)]  # only the new one survives
    assert store.snapshot_path(old_snap) is None  # old file gone
    assert store.snapshot_path(new_snap) is not None  # new file kept
    store.close()


def test_purge_keeps_a_clip_still_referenced_by_a_surviving_event(tmp_path: Path) -> None:
    # One incident clip shared by two events across the cutoff: it must survive.
    store = HomeStore(tmp_path)
    clip_dir = tmp_path / "clips" / "20260101"
    clip_dir.mkdir(parents=True)
    (clip_dir / "c.mp4").write_bytes(b"clip")
    rel = "clips/20260101/c.mp4"
    e_old = store.log_event("unknown_person", ts=100.0)
    e_new = store.log_event("person_detected", ts=1_000_000_000.0)
    store.set_event_clip(e_old, rel)
    store.set_event_clip(e_new, rel)

    removed = store.purge_older_than(cutoff_ts=1000.0)
    assert removed == 0  # the clip is still referenced by the surviving event
    assert store.clip_path(rel) is not None
    store.close()


def test_media_path_is_scoped_to_its_subdir(tmp_path: Path) -> None:
    store = HomeStore(tmp_path)
    # The DB at the data_dir root must not resolve through the media helpers.
    assert store.snapshot_path("camera.db") is None
    assert store.clip_path("camera.db") is None
    assert store.snapshot_path("../camera.db") is None
    store.close()


def test_kv_settings(tmp_path: Path) -> None:
    store = HomeStore(tmp_path)
    assert store.get_setting("presence_mode", "home") == "home"
    store.set_setting("presence_mode", "away")
    assert store.get_setting("presence_mode", "home") == "away"
    store.set_setting("presence_mode", "home")
    assert store.get_setting("presence_mode", "away") == "home"
    store.close()


@pytest.mark.parametrize(
    "vector",
    [
        [],
        [1.0] * 511,
        [0.0] * 512,
        [float("nan")] + [1.0] * 511,
        [float("inf")] + [1.0] * 511,
        [[1.0] * 512],
    ],
)
def test_invalid_embeddings_never_enter_storage(tmp_path: Path, vector: list) -> None:
    store = HomeStore(tmp_path)
    identifier = store.add_person("Example Owner")
    with pytest.raises(ValueError):
        store.add_embedding(identifier, vector)
    assert store.gallery_entries() == []
    store.close()


def test_large_finite_embedding_normalizes_without_overflow(tmp_path: Path) -> None:
    store = HomeStore(tmp_path)
    identifier = store.add_person("Example Owner")
    store.add_embedding(identifier, [1e300] * 512)
    vector = store.gallery_entries()[0].vector
    assert np.isfinite(vector).all()
    assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-6)
    store.close()


def test_existing_corrupt_gallery_fails_closed(tmp_path: Path) -> None:
    import sqlite3

    store = HomeStore(tmp_path)
    identifier = store.add_person("Example Owner")
    store.close()
    with sqlite3.connect(tmp_path / "camera.db") as conn:
        conn.execute(
            "INSERT INTO embeddings(person_id,vector,created_at) VALUES (?,?,?)",
            (identifier, np.full(512, np.nan, dtype=np.float32).tobytes(), 0),
        )
    store = HomeStore(tmp_path)
    with pytest.raises(ValueError):
        store.gallery_entries()
    store.close()
