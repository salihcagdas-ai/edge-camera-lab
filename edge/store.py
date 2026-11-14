"""Local SQLite storage for a small gallery, events and settings.

A WAL database stores face embeddings; snapshots and clips are separate files.
SQLite access is serialized between the analysis loop and web thread pool.
"""

from __future__ import annotations

import contextlib
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_SCHEMA = """
CREATE TABLE IF NOT EXISTS persons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    is_owner INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    vector BLOB NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    kind TEXT NOT NULL,              -- known_person | unknown_person | mode_change
                                     -- | person_detected | cat_activity
    person_name TEXT,
    detail TEXT,
    snapshot TEXT,                   -- path relative to data_dir, nullable
    clip TEXT                        -- event video, relative to data_dir, nullable
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts DESC);
CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _restrict(path: Path) -> None:
    """chmod to owner-only (0700 dir / 0600 file). No-op where unsupported."""
    with contextlib.suppress(OSError):
        os.chmod(path, 0o700 if path.is_dir() else 0o600)


@dataclass(slots=True)
class GalleryEntry:
    person_id: int
    name: str
    is_owner: bool
    vector: np.ndarray  # float32, unit-norm, shape (512,)


@dataclass(slots=True)
class Event:
    id: int
    ts: float
    kind: str
    person_name: str | None
    detail: str | None
    snapshot: str | None
    clip: str | None = None


class HomeStore:
    def __init__(self, data_dir: Path) -> None:
        self._data_dir = Path(data_dir)
        self._snapshot_dir = self._data_dir / "snapshots"
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)
        (self._data_dir / "clips").mkdir(parents=True, exist_ok=True)
        # Owner-only perms: footage + biometric DB must not be readable by a
        # second local user/service. Best-effort (no-op on Windows dev). The
        # 0700 dirs also gate the JPEG/MP4 files written inside them.
        _restrict(self._data_dir)
        _restrict(self._snapshot_dir)
        _restrict(self._data_dir / "clips")
        self._lock = threading.Lock()
        db_path = self._data_dir / "camera.db"
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        _restrict(db_path)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(_SCHEMA)
            # Migration for databases created before the clip column existed.
            columns = {r["name"] for r in self._conn.execute("PRAGMA table_info(events)")}
            if "clip" not in columns:
                self._conn.execute("ALTER TABLE events ADD COLUMN clip TEXT")
            self._conn.commit()

    # --- people / gallery ---

    def add_person(self, name: str, *, is_owner: bool = False) -> int:
        with self._lock:
            row = self._conn.execute("SELECT id FROM persons WHERE name = ?", (name,)).fetchone()
            if row is not None:
                return int(row["id"])
            cursor = self._conn.execute(
                "INSERT INTO persons (name, is_owner, created_at) VALUES (?, ?, ?)",
                (name, int(is_owner), time.time()),
            )
            self._conn.commit()
            return int(cursor.lastrowid or 0)

    def add_embedding(self, person_id: int, vector: list[float] | np.ndarray) -> None:
        arr = _unit_embedding(vector)
        with self._lock:
            self._conn.execute(
                "INSERT INTO embeddings (person_id, vector, created_at) VALUES (?, ?, ?)",
                (person_id, arr.tobytes(), time.time()),
            )
            self._conn.commit()

    def remove_person(self, person_id: int) -> bool:
        with self._lock:
            cursor = self._conn.execute("DELETE FROM persons WHERE id = ?", (person_id,))
            self._conn.commit()
            return cursor.rowcount > 0

    def people(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT p.id, p.name, p.is_owner, COUNT(e.id) AS embeddings
                FROM persons p LEFT JOIN embeddings e ON e.person_id = p.id
                GROUP BY p.id ORDER BY p.name
                """
            ).fetchall()
        return [
            {
                "id": int(r["id"]),
                "name": r["name"],
                "is_owner": bool(r["is_owner"]),
                "embeddings": int(r["embeddings"]),
            }
            for r in rows
        ]

    def gallery_entries(self) -> list[GalleryEntry]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT e.person_id, p.name, p.is_owner, e.vector
                FROM embeddings e JOIN persons p ON p.id = e.person_id
                """
            ).fetchall()
        return [
            GalleryEntry(
                person_id=int(r["person_id"]),
                name=r["name"],
                is_owner=bool(r["is_owner"]),
                vector=_unit_embedding(np.frombuffer(r["vector"], dtype=np.float32)),
            )
            for r in rows
        ]

    # --- events + snapshots ---

    def save_snapshot(self, jpeg: bytes, ts: float | None = None) -> str:
        """Write an event snapshot; return its path relative to data_dir."""
        ts = ts if ts is not None else time.time()
        day = time.strftime("%Y%m%d", time.localtime(ts))
        directory = self._snapshot_dir / day
        directory.mkdir(parents=True, exist_ok=True)
        name = f"{time.strftime('%H%M%S', time.localtime(ts))}_{uuid.uuid4().hex[:8]}.jpg"
        path = directory / name
        path.write_bytes(jpeg)
        # POSIX form so the path is URL-safe in /snapshots/{relative} links.
        return path.relative_to(self._data_dir).as_posix()

    @property
    def data_dir(self) -> Path:
        return self._data_dir

    def _resolve_under(self, subdir: str, relative: str) -> Path | None:
        """Resolve ``relative`` but ONLY if it lands inside ``data_dir/subdir``.

        Scoping to the media subdirectory (not just data_dir) is what stops
        ``/snapshots/camera.db`` or ``/clips/camera.db`` from serving the
        biometric database — every servable file must live under its own
        subdir, never at the data_dir root.
        """
        root = (self._data_dir / subdir).resolve()
        candidate = (self._data_dir / relative).resolve()
        if candidate != root and root not in candidate.parents:
            return None
        return candidate if candidate.is_file() else None

    def snapshot_path(self, relative: str) -> Path | None:
        return self._resolve_under("snapshots", relative)

    def clip_path(self, relative: str) -> Path | None:
        return self._resolve_under("clips", relative)

    def log_event(
        self,
        kind: str,
        *,
        person_name: str | None = None,
        detail: str | None = None,
        snapshot: str | None = None,
        ts: float | None = None,
    ) -> int:
        with self._lock:
            cursor = self._conn.execute(
                "INSERT INTO events (ts, kind, person_name, detail, snapshot) "
                "VALUES (?, ?, ?, ?, ?)",
                (ts if ts is not None else time.time(), kind, person_name, detail, snapshot),
            )
            self._conn.commit()
            return int(cursor.lastrowid or 0)

    def set_event_clip(self, event_id: int, clip: str) -> None:
        """Attach a clip path to an event (clips finalize after their post-roll)."""
        with self._lock:
            self._conn.execute("UPDATE events SET clip = ? WHERE id = ?", (clip, event_id))
            self._conn.commit()

    def purge_older_than(self, cutoff_ts: float) -> int:
        """Delete events (and their unreferenced snapshot/clip files) older than
        ``cutoff_ts``. A media file is unlinked only if no surviving event still
        references it (clips can be shared by a merged incident). Returns the
        number of files removed."""
        with self._lock:
            old = self._conn.execute(
                "SELECT snapshot, clip FROM events WHERE ts < ?", (cutoff_ts,)
            ).fetchall()
            self._conn.execute("DELETE FROM events WHERE ts < ?", (cutoff_ts,))
            self._conn.commit()
            keep: set[str] = set()
            for row in self._conn.execute("SELECT snapshot, clip FROM events"):
                keep.update(v for v in (row["snapshot"], row["clip"]) if v)

        removed = 0
        handled: set[str] = set()
        for row in old:
            for relative in (row["snapshot"], row["clip"]):
                if not relative or relative in keep or relative in handled:
                    continue
                handled.add(relative)
                target = self._resolve_media(relative)
                if target is not None:
                    with contextlib.suppress(OSError):
                        target.unlink()
                        removed += 1
        return removed

    def _resolve_media(self, relative: str) -> Path | None:
        """Traversal-safe resolve of a stored snapshot/clip path for deletion."""
        return self._resolve_under("snapshots", relative) or self._resolve_under("clips", relative)

    def recent_events(self, limit: int = 50) -> list[Event]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM events ORDER BY ts DESC LIMIT ?", (int(limit),)
            ).fetchall()
        return [
            Event(
                id=int(r["id"]),
                ts=float(r["ts"]),
                kind=r["kind"],
                person_name=r["person_name"],
                detail=r["detail"],
                snapshot=r["snapshot"],
                clip=r["clip"],
            )
            for r in rows
        ]

    # --- kv (presence mode, etc.) ---

    def get_setting(self, key: str, default: str) -> str:
        with self._lock:
            row = self._conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return str(row["value"]) if row is not None else default

    def set_setting(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO kv (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _unit_embedding(vector: list[float] | np.ndarray) -> np.ndarray:
    """Reject corrupt biometric vectors before they enter matching or storage."""
    arr = np.asarray(vector, dtype=np.float64)
    if arr.shape != (512,) or not np.isfinite(arr).all():
        raise ValueError("Face embedding must contain exactly 512 finite values")
    # Scale first so valid large finite vectors cannot overflow the norm.
    scale = float(np.max(np.abs(arr)))
    if scale == 0:
        raise ValueError("Face embedding must be nonzero")
    arr = arr / scale
    return (arr / np.linalg.norm(arr)).astype(np.float32)
