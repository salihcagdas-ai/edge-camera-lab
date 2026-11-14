"""Event clips with a pre-event buffer and presence-extended post-roll.

Analysis-frame JPEGs are buffered in memory. Recording starts on a confirmed
trigger, continues while subjects are seen, and ends after a post-roll or
maximum active duration. A writer saves the clip and links its trigger events.
This path is not a full-rate camera recorder.
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)


class ClipWriter(Protocol):
    def write(self, frames: list[tuple[float, bytes]], ts: float) -> str | None:
        """Persist ``(timestamp, jpeg)`` frames; return a data_dir-relative path."""
        ...


class Mp4ClipWriter:
    """Decode buffered JPEGs and write an MP4 (OpenCV, lazily imported)."""

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = Path(data_dir)
        self._clip_dir = self._data_dir / "clips"

    def write(self, frames: list[tuple[float, bytes]], ts: float) -> str | None:
        import cv2  # lazy: heavy, not needed on the fake/test path
        import numpy as np

        if not frames:
            return None
        duration = frames[-1][0] - frames[0][0]
        fps = min(30.0, max(1.0, (len(frames) - 1) / duration)) if duration > 0 else 3.0

        day = time.strftime("%Y%m%d", time.localtime(ts))
        directory = self._clip_dir / day
        directory.mkdir(parents=True, exist_ok=True)
        name = f"{time.strftime('%H%M%S', time.localtime(ts))}_{uuid.uuid4().hex[:8]}.mp4"
        path = directory / name

        writer = None
        written = 0
        try:
            for _frame_ts, jpeg in frames:
                image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("Clip contains an undecodable JPEG frame")
                if writer is None:
                    height, width = image.shape[:2]
                    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
                    if not writer.isOpened():
                        raise RuntimeError("MP4 encoder is unavailable")
                elif image.shape[:2] != (height, width):
                    raise ValueError("Clip frame dimensions changed")
                writer.write(image)
                written += 1
        except Exception:
            if writer is not None:
                writer.release()
                writer = None
            path.unlink(missing_ok=True)  # only this newly generated, incomplete clip
            raise
        finally:
            if writer is not None:
                writer.release()
        if written == 0:
            return None
        if not path.is_file() or path.stat().st_size == 0:
            path.unlink(missing_ok=True)
            raise RuntimeError("MP4 encoder did not produce a nonempty file")
        return path.relative_to(self._data_dir).as_posix()


class ClipRecorder:
    """Pure clip state machine — fed one analysis frame per tick.

    Fully unit-testable: the writer, the event-attach callback and time all
    come from outside. A trigger while a clip is already open merges into it
    (one continuous incident = one video, all its events linked to it).
    """

    def __init__(
        self,
        *,
        writer: ClipWriter,
        attach: Callable[[int, str], None],
        pre_seconds: float = 30.0,
        post_seconds: float = 30.0,
        max_seconds: float = 300.0,
    ) -> None:
        if (
            any(not math.isfinite(value) for value in (pre_seconds, post_seconds, max_seconds))
            or pre_seconds < 0
            or post_seconds < 0
            or max_seconds <= 0
        ):
            raise ValueError("Clip durations must be finite; pre/post nonnegative and max positive")
        self._writer = writer
        self._attach = attach
        self._pre = pre_seconds
        self._post = post_seconds
        self._max = max_seconds
        self._buffer: deque[tuple[float, bytes]] = deque()
        self._recording = False
        self._frames: list[tuple[float, bytes]] = []
        self._event_ids: list[int] = []
        self._started = 0.0
        self._last_seen = 0.0

    @property
    def recording(self) -> bool:
        return self._recording

    def tick(
        self,
        now: float,
        jpeg: bytes | None,
        *,
        subjects_seen: bool,
        trigger_event_ids: list[int] | None = None,
    ) -> None:
        if not math.isfinite(now):
            raise ValueError("Clip timestamp must be finite")
        # Ring buffer runs unconditionally so the NEXT clip has its pre-roll.
        if jpeg is not None:
            self._buffer.append((now, jpeg))
        while self._buffer and self._buffer[0][0] < now - self._pre:
            self._buffer.popleft()

        if not self._recording:
            if trigger_event_ids:
                self._frames = list(self._buffer)  # pre-roll (includes this frame)
                self._event_ids = list(trigger_event_ids)
                self._started = now
                self._last_seen = now
                self._recording = True
            return

        if jpeg is not None:
            self._frames.append((now, jpeg))
        if trigger_event_ids:
            self._event_ids.extend(i for i in trigger_event_ids if i not in self._event_ids)
            self._last_seen = now
        if subjects_seen:
            self._last_seen = now  # subject still in view: the post-roll timer waits

        if (now - self._last_seen) > self._post or (now - self._started) >= self._max:
            self._finalize(now)

    def flush(self, now: float | None = None) -> None:
        """Finalize the current clip before shutdown; call after the analysis loop stops."""
        if self._recording:
            self._finalize(time.time() if now is None else now)

    def _finalize(self, now: float) -> None:
        frames, event_ids = self._frames, self._event_ids
        self._recording = False
        self._frames = []
        self._event_ids = []
        try:
            path = self._writer.write(frames, now) if frames else None
        except Exception:  # a broken write must not wedge the recorder
            logger.exception("clip write failed (%d frames dropped)", len(frames))
            return
        if path is None:
            return
        logger.info("clip saved: %s (%d frames, events %s)", path, len(frames), event_ids)
        for event_id in event_ids:
            self._attach(event_id, path)
