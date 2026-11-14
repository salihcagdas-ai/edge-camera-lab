"""Recognition pipeline: frame -> motion gate -> embed all faces (+ optional
object detection) -> open-set match -> N-consecutive-frame confirmation ->
events + snapshot + clip + presence + notifications.

All collaborators are injected (fakes in tests), so the decision logic is
verifiable without a camera, a model, or an SMTP server. Blocking inference is
pushed off the event loop with ``asyncio.to_thread``.

Object detection policy: a confirmed **person** detection only alarms while the mode is
"away" AND no known face matched the same frame (a person without a
recognizable face is exactly the case face recognition cannot cover). A
confirmed **cat** is logged as activity in any mode; mail is opt-in.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import numpy as np

from edge.clips import ClipRecorder
from edge.debounce import Cooldown, MultiTrackConfirmer
from edge.detect import LABEL_CAT, LABEL_PERSON, Detection, ObjectDetector
from edge.embedder import FaceEmbedder
from edge.matcher import Match, OpenSetMatcher
from edge.motion import MotionGate
from edge.notify import Notifier
from edge.presence import PresenceState
from edge.store import HomeStore

UNKNOWN_KEY = "unknown"
PERSON_KEY = "object:person"
CAT_KEY = "object:cat"


@dataclass(slots=True)
class FrameOutcome:
    """What one processed frame produced (empty for gated/idle frames)."""

    known: list[Match] = field(default_factory=list)
    unknown_confirmed: bool = False
    person_detected: bool = False  # confirmed person (no known face) while away
    cat_detected: bool = False


class RecognitionPipeline:
    def __init__(
        self,
        *,
        embedder: FaceEmbedder,
        matcher: Callable[[], OpenSetMatcher],
        confirmer: MultiTrackConfirmer,
        motion: MotionGate,
        presence: PresenceState,
        store: HomeStore,
        notifier: Notifier,
        notify_cooldown: Cooldown,
        encode_jpeg: Callable[[np.ndarray], bytes | None],
        notify_on_known: bool = True,
        notify_on_owner: bool = True,
        detector: ObjectDetector | None = None,
        notify_on_person: bool = True,
        notify_on_cat: bool = False,
        recorder: ClipRecorder | None = None,
        clip_on_known: bool = True,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._embedder = embedder
        self._matcher = matcher  # callable: enrollment swaps the matcher at runtime
        self._confirmer = confirmer
        self._motion = motion
        self._presence = presence
        self._store = store
        self._notifier = notifier
        self._cooldown = notify_cooldown
        self._encode = encode_jpeg
        self._notify_on_known = notify_on_known
        self._notify_on_owner = notify_on_owner
        self._detector = detector
        self._notify_on_person = notify_on_person
        self._notify_on_cat = notify_on_cat
        self._recorder = recorder
        self._clip_on_known = clip_on_known
        self._clock = clock

    async def process(self, frame: np.ndarray) -> FrameOutcome:
        now = self._clock()
        motion_active = self._motion.active(frame, now)

        faces: list[tuple[list[float], float]] = []
        detections: list[Detection] = []
        if motion_active:
            faces = await asyncio.to_thread(self._embedder.embed_all, frame)
            if self._detector is not None:
                detections = await asyncio.to_thread(self._detector.detect, frame)

        matches: dict[str, Match] = {}
        unknown_seen = False
        matcher = self._matcher()
        for embedding, _score in faces:
            match = matcher.match(embedding)
            if match is not None:
                matches[f"person:{match.person_id}"] = match
            else:
                unknown_seen = True
        persons = [d for d in detections if d.label == LABEL_PERSON]
        cats = [d for d in detections if d.label == LABEL_CAT]

        present = set(matches)
        if unknown_seen:
            present.add(UNKNOWN_KEY)
        # Only arm this event while its policy is eligible. Otherwise a person
        # first seen at home would consume confirmation before switching away.
        if persons and self._presence.away and not matches:
            present.add(PERSON_KEY)
        if cats:
            present.add(CAT_KEY)
        # A gated interval breaks consecutive evidence and re-arms appearances.
        confirmed = self._confirmer.observe(present if motion_active else set())

        outcome = FrameOutcome()
        trigger_ids: list[int] = []
        jpeg: bytes | None = None
        if confirmed:
            jpeg = self._encode(frame)
            snapshot = self._store.save_snapshot(jpeg, ts=now) if jpeg is not None else None
            outcome.known = [matches[key] for key in confirmed if key in matches]
            outcome.unknown_confirmed = UNKNOWN_KEY in confirmed
            for match in outcome.known:
                event_id = await self._handle_known(match, snapshot, jpeg, now)
                if self._clip_on_known:
                    trigger_ids.append(event_id)
            if outcome.unknown_confirmed:
                trigger_ids.append(await self._handle_unknown(snapshot, jpeg, now))
            if PERSON_KEY in confirmed and self._presence.away and not matches:
                outcome.person_detected = True
                trigger_ids.append(await self._handle_person(persons, snapshot, jpeg, now))
            if CAT_KEY in confirmed:
                outcome.cat_detected = True
                trigger_ids.append(await self._handle_cat(cats, snapshot, jpeg, now))

        if self._recorder is not None:
            if jpeg is None:
                jpeg = self._encode(frame)
            await asyncio.to_thread(
                self._recorder.tick,
                now,
                jpeg,
                subjects_seen=bool(faces or detections),
                trigger_event_ids=trigger_ids or None,
            )
        return outcome

    async def _handle_known(
        self, match: Match, snapshot: str | None, jpeg: bytes | None, now: float
    ) -> int:
        was_away = self._presence.away
        event_id = self._store.log_event(
            "known_person",
            person_name=match.name,
            detail=f"distance={match.distance:.3f}",
            snapshot=snapshot,
            ts=now,
        )
        if match.is_owner:
            self._presence.on_owner_seen()
            if self._notify_on_owner and self._cooldown.ready(f"person:{match.person_id}", now):
                await self._notifier.send(
                    f"Edge Camera Lab: {match.name} eve geldi",
                    f"{match.name} kamerada tanındı (mesafe {match.distance:.2f}). "
                    "Sistem 'evde' moduna geçti.",
                    jpeg,
                )
        elif self._notify_on_known and self._cooldown.ready(f"person:{match.person_id}", now):
            prefix = "Sen dışarıdayken " if was_away else ""
            await self._notifier.send(
                f"Edge Camera Lab: {match.name} kapıda",
                f"{prefix}{match.name} kamerada tanındı (mesafe {match.distance:.2f}).",
                jpeg,
            )
        return event_id

    async def _handle_unknown(self, snapshot: str | None, jpeg: bytes | None, now: float) -> int:
        event_id = self._store.log_event("unknown_person", snapshot=snapshot, ts=now)
        if self._cooldown.ready(UNKNOWN_KEY, now):
            away = " Şu an 'dışarıda' modundasın!" if self._presence.away else ""
            await self._notifier.send(
                "Edge Camera Lab ALARM: Tanınmayan kişi",
                f"Kamerada tanınmayan bir yüz doğrulandı.{away} Fotoğraf ekte.",
                jpeg,
            )
        return event_id

    async def _handle_person(
        self, persons: list[Detection], snapshot: str | None, jpeg: bytes | None, now: float
    ) -> int:
        top = max(d.confidence for d in persons)
        event_id = self._store.log_event(
            "person_detected",
            detail=f"count={len(persons)} confidence={top:.2f}",
            snapshot=snapshot,
            ts=now,
        )
        if self._notify_on_person and self._cooldown.ready(PERSON_KEY, now):
            await self._notifier.send(
                "Edge Camera Lab ALARM: İnsan algılandı (yüz tanınamadı)",
                "Sen dışarıdayken kamerada insan tespit edildi ama tanınabilir bir "
                f"yüz görülemedi (algı güveni {top:.2f}). Fotoğraf ekte.",
                jpeg,
            )
        return event_id

    async def _handle_cat(
        self, cats: list[Detection], snapshot: str | None, jpeg: bytes | None, now: float
    ) -> int:
        top = max(d.confidence for d in cats)
        event_id = self._store.log_event(
            "cat_activity",
            detail=f"confidence={top:.2f}",
            snapshot=snapshot,
            ts=now,
        )
        if self._notify_on_cat and self._cooldown.ready(CAT_KEY, now):
            await self._notifier.send(
                "Edge Camera Lab: Kedi hareketi",
                f"Kedi kamerada görüldü (algı güveni {top:.2f}). Fotoğraf ekte.",
                jpeg,
            )
        return event_id


async def run(frames: Iterable[np.ndarray], pipeline: RecognitionPipeline) -> None:
    """Convenience driver for a finite frame source (tests/dev)."""
    for frame in frames:
        await pipeline.process(frame)
