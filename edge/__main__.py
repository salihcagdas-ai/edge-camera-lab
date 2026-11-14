"""Service entrypoint: camera thread + analysis loop + web UI in one process.

    python -m edge          # uses EDGE_CAMERA_* env config

The capture thread publishes frames to the FrameHub; the asyncio analysis loop
samples the newest frame at ``analysis_fps``; uvicorn serves the web UI/live
stream from the same loop.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import subprocess
import threading
import time

import uvicorn

from edge import jpeg
from edge.clips import ClipRecorder, Mp4ClipWriter
from edge.config import EdgeSettings
from edge.debounce import Cooldown, MultiTrackConfirmer
from edge.detect import HailoDetector, ObjectDetector
from edge.embedder import FaceEmbedder, FakeEmbedder, InsightFaceEmbedder
from edge.hub import FrameHub
from edge.matcher import MatcherRef, OpenSetMatcher
from edge.motion import MotionGate
from edge.notify import EmailNotifier, Notifier, NullNotifier
from edge.pipeline import RecognitionPipeline
from edge.presence import MODE_AWAY, MODE_HOME, PresenceState
from edge.source import PiCameraSource, RTSPSource, SyntheticSource
from edge.store import HomeStore
from edge.web import AppState, create_app

logger = logging.getLogger("edge")

FrameSource = PiCameraSource | RTSPSource | SyntheticSource


def build_embedder(settings: EdgeSettings) -> FaceEmbedder:
    if settings.embedder == "fake":
        return FakeEmbedder()
    return InsightFaceEmbedder(settings.insightface_model, model_root=settings.insightface_root)


def build_source(settings: EdgeSettings) -> FrameSource:
    if settings.source == "rtsp":
        if not settings.rtsp_url:
            raise RuntimeError("EDGE_CAMERA_RTSP_URL is required for the rtsp source")
        return RTSPSource(settings.rtsp_url, fps=settings.capture_fps)
    if settings.source == "picamera":
        return PiCameraSource(settings.frame_width, settings.frame_height, fps=settings.capture_fps)
    return SyntheticSource([])  # synthetic mode emits nothing by default


def build_detector(settings: EdgeSettings) -> ObjectDetector | None:
    if settings.detector == "none":
        return None
    if settings.detector == "hailo":
        detector = HailoDetector(settings.hef_path, settings.detect_confidence)
        detector.load()  # fail fast: a mis-set .hef must not look like "no detections"
        return detector
    raise RuntimeError(f"unknown detector: {settings.detector!r} (use 'none' or 'hailo')")


def build_recorder(settings: EdgeSettings, store: HomeStore) -> ClipRecorder | None:
    if not settings.clip_enabled:
        return None
    try:
        import cv2  # noqa: F401  # clips need real JPEG/MP4 encoding
    except ImportError:
        logger.warning("event clips disabled: OpenCV not installed (pip install -e '.[cv]')")
        return None
    return ClipRecorder(
        writer=Mp4ClipWriter(store.data_dir),
        attach=store.set_event_clip,
        pre_seconds=settings.clip_pre_seconds,
        post_seconds=settings.clip_post_seconds,
        max_seconds=settings.clip_max_seconds,
    )


def build_notifier(settings: EdgeSettings) -> Notifier:
    if not settings.smtp_host:
        logger.info("SMTP not configured; external notifications are disabled")
        return NullNotifier()
    return EmailNotifier(
        settings.smtp_host,
        settings.smtp_port,
        settings.smtp_user,
        settings.smtp_password,
        settings.mail_from,
        settings.mail_to,
    )


def _capture_loop(source: FrameSource, hub: FrameHub) -> None:
    try:
        for frame in source.frames():
            hub.publish(frame)
    except Exception:
        logger.exception("capture loop crashed")


async def _analysis_loop(
    hub: FrameHub, pipeline: RecognitionPipeline, interval: float, stop: asyncio.Event
) -> None:
    last_seq = -1
    while not stop.is_set():
        frame, seq = hub.latest()
        if frame is not None and seq != last_seq:
            last_seq = seq
            try:
                await pipeline.process(frame)
            except Exception:
                logger.exception("frame processing failed")
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except TimeoutError:
            pass


async def _retention_loop(store: HomeStore, days: int, interval_s: float = 6 * 3600) -> None:
    """Periodically delete footage older than ``days`` (data minimization)."""
    while True:
        cutoff = time.time() - days * 86400
        try:
            removed = await asyncio.to_thread(store.purge_older_than, cutoff)
            if removed:
                logger.info(
                    "retention: purged %d media file(s) older than %d day(s)", removed, days
                )
        except Exception:
            logger.exception("retention purge failed")
        await asyncio.sleep(interval_s)


def _ping(host: str) -> bool:
    """True if ``host`` answers a single ping within 2 s (phone-presence probe)."""
    try:
        result = subprocess.run(
            ["ping", "-c", "1", "-W", "2", host], capture_output=True, check=False, timeout=4
        )
        return result.returncode == 0
    except Exception:
        return False


async def _presence_loop(
    presence: PresenceState, host: str, interval_s: float, absent_after_s: float
) -> None:
    """Auto-away: ping the owner's phone; after a sustained absence flip to
    "away", and back to "home" when it returns — but only revert an away this
    loop itself set (a manual "away" in the UI is left alone)."""
    last_seen = time.time()
    auto_away = False
    while True:
        alive = await asyncio.to_thread(_ping, host)
        now = time.time()
        if alive:
            last_seen = now
            if auto_away and presence.away:
                presence.set_mode(MODE_HOME)
                auto_away = False
                logger.info("presence probe reachable -> home")
        elif not presence.away and now - last_seen >= absent_after_s:
            presence.set_mode(MODE_AWAY)
            auto_away = True
            logger.info("presence probe absent %.0fs -> away", now - last_seen)
        await asyncio.sleep(interval_s)


async def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    settings = EdgeSettings()

    embedder = build_embedder(settings)
    detector = build_detector(settings)
    source = build_source(settings)
    store = HomeStore(settings.data_dir)
    matcher_ref = MatcherRef(OpenSetMatcher(store.gallery_entries(), settings.match_max_distance))
    presence = PresenceState(store)
    notifier = build_notifier(settings)
    hub = FrameHub()
    recorder = build_recorder(settings, store)
    pipeline = RecognitionPipeline(
        embedder=embedder,
        matcher=lambda: matcher_ref.current,
        confirmer=MultiTrackConfirmer(settings.consecutive_frames),
        motion=MotionGate(
            min_change=settings.motion_min_change,
            cooldown_s=settings.motion_cooldown_s,
            enabled=settings.motion_enabled,
        ),
        presence=presence,
        store=store,
        notifier=notifier,
        notify_cooldown=Cooldown(settings.notify_cooldown_s),
        encode_jpeg=jpeg.encode_jpeg,
        notify_on_known=settings.notify_on_known,
        notify_on_owner=settings.notify_on_owner,
        detector=detector,
        notify_on_person=settings.notify_on_person,
        notify_on_cat=settings.notify_on_cat,
        recorder=recorder,
        clip_on_known=settings.clip_on_known,
    )
    logger.info("gallery: %d embedding(s); mode: %s", matcher_ref.current.size, presence.mode)

    capture = threading.Thread(
        target=_capture_loop, args=(source, hub), daemon=True, name="capture"
    )
    capture.start()

    if not settings.web_pin:
        logger.warning(
            "EDGE_CAMERA_WEB_PIN is unset; local web access has no password. "
            "Do not forward this port or proxy it to a network without authentication."
        )
    app = create_app(
        AppState(
            settings=settings,
            store=store,
            presence=presence,
            matcher_ref=matcher_ref,
            embedder=embedder,
            hub=hub,
            decode_image=jpeg.decode_bgr,
            encode_jpeg=jpeg.encode_jpeg,
            web_pin=settings.web_pin,
            session_secret=secrets.token_hex(32),
            cookie_secure=settings.web_cookie_secure,
        )
    )
    server = uvicorn.Server(
        uvicorn.Config(app, host=settings.web_host, port=settings.web_port, log_level="info")
    )
    analysis_stop = asyncio.Event()
    loops = [
        server.serve(),
        _analysis_loop(hub, pipeline, 1.0 / settings.analysis_fps, analysis_stop),
    ]
    if settings.retention_days > 0:
        loops.append(_retention_loop(store, settings.retention_days))
    if settings.presence_host:
        logger.info("optional presence probe enabled")
        loops.append(
            _presence_loop(
                presence,
                settings.presence_host,
                settings.presence_interval_s,
                settings.presence_absent_after_s,
            )
        )
    tasks = [asyncio.create_task(loop) for loop in loops]
    try:
        # Uvicorn returns after a normal shutdown; stop the infinite workers too.
        await tasks[0]
    finally:
        # Let an in-flight frame/clip write finish before flushing or closing SQLite.
        analysis_stop.set()
        await tasks[1]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if recorder is not None:
            await asyncio.to_thread(recorder.flush)
        store.close()


if __name__ == "__main__":
    asyncio.run(_main())
