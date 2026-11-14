"""Exercise the event pipeline with generated pixels and a fake face embedder.

This does not open a camera, load model weights, or send notifications.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path

import numpy as np

from edge.debounce import Cooldown, MultiTrackConfirmer
from edge.embedder import FakeEmbedder
from edge.jpeg import encode_jpeg
from edge.matcher import OpenSetMatcher
from edge.motion import MotionGate
from edge.notify import NullNotifier
from edge.pipeline import RecognitionPipeline, run
from edge.presence import PresenceState
from edge.store import HomeStore


def sample_frame(variant: int) -> np.ndarray:
    """Two abstract scenes; no faces, photographs, or real identity data."""
    import cv2

    canvas = np.full((360, 640, 3), (30, 24, 18), dtype=np.uint8)
    color = (180, 200, 50) if variant == 1 else (75, 140, 240)
    cv2.rectangle(canvas, (65, 80), (265, 280), color, thickness=-1)
    cv2.circle(canvas, (445, 180), 85, color, thickness=6)
    cv2.putText(canvas, f"GENERATED SCENE {variant}", (25, 40), 0, 0.8, (240, 240, 240), 2)
    cv2.putText(
        canvas, "Fake embeddings / no face inference", (25, 330), 0, 0.65, (240, 240, 240), 1
    )
    return canvas


async def run_demo(output: Path) -> dict:
    import cv2

    # Never mix demo data with an existing gallery or overwrite earlier output.
    await asyncio.to_thread(output.mkdir, parents=True, exist_ok=False)
    store = HomeStore(output / "data")
    try:
        known, unknown = sample_frame(1), sample_frame(2)
        embedder = FakeEmbedder()
        person_id = store.add_person("Demo person")
        store.add_embedding(person_id, embedder.embed_all(known)[0][0])
        matcher = OpenSetMatcher(store.gallery_entries(), max_distance=0.55)
        pipeline = RecognitionPipeline(
            embedder=embedder,
            matcher=lambda: matcher,
            confirmer=MultiTrackConfirmer(required=2),
            motion=MotionGate(enabled=False),
            presence=PresenceState(store),
            store=store,
            notifier=NullNotifier(),
            notify_cooldown=Cooldown(seconds=600),
            encode_jpeg=encode_jpeg,
        )
        blank = np.zeros_like(known)
        await run([known, known, blank, blank, unknown, unknown], pipeline)
        counts = dict(Counter(event.kind for event in store.recent_events()))
        expected = {"known_person": 1, "unknown_person": 1}
        if counts != expected:
            raise RuntimeError(f"Unexpected synthetic event counts: {counts}")

        preview = np.concatenate([known, unknown], axis=1)
        if not cv2.imwrite(str(output / "synthetic-preview.png"), preview):
            raise RuntimeError("Could not save the synthetic preview")
        summary = {
            "demo": "synthetic workflow test",
            "model_inference": False,
            "camera_opened": False,
            "notifications_sent": False,
            "frames_processed": 6,
            "event_counts": counts,
            "expected_event_counts": expected,
            "note": "Generated pixels and deterministic fake embeddings; no accuracy benchmark.",
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return summary
    finally:
        store.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/demo"))
    args = parser.parse_args(argv)
    try:
        summary = asyncio.run(run_demo(args.output))
    except FileExistsError:
        parser.error("output already exists; choose a fresh --output directory")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
