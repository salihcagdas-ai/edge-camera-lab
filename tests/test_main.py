"""Shutdown ordering without cameras, sockets, model weights, or SMTP."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from edge import __main__ as service
from edge.config import EdgeSettings
from edge.source import SyntheticSource
from edge.store import HomeStore


async def test_shutdown_finishes_inflight_frame_before_clip_flush_and_store_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame_started = asyncio.Event()
    release_frame = asyncio.Event()
    server_stopped = asyncio.Event()
    sequence: list[str] = []

    class BlockingPipeline:
        def __init__(self, **kwargs: object) -> None:
            pass

        async def process(self, frame: np.ndarray) -> None:
            sequence.append("frame_started")
            frame_started.set()
            await release_frame.wait()
            sequence.append("frame_finished")

    class RecordingStore(HomeStore):
        def close(self) -> None:
            sequence.append("store_closed")
            super().close()

    class RecordingRecorder:
        def flush(self) -> None:
            sequence.append("clip_flushed")

    class FakeServer:
        def __init__(self, config: object) -> None:
            pass

        async def serve(self) -> None:
            await frame_started.wait()
            sequence.append("server_stopped")
            server_stopped.set()

    class ImmediateCapture:
        """Publish one synthetic frame; no background capture thread is needed."""

        def __init__(self, *, target, args, daemon: bool, name: str) -> None:
            self.target = target
            self.args = args

        def start(self) -> None:
            self.target(*self.args)

    settings = EdgeSettings(
        source="synthetic",
        embedder="fake",
        data_dir=tmp_path,
        retention_days=0,
        presence_host="",
        analysis_fps=3,
    )
    monkeypatch.setattr(service, "EdgeSettings", lambda: settings)
    monkeypatch.setattr(service, "HomeStore", RecordingStore)
    monkeypatch.setattr(service, "RecognitionPipeline", BlockingPipeline)
    monkeypatch.setattr(service, "build_recorder", lambda settings, store: RecordingRecorder())
    monkeypatch.setattr(
        service, "build_source", lambda settings: SyntheticSource([np.zeros((4, 4, 3))])
    )
    # Replace the entrypoint's module references, not global threading/uvicorn APIs.
    monkeypatch.setattr(service, "threading", SimpleNamespace(Thread=ImmediateCapture))
    monkeypatch.setattr(
        service, "uvicorn", SimpleNamespace(Config=lambda *args, **kwargs: None, Server=FakeServer)
    )

    task = asyncio.create_task(service._main())
    try:
        await asyncio.wait_for(server_stopped.wait(), timeout=2)
        await asyncio.sleep(0)  # let the entrypoint enter its shutdown finally block
        assert not task.done()
        assert sequence == ["frame_started", "server_stopped"]
    finally:
        release_frame.set()
        await asyncio.wait_for(task, timeout=2)

    assert sequence == [
        "frame_started",
        "server_stopped",
        "frame_finished",
        "clip_flushed",
        "store_closed",
    ]
