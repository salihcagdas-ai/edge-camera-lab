"""Face embedders.

``FaceEmbedder`` is the interface the pipeline depends on: every detected face
in a frame -> (512-d unit embedding, detection score). ``FakeEmbedder`` is a
deterministic, dependency-light implementation for tests/dev.
``InsightFaceEmbedder`` is the real SCRFD + ArcFace path; its heavy deps
(insightface/onnxruntime) are imported lazily. On the Pi 5 we run the light
``buffalo_s`` pack on CPU. Hailo object detection is a separate optional adapter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np


class FaceEmbedder(Protocol):
    def embed_all(self, frame: np.ndarray) -> list[tuple[list[float], float]]:
        """Return ``(embedding, det_score)`` for every detected face."""
        ...


class FakeEmbedder:
    """Deterministic pseudo-embedding from a frame's pixels (no model/GPU).

    An all-zero frame stands in for "no face detected". Identical frames give
    identical embeddings, so matcher/pipeline behavior is fully testable.
    """

    def embed_all(self, frame: np.ndarray) -> list[tuple[list[float], float]]:
        arr = np.asarray(frame, dtype=np.float64)
        if arr.size == 0 or not np.any(arr):
            return []
        seed = int(abs(arr.sum())) % (2**32)
        vec = np.random.default_rng(seed).standard_normal(512)
        unit = vec / np.linalg.norm(vec)
        return [([float(x) for x in unit], 0.99)]

    def embed(self, frame: np.ndarray) -> list[float] | None:
        faces = self.embed_all(frame)
        return faces[0][0] if faces else None


class InsightFaceEmbedder:
    """Real SCRFD (detect/align) + ArcFace (512-d) via insightface."""

    def __init__(
        self,
        model_name: str = "buffalo_s",
        providers: list[str] | None = None,
        ctx_id: int = -1,
        model_root: Path = Path("models/insightface"),
    ) -> None:
        # Fail before importing the optional runtime: never bootstrap model downloads.
        # InsightFace 0.7.x reuses root/models/name when that directory exists.
        if not model_name or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for c in model_name
        ):
            raise ValueError("model_name must be a simple local model-pack name")
        model_dir = model_root.resolve() / "models" / model_name
        if not model_dir.is_dir() or not any(model_dir.glob("*.onnx")):
            raise FileNotFoundError(
                "Provision the licensed ONNX model pack under "
                "EDGE_CAMERA_INSIGHTFACE_ROOT/models/<model-name> before starting. "
                "Automatic downloads are disabled by this adapter."
            )
        import onnxruntime
        from insightface.app import FaceAnalysis  # lazy: heavy + optional

        onnxruntime.disable_telemetry_events()

        self._app = FaceAnalysis(
            name=model_name,
            root=str(model_root.resolve()),
            allowed_modules=["detection", "recognition"],
            providers=providers or ["CPUExecutionProvider"],
        )
        # ctx_id=-1 -> CPU (the Pi 5 path); >=0 -> GPU index on a dev box.
        self._app.prepare(ctx_id=ctx_id)

    def embed_all(self, frame: np.ndarray) -> list[tuple[list[float], float]]:
        return [
            (list(face.normed_embedding.astype(float)), float(face.det_score))
            for face in self._app.get(frame)
        ]

    def embed(self, frame: np.ndarray) -> list[float] | None:
        faces = self._app.get(frame)
        if not faces:
            return None
        best = max(faces, key=lambda f: f.det_score)  # highest-quality detection
        return list(best.normed_embedding.astype(float))
