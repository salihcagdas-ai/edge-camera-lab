"""Optional Hailo-8L COCO person/cat detection.

The device adapter uses picamera2's Hailo InferModel helper, BGR-to-RGB
conversion, and aspect-preserving letterboxing. It expects 80 per-class
(N,5) arrays: normalized y_min, x_min, y_max, x_max, score. Verify this
contract against the chosen compiled HEF before hardware use. Wrong output
raises an error instead of reporting an empty scene.

Pure parser tests require no device. Hardware integration is separate.
"""

from __future__ import annotations

import contextlib
import logging
import math
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

logger = logging.getLogger(__name__)

COCO_PERSON = 0
COCO_CAT = 15
LABELS: dict[int, str] = {COCO_PERSON: "person", COCO_CAT: "cat", 16: "dog"}

LABEL_PERSON = "person"
LABEL_CAT = "cat"


@dataclass(slots=True)
class Detection:
    class_id: int
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in original-frame pixels


class ObjectDetector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Detection]: ...


class FakeObjectDetector:
    """Scripted detections for tests: one list per call, last entry repeats."""

    def __init__(self, script: list[list[Detection]]) -> None:
        self._script = list(script)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if not self._script:
            return []
        if len(self._script) == 1:
            return list(self._script[0])
        return list(self._script.pop(0))


def letterbox(frame_bgr: np.ndarray, target_h: int, target_w: int) -> tuple[Any, float, int, int]:
    """Aspect-preserving resize + centered 114-gray pad (uint8 HWC BGR)."""
    import cv2  # lazy: heavy, only needed on the real detector path

    h, w = frame_bgr.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    pad_w = (target_w - new_w) // 2
    pad_h = (target_h - new_h) // 2
    canvas[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized
    return canvas, scale, pad_w, pad_h


def parse_nms_by_class(
    output: Any,
    *,
    model_h: int,
    model_w: int,
    scale: float,
    pad_w: int,
    pad_h: int,
    min_confidence: float,
) -> list[Detection]:
    """Parse NMS-by-class output; map boxes back through the letterbox.

    Pure numpy/python so it is testable without hailo_platform. Malformed
    COCO output raises ValueError; a broken detector must not masquerade as
    a successfully analyzed empty scene.
    """
    if (
        model_h <= 0
        or model_w <= 0
        or scale <= 0
        or not math.isfinite(scale)
        or not math.isfinite(min_confidence)
        or not 0 <= min_confidence <= 1
        or not 0 <= pad_w < model_w / 2
        or not 0 <= pad_h < model_h / 2
    ):
        raise ValueError("Invalid detector dimensions, scale, padding or confidence threshold")
    detections: list[Detection] = []
    try:
        if len(output) != 80:
            raise ValueError("Expected 80 COCO class arrays")
        for class_id, boxes in enumerate(output):
            if boxes is None or len(boxes) == 0:
                continue
            rows = np.asarray(boxes, dtype=np.float64)
            if rows.ndim != 2 or rows.shape[1] != 5 or not np.isfinite(rows).all():
                raise ValueError("Expected finite (N,5) class arrays")
            for y_min, x_min, y_max, x_max, score in rows:
                if not 0 <= score <= 1:
                    raise ValueError("Confidence must be in [0,1]")
                if score <= 0 or score < min_confidence:
                    continue
                if not 0 <= x_min < x_max <= 1 or not 0 <= y_min < y_max <= 1:
                    raise ValueError("Detection coordinates must be normalized ordered boxes")
                # normalized -> letterboxed pixels -> original-frame xyxy
                x1 = max(0.0, (x_min * model_w - pad_w) / scale)
                y1 = max(0.0, (y_min * model_h - pad_h) / scale)
                x2 = min((model_w - 2 * pad_w) / scale, (x_max * model_w - pad_w) / scale)
                y2 = min((model_h - 2 * pad_h) / scale, (y_max * model_h - pad_h) / scale)
                if x2 <= x1 or y2 <= y1:  # prediction lies entirely in letterbox padding
                    continue
                detections.append(
                    Detection(
                        class_id=class_id,
                        label=LABELS.get(class_id, f"class_{class_id}"),
                        confidence=score,
                        bbox=(x1, y1, x2, y2),
                    )
                )
    except (TypeError, IndexError, ValueError) as error:
        raise ValueError(
            "Invalid Hailo COCO NMS output; verify the compiled HEF contract"
        ) from error
    return detections


class HailoDetector:
    """Hailo ``.hef`` person+cat detector via the async InferModel engine.

    Wraps ``picamera2.devices.Hailo`` (HailoRT ``InferModel`` + round-robin
    scheduler) — the camera-safe path (see the module docstring, fix #1).
    """

    def __init__(self, hef_path: str, min_confidence: float = 0.5) -> None:
        self._hef_path = hef_path
        self._min_confidence = min_confidence
        self._hailo: Any = None
        self._h = 0
        self._w = 0

    def load(self) -> None:
        try:
            from picamera2.devices import Hailo
        except ImportError as error:  # pragma: no cover - Pi-only dependency
            raise RuntimeError(
                "picamera2 Hailo helper not available (needs hailo-all + "
                "python3-picamera2 on the Pi)"
            ) from error

        self._hailo = Hailo(self._hef_path)
        shape = self._hailo.get_input_shape()  # (H, W, C)
        self._h, self._w = int(shape[0]), int(shape[1])
        logger.info(
            "Hailo HEF loaded (InferModel API): %s (input %dx%d)", self._hef_path, self._w, self._h
        )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if self._hailo is None:
            raise RuntimeError("call load() first")
        canvas, scale, pad_w, pad_h = letterbox(frame, self._h, self._w)
        rgb = np.ascontiguousarray(canvas[:, :, ::-1])  # BGR -> RGB
        # NMS-by-class output: a list of ``num_classes`` (N, 5) arrays.
        output = self._hailo.run(rgb)
        return parse_nms_by_class(
            output,
            model_h=self._h,
            model_w=self._w,
            scale=scale,
            pad_w=pad_w,
            pad_h=pad_h,
            min_confidence=self._min_confidence,
        )

    def close(self) -> None:
        if self._hailo is not None:
            with contextlib.suppress(Exception):
                self._hailo.close()
            self._hailo = None
