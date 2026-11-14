"""Synthetic COCO NMS parser checks, including letterbox coordinate mapping."""

from __future__ import annotations

import numpy as np
import pytest

from edge.detect import COCO_CAT, COCO_PERSON, Detection, FakeObjectDetector, parse_nms_by_class

# A 1280x720 frame letterboxed into a 640x640 model input:
# scale = 0.5, resized 640x360, pad_w = 0, pad_h = (640 - 360) // 2 = 140.
_MODEL = {"model_h": 640, "model_w": 640, "scale": 0.5, "pad_w": 0, "pad_h": 140}


def _output(person_rows: list[list[float]], cat_rows: list[list[float]] | None = None) -> list:
    """Hailo Model Zoo NMS-by-class shape: list indexed by class_id."""
    output: list = [np.zeros((0, 5), dtype=np.float32)] * 80
    output[COCO_PERSON] = np.asarray(person_rows, dtype=np.float32)
    if cat_rows is not None:
        output[COCO_CAT] = np.asarray(cat_rows, dtype=np.float32)
    return output


def test_boxes_unmap_through_the_letterbox_to_frame_pixels() -> None:
    # Normalized [y_min, x_min, y_max, x_max, score] centered person box.
    rows = [[0.34375, 0.25, 0.65625, 0.75, 0.9]]  # y: 220..420, x: 160..480 in model px
    detections = parse_nms_by_class(_output(rows), min_confidence=0.5, **_MODEL)
    assert len(detections) == 1
    detection = detections[0]
    assert detection.label == "person"
    x1, y1, x2, y2 = detection.bbox
    # x: (160 - 0) / 0.5 = 320 .. 960; y: (220 - 140) / 0.5 = 160 .. 560
    assert (round(x1), round(y1), round(x2), round(y2)) == (320, 160, 960, 560)


def test_confidence_threshold_and_cat_label() -> None:
    detections = parse_nms_by_class(
        _output(
            person_rows=[[0.1, 0.1, 0.2, 0.2, 0.30]],  # below threshold
            cat_rows=[[0.5, 0.5, 0.6, 0.6, 0.80]],
        ),
        min_confidence=0.5,
        **_MODEL,
    )
    assert [d.label for d in detections] == ["cat"]
    assert detections[0].class_id == COCO_CAT


def test_unexpected_layout_fails_instead_of_claiming_an_empty_scene() -> None:
    for output in (3.14, [[["bad"]]], [], {"output": []}):
        with pytest.raises(ValueError, match="HEF contract"):
            parse_nms_by_class(output, min_confidence=0.5, **_MODEL)


def test_short_rows_fail_instead_of_hiding_layout_mismatch() -> None:
    output = _output([[0.1, 0.2, 0.3]])  # len < 5
    with pytest.raises(ValueError, match="HEF contract"):
        parse_nms_by_class(output, min_confidence=0.1, **_MODEL)


@pytest.mark.parametrize(
    "row",
    [
        [0.2, 0.2, 0.5, 0.5, float("nan")],
        [0.2, 0.2, 0.5, float("inf"), 0.9],
        [0.2, 0.2, 0.5, 0.5, 1.1],
        [0.5, 0.2, 0.2, 0.5, 0.9],
        [0.2, -0.1, 0.5, 0.5, 0.9],
    ],
)
def test_invalid_nms_rows_cannot_trigger_detections(row: list[float]) -> None:
    with pytest.raises(ValueError):
        parse_nms_by_class(_output([row]), min_confidence=0.5, **_MODEL)


def test_padding_only_prediction_is_not_a_person_in_the_scene() -> None:
    assert (
        parse_nms_by_class(_output([[0, 0.2, 0.1, 0.4, 0.9]]), min_confidence=0.5, **_MODEL) == []
    )


def test_fake_detector_repeats_last_script_entry() -> None:
    person = Detection(class_id=0, label="person", confidence=0.9, bbox=(0, 0, 10, 10))
    detector = FakeObjectDetector([[person], []])
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    assert detector.detect(frame) == [person]
    assert detector.detect(frame) == []
    assert detector.detect(frame) == []  # last entry repeats
