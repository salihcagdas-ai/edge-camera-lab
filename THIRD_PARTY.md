# Dependencies and model provenance

This snapshot contains the repository owner's camera analytics code, curated as a
portfolio prototype. It does not bundle third-party neural-network weights, camera
recordings, face datasets, or compiled HEF files. No open-source license is granted
for the original application code in this snapshot.

Dependencies are installed separately and retain their own licenses. The tested
core/media environment is recorded in `requirements-tested.txt`; optional hardware
and recognition dependencies are declared in `pyproject.toml`.

- [InsightFace](https://github.com/deepinsight/insightface/blob/master/python-package/README.md):
  library code is MIT licensed; the supplied pretrained models are restricted to
  non-commercial research, including manually downloaded models. The adapter here
  targets the 0.7.x API. No rights to its model weights are conveyed by this project.
- [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo): a possible source
  of device-specific model configurations. Check the selected model's original
  source, license, and Hailo runtime/device compatibility separately.
- [Ultralytics YOLO](https://docs.ultralytics.com/models/yolov8/): if a YOLO-derived
  HEF is selected, the source model has its own licensing requirements. This
  repository provides only an output adapter and includes no YOLO weights.

For any model used in an experiment, record its source URL, exact version, SHA-256,
license, input/output contract, and device/runtime version in your private run
records. Synthetic demo results are not evidence about those external models.
