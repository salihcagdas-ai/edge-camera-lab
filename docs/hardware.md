# Optional hardware integration

The offline demo is the verified entry point. The following adapters are included
for further integration; this release does not include device benchmark results.

## Face inference and camera

Install the optional CV dependencies in a separate environment:

```console
python -m pip install -e ".[cv]"
```

The adapter targets InsightFace 0.7.x and CPU ONNX Runtime. The optional package may
require native build tools. Provision a model pack separately, under
`models/insightface/models/buffalo_s/*.onnx`, after checking its model license.
`EDGE_CAMERA_INSIGHTFACE_ROOT` changes the root and `EDGE_CAMERA_INSIGHTFACE_MODEL`
changes the pack name. Missing packs cause an error before model initialization.

On Raspberry Pi OS, install Picamera2 using the platform's package instructions.
The default `EDGE_CAMERA_SOURCE=picamera` expects a Pi camera. An RTSP source requires
`EDGE_CAMERA_SOURCE=rtsp` and `EDGE_CAMERA_RTSP_URL`; keep credentials in a private
environment configuration and use a trusted camera/network. No public camera is
configured or contacted by this repository.

Set `EDGE_CAMERA_WEB_PIN` to your own strong passphrase and choose a private
`EDGE_CAMERA_DATA_DIR` outside the checkout before starting `python -m edge`.
Settings are read from process environment variables; `.env.example` is reference
material and is not loaded automatically.

Enroll only authorized, single-person photos with:

```console
python -m edge.enroll --name "Example person" --photos ./private-photos/
```

Calibration requires representative known and unknown people under the intended
lighting and placement. No recognition threshold is validated by the synthetic demo.

## Hailo object detection

`EDGE_CAMERA_DETECTOR=hailo` enables the Picamera2 Hailo adapter and
`EDGE_CAMERA_HEF_PATH` selects a local compiled HEF. The parser expects **80 COCO
classes**, each with rows `[ymin, xmin, ymax, xmax, confidence]`, normalized boxes,
and finite values. It recognizes the person and cat class indices. This is a
specific NMS-by-class contract, not a generic HEF loader. Verify the output layout,
runtime compatibility, and model license for your device before using it.

## Optional behaviors

SMTP is disabled without a host. Enabling it requires sender/recipient settings
and transmits event snapshots to the chosen mail service. Delivery is best effort;
there is no durable retry queue, and SMTP waits can delay analysis.

The device-presence probe is disabled by default and uses Linux `ping` flags.
It is not supported by the Windows demo. Device reachability is neither reliable
occupancy detection nor an authentication factor.
