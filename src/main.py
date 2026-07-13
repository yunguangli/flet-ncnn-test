"""
flet-camera → YOLO (NCNN): High-Level Data Pipeline
====================================================

1. CAPTURE — flet-camera produces JPEG bytes
------------------------------------------------
`fc.Camera` is a Flutter-wrapped control that taps into the device camera.
On web it creates a `<video>` element backed by `getUserMedia`; on mobile
it uses the native camera API. The control sits in a Flet `Stack` so the
user sees a live preview.

Frame extraction uses one of two strategies depending on platform:

  - **Image streaming** (Android/iOS): `start_image_stream()` registers a
    Dart-level `onStreamedFrameAvailable` callback. Each video frame is
    JPEG-encoded on the client side and delivered to Python via
    `on_stream_image` as a `CameraImageEvent` containing `bytes`,
    `width`, `height`, and `format` metadata.

  - **take_picture() polling** (web — fallback): The web camera plugin
    does not implement image streaming. Instead, the controller polls
    `take_picture()` every ~250 ms. Each call snapshots the `<video>`
    element through a `<canvas>` and returns a single JPEG frame.

  - **OpenCV fallback** (when flet-camera capture is broken): On desktop
    web, if the flet-camera capture methods are unavailable or buggy,
    `cv2.VideoCapture(0)` opens the camera directly from the Python
    process (V4L2 on Linux) and reads raw BGR frames.

2. TRANSMISSION — browser → Python via Flet WebSocket
-------------------------------------------------------
All Flet controls live in the browser (Flutter Web). When a JPEG frame
is captured, it is serialised through Flet's msgpack-based WebSocket
protocol and arrives at the Python server as a plain `bytes` object.
No custom networking code is needed — Flet handles the transport.

3. TRANSFORM — JPEG bytes → NCNN input tensor
-----------------------------------------------
The JPEG bytes must be converted into the tensor format that the YOLO
model expects. The transformation pipeline inside
`detect_and_render_to_bytes()`:

  a. `cv2.imdecode()` → BGR numpy array (original dimensions)
  b. `ncnn.Mat.from_pixels_resize()` → RGB, resized to 640×640
  c. `substract_mean_normalize(mean=[0,0,0], norm=[1/255,1/255,1/255])`
     → uint8 pixels scaled to float32 in [0, 1]
  d. Network forward pass: `ex.input("in0")`, `ex.extract("out0")`
     → output tensor shape [84, N] where rows 0-3 are cx/cy/w/h
     (relative to 640×640) and rows 4-83 are 80 COCO class probabilities.
  e. Confidence filtering + NMS → keeps only strong, non-duplicate boxes.
  f. Boxes scaled back to original frame dimensions.
  g. Bounding boxes + labels rendered onto the original BGR frame.
  h. `cv2.imencode(".jpg")` → annotated JPEG bytes returned.

4. DETECT — running the NCNN model
-----------------------------------
The NCNN `Net` object loads a `.param` (architecture) and `.bin`
(weights) pair exported from Ultralytics. The model runs entirely on
CPU by default (`use_vulkan=False`); on Android you can flip to GPU with
Vulkan. Input/output tensor names are standard YOLOv8 ONNX naming
("in0" / "out0").

5. DISPLAY — annotated frame back to the browser
--------------------------------------------------
The controller pushes the annotated JPEG bytes plus a `DetectionResult`
dataclass (boxes, class names, target flags) to the view. The view sets
`ft.Image.src = annotated_bytes` — Flet accepts raw bytes directly (no
base64, no temp files). Flet's WebSocket protocol carries the bytes back
to the browser, where Flutter decodes and renders them, completing the
round trip at ~10 FPS.


How to use ObjectDetectionCamera
--------------------------------
Import the control and pass model paths + COCO target_classes.

Example (this product: detect laptops only)::

    from app.controls.object_detection_camera import ObjectDetectionCamera
    import os

    asset_dir = os.path.join(os.path.dirname(__file__), "assets")
    page.add(
        ObjectDetectionCamera(
            param_path=os.path.join(asset_dir, "model.ncnn.param"),
            bin_path=os.path.join(asset_dir, "model.ncnn.bin"),
            target_classes=["laptop"],   # [] keeps all COCO classes
            conf_threshold=0.30,
        )
    )

Multi-class::

    ObjectDetectionCamera(
        param_path=...,
        bin_path=...,
        target_classes=["laptop", "cell phone", "keyboard"],
    )

Optional result hook (Python callback; set after construction)::

    panel = ObjectDetectionCamera(param_path=..., bin_path=..., target_classes=["laptop"])
    panel.set_on_detection_result(lambda r: print(r.summary, r.class_counts))
    page.add(panel)

Full module docs: app/controls/object_detection_camera.py

Legacy person/holding demos (untouched): flet_detect_holding.py, ncnn_detector.py
"""

import os
import logging

import flet as ft

from app.controls.object_detection_camera import ObjectDetectionCamera

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def resolve_asset_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "assets")


async def main(page: ft.Page):
    page.title = "Laptop Detect"
    page.theme_mode = ft.ThemeMode.DARK
    page.horizontal_alignment = ft.CrossAxisAlignment.STRETCH
    page.vertical_alignment = ft.MainAxisAlignment.START
    page.padding = 16

    asset_dir = resolve_asset_dir()
    param_path = os.path.join(asset_dir, "model.ncnn.param")
    bin_path = os.path.join(asset_dir, "model.ncnn.bin")

    if not os.path.isfile(param_path) or not os.path.isfile(bin_path):
        page.add(
            ft.Text(
                f"Model files missing under {asset_dir}",
                color=ft.Colors.RED,
            )
        )
        return

    # Product config: retarget via target_classes (COCO names).
    camera_detect = ObjectDetectionCamera(
        param_path=param_path,
        bin_path=bin_path,
        target_classes=["laptop"],
        conf_threshold=0.30,
        nms_threshold=0.45,
        use_vulkan=False,
    )
    camera_detect.set_on_detection_result(
        lambda r: logger.debug("detect: %s counts=%s", r.summary, r.class_counts)
    )

    page.appbar = ft.AppBar(
        title=ft.Text("Laptop Detect (NCNN)"),
        bgcolor=ft.Colors.BLUE_GREY_900,
    )
    logger.info("ObjectDetectionCamera host (targets=laptop)")
    page.add(camera_detect)
    page.update()


if __name__ == "__main__":
    ft.run(main)
