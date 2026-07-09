# Core — flet-ncnn-test

## Purpose
Real-time object detection GUI app using Flet + NCNN. Camera-based live inference with YOLO models, targeting desktop web first, mobile Android later.

## Source map
- `src/main.py` — Entry point; wires Model → Controller → View (CVM pattern)
- `src/app/models/` — Pure data & inference (no Flet imports)
  - `types.py` — Dataclasses: BoundingBox, DetectionResult, DetectorConfig, CameraConfig
  - `detector.py` — YOLOv8NCNNDetector class wrapping ncnn.Net
- `src/app/controllers/` — Business logic & state
  - `detection_controller.py` — Camera lifecycle, detection loop, event routing
- `src/app/views/` — Flet UI
  - `detection_view.py` — Stack layout (camera + annotated overlay), button bindings
- `src/assets/model.ncnn.param|bin` — YOLO NCNN model files (exported via Ultralytics)
- `src/ncnn_detector.py` — Original single-file detector (kept for reference)
- `src/flet_detect_holding.py` — Original static-image demo (kept for reference)

## Architecture invariants
- CVM (Control-View-Model): models/ has no Flet imports, views/ has no business logic, controllers/ owns runtime state.
- Camera backend chosen by platform in `main.py`: flet-camera (web/mobile) or OpenCV (desktop). Controller takes `use_flet_camera` flag.
- flet_camera.Camera control created in View, shared with Controller via `controller.attach_camera(camera)`.
- flet-camera works on web: `start_image_stream()` + `on_stream_image` (CameraImageEvent.bytes = JPEG when initialized with ImageFormatGroup.JPEG). Check `supports_image_streaming()` before streaming.
- Two-button UX: "Start/Stop Camera" (init camera — required for web user-gesture) + "Start/Stop Detection" (toggle NCNN inference). No auto-start.
- Stack layout in View: Camera control (native preview, bottom) + Image overlay (annotated frames during detection, transparent during preview).
- OpenCV fallback (desktop only): cv2.VideoCapture + asyncio preview/detection loops. Server-side camera — wrong device on web, hence flet-camera is primary there.
- `pyproject.toml` is the single source of truth for deps, no requirements.txt.
- `flet run --web` is the primary dev command; `flet run` (desktop) uses OpenCV (flet-camera unsupported on desktop).

## Key constraints
- Browser media API requires user gesture → camera starts only on "Start Camera" button click (not auto-started).
- Camera control must be visible (not `visible=False`) in page tree for invoke-method handlers to register.
- Image `src` supports raw `bytes` directly (not `src_base64`, which was removed in Flet 1.0+).
- NCNN model input is 640×640, output is `out0` tensor with COCO 80-class format.
