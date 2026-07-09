# Core — flet-ncnn-test (product: laptop-detect)

## Purpose
Real-time object detection GUI: Flet + NCNN YOLO inference from camera. Desktop web primary; Android target later.

## Source map
- `src/main.py` — Entry: resolve assets → detector → Controller/View; platform chooses camera backend
- `src/app/models/` — Pure data & inference (no Flet)
  - `types.py` — BoundingBox, DetectionResult, DetectorConfig, CameraConfig
  - `detector.py` — `YOLOv8NCNNDetector` (ncnn.Net wrapper; class name legacy, model is YOLOv11n-exported)
- `src/app/controllers/detection_controller.py` — Camera lifecycle, detection loops, events
- `src/app/views/detection_view.py` — Stack UI (camera + annotated overlay), buttons
- `src/assets/model.ncnn.param|bin` — Bundled NCNN model used at runtime (`build_detector_config`)
- `src/yolo11n_ncnn_model/`, `src/yolov8n_ncnn_model/` — Extra exported models (not wired by default)
- `src/ncnn_detector.py`, `src/flet_detect_holding.py` — Reference/legacy single-file scripts
- `src/tests/` — Ad-hoc YOLO/NCNN experiments; `tests/test_main.py` — template Flet test (counter stub, not app-aligned)
- Spec notes: `todo.txt`; Flet skill docs under `.github/skills/`

## Architecture invariants
- CVM: models/ no Flet; views/ no business logic; controllers/ own runtime state
- Camera: flet-camera on web/Android/iOS (`page.web` or platform); OpenCV on desktop (`use_flet_camera` flag)
- Camera control created in View, attached via `controller.attach_camera(camera)`
- flet-camera web: `start_image_stream()` + `on_stream_image` (JPEG when ImageFormatGroup.JPEG); gate with `supports_image_streaming()`
- UX: separate "Start/Stop Camera" and "Start/Stop Detection" — no auto-start (browser user-gesture)
- View Stack: native Camera preview under Image overlay (annotated frames when detecting; transparent placeholder in preview)
- OpenCV path: `cv2.VideoCapture` + asyncio preview/detection loops (desktop only)
- `pyproject.toml` sole dep source; app path `src` via `[tool.flet.app]`

## Key constraints
- Browser media needs user gesture → camera only on button click
- Camera control must stay in page tree (visible) for invoke-method handlers
- Image `src` accepts raw `bytes` (no `src_base64` in Flet 1.0+)
- NCNN: 640 input, `out0` tensor, COCO 80-class; `use_vulkan=False` by default (True for Android GPU builds)
- Related: `mem:tech_stack`, `mem:conventions`, `mem:suggested_commands`, `mem:task_completion`
