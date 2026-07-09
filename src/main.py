"""
Real-time Object Detection App with Flet + NCNN
================================================

Architecture: CVM (Control-View-Model)
---------------------------------------
A layered architecture that separates concerns cleanly:

  ┌─────────────────────────────────────────────────┐
  │                    main.py                       │
  │  Entry point: creates Model, Controller, View,   │
  │  wires them together, and launches the app.      │
  └────────────────────┬────────────────────────────┘
                       │
  ┌────────────────────┴────────────────────────────┐
  │                 Controller                       │
  │  app/controllers/detection_controller.py         │
  │                                                  │
  │  - Owns all runtime state (is_detecting, FPS)    │
  │  - Handles camera lifecycle (init, stream)       │
  │  - Processes frames: camera → detect → render   │
  │  - Pushes results to View via callbacks          │
  └────┬───────────────────────────────┬─────────────┘
       │                               │
  ┌────┴──────────┐          ┌────────┴──────────┐
  │    Model      │          │      View          │
  │  models/      │          │  views/            │
  │               │          │                    │
  │  - Detector   │          │  - Flet UI layout  │
  │  - Data types │          │  - Event bindings  │
  │  - No Flet    │          │  - No business     │
  │    imports    │          │    logic           │
  └───────────────┘          └────────────────────┘

Data flow (frame pipeline):
  Camera (flet-camera) → JPEG bytes
       ↓
  Controller._on_stream_image()
       ↓
  Detector.detect_and_render_to_bytes(bytes)
       ↓
  (decode → NCNN inference → render boxes → encode)
       ↓
  Annotated JPEG bytes + DetectionResult
       ↓
  View._on_frame_updated()
       ↓
  Image.src = bytes  →  UI updates

Directory structure:
  src/
  ├── main.py                     Entry point (this file)
  ├── app/
  │   ├── models/
  │   │   ├── types.py            Dataclasses (BoundingBox, DetectionResult, etc.)
  │   │   └── detector.py         NCNN inference engine (YOLOv8NCNNDetector)
  │   ├── controllers/
  │   │   └── detection_controller.py  App state, camera lifecycle, event routing
  │   └── views/
  │       └── detection_view.py   Flet UI controls and layout
  ├── assets/                     Model files, images
  ├── tests/                      Unit tests
  ├── ncnn_detector.py            Original single-file detector (kept for reference)
  └── flet_detect_holding.py      Original static-image app (kept for reference)

Dependencies:
  - flet (>=0.86.0.dev1)          UI framework
  - flet-camera                   Camera control (web, Android, iOS)
  - ncnn                          Neural network inference
  - opencv-python                 Image processing
  - numpy                         Array operations
"""

import os
import logging

import flet as ft

from app.models.detector import YOLOv8NCNNDetector
from app.models.types import DetectorConfig
from app.controllers.detection_controller import DetectionController
from app.views.detection_view import DetectionView

# ── Logging setup ──────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def resolve_asset_dir() -> str:
    """Return the directory containing model files and assets.

    When running via 'flet run' or 'flet build apk', assets are resolved
    relative to the src/ directory. On Android, they are bundled into the APK.
    On desktop/web, they are read from the filesystem.
    """
    return os.path.join(os.path.dirname(__file__), "assets")


def build_detector_config(asset_dir: str) -> DetectorConfig:
    """Create the detector configuration pointing at the NCNN model files.

    The model files (model.ncnn.param, model.ncnn.bin) are exported from
    YOLOv11n via Ultralytics. They live in src/assets/ and are bundled
    with the app on mobile.
    """
    return DetectorConfig(
        param_path=os.path.join(asset_dir, "model.ncnn.param"),
        bin_path=os.path.join(asset_dir, "model.ncnn.bin"),
        use_vulkan=False,       # Set True when building APK for Android GPU
        conf_threshold=0.30,    # Minimum confidence to report a detection
        nms_threshold=0.45,     # IoU threshold for Non-Maximum Suppression
        input_size=640,         # YOLO training resolution
    )


# ── App entry point ───────────────────────────────────────
async def main(page: ft.Page):
    """Main Flet application entry point.

    This function runs once when the app starts. It:
      1. Resolves asset paths for the current platform
      2. Creates the Model (detector)
      3. Chooses a camera backend (flet-camera on web/mobile, OpenCV on desktop)
      4. Creates the Controller + View
      5. Adds the UI to the page

    The camera is not auto-started: the user clicks "Start Camera" because
    browsers require a user gesture before granting camera access.
    """
    use_flet_camera = page.web or page.platform in (
        ft.PagePlatform.ANDROID,
        ft.PagePlatform.IOS,
    )
    backend = "flet-camera" if use_flet_camera else "OpenCV"
    logger.info("Starting NCNN Real-time Detection app (%s backend)", backend)

    # 1. Resolve paths
    asset_dir = resolve_asset_dir()
    detector_config = build_detector_config(asset_dir)

    # 2. Create Model — the NCNN detector
    try:
        detector = YOLOv8NCNNDetector(detector_config)
        logger.info("Detector initialized: %s", detector_config.param_path)
    except Exception as e:
        logger.error("Failed to initialize detector: %s", e)
        page.add(ft.Text(f"Detector init failed: {e}", color=ft.Colors.RED))
        return

    # 3. Create Controller — business logic. Camera backend is chosen by platform:
    #    flet-camera on web/mobile (uses the browser/device camera), OpenCV on
    #    desktop (accesses the local camera directly).
    controller = DetectionController(
        detector=detector, use_flet_camera=use_flet_camera
    )

    # 4. Create View — Flet UI (creates the flet-camera control when needed).
    view = DetectionView(controller=controller)
    root = view.build(page)

    # 5. Add UI to the page.
    page.add(root)
    page.update()


if __name__ == "__main__":
    ft.run(main)
