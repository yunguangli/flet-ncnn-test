# Flet NCNN YOLO Laptop Detection App

## Overview
Flet-based cross-platform (desktop/web/Android) YOLO NCNN object detection app. Detects laptops (and other COCO objects) via camera preview with real-time NCNN inference.

## Architecture
- **CVM pattern**: Controls (Views) → Controller → Model
- **`src/main.py`**: Entry point; creates `ObjectDetectionCamera` control and adds to page
- **`src/app/controls/object_detection_camera.py`**: Custom Flet control (`@ft.control(kw_only=True)`, extends `ft.Column`). Builds the UI (camera area, buttons, status, detection results). Creates detector and controller in `build()`.
- **`src/app/controllers/detection_controller.py`**: Manages camera lifecycle, frame capture, detection loop. Uses `run_in_executor` for blocking OpenCV calls.
- **`src/app/models/detector.py`**: `YOLOv8NCNNDetector` class. Guarded imports for `ncnn` and `cv2` (fallback to PIL). 
- **`src/app/models/types.py`**: Dataclasses for `BoundingBox`, `DetectionResult`, `DetectorConfig`.

## Key Design Decisions
- Use `object.__setattr__` for non-serializable fields (detector, controller) to avoid msgpack issues
- `@ft.control(kw_only=True)` (no `isolated=True`) — removed to fix mobile APK rendering
- On mobile: uses `flet_camera` (Android permission required), on desktop: uses `cv2.VideoCapture`

## Build & Deployment
- Desktop: `flet run src/main.py`
- Web: `flet run --web src/main.py`
- APK: `flet build apk --include-packages flet_camera`
- APK post-processing: `python tools/patch_apk_ncnn.py <apk>` — injects missing ncnn .so from pypi.flet.dev into jniLibs/ and adds .soref marker
- Sign: `apksigner sign --ks ~/.android/debug.keystore --ks-pass pass:android --ks-key-alias androiddebugkey <patched-apk>`

## Known Issues
- **NCNN on Android**: `serious_python` only scans for `.abi3.so` extensions; ncnn has platform-specific naming, so the .so is stripped from sitepackages.zip without being relocated to jniLibs/.
  Fix: `tools/patch_apk_ncnn.py` downloads the correct Android wheel from pypi.flet.dev and injects it post-build.
- **OpenCV on Android**: Module-level `import cv2` crashes on Android APK. Fixed with `try/except ImportError` + `_CV2_OK` flag + PIL fallback.
- **Desktop freeze**: OpenCV blocking calls on asyncio event loop. Fixed with `loop.run_in_executor()`.
- **Mobile rendering issue**: `ObjectDetectionCamera` content invisible on APK. Trying without SafeArea wrapper and without `isolated=True`.

## Tools
- `tools/patch_apk_ncnn.py`: Post-processes APK to inject ncnn native module.
