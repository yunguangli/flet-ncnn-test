"""
DEPRECATED — superseded by the reusable control:

    from app.controls.object_detection_camera import ObjectDetectionCamera

See that module and src/main.py for usage. This file is kept only so old
import paths fail loudly if something still imports DetectionView after the
Type/UI refactor (pure class detection, no holding).
"""

raise ImportError(
    "app.views.detection_view is deprecated. "
    "Use app.controls.object_detection_camera.ObjectDetectionCamera instead."
)
