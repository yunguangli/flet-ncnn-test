"""
Laptop-detect entry point — hosts the reusable ObjectDetectionCamera control.

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
    page.padding = 16
    page.scroll = ft.ScrollMode.AUTO

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

    logger.info("ObjectDetectionCamera host (targets=laptop)")
    page.add(
        ft.SafeArea(
            content=ft.Column(
                spacing=12,
                controls=[
                    ft.AppBar(
                        title=ft.Text("Laptop Detect (NCNN)"),
                        bgcolor=ft.Colors.BLUE_GREY_900,
                    ),
                    camera_detect,
                ],
            ),
        )
    )


if __name__ == "__main__":
    ft.run(main)
