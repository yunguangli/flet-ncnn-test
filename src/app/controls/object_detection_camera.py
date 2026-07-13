"""
ObjectDetectionCamera — reusable Flet composite control.

Live camera preview + optional NCNN object detection with configurable COCO
target classes. See https://flet.dev/docs/cookbook/custom-controls

Non-serializable objects (NCNN detector, controller) are created inside
build() so Flet does not try to msgpack them into the session.

Usage
-----
    from app.controls.object_detection_camera import ObjectDetectionCamera

    page.add(
        ObjectDetectionCamera(
            param_path="/path/to/model.ncnn.param",
            bin_path="/path/to/model.ncnn.bin",
            target_classes=["laptop"],          # or ["laptop", "cell phone"]; [] = all COCO
            conf_threshold=0.30,
            # optional size:
            # display_width=640,
            # display_height=480,
        )
    )

Optional result hook (set AFTER construction; not a Flet control field)::

    panel = ObjectDetectionCamera(param_path=..., bin_path=..., target_classes=["laptop"])
    panel.set_on_detection_result(lambda r: print(r.summary, r.class_counts))
    page.add(panel)

Buttons
-------
- Start / Stop Camera — preview (flet-camera on web/Android/iOS; OpenCV desktop)
- Start / Stop Detection — NCNN; web uses take_picture() when streaming unsupported

Public async methods (same as buttons): start_camera, stop_camera,
start_detection, stop_detection, toggle_detection.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from dataclasses import field
from typing import Any, Callable

import flet as ft
import flet_camera as fc

from app.controllers.detection_controller import DetectionController
from app.models.detector import YOLOv8NCNNDetector
from app.models.types import DetectionResult, DetectorConfig

logger = logging.getLogger(__name__)

_TRANSPARENT_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwC"
    "AAAAC0lEQVR42mP8/wIAAgMBAp0YVwAAAABJRU5ErkJggg=="
)


@ft.control(kw_only=True)
class ObjectDetectionCamera(ft.Column):
    """Camera + NCNN detection panel. All host-facing options are serializable fields."""

    param_path: str = ""
    bin_path: str = ""
    target_classes: list[str] = field(default_factory=lambda: ["laptop"])
    conf_threshold: float = 0.30
    nms_threshold: float = 0.45
    input_size: int = 640
    use_vulkan: bool = False
    display_width: int = 640
    display_height: int = 480

    spacing: int = 12
    scroll: ft.ScrollMode = ft.ScrollMode.AUTO
    horizontal_alignment: ft.CrossAxisAlignment = ft.CrossAxisAlignment.STRETCH

    def set_on_detection_result(
        self, callback: Callable[[DetectionResult], None] | None
    ) -> None:
        """Register a Python-only callback (not a Flet-serialized field)."""
        object.__setattr__(self, "_on_detection_result", callback)

    def _build_error(self, title: str, detail: str):
        self.controls = [
            ft.Container(
                content=ft.Column(
                    spacing=8,
                    controls=[
                        ft.Text(title, color=ft.Colors.RED, weight=ft.FontWeight.BOLD, size=16),
                        ft.Text(detail, size=13, color=ft.Colors.GREY_400, selectable=True),
                    ],
                ),
                padding=20,
            )
        ]

    def build(self):
        if not self.param_path or not self.bin_path:
            self._build_error(
                "Missing model paths",
                "ObjectDetectionCamera requires param_path and bin_path.",
            )
            return

        try:
            detector = YOLOv8NCNNDetector(
                DetectorConfig(
                    param_path=self.param_path,
                    bin_path=self.bin_path,
                    use_vulkan=self.use_vulkan,
                    conf_threshold=self.conf_threshold,
                    nms_threshold=self.nms_threshold,
                    input_size=self.input_size,
                    target_classes=list(self.target_classes),
                )
            )
        except RuntimeError as e:
            self._build_error("NCNN not available", str(e))
            return
        except FileNotFoundError as e:
            self._build_error("Model files not found", str(e))
            return
        except Exception as e:
            logger.exception("Unexpected error creating detector")
            self._build_error("Detector init failed", f"{type(e).__name__}: {e}")
            return

        use_flet_camera = False
        if self.page is not None:
            use_flet_camera = self.page.web or self.page.platform in (
                ft.PagePlatform.ANDROID,
                ft.PagePlatform.IOS,
            )

        lens_dir = getattr(self, "_lens_direction", fc.CameraLensDirection.BACK)
        controller = DetectionController(
            detector=detector,
            use_flet_camera=use_flet_camera,
            lens_direction=lens_dir,
        )

        # Python-only runtime state (not declared as control fields → not msgpacked)
        object.__setattr__(self, "_detector", detector)
        object.__setattr__(self, "_controller", controller)
        object.__setattr__(self, "_use_flet_camera", use_flet_camera)
        object.__setattr__(self, "_camera", None)
        object.__setattr__(self, "_preview_active", False)
        object.__setattr__(self, "_frame_counter", 0)
        object.__setattr__(self, "_lens_direction", lens_dir)
        if not hasattr(self, "_on_detection_result"):
            object.__setattr__(self, "_on_detection_result", None)

        display_image = ft.Image(
            src=_TRANSPARENT_1PX,
            width=self.display_width,
            height=self.display_height,
            fit=ft.BoxFit.CONTAIN,
            gapless_playback=True,
        )
        display_stack = ft.Stack(
            controls=[display_image],
            width=self.display_width,
            height=self.display_height,
        )
        object.__setattr__(self, "_display_image", display_image)
        object.__setattr__(self, "_display_stack", display_stack)

        display_area = ft.Container(
            content=display_stack,
            width=self.display_width,
            height=self.display_height,
            bgcolor=ft.Colors.BLACK,
            border_radius=ft.BorderRadius.all(12),
            alignment=ft.Alignment.CENTER,
        )

        camera_btn = ft.FilledButton(
            content=ft.Text("Start Camera"),
            icon=ft.Icons.VIDEOCAM,
            on_click=self._on_toggle_camera,
            style=ft.ButtonStyle(
                color=ft.Colors.WHITE,
                bgcolor=ft.Colors.BLUE_700,
                padding=ft.Padding.symmetric(horizontal=24, vertical=16),
                text_style=ft.TextStyle(size=16, weight=ft.FontWeight.BOLD),
            ),
        )
        flip_btn = ft.IconButton(
            icon=ft.Icons.FLIP_CAMERA_ANDROID,
            tooltip="Switch front/back camera",
            on_click=self._on_flip_camera,
            visible=use_flet_camera,
        )
        toggle_btn = ft.FilledButton(
            content=ft.Text("Start Detection"),
            icon=ft.Icons.PLAY_ARROW,
            on_click=self._on_toggle_detection,
            disabled=True,
            style=ft.ButtonStyle(
                color=ft.Colors.WHITE,
                bgcolor=ft.Colors.GREEN_700,
                padding=ft.Padding.symmetric(horizontal=32, vertical=16),
                text_style=ft.TextStyle(size=18, weight=ft.FontWeight.BOLD),
            ),
        )
        status_text = ft.Text(
            'Click "Start Camera" to begin.', size=14, color=ft.Colors.GREY_400,
        )
        fps_text = ft.Text(
            "FPS: --", size=16, weight=ft.FontWeight.BOLD, color=ft.Colors.GREEN_400,
        )
        backend_label = "flet-camera" if use_flet_camera else "OpenCV"
        backend_text = ft.Text(
            f"Backend: {backend_label}",
            size=12,
            color=ft.Colors.GREY_500,
            italic=True,
        )
        result_text = ft.Text(size=14, selectable=True)
        detection_list = ft.Column(
            spacing=2, visible=False, height=150, scroll=ft.ScrollMode.AUTO,
        )
        object.__setattr__(self, "_camera_btn", camera_btn)
        object.__setattr__(self, "_flip_btn", flip_btn)
        object.__setattr__(self, "_toggle_btn", toggle_btn)
        object.__setattr__(self, "_status_text", status_text)
        object.__setattr__(self, "_fps_text", fps_text)
        object.__setattr__(self, "_result_text", result_text)
        object.__setattr__(self, "_detection_list", detection_list)
        object.__setattr__(self, "_backend_text", backend_text)

        self.controls = [
            ft.Row(
                controls=[camera_btn, flip_btn, toggle_btn, fps_text, backend_text],
                alignment=ft.MainAxisAlignment.CENTER,
                wrap=True,
            ),
            ft.Row(
                controls=[display_area],
                alignment=ft.MainAxisAlignment.CENTER,
            ),
            status_text,
            ft.Divider(height=1),
            ft.Text("Detection Results:", size=16, weight=ft.FontWeight.BOLD),
            result_text,
            detection_list,
        ]

        self._wire_controller()

    def will_unmount(self):
        ctrl: DetectionController | None = getattr(self, "_controller", None)
        if ctrl is None:
            return

        async def _cleanup():
            try:
                if ctrl.is_detecting or ctrl.is_previewing:
                    await ctrl.stop_camera()
            except Exception as e:
                logger.warning("Cleanup stop_camera: %s", e)
            if ctrl.use_flet_camera:
                await self._remove_camera_from_stack()

        try:
            if self.page is not None:
                self.page.run_task(_cleanup)
        except Exception as e:
            logger.warning("will_unmount cleanup: %s", e)

    # ── Public API ─────────────────────────────────────────────

    async def start_camera(self) -> bool:
        ctrl = self._ctrl()
        if ctrl is None:
            return False
        if ctrl.is_previewing:
            return True
        if ctrl.use_flet_camera:
            await self._ensure_camera_in_stack()
        success = await ctrl.start_camera()
        if not success and ctrl.use_flet_camera:
            await self._remove_camera_from_stack()
        return success

    async def stop_camera(self) -> bool:
        ctrl = self._ctrl()
        if ctrl is None:
            return False
        ok = await ctrl.stop_camera()
        if ctrl.use_flet_camera:
            await self._remove_camera_from_stack()
        return ok

    async def toggle_detection(self) -> bool:
        ctrl = self._ctrl()
        if ctrl is None:
            return False
        return await ctrl.toggle_detection()

    async def start_detection(self) -> bool:
        ctrl = self._ctrl()
        if ctrl is None:
            return False
        if ctrl.is_detecting:
            return True
        return await ctrl.toggle_detection()

    async def stop_detection(self) -> bool:
        ctrl = self._ctrl()
        if ctrl is None or not ctrl.is_detecting:
            return True
        return await ctrl.toggle_detection()

    def _ctrl(self) -> DetectionController | None:
        return getattr(self, "_controller", None)

    # ── Controller wiring ──────────────────────────────────────

    def _wire_controller(self):
        ctrl = self._ctrl()
        if ctrl is None:
            return
        ctrl.on_status_changed = self._update_status
        ctrl.on_fps_changed = self._update_fps
        ctrl.on_camera_error = self._on_error
        ctrl.on_detection_toggled = self._on_detection_state_changed
        ctrl.on_preview_mode = self._on_preview_mode
        ctrl.on_frame_updated = self._on_frame_updated
        ctrl.on_camera_toggled = self._on_camera_state_changed

    def _create_camera_control(self) -> fc.Camera:
        return fc.Camera(
            expand=True,
            preview_enabled=True,
            content=ft.Container(
                alignment=ft.Alignment.CENTER,
                content=ft.Icon(
                    ft.Icons.CENTER_FOCUS_STRONG,
                    color=ft.Colors.WHITE_70,
                    size=48,
                ),
            ),
        )

    async def _ensure_camera_in_stack(self):
        camera = getattr(self, "_camera", None)
        stack = getattr(self, "_display_stack", None)
        ctrl = self._ctrl()
        if camera is not None or stack is None or ctrl is None:
            return
        camera = self._create_camera_control()
        object.__setattr__(self, "_camera", camera)
        ctrl.attach_camera(camera)
        stack.controls.insert(0, camera)
        stack.update()
        await asyncio.sleep(0.3)

    async def _remove_camera_from_stack(self):
        camera = getattr(self, "_camera", None)
        stack = getattr(self, "_display_stack", None)
        if camera is None or stack is None:
            return
        if camera in stack.controls:
            stack.controls.remove(camera)
        stack.update()
        ctrl = self._ctrl()
        if ctrl is not None:
            ctrl.detach_camera()
        object.__setattr__(self, "_camera", None)
        await asyncio.sleep(0.3)

    async def _on_toggle_camera(self, e: Any):
        ctrl = self._ctrl()
        if ctrl is None:
            return
        if ctrl.is_previewing:
            await self.stop_camera()
        else:
            await self.start_camera()

    async def _on_flip_camera(self, e: Any):
        ctrl = self._ctrl()
        if ctrl is None or not ctrl.use_flet_camera:
            return
        was_previewing = ctrl.is_previewing
        if was_previewing:
            await self.stop_camera()
        ctrl.lens_direction = (
            fc.CameraLensDirection.BACK if ctrl.lens_direction == fc.CameraLensDirection.FRONT
            else fc.CameraLensDirection.FRONT
        )
        if was_previewing:
            await self.start_camera()

    async def _on_toggle_detection(self, e: Any):
        await self.toggle_detection()

    def _update_status(self, message: str):
        status = getattr(self, "_status_text", None)
        if status:
            status.value = message
            status.update()

    def _update_fps(self, fps: float):
        fps_text = getattr(self, "_fps_text", None)
        if fps_text:
            fps_text.value = f"FPS: {fps:.1f}"
            fps_text.update()

    def _on_error(self, message: str):
        self._update_status(f"Error: {message}")

    def _on_preview_mode(self, is_previewing: bool):
        object.__setattr__(self, "_preview_active", is_previewing)
        ctrl = self._ctrl()
        display_image = getattr(self, "_display_image", None)
        if display_image is None:
            return
        if is_previewing and ctrl is not None and ctrl.use_flet_camera:
            display_image.src = _TRANSPARENT_1PX
            display_image.update()
        elif not is_previewing and display_image.src != _TRANSPARENT_1PX:
            display_image.src = _TRANSPARENT_1PX
            display_image.update()

    def _on_camera_state_changed(self, is_on: bool):
        camera_btn = getattr(self, "_camera_btn", None)
        toggle_btn = getattr(self, "_toggle_btn", None)
        if camera_btn:
            camera_btn.content = ft.Text("Stop Camera" if is_on else "Start Camera")
            camera_btn.icon = ft.Icons.VIDEOCAM_OFF if is_on else ft.Icons.VIDEOCAM
            camera_btn.update()
        if toggle_btn:
            toggle_btn.disabled = not is_on
            toggle_btn.update()

    def _on_detection_state_changed(self, is_running: bool):
        if is_running:
            object.__setattr__(self, "_frame_counter", 0)
        toggle_btn = getattr(self, "_toggle_btn", None)
        if toggle_btn:
            toggle_btn.content = ft.Text(
                "Stop Detection" if is_running else "Start Detection"
            )
            toggle_btn.icon = ft.Icons.STOP if is_running else ft.Icons.PLAY_ARROW
            toggle_btn.update()

    def _on_frame_updated(self, frame_bytes: bytes, result: DetectionResult):
        if result.error:
            self._update_status(f"Error: {result.error}")
            return
        n = getattr(self, "_frame_counter", 0) + 1
        object.__setattr__(self, "_frame_counter", n)

        display_image = getattr(self, "_display_image", None)
        if display_image is not None:
            display_image.src = frame_bytes
            display_image.update()

        ctrl = self._ctrl()
        if getattr(self, "_preview_active", False) and ctrl is not None and not ctrl.is_detecting:
            return

        self._update_status(f"Frame #{n} — {result.summary}")

        result_text = getattr(self, "_result_text", None)
        if result_text is not None:
            result_text.value = result.summary
            result_text.color = (
                ft.Colors.GREEN_400 if result.has_targets else ft.Colors.ORANGE_400
            )
            result_text.update()

        detection_list = getattr(self, "_detection_list", None)
        if detection_list is not None:
            if result.boxes:
                detection_list.controls = [
                    ft.Text(f"  {box.label}  [{box.width}x{box.height}px]", size=12)
                    for box in result.boxes
                ]
                detection_list.visible = True
            else:
                detection_list.controls = []
                detection_list.visible = False
            detection_list.update()

        cb = getattr(self, "_on_detection_result", None)
        if cb is not None:
            try:
                cb(result)
            except Exception:
                logger.exception("on_detection_result callback failed")
