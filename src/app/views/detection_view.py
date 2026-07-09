"""
Detection View — Flet UI controls and layout.

Display strategy:
  - flet-camera backend: a Stack layers the native Camera preview (bottom)
    and an Image overlay (top). During preview the overlay is transparent
    so the native camera feed shows through; during detection the overlay
    shows annotated frames.
  - OpenCV backend: the Image control alone shows raw frames (preview) or
    annotated frames (detection).

Camera control lifecycle (flet-camera):
  The Camera control is created and added to the Stack when the user clicks
  "Start Camera", and removed when they click "Stop Camera". Removing it
  from the page tree triggers Flutter-side disposal of the camera controller,
  which releases the hardware (camera light turns off). pause_preview() alone
  does NOT release the hardware.

Buttons:
  - Start/Stop Camera: opens/closes the camera (flet-camera or OpenCV).
  - Start/Stop Detection: toggles NCNN inference on the live feed.
"""

import asyncio
import base64

import flet as ft
import flet_camera as fc

from app.controllers.detection_controller import DetectionController
from app.models.types import DetectionResult

DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

_TRANSPARENT_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwC"
    "AAAAC0lEQVR42mP8/wIAAgMBAp0YVwAAAABJRU5ErkJggg=="
)

_PLACEHOLDER_ICON = ft.Icon(
    ft.Icons.VIDEOCAM_OFF,
    color=ft.Colors.WHITE_70,
    size=48,
)


class DetectionView:
    def __init__(self, controller: DetectionController):
        self.controller = controller

        self.page: ft.Page | None = None
        self.camera: fc.Camera | None = None
        self.display_stack: ft.Stack | None = None
        self.display_image: ft.Image | None = None
        self.camera_btn: ft.FilledButton | None = None
        self.toggle_btn: ft.FilledButton | None = None
        self.status_text: ft.Text | None = None
        self.fps_text: ft.Text | None = None
        self.result_text: ft.Text | None = None
        self.detection_list: ft.Column | None = None

        self._preview_active = False
        self._frame_counter: int = 0

    def build(self, page: ft.Page) -> ft.Control:
        self.page = page
        page.title = "NCNN Real-time Detection"
        page.theme_mode = ft.ThemeMode.DARK
        page.horizontal_alignment = ft.CrossAxisAlignment.STRETCH
        page.padding = 16
        page.scroll = ft.ScrollMode.AUTO

        # ── Image overlay (annotated frames / raw OpenCV frames) ──
        self.display_image = ft.Image(
            src=_TRANSPARENT_1PX,
            width=DISPLAY_WIDTH,
            height=DISPLAY_HEIGHT,
            fit=ft.BoxFit.CONTAIN,
            gapless_playback=True,
        )

        # ── Stack: camera preview (bottom) + image overlay (top) ──
        # The Camera control is added/removed dynamically (see _ensure_camera_in_stack).
        self.display_stack = ft.Stack(
            controls=[self.display_image],
            width=DISPLAY_WIDTH,
            height=DISPLAY_HEIGHT,
        )

        display_area = ft.Container(
            content=self.display_stack,
            width=DISPLAY_WIDTH,
            height=DISPLAY_HEIGHT,
            bgcolor=ft.Colors.BLACK,
            border_radius=ft.BorderRadius.all(12),
            alignment=ft.Alignment.CENTER,
        )

        # ── Camera button ───────────────────────────────────────
        self.camera_btn = ft.FilledButton(
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

        # ── Detection toggle button ────────────────────────────
        self.toggle_btn = ft.FilledButton(
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

        # ── Status / FPS / Results ──────────────────────────────
        self.status_text = ft.Text(
            "Click \"Start Camera\" to begin.", size=14, color=ft.Colors.GREY_400,
        )
        self.fps_text = ft.Text(
            "FPS: --", size=16, weight=ft.FontWeight.BOLD, color=ft.Colors.GREEN_400,
        )
        self.result_text = ft.Text(size=14, selectable=True)

        self.detection_list = ft.Column(
            spacing=2, visible=False, height=150, scroll=ft.ScrollMode.AUTO,
        )

        # ── Layout assembly ─────────────────────────────────────
        root = ft.SafeArea(
            content=ft.Column(
                spacing=12,
                controls=[
                    ft.AppBar(
                        title=ft.Text("NCNN Real-time Detection"),
                        bgcolor=ft.Colors.BLUE_GREY_900,
                    ),
                    ft.Row(
                        controls=[self.camera_btn, self.toggle_btn, self.fps_text],
                        alignment=ft.MainAxisAlignment.CENTER,
                        wrap=True,
                    ),
                    ft.Row(
                        controls=[display_area],
                        alignment=ft.MainAxisAlignment.CENTER,
                    ),
                    self.status_text,
                    ft.Divider(height=1),
                    ft.Text("Detection Results:", size=16, weight=ft.FontWeight.BOLD),
                    self.result_text,
                    self.detection_list,
                ],
            ),
        )

        self._wire_callbacks()
        return root

    # ── Wire controller → UI ───────────────────────────────────

    def _wire_callbacks(self):
        ctrl = self.controller
        ctrl.on_status_changed = self._update_status
        ctrl.on_fps_changed = self._update_fps
        ctrl.on_camera_error = self._on_error
        ctrl.on_detection_toggled = self._on_detection_state_changed
        ctrl.on_preview_mode = self._on_preview_mode
        ctrl.on_frame_updated = self._on_frame_updated
        ctrl.on_camera_toggled = self._on_camera_state_changed

    # ── flet-camera control lifecycle ─────────────────────────

    def _create_camera_control(self) -> fc.Camera:
        """Create a fresh flet-camera control (not yet mounted)."""
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
        """Create and mount the Camera control if not already present."""
        if self.camera is not None or self.display_stack is None:
            return

        self.camera = self._create_camera_control()
        self.controller.attach_camera(self.camera)
        self.display_stack.controls.insert(0, self.camera)
        self.display_stack.update()
        await asyncio.sleep(0.3)

    async def _remove_camera_from_stack(self):
        """Remove the Camera control from the Stack to release hardware."""
        if self.camera is None or self.display_stack is None:
            return

        if self.camera in self.display_stack.controls:
            self.display_stack.controls.remove(self.camera)
        self.display_stack.update()
        self.camera = None
        await asyncio.sleep(0.3)

    # ── Event handlers (user → controller) ─────────────────────

    async def _on_toggle_camera(self, e):
        if self.controller.is_previewing:
            await self.controller.stop_camera()
            if self.controller.use_flet_camera:
                await self._remove_camera_from_stack()
        else:
            if self.controller.use_flet_camera:
                await self._ensure_camera_in_stack()
            success = await self.controller.start_camera()
            if not success and self.controller.use_flet_camera:
                await self._remove_camera_from_stack()

    async def _on_toggle_detection(self, e):
        await self.controller.toggle_detection()

    # ── Controller callbacks (controller → UI update) ──────────

    def _update_status(self, message: str):
        if self.status_text:
            self.status_text.value = message
            self.status_text.update()

    def _update_fps(self, fps: float):
        if self.fps_text:
            self.fps_text.value = f"FPS: {fps:.1f}"
            self.fps_text.update()

    def _on_error(self, message: str):
        self._update_status(f"Error: {message}")

    def _on_preview_mode(self, is_previewing: bool):
        self._preview_active = is_previewing
        if is_previewing and self.controller.use_flet_camera and self.display_image:
            self.display_image.src = _TRANSPARENT_1PX
            self.display_image.update()

    def _on_camera_state_changed(self, is_on: bool):
        if self.camera_btn:
            self.camera_btn.content = ft.Text("Stop Camera" if is_on else "Start Camera")
            self.camera_btn.icon = ft.Icons.VIDEOCAM_OFF if is_on else ft.Icons.VIDEOCAM
            self.camera_btn.update()
        if self.toggle_btn:
            self.toggle_btn.disabled = not is_on
            self.toggle_btn.update()

    def _on_detection_state_changed(self, is_running: bool):
        self._frame_counter = 0 if is_running else self._frame_counter
        if self.toggle_btn:
            self.toggle_btn.content = ft.Text(
                "Stop Detection" if is_running else "Start Detection"
            )
            self.toggle_btn.icon = ft.Icons.STOP if is_running else ft.Icons.PLAY_ARROW
            self.toggle_btn.update()

    def _on_frame_updated(self, frame_bytes: bytes, result: DetectionResult):
        if result.error:
            self._update_status(f"Error: {result.error}")
            return
        self._frame_counter += 1

        if self.display_image:
            self.display_image.src = frame_bytes
            self.display_image.update()

        if self._preview_active and not self.controller.is_detecting:
            return

        heartbeat_msg = f"Frame #{self._frame_counter} — {result.summary}"
        self._update_status(heartbeat_msg)

        if self.result_text:
            self.result_text.value = result.summary
            self.result_text.color = (
                ft.Colors.GREEN_400 if result.has_person else ft.Colors.ORANGE_400
            )
            self.result_text.update()

        if self.detection_list and result.boxes:
            rows = []
            for box in result.boxes:
                icon = "🟢" if box.is_held else "🟠" if box.class_name == "person" else "🔵"
                rows.append(
                    ft.Text(f"{icon} {box.label}  [{box.width}x{box.height}px]", size=12)
                )
            self.detection_list.controls = rows
            self.detection_list.visible = True
            self.detection_list.update()
