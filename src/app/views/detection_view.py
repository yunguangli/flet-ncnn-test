"""
Detection View — Flet UI controls and layout.

Display strategy:
  - A display area shows the camera feed at all times once connected.
  - When detection is OFF the raw camera feed is displayed.
  - When detection is ON the annotated (processed) image is displayed.
  - The "Start Detection" button toggles detection on/off.
"""

import base64

import flet as ft

from app.controllers.detection_controller import DetectionController
from app.models.types import DetectionResult

DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480


class DetectionView:
    def __init__(self, controller: DetectionController):
        self.controller = controller

        self.page: ft.Page | None = None
        self.display_image: ft.Image | None = None
        self.toggle_btn: ft.FilledButton | None = None
        self.status_text: ft.Text | None = None
        self.fps_text: ft.Text | None = None
        self.result_text: ft.Text | None = None
        self.detection_list: ft.Column | None = None

        self._preview_active = False

    def build(self, page: ft.Page) -> ft.Control:
        self.page = page
        page.title = "NCNN Real-time Detection"
        page.theme_mode = ft.ThemeMode.DARK
        page.horizontal_alignment = ft.CrossAxisAlignment.STRETCH
        page.padding = 16
        page.scroll = ft.ScrollMode.AUTO

        # ── Image display (shows raw feed or annotated feed) ──
        _TRANSPARENT_1PX = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwC"
            "AAAAC0lEQVR42mP8/wIAAgMBAp0YVwAAAABJRU5ErkJggg=="
        )
        self.display_image = ft.Image(
            src=_TRANSPARENT_1PX,
            width=DISPLAY_WIDTH,
            height=DISPLAY_HEIGHT,
            fit=ft.BoxFit.CONTAIN,
            gapless_playback=True,
        )

        display_area = ft.Container(
            content=self.display_image,
            width=DISPLAY_WIDTH,
            height=DISPLAY_HEIGHT,
            bgcolor=ft.Colors.BLACK,
            border_radius=ft.BorderRadius.all(12),
            alignment=ft.Alignment.CENTER,
        )

        # ── Toggle button ───────────────────────────────────────
        self.toggle_btn = ft.FilledButton(
            content=ft.Text("Start Detection"),
            icon=ft.Icons.PLAY_ARROW,
            on_click=self._on_toggle_detection,
            style=ft.ButtonStyle(
                color=ft.Colors.WHITE,
                bgcolor=ft.Colors.GREEN_700,
                padding=ft.Padding.symmetric(horizontal=32, vertical=16),
                text_style=ft.TextStyle(size=18, weight=ft.FontWeight.BOLD),
            ),
        )

        # ── Status / FPS / Results ──────────────────────────────
        self.status_text = ft.Text(
            "Opening camera...", size=14, color=ft.Colors.GREY_400,
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
                        controls=[self.toggle_btn, self.fps_text],
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

    # ── Event handlers (user → controller) ─────────────────────

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

    def _on_detection_state_changed(self, is_running: bool):
        self._frame_counter = 0 if is_running else self._frame_counter
        self.toggle_btn.content = ft.Text(
            "Stop Detection" if is_running else "Start Detection"
        )
        self.toggle_btn.icon = ft.Icons.STOP if is_running else ft.Icons.PLAY_ARROW
        self.toggle_btn.update()

    _frame_counter: int = 0

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
