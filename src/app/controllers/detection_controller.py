"""
Detection Controller — Business logic and state management.

Camera backends:
  - flet-camera (web, Android, iOS): uses the browser/device camera via
    getUserMedia. Native preview + start_image_stream() for frame capture.
  - OpenCV cv2.VideoCapture (desktop): accesses the local camera directly
    from the Python process.

The backend is chosen by the caller (main.py) based on the platform.

Modes:
  - Preview: shows raw camera feed (no detection).
  - Detection: runs NCNN inference on each frame, shows annotated result.
"""

import asyncio
import time
import logging
from typing import Callable

import cv2

import flet_camera as fc

from app.models.detector import YOLOv8NCNNDetector
from app.models.types import DetectionResult

logger = logging.getLogger(__name__)

CAPTURE_INTERVAL = 0.10       # ~10 FPS target
POLL_INTERVAL = 0.25          # ~4 FPS for take_picture() polling (web)
MAX_CAMERA_INDEX = 4         # Try indices 0..MAX_CAMERA_INDEX-1


class DetectionController:
    def __init__(
        self,
        detector: YOLOv8NCNNDetector,
        use_flet_camera: bool = False,
    ):
        self.detector = detector
        self.use_flet_camera = use_flet_camera

        # flet-camera control (set by View via attach_camera())
        self.camera: fc.Camera | None = None

        # State
        self.is_previewing: bool = False
        self.is_detecting: bool = False
        self._cap: cv2.VideoCapture | None = None
        self._preview_task: asyncio.Task | None = None
        self._detection_task: asyncio.Task | None = None

        # flet-camera stream frame buffer
        self._latest_stream_bytes: bytes | None = None
        self._stream_frame_pending: bool = False
        self._stream_supported: bool = False

        # Performance tracking
        self.frame_count: int = 0
        self.fps: float = 0.0
        self._last_fps_time: float = time.time()

        # Latest results
        self.last_result: DetectionResult | None = None
        self.last_annotated_bytes: bytes | None = None

        # Callbacks (set by View)
        self.on_status_changed: Callable[[str], None] | None = None
        self.on_fps_changed: Callable[[float], None] | None = None
        self.on_frame_updated: Callable[[bytes, DetectionResult], None] | None = None
        self.on_camera_error: Callable[[str], None] | None = None
        self.on_detection_toggled: Callable[[bool], None] | None = None
        self.on_preview_mode: Callable[[bool], None] | None = None
        self.on_camera_toggled: Callable[[bool], None] | None = None

    # ── Public API: camera backend wiring ─────────────────────

    def attach_camera(self, camera: fc.Camera):
        """Attach the flet-camera control and wire its event handlers."""
        self.camera = camera
        camera.on_stream_image = self._on_stream_image
        camera.on_state_change = self._on_camera_state_change

    def detach_camera(self):
        """Clear the flet-camera reference (called when View removes it from the Stack).

        This prevents the detection loop from calling take_picture() on a
        control that has been disposed by Flutter.
        """
        self.camera = None

    # ── Public API: camera lifecycle ──────────────────────────

    async def start_camera(self) -> bool:
        """Open the camera and show a raw preview feed."""
        if self.use_flet_camera:
            return await self._start_flet_camera()
        return await self._start_opencv_camera()

    async def stop_camera(self) -> bool:
        """Close the camera and stop any active preview/detection."""
        if self.is_detecting:
            await self._stop_detection()

        if self.use_flet_camera:
            return await self._stop_flet_camera()
        return await self._stop_opencv_camera()

    async def toggle_detection(self) -> bool:
        if self.is_detecting:
            return await self._stop_detection()
        return await self._start_detection()

    # ── flet-camera backend ────────────────────────────────────

    async def _start_flet_camera(self) -> bool:
        if self.camera is None:
            self._emit_error("flet-camera control not attached.")
            return False

        self._emit_status("Listing cameras...")
        try:
            cameras = await self.camera.get_available_cameras()
        except Exception as e:
            self._emit_error(f"Failed to list cameras: {e}")
            return False

        if not cameras:
            self._emit_error("No cameras found.")
            return False

        selected = self._select_camera(cameras)
        self._emit_status(f"Initializing camera: {selected.name}...")

        try:
            await self.camera.initialize(
                description=selected,
                resolution_preset=fc.ResolutionPreset.MEDIUM,
                enable_audio=False,
                image_format_group=fc.ImageFormatGroup.JPEG,
            )
        except Exception as e:
            self._emit_error(f"Camera init failed: {e}")
            return False

        self.is_previewing = True
        if self.on_preview_mode:
            self.on_preview_mode(True)
        if self.on_camera_toggled:
            self.on_camera_toggled(True)
        self._emit_status("Camera preview active.")
        return True

    async def _stop_flet_camera(self) -> bool:
        if self.camera is None:
            return False
        try:
            if self.is_detecting:
                await self.camera.stop_image_stream()
            await self.camera.pause_preview()
        except Exception as e:
            logger.warning("Error stopping flet-camera: %s", e)

        self.is_previewing = False
        self._latest_stream_bytes = None
        self._stream_frame_pending = False
        if self.on_preview_mode:
            self.on_preview_mode(False)
        if self.on_camera_toggled:
            self.on_camera_toggled(False)
        self._emit_status("Camera stopped.")
        return True

    def _select_camera(self, cameras: list) -> fc.CameraDescription:
        """Prefer a front-facing camera; fall back to the first available."""
        for cam in cameras:
            if cam.lens_direction == fc.CameraLensDirection.FRONT:
                return cam
        return cameras[0]

    def _on_stream_image(self, e: fc.CameraImageEvent):
        """Lightweight handler: just stash the latest JPEG frame."""
        self._latest_stream_bytes = e.bytes
        self._stream_frame_pending = True

    async def _on_camera_state_change(self, e: fc.CameraStateEvent):
        if e.has_error:
            self._emit_error(f"Camera error: {e.error_description}")
        elif e.is_streaming_images:
            self._emit_status("Streaming images...")

    # ── OpenCV backend ─────────────────────────────────────────

    async def _start_opencv_camera(self) -> bool:
        if not self._open_camera():
            return False
        self.is_previewing = True
        if self.on_preview_mode:
            self.on_preview_mode(True)
        if self.on_camera_toggled:
            self.on_camera_toggled(True)
        self._preview_task = asyncio.create_task(self._preview_loop())
        self._emit_status("Camera preview active.")
        return True

    async def _stop_opencv_camera(self) -> bool:
        self.is_previewing = False
        if self._preview_task and not self._preview_task.done():
            self._preview_task.cancel()
            try:
                await self._preview_task
            except asyncio.CancelledError:
                pass
        self._preview_task = None
        self._close_camera()
        if self.on_preview_mode:
            self.on_preview_mode(False)
        if self.on_camera_toggled:
            self.on_camera_toggled(False)
        self._emit_status("Camera stopped.")
        return True

    def _open_camera(self) -> bool:
        """Try camera indices 0..MAX_CAMERA_INDEX-1."""
        self._emit_status("Opening camera via OpenCV...")
        for idx in range(MAX_CAMERA_INDEX):
            try:
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self._cap = cap
                    logger.info("OpenCV camera opened at index %d", idx)
                    return True
                cap.release()
            except Exception:
                continue
        self._emit_error("Could not open any camera via OpenCV. Is a camera connected?")
        return False

    def _close_camera(self):
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    # ── Preview loop (OpenCV raw feed, no detection) ───────────

    async def _preview_loop(self):
        """Read frames and display raw (no detection). Cancelled when detection starts."""
        frame_count = 0
        while self.is_previewing and not self.is_detecting:
            if self._cap is None or not self._cap.isOpened():
                await asyncio.sleep(CAPTURE_INTERVAL)
                continue

            ret, frame = self._cap.read()
            if not ret:
                await asyncio.sleep(CAPTURE_INTERVAL)
                continue

            try:
                _, jpeg_bytes = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                raw_result = DetectionResult(summary="Preview — no detection running")
                if self.on_frame_updated:
                    self.on_frame_updated(jpeg_bytes.tobytes(), raw_result)
                frame_count += 1
                if frame_count % 30 == 0:
                    logger.info("Preview loop: %d frames sent", frame_count)
            except Exception as e:
                logger.warning("Preview encode/send error: %s", e)

            await asyncio.sleep(CAPTURE_INTERVAL)

    # ── Detection loop ──────────────────────────────────────────

    async def _start_detection(self) -> bool:
        """Ensure camera is open and start the detection loop."""
        if self.use_flet_camera:
            return await self._start_flet_detection()
        return await self._start_opencv_detection()

    async def _start_flet_detection(self) -> bool:
        if self.camera is None or not self.is_previewing:
            self._emit_error("Start the camera first.")
            return False

        try:
            self._stream_supported = await self.camera.supports_image_streaming()
        except Exception as e:
            self._emit_error(f"Cannot check streaming support: {e}")
            return False

        if self._stream_supported:
            self._latest_stream_bytes = None
            self._stream_frame_pending = False
            try:
                await self.camera.start_image_stream()
            except Exception as e:
                self._emit_error(f"Failed to start image stream: {e}")
                return False
        else:
            self._emit_status("Streaming unsupported; using take_picture() polling...")

        self.is_detecting = True
        self.frame_count = 0
        self._last_fps_time = time.time()
        if self.on_detection_toggled:
            self.on_detection_toggled(True)
        self._detection_task = asyncio.create_task(self._flet_detection_loop())
        mode = "image stream" if self._stream_supported else "take_picture polling"
        self._emit_status(f"Detection running (flet-camera, {mode})...")
        return True

    async def _start_opencv_detection(self) -> bool:
        if not self.is_previewing or self._cap is None or not self._cap.isOpened():
            self._emit_error("Start the camera first.")
            return False

        self.is_detecting = True
        self.frame_count = 0
        self._last_fps_time = time.time()
        if self.on_detection_toggled:
            self.on_detection_toggled(True)
        self._detection_task = asyncio.create_task(self._opencv_detection_loop())
        self._emit_status("Detection running (OpenCV)...")
        return True

    async def _stop_detection(self) -> bool:
        self.is_detecting = False

        if self._detection_task and not self._detection_task.done():
            self._detection_task.cancel()
            try:
                await self._detection_task
            except asyncio.CancelledError:
                pass
        self._detection_task = None

        if self.on_detection_toggled:
            self.on_detection_toggled(False)

        if self.use_flet_camera:
            if self.camera and self.is_previewing and self._stream_supported:
                try:
                    await self.camera.stop_image_stream()
                except Exception as e:
                    logger.warning("Error stopping image stream: %s", e)
            self._emit_status("Preview resumed." if self.is_previewing else "Detection stopped.")
            return True

        # OpenCV: resume preview if camera still open
        if self._cap and self._cap.isOpened():
            self.is_previewing = True
            if self.on_preview_mode:
                self.on_preview_mode(True)
            self._preview_task = asyncio.create_task(self._preview_loop())
            self._emit_status("Preview resumed.")
        else:
            self._close_camera()
            self._emit_status("Detection stopped.")
        return True

    async def _flet_detection_loop(self):
        """Capture frames → detect → render → push to view.

        Uses start_image_stream() when supported (Android/iOS), or falls back
        to take_picture() polling on web (streaming is unsupported there).
        """
        while self.is_detecting:
            loop_start = time.time()

            if self.camera is None:
                break

            if self._stream_supported:
                if not self._stream_frame_pending or self._latest_stream_bytes is None:
                    await asyncio.sleep(0.03)
                    continue
                self._stream_frame_pending = False
                frame_bytes = self._latest_stream_bytes
            else:
                try:
                    frame_bytes = await self.camera.take_picture()
                except Exception as e:
                    logger.warning("take_picture() failed: %s", e)
                    await asyncio.sleep(POLL_INTERVAL)
                    continue

            try:
                annotated_bytes, result = self.detector.detect_and_render_to_bytes(frame_bytes)

                self.last_result = result
                self.last_annotated_bytes = annotated_bytes

                self.frame_count += 1
                elapsed = time.time() - self._last_fps_time
                if elapsed >= 1.0:
                    self.fps = self.frame_count / elapsed
                    self.frame_count = 0
                    self._last_fps_time = time.time()
                    if self.on_fps_changed:
                        self.on_fps_changed(round(self.fps, 1))

                if self.on_frame_updated:
                    self.on_frame_updated(annotated_bytes, result)

            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Detection error on frame")

            elapsed = time.time() - loop_start
            interval = CAPTURE_INTERVAL if self._stream_supported else POLL_INTERVAL
            sleep_time = max(0.0, interval - elapsed)
            if sleep_time > 0:
                try:
                    await asyncio.sleep(sleep_time)
                except asyncio.CancelledError:
                    break

    async def _opencv_detection_loop(self):
        """Read frame → detect → render → push to view."""
        while self.is_detecting:
            loop_start = time.time()

            if self._cap is None or not self._cap.isOpened():
                await asyncio.sleep(CAPTURE_INTERVAL)
                continue

            ret, frame = self._cap.read()
            if not ret:
                await asyncio.sleep(CAPTURE_INTERVAL)
                continue

            try:
                _, jpeg_bytes = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                jpeg_bytes = jpeg_bytes.tobytes()

                annotated_bytes, result = self.detector.detect_and_render_to_bytes(jpeg_bytes)

                self.last_result = result
                self.last_annotated_bytes = annotated_bytes

                self.frame_count += 1
                elapsed = time.time() - self._last_fps_time
                if elapsed >= 1.0:
                    self.fps = self.frame_count / elapsed
                    self.frame_count = 0
                    self._last_fps_time = time.time()
                    if self.on_fps_changed:
                        self.on_fps_changed(round(self.fps, 1))

                if self.on_frame_updated:
                    self.on_frame_updated(annotated_bytes, result)

            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Detection error on frame")

            elapsed = time.time() - loop_start
            sleep_time = max(0.0, CAPTURE_INTERVAL - elapsed)
            if sleep_time > 0:
                try:
                    await asyncio.sleep(sleep_time)
                except asyncio.CancelledError:
                    break

    # ── Internal helpers ─────────────────────────────────────

    def _emit_status(self, message: str):
        if self.on_status_changed:
            self.on_status_changed(message)

    def _emit_error(self, message: str):
        logger.error(message)
        if self.on_camera_error:
            self.on_camera_error(message)
        elif self.on_status_changed:
            self.on_status_changed(message)
