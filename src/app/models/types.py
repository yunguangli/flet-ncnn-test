"""
Types module — Pure data structures for detection results and configuration.

This module contains only dataclasses and enums. No Flet or NCNN imports.
These types are shared between the Model, Controller, and View layers.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class BoundingBox:
    """A single detected object's bounding box and classification."""
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    is_held: bool = False

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def label(self) -> str:
        status = " [HELD]" if self.is_held else ""
        return f"{self.class_name} {self.confidence:.2f}{status}"


@dataclass
class DetectionResult:
    """Complete result of a single frame's detection pass."""
    boxes: List[BoundingBox] = field(default_factory=list)
    person_count: int = 0
    held_items: List[str] = field(default_factory=list)
    summary: str = ""
    error: str | None = None

    @property
    def has_person(self) -> bool:
        return self.person_count > 0

    @property
    def object_count(self) -> int:
        return len(self.boxes)


@dataclass
class DetectorConfig:
    """Configuration for the NCNN detector."""
    param_path: str
    bin_path: str
    use_vulkan: bool = False
    conf_threshold: float = 0.30
    nms_threshold: float = 0.45
    input_size: int = 640


@dataclass
class CameraConfig:
    """Configuration for the camera capture."""
    resolution_preset: str = "MEDIUM"  # LOW, MEDIUM, HIGH, VERY_HIGH, ULTRA_HIGH, MAX
    fps: int | None = 15  # Target FPS for camera stream
    use_front_camera: bool = False
