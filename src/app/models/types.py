"""
Types module — Pure data structures for detection results and configuration.

No Flet or NCNN imports. Shared between Model, Controller, and the
ObjectDetectionCamera control.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class BoundingBox:
    """A single detected object's bounding box and classification."""
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def label(self) -> str:
        return f"{self.class_name} {self.confidence:.2f}"


@dataclass
class DetectionResult:
    """Complete result of a single frame's detection pass."""
    boxes: List[BoundingBox] = field(default_factory=list)
    class_counts: dict[str, int] = field(default_factory=dict)
    summary: str = ""
    error: str | None = None

    @property
    def object_count(self) -> int:
        return len(self.boxes)

    @property
    def has_targets(self) -> bool:
        return len(self.boxes) > 0


@dataclass
class DetectorConfig:
    """Configuration for the NCNN detector."""
    param_path: str
    bin_path: str
    use_vulkan: bool = False
    conf_threshold: float = 0.30
    nms_threshold: float = 0.45
    input_size: int = 640
    # Empty list = keep all COCO classes after threshold/NMS.
    # e.g. ["laptop"] or ["laptop", "cell phone"]
    target_classes: List[str] = field(default_factory=list)
