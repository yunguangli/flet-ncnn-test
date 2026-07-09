"""
Detector model — Wraps NCNN inference engine for object detection.

This is the pure Model layer: no Flet imports, no UI logic.
It handles:
  - Loading NCNN model (param + bin)
  - Image preprocessing (resize, normalize)
  - Forward pass inference
  - Post-processing (NMS, scale to original size)
  - Spatial "holding" analysis via bounding box overlap
  - Both file-path and in-memory (bytes) frame input
"""

import cv2
import numpy as np
import ncnn

from .types import BoundingBox, DetectionResult, DetectorConfig


class YOLOv8NCNNDetector:
    """NCNN-based YOLOv8 object detector with spatial holding analysis.

    Uses Tencent's NCNN inference framework for fast on-device inference.
    The "holding" detection works by checking bounding box overlap between
    "person" detections and other objects.
    """

    # Standard COCO 80-class labels mapped to YOLOv8 output indices
    COCO_CLASS_NAMES = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
        "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush",
    ]

    def __init__(self, config: DetectorConfig):
        """Initialize the NCNN network with a given config.

        Args:
            config: DetectorConfig with param/bin paths and settings.
        """
        self.config = config
        self.class_names = self.COCO_CLASS_NAMES

        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = config.use_vulkan

        if self.net.load_param(config.param_path) != 0:
            raise FileNotFoundError(
                f"Failed to load NCNN network structure: {config.param_path}"
            )
        if self.net.load_model(config.bin_path) != 0:
            raise FileNotFoundError(
                f"Failed to load NCNN network weights: {config.bin_path}"
            )

    # ── Public API ──────────────────────────────────────────────

    def detect_from_file(self, image_path: str) -> DetectionResult:
        """Run detection on an image loaded from disk.

        Args:
            image_path: Path to the image file.

        Returns:
            DetectionResult with bounding boxes and summary.
        """
        bgr_img = cv2.imread(image_path)
        if bgr_img is None:
            return DetectionResult(error=f"Cannot read image: {image_path}")
        return self._process_mat(bgr_img)

    def detect_from_bytes(self, image_bytes: bytes) -> DetectionResult:
        """Run detection on an in-memory JPEG/PNG image (e.g. from camera stream).

        Args:
            image_bytes: Raw encoded image bytes (JPEG, PNG, etc.).

        Returns:
            DetectionResult with bounding boxes and summary.
        """
        np_arr = np.frombuffer(image_bytes, np.uint8)
        bgr_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if bgr_img is None:
            return DetectionResult(error="Failed to decode image bytes")
        return self._process_mat(bgr_img)

    def detect_and_render_to_bytes(self, image_bytes: bytes) -> tuple[bytes, DetectionResult]:
        """Run detection and return annotated JPEG bytes + result.

        This is the main entry point for the camera pipeline.
        The annotated image has bounding boxes drawn directly on it.

        Args:
            image_bytes: Raw encoded image bytes (JPEG from camera).

        Returns:
            Tuple of (annotated JPEG bytes, DetectionResult).
        """
        result = self.detect_from_bytes(image_bytes)
        if result.error:
            return image_bytes, result

        # Decode again for rendering (we already have the mat in _process_mat,
        # but for clean API we re-decode — can optimize later)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        bgr_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self._render_boxes(bgr_img, result)

        _, buffer = cv2.imencode(".jpg", bgr_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buffer.tobytes(), result

    # ── Internal processing pipeline ────────────────────────────

    def _process_mat(self, bgr_img: np.ndarray) -> DetectionResult:
        """Full detection pipeline on a decoded BGR OpenCV matrix."""
        orig_h, orig_w = bgr_img.shape[:2]
        input_size = self.config.input_size

        # 1. Preprocess: resize to network input size + normalize
        ncnn_img = ncnn.Mat.from_pixels_resize(
            bgr_img, ncnn.Mat.PixelType.PIXEL_BGR2RGB,
            orig_w, orig_h, input_size, input_size,
        )
        mean_vals = [0.0, 0.0, 0.0]
        norm_vals = [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0]
        ncnn_img.substract_mean_normalize(mean_vals, norm_vals)

        # 2. Run inference
        ex = self.net.create_extractor()
        ex.input("in0", ncnn_img)
        mat_out = ncnn.Mat()
        ex.extract("out0", mat_out)

        out_array = np.array(mat_out)
        cx, cy, nw, nh = out_array[0, :], out_array[1, :], out_array[2, :], out_array[3, :]
        classes_conf = out_array[4:, :]

        best_class_indices = np.argmax(classes_conf, axis=0)
        best_confidences = np.max(classes_conf, axis=0)

        # 3. Filter by confidence threshold
        valid_mask = best_confidences > self.config.conf_threshold

        x1 = cx - nw / 2
        y1 = cy - nh / 2
        x2 = cx + nw / 2
        y2 = cy + nh / 2

        boxes = np.stack([x1, y1, x2, y2], axis=1)[valid_mask]
        scores = best_confidences[valid_mask]
        class_ids = best_class_indices[valid_mask]

        if len(boxes) == 0:
            return DetectionResult(summary="No objects detected.")

        # 4. Apply Non-Maximum Suppression
        keep_indices = self._nms(boxes, scores, self.config.nms_threshold)

        # 5. Scale boxes back to original image size
        scale_x = orig_w / input_size
        scale_y = orig_h / input_size

        person_boxes: list[tuple[int, int, int, int]] = []
        all_boxes: list[BoundingBox] = []

        for idx in keep_indices:
            class_id = class_ids[idx]
            name = self.class_names[class_id]
            conf = float(scores[idx])

            bx1 = int(boxes[idx, 0] * scale_x)
            by1 = int(boxes[idx, 1] * scale_y)
            bx2 = int(boxes[idx, 2] * scale_x)
            by2 = int(boxes[idx, 3] * scale_y)

            box = BoundingBox(
                class_name=name, confidence=conf,
                x1=bx1, y1=by1, x2=bx2, y2=by2,
            )

            if name == "person":
                person_boxes.append((bx1, by1, bx2, by2))

            all_boxes.append(box)

        # 6. Spatial holding analysis: check overlap with person boxes
        held_items: list[str] = []
        for box in all_boxes:
            if box.class_name == "person":
                continue
            for (px1, py1, px2, py2) in person_boxes:
                horizontal_overlap = (box.x1 < px2) and (box.x2 > px1)
                vertical_overlap = (box.y1 < py2) and (box.y2 > py1)
                if horizontal_overlap and vertical_overlap:
                    box.is_held = True
                    held_items.append(box.class_name)
                    break

        # 7. Build summary string
        if person_boxes:
            if held_items:
                summary = f"Person is holding: {', '.join(set(held_items))}"
            else:
                summary = "Person is not holding anything recognized."
        else:
            summary = "No person detected in the frame."

        # Add object count to summary for FPS context
        summary += f" ({len(all_boxes)} objects)"

        return DetectionResult(
            boxes=all_boxes,
            person_count=len(person_boxes),
            held_items=list(set(held_items)),
            summary=summary,
        )

    def _render_boxes(self, bgr_img: np.ndarray, result: DetectionResult) -> None:
        """Draw bounding boxes and labels directly onto the image (in-place)."""
        for box in result.boxes:
            if box.class_name == "person":
                color = (255, 0, 0)  # Blue for person
            elif box.is_held:
                color = (0, 255, 0)  # Green for held objects
            else:
                color = (0, 165, 255)  # Orange for other objects

            cv2.rectangle(bgr_img, (box.x1, box.y1), (box.x2, box.y2), color, 2)

            label = box.label
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                bgr_img,
                (box.x1, box.y1 - text_h - 4),
                (box.x1 + text_w, box.y1),
                color,
                -1,  # filled
            )
            cv2.putText(
                bgr_img, label,
                (box.x1, box.y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 1, cv2.LINE_AA,
            )

    @staticmethod
    def _nms(
        boxes: np.ndarray, scores: np.ndarray, nms_threshold: float,
    ) -> list[int]:
        """Non-Maximum Suppression: remove duplicate overlapping boxes.

        Args:
            boxes: (N, 4) array of [x1, y1, x2, y2] in normalized coordinates.
            scores: (N,) array of confidence scores.
            nms_threshold: IoU threshold — boxes with overlap above this are removed.

        Returns:
            List of indices to keep.
        """
        keep = []
        if len(boxes) == 0:
            return keep

        order = np.argsort(scores)[::-1]
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break

            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

            inter_w = np.maximum(0.0, xx2 - xx1)
            inter_h = np.maximum(0.0, yy2 - yy1)
            inter_area = inter_w * inter_h

            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_others = (
                (boxes[order[1:], 2] - boxes[order[1:], 0])
                * (boxes[order[1:], 3] - boxes[order[1:], 1])
            )
            union_area = area_i + area_others - inter_area

            iou = inter_area / union_area
            inds = np.where(iou <= nms_threshold)[0]
            order = order[inds + 1]

        return keep
