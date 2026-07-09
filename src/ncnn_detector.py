import ncnn
import cv2
import numpy as np

class YOLOv8NCNNDetector:
    def __init__(self, param_path: str, bin_path: str, use_vulkan: bool = False):
        """Initializes and loads the NCNN network."""
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = use_vulkan
        
        if self.net.load_param(param_path) != 0:
            raise FileNotFoundError(f"Failed to load NCNN network structure: {param_path}")
        if self.net.load_model(bin_path) != 0:
            raise FileNotFoundError(f"Failed to load NCNN network weights: {bin_path}")

        # Standard COCO dataset labels mapped to YOLOv8 indices
        self.class_names = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
            "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
            "scissors", "teddy bear", "hair drier", "toothbrush"
        ]

    def _nms(self, boxes: np.ndarray, scores: np.ndarray, nms_threshold: float) -> list:
        """Executes Non-Maximum Suppression to remove duplicates."""
        keep_indices = []
        if len(boxes) == 0:
            return keep_indices

        order = np.argsort(scores)[::-1]
        while order.size > 0:
            i = order[0]
            keep_indices.append(i)
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
            area_others = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
            union_area = area_i + area_others - inter_area
            
            iou = inter_area / union_area
            inds = np.where(iou <= nms_threshold)[0]
            order = order[inds + 1]
            
        return keep_indices

    def detect_and_render(self, img_path: str, conf_thresh: float = 0.10, nms_thresh: float = 0.45) -> tuple:
        """Processes the image, extracts detections, runs spatial logic, and redraws frames."""
        bgr_img = cv2.imread(img_path)
        if bgr_img is None:
            raise FileNotFoundError(f"Source file not readable at {img_path}")

        orig_h, orig_w, _ = bgr_img.shape
        
        # 1. Transform pixels to scaled 640x640 matrix tensors
        ncnn_img = ncnn.Mat.from_pixels_resize(
            bgr_img, ncnn.Mat.PixelType.PIXEL_BGR2RGB, orig_w, orig_h, 640, 640
        )
        mean_vals = [0.0, 0.0, 0.0]
        norm_vals = [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0]
        ncnn_img.substract_mean_normalize(mean_vals, norm_vals)

        # 2. Network Extractor forward pass execution
        ex = self.net.create_extractor()
        ex.input("in0", ncnn_img)
        mat_out = ncnn.Mat()
        ex.extract("out0", mat_out)
        
        out_array = np.array(mat_out)
        cx, cy, nw, nh = out_array[0, :], out_array[1, :], out_array[2, :], out_array[3, :]
        classes_conf = out_array[4:, :]
        
        best_class_indices = np.argmax(classes_conf, axis=0)
        best_confidences = np.max(classes_conf, axis=0)
        
        # Filter array based on threshold configuration
        valid_mask = best_confidences > conf_thresh
        
        x1 = cx - nw / 2
        y1 = cy - nh / 2
        x2 = cx + nw / 2
        y2 = cy + nh / 2
        
        boxes = np.stack([x1, y1, x2, y2], axis=1)[valid_mask]
        scores = best_confidences[valid_mask]
        class_ids = best_class_indices[valid_mask]
        
        # 3. Apply NMS Filtering
        keep_indices = self._nms(boxes, scores, nms_thresh)
        
        person_boxes = []
        other_detections = []
        output_string = ""
        
        # 4. Group targets and scale back bounding box boundaries
        for idx in keep_indices:
            class_id = class_ids[idx]
            name = self.class_names[class_id]
            conf = round(float(scores[idx]), 2)
            
            bx1 = int(boxes[idx, 0] * orig_w / 640.0)
            by1 = int(boxes[idx, 1] * orig_h / 640.0)
            bx2 = int(boxes[idx, 2] * orig_w / 640.0)
            by2 = int(boxes[idx, 3] * orig_h / 640.0)
            
            if name == "person":
                person_boxes.append((bx1, by1, bx2, by2))
                cv2.rectangle(bgr_img, (bx1, by1), (bx2, by2), (255, 0, 0), 2)
            else:
                other_detections.append((name, conf, (bx1, by1, bx2, by2)))

        # 5. Spatial Boundary Intersection Verification Check
        held_items = []
        for name, conf, (ox1, oy1, ox2, oy2) in other_detections:
            is_held = False
            for (px1, py1, px2, py2) in person_boxes:
                # Intersecting boundary overlap test handles extreme cutout backgrounds safely
                horizontal_overlap = (ox1 < px2) and (ox2 > px1)
                vertical_overlap = (oy1 < py2) and (oy2 > py1)
                if horizontal_overlap and vertical_overlap:
                    is_held = True
                    break
                    
            color = (0, 255, 0) if is_held else (0, 165, 255)
            status = " [HELD]" if is_held else ""
            output_string += f"Detected: {name} | Conf: {conf}{status}\n"
            
            if is_held:
                held_items.append(name)
                
            cv2.rectangle(bgr_img, (ox1, oy1), (ox2, oy2), color, 2)
            label = f"{name} {conf}{status}"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(bgr_img, (ox1, oy1 - text_h - 4), (ox1 + text_w, oy1), color, -1)
            cv2.putText(bgr_img, label, (ox1, oy1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # 6. Parse textual summaries
        if person_boxes:
            summary = f"Man is holding: {', '.join(held_items)}" if held_items else "Man is not holding anything recognized."
            output_string = f"{summary}\n\n" + output_string
        else:
            output_string = "No person detected in the image.\n\n" + output_string

        return bgr_img, output_string.strip()
