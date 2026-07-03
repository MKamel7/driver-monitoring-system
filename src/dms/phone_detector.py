"""Phone-usage detection with a custom-trained YOLOv8 model.

The bundled model (models/yolov8m_custom.pt) was trained on a custom
dataset of hands holding phones. Measured on the original project
hardware: 97.8% mAP@0.5, 0.75 mAP@[0.5:0.95], ~10 FPS on Raspberry Pi.

Fixes applied in the 2026 refactor:
- `cap.relaese()` / `cv2.destroyAllWindow()` typos crashed the demo on exit.
- `if classes == 0:` raised "truth value of an array is ambiguous" as soon
  as two detections appeared in one frame.
- CUDA device was hard-coded; now auto-selected.
"""

import cv2
import numpy as np
from ultralytics import YOLO

PHONE_CLASS_ID = 0  # the custom model has a single class: phone


class PhoneDetector:

    def __init__(self, model_path: str = "models/yolov8m_custom.pt",
                 confidence: float = 0.5, device: str | None = None):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.device = device  # None lets ultralytics pick CUDA if available

    def process(self, frame_bgr, draw: bool = True):
        """Run one frame. Returns (frame, phone_detected, boxes)."""
        results = self.model(frame_bgr, conf=self.confidence,
                             device=self.device, verbose=False)
        result = results[0]
        boxes = np.array(result.boxes.xyxy.cpu(), dtype=int)
        classes = np.array(result.boxes.cls.cpu(), dtype=int)

        phone_boxes = [box for cls, box in zip(classes, boxes)
                       if cls == PHONE_CLASS_ID]

        if draw:
            for (x1, y1, x2, y2) in phone_boxes:
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame_bgr, "phone", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

        return frame_bgr, len(phone_boxes) > 0, phone_boxes
