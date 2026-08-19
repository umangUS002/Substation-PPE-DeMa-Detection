"""
YOLOv8 Model Adapter.
Provides a unified prediction interface for YOLOv8 weights to feed directly
into the PPE DeMa Engine.
"""

import os
from typing import List, Dict, Any, Union
import numpy as np

class YOLOv8Adapter:
    def __init__(self, weights_path: str = "models/best.pt", conf_threshold: float = 0.25):
        self.conf_threshold = conf_threshold
        self.model = None
        
        from ultralytics import YOLO
        
        target_weights = weights_path if (weights_path and os.path.exists(weights_path) and not weights_path.endswith("best_ssd_vgg16.pt")) else "models/best.pt"

        if os.path.exists(target_weights):
            try:
                self.model = YOLO(target_weights)
                print(f"[OK] Loaded YOLOv8 model from '{target_weights}'")
            except Exception as e:
                print(f"[WARN] Could not load '{target_weights}': {e}. Falling back to 'yolov8n.pt'...")
                self.model = YOLO("yolov8n.pt")
        else:
            print(f"[INFO] Loading standard 'yolov8n.pt' model...")
            self.model = YOLO("yolov8n.pt")



    def predict_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Runs object detection on a single image frame and returns formatted detection list:
        [{'label': str, 'confidence': float, 'bbox': [xmin, ymin, xmax, ymax]}]
        """
        if self.model is None:
            return []

        results = self.model.predict(source=frame, conf=self.conf_threshold, verbose=False)
        if not results:
            return []

        res = results[0]
        names = self.model.names
        formatted_detections = []

        boxes = res.boxes
        for box in boxes:
            cls_id = int(box.cls[0].item())
            cls_name = names.get(cls_id, f"class_{cls_id}")
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()

            formatted_detections.append({
                "label": cls_name,
                "confidence": conf,
                "bbox": xyxy
            })

        return formatted_detections
