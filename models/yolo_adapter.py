"""
YOLOv8 Model Adapter.
Provides a unified prediction interface for YOLOv8 weights to feed directly
into the PPE DeMa Engine.
"""

import os
from typing import List, Dict, Any, Union
import numpy as np
import torch
import torchvision

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

        results = self.model.predict(source=frame, conf=self.conf_threshold, verbose=False, imgsz=1280)
        if not results:
            return []

        res = results[0]
        names = self.model.names
        formatted_detections = []

        boxes = res.boxes
        if boxes is not None:
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

    def _generate_tiles(self, frame_h, frame_w, rows=2, cols=2, overlap=0.15):
        tile_h = frame_h // rows
        tile_w = frame_w // cols
        overlap_h = int(tile_h * overlap)
        overlap_w = int(tile_w * overlap)
        
        tiles = []
        for r in range(rows):
            for c in range(cols):
                y1 = max(0, r * tile_h - overlap_h)
                x1 = max(0, c * tile_w - overlap_w)
                y2 = min(frame_h, (r + 1) * tile_h + overlap_h)
                x2 = min(frame_w, (c + 1) * tile_w + overlap_w)
                tiles.append((y1, x1, y2, x2))
        return tiles

    def predict_batch_frames(self, frames: List[np.ndarray], use_tiling: bool = True) -> List[List[Dict[str, Any]]]:
        """
        Runs object detection on a batch of frames using Tiling (same strategy as DINO).
        This guarantees small/distant objects are caught while keeping YOLO's speed.
        """
        if not frames or self.model is None:
            return [[] for _ in frames]

        all_crops = []
        crop_to_frame_idx = []
        crop_offsets = []
        
        for i, frame in enumerate(frames):
            # Always add full frame
            all_crops.append(frame)
            crop_to_frame_idx.append(i)
            crop_offsets.append((0, 0))
            
            if use_tiling:
                h, w = frame.shape[:2]
                tiles = self._generate_tiles(h, w)
                for (y1, x1, y2, x2) in tiles:
                    all_crops.append(frame[y1:y2, x1:x2])
                    crop_to_frame_idx.append(i)
                    crop_offsets.append((y1, x1))

        # Process all crops simultaneously (e.g. 60 cameras * 5 crops = 300 images in 1 batch)
        # Using 1024 for crops is plenty since they are already zoomed in
        results = self.model.predict(source=all_crops, conf=self.conf_threshold, verbose=False, stream=False, imgsz=1024)
        names = self.model.names
        
        raw_frame_results = [[] for _ in frames]
        
        for crop_idx, res in enumerate(results):
            frame_idx = crop_to_frame_idx[crop_idx]
            y_off, x_off = crop_offsets[crop_idx]
            
            if res.boxes is not None:
                for box in res.boxes:
                    cls_id = int(box.cls[0].item())
                    cls_name = names.get(cls_id, f"class_{cls_id}")
                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                    
                    remapped_box = [
                        xyxy[0] + x_off,
                        xyxy[1] + y_off,
                        xyxy[2] + x_off,
                        xyxy[3] + y_off
                    ]
                    
                    raw_frame_results[frame_idx].append({
                        "label": cls_name,
                        "confidence": conf,
                        "bbox": remapped_box
                    })
                    
        # Apply NMS (Non-Maximum Suppression) to merge overlapping boxes from tiles
        final_results = []
        for frame_dets in raw_frame_results:
            if not frame_dets:
                final_results.append([])
                continue
                
            label_groups = {}
            for det in frame_dets:
                lbl = det["label"]
                if lbl not in label_groups:
                    label_groups[lbl] = {"boxes": [], "scores": []}
                label_groups[lbl]["boxes"].append(det["bbox"])
                label_groups[lbl]["scores"].append(det["confidence"])
                
            merged_dets = []
            for lbl, grp in label_groups.items():
                b_tensor = torch.tensor(grp["boxes"], dtype=torch.float32)
                s_tensor = torch.tensor(grp["scores"], dtype=torch.float32)
                keep = torchvision.ops.nms(b_tensor, s_tensor, 0.45)
                for idx in keep:
                    merged_dets.append({
                        "label": lbl,
                        "confidence": float(grp["scores"][idx]),
                        "bbox": grp["boxes"][idx]
                    })
            final_results.append(merged_dets)
            
        return final_results
