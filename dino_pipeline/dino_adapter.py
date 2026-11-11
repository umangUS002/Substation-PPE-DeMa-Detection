import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import functional as F
import numpy as np
import cv2

try:
    from .rcnn_detector import get_faster_rcnn_detector
except (ImportError, ValueError):
    from dino_pipeline.rcnn_detector import get_faster_rcnn_detector

class PPEClassifierHead(nn.Module):
    """
    Lightweight Multi-Label Classifier to run on top of DINOv2 features.
    Predicts: [Helmet, Gloves, Footwear]
    """
    def __init__(self, embed_dim=384, num_classes=3):
        super().__init__()
        self.classifier = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        return self.classifier(x)

class DINOAdapter:
    """
    Two-Stage DINO Pipeline optimized for CCTV footage:
    - Stage 1: Faster R-CNN MobileNetV3 (320px, fast) with 2x2 tiling for small-person detection
    - Stage 2: DINOv2 + PPE Classifier on person crops
    
    Tiling Strategy: The CCTV frame is split into overlapping tiles so that 
    distant workers (who appear tiny in the full frame) become large enough 
    to be detected by the 320px person detector.
    """
    def __init__(self, classifier_weights_path=None, person_threshold=0.35, ppe_threshold=0.25, 
                 device=None, tile_grid=(2, 2), tile_overlap=0.15, min_person_area=600,
                 rcnn_model_size="medium"):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.person_threshold = person_threshold
        self.ppe_threshold = ppe_threshold
        self.tile_grid = tile_grid          # (rows, cols) tiling grid
        self.tile_overlap = tile_overlap    # overlap fraction between tiles
        self.min_person_area = min_person_area  # minimum bbox area to filter false positives
        self.rcnn_model_size = rcnn_model_size
        
        print(f"[INFO] Initializing Two-Stage DINO Pipeline on {self.device}...")
        print(f"[INFO] Tiling: {tile_grid[0]}x{tile_grid[1]} grid, {tile_overlap:.0%} overlap")
        
        # 1. Person Detector (Faster R-CNN: ResNet-50 FPN V2 for 'medium', MobileNetV3 for 'small')
        self.person_detector = get_faster_rcnn_detector(model_size=self.rcnn_model_size, device=self.device)
        self.PERSON_CLASS_ID = 1 
        
        # 2. Feature Extractor
        print("[INFO] Loading Meta DINOv2 (dinov2_vits14)...")
        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.dino.to(self.device)
        self.dino.eval()
        
        # 3. PPE Classifier Head
        self.ppe_classes = ["Helmet", "Gloves", "Footwear"]
        self.classifier = PPEClassifierHead(embed_dim=384, num_classes=len(self.ppe_classes))
        
        if classifier_weights_path:
            try:
                self.classifier.load_state_dict(torch.load(classifier_weights_path, map_location=self.device))
                print(f"[INFO] Loaded custom PPE Classifier weights from {classifier_weights_path}")
            except Exception as e:
                print(f"[WARNING] Could not load classifier weights: {e}")
                
        self.classifier.to(self.device)
        self.classifier.eval()

    def _generate_tiles(self, frame_h, frame_w):
        """
        Generates tile coordinates (y1, x1, y2, x2) for the given frame size.
        Each tile overlaps its neighbors by tile_overlap fraction.
        """
        rows, cols = self.tile_grid
        overlap = self.tile_overlap
        
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

    def _nms_persons(self, person_boxes, person_scores, iou_threshold=0.5):
        """
        Non-Maximum Suppression to remove duplicate person detections 
        from overlapping tiles.
        """
        if len(person_boxes) == 0:
            return [], []
        
        boxes_tensor = torch.tensor(person_boxes, dtype=torch.float32)
        scores_tensor = torch.tensor(person_scores, dtype=torch.float32)
        
        keep = torchvision.ops.nms(boxes_tensor, scores_tensor, iou_threshold)
        
        kept_boxes = [person_boxes[i] for i in keep]
        kept_scores = [person_scores[i] for i in keep]
        return kept_boxes, kept_scores

    def preprocess_for_dino(self, crop_bgr):
        # 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        crop_enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        # 2. Letterbox to preserve aspect ratio
        h, w = crop_enhanced.shape[:2]
        max_side = max(h, w)
        
        # Create gray square canvas
        canvas = np.full((max_side, max_side, 3), 128, dtype=np.uint8)
        
        # Center the crop on the canvas
        x_offset = (max_side - w) // 2
        y_offset = (max_side - h) // 2
        canvas[y_offset:y_offset+h, x_offset:x_offset+w] = crop_enhanced
        
        # 3. Resize safely and normalize
        crop_resized = cv2.resize(canvas, (224, 224))
        crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
        
        tensor = F.to_tensor(crop_rgb)
        tensor = F.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return tensor

    def _detect_persons_tiled(self, frame_bgr):
        """
        Runs person detection on overlapping tiles of the frame, then merges 
        results with NMS to deduplicate.
        """
        frame_h, frame_w = frame_bgr.shape[:2]
        tiles = self._generate_tiles(frame_h, frame_w)
        
        # Prepare all tile tensors
        tile_tensors = []
        tile_offsets = []  # (y_offset, x_offset) for each tile
        
        for (y1, x1, y2, x2) in tiles:
            tile_img = frame_bgr[y1:y2, x1:x2]
            tile_rgb = cv2.cvtColor(tile_img, cv2.COLOR_BGR2RGB)
            tile_tensors.append(F.to_tensor(tile_rgb).to(self.device))
            tile_offsets.append((y1, x1))
        
        # Batch all tiles through the detector at once
        with torch.no_grad():
            tile_outputs = self.person_detector(tile_tensors)
        
        # Collect all person detections, remapping coordinates to full frame
        all_boxes = []
        all_scores = []
        
        for tile_idx, pd_out in enumerate(tile_outputs):
            y_off, x_off = tile_offsets[tile_idx]
            boxes = pd_out['boxes'].cpu().numpy()
            scores = pd_out['scores'].cpu().numpy()
            labels = pd_out['labels'].cpu().numpy()
            
            mask = (labels == self.PERSON_CLASS_ID) & (scores > self.person_threshold)
            
            for box, score in zip(boxes[mask], scores[mask]):
                # Remap tile coordinates to full frame coordinates
                full_box = [
                    box[0] + x_off,
                    box[1] + y_off,
                    box[2] + x_off,
                    box[3] + y_off
                ]
                # Filter by minimum area to remove tiny false positives
                area = (full_box[2] - full_box[0]) * (full_box[3] - full_box[1])
                if area >= self.min_person_area:
                    all_boxes.append(full_box)
                    all_scores.append(float(score))
        
        # NMS to merge overlapping detections from adjacent tiles
        merged_boxes, merged_scores = self._nms_persons(all_boxes, all_scores, iou_threshold=0.4)
        return merged_boxes, merged_scores

    def predict_batch_frames(self, frames_bgr):
        """
        Runs the Two-Stage Pipeline on a batch of frames.
        Returns a list of result lists (one result list per frame).
        """
        if not frames_bgr:
            return []
            
        results_per_frame = [[] for _ in range(len(frames_bgr))]
        all_crops_tensors = []
        crop_to_frame_idx = []
        crop_to_box = []

        # --- STAGE 1: Person Detection with Tiling ---
        for frame_idx, frame in enumerate(frames_bgr):
            person_boxes, person_scores = self._detect_persons_tiled(frame)
            
            for i, (box, score) in enumerate(zip(person_boxes, person_scores)):
                results_per_frame[frame_idx].append({
                    "label": "Person",
                    "confidence": score,
                    "bbox": box
                })

                x1, y1, x2, y2 = map(int, box)
                # Add context padding: 10% horizontal, 20% vertical for better head/feet coverage
                w, h = x2 - x1, y2 - y1
                pad_x, pad_y = int(w * 0.10), int(h * 0.20)
                x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
                x2 = min(frame.shape[1], x2 + pad_x)
                y2 = min(frame.shape[0], y2 + pad_y)

                if x2 - x1 < 10 or y2 - y1 < 10:
                    continue

                crop_img = frame[y1:y2, x1:x2]
                all_crops_tensors.append(self.preprocess_for_dino(crop_img))
                crop_to_frame_idx.append(frame_idx)
                crop_to_box.append(box)

        # --- STAGE 2 & 3: DINO + Classifier ---
        if len(all_crops_tensors) > 0:
            batch_tensor = torch.stack(all_crops_tensors).to(self.device)
            
            with torch.no_grad():
                features = self.dino(batch_tensor)
                logits = self.classifier(features)
                probs = torch.sigmoid(logits).cpu().numpy()
                
            for i, prob_array in enumerate(probs):
                f_idx = crop_to_frame_idx[i]
                box = crop_to_box[i]
                
                for class_idx, ppe_name in enumerate(self.ppe_classes):
                    if ppe_name == "Gloves":
                        continue
                    conf = float(prob_array[class_idx])
                    if conf > self.ppe_threshold:
                        results_per_frame[f_idx].append({
                            "label": ppe_name,
                            "confidence": conf,
                            "bbox": box
                        })
                        
        return results_per_frame

    def predict_frame(self, frame_bgr):
        return self.predict_batch_frames([frame_bgr])[0]
