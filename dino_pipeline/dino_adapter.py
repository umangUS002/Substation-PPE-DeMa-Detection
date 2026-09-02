import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
from torchvision.transforms import functional as F
import numpy as np
import cv2

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
    def __init__(self, classifier_weights_path=None, conf_threshold=0.75, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.conf_threshold = conf_threshold
        
        print(f"[INFO] Initializing Two-Stage DINO Pipeline on {self.device}...")
        
        # 1. Person Detector
        print("[INFO] Loading Person Detector (Faster R-CNN MobileNetV3)...")
        weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
        self.person_detector = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
        self.person_detector.to(self.device)
        self.person_detector.eval()
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

    def preprocess_for_dino(self, crop_bgr):
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        crop_resized = cv2.resize(crop_rgb, (224, 224))
        tensor = F.to_tensor(crop_resized)
        tensor = F.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return tensor

    def predict_batch_frames(self, frames_bgr):
        """
        Runs the Two-Stage Pipeline on a batch of frames (e.g., from 24 cameras).
        Returns a list of result lists (one result list per frame).
        """
        if not frames_bgr:
            return []
            
        results_per_frame = [[] for _ in range(len(frames_bgr))]
        
        # --- STAGE 1: Person Detection (Batched) ---
        tensors = []
        for f in frames_bgr:
            frame_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            tensors.append(F.to_tensor(frame_rgb).to(self.device))
            
        with torch.no_grad():
            pd_outs = self.person_detector(tensors)
            
        # Collect all valid crops across all frames
        all_crops_tensors = []
        crop_to_frame_idx = [] # Maps crop index back to frame index
        crop_to_box = []       # Maps crop index to bounding box
        
        for frame_idx, pd_out in enumerate(pd_outs):
            boxes = pd_out['boxes'].cpu().numpy()
            scores = pd_out['scores'].cpu().numpy()
            labels = pd_out['labels'].cpu().numpy()
            
            mask = (labels == self.PERSON_CLASS_ID) & (scores > self.conf_threshold)
            person_boxes = boxes[mask]
            person_scores = scores[mask]
            
            for i, box in enumerate(person_boxes):
                results_per_frame[frame_idx].append({
                    "label": "Person",
                    "confidence": float(person_scores[i]),
                    "bbox": box.tolist()
                })
                
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frames_bgr[frame_idx].shape[1], x2), min(frames_bgr[frame_idx].shape[0], y2)
                
                if x2 - x1 < 10 or y2 - y1 < 10:
                    continue
                    
                crop_img = frames_bgr[frame_idx][y1:y2, x1:x2]
                all_crops_tensors.append(self.preprocess_for_dino(crop_img))
                crop_to_frame_idx.append(frame_idx)
                crop_to_box.append(box.tolist())
                
        # --- STAGE 2 & 3: DINO + Classifier (Massive Batch) ---
        if len(all_crops_tensors) > 0:
            # For 24 cameras, this batch might be size ~50-100 depending on people per frame
            batch_tensor = torch.stack(all_crops_tensors).to(self.device)
            
            with torch.no_grad():
                features = self.dino(batch_tensor)
                logits = self.classifier(features)
                probs = torch.sigmoid(logits).cpu().numpy()
                
            for i, prob_array in enumerate(probs):
                f_idx = crop_to_frame_idx[i]
                box = crop_to_box[i]
                
                for class_idx, ppe_name in enumerate(self.ppe_classes):
                    conf = float(prob_array[class_idx])
                    if conf > self.conf_threshold:
                        results_per_frame[f_idx].append({
                            "label": ppe_name,
                            "confidence": conf,
                            "bbox": box
                        })
                        
        return results_per_frame

    def predict_frame(self, frame_bgr):
        return self.predict_batch_frames([frame_bgr])[0]
