"""
Single Shot MultiBox Detector (SSD) with VGG-16 Backbone (PyTorch Implementation).
As described in Section III-A of the paper:
Zhao & Barati (IEEE Transactions on Industry Applications, 2023).
Customized for Helmet, Gloves, and Footwear detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.models.detection.ssd import SSDClassificationHead

# Target PPE Categories + Background
TARGET_CLASSES_SSD = ["__background__", "Helmet", "Gloves", "Footwear"]

def build_ssd_vgg16_model(num_classes: int = len(TARGET_CLASSES_SSD), pretrained: bool = True):
    """
    Constructs an SSD300 model with VGG-16 backbone.
    Adjusts the classification head to target PPE classes.

    pretrained=False is used at call sites that load a full state_dict right
    after construction (--resume, loading a custom checkpoint), so the
    ImageNet-pretrained backbone weights torchvision would otherwise download
    by default are skipped too — they'd just be overwritten anyway.
    """
    if pretrained:
        # weights=DEFAULT already implies fully-pretrained backbone weights.
        model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
    else:
        model = ssd300_vgg16(weights=None, weights_backbone=None)

    # Retrieve in_channels list and num_anchors from existing SSD classification head
    in_channels = [module.in_channels for module in model.head.classification_head.module_list]
    num_anchors = model.anchor_generator.num_anchors_per_location()

    # Custom SSD Classification Head for target classes
    model.head.classification_head = SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes
    )
    return model



class MultiBoxLoss(nn.Module):
    """
    MultiBox Objective Loss Function from Equation (1) & (2) in paper:
    L(x, c, l, g) = (1 / N) * (L_conf(x, c) + alpha * L_loc(x, l, g))
    where alpha = 1.0.
    """

    def __init__(self, alpha: float = 1.0):
        super(MultiBoxLoss, self).__init__()
        self.alpha = alpha
        self.smooth_l1 = nn.SmoothL1Loss(reduction='sum')
        self.cross_entropy = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, confidence_preds, localization_preds, confidence_targets, localization_targets, num_matched):
        """
        Calculates confidence loss (Softmax) and localization loss (Smooth L1).
        """
        N = max(num_matched, 1.0)
        
        # Localization Loss: Smooth L1 over matched default boxes
        loc_loss = self.smooth_l1(localization_preds, localization_targets) / N
        
        # Confidence Loss: Cross-Entropy Softmax Loss
        conf_loss = self.cross_entropy(confidence_preds, confidence_targets) / N

        # Combined MultiBox Loss
        total_loss = conf_loss + (self.alpha * loc_loss)
        
        return {
            "total_loss": total_loss,
            "conf_loss": conf_loss,
            "loc_loss": loc_loss
        }


import os
import cv2
import numpy as np

class SSDVGG16Adapter:
    """
    Adapter for PyTorch SSD300 with VGG-16 Backbone (Exact Paper Model).
    Calculates predictions using PyTorch SSD-VGG16 and maps outputs to DeMa Engine.
    """

    def __init__(self, weights_path: str = "models/best_ssd_vgg16.pt", conf_threshold: float = 0.25):
        self.conf_threshold = conf_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

        try:
            if weights_path and os.path.exists(weights_path):
                checkpoint = torch.load(weights_path, map_location=self.device)
                if isinstance(checkpoint, dict) and "synthetic" in checkpoint:
                    self.model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
                    self.id_to_label = {1: "Person", 37: "Helmet", 80: "Footwear"}
                    print(f"[OK] Loaded Paper Model: PyTorch SSD300-VGG16 (Pretrained VGG-16 Backbone)")
                else:
                    self.model = build_ssd_vgg16_model(num_classes=len(TARGET_CLASSES_SSD), pretrained=False)
                    self.model.load_state_dict(checkpoint)
                    self.id_to_label = {1: "Helmet", 2: "Gloves", 3: "Footwear"}
                    print(f"[OK] Loaded Custom Paper Model: PyTorch SSD300-VGG16 from '{weights_path}'")
            else:
                self.model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
                self.id_to_label = {1: "Person", 37: "Helmet", 80: "Footwear"}
                print(f"[OK] Loaded Paper Model: PyTorch SSD300-VGG16 (COCO VGG-16 Weights)")

            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"[WARN] Error initializing SSD-VGG16 model: {e}")

    def predict_frame(self, frame: np.ndarray) -> list:
        if self.model is None:
            return []

        import torchvision.transforms.functional as TF
        from PIL import Image

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        img_tensor = TF.to_tensor(pil_img).to(self.device)

        with torch.no_grad():
            predictions = self.model([img_tensor])[0]

        boxes = predictions["boxes"].cpu().numpy()
        labels = predictions["labels"].cpu().numpy()
        scores = predictions["scores"].cpu().numpy()

        formatted_detections = []
        for box, label_id, score in zip(boxes, labels, scores):
            if score >= self.conf_threshold:
                label_name = self.id_to_label.get(int(label_id), f"Class_{label_id}")
                formatted_detections.append({
                    "label": label_name,
                    "confidence": float(score),
                    "bbox": [int(b) for b in box]
                })

        return formatted_detections

