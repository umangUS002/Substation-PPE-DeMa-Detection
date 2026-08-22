"""
Dataset Loader for Substation PPE Detection.
Supports PASCAL VOC XML format annotations and PyTorch DataLoader integration.
Focused on Helmet, Gloves, and Footwear categories.
"""

import os
import random
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset
from PIL import Image
from .augmentation import FewShotAugmenter

LABEL_TO_ID = {
    "background": 0,
    "Helmet": 1,
    "Gloves": 2,
    "Footwear": 3
}

ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

class PPEDataset(Dataset):
    def __init__(self, images_dir: str, annotations_dir: str, augment: bool = True):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.augment = augment
        self.augmenter = FewShotAugmenter()

        self.image_files = []
        if os.path.exists(images_dir):
            # Sorted for a deterministic file order, so a fixed-seed train/val
            # split (see split_indices below) lands on the same images every run.
            self.image_files = sorted(f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png')))

    def __len__(self):
        return len(self.image_files)

    def parse_voc_xml(self, xml_path: str):
        """Parses Pascal VOC XML annotation file."""
        boxes = []
        labels = []
        if not os.path.exists(xml_path):
            return boxes, labels

        tree = ET.parse(xml_path)
        root = tree.getroot()

        for obj in root.findall('object'):
            name = obj.find('name').text.strip()
            
            # Map alias to target class
            from dema_engine.ppe_rules import normalize_class_name
            norm_name = normalize_class_name(name)

            if norm_name in LABEL_TO_ID:
                bndbox = obj.find('bndbox')
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)

                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(LABEL_TO_ID[norm_name])

        return boxes, labels

    def parse_yolo_txt(self, txt_path: str, img_width: int, img_height: int, class_mapping: dict = None):
        """Parses Roboflow YOLO format (.txt) normalized bounding box annotations."""
        boxes = []
        labels = []
        if not os.path.exists(txt_path):
            return boxes, labels

        if class_mapping is None:
            # Dataset mapping for target classes: Helmet (1), Gloves (2), Footwear (3)
            # Class 3 = Helmet/Hardhat, Class 1 = Gloves, Class 0 = Footwear/Boots
            class_mapping = {3: 1, 1: 2, 0: 3}

        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    if cls_id in class_mapping:
                        target_label = class_mapping[cls_id]
                        xc, yc, w, h = [float(x) for x in parts[1:5]]

                        xmin = (xc - w / 2.0) * img_width
                        ymin = (yc - h / 2.0) * img_height
                        xmax = (xc + w / 2.0) * img_width
                        ymax = (yc + h / 2.0) * img_height

                        boxes.append([xmin, ymin, xmax, ymax])
                        labels.append(target_label)

        return boxes, labels

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        base_name = os.path.splitext(img_name)[0]
        
        xml_path = os.path.join(self.annotations_dir, base_name + ".xml")
        txt_path = os.path.join(self.annotations_dir, base_name + ".txt")
        
        labels_dir = self.annotations_dir.replace("annotations", "labels")
        labels_txt_path = os.path.join(labels_dir, base_name + ".txt")

        img_xml_path = os.path.join(self.images_dir, base_name + ".xml")
        img_txt_path = os.path.join(self.images_dir, base_name + ".txt")

        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        if os.path.exists(xml_path):
            boxes, labels = self.parse_voc_xml(xml_path)
        elif os.path.exists(txt_path):
            boxes, labels = self.parse_yolo_txt(txt_path, orig_w, orig_h)
        elif os.path.exists(labels_txt_path):
            boxes, labels = self.parse_yolo_txt(labels_txt_path, orig_w, orig_h)
        elif os.path.exists(img_xml_path):
            boxes, labels = self.parse_voc_xml(img_xml_path)
        elif os.path.exists(img_txt_path):
            boxes, labels = self.parse_yolo_txt(img_txt_path, orig_w, orig_h)
        else:
            boxes, labels = [], []

        target_w, target_h = self.augmenter.target_size

        if self.augment:
            image, boxes, labels = self.augmenter.apply_augmentation(image, boxes, labels)
        else:
            image, boxes = self.augmenter.resize_standard(image, boxes)

        # Strictly clamp bounding boxes to target dimensions (flip/crop can push them to the edge)
        valid_boxes = []
        valid_labels = []

        for box, label in zip(boxes, labels):
            x1 = max(0.0, min(box[0], target_w - 1.0))
            y1 = max(0.0, min(box[1], target_h - 1.0))
            x2 = max(x1 + 1.0, min(box[2], float(target_w)))
            y2 = max(y1 + 1.0, min(box[3], float(target_h)))

            if (x2 - x1) >= 1.0 and (y2 - y1) >= 1.0:
                valid_boxes.append([x1, y1, x2, y2])
                valid_labels.append(label)

        # Convert to Tensor
        import torchvision.transforms.functional as TF
        img_tensor = TF.to_tensor(image)
        
        target = {
            "boxes": torch.tensor(valid_boxes, dtype=torch.float32) if valid_boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(valid_labels, dtype=torch.int64) if valid_labels else torch.zeros((0,), dtype=torch.int64)
        }

        return img_tensor, target


def split_indices(n: int, val_ratio: float = 0.15, seed: int = 42):
    """
    Deterministic train/val index split. Given two PPEDataset instances built
    from the same images_dir (one with augment=True, one augment=False), apply
    the returned indices to each via torch.utils.data.Subset so training gets
    augmentation and validation stays on clean, fixed images.
    """
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    val_size = int(n * val_ratio) if n > 1 else 0
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return train_indices, val_indices
