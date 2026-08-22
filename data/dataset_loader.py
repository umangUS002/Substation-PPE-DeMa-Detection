"""
Dataset Loader for Substation PPE Detection.
Supports PASCAL VOC XML format annotations and PyTorch DataLoader integration.
Focused on Helmet, Gloves, and Footwear categories.
"""

import os
import random
import xml.etree.ElementTree as ET
import yaml
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
    def __init__(self, images_dir: str, annotations_dir: str, augment: bool = True, class_mapping: dict = None):
        """
        class_mapping: optional {yolo_class_id: LABEL_TO_ID value}, normally built
        by load_class_mapping() from the dataset's data.yaml. Only used for YOLO
        .txt annotations. If None, parse_yolo_txt falls back to a hardcoded
        3-class mapping that matches this repo's original sample dataset only —
        pass a real mapping for any other YOLO-format dataset.
        """
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.augment = augment
        self.class_mapping = class_mapping
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
            # Fallback only: matches this repo's original sample dataset class
            # order. Any other YOLO dataset needs a real mapping built from its
            # data.yaml via load_class_mapping() — otherwise labels are wrong.
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
            boxes, labels = self.parse_yolo_txt(txt_path, orig_w, orig_h, self.class_mapping)
        elif os.path.exists(labels_txt_path):
            boxes, labels = self.parse_yolo_txt(labels_txt_path, orig_w, orig_h, self.class_mapping)
        elif os.path.exists(img_xml_path):
            boxes, labels = self.parse_voc_xml(img_xml_path)
        elif os.path.exists(img_txt_path):
            boxes, labels = self.parse_yolo_txt(img_txt_path, orig_w, orig_h, self.class_mapping)
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


def find_data_yaml(data_dir: str):
    """Looks for a Roboflow/YOLO data.yaml at the dataset root or inside a split folder."""
    candidates = [
        os.path.join(data_dir, "data.yaml"),
        os.path.join(data_dir, "train", "data.yaml"),
        os.path.join(data_dir, "valid", "data.yaml"),
        os.path.join(data_dir, "test", "data.yaml"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def load_class_mapping(data_yaml_path: str):
    """
    Reads a YOLO data.yaml's class names and maps each one that matches (via
    dema_engine.ppe_rules.normalize_class_name) a target class onto our
    LABEL_TO_ID ids. Returns (mapping, unmatched_names) — unmatched classes
    (e.g. "Person", "NO-Hardhat") are expected and simply dropped, since this
    repo only trains on the 3 positive PPE-presence classes.
    """
    from dema_engine.ppe_rules import normalize_class_name

    with open(data_yaml_path, "r") as f:
        config = yaml.safe_load(f)

    names = config.get("names", [])
    id_to_name = {int(k): v for k, v in names.items()} if isinstance(names, dict) else dict(enumerate(names))

    mapping = {}
    unmatched = []
    for yolo_id, name in id_to_name.items():
        norm_name = normalize_class_name(name)
        if norm_name in LABEL_TO_ID and norm_name != "background":
            mapping[yolo_id] = LABEL_TO_ID[norm_name]
        else:
            unmatched.append(name)

    return mapping, unmatched


def resolve_split_dirs(data_dir: str):
    """
    Supports two dataset layouts:
    1. Roboflow-style: data_dir/{train,valid,test}/{images,labels}/
    2. Flat: data_dir/images/ + data_dir/annotations/ (this repo's original layout,
       VOC XML or a single unsplit YOLO folder)

    Returns {"train": (images_dir, labels_dir) or None, "val": ..., "test": ...}.
    For the flat layout, only "train" is set — the caller derives its own
    train/val split from it via split_indices().
    """
    train_images = os.path.join(data_dir, "train", "images")
    if os.path.exists(train_images):
        def split(name):
            img_dir = os.path.join(data_dir, name, "images")
            lbl_dir = os.path.join(data_dir, name, "labels")
            return (img_dir, lbl_dir) if os.path.exists(img_dir) else None

        return {
            "train": split("train"),
            "val": split("valid") or split("val"),
            "test": split("test"),
        }

    images_dir = os.path.join(data_dir, "images")
    annotations_dir = os.path.join(data_dir, "annotations")
    if os.path.exists(images_dir):
        return {"train": (images_dir, annotations_dir), "val": None, "test": None}

    return {"train": None, "val": None, "test": None}


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
