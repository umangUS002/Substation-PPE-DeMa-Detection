import os
import cv2
import json
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
from torchvision.transforms import functional as F
import argparse

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.dataset_loader import PPEDataset, ID_TO_LABEL

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0
        
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter)

def prepare_dataset(images_dir, annotations_dir, output_dir, device='cpu'):
    print(f"[INFO] Preparing DINO Classification Dataset from {images_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    print("[INFO] Loading Person Detector...")
    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    person_detector = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
    person_detector.to(device)
    person_detector.eval()
    
    # We use PPEDataset to parse the XML/TXT ground truths
    # We disable augmentation to get the original raw images and boxes
    dataset = PPEDataset(images_dir, annotations_dir, augment=False)
    
    metadata = []
    crop_id = 0
    
    for idx in range(len(dataset)):
        img_name = dataset.image_files[idx]
        img_path = os.path.join(dataset.images_dir, img_name)
        
        # PPEDataset resizes images internally. To avoid coordinate mismatch, 
        # we will load the raw image and parse annotations manually, or use the dataset's output.
        # Since dataset resizes to target_size, we just use the raw image directly 
        # and re-parse annotations using the dataset's helper methods.
        
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
            
        orig_h, orig_w = img_bgr.shape[:2]
        
        # Parse ground truths for this image
        base_name = os.path.splitext(img_name)[0]
        xml_path = os.path.join(annotations_dir, base_name + ".xml")
        txt_path = os.path.join(annotations_dir, base_name + ".txt")
        
        gt_boxes, gt_labels = [], []
        if os.path.exists(xml_path):
            gt_boxes, gt_labels = dataset.parse_voc_xml(xml_path)
        elif os.path.exists(txt_path):
            gt_boxes, gt_labels = dataset.parse_yolo_txt(txt_path, orig_w, orig_h)
            
        if not gt_boxes:
            continue
            
        # Detect people
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor_img = F.to_tensor(img_rgb).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pd_out = person_detector(tensor_img)[0]
            
        pred_boxes = pd_out['boxes'].cpu().numpy()
        scores = pd_out['scores'].cpu().numpy()
        pred_labels = pd_out['labels'].cpu().numpy()
        
        mask = (pred_labels == 1) & (scores > 0.5)
        person_boxes = pred_boxes[mask]
        
        # Match GT PPE boxes to detected Person boxes
        for p_box in person_boxes:
            x1, y1, x2, y2 = map(int, p_box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_w, x2), min(orig_h, y2)
            
            if x2 - x1 < 20 or y2 - y1 < 20:
                continue
                
            crop = img_bgr[y1:y2, x1:x2]
            
            # Determine Multi-Label vector: [Helmet, Gloves, Footwear]
            labels = {"Helmet": 0, "Gloves": 0, "Footwear": 0}
            
            for gt_b, gt_l in zip(gt_boxes, gt_labels):
                gt_name = ID_TO_LABEL.get(gt_l, "")
                if gt_name in labels:
                    # Check if PPE box is inside/overlaps the Person box
                    iou = compute_iou(p_box, gt_b)
                    
                    # Alternatively, check if center of PPE is inside person box
                    gt_cx = (gt_b[0] + gt_b[2]) / 2
                    gt_cy = (gt_b[1] + gt_b[3]) / 2
                    
                    if iou > 0.1 or (x1 <= gt_cx <= x2 and y1 <= gt_cy <= y2):
                        labels[gt_name] = 1
                        
            # Save crop
            crop_filename = f"crop_{crop_id:05d}.jpg"
            cv2.imwrite(os.path.join(output_dir, crop_filename), crop)
            
            metadata.append({
                "image": crop_filename,
                "labels": [labels["Helmet"], labels["Gloves"], labels["Footwear"]]
            })
            crop_id += 1
            
    # Save metadata JSON
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"[INFO] Generated {crop_id} training crops for DINO classifier.")
    print(f"[INFO] Metadata saved to {os.path.join(output_dir, 'metadata.json')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="data/images", help="Path to raw dataset images")
    parser.add_argument("--annotations", default="data/annotations", help="Path to raw XML/TXT annotations")
    parser.add_argument("--output", default="data/dino_crops", help="Output directory for crops")
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    prepare_dataset(args.images, args.annotations, args.output, device)
