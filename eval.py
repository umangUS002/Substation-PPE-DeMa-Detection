"""
Evaluation script for the SSD-VGG16 PPE model.

Computes mAP@0.5 and mAP@0.5:0.95 (COCO-style, all-point interpolation) on a
held-out validation split, using the same metric definitions Ultralytics
reports for YOLOv8 ("mAP50" / "mAP50-95") so results are directly comparable
to a YOLOv8-medium benchmark run on the same dataset.

Usage:
    python eval.py --weights models/best_ssd_vgg16.pt --data-dir data/sample_dataset
"""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import Subset

from models.ssd_vgg16 import build_ssd_vgg16_model
from data.dataset_loader import (
    PPEDataset, LABEL_TO_ID, ID_TO_LABEL, split_indices,
    find_data_yaml, load_class_mapping, resolve_split_dirs,
)


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU of one box against an array of boxes, all in [x1, y1, x2, y2] format."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_box + area_boxes - inter
    return inter / np.clip(union, 1e-9, None)


def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """COCO-style all-point interpolated AP from a precision/recall curve."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def ap_for_class(preds, gts_by_image, iou_thresh):
    """
    preds: list of (image_idx, score, box) for one class, across all images.
    gts_by_image: dict image_idx -> np.ndarray[N, 4] ground-truth boxes for that class.
    Returns None if the class has no ground-truth boxes at all (undefined AP).
    """
    n_positive = sum(len(v) for v in gts_by_image.values())
    if n_positive == 0:
        return None

    preds = sorted(preds, key=lambda p: -p[1])
    tp = np.zeros(len(preds))
    fp = np.zeros(len(preds))
    matched = {img_idx: np.zeros(len(boxes), dtype=bool) for img_idx, boxes in gts_by_image.items()}

    for i, (img_idx, _score, box) in enumerate(preds):
        gt_boxes = gts_by_image.get(img_idx)
        if gt_boxes is None or len(gt_boxes) == 0:
            fp[i] = 1
            continue
        ious = box_iou(box, gt_boxes)
        best = int(np.argmax(ious))
        if ious[best] >= iou_thresh and not matched[img_idx][best]:
            tp[i] = 1
            matched[img_idx][best] = True
        else:
            fp[i] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / n_positive
    precision = tp_cum / np.clip(tp_cum + fp_cum, 1e-9, None)
    return average_precision(recall, precision)


@torch.no_grad()
def run_evaluation(model, dataset, device, iou_thresholds, conf_threshold=0.01):
    model.eval()
    class_ids = [c for c in ID_TO_LABEL if c != 0]
    preds_by_class = {c: [] for c in class_ids}
    gts_by_class = {c: {} for c in class_ids}

    for idx in range(len(dataset)):
        image, target = dataset[idx]
        output = model([image.to(device)])[0]

        boxes = output["boxes"].cpu().numpy()
        labels = output["labels"].cpu().numpy()
        scores = output["scores"].cpu().numpy()

        for box, label, score in zip(boxes, labels, scores):
            if score >= conf_threshold and int(label) in preds_by_class:
                preds_by_class[int(label)].append((idx, float(score), box))

        gt_boxes = target["boxes"].numpy()
        gt_labels = target["labels"].numpy()
        for c in class_ids:
            gts_by_class[c][idx] = gt_boxes[gt_labels == c]

    per_iou_results = {}
    for iou_t in iou_thresholds:
        class_aps = {}
        for c in class_ids:
            ap = ap_for_class(preds_by_class[c], gts_by_class[c], iou_t)
            if ap is not None:
                class_aps[ID_TO_LABEL[c]] = ap
        per_iou_results[iou_t] = class_aps

    return per_iou_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate SSD-VGG16 PPE model: mAP@0.5 and mAP@0.5:0.95")
    parser.add_argument("--weights", type=str, default="models/best_ssd_vgg16.pt")
    parser.add_argument("--data-dir", type=str, default="data/sample_dataset")
    parser.add_argument("--val-split", type=float, default=0.15, help="Must match the split used at training time")
    parser.add_argument("--conf-threshold", type=float, default=0.01, help="Low threshold recommended for mAP curves")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.weights):
        raise SystemExit(f"Weights file not found: '{args.weights}'")

    checkpoint = torch.load(args.weights, map_location=device)
    if isinstance(checkpoint, dict) and "synthetic" in checkpoint:
        raise SystemExit(
            f"'{args.weights}' is a synthetic placeholder checkpoint (train.py found no real dataset, "
            f"so nothing was actually trained). Train on a real dataset first, then re-run eval.py."
        )

    model = build_ssd_vgg16_model(num_classes=len(LABEL_TO_ID), pretrained=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)

    data_yaml = find_data_yaml(args.data_dir)
    class_mapping = None
    if data_yaml:
        class_mapping, unmatched = load_class_mapping(data_yaml)
        print(f"[INFO] Loaded class mapping from '{data_yaml}': {class_mapping}")
        if unmatched:
            print(f"[INFO] Ignoring non-target classes in data.yaml: {unmatched}")

    splits = resolve_split_dirs(args.data_dir)
    if splits["train"] is None:
        raise SystemExit(f"No dataset found under '{args.data_dir}' (looked for 'test/images/', 'valid/images/', or 'images/').")

    if splits["test"] is not None:
        eval_images, eval_ann = splits["test"]
        print(f"[INFO] Evaluating on the dataset's own held-out test split: '{eval_images}'")
        val_dataset = PPEDataset(images_dir=eval_images, annotations_dir=eval_ann, augment=False, class_mapping=class_mapping)
    elif splits["val"] is not None:
        eval_images, eval_ann = splits["val"]
        print(f"[INFO] No test split found - evaluating on the dataset's val split: '{eval_images}'")
        val_dataset = PPEDataset(images_dir=eval_images, annotations_dir=eval_ann, augment=False, class_mapping=class_mapping)
    else:
        # Flat layout: reproduce the same deterministic split train.py used.
        train_images, train_ann = splits["train"]
        full_dataset = PPEDataset(images_dir=train_images, annotations_dir=train_ann, augment=False, class_mapping=class_mapping)
        if len(full_dataset) == 0:
            raise SystemExit(f"No images found in '{train_images}'.")
        _, val_indices = split_indices(len(full_dataset), val_ratio=args.val_split)
        if not val_indices:
            print("[WARN] val-split produced 0 images; evaluating on the full dataset instead "
                  "(scores will be optimistic - not a true held-out result).")
            val_dataset = full_dataset
        else:
            val_dataset = Subset(full_dataset, val_indices)

    if len(val_dataset) == 0:
        raise SystemExit("Evaluation split has 0 images.")

    iou_thresholds = [round(x, 2) for x in np.arange(0.5, 1.0, 0.05)]
    results = run_evaluation(model, val_dataset, device, iou_thresholds, args.conf_threshold)

    def mean_ap(class_aps):
        return float(np.mean(list(class_aps.values()))) if class_aps else 0.0

    map50 = mean_ap(results[0.5])
    map_50_95 = float(np.mean([mean_ap(results[t]) for t in iou_thresholds]))

    print(f"\n[RESULTS] Validation images: {len(val_dataset)}")
    print(f"[RESULTS] mAP@0.5      : {map50:.4f}")
    print(f"[RESULTS] mAP@0.5:0.95 : {map_50_95:.4f}   (comparable to Ultralytics YOLOv8 'mAP50-95')")
    print("\nPer-class AP@0.5:")
    for cname, ap in results[0.5].items():
        print(f"  {cname:12s}: {ap:.4f}")
    missing = set(LABEL_TO_ID) - {"background"} - set(results[0.5])
    if missing:
        print(f"\n[NOTE] No ground-truth boxes for {sorted(missing)} in the validation split - AP undefined for them.")


if __name__ == "__main__":
    main()
