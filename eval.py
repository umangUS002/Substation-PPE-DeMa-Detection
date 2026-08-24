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
import json
import os

import matplotlib.pyplot as plt
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


def _cumulative_tp_fp(preds, gts_by_image, iou_thresh):
    """
    Sorted-by-score cumulative TP/FP arrays, shared by ap_for_class() and
    pr_curve_for_class() so the matching logic (each GT box claimable once,
    IoU >= iou_thresh) only lives in one place.
    """
    preds_sorted = sorted(preds, key=lambda p: -p[1])
    tp = np.zeros(len(preds_sorted))
    fp = np.zeros(len(preds_sorted))
    matched = {img_idx: np.zeros(len(boxes), dtype=bool) for img_idx, boxes in gts_by_image.items()}

    for i, (img_idx, _score, box) in enumerate(preds_sorted):
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

    return np.cumsum(tp), np.cumsum(fp)


def ap_for_class(preds, gts_by_image, iou_thresh):
    """
    preds: list of (image_idx, score, box) for one class, across all images.
    gts_by_image: dict image_idx -> np.ndarray[N, 4] ground-truth boxes for that class.
    Returns None if the class has no ground-truth boxes at all (undefined AP).
    """
    n_positive = sum(len(v) for v in gts_by_image.values())
    if n_positive == 0:
        return None
    tp_cum, fp_cum = _cumulative_tp_fp(preds, gts_by_image, iou_thresh)
    recall = tp_cum / n_positive
    precision = tp_cum / np.clip(tp_cum + fp_cum, 1e-9, None)
    return average_precision(recall, precision)


def pr_curve_for_class(preds, gts_by_image, iou_thresh):
    """Returns (recall, precision) arrays for plotting, or None if no GT boxes."""
    n_positive = sum(len(v) for v in gts_by_image.values())
    if n_positive == 0:
        return None
    tp_cum, fp_cum = _cumulative_tp_fp(preds, gts_by_image, iou_thresh)
    recall = tp_cum / n_positive
    precision = tp_cum / np.clip(tp_cum + fp_cum, 1e-9, None)
    return recall, precision


@torch.no_grad()
def collect_predictions(model, dataset, device, conf_threshold=0.01):
    """
    Runs inference once over the whole dataset. Returns:
      preds_by_class: {class_id: [(image_idx, score, box), ...]}
      gts_by_class:   {class_id: {image_idx: np.ndarray[N,4]}}
      preds_by_image: {image_idx: [(class_id, score, box), ...]}   (for the confusion matrix)
      gts_by_image:   {image_idx: [(class_id, box), ...]}          (for the confusion matrix)
    conf_threshold should stay very low (default 0.01) here — it's just the
    floor for what counts as a "prediction" at all; the AP curve needs the
    full range of scores to integrate over, and the threshold sweep / confusion
    matrix filter down from this pool per confidence level.
    """
    model.eval()
    class_ids = [c for c in ID_TO_LABEL if c != 0]
    preds_by_class = {c: [] for c in class_ids}
    gts_by_class = {c: {} for c in class_ids}
    preds_by_image = {}
    gts_by_image = {}

    for idx in range(len(dataset)):
        image, target = dataset[idx]
        output = model([image.to(device)])[0]

        boxes = output["boxes"].cpu().numpy()
        labels = output["labels"].cpu().numpy()
        scores = output["scores"].cpu().numpy()

        img_preds = []
        for box, label, score in zip(boxes, labels, scores):
            if score >= conf_threshold and int(label) in preds_by_class:
                preds_by_class[int(label)].append((idx, float(score), box))
                img_preds.append((int(label), float(score), box))
        preds_by_image[idx] = img_preds

        gt_boxes = target["boxes"].numpy()
        gt_labels = target["labels"].numpy()
        for c in class_ids:
            gts_by_class[c][idx] = gt_boxes[gt_labels == c]
        gts_by_image[idx] = list(zip(gt_labels.tolist(), gt_boxes))

    return preds_by_class, gts_by_class, preds_by_image, gts_by_image


def compute_map(preds_by_class, gts_by_class, iou_thresholds):
    """AP per class at each IoU threshold, using every collected prediction."""
    per_iou_results = {}
    for iou_t in iou_thresholds:
        class_aps = {}
        for c in preds_by_class:
            ap = ap_for_class(preds_by_class[c], gts_by_class[c], iou_t)
            if ap is not None:
                class_aps[ID_TO_LABEL[c]] = ap
        per_iou_results[iou_t] = class_aps
    return per_iou_results


def run_evaluation(model, dataset, device, iou_thresholds, conf_threshold=0.01):
    """Kept for backward compatibility: collect + compute_map in one call."""
    preds_by_class, gts_by_class, _, _ = collect_predictions(model, dataset, device, conf_threshold)
    return compute_map(preds_by_class, gts_by_class, iou_thresholds)


def precision_recall_at_threshold(preds, gts_by_image, iou_thresh, conf_thresh):
    """
    Precision/recall/F1 for one class using only predictions with score >=
    conf_thresh — i.e. "if I actually deployed at this confidence cutoff,
    what would I see?" (unlike AP, which sweeps every threshold at once).
    """
    n_positive = sum(len(v) for v in gts_by_image.values())
    filtered = sorted((p for p in preds if p[1] >= conf_thresh), key=lambda p: -p[1])
    matched = {img_idx: np.zeros(len(boxes), dtype=bool) for img_idx, boxes in gts_by_image.items()}

    tp = fp = 0
    for img_idx, _score, box in filtered:
        gt_boxes = gts_by_image.get(img_idx)
        if gt_boxes is None or len(gt_boxes) == 0:
            fp += 1
            continue
        ious = box_iou(box, gt_boxes)
        best = int(np.argmax(ious))
        if ious[best] >= iou_thresh and not matched[img_idx][best]:
            tp += 1
            matched[img_idx][best] = True
        else:
            fp += 1

    fn = n_positive - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / n_positive if n_positive > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1, tp, fp, fn


def threshold_sweep(preds_by_class, gts_by_class, iou_thresh, conf_thresholds):
    """
    Precision/recall/F1 per class and micro-averaged overall, at each
    confidence threshold in conf_thresholds — the practical "what threshold
    should I actually deploy at" report.
    """
    rows = []
    for conf_t in conf_thresholds:
        per_class = {}
        tp_total = fp_total = fn_total = 0
        for c, preds in preds_by_class.items():
            p, r, f1, tp, fp, fn = precision_recall_at_threshold(preds, gts_by_class[c], iou_thresh, conf_t)
            per_class[ID_TO_LABEL[c]] = (p, r, f1)
            tp_total += tp
            fp_total += fp
            fn_total += fn

        micro_p = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
        micro_r = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
        rows.append({"conf": conf_t, "per_class": per_class, "precision": micro_p, "recall": micro_r, "f1": micro_f1})
    return rows


def build_confusion_matrix(preds_by_image, gts_by_image, iou_thresh, conf_thresh):
    """
    Class-confusion matrix at a fixed confidence threshold. Rows = ground-truth
    class, columns = predicted class, plus a "background" row (false-positive
    predictions matching no GT box) and "background" column (missed/unmatched
    GT boxes, i.e. false negatives). Matching is per-image and greedy by score
    across ALL classes at once (unlike the per-class AP matching above), so it
    can actually surface cross-class confusion -- e.g. a Helmet GT box that a
    Gloves prediction incorrectly claimed -- not just presence/absence.
    """
    class_names = [ID_TO_LABEL[c] for c in sorted(ID_TO_LABEL) if c != 0]
    names = class_names + ["background"]
    idx_of = {name: i for i, name in enumerate(names)}
    matrix = np.zeros((len(names), len(names)), dtype=int)

    for img_idx, gt_list in gts_by_image.items():
        preds = [p for p in preds_by_image.get(img_idx, []) if p[1] >= conf_thresh]
        preds = sorted(preds, key=lambda p: -p[1])
        gt_matched = [False] * len(gt_list)
        gt_boxes = np.array([g[1] for g in gt_list], dtype=float) if gt_list else None

        for label, _score, box in preds:
            pred_name = ID_TO_LABEL[label]
            if gt_boxes is None:
                matrix[idx_of["background"], idx_of[pred_name]] += 1
                continue
            ious = box_iou(box, gt_boxes)
            best = int(np.argmax(ious))
            if ious[best] >= iou_thresh and not gt_matched[best]:
                gt_matched[best] = True
                gt_name = ID_TO_LABEL[gt_list[best][0]]
                matrix[idx_of[gt_name], idx_of[pred_name]] += 1
            else:
                matrix[idx_of["background"], idx_of[pred_name]] += 1

        for matched, (label, _box) in zip(gt_matched, gt_list):
            if not matched:
                matrix[idx_of[ID_TO_LABEL[label]], idx_of["background"]] += 1

    return matrix, names


def plot_pr_curves(preds_by_class, gts_by_class, iou_thresh, results_dir):
    plt.figure(figsize=(7, 6))
    any_curve = False
    for c, preds in preds_by_class.items():
        curve = pr_curve_for_class(preds, gts_by_class[c], iou_thresh)
        if curve is None:
            continue
        recall, precision = curve
        plt.plot(recall, precision, label=ID_TO_LABEL[c])
        any_curve = True

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curves @ IoU {iou_thresh}")
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.grid(True)
    if any_curve:
        plt.legend()
    plt.tight_layout()
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "pr_curves.png")
    plt.savefig(path)
    plt.close()
    return path


def plot_confusion_matrix(matrix, names, results_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title("Confusion Matrix")
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                     color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(path)
    plt.close()
    return path


def main():
    parser = argparse.ArgumentParser(description="Evaluate SSD-VGG16 PPE model: mAP@0.5 and mAP@0.5:0.95")
    parser.add_argument("--weights", type=str, default="models/best_ssd_vgg16.pt")
    parser.add_argument("--data-dir", type=str, default="data/sample_dataset")
    parser.add_argument("--results-dir", type=str, default="results", help="Where eval_results.json / pr_curves.png / confusion_matrix.png are written")
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
    # Single inference pass — reused for AP, the threshold sweep, PR curves, and the confusion matrix.
    preds_by_class, gts_by_class, preds_by_image, gts_by_image = collect_predictions(
        model, val_dataset, device, args.conf_threshold
    )
    results = compute_map(preds_by_class, gts_by_class, iou_thresholds)

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

    # Practical "what should I actually deploy at" report — AP sweeps every
    # threshold internally, but doesn't tell you what a fixed cutoff looks like.
    sweep_thresholds = [0.10, 0.25, 0.50, 0.75, 0.90]
    sweep_rows = threshold_sweep(preds_by_class, gts_by_class, iou_thresh=0.5, conf_thresholds=sweep_thresholds)

    print("\n[THRESHOLD SWEEP] Precision / Recall / F1 @ IoU 0.5 (micro-averaged across classes):")
    print(f"  {'conf>=':>7} | {'precision':>9} | {'recall':>7} | {'f1':>7}")
    for row in sweep_rows:
        print(f"  {row['conf']:>7.2f} | {row['precision']:>9.4f} | {row['recall']:>7.4f} | {row['f1']:>7.4f}")

    best_row = max(sweep_rows, key=lambda r: r["f1"])
    print(f"\n[SUGGESTED THRESHOLD] conf>={best_row['conf']:.2f} gives the best F1 ({best_row['f1']:.4f}) "
          f"among {sweep_thresholds} -- a reasonable starting point for --conf-threshold in detect.py.")

    # Combined per-class table: Class | Precision | Recall | AP@0.50 | #GT | #predictions
    print(f"\n[PER-CLASS TABLE] (Precision/Recall @ conf>={best_row['conf']:.2f}, AP @ IoU 0.5)")
    print(f"  {'Class':12s} | {'Precision':>9} | {'Recall':>7} | {'AP@0.50':>7} | {'#GT':>6} | {'#Preds':>8}")
    per_class_table = []
    for c in sorted(preds_by_class):
        cname = ID_TO_LABEL[c]
        p, r, _f1 = best_row["per_class"][cname]
        ap = results[0.5].get(cname, 0.0)
        n_gt = sum(len(v) for v in gts_by_class[c].values())
        n_pred = len(preds_by_class[c])
        print(f"  {cname:12s} | {p:>9.4f} | {r:>7.4f} | {ap:>7.4f} | {n_gt:>6} | {n_pred:>8}")
        per_class_table.append({"class": cname, "precision": p, "recall": r, "ap50": ap,
                                 "num_ground_truth": n_gt, "num_predictions": n_pred})

    # PR curves + confusion matrix artifacts
    pr_path = plot_pr_curves(preds_by_class, gts_by_class, iou_thresh=0.5, results_dir=args.results_dir)
    print(f"\n[SAVE] Saved PR curves to: '{pr_path}'")

    confusion_matrix, confusion_names = build_confusion_matrix(
        preds_by_image, gts_by_image, iou_thresh=0.5, conf_thresh=best_row["conf"]
    )
    cm_path = plot_confusion_matrix(confusion_matrix, confusion_names, results_dir=args.results_dir)
    print(f"[SAVE] Saved confusion matrix (@ conf>={best_row['conf']:.2f}) to: '{cm_path}'")

    # Full results dump for the paper / later comparison
    os.makedirs(args.results_dir, exist_ok=True)
    results_path = os.path.join(args.results_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "weights": args.weights,
            "num_val_images": len(val_dataset),
            "map50": map50,
            "map50_95": map_50_95,
            "ap_by_iou": {str(t): v for t, v in results.items()},
            "threshold_sweep": sweep_rows,
            "suggested_threshold": best_row["conf"],
            "per_class_table": per_class_table,
            "confusion_matrix": {"labels": confusion_names, "matrix": confusion_matrix.tolist()},
        }, f, indent=2)
    print(f"[SAVE] Saved full evaluation results to: '{results_path}'")


if __name__ == "__main__":
    main()
