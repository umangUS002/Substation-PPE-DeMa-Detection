"""
Enhanced Video/CCTV PPE Detection Script.
Optimized for real-world surveillance footage with:
- Frame preprocessing (CLAHE, denoising, sharpening)
- Multi-scale detection for small/distant workers
- Temporal smoothing to suppress flickering detections
- Smart frame skipping for performance
- Support for SSD, YOLO, and DINO model backends

Usage:
    python detect_video.py --source v1.mov --model-type yolo --save
    python detect_video.py --source 0 --model-type dino
    python detect_video.py --source "rtsp://camera_ip/stream" --model-type yolo --save
"""

import argparse
import os
import sys
import time
import cv2
import numpy as np
import torch
import torchvision

from dema_engine.dema_algorithm import PPEDeMaEngine
from dema_engine.ppe_rules import TARGET_CLASSES
from data.video_preprocessor import VideoFramePreprocessor, FrameSkipController


# ─── Multi-Scale Detection ─────────────────────────────────────────────────────

def run_multiscale_detection(detector, frame, model_type, scales=[1.0, 0.6]):
    """
    Runs detection at multiple scales and merges results with NMS.
    
    For CCTV footage, distant workers are tiny in the full frame.
    By running the detector on the full frame AND on zoomed-in crops,
    we catch both nearby and far-away workers.
    
    Args:
        detector: The model adapter (SSD, YOLO, or DINO)
        frame: BGR frame
        model_type: 'ssd', 'yolo', or 'dino'
        scales: List of scales. 1.0 = full frame, 0.5 = 2x2 grid of crops.
    
    Returns:
        List of merged detections after NMS.
    """
    all_boxes = []
    all_scores = []
    all_labels = []
    
    frame_h, frame_w = frame.shape[:2]
    
    for scale in scales:
        if scale >= 1.0:
            # Full frame detection
            dets = detector.predict_frame(frame)
            for d in dets:
                all_boxes.append(d["bbox"])
                all_scores.append(d["confidence"])
                all_labels.append(d["label"])
        else:
            # Crop-based detection: split into grid based on scale
            # scale=0.5 means each crop is 50% of the frame → 2x2 grid with overlap
            crop_h = int(frame_h * scale)
            crop_w = int(frame_w * scale)
            overlap = 0.15  # 15% overlap between crops
            
            step_h = int(crop_h * (1 - overlap))
            step_w = int(crop_w * (1 - overlap))
            
            for y in range(0, frame_h - crop_h + 1, step_h):
                for x in range(0, frame_w - crop_w + 1, step_w):
                    crop = frame[y:y + crop_h, x:x + crop_w]
                    
                    # Upscale the crop to full detection resolution
                    upscaled = cv2.resize(crop, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)
                    
                    dets = detector.predict_frame(upscaled)
                    
                    # Remap bounding boxes back to the original frame coordinates
                    scale_x = crop_w / frame_w
                    scale_y = crop_h / frame_h
                    
                    for d in dets:
                        bbox = d["bbox"]
                        remapped_box = [
                            int(bbox[0] * scale_x + x),
                            int(bbox[1] * scale_y + y),
                            int(bbox[2] * scale_x + x),
                            int(bbox[3] * scale_y + y),
                        ]
                        
                        # Filter out tiny false positives
                        area = (remapped_box[2] - remapped_box[0]) * (remapped_box[3] - remapped_box[1])
                        if area >= 400:
                            all_boxes.append(remapped_box)
                            all_scores.append(d["confidence"])
                            all_labels.append(d["label"])
    
    # NMS to merge duplicates from multi-scale
    if len(all_boxes) == 0:
        return []
    
    merged = _nms_merge(all_boxes, all_scores, all_labels, iou_threshold=0.45)
    return merged


def _nms_merge(boxes, scores, labels, iou_threshold=0.45):
    """
    Applies per-class NMS to merge duplicate detections from multi-scale inference.
    """
    # Group by label for per-class NMS
    label_groups = {}
    for box, score, label in zip(boxes, scores, labels):
        if label not in label_groups:
            label_groups[label] = {"boxes": [], "scores": []}
        label_groups[label]["boxes"].append(box)
        label_groups[label]["scores"].append(score)
    
    merged_detections = []
    
    for label, group in label_groups.items():
        if len(group["boxes"]) == 0:
            continue
        
        boxes_tensor = torch.tensor(group["boxes"], dtype=torch.float32)
        scores_tensor = torch.tensor(group["scores"], dtype=torch.float32)
        
        keep = torchvision.ops.nms(boxes_tensor, scores_tensor, iou_threshold)
        
        for idx in keep:
            merged_detections.append({
                "label": label,
                "confidence": group["scores"][idx],
                "bbox": group["boxes"][idx]
            })
    
    return merged_detections


# ─── Temporal Smoothing ─────────────────────────────────────────────────────────

class TemporalDetectionSmoother:
    """
    Smooths detections across multiple frames to suppress flickering.
    
    Strategy:
    - Maintains a sliding window of detections from the last N processed frames
    - For each candidate detection, checks if a similar detection (high IoU) existed 
      in recent frames
    - Only outputs detections that appear consistently (confirmed by temporal evidence)
    - Carries forward recent confirmed detections for skipped frames
    """
    
    def __init__(self, window_size: int = 5, min_confirmations: int = 2, iou_threshold: float = 0.35):
        self.window_size = window_size
        self.min_confirmations = min_confirmations
        self.iou_threshold = iou_threshold
        self.detection_history = []  # List of (frame_detections, timestamp)
        self.last_confirmed_detections = []
    
    def update(self, detections: list) -> list:
        """
        Adds new detections and returns temporally-confirmed detections.
        """
        # Add to history
        self.detection_history.append(detections)
        if len(self.detection_history) > self.window_size:
            self.detection_history.pop(0)
        
        # For each detection in the current frame, count how many recent frames
        # had a similar detection (same label + high IoU overlap)
        confirmed = []
        
        for det in detections:
            confirmation_count = 0
            
            # Check against all past frames in the window (excluding current)
            for past_dets in self.detection_history[:-1]:
                for past_det in past_dets:
                    if past_det["label"] == det["label"]:
                        iou = self._compute_iou(det["bbox"], past_det["bbox"])
                        if iou > self.iou_threshold:
                            confirmation_count += 1
                            break  # One match per frame is enough
            
            # Confirmed if seen in enough recent frames.
            # PPE labels (Helmet/Gloves/Footwear) ALWAYS need temporal confirmation
            # to prevent single-frame flickering. Only Person boxes (which are already
            # stabilised by the tracker) can bypass with high confidence alone.
            is_ppe_label = det["label"] in ("Helmet", "Gloves", "Footwear")
            if is_ppe_label:
                # Strict: PPE labels must be confirmed across multiple frames, no shortcuts
                if confirmation_count >= self.min_confirmations:
                    confirmed.append(det)
            else:
                # Person/other labels: original logic with high-confidence bypass
                if confirmation_count >= self.min_confirmations or det["confidence"] > 0.65:
                    confirmed.append(det)
        
        # If no detections at all this frame but we had recent ones, carry forward briefly
        if len(confirmed) == 0 and len(self.last_confirmed_detections) > 0:
            # Carry forward last confirmed detections with slightly reduced confidence
            carryover = []
            for det in self.last_confirmed_detections:
                faded = det.copy()
                faded["confidence"] = max(0.15, det["confidence"] * 0.85)
                carryover.append(faded)
            self.last_confirmed_detections = carryover
            return carryover
        
        self.last_confirmed_detections = confirmed
        return confirmed
    
    def get_last_detections(self) -> list:
        """Returns the last confirmed detections (for skipped frames)."""
        return self.last_confirmed_detections
    
    @staticmethod
    def _compute_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        denom = float(boxAArea + boxBArea - interArea)
        if denom <= 0:
            return 0.0
        return interArea / denom


# ─── Drawing Overlay ────────────────────────────────────────────────────────────

def draw_video_overlay(frame: np.ndarray, dema_result: dict, fps: float = 0.0, frame_num: int = 0) -> np.ndarray:
    """
    Draws detection boxes, alert banners, and a professional FPS/status overlay
    optimized for video output and presentation demos.
    """
    annotated = frame.copy()
    matching_result = dema_result["matching_result"]
    color_bgr = dema_result["color_bgr"]
    alert_msg = dema_result["alert_message"]

    # Draw Bounding Boxes for All Detections
    all_dets = matching_result.get("all_detections", matching_result.get("detected_ppe", []))

    for det in all_dets:
        raw_label = det.get("raw_label", det.get("normalized_label", ""))
        norm_label = det["normalized_label"]
        prob = det["similarity_probability"]
        bbox = det["bbox"]

        xmin, ymin, xmax, ymax = [int(v) for v in bbox]

        # Determine Box Color & Tag
        if norm_label.lower() in ["person", "human"]:
            box_color = (255, 191, 0)
            tag = f"Person ({prob:.0%})"
        elif norm_label.startswith("NO-"):
            box_color = (0, 0, 255)
            tag = f"WARNING: {raw_label} ({prob:.0%})"
        elif norm_label in TARGET_CLASSES or norm_label.lower() in ["hardhat", "helmet", "footwear", "boots"]:
            box_color = (0, 255, 0)
            tag = f"{raw_label} ({prob:.0%})"
        else:
            box_color = (220, 220, 0)
            tag = f"{raw_label} ({prob:.0%})"

        # Draw thicker bounding boxes for video visibility
        cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), box_color, 3)

        # Label tag background & text
        (w, h), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (xmin, max(0, ymin - 28)), (xmin + w + 12, max(0, ymin)), box_color, -1)
        text_color = (0, 0, 0) if box_color in [(0, 255, 0), (255, 191, 0), (220, 220, 0)] else (255, 255, 255)
        cv2.putText(annotated, tag, (xmin + 5, max(15, ymin - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

    # ── Top Alert Banner ──
    banner_height = 50
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (annotated.shape[1], banner_height), color_bgr, -1)
    cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)
    cv2.putText(annotated, alert_msg, (15, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # ── Bottom Status Bar (FPS + Frame Counter) ──
    h_frame = annotated.shape[0]
    w_frame = annotated.shape[1]
    status_bar_y = h_frame - 35
    
    overlay2 = annotated.copy()
    cv2.rectangle(overlay2, (0, status_bar_y), (w_frame, h_frame), (30, 30, 30), -1)
    cv2.addWeighted(overlay2, 0.7, annotated, 0.3, 0, annotated)
    
    fps_text = f"FPS: {fps:.1f}"
    frame_text = f"Frame: {frame_num}"
    model_text = "PPE DeMa Detection | Video Enhanced"
    
    cv2.putText(annotated, fps_text, (15, h_frame - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1, cv2.LINE_AA)
    cv2.putText(annotated, frame_text, (150, h_frame - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(annotated, model_text, (w_frame - 380, h_frame - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

    return annotated


# ─── Load Model ─────────────────────────────────────────────────────────────────

def load_detector(model_type: str, weights_path: str = None, conf_thresh: float = 0.20, enable_tiling: bool = True, person_weights: str = None, person_detector: str = "rcnn"):
    """
    Loads the appropriate model backend.
    """
    grid = (2, 2) if enable_tiling else (1, 1)

    if model_type == "dino":
        from dino_pipeline.dino_adapter import DINOAdapter
        classifier_path = "models/dino_classifier.pt" if os.path.exists("models/dino_classifier.pt") else None
        detector = DINOAdapter(
            classifier_weights_path=classifier_path,
            person_threshold=max(0.30, conf_thresh),
            ppe_threshold=conf_thresh,
            tile_grid=grid
        )
        print(f"[OK] Loaded DINO Two-Stage Pipeline (best for CCTV)")
        return detector
    elif model_type == "optimized-dino":
        # Load the new optimized architecture side-by-side
        from dino_pipeline.optimized_adapter import DINOAdapterOptimized

        # Check if distilled student weights exist — if so, use the fast backbone
        student_weights = "models/student_backbone.pt"
        p2_weights = person_weights

        if os.path.exists(student_weights):
            # Distilled mode: use MobileNetV3 backbone + its own classifier
            classifier_path = "models/dino_classifier_distilled.pt" if os.path.exists("models/dino_classifier_distilled.pt") else None
            detector = DINOAdapterOptimized(
                classifier_weights_path=classifier_path,
                person_threshold=max(0.20, conf_thresh),
                ppe_threshold=conf_thresh,
                tile_grid=grid,
                backbone_mode="distilled",
                student_weights_path=student_weights,
                person_detector_type=person_detector,
                person_weights_path=p2_weights
            )
            print(f"[OK] Loaded FAST Distilled Architecture (MobileNetV3 student)!")
        else:
            # Fallback: use real DINOv2 backbone + original classifier
            classifier_path = "models/dino_classifier.pt" if os.path.exists("models/dino_classifier.pt") else None
            finetuned_dino_path = "models/dino_backbone_finetuned.pt" if os.path.exists("models/dino_backbone_finetuned.pt") else None
            detector = DINOAdapterOptimized(
                classifier_weights_path=classifier_path,
                person_threshold=max(0.20, conf_thresh),
                ppe_threshold=conf_thresh,
                tile_grid=grid,
                backbone_mode="dino",
                dino_weights_path=finetuned_dino_path,
                person_detector_type=person_detector,
                person_weights_path=p2_weights
            )
            if finetuned_dino_path:
                print(f"[OK] Loaded Optimized DINO Architecture (FINE-TUNED DINOv2 backbone)!")
            else:
                print(f"[OK] Loaded Optimized DINO Architecture (DINOv2 backbone)!")
        return detector
    elif model_type == "yolo":
        from models.yolo_adapter import YOLOv8Adapter
        yolo_w = weights_path if weights_path else "models/best.pt"
        detector = YOLOv8Adapter(weights_path=yolo_w, conf_threshold=conf_thresh)
        print(f"[OK] Loaded YOLOv8 Adapter")
        return detector
    else:
        from models.ssd_vgg16 import SSDVGG16Adapter
        ssd_w = weights_path if weights_path else "models/best_ssd_vgg16.pt"
        detector = SSDVGG16Adapter(weights_path=ssd_w, conf_threshold=conf_thresh)
        print(f"[OK] Loaded SSD-VGG16 Paper Model")
        return detector


# ─── Main Video Detection Loop ──────────────────────────────────────────────────

def run_video_detection(
    source: str,
    weights_path: str = None,
    conf_thresh: float = 0.20,
    hazard_level: str = "Level_2_High_Risk",
    model_type: str = "optimized-dino",
    save_output: bool = True,
    show_window: bool = True,
    enable_preprocess: bool = True,
    enable_multiscale: bool = True,
    enable_temporal: bool = True,
    frame_skip: int = 2,
    denoise: bool = False,
    person_weights: str = None,
    person_detector: str = "rcnn",
):
    """
    Enhanced video detection pipeline with preprocessing, multi-scale, and temporal smoothing.
    """
    print("=" * 65)
    print("  Substation PPE Detection — Video Enhanced Mode")
    print("=" * 65)
    print(f"  Model:          {model_type.upper()}")
    if model_type == "optimized-dino":
        pd_name = "Faster R-CNN MobileNetV3 Small (RCNN)" if person_detector == "rcnn" else "YOLOv8"
        print(f"  Person Detector:{pd_name}")
    print(f"  Source:          {source}")
    print(f"  Hazard Level:   {hazard_level}")
    print(f"  Preprocessing:  {'ON' if enable_preprocess else 'OFF'}")
    print(f"  Multi-Scale:    {'ON' if enable_multiscale else 'OFF'}")
    print(f"  Temporal:       {'ON' if enable_temporal else 'OFF'}")
    print(f"  Frame Skip:     {frame_skip}")
    print(f"  Conf Threshold: {conf_thresh}")
    if person_weights:
        print(f"  Person Weights: {person_weights}")
    print("=" * 65 + "\n")

    # ── Initialize Components ──
    detector = load_detector(model_type, weights_path, conf_thresh, enable_tiling=enable_multiscale, person_weights=person_weights, person_detector=person_detector)

    dema_engine = PPEDeMaEngine(
        default_level=hazard_level,
        min_similarity_thresh=conf_thresh,
        temporal_window=25  # Increased for video stability
    )

    preprocessor = VideoFramePreprocessor(
        enable_clahe=enable_preprocess,
        enable_denoise=denoise,     # Off by default (slow), enable with --denoise
        enable_sharpen=enable_preprocess,
    ) if enable_preprocess else None

    skip_controller = FrameSkipController(base_skip=frame_skip)
    temporal_smoother = TemporalDetectionSmoother(
        window_size=10,
        min_confirmations=2,
        iou_threshold=0.25
    ) if enable_temporal else None

    # Multi-scale config
    if enable_multiscale and model_type not in ["dino", "optimized-dino"]:
        # DINO already has tiling built in, skip multi-scale for it
        ms_scales = [1.0, 0.6]
    else:
        ms_scales = [1.0]

    # ── Open Video Source ──
    is_webcam = str(source).isdigit()
    src_val = int(source) if is_webcam else source
    cap = cv2.VideoCapture(src_val)

    if not cap.isOpened():
        print(f"[ERROR] Unable to open video source '{source}'.")
        return

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 800
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 600
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    print(f"[INFO] Video: {orig_w}x{orig_h} @ {src_fps:.1f} FPS, ~{total_frames} frames")
    if show_window:
        print("[INFO] Controls: [SPACE / P] = Pause/Resume | [D / N] = Step 1 Frame | [Q] = Quit")

    # ── Output Writer ──
    writer = None
    if save_output:
        os.makedirs("output", exist_ok=True)
        save_path = os.path.join("output", "dema_video_enhanced_output.mp4")
        writer = cv2.VideoWriter(
            save_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            src_fps,
            (orig_w, orig_h)
        )
        print(f"[INFO] Saving output to: {save_path}")

    # ── Detection Loop ──
    fps_history = []
    frame_num = 0
    processed_count = 0
    last_dema_result = None
    is_paused = False

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1
            start_time = time.time()

            # Check if we should process this frame or skip it
            should_process = skip_controller.should_process(frame)

            if should_process:
                # ── Step 1: Preprocess ──
                if preprocessor:
                    proc_frame = preprocessor.preprocess(frame)
                else:
                    proc_frame = frame

                # ── Step 2: Detect (optionally multi-scale) ──
                if enable_multiscale and model_type not in ["dino", "optimized-dino"]:
                    raw_detections = run_multiscale_detection(
                        detector, proc_frame, model_type, scales=ms_scales
                    )
                else:
                    raw_detections = detector.predict_frame(proc_frame)

                # Only allow specific target classes (person, helmet, footwear, gloves)
                allowed_keywords = ["person", "human", "helmet", "hardhat", "footwear", "boots", "shoe", "glove"]
                filtered_dets = []
                for d in raw_detections:
                    l = d.get("label", "").lower()
                    # Check if any allowed keyword is in the label
                    if any(kw in l for kw in allowed_keywords):
                        filtered_dets.append(d)
                raw_detections = filtered_dets

                # ── Step 3: Temporal Smoothing ──
                if temporal_smoother:
                    raw_detections = temporal_smoother.update(raw_detections)

                # ── Step 4: DeMa Algorithm ──
                dema_result = dema_engine.process_frame_pipeline(
                    raw_detections, hazard_level=hazard_level
                )
                last_dema_result = dema_result
                processed_count += 1

            else:
                # Skipped frame — reuse last result
                if last_dema_result is None:
                    continue
                dema_result = last_dema_result

            # ── Step 5: Render Overlay ──
            elapsed = time.time() - start_time
            if elapsed > 0:
                fps_history.append(1.0 / elapsed)
                if len(fps_history) > 30:
                    fps_history.pop(0)
            avg_fps = sum(fps_history) / max(len(fps_history), 1)

            annotated = draw_video_overlay(frame, dema_result, fps=avg_fps, frame_num=frame_num)

            if writer is not None:
                annotated_resized = cv2.resize(annotated, (orig_w, orig_h)) if annotated.shape[:2] != (orig_h, orig_w) else annotated
                writer.write(annotated_resized)

            if show_window:
                display = cv2.resize(annotated, (min(orig_w, 1280), min(orig_h, 720)))
                cv2.imshow("PPE DeMa Detection - Video Enhanced", display)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key in [ord(' '), ord('p'), ord('P')]:
                    is_paused = True

                if is_paused:
                    # Draw pause status overlay bar at bottom
                    h, w = display.shape[:2]
                    banner_w = min(620, w - 24)
                    cv2.rectangle(display, (12, h - 45), (12 + banner_w, h - 10), (0, 0, 0), -1)
                    cv2.rectangle(display, (12, h - 45), (12 + banner_w, h - 10), (0, 215, 255), 2)
                    cv2.putText(
                        display,
                        "[PAUSED] SPACE / P: Resume | D / N: Next Frame | Q: Quit",
                        (22, h - 22),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.50,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow("PPE DeMa Detection - Video Enhanced", display)

                    quit_requested = False
                    while is_paused:
                        k = cv2.waitKey(30) & 0xFF
                        if k == ord('q'):
                            quit_requested = True
                            break
                        elif k in [ord(' '), ord('p'), ord('P')]:
                            is_paused = False
                            break
                        elif k in [ord('d'), ord('D'), ord('n'), ord('N')]:
                            # Advance single frame while staying paused
                            break

                    if quit_requested:
                        break

            # Progress log
            if frame_num % 50 == 0:
                pct = (frame_num / total_frames * 100) if total_frames > 0 else 0
                print(f"  Frame {frame_num}/{total_frames} ({pct:.0f}%) | FPS: {avg_fps:.1f} | Processed: {processed_count}")

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    print(f"\n[DONE] Processed {processed_count}/{frame_num} frames.")
    if save_output:
        print(f"[SAVE] Output saved to: output/dema_video_enhanced_output.mp4")


# ─── CLI ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enhanced PPE Detection for CCTV/Video Footage",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--source", type=str, default="0",
                        help="Video source: 0 (webcam), file.mp4, or rtsp:// URL")
    parser.add_argument("--weights", type=str, default=None,
                        help="Model weights path")
    parser.add_argument("--model-type", type=str, default="yolo",
                        choices=["ssd", "yolo", "dino", "optimized-dino"],
                        help="Model: 'ssd' (paper), 'yolo' (fast), 'dino' (accurate), 'optimized-dino' (fast \u0026 accurate)")
    parser.add_argument("--conf", type=float, default=0.20,
                        help="Confidence threshold (lower for video, default 0.20)")
    parser.add_argument("--level", type=str, default="Level_2_High_Risk",
                        help="DeMa Hazard Level")
    parser.add_argument("--save", action="store_true",
                        help="Save output video to output/")
    parser.add_argument("--no-show", action="store_true",
                        help="Disable GUI display window")
    parser.add_argument("--no-preprocess", action="store_true",
                        help="Disable frame preprocessing (CLAHE + sharpening)")
    parser.add_argument("--no-multiscale", action="store_true",
                        help="Disable multi-scale detection")
    parser.add_argument("--no-temporal", action="store_true",
                        help="Disable temporal smoothing")
    parser.add_argument("--denoise", action="store_true",
                        help="Enable denoising (slower but reduces compression artifacts)")
    parser.add_argument("--frame-skip", type=int, default=2,
                        help="Base frame skip interval (1=every frame, 2=every other, etc.)")
    parser.add_argument("--person-weights", type=str, default=None,
                        help="Optional custom trained YOLO person detector weights (e.g. models/yolov8s_p2_best.pt)")
    parser.add_argument("--person-detector", type=str, default="rcnn", choices=["rcnn", "yolo"],
                        help="Person detector for optimized-dino: 'rcnn' (Faster R-CNN MobileNetV3) or 'yolo' (YOLOv8, default 'rcnn')")

    args = parser.parse_args()

    run_video_detection(
        source=args.source,
        weights_path=args.weights,
        conf_thresh=args.conf,
        hazard_level=args.level,
        model_type=args.model_type,
        save_output=args.save,
        show_window=not args.no_show,
        enable_preprocess=not args.no_preprocess,
        enable_multiscale=not args.no_multiscale,
        enable_temporal=not args.no_temporal,
        frame_skip=args.frame_skip,
        denoise=args.denoise,
        person_weights=args.person_weights,
        person_detector=args.person_detector,
    )
