"""
Real-time PPE Detection and DeMa Safety Compliance Monitoring Script.
Supports image, video, webcam, or RTSP feeds.
Target PPE: Helmet, Gloves, Footwear
"""

import argparse
import os
import time
import cv2
import numpy as np
import torch

from dema_engine.dema_algorithm import PPEDeMaEngine
from dema_engine.ppe_rules import TARGET_CLASSES
from models.ssd_vgg16 import SSDVGG16Adapter
from models.yolo_adapter import YOLOv8Adapter

def draw_detection_overlay(frame: np.ndarray, dema_result: dict) -> np.ndarray:
    """
    Draws bounding boxes for Person, detected PPE items, and non-compliance alerts,
    along with system warning alert banners onto the frame.
    """
    annotated = frame.copy()
    matching_result = dema_result["matching_result"]
    color_bgr = dema_result["color_bgr"]
    alert_msg = dema_result["alert_message"]

    # Draw Bounding Boxes for All Detections (Person, Helmet, Gloves, Footwear, Mask, Safety Vest, etc.)
    all_dets = matching_result.get("all_detections", matching_result.get("detected_ppe", []))

    for det in all_dets:
        raw_label = det.get("raw_label", det.get("normalized_label", ""))
        norm_label = det["normalized_label"]
        prob = det["similarity_probability"]
        bbox = det["bbox"]

        xmin, ymin, xmax, ymax = [int(v) for v in bbox]

        # Determine Box Color & Tag
        if norm_label.lower() in ["person", "human"]:
            box_color = (255, 191, 0) # Deep Cyan/Yellow-Blue for Person detection
            tag = f"Person ({prob:.0%})"
        elif norm_label.startswith("NO-"):
            box_color = (0, 0, 255) # Bright Red for violations
            tag = f"WARNING: {raw_label} ({prob:.0%})"
        elif norm_label in TARGET_CLASSES or norm_label.lower() in ["hardhat", "helmet", "safety vest", "mask", "gloves", "footwear", "boots"]:
            box_color = (0, 255, 0) # Green for detected PPE
            tag = f"{raw_label} ({prob:.0%})"
        else:
            box_color = (220, 220, 0) # Yellow for general objects
            tag = f"{raw_label} ({prob:.0%})"

        # Draw bounding box
        cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), box_color, 2)

        # Label tag background & text
        (w, h), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(annotated, (xmin, max(0, ymin - 25)), (xmin + w + 10, max(0, ymin)), box_color, -1)
        text_color = (0, 0, 0) if box_color in [(0, 255, 0), (255, 191, 0), (220, 220, 0)] else (255, 255, 255)
        cv2.putText(annotated, tag, (xmin + 5, max(15, ymin - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 2)

    # System Warning Banner (Top Status Bar)
    banner_height = 45
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], banner_height), color_bgr, -1)
    cv2.putText(annotated, alert_msg, (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    return annotated

def run_detection_stream(
    source: str, 
    weights_path: str = None, 
    conf_thresh: float = 0.25,
    hazard_level: str = "Level_2_High_Risk",
    model_type: str = "ssd",
    save_output: bool = True,
    show_window: bool = True
):
    model_name = "PyTorch SSD-VGG16 (Paper Model)" if model_type.lower() == "ssd" else "YOLOv8 Adapter"
    print(f"[INFO] Initializing PPE DeMa Detection System using {model_name}...")
    print(f"[INFO] Target Categories: Helmet, Gloves, Footwear")
    print(f"[INFO] Hazard Level: {hazard_level}")

    dema_engine = PPEDeMaEngine(default_level=hazard_level, min_similarity_thresh=conf_thresh)
    
    if model_type.lower() == "yolo":
        yolo_w = weights_path if (weights_path and not weights_path.endswith("best_ssd_vgg16.pt")) else "models/best.pt"
        detector = YOLOv8Adapter(weights_path=yolo_w, conf_threshold=conf_thresh)
    else:
        ssd_w = weights_path if weights_path else "models/best_ssd_vgg16.pt"
        detector = SSDVGG16Adapter(weights_path=ssd_w, conf_threshold=conf_thresh)


    is_webcam = str(source).isdigit()
    src_val = int(source) if is_webcam else source
    cap = cv2.VideoCapture(src_val)

    if not cap.isOpened():
        print(f"[ERROR] Unable to open video source '{source}'.")
        return

    writer = None
    if save_output:
        os.makedirs("output", exist_ok=True)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 800
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 600
        save_path = os.path.join("output", "dema_detection_output.mp4")
        writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 25.0, (w, h))

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Step 1: Detect objects using paper's SSD-VGG16 detector
            raw_detections = detector.predict_frame(frame)

            # Step 2 & 3: Run DeMa Algorithm
            dema_res = dema_engine.process_frame_pipeline(raw_detections, hazard_level=hazard_level)

            # Render overlay (draw boxes for Person, PPE, & Alerts)
            annotated_frame = draw_detection_overlay(frame, dema_res)

            if writer is not None:
                writer.write(annotated_frame)

            if show_window:
                cv2.imshow("Substation PPE DeMa Detection (SSD-VGG16 Paper Model)", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        print("[INFO] Stream processing finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Substation Safety PPE Detection")
    parser.add_argument("--source", type=str, default="0", help="Video source (0 for webcam, file.mp4, or RTSP url)")
    parser.add_argument("--weights", type=str, default=None, help="Model weights path (defaults based on --model-type)")
    parser.add_argument("--model-type", type=str, default="ssd", choices=["ssd", "yolo"], help="Model architecture: 'ssd' (Paper Model) or 'yolo'")
    parser.add_argument("--conf", type=float, default=0.25, help="Similarity threshold")
    parser.add_argument("--level", type=str, default="Level_2_High_Risk", help="Hazard Level (Level_1_Standard, Level_2_High_Risk)")
    parser.add_argument("--save", action="store_true", help="Save output video")
    parser.add_argument("--show", action="store_true", default=True, help="Show GUI display window")
    parser.add_argument("--no-show", action="store_true", help="Disable GUI display window")

    args = parser.parse_args()
    show_window = not args.no_show if args.no_show else args.show

    weights_path = args.weights
    if weights_path is None:
        weights_path = "models/best.pt" if args.model_type.lower() == "yolo" else "models/best_ssd_vgg16.pt"

    run_detection_stream(
        source=args.source,
        weights_path=weights_path,
        conf_thresh=args.conf,
        hazard_level=args.level,
        model_type=args.model_type,
        save_output=args.save,
        show_window=show_window
    )



