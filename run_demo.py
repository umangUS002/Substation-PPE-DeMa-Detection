"""
Self-contained Verification Demo Script.
Demonstrates the 3-step PPE DeMa Algorithm (Determination, Matching, System Warning)
for Helmet, Gloves, and Footwear safety compliance.
"""

import os
import cv2
import numpy as np
from dema_engine.dema_algorithm import PPEDeMaEngine
from detect import draw_detection_overlay

def create_synthetic_worker_image(detected_items: list, title: str) -> np.ndarray:
    """Creates a synthetic test image representing a worker with specified PPE items."""
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    img[:] = (45, 45, 45) # Dark gray background

    # Draw Worker silhouette box
    cv2.rectangle(img, (250, 100), (550, 550), (90, 90, 90), -1)
    cv2.circle(img, (400, 180), 50, (120, 120, 120), -1) # Head
    cv2.putText(img, title, (20, 580), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

    return img

def run_verification_demo():
    print("=" * 60)
    print("  Substation Safety PPE Detection & DeMa Algorithm Demo")
    print("  Focusing on: Helmet, Gloves, Footwear")
    print("=" * 60 + "\n")

    os.makedirs("output", exist_ok=True)
    dema_engine = PPEDeMaEngine(default_level="Level_2_High_Risk", min_similarity_thresh=0.50)

    # Test Case 1: Fully Compliant Worker (Has Helmet, Gloves, Footwear)
    raw_detections_compliant = [
        {"label": "Helmet", "confidence": 0.98, "bbox": [350, 130, 450, 190]},
        {"label": "Gloves", "confidence": 0.89, "bbox": [230, 330, 290, 400]},
        {"label": "Gloves", "confidence": 0.91, "bbox": [510, 330, 570, 400]},
        {"label": "Footwear", "confidence": 0.94, "bbox": [320, 500, 480, 550]}
    ]

    # Test Case 2: Non-Compliant Worker (Missing Gloves & Footwear)
    raw_detections_violating = [
        {"label": "Helmet", "confidence": 0.96, "bbox": [350, 130, 450, 190]},
        {"label": "NO-Gloves", "confidence": 0.85, "bbox": [230, 330, 290, 400]}
    ]

    scenarios = [
        ("Scenario 1: Fully Compliant Worker", raw_detections_compliant, "compliant_worker_output.jpg"),
        ("Scenario 2: Non-Compliant Worker (Missing Gloves & Footwear)", raw_detections_violating, "non_compliant_worker_output.jpg")
    ]

    for title, detections, filename in scenarios:
        print(f"[INFO] Testing {title}...")
        
        # Run 3-Step DeMa Algorithm
        dema_result = dema_engine.process_frame_pipeline(detections, hazard_level="Level_2_High_Risk")
        
        print(f"   Status: {dema_result['status_code']}")
        print(f"   Alert Message: {dema_result['alert_message']}")
        print(f"   Detected Items: {dema_result['matching_result']['detected_set']}")
        print(f"   Missing Items: {dema_result['matching_result']['missing_ppe']}\n")

        # Render visual verification image
        base_img = create_synthetic_worker_image(detections, title)
        annotated_img = draw_detection_overlay(base_img, dema_result)

        save_path = os.path.join("output", filename)
        cv2.imwrite(save_path, annotated_img)
        print(f"   [SAVED] Output image saved to: '{save_path}'\n" + "-" * 50)

    print("[SUCCESS] Verification Demo finished successfully!")

if __name__ == "__main__":
    run_verification_demo()

