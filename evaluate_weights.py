import cv2
import os
import glob
import torch
import numpy as np
from models.ssd_vgg16 import SSDVGG16Adapter
from dema_engine.dema_algorithm import PPEDeMaEngine
from detect import draw_detection_overlay

def evaluate_models():
    os.makedirs("output", exist_ok=True)
    images = glob.glob("data/my_dataset/images/*.jpg")
    print(f"Total available images in dataset: {len(images)}", flush=True)
    
    # Pick a representative sample of 4 test images
    sample_images = images[:4]
    
    models = {
        "train1.pt": "models/train1.pt",
        "train2.pt": "models/train2.pt"
    }
    
    dema_engine = PPEDeMaEngine(default_level="Level_2_High_Risk", min_similarity_thresh=0.20)
    
    stats = {}
    
    for name, path in models.items():
        print(f"\n{'='*50}\nEvaluating Model: {name} (Weights: {path})\n{'='*50}")
        adapter = SSDVGG16Adapter(weights_path=path, conf_threshold=0.20)
        
        total_detections = 0
        class_counts = {"Helmet": 0, "Footwear": 0}
        
        for idx, img_p in enumerate(sample_images):
            frame = cv2.imread(img_p)
            if frame is None:
                continue
            
            raw_dets = adapter.predict_frame(frame)
            dema_res = dema_engine.process_frame_pipeline(raw_dets, hazard_level="Level_2_High_Risk")
            
            total_detections += len(raw_dets)
            for d in raw_dets:
                lbl = d["label"]
                class_counts[lbl] = class_counts.get(lbl, 0) + 1
            
            # Draw visual overlay
            annotated = draw_detection_overlay(frame, dema_res)
            out_filename = f"output/eval_{name.replace('.pt', '')}_sample_{idx+1}.jpg"
            cv2.imwrite(out_filename, annotated)
            
            base_name = os.path.basename(img_p)
            print(f"Sample {idx+1} [{base_name[:25]}...]:")
            print(f"  - Detected Objects ({len(raw_dets)}):")
            for d in raw_dets:
                print(f"      * {d['label']} (Confidence: {d['confidence']:.1%}) at {d['bbox']}")
            print(f"  - DeMa Decision: {dema_res['alert_message']}")
            print(f"  - Saved visual overlay: {out_filename}")
            
        stats[name] = {
            "total_dets": total_detections,
            "class_counts": class_counts
        }
        
    print("\n" + "="*50)
    print("SUMMARY COMPARISON")
    print("="*50)
    for name, s in stats.items():
        print(f"Model: {name} | Total Detections on sample: {s['total_dets']} | Counts: {s['class_counts']}")

if __name__ == "__main__":
    evaluate_models()
