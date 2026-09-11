import cv2
import threading
import time
import queue
import argparse
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dino_pipeline.dino_adapter import DINOAdapter
from models.yolo_adapter import YOLOv8Adapter
from dema_engine.dema_algorithm import PPEDeMaEngine
from data.video_preprocessor import VideoFramePreprocessor
from detect_video import TemporalDetectionSmoother

class CameraStream:
    def __init__(self, cam_id, source):
        self.cam_id = cam_id
        self.source = source
        self.cap = cv2.VideoCapture(source)
        self.q = queue.Queue(maxsize=2)
        self.running = True
        self.latest_frame = None
        
        if not self.cap.isOpened():
            print(f"[ERROR] Camera {cam_id} failed to open source: {source}")
            self.running = False
            
        self.thread = threading.Thread(target=self._update, daemon=True)
        if self.running:
            self.thread.start()
            
    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print(f"[WARNING] Camera {self.cam_id} stream lost.")
                self.running = False
                break
                
            self.latest_frame = frame
            # Throttle slightly if needed, or rely on main thread pulling
            time.sleep(0.01) 
            
    def read_latest(self):
        return self.latest_frame
        
    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        if self.thread.is_alive():
            self.thread.join()

def custom_draw_overlay(frame, dema_result):
    annotated = frame.copy()
    matching_result = dema_result["matching_result"]
    color_bgr = dema_result["color_bgr"]
    alert_msg = dema_result["alert_message"]
    
    all_dets = matching_result.get("all_detections", matching_result.get("detected_ppe", []))
    
    # Group detections by their bounding box coordinates
    box_groups = {}
    for det in all_dets:
        box_tuple = tuple(map(int, det["bbox"]))
        if box_tuple not in box_groups:
            box_groups[box_tuple] = []
        box_groups[box_tuple].append(det)
        
    for box, dets in box_groups.items():
        xmin, ymin, xmax, ymax = box
        
        # Build a combined tag (e.g., "Person (100%) | Helmet (85%)")
        tag_parts = []
        box_color = (255, 191, 0) # Default cyan/yellow for Person
        
        for det in dets:
            raw_label = det.get("raw_label", det.get("normalized_label", ""))
            prob = det["similarity_probability"]
            tag_parts.append(f"{raw_label} ({prob:.0%})")
            
            # If there's a PPE detection or WARNING, change the box color
            norm_label = det.get("normalized_label", "")
            if norm_label.startswith("NO-"):
                box_color = (0, 0, 255)
            elif norm_label.lower() in ["helmet", "footwear", "gloves", "hardhat", "boots"]:
                box_color = (0, 255, 0)
                
        tag_str = " | ".join(tag_parts)
        
        # Draw bounding box
        cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), box_color, 2)
        
        # Label tag background & text
        (w, h), _ = cv2.getTextSize(tag_str, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(annotated, (xmin, max(0, ymin - 25)), (xmin + w + 10, max(0, ymin)), box_color, -1)
        text_color = (0, 0, 0) if box_color in [(0, 255, 0), (255, 191, 0), (220, 220, 0)] else (255, 255, 255)
        cv2.putText(annotated, tag_str, (xmin + 5, max(15, ymin - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 2)
                    
    # System Warning Banner
    banner_height = 45
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], banner_height), color_bgr, -1)
    cv2.putText(annotated, alert_msg, (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
                
    return annotated

def run_multi_camera_system(sources, hazard_level="Level_2_High_Risk", display=True, model_type="yolo"):
    num_cameras = len(sources)
    print(f"[INFO] Initializing {num_cameras} Camera Streams...")
    
    streams = []
    for i, src in enumerate(sources):
        src_val = int(src) if str(src).isdigit() else src
        streams.append(CameraStream(i, src_val))
        
    # Wait for streams to initialize
    time.sleep(2)
    
    if model_type.lower() == "yolo":
        print("[INFO] Loading YOLOv8 for Multi-Camera Batch Inference (Best for 60 cameras)...")
        adapter = YOLOv8Adapter(weights_path="models/best.pt", conf_threshold=0.20)
    else:
        print("[INFO] Loading DINO Two-Stage Pipeline...")
        classifier_path = "models/dino_classifier.pt" if os.path.exists("models/dino_classifier.pt") else None
        adapter = DINOAdapter(classifier_weights_path=classifier_path)
    
    # Initialize one DeMa Engine per camera
    dema_engines = [PPEDeMaEngine(default_level=hazard_level, temporal_window=25) for _ in range(num_cameras)]
    
    # Initialize Preprocessors and Smoothers per camera to guarantee high accuracy
    preprocessors = [VideoFramePreprocessor(enable_denoise=False) for _ in range(num_cameras)]
    smoothers = [TemporalDetectionSmoother(window_size=5, min_confirmations=2) for _ in range(num_cameras)]
    
    print("[INFO] Multi-Camera System Running. Press 'q' in any window to quit.")
    
    try:
        fps_history = []
        while True:
            start_time = time.time()
            
            # 1. Pull latest frame from all active cameras
            batch_frames = []
            display_frames = []
            active_indices = []
            
            for i, stream in enumerate(streams):
                if stream.running:
                    frame = stream.read_latest()
                    if frame is not None:
                        # Preprocess frame (CLAHE + Sharpen) to boost accuracy in bad lighting
                        proc_frame = preprocessors[i].preprocess(frame.copy())
                        batch_frames.append(proc_frame)
                        display_frames.append(frame.copy())
                        active_indices.append(i)
                        
            if not batch_frames:
                print("[INFO] All camera streams closed.")
                break
                
            # 2. Run Massive Batch Inference
            # YOLOv8 will process all 60 frames in a single GPU batch, making it vastly faster
            batch_results = adapter.predict_batch_frames(batch_frames)
            
            # 3. Process DeMa Rules & Display per camera
            for idx_in_batch, cam_idx in enumerate(active_indices):
                frame = display_frames[idx_in_batch]
                raw_dets = batch_results[idx_in_batch]
                
                # Filter out irrelevant classes
                filtered_dets = [d for d in raw_dets if not ("vest" in d.get("label", "").lower() or "mask" in d.get("label", "").lower())]
                
                # Temporal smoothing to prevent flickering accuracy
                smoothed_dets = smoothers[cam_idx].update(filtered_dets)
                
                # Run DeMa
                dema_res = dema_engines[cam_idx].process_frame_pipeline(smoothed_dets, hazard_level=hazard_level)
                
                # Render using the custom overlay drawing function
                annotated = custom_draw_overlay(frame, dema_res)
                
                # Display
                if display and cam_idx < 9: # Only show first 9 cameras to avoid crashing the OS UI
                    display_frame = cv2.resize(annotated, (400, 300))
                    cv2.imshow(f"Camera {cam_idx}", display_frame)
                    
            elapsed = time.time() - start_time
            if elapsed > 0:
                fps_history.append(1.0 / elapsed)
                if len(fps_history) > 30:
                    fps_history.pop(0)
            avg_fps = sum(fps_history) / max(len(fps_history), 1)
            
            # This is the System Loop FPS, not per-camera FPS. 
            # Total processed frames per second = avg_fps * len(batch_frames)
            print(f"System Loop FPS: {avg_fps:.2f} (Total Frames/Sec: {avg_fps * len(batch_frames):.1f} across {len(batch_frames)} cameras)")
            
            # For 60 cameras, sleep to prevent locking up the CPU/GPU and to run at ~1-2 FPS per camera
            time.sleep(0.5)
            
            if display and cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        for s in streams:
            s.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", required=True, help="List of video sources (e.g. 0 1 video.mp4 rtsp://...)")
    parser.add_argument("--level", default="Level_2_High_Risk", help="DeMa Hazard Level")
    parser.add_argument("--model", default="yolo", choices=["yolo", "dino"], help="Model architecture")
    args = parser.parse_args()
    
    run_multi_camera_system(args.sources, hazard_level=args.level, model_type=args.model)
