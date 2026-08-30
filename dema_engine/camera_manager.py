"""
Multi-Camera Stream & Inference Engine for Industrial Plant PPE Compliance Monitoring.
Manages 20+ simultaneous camera channels across industrial plant zones.
Integrates 3-Step DeMa Engine (Determination, Matching, Warning) with YOLOv8 trained model,
real-time live webcam/CCTV hardware support, and automated violation snapshot logging.
"""

import os
import time
import math
import random
import cv2
import numpy as np
import threading
import asyncio
from typing import List, Dict, Any, Optional, Tuple

from dema_engine.dema_algorithm import PPEDeMaEngine
from dema_engine.ppe_rules import TARGET_CLASSES, HAZARD_LEVEL_REQUIREMENTS
from models.yolo_adapter import YOLOv8Adapter
from database.db_logger import (
    init_db, seed_default_cameras,
    log_violation_event, increment_camera_scans, get_all_cameras
)

# Snapshots directory
SNAPSHOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "snapshots")
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)

# 24 Default Industrial Plant Cameras across critical hazard zones (CAM-01 defaults to Live Webcam)
DEFAULT_24_CAMERAS = [
    {"camera_id": "CAM-01", "name": "Live Primary CCTV (Your Webcam)", "zone": "Plant Entrance & Gate", "hazard_level": "Level_2_High_Risk", "rtsp_url": "0"},
    {"camera_id": "CAM-02", "name": "Substation Bay 2 - Breakers", "zone": "HV Switchyard", "hazard_level": "Level_2_High_Risk", "rtsp_url": "simulated"},
    {"camera_id": "CAM-03", "name": "Substation Bay 3 - Busbar Area", "zone": "HV Switchyard", "hazard_level": "Level_2_High_Risk", "rtsp_url": "simulated"},
    {"camera_id": "CAM-04", "name": "Transformer Yard East", "zone": "Transformer Yard", "hazard_level": "Level_2_High_Risk", "rtsp_url": "simulated"},
    {"camera_id": "CAM-05", "name": "Transformer Yard West", "zone": "Transformer Yard", "hazard_level": "Level_2_High_Risk", "rtsp_url": "simulated"},
    {"camera_id": "CAM-06", "name": "Control Room Entrance Gate", "zone": "Control Building", "hazard_level": "Level_1_Standard", "rtsp_url": "simulated"},
    {"camera_id": "CAM-07", "name": "Relay & Protection Panel Room", "zone": "Control Building", "hazard_level": "Level_1_Standard", "rtsp_url": "simulated"},
    {"camera_id": "CAM-08", "name": "Battery & UPS Backup Room", "zone": "Control Building", "hazard_level": "Level_2_High_Risk", "rtsp_url": "simulated"},
    {"camera_id": "CAM-09", "name": "Cable Trench Vault North", "zone": "Subterranean Trench", "hazard_level": "Level_2_High_Risk", "rtsp_url": "simulated"},
    {"camera_id": "CAM-10", "name": "Switchgear Room Hallway A", "zone": "Switchgear Hall", "hazard_level": "Level_2_High_Risk", "rtsp_url": "simulated"},
    {"camera_id": "CAM-11", "name": "Switchgear Room Hallway B", "zone": "Switchgear Hall", "hazard_level": "Level_2_High_Risk", "rtsp_url": "simulated"},
    {"camera_id": "CAM-12", "name": "HV Disconnect Switch Bay", "zone": "HV Switchyard", "hazard_level": "Level_2_High_Risk", "rtsp_url": "simulated"},
    {"camera_id": "CAM-13", "name": "Maintenance Overhaul Bay", "zone": "Maintenance Yard", "hazard_level": "Level_2_High_Risk", "rtsp_url": "simulated"},
    {"camera_id": "CAM-14", "name": "Solar Inverter Enclosure", "zone": "Solar Array Field", "hazard_level": "Level_1_Standard", "rtsp_url": "simulated"},
    {"camera_id": "CAM-15", "name": "Diesel Generator Enclosure", "zone": "Auxiliary Power", "hazard_level": "Level_2_High_Risk", "rtsp_url": "simulated"},
    {"camera_id": "CAM-16", "name": "PPE Checkpoint Gate 1", "zone": "Plant Entrance", "hazard_level": "Level_1_Standard", "rtsp_url": "simulated"},
    {"camera_id": "CAM-17", "name": "PPE Checkpoint Gate 2", "zone": "Plant Entrance", "hazard_level": "Level_1_Standard", "rtsp_url": "simulated"},
    {"camera_id": "CAM-18", "name": "Perimeter Wall East", "zone": "Perimeter Yard", "hazard_level": "Level_1_Standard", "rtsp_url": "simulated"},
    {"camera_id": "CAM-19", "name": "Perimeter Wall West", "zone": "Perimeter Yard", "hazard_level": "Level_1_Standard", "rtsp_url": "simulated"},
    {"camera_id": "CAM-20", "name": "Grounding Grid Test Site", "zone": "Testing Field", "hazard_level": "Level_2_High_Risk", "rtsp_url": "simulated"},
    {"camera_id": "CAM-21", "name": "GIS (Gas Insulated) Hall", "zone": "Indoor Substation", "hazard_level": "Level_2_High_Risk", "rtsp_url": "simulated"},
    {"camera_id": "CAM-22", "name": "Safety Gear & Warehouse", "zone": "Logistics Hub", "hazard_level": "Level_1_Standard", "rtsp_url": "simulated"},
    {"camera_id": "CAM-23", "name": "Overhead Line Gantry Alpha", "zone": "HV Switchyard", "hazard_level": "Level_2_High_Risk", "rtsp_url": "simulated"},
    {"camera_id": "CAM-24", "name": "Auxiliary Transformer Yard", "zone": "Transformer Yard", "hazard_level": "Level_2_High_Risk", "rtsp_url": "simulated"},
]


class SyntheticIndustrialFrameGenerator:
    """
    Generates realistic industrial plant video frames with worker overlays,
    PPE detections, and non-compliance simulation.
    """

    @staticmethod
    def generate_frame(
        camera_info: Dict[str, Any],
        tick: int
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
        width, height = 640, 360
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Industrial Background
        cam_num = int(camera_info["camera_id"].split("-")[1]) if "-" in camera_info["camera_id"] else 1
        
        # Grid lines
        for y in range(0, height, 40):
            cv2.line(frame, (0, y), (width, y), (25, 30, 35), 1)
        for x in range(0, width, 40):
            cv2.line(frame, (x, 0), (x, height), (25, 30, 35), 1)

        # Substation Structure
        cv2.rectangle(frame, (50, 100), (220, 300), (45, 55, 65), -1)
        cv2.rectangle(frame, (50, 100), (220, 300), (80, 95, 110), 2)
        cv2.putText(frame, f"EQUIPMENT BAY-{cam_num:02d}", (60, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 200, 220), 1)

        # High Voltage Bushings
        cv2.line(frame, (80, 100), (80, 40), (120, 140, 160), 3)
        cv2.line(frame, (135, 100), (135, 40), (120, 140, 160), 3)
        cv2.line(frame, (190, 100), (190, 40), (120, 140, 160), 3)

        # Worker 1
        worker1_x = int(220 + math.sin(tick * 0.05 + cam_num) * 100)
        worker1_y = 140
        worker_w, worker_h = 90, 170

        cv2.rectangle(frame, (worker1_x, worker1_y), (worker1_x + worker_w, worker1_y + worker_h), (50, 80, 60), 1)
        cv2.circle(frame, (worker1_x + worker_w // 2, worker1_y + 25), 20, (200, 180, 160), -1)
        cv2.rectangle(frame, (worker1_x + 15, worker1_y + 48), (worker1_x + worker_w - 15, worker1_y + 120), (30, 90, 150), -1)

        # Worker 2
        worker2_x = int(450 - math.cos(tick * 0.04 + cam_num) * 60)
        worker2_y = 120
        w2_w, w2_h = 70, 150
        cv2.rectangle(frame, (worker2_x, worker2_y), (worker2_x + w2_w, worker2_y + w2_h), (40, 60, 80), 1)
        cv2.circle(frame, (worker2_x + w2_w // 2, worker2_y + 20), 16, (190, 170, 150), -1)
        cv2.rectangle(frame, (worker2_x + 10, worker2_y + 40), (worker2_x + w2_w - 10, worker2_y + 100), (20, 120, 70), -1)

        # Dynamic PPE violation cycles for simulated channels
        is_violation_cycle = (tick // 15) % 4 == 2

        raw_detections = [
            {"label": "Person", "confidence": 0.94, "bbox": [worker1_x, worker1_y, worker1_x + worker_w, worker1_y + worker_h]},
            {"label": "Person", "confidence": 0.91, "bbox": [worker2_x, worker2_y, worker2_x + w2_w, worker2_y + w2_h]},
            {"label": "Helmet", "confidence": 0.95, "bbox": [worker2_x + 15, worker2_y + 5, worker2_x + w2_w - 15, worker2_y + 35]},
            {"label": "Gloves", "confidence": 0.88, "bbox": [worker2_x + 5, worker2_y + 90, worker2_x + 25, worker2_y + 120]},
            {"label": "Footwear", "confidence": 0.89, "bbox": [worker2_x + 10, worker2_y + 130, worker2_x + w2_w - 10, worker2_y + 150]},
            {"label": "Safety Vest", "confidence": 0.92, "bbox": [worker2_x + 10, worker2_y + 40, worker2_x + w2_w - 10, worker2_y + 100]}
        ]

        if not is_violation_cycle:
            raw_detections.append({"label": "Helmet", "confidence": 0.96, "bbox": [worker1_x + 20, worker1_y + 5, worker1_x + worker_w - 20, worker1_y + 45]})
            raw_detections.append({"label": "Gloves", "confidence": 0.91, "bbox": [worker1_x + 5, worker1_y + 100, worker1_x + 30, worker1_y + 130]})
            raw_detections.append({"label": "Footwear", "confidence": 0.93, "bbox": [worker1_x + 15, worker1_y + 150, worker1_x + worker_w - 15, worker1_y + 170]})
            raw_detections.append({"label": "Safety Vest", "confidence": 0.94, "bbox": [worker1_x + 15, worker1_y + 48, worker1_x + worker_w - 15, worker1_y + 120]})
        else:
            raw_detections.append({"label": "NO-Hardhat", "confidence": 0.92, "bbox": [worker1_x + 20, worker1_y + 5, worker1_x + worker_w - 20, worker1_y + 45]})
            raw_detections.append({"label": "Footwear", "confidence": 0.90, "bbox": [worker1_x + 15, worker1_y + 150, worker1_x + worker_w - 15, worker1_y + 170]})

        meta = {
            "total_detected_workers": 2
        }

        return frame, raw_detections, meta


class CameraChannel:
    """Manages an individual camera feed, live capture, YOLO inference & frame rendering."""

    def __init__(self, camera_info: Dict[str, Any], dema_engine: PPEDeMaEngine, detector: Optional[YOLOv8Adapter] = None):
        self.info = camera_info
        self.camera_id = camera_info["camera_id"]
        self.dema_engine = dema_engine
        self.detector = detector
        self.tick = random.randint(0, 100)
        self.latest_jpeg: Optional[bytes] = None
        self.latest_raw_frame: Optional[np.ndarray] = None
        self.latest_result: Dict[str, Any] = {}
        self.is_running = True
        self.lock = threading.Lock()
        self.last_alert_time = 0.0
        
        # Hardware capture setup
        self.rtsp_url = str(camera_info.get("rtsp_url", "simulated"))
        self.is_live = self.rtsp_url.isdigit() or (self.rtsp_url.startswith(("rtsp://", "http://", "https://")) or self.rtsp_url.endswith((".mp4", ".avi")))
        self.cap: Optional[cv2.VideoCapture] = None
        
        if self.is_live:
            src = int(self.rtsp_url) if self.rtsp_url.isdigit() else self.rtsp_url
            try:
                self.cap = cv2.VideoCapture(src)
                if self.cap.isOpened():
                    print(f"[OK] Opened Live Video Source for {self.camera_id}: '{self.rtsp_url}'")
                else:
                    print(f"[WARN] Could not open live source for {self.camera_id}: '{self.rtsp_url}'. Falling back to synthetic.")
                    self.is_live = False
            except Exception as e:
                print(f"[WARN] Error opening source {self.rtsp_url}: {e}")
                self.is_live = False

    def process_step(self):
        """Executes 1 frame capture/generation, YOLO inference, DeMa matching, snapshot saving, and rendering."""
        self.tick += 1
        raw_dets = []
        worker_count = 1
        frame = None

        if self.is_live and self.cap is not None and self.cap.isOpened():
            ret, captured_frame = self.cap.read()
            if ret and captured_frame is not None:
                frame = cv2.resize(captured_frame, (640, 360))
                # Run real-time YOLO detection
                if self.detector is not None:
                    raw_dets = self.detector.predict_frame(frame)
                
                # Count detected workers/persons
                person_count = sum(1 for d in raw_dets if d.get("label", "").lower() in ["person", "human", "worker"])
                worker_count = max(person_count, 1 if len(raw_dets) > 0 else 0)
            else:
                src = int(self.rtsp_url) if self.rtsp_url.isdigit() else self.rtsp_url
                self.cap.open(src)
        
        if frame is None:
            frame, raw_dets, meta = SyntheticIndustrialFrameGenerator.generate_frame(self.info, self.tick)
            worker_count = meta.get("total_detected_workers", 2)

        # Run 3-Step DeMa Engine Pipeline
        dema_res = self.dema_engine.process_frame_pipeline(
            raw_dets, hazard_level=self.info["hazard_level"]
        )

        matching_res = dema_res["matching_result"]
        is_compliant = matching_res["is_compliant"]
        color_bgr = dema_res["color_bgr"]
        alert_msg = dema_res["alert_message"]

        # Draw Minimalist, Clean Visual Overlay
        annotated = frame.copy()

        # Render Bounding Boxes
        all_dets = matching_res.get("all_detections", matching_res.get("detected_ppe", []))
        for det in all_dets:
            raw_lbl = det.get("raw_label", det.get("normalized_label", ""))
            norm_lbl = det.get("normalized_label", raw_lbl)
            score = det.get("similarity_probability", 0.85)
            bbox = det.get("bbox", [0, 0, 0, 0])
            xmin, ymin, xmax, ymax = [int(v) for v in bbox]

            if norm_lbl.lower() in ["person", "human"]:
                box_color = (255, 191, 0)
                tag = f"Worker ({score:.0%})"
            elif norm_lbl.startswith("NO-") or "NO-" in raw_lbl:
                box_color = (0, 0, 255)
                tag = f"NO {norm_lbl.replace('NO-', '')} ({score:.0%})"
            elif norm_lbl in TARGET_CLASSES or norm_lbl.lower() in ["hardhat", "helmet", "safety vest", "mask", "gloves", "footwear", "boots"]:
                box_color = (0, 255, 0)
                tag = f"{raw_lbl} ({score:.0%})"
            else:
                box_color = (220, 220, 0)
                tag = f"{raw_lbl} ({score:.0%})"

            cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), box_color, 2)
            (w, h), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(annotated, (xmin, max(0, ymin - 20)), (xmin + w + 8, ymin), box_color, -1)
            text_color = (0, 0, 0) if box_color in [(0, 255, 0), (255, 191, 0), (220, 220, 0)] else (255, 255, 255)
            cv2.putText(annotated, tag, (xmin + 3, max(12, ymin - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, text_color, 1)

        # Sleek Top Status Bar
        banner_h = 28
        cv2.rectangle(annotated, (0, 0), (640, banner_h), color_bgr, -1)
        cam_tag = "LIVE WEBCAM" if self.is_live else "CCTV FEED"
        header_text = f"[{self.camera_id}] {self.info['name']} ({cam_tag}) | {worker_count} Workers"
        cv2.putText(annotated, header_text, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

        # Sleek Bottom Alert Bar
        cv2.rectangle(annotated, (0, 332), (640, 360), (0, 0, 180) if not is_compliant else (20, 120, 40), -1)
        cv2.putText(annotated, f"Status: {alert_msg}", (10, 351), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        # Encode Frame to JPEG
        _, buffer = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        jpeg_bytes = buffer.tobytes()

        now = time.time()
        snapshot_url = ""

        # Logging into Database & Saving Violation Snapshot (throttled to max 1 per camera every 6s)
        if not is_compliant:
            if now - self.last_alert_time > 6.0:
                self.last_alert_time = now
                event_id = f"EVT-{int(now * 1000)}-{self.camera_id}"
                snapshot_filename = f"{event_id}.jpg"
                snapshot_filepath = os.path.join(SNAPSHOTS_DIR, snapshot_filename)
                
                # Save annotated frame as evidence snapshot
                cv2.imwrite(snapshot_filepath, annotated)
                snapshot_url = f"/api/snapshots/{snapshot_filename}"

                log_violation_event({
                    "event_id": event_id,
                    "camera_id": self.camera_id,
                    "camera_name": self.info["name"],
                    "zone": self.info["zone"],
                    "hazard_level": self.info["hazard_level"],
                    "worker_count": worker_count,
                    "missing_ppe": matching_res["missing_ppe"],
                    "explicit_violations": matching_res["explicit_violations"],
                    "required_ppe": matching_res["required_ppe"],
                    "detected_ppe": matching_res["detected_ppe"],
                    "confidence": 0.92,
                    "severity": "CRITICAL" if "Helmet" in matching_res["missing_ppe"] or "NO-Hardhat" in matching_res["explicit_violations"] else "HIGH",
                    "snapshot_url": snapshot_url
                })
        else:
            increment_camera_scans(self.camera_id)

        with self.lock:
            self.latest_jpeg = jpeg_bytes
            self.latest_raw_frame = frame
            self.latest_result = {
                "camera_id": self.camera_id,
                "name": self.info["name"],
                "zone": self.info["zone"],
                "hazard_level": self.info["hazard_level"],
                "status": "VIOLATION" if not is_compliant else "COMPLIANT",
                "is_compliant": is_compliant,
                "missing_ppe": matching_res["missing_ppe"],
                "suppressed_occlusions": matching_res.get("suppressed_occlusions", []),
                "required_ppe": matching_res["required_ppe"],
                "explicit_violations": matching_res["explicit_violations"],
                "alert_message": alert_msg,
                "total_detected_workers": worker_count,
                "temporal_consensus": dema_res.get("temporal_consensus", "5/5"),
                "is_live": self.is_live,
                "snapshot_url": snapshot_url,
                "timestamp": now
            }

    def get_jpeg(self) -> Optional[bytes]:
        with self.lock:
            return self.latest_jpeg

    def get_status(self) -> Dict[str, Any]:
        with self.lock:
            if not self.latest_result:
                return {
                    "camera_id": self.camera_id,
                    "name": self.info["name"],
                    "zone": self.info["zone"],
                    "hazard_level": self.info["hazard_level"],
                    "status": "COMPLIANT",
                    "is_compliant": True,
                    "missing_ppe": [],
                    "suppressed_occlusions": [],
                    "required_ppe": ["Helmet", "Gloves", "Footwear"],
                    "explicit_violations": [],
                    "alert_message": "[SAFE] Initializing stream...",
                    "total_detected_workers": 1,
                    "temporal_consensus": "5/5",
                    "is_live": self.is_live,
                    "timestamp": time.time()
                }
            return dict(self.latest_result)

    def close(self):
        self.is_running = False
        if self.cap is not None:
            self.cap.release()


class MultiCameraManager:
    """Enterprise Manager for 20+ Industrial Plant Camera Channels with YOLOv8 inference."""

    def __init__(self):
        init_db()
        seed_default_cameras(DEFAULT_24_CAMERAS)

        # Initialize YOLOv8 Trained Model
        yolo_weights = "models/best.pt"
        self.yolo_detector = YOLOv8Adapter(weights_path=yolo_weights, conf_threshold=0.25)
        print(f"[OK] MultiCameraManager initialized with YOLOv8 detector '{yolo_weights}'")

        self.dema_engine = PPEDeMaEngine(default_level="Level_2_High_Risk", min_similarity_thresh=0.25)
        self.channels: Dict[str, CameraChannel] = {}
        self.is_running = False
        self.thread: Optional[threading.Thread] = None

        # Load all cameras from DB
        db_cams = get_all_cameras()
        for cam in db_cams:
            self.channels[cam["camera_id"]] = CameraChannel(cam, self.dema_engine, self.yolo_detector)

    def start(self):
        """Starts multi-camera processing loop."""
        if self.is_running:
            return
        self.is_running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        print(f"[OK] MultiCameraManager started with {len(self.channels)} active channels.")

    def _run_loop(self):
        while self.is_running:
            for ch in self.channels.values():
                try:
                    ch.process_step()
                except Exception as e:
                    print(f"[WARN] Camera {ch.camera_id} loop error: {e}")
            time.sleep(0.04) # ~25 FPS loop

    def stop(self):
        self.is_running = False
        for ch in self.channels.values():
            ch.close()

    def get_camera_jpeg(self, camera_id: str) -> Optional[bytes]:
        ch = self.channels.get(camera_id)
        return ch.get_jpeg() if ch else None

    def get_all_statuses(self) -> List[Dict[str, Any]]:
        statuses = []
        for ch in self.channels.values():
            st = ch.get_status()
            if st:
                statuses.append(st)
        return statuses

    def add_or_update_camera(self, cam_data: Dict[str, Any]):
        cam_id = cam_data["camera_id"]
        if cam_id in self.channels:
            self.channels[cam_id].close()
        ch = CameraChannel(cam_data, self.dema_engine, self.yolo_detector)
        self.channels[cam_id] = ch
        return cam_data


# Singleton Instance
camera_manager_instance = MultiCameraManager()
