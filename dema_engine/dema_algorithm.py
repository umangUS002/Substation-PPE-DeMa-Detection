"""
PPE DeMa (Detection & Matching) Algorithm Engine with False Alarm Suppression & Occlusion Handling.
Implements:
1. 3-Step DeMa Algorithm (Determination, Matching, Warning)
2. Bounding Box IoU Inter-Person Occlusion Analysis (suppresses false alarms when feet/gloves are blocked)
3. Multi-Frame Temporal Persistence Filter (sliding window verification)
"""

import time
from typing import List, Dict, Any, Tuple, Optional
from .ppe_rules import HAZARD_LEVEL_REQUIREMENTS, normalize_class_name, TARGET_CLASSES


def compute_iou(boxA: List[int], boxB: List[int]) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes [xmin, ymin, xmax, ymax]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    if float(boxAArea + boxBArea - interArea) <= 0:
        return 0.0
    return interArea / float(boxAArea + boxBArea - interArea)


class PPEDeMaEngine:
    """
    Enhanced 3-Step PPE DeMa Engine with False-Alarm Suppression:
    - Step 1: PPE Determination (Extract bounding boxes, labels, confidences)
    - Step 2: Occlusion Analysis & PPE Matching
    - Step 3: Temporal Persistence & System Warning
    """

    def __init__(
        self, 
        default_level: str = "Level_2_High_Risk", 
        min_similarity_thresh: float = 0.25,
        temporal_window: int = 4
    ):
        self.default_level = default_level
        self.min_similarity_thresh = min_similarity_thresh
        self.temporal_window = temporal_window
        self.history_buffer = []
        self.violation_history = []

    def step1_ppe_determination(self, raw_detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Step 1 (PPE Determination): Filters confidence and normalizes labels."""
        processed = []
        for det in raw_detections:
            raw_label = det.get("label", "")
            confidence = det.get("confidence", 0.0)
            bbox = det.get("bbox", [0, 0, 0, 0])

            normalized_label = normalize_class_name(raw_label)
            if confidence >= self.min_similarity_thresh:
                processed.append({
                    "raw_label": raw_label,
                    "normalized_label": normalized_label,
                    "similarity_probability": confidence,
                    "bbox": bbox
                })
        return processed

    def detect_limb_occlusion(self, processed_detections: List[Dict[str, Any]]) -> Dict[str, bool]:
        """
        Detects if feet (boots) are occluded by other workers or out-of-frame.
        """
        persons = [d for d in processed_detections if d["normalized_label"].lower() in ["person", "human", "worker"]]
        occlusion_map = {"Footwear": False, "Helmet": False}

        # If a single person is in upper-body view (webcam close-up / desk), suppress feet
        if len(persons) >= 1:
            for p in persons:
                box = p["bbox"]
                box_h = box[3] - box[1]
                # If person box occupies most of the frame height from top but bottom is cut off
                if box[3] >= 340 and box[1] < 100:
                    occlusion_map["Footwear"] = True

        if len(persons) >= 2:
            for i in range(len(persons)):
                for j in range(i + 1, len(persons)):
                    p1_box = persons[i]["bbox"]
                    p2_box = persons[j]["bbox"]
                    iou = compute_iou(p1_box, p2_box)
                    if iou > 0.20:
                        occlusion_map["Footwear"] = True

        return occlusion_map

    def step2_ppe_matching(
        self, 
        processed_detections: List[Dict[str, Any]], 
        hazard_level: str = None
    ) -> Dict[str, Any]:
        """Step 2 (PPE Matching with Occlusion Suppression)."""
        level = hazard_level or self.default_level
        rule_spec = HAZARD_LEVEL_REQUIREMENTS.get(level, HAZARD_LEVEL_REQUIREMENTS["Level_2_High_Risk"])
        required_ppe = rule_spec["required"]

        detected_ppe_map = {}
        explicit_violations = set()

        for det in processed_detections:
            norm_name = det["normalized_label"]
            score = det["similarity_probability"]

            if norm_name.startswith("NO-") or "NO-" in det.get("raw_label", ""):
                clean_item = norm_name.replace("NO-", "").replace("NO ", "")
                explicit_violations.add(clean_item)
            elif norm_name in TARGET_CLASSES:
                if norm_name not in detected_ppe_map or score > detected_ppe_map[norm_name]["similarity_probability"]:
                    detected_ppe_map[norm_name] = det

        detected_set = set(detected_ppe_map.keys())
        
        # Missing items = (required - detected) + any explicitly detected non-compliances
        missing_ppe = (required_ppe - detected_set) | explicit_violations

        # Detect limb occlusion to suppress false alarms for feet/hands when out of frame
        occlusion_map = self.detect_limb_occlusion(processed_detections)
        suppressed_occlusions = []
        true_missing = []

        for item in missing_ppe:
            # Explicit violations (e.g. detected NO-Hardhat) are NEVER suppressed
            if item in explicit_violations:
                true_missing.append(item)
            elif occlusion_map.get(item, False):
                suppressed_occlusions.append(item)
            else:
                true_missing.append(item)

        is_compliant = len(true_missing) == 0

        return {
            "timestamp": time.time(),
            "hazard_level": level,
            "description": rule_spec["description"],
            "all_detections": processed_detections,
            "required_ppe": sorted(list(required_ppe)),
            "detected_ppe": list(detected_ppe_map.values()),
            "detected_set": sorted(list(detected_set)),
            "missing_ppe": sorted(list(set(true_missing))),
            "suppressed_occlusions": sorted(list(set(suppressed_occlusions))),
            "explicit_violations": sorted(list(explicit_violations)),
            "is_compliant": is_compliant
        }

    def step3_system_warning(self, matching_result: Dict[str, Any]) -> Dict[str, Any]:
        """Step 3 (System Warning with Temporal Consensus Filter)."""
        is_compliant = matching_result["is_compliant"]
        missing = matching_result["missing_ppe"]
        occluded = matching_result.get("suppressed_occlusions", [])
        level = matching_result["hazard_level"]

        # Temporal Sliding Window Consensus Filter
        self.history_buffer.append(is_compliant)
        if len(self.history_buffer) > self.temporal_window:
            self.history_buffer.pop(0)

        non_compliant_count = sum(1 for status in self.history_buffer if not status)
        confirmed_violation = non_compliant_count >= 1 if not is_compliant else (non_compliant_count >= 2)

        if not confirmed_violation and is_compliant:
            status_code = "COMPLIANT"
            if occluded:
                alert_message = f"[SAFE] PPE compliant (Suppressed {', '.join(occluded)})"
            else:
                alert_message = f"[SAFE] All required PPE worn"
            color_bgr = (0, 200, 0)
        else:
            status_code = "VIOLATION"
            items_str = ", ".join(missing) if missing else "Mandatory PPE"
            alert_message = f"[WARNING] Missing PPE: {items_str}"
            color_bgr = (0, 0, 220)
            self.violation_history.append({
                "timestamp": matching_result["timestamp"],
                "missing": missing,
                "hazard_level": level
            })

        output = {
            "status_code": status_code,
            "alert_message": alert_message,
            "color_bgr": color_bgr,
            "matching_result": matching_result,
            "temporal_consensus": f"{non_compliant_count}/{len(self.history_buffer)}"
        }
        return output

    def process_frame_pipeline(
        self, 
        raw_detections: List[Dict[str, Any]], 
        hazard_level: str = None
    ) -> Dict[str, Any]:
        step1_res = self.step1_ppe_determination(raw_detections)
        step2_res = self.step2_ppe_matching(step1_res, hazard_level=hazard_level)
        step3_res = self.step3_system_warning(step2_res)
        return step3_res
