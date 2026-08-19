"""
PPE DeMa (Detection & Matching) Algorithm Engine.
Implements the 3-Step DeMa Algorithm presented in the IEEE 2023 Paper:
Zhao & Barati (IEEE Transactions on Industry Applications, 2023).
"""

import time
from typing import List, Dict, Any, Tuple
from .ppe_rules import HAZARD_LEVEL_REQUIREMENTS, normalize_class_name, TARGET_CLASSES

class PPEDeMaEngine:
    """
    3-Step PPE DeMa Engine:
    - Step 1: Determination (Extract detected PPE bounding boxes, classes, & similarity scores)
    - Step 2: Matching (Compare detected PPE against required Helmet, Gloves, & Footwear rules)
    - Step 3: System Warning (Generate real-time alerts if compliance checks fail)
    """

    def __init__(self, default_level: str = "Level_2_High_Risk", min_similarity_thresh: float = 0.50):
        self.default_level = default_level
        self.min_similarity_thresh = min_similarity_thresh
        self.violation_history = []

    def step1_ppe_determination(self, raw_detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Step 1 (PPE Determination):
        Filters and normalizes bounding boxes, labels, and similarity probabilities (confidences).
        """
        processed_detections = []
        for det in raw_detections:
            raw_label = det.get("label", "")
            confidence = det.get("confidence", 0.0)
            bbox = det.get("bbox", [0, 0, 0, 0])  # [xmin, ymin, xmax, ymax]

            normalized_label = normalize_class_name(raw_label)
            
            if confidence >= self.min_similarity_thresh:
                processed_detections.append({
                    "raw_label": raw_label,
                    "normalized_label": normalized_label,
                    "similarity_probability": confidence,
                    "bbox": bbox
                })
        return processed_detections

    def step2_ppe_matching(
        self, 
        processed_detections: List[Dict[str, Any]], 
        hazard_level: str = None
    ) -> Dict[str, Any]:
        """
        Step 2 (PPE Matching):
        Compares detected items against required set (Helmet, Gloves, Footwear).
        """
        level = hazard_level or self.default_level
        rule_spec = HAZARD_LEVEL_REQUIREMENTS.get(level, HAZARD_LEVEL_REQUIREMENTS["Level_2_High_Risk"])
        required_ppe = rule_spec["required"]

        detected_ppe_map = {}
        explicit_violations = set()

        for det in processed_detections:
            norm_name = det["normalized_label"]
            score = det["similarity_probability"]

            if norm_name.startswith("NO-"):
                clean_item = norm_name.replace("NO-", "")
                explicit_violations.add(clean_item)
            elif norm_name in TARGET_CLASSES:
                if norm_name not in detected_ppe_map or score > detected_ppe_map[norm_name]["similarity_probability"]:
                    detected_ppe_map[norm_name] = det

        detected_set = set(detected_ppe_map.keys())
        missing_ppe = (required_ppe - detected_set) | (required_ppe & explicit_violations)

        is_compliant = len(missing_ppe) == 0

        return {
            "timestamp": time.time(),
            "hazard_level": level,
            "description": rule_spec["description"],
            "all_detections": processed_detections,
            "required_ppe": sorted(list(required_ppe)),
            "detected_ppe": list(detected_ppe_map.values()),
            "detected_set": sorted(list(detected_set)),
            "missing_ppe": sorted(list(missing_ppe)),
            "explicit_violations": sorted(list(explicit_violations)),
            "is_compliant": is_compliant
        }


    def step3_system_warning(self, matching_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 3 (System Warning):
        Constructs warning alert notification and logs non-compliance events.
        """
        is_compliant = matching_result["is_compliant"]
        missing = matching_result["missing_ppe"]
        level = matching_result["hazard_level"]

        if is_compliant:
            status_code = "COMPLIANT"
            alert_message = f"[SAFE] All required PPE worn ({', '.join(matching_result['required_ppe'])})"
            color_bgr = (0, 200, 0) # Green
        else:
            status_code = "VIOLATION"
            alert_message = f"[WARNING] Missing PPE for {level}: {', '.join(missing)}"
            color_bgr = (0, 0, 220) # Red
            self.violation_history.append({
                "timestamp": matching_result["timestamp"],
                "missing": missing,
                "hazard_level": level
            })


        output = {
            "status_code": status_code,
            "alert_message": alert_message,
            "color_bgr": color_bgr,
            "matching_result": matching_result
        }
        return output

    def process_frame_pipeline(
        self, 
        raw_detections: List[Dict[str, Any]], 
        hazard_level: str = None
    ) -> Dict[str, Any]:
        """
        Full 3-Step DeMa Pipeline Execution.
        """
        step1_res = self.step1_ppe_determination(raw_detections)
        step2_res = self.step2_ppe_matching(step1_res, hazard_level=hazard_level)
        step3_res = self.step3_system_warning(step2_res)
        return step3_res
