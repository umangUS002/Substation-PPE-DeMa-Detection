import numpy as np
from typing import List, Dict, Tuple
from collections import deque

class ContextGate:
    """
    Context Gate: Suppresses detections that occur in known "safe" but highly confusing zones
    (e.g., active steam vents or soot-blowing areas).
    """
    def __init__(self):
        # List of polygons defining masked regions. Each polygon is a list of (x, y) coordinates.
        self.masked_regions: List[List[Tuple[int, int]]] = []
        
    def add_masked_region(self, polygon: List[Tuple[int, int]]):
        """Adds a new exclusion zone (e.g., the coordinates of a steam vent)."""
        self.masked_regions.append(polygon)
        
    def is_suppressed(self, bbox: List[int]) -> bool:
        """
        Checks if the center of the bounding box falls within any masked region.
        bbox format: [xmin, ymin, xmax, ymax]
        """
        if not self.masked_regions:
            return False
            
        import cv2
        center_x = (bbox[0] + bbox[2]) // 2
        center_y = (bbox[1] + bbox[3]) // 2
        point = (float(center_x), float(center_y))
        
        for polygon in self.masked_regions:
            contour = np.array(polygon, dtype=np.int32)
            # pointPolygonTest returns > 0 if inside, 0 if on boundary, < 0 if outside
            if cv2.pointPolygonTest(contour, point, False) >= 0:
                return True
                
        return False

class FireRollingWindow:
    """
    Rolling Window Confirmation: A candidate must be confirmed as 'Combustion' 
    in a majority of the rolling window before triggering the alarm.
    """
    def __init__(self, window_size: int = 5, threshold_ratio: float = 0.5):
        self.window_size = window_size
        self.threshold_ratio = threshold_ratio
        # Store boolean history: True = Combustion detected, False = Not detected
        self.history = deque(maxlen=window_size)
        
    def update(self, is_combustion: bool) -> bool:
        """
        Updates the window with the latest frame's prediction.
        Returns True if the majority consensus indicates confirmed combustion.
        """
        self.history.append(is_combustion)
        
        # If the buffer isn't full yet, wait (return False) or evaluate based on current size?
        # Typically we want to wait for at least a few frames.
        if len(self.history) < self.window_size // 2:
            return False
            
        combustion_count = sum(self.history)
        ratio = combustion_count / len(self.history)
        
        return ratio >= self.threshold_ratio

class FireDecisionEngine:
    """
    Integrates the Context Gate and Rolling Window logic for Fire and Ash detection.
    """
    def __init__(self, window_size: int = 5):
        self.context_gate = ContextGate()
        self.rolling_window = FireRollingWindow(window_size=window_size)
        
    def process_candidate(self, bbox: List[int], is_combustion_pred: bool) -> Dict:
        """
        Processes a single detection candidate through the Context Gate and Temporal Window.
        """
        # 1. Context Gate Check (Spatial suppression)
        if self.context_gate.is_suppressed(bbox):
            # Suppressed due to being in a known steam/soot zone
            return {
                "status": "SUPPRESSED",
                "reason": "Inside Context Gate (Steam/Soot zone)",
                "alarm_triggered": False
            }
            
        # 2. Rolling Window Consensus (Temporal confirmation)
        alarm = self.rolling_window.update(is_combustion_pred)
        
        if alarm:
            return {
                "status": "CONFIRMED_COMBUSTION",
                "reason": "Majority consensus in rolling window",
                "alarm_triggered": True
            }
        else:
            return {
                "status": "PENDING",
                "reason": "Waiting for rolling window consensus",
                "alarm_triggered": False
            }
