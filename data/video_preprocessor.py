"""
Video Frame Preprocessor for CCTV / Surveillance Footage.
Applies adaptive preprocessing to bridge the domain gap between 
clean training images and noisy, low-res, poorly-lit CCTV frames.

Pipeline:
1. CLAHE (Contrast Limited Adaptive Histogram Equalization) — fixes uneven/poor lighting
2. Fast Denoising — reduces H.264/H.265 compression artifacts and sensor noise
3. Unsharp Masking — counteracts motion blur and low-res blurriness
4. Optional resolution upscaling — for very low-res streams
"""

import cv2
import numpy as np
from typing import Optional, Tuple


class VideoFramePreprocessor:
    """
    Preprocesses CCTV/video frames to improve detection quality.
    All operations are optimized for real-time performance (~2-4ms per frame on CPU).
    """

    def __init__(
        self,
        enable_clahe: bool = True,
        enable_denoise: bool = True,
        enable_sharpen: bool = True,
        clahe_clip_limit: float = 2.5,
        clahe_grid_size: Tuple[int, int] = (8, 8),
        denoise_strength: int = 6,
        sharpen_amount: float = 0.4,
        min_resolution: int = 480,
    ):
        self.enable_clahe = enable_clahe
        self.enable_denoise = enable_denoise
        self.enable_sharpen = enable_sharpen
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = clahe_grid_size
        self.denoise_strength = denoise_strength
        self.sharpen_amount = sharpen_amount
        self.min_resolution = min_resolution

        # Pre-create CLAHE object (reusable, thread-safe)
        if self.enable_clahe:
            self._clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit,
                tileGridSize=self.clahe_grid_size
            )

        # Pre-create sharpening kernel
        self._sharpen_kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=np.float32)

    def apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """
        Applies CLAHE to the L-channel of the frame (LAB color space).
        Dramatically improves visibility in dark/unevenly-lit CCTV footage
        without destroying color information.
        """
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Apply CLAHE only to the luminance channel
        l_enhanced = self._clahe.apply(l_channel)

        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    def apply_denoise(self, frame: np.ndarray) -> np.ndarray:
        """
        Fast non-local means denoising to suppress compression artifacts.
        Uses fastNlMeansDenoisingColored which is optimized for real-time.
        """
        return cv2.fastNlMeansDenoisingColored(
            frame,
            None,
            h=self.denoise_strength,
            hForColorComponents=self.denoise_strength,
            templateWindowSize=7,
            searchWindowSize=15
        )

    def apply_sharpen(self, frame: np.ndarray) -> np.ndarray:
        """
        Unsharp masking to counteract motion blur and low-res blurriness.
        Blends the sharpened result with the original to avoid over-sharpening.
        """
        # Gaussian blur to create the "unsharp mask"
        blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=3)

        # Unsharp mask: original + amount * (original - blurred)
        sharpened = cv2.addWeighted(
            frame, 1.0 + self.sharpen_amount,
            blurred, -self.sharpen_amount,
            0
        )
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def upscale_if_needed(self, frame: np.ndarray) -> np.ndarray:
        """
        Upscales very low-resolution frames so the detector has enough pixels.
        Only triggers if the shorter side is below min_resolution.
        """
        h, w = frame.shape[:2]
        shorter_side = min(h, w)

        if shorter_side < self.min_resolution:
            scale = self.min_resolution / shorter_side
            new_w = int(w * scale)
            new_h = int(h * scale)
            return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        return frame

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Full preprocessing pipeline. Applies all enabled stages in order.
        Returns the preprocessed frame (same shape or upscaled).
        
        Typical cost: ~2-4ms on CPU for a 1080p frame (excluding denoise).
        With denoise: ~15-25ms (can disable for real-time needs).
        """
        result = frame.copy()

        # 1. Upscale very low-res frames first
        result = self.upscale_if_needed(result)

        # 2. CLAHE — fix lighting
        if self.enable_clahe:
            result = self.apply_clahe(result)

        # 3. Denoise — remove compression artifacts
        if self.enable_denoise:
            result = self.apply_denoise(result)

        # 4. Sharpen — counteract blur
        if self.enable_sharpen:
            result = self.apply_sharpen(result)

        return result


class FrameSkipController:
    """
    Smart frame skipping for video streams.
    Processes every Nth frame and holds the last detection for skipped frames.
    Adapts skip rate based on whether motion/changes are detected.
    """

    def __init__(self, base_skip: int = 2, max_skip: int = 5):
        self.base_skip = base_skip
        self.max_skip = max_skip
        self.frame_count = 0
        self.current_skip = base_skip
        self.prev_frame_gray = None
        self.motion_threshold = 15.0  # Mean absolute difference threshold

    def should_process(self, frame: np.ndarray) -> bool:
        """
        Returns True if this frame should be processed by the detector.
        Uses frame differencing to detect motion — processes more frames when
        there's movement, skips more when the scene is static.
        """
        self.frame_count += 1

        # Always process the first frame
        if self.frame_count == 1:
            self.prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return True

        # Check if we've hit the skip interval
        if self.frame_count % self.current_skip != 0:
            return False

        # Compute motion score (fast — just mean absolute difference)
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_frame_gray is not None:
            # Resize for faster comparison if frames are large
            h, w = current_gray.shape
            if h > 240:
                small_curr = cv2.resize(current_gray, (320, 240))
                small_prev = cv2.resize(self.prev_frame_gray, (320, 240))
            else:
                small_curr = current_gray
                small_prev = self.prev_frame_gray

            motion_score = np.mean(cv2.absdiff(small_curr, small_prev))

            # Adapt skip rate: more motion → process more frames
            if motion_score > self.motion_threshold:
                self.current_skip = max(1, self.base_skip - 1)
            else:
                self.current_skip = min(self.max_skip, self.base_skip + 1)

        self.prev_frame_gray = current_gray
        return True

    def reset(self):
        """Resets the controller state."""
        self.frame_count = 0
        self.current_skip = self.base_skip
        self.prev_frame_gray = None
