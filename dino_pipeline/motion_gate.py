"""
motion_gate.py
---------------
Cheap, CPU-only pre-filter that decides *which tiles of which cameras*
actually need to go through the expensive GPU pipeline this frame.

Why this exists
================
In the original pipeline, every one of the 25 streams pushes all 4
tiles (2x2 grid) through Faster R-CNN on *every* frame, regardless of
whether anything in the scene changed. For static substation / plant
cameras, the overwhelming majority of tiles are unchanged frame-to-frame
(empty yard, idle equipment, a person standing still at a desk, etc.).
Skipping GPU work on those tiles is the single highest-leverage
optimization available before touching the neural nets at all -- it
costs a few hundred microseconds of CPU/OpenCV time per tile instead of
a full detector forward pass.

Design
======
- One `cv2.BackgroundSubtractorMOG2` per camera, run on a small
  downscaled grayscale copy of the *full* frame (cheap: ~320x180).
- Per tile, we count foreground pixels that fall inside that tile's
  region. If the count clears `min_motion_pixels`, the tile is marked
  "active" and gets a real detector pass this frame.
- Two safety nets against motion-gating silently hiding a real person:
    1. `force_interval`: every N frames (default 45 @ ~15fps ~= 3s),
       *all* tiles are forced active regardless of motion, so a person
       who has gone still (and faded into the MOG2 background model)
       is still re-detected periodically. This is what bounds the
       worst-case detection latency contribution from motion gating.
    2. `warmup_frames`: MOG2 needs a few frames to build a background
       model; until then every tile is treated as active.
- This module has no dependency on torch/torchvision so it can be
  unit-tested and reasoned about independently of the GPU pipeline.
"""

import time
import numpy as np
import cv2


class MotionGate:
    """
    Tracks per-camera motion state and decides which tiles need a real
    detector pass on the current frame.

    Usage:
        gate = MotionGate()
        active_idx, forced = gate.active_tile_indices(camera_id, frame_bgr, tiles)
        # tiles = list of (y1, x1, y2, x2) as produced by DINOAdapter._generate_tiles
        # active_idx = list of indices into `tiles` that should be run through the detector
    """

    def __init__(
        self,
        history=300,
        var_threshold=24,
        min_motion_frac=0.0015,   # fraction of a tile's downscaled pixel area that must be "foreground"
        force_interval=45,       # force a full (all-tiles) detection pass every N frames per camera
        warmup_frames=5,
        downscale_width=320,
    ):
        self.history = history
        self.var_threshold = var_threshold
        self.min_motion_frac = min_motion_frac
        self.force_interval = force_interval
        self.warmup_frames = warmup_frames
        self.downscale_width = downscale_width

        # per-camera state
        self._subtractors = {}   # camera_id -> cv2.BackgroundSubtractorMOG2
        self._frame_count = {}   # camera_id -> int

    def _get_subtractor(self, camera_id):
        sub = self._subtractors.get(camera_id)
        if sub is None:
            sub = cv2.createBackgroundSubtractorMOG2(
                history=self.history,
                varThreshold=self.var_threshold,
                detectShadows=False,
            )
            self._subtractors[camera_id] = sub
            self._frame_count[camera_id] = 0
        return sub

    def active_tile_indices(self, camera_id, frame_bgr, tiles):
        """
        Returns (active_indices, forced) where:
          - active_indices: sorted list of tile indices to run detection on this frame
          - forced: True if this was a forced full-detection frame (all tiles active)
        """
        frame_h, frame_w = frame_bgr.shape[:2]
        sub = self._get_subtractor(camera_id)
        count = self._frame_count[camera_id]
        self._frame_count[camera_id] = count + 1

        # Downscale for cheap background subtraction.
        scale = self.downscale_width / float(frame_w)
        small_w = self.downscale_width
        small_h = max(1, int(frame_h * scale))
        small = cv2.resize(frame_bgr, (small_w, small_h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        fg_mask = sub.apply(gray)  # updates the background model as a side effect

        forced = (count < self.warmup_frames) or (
            self.force_interval > 0 and count % self.force_interval == 0
        )
        if forced:
            return list(range(len(tiles))), True

        # Light denoise so single noisy pixels don't count as "motion".
        fg_mask = cv2.medianBlur(fg_mask, 3)

        active = []
        for idx, (y1, x1, y2, x2) in enumerate(tiles):
            sy1, sx1 = int(y1 * scale), int(x1 * scale)
            sy2, sx2 = int(y2 * scale), int(x2 * scale)
            sy2, sx2 = max(sy2, sy1 + 1), max(sx2, sx1 + 1)
            region = fg_mask[sy1:sy2, sx1:sx2]
            if region.size == 0:
                continue
            motion_frac = np.count_nonzero(region) / float(region.size)
            if motion_frac >= self.min_motion_frac:
                active.append(idx)

        return active, False

    def reset(self, camera_id=None):
        """Drop background models (e.g. after a camera reconnect / scene cut)."""
        if camera_id is None:
            self._subtractors.clear()
            self._frame_count.clear()
        else:
            self._subtractors.pop(camera_id, None)
            self._frame_count.pop(camera_id, None)
