"""
person_tracker.py
------------------
A deliberately simple, dependency-free (numpy only) multi-object tracker
used to propagate person bounding boxes across frames where we *skip*
the expensive tiled Faster R-CNN pass (see `detect_stride` in
optimized_adapter.py).

Why not a "real" tracker (DeepSORT / ByteTrack / optical flow)?
=================================================================
Those are better trackers, but they cost CPU/GPU time we don't have to
spare -- the whole point of this module is to replace an expensive
detector call with a near-zero-cost bookkeeping step. At 12-15 fps on
a largely static industrial camera, a person's bounding box moves only
a few pixels between consecutive frames, so:

  - greedy IOU matching between "last known boxes" and the newest
    detector output (when we do run the detector), and
  - constant-velocity extrapolation (from the last two known centroids)
    to fill in the skipped frames in between

...are enough to keep boxes tracking a walking worker without drift
accumulating faster than the next real detection can correct it. This
is the standard "detect every N frames, track in between" pattern used
in real-time CCTV analytics pipelines.

If accuracy testing later shows this isn't tight enough (e.g. workers
moving fast near the camera), swap in Lucas-Kanade sparse optical flow
per track -- the interface (`update_from_detections` /
`propagate` / `tracks`) doesn't need to change.
"""

import itertools


def _iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class Track:
    __slots__ = ("track_id", "bbox", "prev_bbox", "score", "age", "misses")

    def __init__(self, track_id, bbox, score):
        self.track_id = track_id
        self.bbox = bbox
        self.prev_bbox = bbox
        self.score = score
        self.age = 0
        self.misses = 0

    def centroid(self, box=None):
        x1, y1, x2, y2 = box if box is not None else self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def extrapolate(self, max_shift_frac=0.5):
        """
        Constant-velocity guess for this frame based on the last two
        observed boxes. Clamped so a bad match can't fling the box
        across the frame; `max_shift_frac` bounds the shift to a
        fraction of the box's own size.
        """
        x1, y1, x2, y2 = self.bbox
        px1, py1, px2, py2 = self.prev_bbox
        w, h = x2 - x1, y2 - y1
        vx, vy = (x1 - px1), (y1 - py1)
        max_dx, max_dy = w * max_shift_frac, h * max_shift_frac
        vx = max(-max_dx, min(max_dx, vx))
        vy = max(-max_dy, min(max_dy, vy))
        return [x1 + vx, y1 + vy, x2 + vx, y2 + vy]


class SimpleIOUTracker:
    """
    One instance per camera stream.

    update_from_detections(detections): call this on frames where the
        real detector ran. `detections` is a list of (bbox, score).
        Returns the list of active tracks (post-match).

    propagate(): call this on frames where detection was skipped.
        Advances every track by constant-velocity extrapolation and
        returns the resulting boxes. Does not consume a "miss" the way
        a failed detection match would -- it's a deliberate hold-over,
        not evidence the person left.
    """

    def __init__(self, iou_threshold=0.35, max_misses=2, max_age_without_detection=6):
        self.iou_threshold = iou_threshold
        self.max_misses = max_misses
        self.max_age_without_detection = max_age_without_detection
        self._tracks = {}
        self._next_id = itertools.count(1)

    @property
    def tracks(self):
        return list(self._tracks.values())

    def update_from_detections(self, detections):
        unmatched_tracks = set(self._tracks.keys())
        unmatched_dets = set(range(len(detections)))

        # Greedy best-IOU matching (fine for the small N typical of a single-camera frame).
        pairs = []
        for tid, track in self._tracks.items():
            for di, (bbox, score) in enumerate(detections):
                iou = _iou(track.bbox, bbox)
                if iou >= self.iou_threshold:
                    pairs.append((iou, tid, di))
        pairs.sort(reverse=True, key=lambda p: p[0])

        for iou, tid, di in pairs:
            if tid in unmatched_tracks and di in unmatched_dets:
                track = self._tracks[tid]
                bbox, score = detections[di]
                track.prev_bbox = track.bbox
                track.bbox = list(bbox)
                track.score = score
                track.age += 1
                track.misses = 0
                unmatched_tracks.discard(tid)
                unmatched_dets.discard(di)

        # Unmatched existing tracks: count a miss, drop if stale too long.
        for tid in list(unmatched_tracks):
            track = self._tracks[tid]
            track.misses += 1
            if track.misses > self.max_misses:
                del self._tracks[tid]

        # Unmatched detections become new tracks.
        for di in unmatched_dets:
            bbox, score = detections[di]
            tid = next(self._next_id)
            self._tracks[tid] = Track(tid, list(bbox), score)

        return self.tracks

    def propagate(self):
        """Returns (track_id, bbox, score) tuples -- track_id is stable across
        calls for the same person, which lets callers cache per-person work
        (e.g. PPE classification) keyed by it instead of redoing it every frame."""
        results = []
        for track in list(self._tracks.values()):
            if track.age > self.max_age_without_detection:
                # Too many frames since a real detection confirmed this
                # box -- don't keep hallucinating a person forever.
                continue
            new_bbox = track.extrapolate()
            track.prev_bbox = track.bbox
            track.bbox = new_bbox
            track.age += 1
            results.append((track.track_id, new_bbox, track.score))
        return results

    def reset(self):
        self._tracks.clear()
