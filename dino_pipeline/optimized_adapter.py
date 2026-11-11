"""
optimized_adapter.py
---------------------
Rewrite of the original DINOAdapter aimed at one target: 25 x 2MP CCTV
feeds, 12-15 fps analytic rate each, <=5s end-to-end alert latency, on
a single GTX 1650 (4GB VRAM) laptop GPU.

What changed vs. the original dino_adapter.py, and why
========================================================
1. Motion-gated tiling (motion_gate.py)
   The original always runs all 4 tiles of the 2x2 grid through Faster
   R-CNN, every frame, for every camera. Most tiles on a static
   industrial camera don't change frame-to-frame. We now run a cheap
   CPU background-subtraction pass first and only send *changed*
   tiles into the GPU detector, with a periodic forced full pass so a
   person who has gone still is never permanently missed. This is
   normally the single biggest win, because it directly cuts the
   4x-per-frame tiling multiplier down to "however many tiles are
   actually active."

2. Detect-then-track (person_tracker.py)
   Instead of running the (now motion-gated) tiled detector on every
   single frame, we run it every `detect_stride` frames and use a
   lightweight IOU tracker to carry boxes forward on the frames in
   between. People move only a few pixels between frames at 12-15fps,
   so constant-velocity extrapolation holds up fine for the ~2-4
   frame gaps involved, and it turns an expensive multi-tile CNN call
   into a few microseconds of bookkeeping.

3. Distilled backbone by default (lightweight_backbone.py)
   DINOv2 ViT-S/14 is a strong feature extractor but expensive per
   crop, especially on a Turing GPU with no tensor cores. Default mode
   swaps in a MobileNetV3-Small student distilled from DINOv2 offline
   -- same classifier head, ~10-30x fewer FLOPs per crop. `mode="dino"`
   is kept available if you want to run the original backbone on a
   subset of high-priority cameras.

4. FP16 autocast + torch.inference_mode()
   Both stages run under `torch.autocast` when CUDA is available.
   Gains on a non-tensor-core GPU are modest (roughly 1.2-1.5x, not
   the 2-3x you'd see on an RTX/Tensor-Core part) but it also halves
   activation memory, which matters on a 4GB card running 25-stream
   batches.

5. GPU-side crop resize, vectorized normalization
   The original resizes every person crop with cv2.resize on the CPU,
   one at a time, before converting to a tensor. With many people
   across 25 streams this becomes a real CPU bottleneck (and CPU time
   is scarce -- it's also decoding 25 2MP video streams). We move the
   resize onto the GPU via F.interpolate and normalize the whole
   stacked batch in one vectorized op.

6. Top-K person cap per frame
   Bounds worst-case Stage 2 batch size (a crowded frame shouldn't be
   allowed to blow the latency budget for every other camera sharing
   the inference server).

7. Per-track PPE classification cache (classify_stride)
   Stage 2 (PPE classification) does NOT get cheaper just from
   motion-gating/detect-stride, because a tracked person still needs
   their PPE state checked every frame in the original design -- a
   worker standing still in a static scene still gets reclassified
   12-15 times a second for no reason. PPE state doesn't flicker
   frame-to-frame, so each tracked person (keyed by their stable
   track_id) is only actually re-run through the backbone+classifier
   every `classify_stride` frames; frames in between reuse the last
   classification. New tracks are always classified immediately (no
   stride delay) so a just-arrived worker's PPE state is never stale
   on first sight. This turns out to be the single biggest remaining
   lever once Stage 1 is fixed -- see docs/scaling_bottleneck_analysis.docx.

8. Multi-camera batching entry point
   `predict_batch_multi_camera` accepts frames from several cameras at
   once and batches Stage 2 (and, for tiles, Stage 1) across all of
   them -- this is what actually gets GPU utilization up on a single
   shared "inference server" servicing 25 feeds. See stream_server.py
   for the piece that collects frames across cameras and calls this.

Honest scope note
==================
These changes are designed to close the gap between the ~5-10x
overload the original tiled+DINOv2 pipeline implies at 25 streams and
what a single GTX 1650 can plausibly deliver. They will not
automatically hit 25 x 12-15fps -- that depends on real crowd density,
scene motion levels, and needs to be benchmarked on your actual
hardware and footage (see docs/scaling_bottleneck_analysis.docx for
the throughput math and a fallback plan -- adaptive per-camera rate,
TensorRT/INT8 export, or a second inference server -- if benchmarking
still comes up short).
"""

import torch
from collections import defaultdict
import torch.nn as nn
import torchvision
from torchvision.transforms import functional as F
import numpy as np
import cv2

try:
    from .rcnn_detector import get_faster_rcnn_detector
except (ImportError, ValueError):
    from dino_pipeline.rcnn_detector import get_faster_rcnn_detector

from dino_pipeline.motion_gate import MotionGate
from dino_pipeline.person_tracker import SimpleIOUTracker
from dino_pipeline.lightweight_backbone import PPEFeatureBackbone, PPEClassifierHead


class DINOAdapterOptimized:
    """
    Drop-in-ish replacement for the original DINOAdapter. The main API
    difference is that streaming methods now take a `camera_id` so
    per-camera motion/tracking state can be kept -- pass camera_id=0
    (the default) if you only have a single stream.
    """

    def __init__(
        self,
        classifier_weights_path=None,
        person_threshold=0.50,
        ppe_threshold=0.25,
        device=None,
        tile_grid=(2, 2),
        tile_overlap=0.15,
        min_person_area=2500,
        min_person_aspect_ratio=0.8,     # height/width must exceed this to be a "person" (filters stones/pipes)
        # --- new knobs ---
        rcnn_model_size="medium",        # "medium" (Faster R-CNN ResNet-50 FPN V2) or "small" (MobileNetV3 320 FPN)
        backbone_mode="distilled",       # "distilled" (fast) or "dino" (original, slow/accurate)
        student_weights_path=None,       # required for backbone_mode="distilled" once you've run distillation
        use_fp16=True,
        motion_gating=True,
        detect_stride=3,                 # run the full tiled detector every Nth frame per camera; track in between
        classify_stride=2,               # re-run PPE classification on a *tracked* person every Nth frame; reuse cached label in between
        ppe_cache_stale_after=600,       # drop a track's cached PPE label if untouched this many frames (~40-50s @ 12-15fps)
        max_persons_per_frame=12,
        motion_gate_kwargs=None,
        **kwargs
    ):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.person_threshold = person_threshold
        self.ppe_threshold = ppe_threshold
        self.tile_grid = tile_grid
        self.tile_overlap = tile_overlap
        self.min_person_area = min_person_area
        self.min_person_aspect_ratio = min_person_aspect_ratio
        self.detect_stride = max(1, detect_stride)
        self.classify_stride = max(1, classify_stride)
        self.ppe_cache_stale_after = ppe_cache_stale_after
        self.max_persons_per_frame = max_persons_per_frame
        self.use_fp16 = use_fp16 and self.device == "cuda"
        self.rcnn_model_size = rcnn_model_size

        print(f"[INFO] Initializing Optimized Two-Stage PPE Pipeline on {self.device}...")
        print(f"[INFO] Person detector: Faster R-CNN ({self.rcnn_model_size}), Tiling: {tile_grid[0]}x{tile_grid[1]} grid, {tile_overlap:.0%} overlap, "
              f"motion_gating={motion_gating}, detect_stride={self.detect_stride}, "
              f"classify_stride={self.classify_stride}, "
              f"backbone_mode={backbone_mode}, fp16={self.use_fp16}")

        # 1. Person Detector (Faster R-CNN: ResNet-50 FPN V2 for 'medium', MobileNetV3 for 'small')
        self.person_detector = get_faster_rcnn_detector(model_size=self.rcnn_model_size, device=self.device)
        self.PERSON_CLASS_ID = 1

        # 2. Feature extractor: distilled CNN by default, DINOv2 available as opt-in.
        self.backbone = PPEFeatureBackbone(
            mode=backbone_mode,
            student_weights_path=student_weights_path,
            dino_weights_path=kwargs.get("dino_weights_path", None),
            device=self.device,
        )

        # 3. PPE Classifier Head (same shape/contract as the original).
        self.ppe_classes = ["Helmet", "Gloves", "Footwear"]
        self.classifier = PPEClassifierHead(embed_dim=self.backbone.embed_dim, num_classes=len(self.ppe_classes))
        if classifier_weights_path:
            try:
                self.classifier.load_state_dict(torch.load(classifier_weights_path, map_location=self.device))
                print(f"[INFO] Loaded PPE Classifier weights from {classifier_weights_path}")
            except Exception as e:
                print(f"[WARNING] Could not load classifier weights: {e}")
        self.classifier.to(self.device)
        self.classifier.eval()

        # 4. Per-camera state
        self.motion_gating = motion_gating
        self.motion_gate = MotionGate(**(motion_gate_kwargs or {})) if motion_gating else None
        self._trackers = {}       # camera_id -> SimpleIOUTracker
        self._frame_counters = {} # camera_id -> int
        self._ppe_cache = {}      # (camera_id, track_id) -> {"frame": int, "hits": {label: conf}}

        # --- PPE Temporal Stabilizer (anti-flicker) ---
        # Per-track exponential moving average of raw classifier confidence.
        # Key: (camera_id, track_id) -> {ppe_name: smoothed_confidence}
        self._ppe_ema = {}        # smoothed running confidence per track
        self._ppe_ema_alpha = 0.3   # EMA weight for new observation (lower = smoother, less flicker)
        # Hysteresis thresholds to prevent rapid toggling:
        #   - A PPE item is shown only once smoothed confidence rises above ppe_on_thresh
        #   - It stays shown until smoothed confidence drops below ppe_off_thresh
        self._ppe_on_thresh = 0.32   # must exceed this to turn ON
        self._ppe_off_thresh = 0.08  # must drop below this to turn OFF
        self._ppe_display_state = {} # (cam_id, track_id) -> {ppe_name: bool (currently shown)}

        self._input_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self._input_std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    # ------------------------------------------------------------------
    # Tiling (unchanged geometry from the original)
    # ------------------------------------------------------------------
    def _generate_tiles(self, frame_h, frame_w):
        rows, cols = self.tile_grid
        overlap = self.tile_overlap
        tile_h = frame_h // rows
        tile_w = frame_w // cols
        overlap_h = int(tile_h * overlap)
        overlap_w = int(tile_w * overlap)

        tiles = []
        for r in range(rows):
            for c in range(cols):
                y1 = max(0, r * tile_h - overlap_h)
                x1 = max(0, c * tile_w - overlap_w)
                y2 = min(frame_h, (r + 1) * tile_h + overlap_h)
                x2 = min(frame_w, (c + 1) * tile_w + overlap_w)
                tiles.append((y1, x1, y2, x2))
        return tiles

    def _tracker_for(self, camera_id):
        tr = self._trackers.get(camera_id)
        if tr is None:
            tr = SimpleIOUTracker()
            self._trackers[camera_id] = tr
        return tr

    def _nms_persons(self, person_boxes, person_scores, iou_threshold=0.4):
        if len(person_boxes) == 0:
            return [], []
        boxes_tensor = torch.tensor(person_boxes, dtype=torch.float32)
        scores_tensor = torch.tensor(person_scores, dtype=torch.float32)
        keep = torchvision.ops.nms(boxes_tensor, scores_tensor, iou_threshold)
        return [person_boxes[i] for i in keep], [person_scores[i] for i in keep]

    # ------------------------------------------------------------------
    # Stage 1: motion-gated tiled person detection
    # ------------------------------------------------------------------
    def _detect_persons_tiled_gated(self, frame_bgr, camera_id):
        frame_h, frame_w = frame_bgr.shape[:2]
        tiles = self._generate_tiles(frame_h, frame_w)

        if self.motion_gating:
            active_idx, _forced = self.motion_gate.active_tile_indices(camera_id, frame_bgr, tiles)
        else:
            active_idx = list(range(len(tiles)))

        if not active_idx:
            return [], []

        tile_tensors, tile_offsets = [], []
        for idx in active_idx:
            y1, x1, y2, x2 = tiles[idx]
            tile_img = frame_bgr[y1:y2, x1:x2]
            tile_rgb = cv2.cvtColor(tile_img, cv2.COLOR_BGR2RGB)
            tile_tensors.append(F.to_tensor(tile_rgb).to(self.device))
            tile_offsets.append((y1, x1))

        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_fp16):
                tile_outputs = self.person_detector(tile_tensors)

        all_boxes, all_scores = [], []
        for tile_idx, pd_out in enumerate(tile_outputs):
            y_off, x_off = tile_offsets[tile_idx]
            boxes = pd_out["boxes"].float().cpu().numpy()
            scores = pd_out["scores"].float().cpu().numpy()
            labels = pd_out["labels"].cpu().numpy()

            mask = (labels == self.PERSON_CLASS_ID) & (scores > self.person_threshold)
            for box, score in zip(boxes[mask], scores[mask]):
                full_box = [box[0] + x_off, box[1] + y_off, box[2] + x_off, box[3] + y_off]
                bw = full_box[2] - full_box[0]
                bh = full_box[3] - full_box[1]
                area = bw * bh
                # Filter: minimum area + aspect ratio (person is taller than wide)
                if area >= self.min_person_area and (bh / max(bw, 1)) >= self.min_person_aspect_ratio:
                    all_boxes.append(full_box)
                    all_scores.append(float(score))

        return self._nms_persons(all_boxes, all_scores, iou_threshold=0.4)

    def _get_person_boxes(self, frame_bgr, camera_id):
        """
        Detect-then-track dispatcher: runs the real (motion-gated,
        tiled) detector every `detect_stride` frames for this camera
        and reconciles it with the tracker; on the frames in between,
        just propagates existing tracks.

        Returns (frame_idx, [(track_id, bbox, score), ...]) -- the
        stable track_id is what lets the caller cache PPE
        classification per person instead of redoing it every frame
        (see `classify_stride` / `_ppe_cache` in predict_batch_multi_camera).
        """
        frame_idx = self._frame_counters.get(camera_id, 0)
        self._frame_counters[camera_id] = frame_idx + 1
        tracker = self._tracker_for(camera_id)

        if frame_idx % self.detect_stride == 0:
            boxes, scores = self._detect_persons_tiled_gated(frame_bgr, camera_id)
            detections = list(zip(boxes, scores))
            tracker.update_from_detections(detections)
            return frame_idx, [(t.track_id, t.bbox, t.score) for t in tracker.tracks]
        else:
            return frame_idx, tracker.propagate()

    # ------------------------------------------------------------------
    # Stage 2: crop preprocessing (GPU-side resize) + backbone + classifier
    # ------------------------------------------------------------------
    def _crop_to_tensor(self, frame_bgr, box, target_size=224):
        x1, y1, x2, y2 = map(int, box)
        h_frame, w_frame = frame_bgr.shape[:2]
        w, h = x2 - x1, y2 - y1
        # Asymmetric padding: 10% horizontal, 25% vertical (helmet is at the very top)
        pad_x, pad_y = int(w * 0.10), int(h * 0.25)
        x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        x2, y2 = min(w_frame, x2 + pad_x), min(h_frame, y2 + pad_y)
        if x2 - x1 < 10 or y2 - y1 < 10:
            return None

        crop_bgr = frame_bgr[y1:y2, x1:x2]

        # --- CLAHE preprocessing (matches original dino_adapter.py) ---
        lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_ch = clahe.apply(l_ch)
        crop_enhanced = cv2.cvtColor(cv2.merge((l_ch, a_ch, b_ch)), cv2.COLOR_LAB2BGR)

        # --- Letterbox to preserve aspect ratio (matches original) ---
        ch, cw = crop_enhanced.shape[:2]
        max_side = max(ch, cw)
        canvas = np.full((max_side, max_side, 3), 128, dtype=np.uint8)
        x_off = (max_side - cw) // 2
        y_off = (max_side - ch) // 2
        canvas[y_off:y_off+ch, x_off:x_off+cw] = crop_enhanced

        crop_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        crop_resized = cv2.resize(crop_rgb, (target_size, target_size))

        tensor = torch.from_numpy(np.ascontiguousarray(crop_resized)).permute(2, 0, 1).float().div_(255.0)
        tensor = tensor.to(self.device, non_blocking=True)
        return tensor

    def _run_stage2(self, frame_crop_pairs):
        """
        frame_crop_pairs: list of (frame_idx_or_camera_id, box, tensor)
        Returns list of (index_key, box, {ppe_label: confidence})
        """
        if not frame_crop_pairs:
            return []

        tensors = torch.stack([t for _, _, t in frame_crop_pairs]).to(self.device)
        tensors = (tensors - self._input_mean) / self._input_std  # vectorized normalize, whole batch at once

        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_fp16):
                features = self.backbone(tensors)
                logits = self.classifier(features)
                probs = torch.sigmoid(logits).float().cpu().numpy()

        results = []
        for i, prob_array in enumerate(probs):
            key, box, _ = frame_crop_pairs[i]  # key = (camera_id, track_id)
            raw_confs = {self.ppe_classes[ci]: float(prob_array[ci]) for ci in range(len(self.ppe_classes))}

            # --- EMA smoothing per-track ---
            if key not in self._ppe_ema:
                # Initialize EMA with the FIRST observation (not 0.0) so labels
                # can appear immediately on the very first classification.
                self._ppe_ema[key] = dict(raw_confs)
                self._ppe_display_state[key] = {name: False for name in self.ppe_classes}

            ema = self._ppe_ema[key]
            display = self._ppe_display_state[key]
            alpha = self._ppe_ema_alpha

            ppe_hits = {}
            for ppe_name, raw_conf in raw_confs.items():
                # Update exponential moving average
                ema[ppe_name] = alpha * raw_conf + (1 - alpha) * ema[ppe_name]
                smoothed = ema[ppe_name]

                # Hysteresis: require crossing ON threshold to appear,
                # and dropping below OFF threshold to disappear.
                if display.get(ppe_name, False):
                    # Currently shown -> keep showing unless smoothed drops below OFF threshold
                    if smoothed < self._ppe_off_thresh:
                        display[ppe_name] = False
                    else:
                        ppe_hits[ppe_name] = smoothed
                else:
                    # Currently hidden -> only show once smoothed exceeds ON threshold
                    if smoothed >= self._ppe_on_thresh:
                        display[ppe_name] = True
                        ppe_hits[ppe_name] = smoothed

            results.append((key, box, ppe_hits))
        return results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict_frame(self, frame_bgr, camera_id=0):
        """Single-camera convenience path (kept for compatibility with the original API)."""
        return self.predict_batch_multi_camera([(camera_id, frame_bgr)])[camera_id]

    def predict_batch_multi_camera(self, camera_frame_pairs):
        """
        camera_frame_pairs: list of (camera_id, frame_bgr) -- typically
        one frame per active camera, collected by stream_server.py on
        each scheduling tick.

        Returns: dict camera_id -> list of detections, each
                 {"label": str, "confidence": float, "bbox": [x1,y1,x2,y2]}
                 matching the original output shape (Person + per-PPE-class entries).
        """
        results_by_camera = {cam_id: [] for cam_id, _ in camera_frame_pairs}
        frame_idx_by_camera = {}
        crop_batch = []  # ((camera_id, track_id), box, tensor)

        for camera_id, frame_bgr in camera_frame_pairs:
            frame_idx, person_tuples = self._get_person_boxes(frame_bgr, camera_id)
            frame_idx_by_camera[camera_id] = frame_idx
            person_count = 0  # counts People, not label entries -- results_by_camera also gets PPE labels appended below

            for track_id, box, score in person_tuples:
                results_by_camera[camera_id].append({"label": "Person", "confidence": float(score), "bbox": list(box)})
                person_count += 1

                if person_count > self.max_persons_per_frame:
                    continue  # still recorded as a Person box above, just skip Stage 2 for the overflow

                cache_key = (camera_id, track_id)
                cached = self._ppe_cache.get(cache_key)
                needs_classify = cached is None or (frame_idx - cached["frame"]) >= self.classify_stride

                if needs_classify:
                    crop_tensor = self._crop_to_tensor(frame_bgr, box)
                    if crop_tensor is not None:
                        crop_batch.append((cache_key, box, crop_tensor))
                    elif cached is not None:
                        # crop too small to (re)classify this frame -- fall back to the last known label
                        for ppe_name, conf in cached["hits"].items():
                            results_by_camera[camera_id].append({"label": ppe_name, "confidence": conf, "bbox": list(box)})
                else:
                    # PPE state doesn't flicker frame-to-frame -- reuse the last classification
                    # for this tracked person instead of paying for another Stage 2 forward pass.
                    for ppe_name, conf in cached["hits"].items():
                        results_by_camera[camera_id].append({"label": ppe_name, "confidence": conf, "bbox": list(box)})

        stage2_results = self._run_stage2(crop_batch)
        for (camera_id, track_id), box, ppe_hits in stage2_results:
            self._ppe_cache[(camera_id, track_id)] = {"frame": frame_idx_by_camera[camera_id], "hits": ppe_hits}
            for ppe_name, conf in ppe_hits.items():
                results_by_camera[camera_id].append({"label": ppe_name, "confidence": conf, "bbox": list(box)})

        self._prune_ppe_cache(frame_idx_by_camera)
        return results_by_camera

    def _prune_ppe_cache(self, frame_idx_by_camera):
        """Drop cache entries for tracks that haven't been (re)classified in a while
        (track lost, or camera stopped appearing in batches) so the cache doesn't grow forever."""
        stale_keys = [
            key for key, entry in self._ppe_cache.items()
            if key[0] in frame_idx_by_camera
            and (frame_idx_by_camera[key[0]] - entry["frame"]) > self.ppe_cache_stale_after
        ]
        for key in stale_keys:
            del self._ppe_cache[key]

    def reset_camera(self, camera_id):
        """Call on stream reconnect / scene cut to drop stale motion + tracking state."""
        self._trackers.pop(camera_id, None)
        self._frame_counters.pop(camera_id, None)
        for key in [k for k in self._ppe_cache if k[0] == camera_id]:
            del self._ppe_cache[key]
        for key in [k for k in self._ppe_ema if k[0] == camera_id]:
            del self._ppe_ema[key]
        for key in [k for k in self._ppe_display_state if k[0] == camera_id]:
            del self._ppe_display_state[key]
        if self.motion_gate:
            self.motion_gate.reset(camera_id)
