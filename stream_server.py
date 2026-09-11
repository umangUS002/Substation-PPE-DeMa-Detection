"""
stream_server.py
-----------------
Reference scaffold for the "single inference server" that services all
25 CCTV feeds off one shared GPU pipeline (optimized_adapter.py).

Architecture
============
  25x DecoderThread (one per camera)          1x InferenceWorker            AlertDispatcher
  ------------------------------------  -->   -------------------   -->    -----------------
  reads frames, timestamps them,              pulls the newest frame       checks elapsed time
  drops stale frames if the queue             per camera each tick,        vs. the 5s latency
  is backed up (never blocks on a              calls the shared             budget, emits/logs
  slow camera)                                DINOAdapterOptimized          alerts

Key design choices, and why
============================
- **Latest-frame-wins queues, not FIFO.** If the GPU falls behind for a
  moment, we want the *next* frame processed to be the current state of
  that camera, not a 2-second-old backlog -- backlogged frames only
  make the 5s latency budget worse. Each camera's queue holds exactly
  one frame; a decoder overwrites it rather than piling up.

- **One inference worker, micro-batched across cameras.** This is what
  "single inference server" means in practice: rather than 25
  independent model instances (which wouldn't fit in 4GB VRAM and
  would thrash the GPU with context switching), one loop gathers
  "whatever's new across the 25 queues right now" into a single call
  to `predict_batch_multi_camera`, so Stage 1 tile batches and Stage 2
  crop batches are as large as possible per GPU dispatch. Bigger
  batches amortize kernel-launch overhead, which matters more on a
  smaller GPU than on a datacenter card.

- **Video decode is intentionally out of scope here.** For 25 x 2MP
  streams, CPU-based decode (plain cv2.VideoCapture pulling RTSP) can
  itself become the bottleneck before the GPU ever sees a frame. On
  the actual deployment box, decode should go through hardware NVDEC
  (e.g. GStreamer's `nvv4l2decoder` / `nvh264dec`, or ffmpeg with
  `-hwaccel cuda`) rather than cv2's default software decoder --
  swap that into `DecoderThread._read_frame` for production. This
  scaffold uses cv2.VideoCapture as a stand-in so the file runs
  against any RTSP URL or video file for testing.

- **Latency accounting.** Every frame is timestamped at capture; the
  dispatcher measures capture-to-alert latency and logs a warning if
  it exceeds the 5s budget, so latency regressions are visible instead
  of silently eating into the SLA.

This is a scaffold, not a production service: no reconnect/backoff
logic, no health checks, no metrics export. Those are the next layer
to add once the core pipeline's throughput is validated against real
footage.
"""

import queue
import threading
import time
import logging

import cv2

from optimized_adapter import DINOAdapterOptimized

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("stream_server")

LATENCY_BUDGET_SECONDS = 5.0


class LatestFrameQueue:
    """A 1-slot mailbox: put() overwrites, get() never blocks past a short timeout."""

    def __init__(self):
        self._lock = threading.Lock()
        self._item = None  # (frame_bgr, capture_ts)

    def put(self, frame_bgr, capture_ts):
        with self._lock:
            self._item = (frame_bgr, capture_ts)

    def get_and_clear(self):
        with self._lock:
            item = self._item
            self._item = None
            return item


class DecoderThread(threading.Thread):
    def __init__(self, camera_id, source, frame_queue, target_fps=15, stop_event=None):
        super().__init__(daemon=True, name=f"decoder-{camera_id}")
        self.camera_id = camera_id
        self.source = source
        self.frame_queue = frame_queue
        self.min_interval = 1.0 / max(1, target_fps)
        self.stop_event = stop_event or threading.Event()

    def run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            log.error(f"camera {self.camera_id}: failed to open source {self.source}")
            return

        last_push = 0.0
        try:
            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    log.warning(f"camera {self.camera_id}: read failed, retrying")
                    time.sleep(0.5)
                    continue

                now = time.monotonic()
                if now - last_push < self.min_interval:
                    continue  # throttle to the analytic rate; decode-only frames are simply dropped
                last_push = now
                self.frame_queue.put(frame, now)
        finally:
            cap.release()

    def stop(self):
        self.stop_event.set()


class InferenceWorker(threading.Thread):
    """
    One shared model instance. Each tick, gathers the latest available
    frame from every camera queue and runs one batched inference call.
    """

    def __init__(self, camera_queues, adapter: DINOAdapterOptimized, alert_queue, tick_interval=None):
        super().__init__(daemon=True, name="inference-worker")
        self.camera_queues = camera_queues  # dict camera_id -> LatestFrameQueue
        self.adapter = adapter
        self.alert_queue = alert_queue
        # If not set, run flat-out (bounded by GPU throughput); a small
        # floor avoids busy-spinning when every camera queue is empty.
        self.tick_interval = tick_interval if tick_interval is not None else 0.01
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.is_set():
            batch = []  # (camera_id, frame_bgr)
            capture_ts_by_camera = {}

            for camera_id, q in self.camera_queues.items():
                item = q.get_and_clear()
                if item is None:
                    continue
                frame, capture_ts = item
                batch.append((camera_id, frame))
                capture_ts_by_camera[camera_id] = capture_ts

            if not batch:
                time.sleep(self.tick_interval)
                continue

            t0 = time.monotonic()
            results_by_camera = self.adapter.predict_batch_multi_camera(batch)
            infer_ms = (time.monotonic() - t0) * 1000.0

            now = time.monotonic()
            for camera_id, detections in results_by_camera.items():
                capture_ts = capture_ts_by_camera.get(camera_id, now)
                latency = now - capture_ts
                self.alert_queue.put((camera_id, detections, capture_ts, latency))

            if len(batch) > 0:
                log.debug(f"inference tick: {len(batch)} cameras, {infer_ms:.1f}ms")

    def stop(self):
        self._stop_event.set()


class AlertDispatcher(threading.Thread):
    """
    Turns raw per-frame detections into alerts (e.g. a person with a
    missing PPE item) and checks the result against the 5s latency
    budget. Replace `_emit_alert` with your actual sink (MQTT, DB,
    NTPC control-room integration, etc.).
    """

    def __init__(self, alert_queue, ppe_classes=("Helmet", "Gloves", "Footwear")):
        super().__init__(daemon=True, name="alert-dispatcher")
        self.alert_queue = alert_queue
        self.ppe_classes = set(ppe_classes)
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.is_set():
            try:
                camera_id, detections, capture_ts, latency = self.alert_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            violations = self._find_violations(detections)
            if violations:
                self._emit_alert(camera_id, violations, latency)

            if latency > LATENCY_BUDGET_SECONDS:
                log.warning(f"camera {camera_id}: end-to-end latency {latency:.2f}s exceeded "
                            f"{LATENCY_BUDGET_SECONDS:.0f}s budget")

    def _find_violations(self, detections):
        """
        Very simple example rule: a Person box with none of the PPE
        classes detected on it is treated as a violation. Replace with
        your actual per-item compliance logic (e.g. spatial overlap
        matching between the Person box and each PPE-class box).
        """
        persons = [d for d in detections if d["label"] == "Person"]
        ppe_present = {d["label"] for d in detections if d["label"] in self.ppe_classes}
        missing = self.ppe_classes - ppe_present
        if persons and missing:
            return {"person_count": len(persons), "missing_ppe": sorted(missing)}
        return None

    def _emit_alert(self, camera_id, violations, latency):
        log.info(f"ALERT camera={camera_id} missing={violations['missing_ppe']} "
                 f"persons={violations['person_count']} latency={latency:.2f}s")

    def stop(self):
        self._stop_event.set()


def build_server(camera_sources: dict, target_fps=15, **adapter_kwargs):
    """
    camera_sources: dict camera_id -> RTSP URL / video path / device index
    Returns (decoders, worker, dispatcher) -- call .start() on each, and
    .stop() on all three (in that order) to shut down cleanly.
    """
    adapter = DINOAdapterOptimized(**adapter_kwargs)

    camera_queues = {cam_id: LatestFrameQueue() for cam_id in camera_sources}
    alert_queue = queue.Queue()

    decoders = [
        DecoderThread(cam_id, src, camera_queues[cam_id], target_fps=target_fps)
        for cam_id, src in camera_sources.items()
    ]
    worker = InferenceWorker(camera_queues, adapter, alert_queue)
    dispatcher = AlertDispatcher(alert_queue)

    return decoders, worker, dispatcher


if __name__ == "__main__":
    # Example wiring for 25 cameras -- replace with your actual RTSP URLs.
    camera_sources = {f"cam{i:02d}": f"rtsp://192.168.1.{100+i}/stream1" for i in range(25)}

    decoders, worker, dispatcher = build_server(
        camera_sources,
        target_fps=15,
        backbone_mode="distilled",
        student_weights_path="student.pt",  # produced by lightweight_backbone.train_distillation + finetune_classifier
        classifier_weights_path="ppe_classifier.pt",
        detect_stride=3,
        classify_stride=5,
        motion_gating=True,
        use_fp16=True,
        max_persons_per_frame=12,
    )

    worker.start()
    dispatcher.start()
    for d in decoders:
        d.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("shutting down...")
        for d in decoders:
            d.stop()
        worker.stop()
        dispatcher.stop()
