"""
Quick self-test for motion_gate.py and person_tracker.py.
No torch dependency -- exercises the pure cv2/numpy logic that the
GPU pipeline builds on top of. Run: python3 test_motion_and_tracker.py
"""
import numpy as np
from dino_pipeline.motion_gate import MotionGate
from dino_pipeline.person_tracker import SimpleIOUTracker, _iou


def make_frame(h, w, moving_box=None, bg_val=40):
    """moving_box is (x1, y1, x2, y2) in pixel coordinates."""
    frame = np.full((h, w, 3), bg_val, dtype=np.uint8)
    if moving_box is not None:
        x1, y1, x2, y2 = moving_box
        frame[y1:y2, x1:x2] = 220
    return frame


def test_motion_gate_skips_static_tiles():
    gate = MotionGate(warmup_frames=3, force_interval=0, min_motion_frac=0.01)
    h, w = 480, 640
    tiles = [(0, 0, 240, 320), (0, 320, 240, 640), (240, 0, 480, 320), (240, 320, 480, 640)]

    # warm up on a static frame so MOG2 has a background model
    static = make_frame(h, w)
    for _ in range(6):
        gate.active_tile_indices("cam0", static, tiles)

    # now introduce motion only in the bottom-right tile (x1,y1,x2,y2 = 340,260,600,420)
    moving = make_frame(h, w, moving_box=(340, 260, 600, 420))
    active, forced = gate.active_tile_indices("cam0", moving, tiles)
    assert not forced
    assert active == [3], f"expected only tile 3 active, got {active}"
    print("test_motion_gate_skips_static_tiles: PASS", active)


def test_motion_gate_forces_periodically():
    gate = MotionGate(warmup_frames=1, force_interval=4, min_motion_frac=0.01)
    h, w = 240, 320
    tiles = [(0, 0, 240, 320)]
    static = make_frame(h, w)
    forced_flags = []
    for i in range(10):
        _, forced = gate.active_tile_indices("cam1", static, tiles)
        forced_flags.append(forced)
    assert forced_flags[0] is True  # warmup
    assert any(forced_flags[4:]), "expected a forced frame within the interval window"
    print("test_motion_gate_forces_periodically: PASS", forced_flags)


def test_tracker_matches_and_extrapolates():
    tracker = SimpleIOUTracker(iou_threshold=0.3)
    tracker.update_from_detections([([100, 100, 150, 200], 0.9)])
    assert len(tracker.tracks) == 1

    # person moves slightly to the right -- should match the same track
    tracker.update_from_detections([([108, 100, 158, 200], 0.91)])
    assert len(tracker.tracks) == 1
    tid = tracker.tracks[0].track_id

    # skip two frames of real detection -- propagate should extrapolate forward
    prop1 = tracker.propagate()
    prop2 = tracker.propagate()
    assert len(prop1) == 1 and len(prop2) == 1
    (tid2, box2, _score) = prop2[0]
    assert box2[0] > 108, f"expected extrapolated box to keep moving right, got {box2}"
    assert tid2 == tid, "track identity should be preserved across propagate()"
    assert tracker.tracks[0].track_id == tid, "track identity should be preserved across propagate()"
    print("test_tracker_matches_and_extrapolates: PASS", prop1, prop2)


def test_tracker_drops_stale_tracks():
    tracker = SimpleIOUTracker(iou_threshold=0.3, max_misses=1)
    tracker.update_from_detections([([0, 0, 10, 10], 0.9)])
    assert len(tracker.tracks) == 1
    # two consecutive frames with no matching detection -> track should be dropped
    tracker.update_from_detections([])
    tracker.update_from_detections([])
    assert len(tracker.tracks) == 0
    print("test_tracker_drops_stale_tracks: PASS")


def test_iou_sanity():
    assert _iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert _iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    print("test_iou_sanity: PASS")


if __name__ == "__main__":
    test_motion_gate_skips_static_tiles()
    test_motion_gate_forces_periodically()
    test_tracker_matches_and_extrapolates()
    test_tracker_drops_stale_tracks()
    test_iou_sanity()
    print("\nAll tests passed.")
