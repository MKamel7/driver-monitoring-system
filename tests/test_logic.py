"""Unit tests for the detector decision logic.

These cover the five decisions the system actually makes once a frame has
been reduced to numbers: the eye aspect ratio, the calibration of a
per-driver baseline, the blink-versus-drowsiness state machine, the head
direction classifier, and the phone class filter.

They import only dms.logic and dms.config, so they run without OpenCV,
MediaPipe, Ultralytics or a camera, and without the model weights.

Every assertion here was checked against a deliberately broken copy of its
target before being kept, so none of them is a test that cannot fail.
"""

import pytest

from dms.config import EarConfig, PoseConfig
from dms.logic import (
    DrowsinessCounter,
    EarCalibrator,
    classify_direction,
    eye_aspect_ratio,
    filter_phone_boxes,
)


def eye_contour(width, height, cx=100.0, cy=100.0):
    """A 16-point eye contour with the given width and height.

    Only the four indices the ratio reads are meaningful: [0] and [8] are the
    horizontal corners, [4] and [12] the vertical extremes. The rest are
    filler, exactly as the unread points of the real MediaPipe contour are.
    """
    points = [(cx, cy)] * 16
    points[0] = (cx - width / 2.0, cy)
    points[8] = (cx + width / 2.0, cy)
    points[4] = (cx, cy - height / 2.0)
    points[12] = (cx, cy + height / 2.0)
    return points


# --------------------------------------------------------------------------
# 1. The eye aspect ratio itself
# --------------------------------------------------------------------------

def test_ear_is_height_over_width_and_falls_as_the_eye_closes():
    wide_open = eye_contour(width=60.0, height=18.0)   # 0.30
    closed = eye_contour(width=60.0, height=3.0)       # 0.05

    assert eye_aspect_ratio(wide_open, wide_open) == 0.30
    assert eye_aspect_ratio(closed, closed) == 0.05

    # Guards the index layout: reading the corners as the vertical pair would
    # invert the ratio to 3.33 rather than lowering it.
    assert eye_aspect_ratio(closed, closed) < eye_aspect_ratio(wide_open, wide_open)


def test_ear_is_invariant_to_how_far_away_the_driver_sits():
    near = eye_contour(width=80.0, height=24.0)
    far = eye_contour(width=20.0, height=6.0)

    # Same eye, quarter the size on the sensor. A ratio must not care.
    assert eye_aspect_ratio(near, near) == eye_aspect_ratio(far, far)


def test_ear_averages_both_eyes_rather_than_reporting_one():
    left = eye_contour(width=60.0, height=18.0)   # 0.30 on its own
    right = eye_contour(width=60.0, height=6.0)   # 0.10 on its own

    combined = eye_aspect_ratio(left, right)

    assert combined == 0.20
    # Strictly between the two, so returning either eye alone fails here.
    assert 0.10 < combined < 0.30


# --------------------------------------------------------------------------
# 2. Calibration of the per-driver baseline
# --------------------------------------------------------------------------

def test_calibration_publishes_the_mean_of_the_accepted_frames():
    cal = EarCalibrator()
    n = EarConfig.N_INITIAL_FRAMES

    # Alternating either side of 0.30 so a mean of 0.30 is not also the
    # value of any single frame.
    for i in range(n):
        cal.update(0.28 if i % 2 else 0.32)

    # The baseline is published on the call after the counter fills, not on
    # the call that fills it. That is the shipped behaviour and callers rely
    # on `calibrated` rather than on a frame count.
    assert cal.initial_ear == 0.0
    assert cal.calibrated is False

    cal.update(0.30)

    assert cal.initial_ear == 0.30
    assert cal.calibrated is True


def test_calibration_skips_blinks_instead_of_averaging_them_in():
    cal = EarCalibrator()
    n = EarConfig.N_INITIAL_FRAMES

    for _ in range(10):
        cal.update(0.05)          # blinks, well below MIN_VALID_EAR
    for _ in range(n):
        cal.update(0.30)          # open-eye frames
    cal.update(0.30)

    # Averaging the blinks in would drag this far below 0.30.
    assert cal.initial_ear == 0.30


def test_a_driver_at_the_validity_floor_never_calibrates():
    cal = EarCalibrator()

    # MIN_VALID_EAR is an exclusive floor: exactly 0.2 is rejected.
    for _ in range(EarConfig.N_INITIAL_FRAMES * 3):
        cal.update(EarConfig.MIN_VALID_EAR)

    # Documented consequence: a driver whose open-eye EAR sits at or below
    # the floor calibrates forever and is never assessed for drowsiness.
    assert cal.calibrated is False
    assert cal.initial_ear == 0.0


# --------------------------------------------------------------------------
# 3. Blink versus drowsiness
# --------------------------------------------------------------------------

BASELINE = 0.30                                    # calibrated open-eye EAR
THRESHOLD = BASELINE * EarConfig.DROWSINESS_FACTOR  # 0.225
CLOSED = 0.20                                      # below the threshold
OPEN = 0.30                                        # above it


def test_a_long_blink_is_not_drowsiness():
    counter = DrowsinessCounter()

    # A closure exactly as long as the holding window is still a blink.
    for _ in range(EarConfig.HOLDING_FRAMES):
        assert counter.update(CLOSED, BASELINE) == "Awake"


def test_one_frame_past_the_holding_window_is_drowsiness():
    counter = DrowsinessCounter()

    for _ in range(EarConfig.HOLDING_FRAMES):
        counter.update(CLOSED, BASELINE)

    assert counter.update(CLOSED, BASELINE) == "Drowsy"


def test_a_single_open_frame_resets_the_run():
    counter = DrowsinessCounter()

    for _ in range(EarConfig.HOLDING_FRAMES):
        counter.update(CLOSED, BASELINE)
    assert counter.update(OPEN, BASELINE) == "Awake"

    # The run has to start over: the whole window again is still a blink.
    for _ in range(EarConfig.HOLDING_FRAMES):
        assert counter.update(CLOSED, BASELINE) == "Awake"
    assert counter.update(CLOSED, BASELINE) == "Drowsy"


def test_the_threshold_itself_counts_as_open():
    counter = DrowsinessCounter()

    for _ in range(EarConfig.HOLDING_FRAMES * 2):
        assert counter.update(THRESHOLD, BASELINE) == "Awake"


# --------------------------------------------------------------------------
# 4. Head direction
# --------------------------------------------------------------------------

def test_direction_reports_each_of_the_five_states():
    assert classify_direction(pitch_deg=0, yaw_deg=0) == "Forward"
    assert classify_direction(pitch_deg=0, yaw_deg=-30) == "Looking Left"
    assert classify_direction(pitch_deg=0, yaw_deg=30) == "Looking Right"
    assert classify_direction(pitch_deg=-30, yaw_deg=0) == "Looking Down"
    assert classify_direction(pitch_deg=30, yaw_deg=0) == "Looking Up"


@pytest.mark.parametrize("pitch,yaw", [
    (0, -PoseConfig.YAW_THRESHOLD_DEG),   # exactly at the yaw limit
    (0, PoseConfig.YAW_THRESHOLD_DEG),
    (PoseConfig.PITCH_DOWN_DEG, 0),       # exactly at the pitch limits
    (PoseConfig.PITCH_UP_DEG, 0),
])
def test_the_thresholds_are_exclusive(pitch, yaw):
    assert classify_direction(pitch_deg=pitch, yaw_deg=yaw) == "Forward"


def test_the_pitch_band_is_asymmetric_on_purpose():
    # Looking down is the drowsiness posture, so it is caught 16 degrees
    # earlier than looking up. Symmetric thresholds would break this.
    assert classify_direction(pitch_deg=-16, yaw_deg=0) == "Looking Down"
    assert classify_direction(pitch_deg=16, yaw_deg=0) == "Forward"


def test_yaw_is_decided_before_pitch():
    # A head both turned away and tilted down reports the turn. This is the
    # known consequence of the ordering: such a frame never increments the
    # looking-down counter that feeds Suspected Drowsiness.
    assert classify_direction(pitch_deg=-40, yaw_deg=-40) == "Looking Left"


# --------------------------------------------------------------------------
# 5. The phone class filter
# --------------------------------------------------------------------------

def test_only_phone_class_boxes_survive_and_order_is_kept():
    boxes = [(0, 0, 10, 10), (20, 20, 30, 30), (40, 40, 50, 50)]
    classes = [0, 1, 0]

    assert filter_phone_boxes(classes, boxes, class_id=0) == [boxes[0], boxes[2]]


def test_no_phone_and_no_detections_both_yield_nothing():
    boxes = [(0, 0, 10, 10), (20, 20, 30, 30)]

    assert filter_phone_boxes([1, 2], boxes, class_id=0) == []
    assert filter_phone_boxes([], [], class_id=0) == []


def test_a_class_and_box_length_mismatch_is_loud():
    # A plain zip would truncate and drop detections with nothing logged.
    with pytest.raises(ValueError):
        filter_phone_boxes([0, 0], [(0, 0, 10, 10)], class_id=0)


def test_two_detections_in_one_frame_do_not_raise():
    """The regression the 2026 refactor fixed.

    The original code compared the whole class array at once (`if classes ==
    0:`), which raised "truth value of an array is ambiguous" the moment a
    second detection appeared. Real numpy arrays are used here because that
    is the only way this can fail.
    """
    np = pytest.importorskip("numpy")

    classes = np.array([0, 0, 1], dtype=int)
    boxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]], dtype=int)

    kept = filter_phone_boxes(classes, boxes, class_id=0)

    assert len(kept) == 2
    assert np.array_equal(kept[0], boxes[0])
    assert np.array_equal(kept[1], boxes[1])
