"""The decision logic of the three detectors, with no third-party imports.

Everything here is either a pure function or a small counter over plain
floats and strings. The detector classes in ear.py, head_pose.py and
phone_detector.py hold the camera, the MediaPipe graph and the YOLO model;
this module holds what they decide once the frame has been reduced to
numbers.

The split exists so the rules can be unit-tested. Importing dms.ear pulls
in OpenCV and MediaPipe, and dms.phone_detector pulls in Ultralytics and
therefore Torch, which is roughly two gigabytes of dependencies to check an
inequality. Importing this module costs the standard library.

Behaviour is byte-for-byte the behaviour of the methods it was lifted from,
including the two quirks documented at EarCalibrator.update.
"""

from math import dist

from .config import PHONE_CLASS_ID, EarConfig, PoseConfig


def eye_aspect_ratio(left_eye, right_eye) -> float:
    """Mean eye aspect ratio of both eyes, rounded to two decimals.

    Each eye is the 16-point MediaPipe contour. Index [0] and [8] are the
    horizontal corners, [4] and [12] the vertical extremes, so the ratio is
    eye height over eye width: it falls as the lid closes and is roughly
    invariant to how far the driver is from the camera.
    """
    left = dist(left_eye[12], left_eye[4]) / dist(left_eye[0], left_eye[8])
    right = dist(right_eye[12], right_eye[4]) / dist(right_eye[0], right_eye[8])
    return round((left + right) / 2.0, 2)


class EarCalibrator:
    """Accumulates a per-driver baseline EAR over the opening frames.

    Two behaviours here are deliberate and are asserted in the tests, because
    both are easy to "tidy" into something different:

    - Frames at or below MIN_VALID_EAR are skipped rather than averaged, so a
      blink during calibration does not drag the baseline down. A driver
      whose open-eye EAR sits below that floor therefore never calibrates.
    - The baseline is published on the call *after* the counter fills, not on
      the call that fills it, so calibration takes N_INITIAL_FRAMES + 1 valid
      frames.
    """

    def __init__(self, config=None):
        self.config = config if config is not None else EarConfig()
        self.initial_ear = 0.0
        self._sum = 0.0
        self._count = 0

    @property
    def calibrated(self) -> bool:
        return self.initial_ear != 0.0

    def update(self, avg_ear: float) -> float:
        """Feed one frame's EAR. Returns the baseline (0.0 until calibrated)."""
        if self._count >= self.config.N_INITIAL_FRAMES:
            self.initial_ear = round(self._sum / self.config.N_INITIAL_FRAMES, 2)
            self._sum = 0.0
            self._count = 0
        elif avg_ear > self.config.MIN_VALID_EAR:
            self._sum += avg_ear
            self._count += 1
        return self.initial_ear


class DrowsinessCounter:
    """Turns a stream of EAR values into Awake / Drowsy.

    The point of the counter is to separate a blink from drowsiness: the eye
    has to stay below the threshold for longer than HOLDING_FRAMES, and any
    single frame above the threshold resets the run.
    """

    def __init__(self, config=None):
        self.config = config if config is not None else EarConfig()
        self.count = 0

    def update(self, avg_ear: float, initial_ear: float) -> str:
        if avg_ear < initial_ear * self.config.DROWSINESS_FACTOR:
            self.count += 1
            if self.count > self.config.HOLDING_FRAMES:
                return "Drowsy"
            return "Awake"  # possibly just a blink
        self.count = 0
        return "Awake"


def classify_direction(pitch_deg: float, yaw_deg: float, config=None) -> str:
    """Where the driver is looking, from head pitch and yaw in degrees.

    Yaw is tested before pitch, so a head turned away *and* tilted down is
    reported as Looking Left or Looking Right. The pitch band is asymmetric
    on purpose: looking down is the drowsiness posture and is caught earlier
    (-15 deg) than looking up (+20 deg).
    """
    c = config if config is not None else PoseConfig()
    if yaw_deg < -c.YAW_THRESHOLD_DEG:
        return "Looking Left"
    if yaw_deg > c.YAW_THRESHOLD_DEG:
        return "Looking Right"
    if pitch_deg < c.PITCH_DOWN_DEG:
        return "Looking Down"
    if pitch_deg > c.PITCH_UP_DEG:
        return "Looking Up"
    return "Forward"


def filter_phone_boxes(classes, boxes, class_id: int = PHONE_CLASS_ID) -> list:
    """Keep only the boxes whose predicted class is the phone class.

    The custom checkpoint has a single class, so this is a guard rather than
    a choice, and it is the reason the original `if classes == 0:` had to go:
    comparing a whole array raised "truth value of an array is ambiguous" the
    moment two detections appeared in one frame.
    """
    # strict=True on purpose: the two arrays come off the same detection
    # result and must be the same length. Silently truncating a mismatch
    # would drop detections without anyone noticing.
    return [box for cls, box in zip(classes, boxes, strict=True) if cls == class_id]
