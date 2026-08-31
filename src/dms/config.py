"""Tunable thresholds for the detectors.

Kept in a module of its own, with no third-party imports, so that the
thresholds and the logic that reads them (dms.logic) can be imported and
tested without pulling in OpenCV, MediaPipe or Ultralytics.

The values are unchanged from the original thesis experiments; the ranges
in the comments are the ranges that were tried.
"""


class EarConfig:
    """Tunable thresholds (ranges from the original thesis experiments)."""
    N_INITIAL_FRAMES = 25       # frames used to calibrate the baseline EAR (20-40)
    HOLDING_FRAMES = 20         # frames below threshold before "Drowsy" (20-40, FPS-dependent)
    DROWSINESS_FACTOR = 0.75    # fraction of baseline EAR treated as "eye closing" (0.5-0.8)
    MIN_VALID_EAR = 0.2         # EAR values below this are ignored during calibration (blinks)


class PoseConfig:
    YAW_THRESHOLD_DEG = 15      # |yaw| beyond this counts as looking away
    PITCH_DOWN_DEG = -15        # pitch below this counts as looking down
    PITCH_UP_DEG = 20           # pitch above this counts as looking up
    AWAY_FRAMES = 40            # consecutive away-frames before NOT FOCUSED
    DROWSY_FRAMES = 40          # consecutive down-frames before Suspected Drowsiness


PHONE_CLASS_ID = 0  # the custom model has a single class: phone
