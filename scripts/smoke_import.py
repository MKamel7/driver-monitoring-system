"""Import every module with the real dependencies present, and build the two
detectors that need no checkpoint.

The unit tests deliberately run without OpenCV, MediaPipe or Ultralytics, so
nothing else in this repository would notice if the versions in
requirements.txt stopped resolving, or if a detector module stopped importing.
This script is that check. It is what the import-smoke CI job runs.

PhoneDetector is not constructed here: it loads models/yolov8m_custom.pt,
which is fetched rather than tracked (scripts/fetch_models.py).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> int:
    import dms.config  # noqa: F401
    import dms.ear  # noqa: F401
    import dms.head_pose  # noqa: F401
    import dms.logic  # noqa: F401
    import dms.phone_detector  # noqa: F401
    from dms import EarDetector, HeadPoseEstimator

    detector = EarDetector()
    assert detector.initial_ear == 0.0, "a fresh detector should not be calibrated"
    HeadPoseEstimator()

    print("imports and detector construction OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
