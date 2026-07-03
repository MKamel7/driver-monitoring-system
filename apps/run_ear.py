"""Standalone EAR drowsiness detection demo (MediaPipe)."""

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dms import EarDetector  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    detector = EarDetector()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera index {args.camera}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame, state = detector.process(frame)
            cv2.putText(frame, f"State: {state}  (baseline EAR: {detector.initial_ear})",
                        (10, 25), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (100, 0, 100), 1)
            cv2.imshow("EAR Drowsiness Detection", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
