"""Standalone phone-usage detection demo (custom YOLOv8)."""

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dms import PhoneDetector  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", default="models/yolov8m_custom.pt")
    args = parser.parse_args()

    detector = PhoneDetector(args.model)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera index {args.camera}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame, detected, _ = detector.process(frame)
            cv2.imshow("Phone Detection", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
