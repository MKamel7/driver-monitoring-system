"""Combined Driver Monitoring System demo — EAR + head pose (+ optional phone detection).

Runs all detectors on a single webcam stream in one process and overlays
the fused result. Phone detection is optional because the YOLOv8 model is
heavy on CPU-only machines.

Usage:
    python apps/run_dms.py               # EAR + head pose
    python apps/run_dms.py --phone       # EAR + head pose + phone detection
    python apps/run_dms.py --camera 1    # different webcam index
"""

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dms import EarDetector, HeadPoseEstimator, PhoneDetector  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="webcam index")
    parser.add_argument("--phone", action="store_true",
                        help="enable YOLOv8 phone detection (heavier)")
    parser.add_argument("--model", default="models/yolov8m_custom.pt",
                        help="path to the custom YOLOv8 weights")
    args = parser.parse_args()

    ear = EarDetector()
    pose = HeadPoseEstimator()
    phone = PhoneDetector(args.model) if args.phone else None

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera index {args.camera}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera stream ended.", file=sys.stderr)
                break

            frame, eye_state = ear.process(frame)
            frame, pose_state, direction = pose.process(frame, draw=False)
            phone_detected = False
            if phone is not None:
                frame, phone_detected, _ = phone.process(frame)

            # Fused status overlay
            alert = (eye_state == "Drowsy"
                     or pose_state == "Suspected Drowsiness"
                     or phone_detected)
            color = (0, 0, 255) if alert else (0, 200, 0)
            cv2.putText(frame, f"Eyes: {eye_state}", (10, 25),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, color, 1)
            cv2.putText(frame, f"Pose: {pose_state} ({direction})", (10, 50),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, color, 1)
            if phone is not None:
                cv2.putText(frame, f"Phone: {'DETECTED' if phone_detected else 'clear'}",
                            (10, 75), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, color, 1)
            if alert:
                cv2.putText(frame, "ALERT", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            cv2.imshow("Driver Monitoring System", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
