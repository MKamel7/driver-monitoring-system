# Driver Monitoring System (DMS)

Real-time driver monitoring on embedded hardware: **drowsiness detection** (Eye Aspect Ratio), **attention tracking** (head pose), and **phone-usage detection** (custom-trained YOLOv8) — originally developed as a mechatronics engineering graduation project (AASTMT, graded A+, 1st place nationally at GPAct Talent Expo 2023) and deployed across a Raspberry Pi + Jetson Nano ROS setup.

## Measured results (original project hardware)

| Component | Metric | Platform |
|---|---|---|
| MediaPipe drowsiness detection | **97.47% accuracy** · ~18–20 FPS | NVIDIA Jetson Nano |
| YOLOv8 phone detection (custom model) | **97.8% mAP@0.5** · 0.75 mAP@[0.5:0.95] · ~10 FPS | Raspberry Pi |

An early TensorFlow prototype ran at ~2 FPS on the Raspberry Pi; switching to a custom-trained YOLOv8 model delivered the ~10 FPS above — the kind of constraint-driven pivot embedded vision work actually demands.

## How it works

```
                 ┌─────────────────┐
 webcam ────────▶│  camera node    │──── frames ────┬───────────────┐
                 └─────────────────┘                │               │
                                            ┌───────▼──────┐ ┌──────▼────────┐
                                            │  EAR         │ │  Head pose    │
                                            │  (MediaPipe) │ │  (solvePnP)   │
                                            └───────┬──────┘ └──────┬────────┘
                                                    │ eye state     │ attention state
                                                  ┌─▼───────────────▼─┐
                                                  │   result node     │──▶ fused alert
                                                  └───────────────────┘
```

- **EAR drowsiness:** calibrates a per-driver baseline Eye Aspect Ratio over the first ~25 frames, then flags drowsiness when EAR stays below 75% of baseline for longer than a blink (`src/dms/ear.py`).
- **Head pose:** solvePnP over six face-mesh landmarks → pitch/yaw → FOCUSED / NOT FOCUSED / Suspected Drowsiness (`src/dms/head_pose.py`).
- **Phone detection:** custom single-class YOLOv8 model, trained on a purpose-built dataset of hands holding phones (`src/dms/phone_detector.py`, weights in `models/`).
- The same detectors run either standalone (`apps/`) or as a distributed **ROS 1 (Noetic)** node graph (`ros_ws/`).

## Quickstart

```bash
pip install -r requirements.txt

python apps/run_dms.py            # EAR + head pose on your webcam
python apps/run_dms.py --phone    # + YOLOv8 phone detection
```

Each detector also runs standalone: `run_ear.py`, `run_head_pose.py`, `run_phone_detection.py`.

### ROS 1 (Noetic)

```bash
cd ros_ws && catkin_make && source devel/setup.bash
roslaunch drowsiness_detection_pkg project.launch
```

## Repository layout

```
src/dms/            core detectors (importable package)
apps/               webcam demos, single-process
ros_ws/             ROS 1 package: camera → EAR + pose → fused result
models/             custom-trained weights (+ download links for standard ones)
alternatives/       comparative studies: 3 dlib EAR variants, YOLOv4-tiny detection
```

The `alternatives/` folder documents the evaluation that led to the final design — dlib (accurate, slow), a custom-trained dlib eye predictor (fast, less accurate), and MediaPipe (chosen: accurate *and* fast), plus a YOLOv4-tiny baseline that the custom YOLOv8 model replaced.

## 2026 maintenance release

This repository is the cleaned-up version of the original project code. Fixes:

- Head-pose module opened its own camera at import time, conflicting with the ROS camera node.
- Image width/height were swapped when building the solvePnP camera matrix.
- ROS publishers were re-created inside subscriber callbacks on every frame.
- `rospy.spin()` wrapped in a busy loop; shutdown hooks registered after spin (never ran).
- Crash-on-exit typos (`cap.relaese()`, `destroyAllWindow()`), and an ambiguous NumPy
  truth-value error in the YOLOv8 demo as soon as two objects were detected.
- Landmark extraction de-duplicated (16 hand-written tuples → index-list comprehensions).

## Credits

Graduation project — Mechatronics, Robotics & Automation Engineering, AASTMT Alexandria (2023). Built by the project team; this repository is maintained by [Mo Kamel](https://github.com/MKamel7).

## License

MIT
