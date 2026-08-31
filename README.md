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
python scripts/fetch_models.py    # only needed for phone detection
python apps/run_dms.py --phone    # + YOLOv8 phone detection
```

Each detector also runs standalone: `run_ear.py`, `run_head_pose.py`, `run_phone_detection.py`.

The two custom checkpoints are published as [Release assets](https://github.com/MKamel7/driver-monitoring-system/releases/tag/v1.0.0) rather than tracked in git, so a clone is text only. `fetch_models.py` verifies each download against a recorded SHA-256. The MediaPipe half of the pipeline needs no checkpoint at all.

### Tests

```bash
pip install pytest
pytest
```

The decision logic lives in `src/dms/logic.py`, which imports nothing but the standard library, so the suite runs in under a second without OpenCV, MediaPipe, Ultralytics, a camera or the checkpoints. It covers the eye aspect ratio, the per-driver calibration, the blink-versus-drowsiness state machine at both of its boundaries, the head direction classifier at its thresholds, and the phone class filter. Every assertion was checked against a deliberately broken copy of its target before it was kept, so none of them is a test that cannot fail.

It does **not** cover MediaPipe or YOLO inference itself, or the accuracy figures above. See *Reproducibility of the measured results*.

### ROS 1 (Noetic)

```bash
cd ros_ws && catkin_make && source devel/setup.bash
roslaunch drowsiness_detection_pkg project.launch
```

## Repository layout

```
src/dms/            core detectors (importable package)
  config.py           thresholds, no third-party imports
  logic.py            the decision logic, no third-party imports
  ear.py              MediaPipe face mesh + EAR drowsiness
  head_pose.py        solvePnP head pose + attention
  phone_detector.py   YOLOv8 phone detection
apps/               webcam demos, single-process
tests/              unit tests for the decision logic (no deps, no weights)
scripts/            fetch_models.py, smoke_import.py
ros_ws/             ROS 1 package: camera → EAR + pose → fused result
models/             checkpoints, fetched not tracked (see models/README.md)
alternatives/       comparative studies: 3 dlib EAR variants, YOLOv4-tiny detection
```

The `alternatives/` folder documents the evaluation that led to the final design — dlib (accurate, slow), a custom-trained dlib eye predictor (fast, less accurate), and MediaPipe (chosen: accurate *and* fast), plus a YOLOv4-tiny baseline that the custom YOLOv8 model replaced.

## Reproducibility of the measured results

The two numbers in the table at the top were measured during the graduation
project, on a Jetson Nano and a Raspberry Pi, against a test set that is not
part of this repository. **No script here regenerates them**, and until one
does they should be read as reported results rather than as something you can
check by cloning.

Two further caveats worth stating plainly, because they are the ones an
engineer would ask about:

- **Accuracy is a weak metric for drowsiness detection.** What matters
  operationally is sensitivity, specificity, false alarms per hour and
  detection latency. A monitor that cries wolf twice an hour is unusable
  whatever its accuracy.
- **The split was not subject-wise.** Reporting per-subject cross-validation
  would very likely lower the headline number, which is exactly why it is the
  right thing to report.

Fixing this means building an evaluation, not adjusting the wording. It is
tracked as the next piece of work on this repository.

## 2026 maintenance release

Checked and made runnable, 2026-08-31:

- Unit tests for the decision logic, and CI that runs them, lints, and proves
  the pinned dependencies still resolve. The repository had none of these.
- The logic moved into `src/dms/logic.py` and `src/dms/config.py`, neither of
  which imports a third-party package, so it can be tested in a second rather
  than behind a two-gigabyte install. The detector classes are unchanged in
  behaviour and keep their existing interfaces; `dms` now exports them lazily.
- `zip()` in the phone class filter is strict, so a class/box length mismatch
  raises instead of silently dropping detections.
- `EarDetector` and `HeadPoseEstimator` no longer share one config object
  between every instance (a mutable default argument evaluated at import).
- The 68 MiB of checkpoints are Release assets fetched by a checksum-verifying
  script instead of tracked blobs.

Earlier fixes, from the original clean-up:

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
