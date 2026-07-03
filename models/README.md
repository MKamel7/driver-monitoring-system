# Models

## Included (custom-trained for this project)

| File | Size | Used by |
|---|---|---|
| `yolov8m_custom.pt` | ~50 MB | Phone-usage detection (single class: phone). 97.8% mAP@0.5 on the project test set. |
| `eye_predictor.dat` | ~18 MB | Custom-trained dlib eye landmark predictor (`alternatives/dlib_ear/dlib__Custom_solution.py`). |

## Download separately (standard, publicly available)

Only needed for the legacy dlib variants in `alternatives/dlib_ear/`:

- `shape_predictor_68_face_landmarks.dat` — http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 (extract into this folder)
- `yolov4-tiny.weights` — https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights (for `alternatives/yolov4_tiny/`)

The main pipeline (MediaPipe EAR + head pose + YOLOv8) needs **no extra downloads**.
