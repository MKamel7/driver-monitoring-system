# Models

**No checkpoint is tracked in git.** Fetch the two custom ones with:

```bash
python scripts/fetch_models.py
```

That downloads them from the [v1.0.0
Release](https://github.com/MKamel7/driver-monitoring-system/releases/tag/v1.0.0)
into this folder and verifies each against a recorded SHA-256. Running it
again is free: a file that already matches is left alone, and a file that does
not match is reported rather than silently replaced.

## Custom-trained for this project (fetched by the script)

| File | Size | Used by |
|---|---|---|
| `yolov8m_custom.pt` | 49.6 MiB | Phone-usage detection (single class: phone). 97.8% mAP@0.5 on the project test set. |
| `eye_predictor.dat` | 18.2 MiB | Custom-trained dlib eye landmark predictor (`alternatives/dlib_ear/dlib__Custom_solution.py`). |

Both were tracked in git until 2026-08-31. They are Release assets now, so a
fresh clone no longer carries 68 MiB of checkpoint blobs. The blobs remain in
the history of the commits before that date, so existing clones still work and
no link breaks.

## Download separately (standard, publicly available)

Only needed for the legacy dlib variants in `alternatives/dlib_ear/`:

- `shape_predictor_68_face_landmarks.dat` — http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 (extract into this folder)
- `yolov4-tiny.weights` — https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights (for `alternatives/yolov4_tiny/`)

## What needs what

The MediaPipe pipeline (EAR drowsiness + head pose) needs **no checkpoint at
all** — MediaPipe ships its own face mesh, so `python apps/run_dms.py` runs on
a bare `pip install -r requirements.txt`. Only `--phone` needs
`yolov8m_custom.pt`.
