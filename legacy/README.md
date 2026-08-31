# legacy/

**The implementations as submitted for the graduation project, kept verbatim.**

This directory was `alternatives/`. The rename is the point: these files are a
record of what was handed in and of the comparison that led to the final
design. They are not maintained, they are not imported by `src/dms` or `apps/`,
and they are excluded from lint for that reason. Nothing here should be
"modernised" — a preserved artefact that has been tidied is no longer evidence
of anything.

## Why they are not merged into one script

The three dlib variants look like copy-paste of each other, and two of them
almost are. Merging them would nevertheless destroy information, so the
differences are written down here instead.

| file | predictor | eye landmarks | drowsiness state machine |
|---|---|---|---|
| `dlib_ear/dlib_solution.py` | `shape_predictor_68_face_landmarks.dat` | 68-point, indices 36-47 | intact |
| `dlib_ear/dlib_GTX_solution.py` | `shape_predictor_68_face_landmarks_GTX.dat` | 68-point, indices 36-47 | intact |
| `dlib_ear/dlib__Custom_solution.py` | `eye_predictor.dat` (custom-trained) | 12-point, indices 0-11 | **commented out** |

- **Base and GTX differ by twelve lines, of which three matter**: the predictor
  filename, a commented-out `argparse` import, and one reworded comment. A
  parameterised script would cover both, and that script would be new code
  standing where submitted code used to be.
- **`dlib__Custom_solution.py` is not a third copy.** It uses the custom
  12-point eye predictor, and its drowsiness state machine is commented out, so
  it sets `STATE = "Drowsy"` on a single low-EAR frame instead of requiring
  `Holding_Frames` consecutive ones. **It reports drowsiness on a blink.** That
  is a behavioural difference, and folding it into a shared script would hide
  the one thing about it worth knowing.

The maintained version of this logic, with the state machine intact and under
test, is `src/dms/logic.py`. That is where the collapse actually happened: one
implementation, tested, with the duplication removed from the code that runs.

## Running them, and what you need first

Every script here **opens the webcam and loads its weights at import time**, so
they can only be executed, never imported, and none of them is safe to pull
into a test.

Two of the three cannot run from a fresh clone at all, because their weights
are large public downloads that are gitignored rather than shipped:

- `shape_predictor_68_face_landmarks.dat` and its GTX variant, see
  [`models/README.md`](../models/README.md) for the links
- `yolov4_tiny/` needs `yolov4-tiny.weights`, same

`dlib__Custom_solution.py` runs once `python scripts/fetch_models.py` has
brought down `eye_predictor.dat`.

dlib itself is not in `requirements.txt`. It needs CMake and a C++ toolchain,
and the shipped pipeline does not use it.

## What the comparison concluded

dlib with the 68-point predictor was accurate and slow. The custom-trained
12-point eye predictor was faster and less accurate. MediaPipe was both
accurate and fast and is what the project uses. `yolov4_tiny/` is the phone
detection baseline that the custom-trained YOLOv8 model replaced, going from
about 2 FPS to about 10 FPS on the Raspberry Pi.
