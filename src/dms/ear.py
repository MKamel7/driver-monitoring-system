"""Eye Aspect Ratio (EAR) drowsiness detection using MediaPipe Face Mesh.

The detector calibrates a per-driver baseline EAR over the first frames,
then flags drowsiness when the current EAR stays below a fraction of that
baseline for a sustained number of frames (i.e. longer than a blink).

Measured on the original project hardware: 97.47% accuracy at ~18-20 FPS
on an NVIDIA Jetson Nano.

Original EAR implementation: Mario Ezzat (graduation project team).
Refactored 2026: module-level camera removed, landmark extraction
de-duplicated, state machine made explicit. The ratio, the calibration and
the state machine now live in dms.logic so they can be tested without
MediaPipe; this class holds the face mesh and the frame handling.
"""

import cv2
import mediapipe as mp

from .config import EarConfig
from .logic import DrowsinessCounter, EarCalibrator, eye_aspect_ratio

# MediaPipe Face Mesh landmark indices for the eye contours (16 points each).
# Index layout is symmetric: [0] and [8] are the horizontal corners,
# [4] and [12] are the vertical extremes used for the EAR ratio.
LEFT_EYE_LANDMARKS = [362, 382, 381, 380, 374, 373, 390, 249,
                      263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_LANDMARKS = [33, 7, 163, 144, 145, 153, 154, 155,
                       133, 173, 157, 158, 159, 160, 161, 246]

__all__ = ["EarConfig", "EarDetector",
           "LEFT_EYE_LANDMARKS", "RIGHT_EYE_LANDMARKS"]


class EarDetector:
    """Stateful per-driver EAR drowsiness detector."""

    def __init__(self, config: EarConfig | None = None):
        # One config per detector. The previous default evaluated EarConfig()
        # once at import, so every detector in a process shared one object.
        self.config = config if config is not None else EarConfig()
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            refine_landmarks=True,
            max_num_faces=1,
            static_image_mode=False,
        )
        self._calibrator = EarCalibrator(config)
        self._drowsy = DrowsinessCounter(config)

    @property
    def initial_ear(self) -> float:
        """The calibrated baseline EAR, 0.0 until calibration finishes."""
        return self._calibrator.initial_ear

    def process(self, frame_bgr):
        """Run one frame through the detector.

        Returns (frame, state) where state is one of
        "No Driver", "Awake", "Drowsy".
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return frame_bgr, "No Driver"

        landmarks = results.multi_face_landmarks[0].landmark
        h, w = frame_bgr.shape[:2]
        left_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h))
                    for i in LEFT_EYE_LANDMARKS]
        right_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h))
                     for i in RIGHT_EYE_LANDMARKS]

        avg_ear = eye_aspect_ratio(left_eye, right_eye)

        if not self._calibrator.calibrated:
            self._calibrator.update(avg_ear)
            return frame_bgr, "Awake"

        return frame_bgr, self._drowsy.update(avg_ear, self.initial_ear)
