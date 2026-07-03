"""Eye Aspect Ratio (EAR) drowsiness detection using MediaPipe Face Mesh.

The detector calibrates a per-driver baseline EAR over the first frames,
then flags drowsiness when the current EAR stays below a fraction of that
baseline for a sustained number of frames (i.e. longer than a blink).

Measured on the original project hardware: 97.47% accuracy at ~18-20 FPS
on an NVIDIA Jetson Nano.

Original EAR implementation: Mario Ezzat (graduation project team).
Refactored 2026: module-level camera removed, landmark extraction
de-duplicated, state machine made explicit.
"""

from math import dist

import cv2
import mediapipe as mp

# MediaPipe Face Mesh landmark indices for the eye contours (16 points each).
# Index layout is symmetric: [0] and [8] are the horizontal corners,
# [4] and [12] are the vertical extremes used for the EAR ratio.
LEFT_EYE_LANDMARKS = [362, 382, 381, 380, 374, 373, 390, 249,
                      263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_LANDMARKS = [33, 7, 163, 144, 145, 153, 154, 155,
                       133, 173, 157, 158, 159, 160, 161, 246]


class EarConfig:
    """Tunable thresholds (ranges from the original thesis experiments)."""
    N_INITIAL_FRAMES = 25       # frames used to calibrate the baseline EAR (20-40)
    HOLDING_FRAMES = 20         # frames below threshold before "Drowsy" (20-40, FPS-dependent)
    DROWSINESS_FACTOR = 0.75    # fraction of baseline EAR treated as "eye closing" (0.5-0.8)
    MIN_VALID_EAR = 0.2         # EAR values below this are ignored during calibration (blinks)


class EarDetector:
    """Stateful per-driver EAR drowsiness detector."""

    def __init__(self, config: EarConfig = EarConfig()):
        self.config = config
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            refine_landmarks=True,
            max_num_faces=1,
            static_image_mode=False,
        )
        self.initial_ear = 0.0
        self._calibration_sum = 0.0
        self._calibration_count = 0
        self._drowsy_counter = 0

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

        avg_ear = self._ear(left_eye, right_eye)

        if self.initial_ear == 0.0:
            self._calibrate(avg_ear)
            return frame_bgr, "Awake"

        return frame_bgr, self._state(avg_ear)

    def _ear(self, left_eye, right_eye) -> float:
        left = dist(left_eye[12], left_eye[4]) / dist(left_eye[0], left_eye[8])
        right = dist(right_eye[12], right_eye[4]) / dist(right_eye[0], right_eye[8])
        return round((left + right) / 2.0, 2)

    def _calibrate(self, avg_ear: float) -> None:
        """Accumulate a baseline EAR, skipping blink frames."""
        if self._calibration_count >= self.config.N_INITIAL_FRAMES:
            self.initial_ear = round(
                self._calibration_sum / self.config.N_INITIAL_FRAMES, 2)
            self._calibration_sum = 0.0
            self._calibration_count = 0
        elif avg_ear > self.config.MIN_VALID_EAR:
            self._calibration_sum += avg_ear
            self._calibration_count += 1

    def _state(self, avg_ear: float) -> str:
        if avg_ear < self.initial_ear * self.config.DROWSINESS_FACTOR:
            self._drowsy_counter += 1
            if self._drowsy_counter > self.config.HOLDING_FRAMES:
                return "Drowsy"
            return "Awake"  # possibly just a blink
        self._drowsy_counter = 0
        return "Awake"
