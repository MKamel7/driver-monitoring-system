"""Head pose estimation using MediaPipe Face Mesh + solvePnP.

Estimates where the driver is looking from six facial landmarks and
classifies attention: FOCUSED, NOT FOCUSED (looking away too long), or
Suspected Drowsiness (head pitched down for a sustained period).

Fixes applied in the 2026 refactor:
- Camera is no longer opened at import time (this module used to grab
  /dev/video0 even when frames arrived over a ROS topic).
- Image width/height were swapped when building the camera matrix.
- Direction labels (Left/Right/Up/Down) are reported again instead of
  being discarded.
"""

import time

import cv2
import mediapipe as mp
import numpy as np

from .config import PoseConfig
from .logic import classify_direction

# Landmark indices used for pose: nose tip, eye corners, mouth corners, chin.
POSE_LANDMARKS = [1, 33, 61, 199, 263, 291]

__all__ = ["PoseConfig", "HeadPoseEstimator", "POSE_LANDMARKS"]


class HeadPoseEstimator:
    """Stateful head-pose attention classifier."""

    def __init__(self, config: PoseConfig | None = None):
        # One config per estimator; see the note in ear.py.
        self.config = config if config is not None else PoseConfig()
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            max_num_faces=1,
            static_image_mode=False,
        )
        self._away_counter = 0
        self._down_counter = 0

    def process(self, frame_bgr, draw: bool = True):
        """Run one frame. Returns (frame, state, direction).

        state: "No Driver" | "FOCUSED" | "NOT FOCUSED" | "Suspected Drowsiness"
        direction: "Forward" | "Looking Left/Right/Up/Down" | None
        """
        start = time.time()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return frame_bgr, "No Driver", None

        pitch_deg, yaw_deg = self._angles(frame_bgr, results)
        direction = classify_direction(pitch_deg, yaw_deg, self.config)
        state = self._state(direction)

        if draw:
            cv2.putText(frame_bgr, f"{state} ({direction})", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            fps = 1.0 / max(time.time() - start, 1e-6)
            cv2.putText(frame_bgr, f"FPS: {int(fps)}", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame_bgr, state, direction

    def _angles(self, frame, results):
        """Solve PnP for the six landmark correspondences -> (pitch, yaw) in degrees."""
        h, w = frame.shape[:2]
        landmarks = results.multi_face_landmarks[0].landmark

        face_3d = np.array([(landmarks[i].x * w, landmarks[i].y * h, landmarks[i].z)
                            for i in POSE_LANDMARKS])
        face_2d = np.array([(landmarks[i].x * w, landmarks[i].y * h)
                            for i in POSE_LANDMARKS])

        focal_length = w  # common pinhole approximation: focal ~= image width
        cam_matrix = np.array([[focal_length, 0, w / 2],
                               [0, focal_length, h / 2],
                               [0, 0, 1]], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        ok, rot_vec, _ = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_coeffs)
        if not ok:
            return 0.0, 0.0

        rmat, _ = cv2.Rodrigues(rot_vec)
        angles, *_ = cv2.RQDecomp3x3(rmat)
        # RQDecomp3x3 returns normalized angles; scale as in the original thesis code.
        pitch_deg = angles[0] * 360
        yaw_deg = angles[1] * -360
        return pitch_deg, yaw_deg

    def _state(self, direction: str) -> str:
        if direction == "Forward":
            self._away_counter = 0
            self._down_counter = 0
            return "FOCUSED"

        self._away_counter += 1
        if direction == "Looking Down":
            self._down_counter += 1
        else:
            self._down_counter = 0

        if self._down_counter >= self.config.DROWSY_FRAMES:
            return "Suspected Drowsiness"
        if self._away_counter >= self.config.AWAY_FRAMES:
            return "NOT FOCUSED"
        return "FOCUSED"
