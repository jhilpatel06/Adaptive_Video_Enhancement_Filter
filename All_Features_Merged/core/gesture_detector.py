import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import math


class GestureDetector:
    """
    Detects a hand via MediaPipe Hand Landmarker and provides:
      1. Temperature value (-100 to +100) from thumb-index pinch distance.
    2. Discrete gesture detection (fist/open) for filter switching.

    Usage:
        detector = GestureDetector()
        temperature, action, annotated_frame = detector.process(frame)
        # action is one of: "left" (fist), "right" (open), or None
    """

    # Distance range (in pixels) that maps to the temperature scale
    MIN_DIST = 20
    MAX_DIST = 200

    # Pose detection thresholds
    POSE_HOLD_TIME = 0.25   # Seconds a pose must be stable before triggering
    POSE_COOLDOWN = 0.8     # Seconds to wait before allowing another trigger
    FINGER_EXTEND_RATIO = 1.2

    def __init__(self, model_path="hand_landmarker.task"):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        # Pose tracking state
        self._current_pose = None
        self._pose_start_time = 0.0
        self._last_emit_time = 0.0
        self._last_emitted_pose = None

    def _finger_extended(self, hand_landmarks, tip_idx, mcp_idx, w, h):
        tip = hand_landmarks[tip_idx]
        mcp = hand_landmarks[mcp_idx]
        wrist = hand_landmarks[0]

        tip_pt = (int(tip.x * w), int(tip.y * h))
        mcp_pt = (int(mcp.x * w), int(mcp.y * h))
        wrist_pt = (int(wrist.x * w), int(wrist.y * h))

        tip_dist = math.hypot(tip_pt[0] - wrist_pt[0], tip_pt[1] - wrist_pt[1])
        mcp_dist = math.hypot(mcp_pt[0] - wrist_pt[0], mcp_pt[1] - wrist_pt[1])
        ratio = tip_dist / max(1.0, mcp_dist)

        return ratio > self.FINGER_EXTEND_RATIO, tip_pt, mcp_pt, ratio

    def _detect_pose(self, hand_landmarks, w, h):
        extended = 0
        ratios = []
        points = []
        for tip_idx, mcp_idx in ((8, 5), (12, 9), (16, 13), (20, 17)):
            is_extended, tip_pt, mcp_pt, ratio = self._finger_extended(
                hand_landmarks, tip_idx, mcp_idx, w, h
            )
            if is_extended:
                extended += 1
            ratios.append(ratio)
            points.append((tip_pt, mcp_pt))

        if extended >= 3:
            return "open", ratios, points
        if extended <= 1:
            return "fist", ratios, points
        return None, ratios, points

    def _map_pose_to_action(self, pose):
        if pose == "open":
            return "right"
        if pose == "fist":
            return "left"
        return None

    def process(self, frame):
        """
        Detect a hand and compute gesture temperature + pose action.

        Args:
            frame: BGR image (OpenCV format).

        Returns:
            temperature: float in [-100, 100]. 0 if no hand is detected.
            swipe:       "left" (fist), "right" (open), or None.
            annotated:   a copy of the frame with tracking overlay drawn on it.
        """
        annotated = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self.detector.detect(mp_image)

        temperature = 0.0
        swipe = None

        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                h, w = frame.shape[:2]

                # --- Temperature (thumb-index pinch) ---
                thumb_tip = hand_landmarks[4]
                index_tip = hand_landmarks[8]

                x1, y1 = int(thumb_tip.x * w), int(thumb_tip.y * h)
                x2, y2 = int(index_tip.x * w), int(index_tip.y * h)

                # Draw tracking overlay
                cv2.circle(annotated, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
                cv2.circle(annotated, (x2, y2), 10, (255, 0, 0), cv2.FILLED)
                cv2.line(annotated, (x1, y1), (x2, y2), (255, 0, 255), 3)

                dist = math.hypot(x2 - x1, y2 - y1)
                dist_clamped = max(self.MIN_DIST, min(self.MAX_DIST, dist))
                temperature = float(np.interp(dist_clamped, [self.MIN_DIST, self.MAX_DIST], [-100, 100]))

                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.putText(annotated, f"Dist: {int(dist)}", (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # --- Pose detection (fist/open) ---
                now = time.time()
                pose, ratios, points = self._detect_pose(hand_landmarks, w, h)

                if pose != self._current_pose:
                    self._current_pose = pose
                    self._pose_start_time = now

                if pose is None:
                    self._last_emitted_pose = None
                elif (
                    pose == self._current_pose
                    and now - self._pose_start_time >= self.POSE_HOLD_TIME
                    and now - self._last_emit_time >= self.POSE_COOLDOWN
                    and pose != self._last_emitted_pose
                ):
                    self._last_emit_time = now
                    self._last_emitted_pose = pose
                    swipe = self._map_pose_to_action(pose)
                    if swipe:
                        print(f"  >> POSE {pose.upper()} detected! -> {swipe.upper()}")

                # Draw finger points and ratios used for fist/open detection
                labels = ["Idx", "Mid", "Ring", "Pky"]
                for (tip_pt, mcp_pt), ratio, label in zip(points, ratios, labels):
                    color = (0, 255, 0) if ratio >= self.FINGER_EXTEND_RATIO else (0, 0, 255)
                    cv2.circle(annotated, mcp_pt, 6, color, cv2.FILLED)
                    cv2.circle(annotated, tip_pt, 6, color, cv2.FILLED)
                    cv2.line(annotated, mcp_pt, tip_pt, color, 2)
                    cv2.putText(
                        annotated,
                        f"{label}:{ratio:.2f}",
                        (mcp_pt[0] + 8, mcp_pt[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                    )

        else:
            self._current_pose = None
            self._last_emitted_pose = None

        return temperature, swipe, annotated

    def close(self):
        """Free resources used by the hand detection model."""
        self.detector.close()
