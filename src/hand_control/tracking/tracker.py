"""Hand tracking module for maintaining consistent hand tracking."""

from collections.abc import Sequence

import numpy as np

from hand_control.types import LandmarkPoint, Vector2D


class HandTracker:
    """Tracks a single hand even when multiple hands are present.

    With MediaPipe VIDEO mode, temporal landmark smoothing is handled
    internally. This class only handles multi-hand selection when
    more than one hand is visible.
    """

    def __init__(self, tracking_threshold: float = 0.3) -> None:
        """Initialize the hand tracker.

        Args:
            tracking_threshold: Max normalized distance to consider same hand.
        """
        self._tracked_hand_center: Vector2D | None = None
        self._tracking_threshold = tracking_threshold

    @property
    def tracked_hand_center(self) -> Vector2D | None:
        """Current tracked hand center position."""
        return self._tracked_hand_center

    def get_hand_center(self, landmarks: Sequence[LandmarkPoint]) -> Vector2D:
        """Calculate center of palm using wrist and middle finger base.

        Args:
            landmarks: List of hand landmarks.

        Returns:
            2D array with normalized x, y coordinates of palm center.
        """
        wrist = landmarks[0]
        middle_mcp = landmarks[9]
        return np.array([(wrist.x + middle_mcp.x) / 2, (wrist.y + middle_mcp.y) / 2])

    def find_tracked_hand(
        self,
        hand_landmarks_list: Sequence[Sequence[LandmarkPoint]],
    ) -> Sequence[LandmarkPoint] | None:
        """Find the hand closest to previously tracked position.

        Args:
            hand_landmarks_list: List of detected hands, each with landmarks.

        Returns:
            The landmarks of the tracked hand, or None if no hand found.
        """
        if not hand_landmarks_list:
            return None

        if self._tracked_hand_center is None:
            self._tracked_hand_center = self.get_hand_center(hand_landmarks_list[0])
            return hand_landmarks_list[0]

        closest_hand: Sequence[LandmarkPoint] | None = None
        closest_center: Vector2D | None = None
        min_dist = float("inf")

        for hand_landmarks in hand_landmarks_list:
            center = self.get_hand_center(hand_landmarks)
            dist = float(np.linalg.norm(center - self._tracked_hand_center))
            if dist < min_dist:
                min_dist = dist
                closest_hand = hand_landmarks
                closest_center = center

        if closest_hand is not None and closest_center is not None:
            self._tracked_hand_center = 0.7 * self._tracked_hand_center + 0.3 * closest_center
            return closest_hand

        return None

    def reset_tracking(self) -> None:
        """Reset tracking state."""
        self._tracked_hand_center = None
