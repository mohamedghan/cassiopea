"""Hand tracking module for maintaining consistent hand tracking."""

import time
from collections.abc import Sequence

import numpy as np

from hand_control.config import config
from hand_control.filters import OneEuroFilter
from hand_control.types import FINGER_NAMES, FingerAngles, FingerName, LandmarkPoint, Vector2D


class HandTracker:
    """Tracks a single hand even when multiple hands are present.

    Maintains tracking of one hand across frames by tracking the hand's
    center position and finding the closest hand in subsequent frames.
    """

    def __init__(
        self,
        tracking_threshold: float = 0.3,
        max_frames_without_hand: int = 30,
    ) -> None:
        """Initialize the hand tracker.

        Args:
            tracking_threshold: Max normalized distance to consider same hand.
            max_frames_without_hand: Frames before resetting tracking.
        """
        self._tracked_hand_center: Vector2D | None = None
        self._tracking_threshold = tracking_threshold
        self._max_frames_without_hand = max_frames_without_hand
        self._frames_without_hand = 0

        # Create filters for each finger
        self._filters: dict[FingerName, OneEuroFilter] = {
            finger: OneEuroFilter(
                min_cutoff=config.filter_min_cutoff,
                beta=config.filter_beta,
            )
            for finger in FINGER_NAMES
        }

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
            self._frames_without_hand += 1
            if self._frames_without_hand > self._max_frames_without_hand:
                self.reset_tracking()
            return None

        self._frames_without_hand = 0

        # If no previous tracking, pick first hand
        if self._tracked_hand_center is None:
            self._tracked_hand_center = self.get_hand_center(hand_landmarks_list[0])
            return hand_landmarks_list[0]

        # Find closest hand to tracked position
        min_dist = float("inf")
        closest_hand: Sequence[LandmarkPoint] | None = None
        closest_center: Vector2D | None = None

        for hand_landmarks in hand_landmarks_list:
            center = self.get_hand_center(hand_landmarks)
            dist = float(np.linalg.norm(center - self._tracked_hand_center))

            if dist < min_dist:
                min_dist = dist
                closest_hand = hand_landmarks
                closest_center = center

        # Update tracked position with smoothing
        if closest_center is not None and min_dist < self._tracking_threshold:
            self._tracked_hand_center = 0.7 * self._tracked_hand_center + 0.3 * closest_center
            return closest_hand
        elif closest_hand is not None:
            # Hand jumped too far, but still track it (might be fast movement)
            self._tracked_hand_center = closest_center
            return closest_hand

        return None

    def filter_angles(self, angles: FingerAngles) -> FingerAngles:
        """Apply One Euro Filter to all finger angles.

        Args:
            angles: Raw finger angles from detection.

        Returns:
            Filtered finger angles with reduced jitter.
        """
        t = time.time()
        return FingerAngles(
            thumb=int(self._filters["thumb"].update(angles["thumb"], t)),
            index=int(self._filters["index"].update(angles["index"], t)),
            middle=int(self._filters["middle"].update(angles["middle"], t)),
            ring=int(self._filters["ring"].update(angles["ring"], t)),
            pinky=int(self._filters["pinky"].update(angles["pinky"], t)),
        )

    def reset_tracking(self) -> None:
        """Reset tracking state."""
        self._tracked_hand_center = None
        for f in self._filters.values():
            f.reset()
