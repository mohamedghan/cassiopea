"""One Euro Filter for high-quality signal smoothing."""

import time

import numpy as np


class OneEuroFilter:
    """One Euro Filter for high-quality smoothing.

    Reduces jitter while maintaining responsiveness by adapting the
    cutoff frequency based on the signal's rate of change.

    Reference: https://cristal.univ-lille.fr/~casiez/1euro/
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ) -> None:
        """Initialize the One Euro Filter.

        Args:
            min_cutoff: Minimum cutoff frequency. Lower = more smoothing at rest.
            beta: Speed coefficient. Higher = less lag during fast movements.
            d_cutoff: Cutoff frequency for derivative computation.
        """
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev: float | None = None
        self._dx_prev: float = 0.0
        self._t_prev: float | None = None

    def _smoothing_factor(self, t_e: float, cutoff: float) -> float:
        """Compute the smoothing factor alpha.

        Args:
            t_e: Time elapsed since last update.
            cutoff: Cutoff frequency.

        Returns:
            Smoothing factor between 0 and 1.
        """
        r = 2 * np.pi * cutoff * t_e
        return float(r / (r + 1))

    def _exponential_smoothing(self, a: float, x: float, x_prev: float) -> float:
        """Apply exponential smoothing.

        Args:
            a: Smoothing factor.
            x: Current value.
            x_prev: Previous smoothed value.

        Returns:
            Smoothed value.
        """
        return a * x + (1 - a) * x_prev

    def update(self, x: float, t: float | None = None) -> float:
        """Update the filter with a new value.

        Args:
            x: The new input value to filter.
            t: Optional timestamp. If not provided, uses current time.

        Returns:
            The filtered output value.
        """
        if t is None:
            t = time.time()

        if self._t_prev is None:
            self._x_prev = x
            self._t_prev = t
            return x

        t_e = t - self._t_prev
        if t_e <= 0:
            t_e = 1 / 60  # Default to 60fps

        # Compute derivative
        a_d = self._smoothing_factor(t_e, self.d_cutoff)
        dx = (x - (self._x_prev or x)) / t_e
        dx_hat = self._exponential_smoothing(a_d, dx, self._dx_prev)

        # Compute adaptive cutoff
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)

        # Compute filtered value
        a = self._smoothing_factor(t_e, cutoff)
        x_hat = self._exponential_smoothing(a, x, self._x_prev or x)

        # Update state
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = t

        return x_hat

    def reset(self) -> None:
        """Reset the filter state."""
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None
