"""Tests for signal filtering modules."""

import time

from hand_control.filters import LowPassFilter, OneEuroFilter


class TestLowPassFilter:
    """Tests for LowPassFilter."""

    def test_initial_value(self):
        """First update should return the input value."""
        f = LowPassFilter(alpha=0.5)
        assert f.update(100.0) == 100.0

    def test_smoothing(self):
        """Filter should smooth values over time."""
        f = LowPassFilter(alpha=0.5)
        f.update(0.0)
        result = f.update(100.0)
        # With alpha=0.5: 0.5 * 100 + 0.5 * 0 = 50
        assert result == 50.0

    def test_low_alpha_more_smoothing(self):
        """Lower alpha should result in more smoothing (slower response)."""
        f = LowPassFilter(alpha=0.1)
        f.update(0.0)
        result = f.update(100.0)
        # With alpha=0.1: 0.1 * 100 + 0.9 * 0 = 10
        assert result == 10.0

    def test_high_alpha_less_smoothing(self):
        """Higher alpha should result in less smoothing (faster response)."""
        f = LowPassFilter(alpha=0.9)
        f.update(0.0)
        result = f.update(100.0)
        # With alpha=0.9: 0.9 * 100 + 0.1 * 0 = 90
        assert result == 90.0

    def test_reset(self):
        """Reset should clear the filter state."""
        f = LowPassFilter(alpha=0.5)
        f.update(100.0)
        f.reset()
        assert f.value is None
        # After reset, should behave like initial call
        assert f.update(50.0) == 50.0


class TestOneEuroFilter:
    """Tests for OneEuroFilter."""

    def test_initial_value(self):
        """First update should return the input value."""
        f = OneEuroFilter()
        assert f.update(100.0) == 100.0

    def test_consistent_output(self):
        """Filter should produce consistent output for constant input."""
        f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
        t = time.time()

        # Feed same value multiple times
        results = []
        for i in range(10):
            result = f.update(50.0, t + i * 0.016)
            results.append(result)

        # All outputs should be close to 50 after initial stabilization
        for result in results[2:]:  # Skip first couple while filter stabilizes
            assert abs(result - 50.0) < 1.0, f"Expected ~50, got {result}"

    def test_responsive_to_fast_movement(self):
        """Filter should be more responsive during fast movements."""
        f = OneEuroFilter(min_cutoff=0.1, beta=1.0)  # High beta for responsiveness
        t = time.time()
        f.update(0.0, t)
        # Large change in short time
        result = f.update(100.0, t + 0.01)
        # Should track faster due to high beta
        assert result > 10.0  # Should be reasonably responsive

    def test_reset(self):
        """Reset should clear the filter state."""
        f = OneEuroFilter()
        f.update(100.0)
        f.reset()
        # After reset, should behave like initial call
        assert f.update(50.0) == 50.0

    def test_convergence(self):
        """Filter should converge to constant input over time."""
        f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
        t = time.time()
        # Feed constant value multiple times
        for i in range(100):
            result = f.update(100.0, t + i * 0.1)
        # Should converge close to 100
        assert abs(result - 100.0) < 1.0
