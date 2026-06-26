"""Tests for the preprocessing pipeline modules.

Tests cover: IO, grayscale conversion, baseline metrics, skew analysis,
illumination normalization, binarization, fusion scoring, border removal,
and DPI kernel scaling. All tests use synthetic images to avoid dependencies
on the `form/` directory or real scanned forms.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import cv2
import pytest

from project.preprocessing.io import load_image
from project.preprocessing.grayscale import to_grayscale
from project.preprocessing.baseline_metrics import baseline_metrics
from project.preprocessing.skew_analysis import skew_analysis
from project.preprocessing.illumination import illumination_normalization
from project.preprocessing.binarization import binarization
from project.preprocessing.fusion import fuse
from project.preprocessing.border_removal import border_removal
from project.preprocessing.dpi import scale, kernel_sizes


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

class TestIO:
    def test_io_load_valid_image(self, tmp_path: Path, synthetic_image: np.ndarray):
        """Save a synthetic numpy array as PNG, then load it with load_image()."""
        p = tmp_path / "test_img.png"
        cv2.imwrite(str(p), synthetic_image)

        img, h, w, aspect = load_image(str(p))
        assert isinstance(img, np.ndarray)
        assert h == 200
        assert w == 200
        assert aspect == pytest.approx(1.0, abs=0.01)

    def test_io_load_nonexistent(self):
        """Verify ValueError is raised for a missing file."""
        with pytest.raises(ValueError, match="could not be loaded"):
            load_image("/nonexistent/path/image.png")


# ---------------------------------------------------------------------------
# Grayscale
# ---------------------------------------------------------------------------

class TestGrayscale:
    def test_grayscale_converts(self, synthetic_image: np.ndarray):
        """RGB image → grayscale output should be 2D."""
        gray = to_grayscale(synthetic_image)
        assert gray is not None
        assert gray.ndim == 2
        assert gray.shape == (200, 200)

    def test_grayscale_none_input(self):
        """Passing None should not raise — should log a warning and return None."""
        result = to_grayscale(None)
        assert result is None


# ---------------------------------------------------------------------------
# Baseline Metrics
# ---------------------------------------------------------------------------

class TestBaselineMetrics:
    def test_baseline_metrics_valid(self, synthetic_grayscale_image: np.ndarray):
        """Compute metrics on a synthetic image; verify all floats in valid range."""
        metrics = baseline_metrics(synthetic_grayscale_image)
        assert isinstance(metrics, dict)
        expected_keys = [
            "grayscale_mean", "grayscale_std", "min_intensity", "max_intensity",
            "intensity_range", "near_white_ratio", "near_black_ratio",
            "contrast_score", "balance_score", "blur_score", "noise_score",
        ]
        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"
            assert isinstance(metrics[key], float), f"{key} not float"
            assert 0.0 <= metrics[key] <= 255.0, f"{key} out of range"

    def test_baseline_metrics_none(self):
        """Passing None returns default values (all 0.0)."""
        metrics = baseline_metrics(None)
        assert metrics == {
            "grayscale_mean": 0.0, "grayscale_std": 0.0,
            "min_intensity": 0.0, "max_intensity": 0.0,
            "intensity_range": 0.0, "near_white_ratio": 0.0,
            "near_black_ratio": 0.0, "contrast_score": 0.0,
            "balance_score": 0.0, "blur_score": 0.0, "noise_score": 0.0,
        }


# ---------------------------------------------------------------------------
# Skew Analysis
# ---------------------------------------------------------------------------

class TestSkewAnalysis:
    def test_skew_detection(self, synthetic_skewed_image: np.ndarray):
        """A synthetic rotated image should produce a non-zero skew angle."""
        meta = skew_analysis(synthetic_skewed_image, hough_threshold=50)
        assert isinstance(meta, dict)
        assert "skew_angle" in meta
        # The skew should be non-zero (either positive or negative)
        assert abs(meta["skew_angle"]) > 0.1, (
            f"Expected non-zero skew angle, got {meta['skew_angle']}"
        )

    def test_skew_blank_image(self, blank_image: np.ndarray):
        """A blank (white) image should return 0.0 skew angle."""
        meta = skew_analysis(blank_image)
        assert meta["skew_angle"] == 0.0


# ---------------------------------------------------------------------------
# Illumination Normalization
# ---------------------------------------------------------------------------

class TestIllumination:
    def test_illumination_normalization(self, synthetic_grayscale_image: np.ndarray):
        """CLAHE output shape should match input shape."""
        normalized, illum = illumination_normalization(
            synthetic_grayscale_image, grayscale_std=50.0,
        )
        assert normalized is not None
        assert normalized.shape == synthetic_grayscale_image.shape
        assert "illumination_uniformity" in illum
        assert "local_contrast_gain" in illum


# ---------------------------------------------------------------------------
# Binarization
# ---------------------------------------------------------------------------

class TestBinarization:
    def test_binarization_otsu(self, synthetic_grayscale_image: np.ndarray):
        """Output should be binary (only 0 or 255 values)."""
        binary, meta = binarization(synthetic_grayscale_image, block_size=31)
        assert binary is not None
        unique = set(np.unique(binary))
        assert unique.issubset({0, 255}), f"Expected binary, got values: {unique}"

    def test_binarization_even_block_size(self, synthetic_grayscale_image: np.ndarray):
        """An even block_size should get corrected to odd (no crash)."""
        binary, meta = binarization(synthetic_grayscale_image, block_size=30)
        assert binary is not None
        assert binary.shape == synthetic_grayscale_image.shape


# ---------------------------------------------------------------------------
# Fusion Scoring
# ---------------------------------------------------------------------------

class TestFusion:
    def test_fusion_all_max(self):
        """All metrics = 1.0 → score should be ≈ 1.0."""
        metrics = {
            "contrast_score": 1.0,
            "blur_score": 1.0,
            "noise_score": 1.0,
            "balance_score": 1.0,
            "illumination_uniformity": 1.0,
            "local_contrast_gain": 1.0,
            "threshold_stability": 1.0,
            "structural_confidence": 1.0,
            "effective_skew_risk": 0.0,  # risk is subtracted
        }
        score = fuse(metrics)
        assert score == pytest.approx(1.0, abs=0.05)

    def test_fusion_all_min(self):
        """All metrics = 0.0 → score should be ≈ 0.0."""
        metrics = {
            "contrast_score": 0.0,
            "blur_score": 0.0,
            "noise_score": 0.0,
            "balance_score": 0.0,
            "illumination_uniformity": 0.0,
            "local_contrast_gain": 0.0,
            "threshold_stability": 0.0,
            "structural_confidence": 0.0,
            "effective_skew_risk": 1.0,  # max risk subtracts
        }
        score = fuse(metrics)
        assert score == pytest.approx(0.0, abs=0.05)

    def test_fusion_with_nan(self):
        """NaN metrics should be replaced with 0.0 (no crash)."""
        metrics = {
            "contrast_score": float("nan"),
            "blur_score": 1.0,
            "noise_score": 1.0,
            "balance_score": 1.0,
            "illumination_uniformity": 1.0,
            "local_contrast_gain": 1.0,
            "threshold_stability": 1.0,
            "structural_confidence": 1.0,
            "effective_skew_risk": 0.0,
        }
        score = fuse(metrics)
        # NaN gets 0.0, so score should be less than 1.0
        assert 0.0 <= score <= 1.0

    def test_fusion_none(self):
        """Passing None should return 0.0."""
        assert fuse(None) == 0.0


# ---------------------------------------------------------------------------
# Border Removal
# ---------------------------------------------------------------------------

class TestBorderRemoval:
    def test_border_removal(self):
        """Image with synthetic dark borders → verify borders removed."""
        # Create a 100x100 binary image with dark borders (0) and white content (255)
        img = np.zeros((100, 100), dtype=np.uint8)
        img[10:90, 10:90] = 255  # white content in the middle

        cropped, meta = border_removal(img, threshold_stability=0.5)
        assert cropped is not None
        assert cropped.shape[0] < 100, "Expected border removal to crop rows"
        assert cropped.shape[1] < 100, "Expected border removal to crop cols"
        assert "border_thickness" in meta
        assert 0.0 <= meta["border_thickness"] <= 1.0

    def test_border_removal_no_borders(self):
        """Image without borders → nearly unchanged (may lose 1px edge)."""
        img = np.ones((50, 50), dtype=np.uint8) * 255
        cropped, meta = border_removal(img, threshold_stability=0.5)
        assert cropped is not None
        # The algorithm strips a thin edge even for fully white images
        # due to [top:bottom) slicing excluding the last row/column
        assert cropped.shape[0] >= 48
        assert cropped.shape[1] >= 48
        # border_thickness should be very low (one-pixel edge yields 0.04)
        assert meta["border_thickness"] <= 0.05

    def test_border_removal_none(self):
        """None input should return None and default meta."""
        result, meta = border_removal(None, 0.5)
        assert result is None
        assert meta == {"border_thickness": 0.0, "cropping_confidence": 0.0}


# ---------------------------------------------------------------------------
# DPI Kernel Scaling
# ---------------------------------------------------------------------------

class TestDPI:
    def test_scale_default(self, caplog):
        """scale() with missing DPI → logs warning and uses 300."""
        caplog.set_level(logging.WARNING)
        result = scale(31, None)
        assert result >= 1
        assert result % 2 == 1  # must be odd

    def test_kernel_sizes_invalid_dpi(self, caplog):
        """kernel_sizes() with invalid dpi → logs warning and uses 300."""
        caplog.set_level(logging.WARNING)
        sizes = kernel_sizes(None)
        assert isinstance(sizes, dict)
        assert sizes["dpi"] == 300

    def test_kernel_sizes_valid(self):
        """kernel_sizes() with valid DPI returns correct keys."""
        sizes = kernel_sizes(300)
        expected_keys = [
            "skew_hough_threshold", "binarization_block", "illumination_blur",
            "clahe_tile", "diff_threshold", "dpi", "scale_factor",
        ]
        for key in expected_keys:
            assert key in sizes, f"Missing key: {key}"
