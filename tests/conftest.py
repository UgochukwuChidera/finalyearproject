"""Shared fixtures for the DAPE test suite."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest


@pytest.fixture
def synthetic_image() -> np.ndarray:
    """Create a small synthetic RGB test image (200x200) with some structure."""
    img = np.ones((200, 200, 3), dtype=np.uint8) * 240  # light gray
    # Draw a dark rectangle
    img[50:150, 50:150] = (60, 60, 60)
    # Add some text-like lines
    cv2.rectangle(img, (30, 30), (170, 50), (0, 0, 0), 2)
    cv2.line(img, (20, 100), (180, 100), (0, 0, 0), 1)
    cv2.line(img, (20, 120), (150, 120), (0, 0, 0), 1)
    return img


@pytest.fixture
def synthetic_grayscale_image() -> np.ndarray:
    """Create a small synthetic grayscale image (200x200)."""
    img = np.ones((200, 200), dtype=np.uint8) * 240
    img[50:150, 50:150] = 60
    cv2.rectangle(img, (30, 30), (170, 50), 0, 2)
    cv2.line(img, (20, 100), (180, 100), 0, 1)
    return img


@pytest.fixture
def synthetic_skewed_image() -> np.ndarray:
    """Create a synthetic grayscale image with a ~5-degree skew."""
    img = np.ones((200, 200), dtype=np.uint8) * 240
    img[50:150, 50:150] = 60
    cv2.line(img, (20, 100), (180, 100), 0, 2)

    h, w = img.shape
    center = (w / 2.0, h / 2.0)
    rot = cv2.getRotationMatrix2D(center, 5.0, 1.0)
    skewed = cv2.warpAffine(
        img, rot, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=240,
    )
    return skewed


@pytest.fixture
def blank_image() -> np.ndarray:
    """Create a completely blank (white) image."""
    return np.ones((100, 100), dtype=np.uint8) * 255


@pytest.fixture
def mock_config() -> dict[str, Any]:
    """Return a minimal valid config dict for testing."""
    return {
        "form_type": "test_form",
        "fields": [
            {
                "name": "full_name",
                "expected_type": "string",
                "critical": True,
                "bounding_box": {"x": 10, "y": 10, "w": 200, "h": 30},
                "dictionary": None,
                "validation": {"min_length": 2},
            },
            {
                "name": "signature",
                "expected_type": "checkbox",
                "critical": False,
                "bounding_box": {"x": 300, "y": 400, "w": 50, "h": 20},
                "validation": {},
            },
        ],
        "template_path": "",
        "confidence_weights": {"w_lp": 0.6, "w_dict": 0.4},
        "thresholds": {"auto_accept": 0.85, "review": 0.70},
        "preprocessing": {"deskew": False},
        "editor_canvas": {"width": 800, "height": 1100},
    }


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for test outputs."""
    d = tmp_path / "test_outputs"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def mock_config_path(tmp_path: Path) -> str:
    """Create a real config JSON file on disk and return its path."""
    config = {
        "form_type": "test_form",
        "fields": [
            {
                "name": "full_name",
                "expected_type": "string",
                "critical": True,
                "bounding_box": {"x": 10, "y": 10, "w": 200, "h": 30},
                "dictionary": None,
                "validation": {"min_length": 2},
            },
        ],
        "template_path": "",
        "confidence_weights": {"w_lp": 0.6, "w_dict": 0.4},
        "thresholds": {"auto_accept": 0.85, "review": 0.70},
        "preprocessing": {"deskew": False},
        "editor_canvas": {"width": 800, "height": 1100},
    }
    p = tmp_path / "test_config.json"
    import json
    with open(str(p), "w") as f:
        json.dump(config, f)
    return str(p)
