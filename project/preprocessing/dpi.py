"""DPI-aware kernel scaling. All defaults calibrated for 300 DPI."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_BASE = 300


def scale(kernel_size: int, dpi: int) -> int:
    if dpi is None or dpi <= 0:
        logger.warning("scale: invalid dpi (%s), using default %d", dpi, _BASE)
        dpi = _BASE
    scaled = max(1, round(kernel_size * dpi / _BASE))
    return scaled if scaled % 2 == 1 else scaled + 1


def kernel_sizes(dpi: int) -> dict[str, int | float]:
    if dpi is None or dpi <= 0:
        logger.warning("kernel_sizes: invalid dpi (%s), using default %d", dpi, _BASE)
        dpi = _BASE
    scale_factor: float = dpi / _BASE
    return {
        "skew_hough_threshold":  round(200 * scale_factor),
        "binarization_block":    scale(31, dpi),
        "illumination_blur":     scale(51, dpi),
        "clahe_tile":            max(1, round(8 * scale_factor)),
        "diff_threshold":        30,
        "morph_open":            scale(3, dpi),
        "morph_close":           scale(5, dpi),
        "morph_dilate":          scale(7, dpi),
        "min_region_area":       round(50 * scale_factor**2),
        "max_features":          round(1000 * scale_factor),
        "checkbox_max_area":     round(3000 * scale_factor**2),
        "textbox_min_area":      round(3000 * scale_factor**2),
        "line_min_width":        round(60 * scale_factor),
        "dpi":                   dpi,
        "scale_factor":          round(scale_factor, 4),
    }
