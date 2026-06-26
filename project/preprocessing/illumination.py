from __future__ import annotations

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


def illumination_normalization(
    gray: np.ndarray, grayscale_std: float,
    blur_size: int = 51, clahe_tile: int = 8,
) -> tuple[np.ndarray | None, dict[str, float]]:
    _empty_meta: dict[str, float] = {"illumination_uniformity": 0.0, "local_contrast_gain": 0.0}
    if gray is None:
        logger.warning("illumination_normalization: gray is None")
        return None, dict(_empty_meta)
    if not isinstance(gray, np.ndarray):
        logger.warning("illumination_normalization: expected ndarray, got %s", type(gray).__name__)
        return gray, dict(_empty_meta)
    if gray.size == 0:
        logger.warning("illumination_normalization: empty image")
        return gray, dict(_empty_meta)
    try:
        bg = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    except Exception as e:
        logger.exception("illumination_normalization: GaussianBlur failed: %s", e)
        return gray, dict(_empty_meta)
    try:
        corrected = cv2.divide(gray, bg + 1, scale=255)
    except Exception as e:
        logger.exception("illumination_normalization: divide failed: %s", e)
        return gray, dict(_empty_meta)
    try:
        clahe = cv2.createCLAHE(2.0, (clahe_tile, clahe_tile))
        normalized = clahe.apply(corrected)
    except Exception as e:
        logger.exception("illumination_normalization: CLAHE failed: %s", e)
        return corrected, dict(_empty_meta)
    try:
        illum_std = float(np.std(bg))
        uniformity = float(np.clip(1 - illum_std / 60, 0, 1))
        gain = float(np.clip(float(np.std(normalized)) / (float(grayscale_std) + 1e-5) / 1.5, 0, 1))
        return normalized, {"illumination_uniformity": uniformity, "local_contrast_gain": gain}
    except Exception as e:
        logger.exception("illumination_normalization: metrics computation failed: %s", e)
        return normalized, dict(_empty_meta)
