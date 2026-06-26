from __future__ import annotations

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


def binarization(
    normalized: np.ndarray, block_size: int = 31,
) -> tuple[np.ndarray | None, dict[str, float]]:
    _empty_meta: dict[str, float] = {
        "foreground_ratio": 0.0, "pixel_entropy": 0.0, "threshold_stability": 0.0,
    }
    if normalized is None:
        logger.warning("binarization: normalized is None")
        return None, dict(_empty_meta)
    if not isinstance(normalized, np.ndarray):
        logger.warning("binarization: expected ndarray, got %s", type(normalized).__name__)
        return normalized, dict(_empty_meta)
    if normalized.size == 0:
        logger.warning("binarization: empty image")
        return normalized, dict(_empty_meta)
    if normalized.ndim != 2:
        logger.warning("binarization: expected 2D, got %dD", normalized.ndim)
        return normalized, dict(_empty_meta)
    # Ensure block_size is odd and at least 3
    if not isinstance(block_size, int) or block_size % 2 == 0:
        old = block_size
        block_size = (block_size + 1) if isinstance(block_size, int) else 31
        if block_size % 2 == 0:
            block_size += 1
        logger.warning("binarization: block_size must be odd, adjusted %s -> %d", old, block_size)
    if block_size < 3:
        block_size = 3
        logger.warning("binarization: block_size too small, clamped to 3")
    try:
        binary = cv2.adaptiveThreshold(
            normalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block_size, 10,
        )
    except Exception as e:
        logger.exception("binarization: adaptiveThreshold failed: %s", e)
        return normalized, dict(_empty_meta)
    try:
        fg = float(np.mean(binary > 0))
        pfg, pbg = fg, 1 - fg
        ent = 0.0
        if pfg > 0:
            ent -= pfg * np.log2(pfg)
        if pbg > 0:
            ent -= pbg * np.log2(pbg)
        h, w = binary.shape
        reg = [np.mean(binary[i*h//4:(i+1)*h//4, j*w//4:(j+1)*w//4] > 0)
               for i in range(4) for j in range(4)]
        stab = float(np.clip(1 - np.var(reg) / 0.02, 0, 1))
        return binary, {"foreground_ratio": fg, "pixel_entropy": ent, "threshold_stability": stab}
    except Exception as e:
        logger.exception("binarization: metrics computation failed: %s", e)
        return binary, dict(_empty_meta)
