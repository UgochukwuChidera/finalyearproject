from __future__ import annotations

import numpy as np
import logging

logger = logging.getLogger(__name__)


def border_removal(
    binary: np.ndarray, threshold_stability: float,
) -> tuple[np.ndarray | None, dict[str, float]]:
    _empty_meta: dict[str, float] = {"border_thickness": 0.0, "cropping_confidence": 0.0}
    if binary is None:
        logger.warning("border_removal: binary is None")
        return None, dict(_empty_meta)
    if not isinstance(binary, np.ndarray):
        logger.warning("border_removal: expected ndarray, got %s", type(binary).__name__)
        return binary, dict(_empty_meta)
    if binary.size == 0:
        logger.debug("border_removal: empty image — no-op")
        return binary, dict(_empty_meta)
    if binary.ndim != 2:
        logger.warning("border_removal: expected 2D, got %dD", binary.ndim)
        return binary, dict(_empty_meta)
    try:
        h, w = binary.shape
        rs = np.sum(binary > 0, axis=1)
        cs = np.sum(binary > 0, axis=0)
        rows = np.where(rs > 0.01 * w)[0]
        cols = np.where(cs > 0.01 * h)[0]
        if len(rows) == 0 or len(cols) == 0:
            logger.debug("border_removal: no content rows/cols — no-op")
            return binary, dict(_empty_meta)
        top, bottom = rows[0], rows[-1]
        left, right = cols[0], cols[-1]
        cropped = binary[top:bottom, left:right]
        removed = (top + (h - bottom)) * w + (left + (w - right)) * h
        thick = float(removed / (h * w))
        ec = 1 - np.mean([
            np.mean(binary[top:top+10, :] > 0),
            np.mean(binary[bottom-10:bottom, :] > 0),
            np.mean(binary[:, left:left+10] > 0),
            np.mean(binary[:, right-10:right] > 0),
        ])
        conf = float(np.clip(0.6 * ec + 0.4 * threshold_stability, 0, 1))
        return cropped, {"border_thickness": thick, "cropping_confidence": conf}
    except Exception as e:
        logger.exception("border_removal: processing failed: %s", e)
        return binary, dict(_empty_meta)
