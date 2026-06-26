from __future__ import annotations

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


def structure_prep(
    binary: np.ndarray, template_size: tuple[int, int],
) -> dict[str, float]:
    _defaults: dict[str, float] = {"feature_count": 0.0, "structural_confidence": 0.0}
    if binary is None:
        logger.warning("structure_prep: binary is None")
        return dict(_defaults)
    if not isinstance(binary, np.ndarray):
        logger.warning("structure_prep: expected ndarray, got %s", type(binary).__name__)
        return dict(_defaults)
    if binary.size == 0:
        logger.warning("structure_prep: empty image")
        return dict(_defaults)
    if binary.ndim != 2:
        logger.warning("structure_prep: expected 2D, got %dD", binary.ndim)
        return dict(_defaults)
    try:
        ch, cw = binary.shape
        if template_size is None or len(template_size) != 2:
            logger.warning("structure_prep: invalid template_size %s, using image dims", template_size)
            tw, th = cw, ch
        else:
            tw, th = template_size
        sc = float(np.clip(1 - abs(tw / cw - th / ch) / 0.1, 0, 1))
        orb = cv2.ORB_create(500)
        kp = orb.detect(binary, None)
        fc = len(kp)
        sconf = float(np.clip(0.4 * sc + 0.6 * min(fc / 500, 1), 0, 1))
        return {"feature_count": float(fc), "structural_confidence": sconf}
    except Exception as e:
        logger.exception("structure_prep: processing failed: %s", e)
        return dict(_defaults)
