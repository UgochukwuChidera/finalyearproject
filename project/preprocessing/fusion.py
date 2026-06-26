from __future__ import annotations

import numpy as np
import logging
import math

logger = logging.getLogger(__name__)

_REQUIRED_KEYS: list[str] = [
    "contrast_score", "blur_score", "noise_score", "balance_score",
    "illumination_uniformity", "local_contrast_gain",
    "threshold_stability", "structural_confidence", "effective_skew_risk",
]


def fuse(metrics: dict[str, float]) -> float:
    if metrics is None:
        logger.error("fuse: metrics is None, returning 0.0")
        return 0.0
    if not isinstance(metrics, dict):
        logger.error("fuse: expected dict, got %s, returning 0.0", type(metrics).__name__)
        return 0.0
    sanitized: dict[str, float] = {}
    for key in _REQUIRED_KEYS:
        val = metrics.get(key)
        if val is None:
            logger.warning("fuse: key '%s' is None, substituting 0.0", key)
            sanitized[key] = 0.0
        elif isinstance(val, (float, int)):
            if math.isnan(val) or math.isinf(val):
                logger.warning("fuse: key '%s' is %s, substituting 0.0", key, val)
                sanitized[key] = 0.0
            else:
                sanitized[key] = float(val)
        else:
            logger.warning("fuse: key '%s' has type %s, substituting 0.0", key, type(val).__name__)
            sanitized[key] = 0.0
    try:
        score = (
            0.22 * sanitized["contrast_score"]
            + 0.18 * sanitized["blur_score"]
            + 0.14 * sanitized["noise_score"]
            + 0.14 * sanitized["balance_score"]
            + 0.10 * sanitized["illumination_uniformity"]
            + 0.08 * sanitized["local_contrast_gain"]
            + 0.07 * sanitized["threshold_stability"]
            + 0.07 * sanitized["structural_confidence"]
            - 0.07 * sanitized["effective_skew_risk"]
        )
        return float(np.clip(score, 0, 1))
    except Exception as e:
        logger.exception("fuse: computation failed: %s", e)
        return 0.0
