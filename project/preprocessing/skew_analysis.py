from __future__ import annotations

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

_DEFAULTS: dict[str, float] = {
    "skew_angle": 0.0, "skew_confidence": 0.0,
    "skew_acceptability": 1.0, "skew_risk": 0.0,
    "effective_skew_risk": 0.0,
}


def skew_analysis(gray: np.ndarray, hough_threshold: int = 200) -> dict[str, float]:
    if gray is None:
        logger.warning("skew_analysis: gray is None, returning defaults")
        return dict(_DEFAULTS)
    if not isinstance(gray, np.ndarray):
        logger.warning("skew_analysis: expected ndarray, got %s", type(gray).__name__)
        return dict(_DEFAULTS)
    if gray.size == 0:
        logger.warning("skew_analysis: empty image, returning defaults")
        return dict(_DEFAULTS)
    if gray.ndim != 2:
        logger.warning("skew_analysis: expected 2D, got %dD, returning defaults", gray.ndim)
        return dict(_DEFAULTS)
    try:
        binary = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY_INV,15,10)
    except Exception as e:
        logger.exception("skew_analysis: adaptiveThreshold failed: %s", e)
        return dict(_DEFAULTS)
    try:
        edges = cv2.Canny(binary,50,150)
    except Exception as e:
        logger.exception("skew_analysis: Canny failed: %s", e)
        return dict(_DEFAULTS)
    try:
        lines = cv2.HoughLines(edges,1,np.pi/180,hough_threshold)
    except Exception as e:
        logger.exception("skew_analysis: HoughLines failed: %s", e)
        return dict(_DEFAULTS)
    skew_angle = 0.0
    skew_confidence = 0.0
    skew_acceptability = 1.0
    if lines is not None:
        try:
            angles = np.array([(t - np.pi/2) * 180/np.pi for r, t in lines[:30, 0]])
            med = np.median(angles)
            mean_angle = np.mean(angles)
            skew_angle = float(0.8 * med + 0.2 * mean_angle)
            cs = 1 - np.std(angles) / 2.0
            ca = 1 - abs(mean_angle - med) / 2.0
            skew_confidence = float(np.clip(0.7 * cs + 0.3 * ca, 0, 1))
            skew_acceptability = float(np.clip(1 - abs(skew_angle) / 3.0, 0, 1))
        except Exception as e:
            logger.warning("skew_analysis: angle computation failed: %s", e)
    risk = float(np.clip(abs(skew_angle) / 5.0, 0, 1))
    return {"skew_angle": skew_angle, "skew_confidence": skew_confidence,
            "skew_acceptability": skew_acceptability, "skew_risk": risk,
            "effective_skew_risk": float(skew_confidence * risk)}
