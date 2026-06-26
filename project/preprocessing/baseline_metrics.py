from __future__ import annotations

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

_DEFAULTS: dict[str, float] = {
    "grayscale_mean": 0.0, "grayscale_std": 0.0,
    "min_intensity": 0.0, "max_intensity": 0.0,
    "intensity_range": 0.0, "near_white_ratio": 0.0,
    "near_black_ratio": 0.0, "contrast_score": 0.0,
    "balance_score": 0.0, "blur_score": 0.0, "noise_score": 0.0,
}


def baseline_metrics(gray: np.ndarray) -> dict[str, float]:
    if gray is None:
        logger.warning("baseline_metrics: gray is None, returning defaults")
        return dict(_DEFAULTS)
    if not isinstance(gray, np.ndarray):
        logger.warning("baseline_metrics: expected ndarray, got %s", type(gray).__name__)
        return dict(_DEFAULTS)
    if gray.size == 0:
        logger.warning("baseline_metrics: empty image, returning defaults")
        return dict(_DEFAULTS)
    if gray.ndim != 2:
        logger.warning("baseline_metrics: expected 2D, got %dD, returning defaults", gray.ndim)
        return dict(_DEFAULTS)
    try:
        mean = float(np.mean(gray)); std = float(np.std(gray))
        mn, mx = int(np.min(gray)), int(np.max(gray))
        rng = float(mx - mn)
        nw = float(np.mean(gray > 240)); nb = float(np.mean(gray < 15))
        contrast = float(np.clip((rng - 80) / 120, 0, 1))
        ws = np.clip(1 - abs(nw - 0.6), 0, 1); bs = np.clip(1 - nb / 0.15, 0, 1)
        balance = float(0.6 * ws + 0.4 * bs)
        lap_img = cv2.Laplacian(gray, cv2.CV_32F)
        _, lap_std = cv2.meanStdDev(lap_img)
        lap = float(lap_std[0][0] ** 2)
        blur = float(np.clip((lap - 50) / 150, 0, 1))
        den = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        noise = float(np.clip(1 - np.mean(cv2.absdiff(gray, den)) / 40, 0, 1))
        return {"grayscale_mean":mean,"grayscale_std":std,"min_intensity":float(mn),
                "max_intensity":float(mx),"intensity_range":rng,"near_white_ratio":nw,
                "near_black_ratio":nb,"contrast_score":contrast,"balance_score":balance,
                "blur_score":blur,"noise_score":noise}
    except Exception as e:
        logger.exception("baseline_metrics: computation failed: %s", e)
        return dict(_DEFAULTS)
