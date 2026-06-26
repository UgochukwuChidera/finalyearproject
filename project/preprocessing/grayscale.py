from __future__ import annotations

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


def to_grayscale(image: np.ndarray) -> np.ndarray | None:
    if image is None:
        logger.warning("to_grayscale: image is None")
        return None
    if not isinstance(image, np.ndarray):
        logger.warning("to_grayscale: expected ndarray, got %s", type(image).__name__)
        return image
    if image.ndim not in (2, 3):
        logger.warning("to_grayscale: unexpected dimensions %d", image.ndim)
        return image
    if image.size == 0:
        logger.warning("to_grayscale: empty image")
        return image
    try:
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image.copy()
    except Exception as e:
        logger.exception("to_grayscale: conversion failed: %s", e)
        return image
