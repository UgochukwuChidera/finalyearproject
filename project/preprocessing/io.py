from __future__ import annotations

import cv2
import logging

logger = logging.getLogger(__name__)


import numpy as np


def load_image(image_path: str) -> tuple[np.ndarray, int, int, float]:
    try:
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Image could not be loaded: {image_path}")
        h, w = image.shape[:2]
        return image, h, w, w / h
    except FileNotFoundError:
        logger.error("load_image: file not found: %s", image_path)
        raise
    except ValueError:
        logger.error("load_image: could not decode image: %s", image_path)
        raise
    except Exception as e:
        logger.exception("load_image: unexpected error loading %s: %s", image_path, e)
        raise ValueError(f"Failed to load image: {image_path}") from e
