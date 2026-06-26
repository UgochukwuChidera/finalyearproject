from __future__ import annotations

import numpy as np


class CheckboxExtractor:
    def __init__(self, marked_threshold: float = 0.15, unmarked_threshold: float = 0.04) -> None:
        self.marked_threshold   = marked_threshold
        self.unmarked_threshold = unmarked_threshold

    def extract(self, field_image: np.ndarray, diff_roi: np.ndarray | None = None) -> dict[str, bool | float]:
        _E: dict[str, bool | float] = {"value": False, "confidence": 0.0, "pixel_density": 0.0}
        region = diff_roi if diff_roi is not None else field_image
        if region is None or region.size == 0:
            return _E
        density = float(np.count_nonzero(region) / region.size)
        mid = (self.marked_threshold + self.unmarked_threshold) / 2.0
        if density >= self.marked_threshold:
            value = True
            conf = float(np.clip(
                (density - self.marked_threshold) / max(0.5 - self.marked_threshold, 1e-6), 0, 1
            ))
        elif density <= self.unmarked_threshold:
            value = False
            conf = float(np.clip(1 - density / max(self.unmarked_threshold, 1e-6), 0, 1))
        else:
            value = density > mid
            conf = 0.25
        return {"value": value, "confidence": float(np.clip(conf, 0, 1)), "pixel_density": density}
