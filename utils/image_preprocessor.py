"""
Image preprocessing utilities for scanned forms.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


def pil_to_cv(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def cv_to_pil(cv_img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))


def deskew(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    lines = cv2.HoughLinesP(binary, rho=1, theta=np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    if lines is None:
        return image

    angles: list[float] = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if -45 < angle < 45:
            angles.append(angle)
    if not angles:
        return image

    median_angle = np.median(angles)
    if abs(median_angle) < 0.3:
        return image

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    return cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def detect_and_crop_dark_borders(
    image: np.ndarray, dark_threshold: int = 50, min_content_percent: int = 70
) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    def find_border(strip: np.ndarray, threshold: int, min_content: int) -> int:
        for i, row in enumerate(strip):
            light_pixels = np.sum(row > threshold)
            if light_pixels > (len(row) * min_content / 100):
                return i
        return 0

    top = find_border(gray, dark_threshold, min_content_percent)
    bottom = h - find_border(gray[::-1], dark_threshold, min_content_percent)
    left = find_border(gray.T, dark_threshold, min_content_percent)
    right = w - find_border(gray.T[::-1], dark_threshold, min_content_percent)

    pad = 10
    top = max(0, top - pad)
    bottom = min(h, bottom + pad)
    left = max(0, left - pad)
    right = min(w, right + pad)

    return image[top:bottom, left:right]


def align_to_template(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_tmpl = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(gray_tmpl, None)
    kp2, des2 = orb.detectAndCompute(gray_img, None)
    if des1 is None or des2 is None:
        return image

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)
    if len(matches) < 10:
        return image

    good = matches[:200]
    pts_template = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_image = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(pts_image, pts_template, cv2.RANSAC, 5.0)
    if H is None:
        return image

    h, w = template.shape[:2]
    return cv2.warpPerspective(image, H, (w, h))


def clean_image(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    return cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=11, C=2)


def preprocess(
    image: np.ndarray,
    template: np.ndarray | None = None,
    do_deskew: bool = True,
    do_crop: bool = True,
    do_align: bool = True,
    do_clean: bool = True,
) -> Image.Image:
    cv_img = pil_to_cv(image) if isinstance(image, Image.Image) else image.copy()
    cv_tmpl = pil_to_cv(template) if isinstance(template, Image.Image) else template

    if do_deskew:
        cv_img = deskew(cv_img)
    if do_crop:
        cv_img = detect_and_crop_dark_borders(cv_img)
    if do_align and cv_tmpl is not None:
        cv_img = align_to_template(cv_img, cv_tmpl)

    if do_clean:
        cleaned = clean_image(cv_img)
        return Image.fromarray(cleaned)
    return cv_to_pil(cv_img)


def preprocess_for_ai(
    image: np.ndarray,
    template: np.ndarray | None = None,
    **kwargs: bool,
) -> Image.Image:
    return preprocess(image, template, **kwargs)


if __name__ == "__main__":
    img_path = Path("scanned_form.tif")
    if img_path.exists():
        img = Image.open(str(img_path))
        preprocess_for_ai(img)
