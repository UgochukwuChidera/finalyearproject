"""
Image Preprocessing Pipeline for Scanned Forms
Fixes:
  1. Skew / rotation        — deskew using Hough line detection
  2. Edge smear / bleed     — crop borders by a safe margin
  3. Template alignment     — homography-based alignment to a reference template
  4. General cleanup        — binarization, denoising for better OCR

Requirements:
    pip install opencv-python numpy pytesseract Pillow
"""

import cv2
import numpy as np
from PIL import Image
import pytesseract
from pathlib import Path


# ================================================================
# 1. LOAD HELPERS
# ================================================================

def pil_to_cv(pil_img):
    """Convert PIL Image to OpenCV BGR array."""
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def cv_to_pil(cv_img):
    """Convert OpenCV BGR array to PIL Image."""
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))


# ================================================================
# 2. DESKEW — fix rotation/tilt
# ================================================================

def deskew(image):
    """
    Detects and corrects skew in a scanned document.
    Works by finding the dominant angle of text lines using
    Hough line transform, then rotating to straighten.

    Returns corrected OpenCV image.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold to binary
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Detect lines using HoughLines
    lines = cv2.HoughLinesP(
        binary,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=100,
        maxLineGap=10
    )

    if lines is None:
        print("  [Deskew] No lines detected, skipping.")
        return image

    # Calculate angle of each line
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Only consider near-horizontal lines (text lines)
        if -45 < angle < 45:
            angles.append(angle)

    if not angles:
        print("  [Deskew] No valid angles found, skipping.")
        return image

    # Use median angle to avoid outliers
    median_angle = np.median(angles)
    print(f"  [Deskew] Detected skew angle: {median_angle:.2f}°")

    # Only rotate if skew is significant (> 0.3 degrees)
    if abs(median_angle) < 0.3:
        print("  [Deskew] Skew within tolerance, no rotation needed.")
        return image

    # Rotate image to correct skew
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    corrected = cv2.warpAffine(
        image,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE  # fill edges with nearby pixels
    )
    print(f"  [Deskew] Rotated by {-median_angle:.2f}°")
    return corrected


# ================================================================
# 3. EDGE CROP — remove scanner edge smear/bleed
# ================================================================

def crop_edges(image, margin_percent=2.5):
    """
    Crops a percentage of pixels from all edges to remove
    scanner bleed, dark borders, and edge smear artifacts.

    margin_percent: what % of width/height to remove per side
                    2.5% is a safe default — increase if smear is heavy
    """
    h, w = image.shape[:2]
    margin_x = int(w * margin_percent / 100)
    margin_y = int(h * margin_percent / 100)

    cropped = image[margin_y:h - margin_y, margin_x:w - margin_x]
    print(f"  [Crop] Removed {margin_percent}% border margins "
          f"({margin_x}px sides, {margin_y}px top/bottom)")
    return cropped


def detect_and_crop_dark_borders(image, dark_threshold=50, min_content_percent=70):
    """
    More intelligent border removal — scans inward from each edge
    until it finds content (non-dark pixels), then crops.
    Better for uneven smear that is heavier on one side.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    def find_border(strip, threshold, min_content):
        for i, row in enumerate(strip):
            light_pixels = np.sum(row > threshold)
            if light_pixels > (len(row) * min_content / 100):
                return i
        return 0

    # Scan from each edge
    top    = find_border(gray, dark_threshold, min_content_percent)
    bottom = h - find_border(gray[::-1], dark_threshold, min_content_percent)
    left   = find_border(gray.T, dark_threshold, min_content_percent)
    right  = w - find_border(gray.T[::-1], dark_threshold, min_content_percent)

    # Add small padding after detected edge
    pad = 10
    top    = max(0, top - pad)
    bottom = min(h, bottom + pad)
    left   = max(0, left - pad)
    right  = min(w, right + pad)

    cropped = image[top:bottom, left:right]
    print(f"  [SmartCrop] Cropped to content area: "
          f"top={top}, bottom={bottom}, left={left}, right={right}")
    return cropped


# ================================================================
# 4. TEMPLATE ALIGNMENT — homography warping to reference
# ================================================================

def align_to_template(image, template):
    """
    Aligns a scanned form image to a reference template using
    feature-based homography (ORB features + RANSAC).

    This corrects:
      - Rotation
      - Perspective distortion
      - Scale differences
      - Translation offset

    image    : the scanned form (OpenCV BGR)
    template : the reference/blank form (OpenCV BGR)

    Returns the warped image aligned to the template's coordinate space.
    """
    gray_img  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_tmpl = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Use ORB feature detector (free, no patent issues)
    orb = cv2.ORB_create(nfeatures=2000)

    kp1, des1 = orb.detectAndCompute(gray_tmpl, None)
    kp2, des2 = orb.detectAndCompute(gray_img, None)

    if des1 is None or des2 is None:
        print("  [Align] Could not detect features, skipping alignment.")
        return image

    # Match features using brute-force Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    print(f"  [Align] Found {len(matches)} feature matches")

    # Need at least 10 good matches for reliable homography
    if len(matches) < 10:
        print("  [Align] Not enough matches for alignment, skipping.")
        return image

    # Use top 200 matches
    good_matches = matches[:200]

    # Extract matched keypoint coordinates
    pts_template = np.float32(
        [kp1[m.queryIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    pts_image = np.float32(
        [kp2[m.trainIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    # Find homography matrix using RANSAC (robust to outliers)
    H, mask = cv2.findHomography(pts_image, pts_template, cv2.RANSAC, 5.0)

    if H is None:
        print("  [Align] Homography failed, skipping alignment.")
        return image

    inliers = np.sum(mask)
    print(f"  [Align] Homography computed with {inliers} inliers")

    # Warp image to match template dimensions
    h, w = template.shape[:2]
    aligned = cv2.warpPerspective(image, H, (w, h))
    print(f"  [Align] Image warped to template size {w}x{h}")
    return aligned


# ================================================================
# 5. BINARIZATION + DENOISE — improve OCR accuracy
# ================================================================

def clean_for_ocr(image):
    """
    Converts to grayscale, denoises, and applies adaptive
    thresholding for crisp black-and-white output ideal for OCR.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    # Adaptive threshold handles uneven lighting across the page
    binary = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2
    )
    print("  [Clean] Applied adaptive threshold and denoise")
    return binary


# ================================================================
# 6. FULL PIPELINE
# ================================================================

def preprocess(
    image,                      # PIL Image or OpenCV array
    template=None,              # optional PIL/CV reference template
    do_deskew=True,
    do_crop=True,
    crop_mode='smart',          # 'smart' or 'fixed'
    crop_margin=2.5,            # used if crop_mode='fixed'
    do_align=True,
    do_clean=True
):
    """
    Full preprocessing pipeline. Pass a PIL Image and optionally
    a template image. Returns a cleaned PIL Image ready for OCR.

    Steps (all toggleable):
      1. Deskew     — fix rotation
      2. Crop       — remove edge smear
      3. Align      — warp to template (if template provided)
      4. Clean      — binarize + denoise for OCR
    """
    print("\n[Preprocess] Starting pipeline...")

    # Convert PIL to OpenCV if needed
    if isinstance(image, Image.Image):
        cv_img = pil_to_cv(image)
    else:
        cv_img = image.copy()

    if template is not None and isinstance(template, Image.Image):
        cv_tmpl = pil_to_cv(template)
    else:
        cv_tmpl = template

    # --- Step 1: Deskew ---
    if do_deskew:
        cv_img = deskew(cv_img)

    # --- Step 2: Crop edges ---
    if do_crop:
        if crop_mode == 'smart':
            cv_img = detect_and_crop_dark_borders(cv_img)
        else:
            cv_img = crop_edges(cv_img, crop_margin)

    # --- Step 3: Template alignment ---
    if do_align and cv_tmpl is not None:
        cv_img = align_to_template(cv_img, cv_tmpl)
    elif do_align and cv_tmpl is None:
        print("  [Align] Skipped — no template provided")

    # --- Step 4: Clean for OCR ---
    if do_clean:
        cleaned = clean_for_ocr(cv_img)
        result = Image.fromarray(cleaned)
    else:
        result = cv_to_pil(cv_img)

    print("[Preprocess] Pipeline complete.\n")
    return result


def preprocess_and_ocr(image, template=None, **kwargs):
    """
    Convenience function — preprocess then OCR in one call.
    Returns extracted text string.
    """
    cleaned = preprocess(image, template, **kwargs)
    text = pytesseract.image_to_string(cleaned)
    return text


# ================================================================
# USAGE EXAMPLES
# ================================================================
if __name__ == '__main__':
    from PIL import Image

    # --- Basic usage: just deskew + crop + clean ---
    img = Image.open('scanned_form.tif')
    text = preprocess_and_ocr(img)
    print(text)

    # --- With template alignment ---
    img      = Image.open('filled_form.tif')
    template = Image.open('blank_template.tif')  # your clean blank form
    text = preprocess_and_ocr(img, template=template)
    print(text)

    # --- Process multi-page TIFF ---
    tiff = Image.open('multi_page.tif')
    template = Image.open('blank_template.tif')
    page = 0
    try:
        while True:
            tiff.seek(page)
            text = preprocess_and_ocr(tiff.copy(), template=template)
            print(f"\n=== PAGE {page + 1} ===\n{text}")
            page += 1
    except EOFError:
        pass
