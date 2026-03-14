import cv2
import numpy as np

def baseline_metrics(gray):
    mean = float(np.mean(gray)); std = float(np.std(gray))
    mn, mx = int(np.min(gray)), int(np.max(gray))
    rng = float(mx - mn)
    nw = float(np.mean(gray > 240)); nb = float(np.mean(gray < 15))
    contrast = float(np.clip((rng - 80) / 120, 0, 1))
    ws = np.clip(1 - abs(nw - 0.6), 0, 1); bs = np.clip(1 - nb / 0.15, 0, 1)
    balance = float(0.6 * ws + 0.4 * bs)
    lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur = float(np.clip((lap - 50) / 150, 0, 1))
    den = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    noise = float(np.clip(1 - np.mean(cv2.absdiff(gray, den)) / 40, 0, 1))
    return {"grayscale_mean":mean,"grayscale_std":std,"min_intensity":mn,
            "max_intensity":mx,"intensity_range":rng,"near_white_ratio":nw,
            "near_black_ratio":nb,"contrast_score":contrast,"balance_score":balance,
            "blur_score":blur,"noise_score":noise}
