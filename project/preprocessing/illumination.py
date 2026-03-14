import cv2
import numpy as np

def illumination_normalization(gray, grayscale_std, blur_size=51, clahe_tile=8):
    bg = cv2.GaussianBlur(gray,(blur_size,blur_size),0)
    corrected = cv2.divide(gray,bg+1,scale=255)
    clahe = cv2.createCLAHE(2.0,(clahe_tile,clahe_tile))
    normalized = clahe.apply(corrected)
    illum_std = float(np.std(bg))
    uniformity = float(np.clip(1-illum_std/60,0,1))
    gain = float(np.clip((np.std(normalized)/(grayscale_std+1e-5))/1.5,0,1))
    return normalized, {"illumination_uniformity":uniformity,"local_contrast_gain":gain}
