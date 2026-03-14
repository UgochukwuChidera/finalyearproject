import cv2
import numpy as np

def binarization(normalized, block_size=31):
    binary = cv2.adaptiveThreshold(normalized,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV,block_size,10)
    fg = float(np.mean(binary>0))
    pfg,pbg = fg, 1-fg
    ent = 0.0
    if pfg>0: ent -= pfg*np.log2(pfg)
    if pbg>0: ent -= pbg*np.log2(pbg)
    h,w = binary.shape
    reg=[np.mean(binary[i*h//4:(i+1)*h//4,j*w//4:(j+1)*w//4]>0)
         for i in range(4) for j in range(4)]
    stab=float(np.clip(1-np.var(reg)/0.02,0,1))
    return binary,{"foreground_ratio":fg,"pixel_entropy":ent,"threshold_stability":stab}
