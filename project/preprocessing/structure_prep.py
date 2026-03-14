import cv2
import numpy as np

def structure_prep(binary, template_size):
    ch,cw = binary.shape; tw,th = template_size
    sc = float(np.clip(1-abs(tw/cw-th/ch)/0.1,0,1))
    orb = cv2.ORB_create(500)
    kp = orb.detect(binary,None)
    fc = len(kp)
    sconf = float(np.clip(0.4*sc+0.6*min(fc/500,1),0,1))
    return {"feature_count":fc,"structural_confidence":sconf}
