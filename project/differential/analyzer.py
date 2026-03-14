import cv2
import numpy as np

class DifferentialAnalyzer:
    def __init__(self, diff_threshold=30, min_region_area=50):
        self.diff_threshold  = diff_threshold
        self.min_region_area = min_region_area

    def analyze(self, aligned_form, template):
        diff = cv2.absdiff(aligned_form, template)
        _,binary_diff = cv2.threshold(diff,self.diff_threshold,255,cv2.THRESH_BINARY)
        k_open  = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
        opened  = cv2.morphologyEx(binary_diff,cv2.MORPH_OPEN,k_open)
        cleaned = cv2.morphologyEx(opened,cv2.MORPH_CLOSE,k_close)
        k_dil   = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
        dilated = cv2.dilate(cleaned,k_dil,iterations=2)
        contours,_ = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in contours if cv2.contourArea(c)>=self.min_region_area]
        mask  = np.zeros_like(dilated,dtype=np.uint8)
        for c in valid:
            cv2.drawContours(mask,[c],-1,255,cv2.FILLED)
        dev = float(np.mean(mask>0))
        return mask,{"binary_diff":binary_diff,"cleaned_diff":cleaned,
                     "diff_region_count":len(valid),"deviation_ratio":dev}
