import cv2
import numpy as np

def skew_analysis(gray, hough_threshold=200):
    binary = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,15,10)
    edges = cv2.Canny(binary,50,150)
    lines = cv2.HoughLines(edges,1,np.pi/180,hough_threshold)
    skew_angle=0.0; skew_confidence=0.0; skew_acceptability=1.0
    if lines is not None:
        angles=np.array([(t-np.pi/2)*180/np.pi for r,t in lines[:30,0]])
        med=np.median(angles); mn=np.mean(angles)
        skew_angle=float(0.8*med+0.2*mn)
        cs=1-np.std(angles)/2.0; ca=1-abs(mn-med)/2.0
        skew_confidence=float(np.clip(0.7*cs+0.3*ca,0,1))
        skew_acceptability=float(np.clip(1-abs(skew_angle)/3.0,0,1))
    risk=float(np.clip(abs(skew_angle)/5.0,0,1))
    return {"skew_angle":skew_angle,"skew_confidence":skew_confidence,
            "skew_acceptability":skew_acceptability,"skew_risk":risk,
            "effective_skew_risk":float(skew_confidence*risk)}
