import numpy as np

def fuse(metrics):
    score=(0.22*metrics["contrast_score"]+0.18*metrics["blur_score"]+
           0.14*metrics["noise_score"]+0.14*metrics["balance_score"]+
           0.10*metrics["illumination_uniformity"]+0.08*metrics["local_contrast_gain"]+
           0.07*metrics["threshold_stability"]+0.07*metrics["structural_confidence"]-
           0.07*metrics["effective_skew_risk"])
    return float(np.clip(score,0,1))
