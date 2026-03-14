import numpy as np

def border_removal(binary, threshold_stability):
    h,w = binary.shape
    rs = np.sum(binary>0,axis=1); cs = np.sum(binary>0,axis=0)
    rows = np.where(rs>0.01*w)[0]; cols = np.where(cs>0.01*h)[0]
    if len(rows)==0 or len(cols)==0:
        return binary,{"border_thickness":0.0,"cropping_confidence":0.0}
    t,b = rows[0],rows[-1]; l,r = cols[0],cols[-1]
    cropped = binary[t:b,l:r]
    removed = (t+(h-b))*w+(l+(w-r))*h
    thick = float(removed/(h*w))
    ec = 1-np.mean([np.mean(binary[t:t+10,:]>0),np.mean(binary[b-10:b,:]>0),
                    np.mean(binary[:,l:l+10]>0),np.mean(binary[:,r-10:r]>0)])
    conf = float(np.clip(0.6*ec+0.4*threshold_stability,0,1))
    return cropped,{"border_thickness":thick,"cropping_confidence":conf}
